"""Train NLCP V3 ConceptPyramidBuilder.

Usage:
    python examples/nlcpV3/train_builder.py -c configs/nlcpV3/GSM8K/xxx.yml
    torchrun --nproc_per_node=2 examples/nlcpV3/train_builder.py -c xxx.yml
"""

import argparse
import datetime
import json
import logging
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

# Ensure project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

from nlcpV3.concept_hybrid_builder import ConceptPyramidBuilder, PyramidOutput
from nlcpV3.data_loader import NLCPV3DataLoader
from lmbase.utils.env_tools import get_device
from ram.utils import load_config, setup_environment


def _log_terminal_entry(log_path: Path, entry: dict):
    """Append a structured JSON line to the terminal output log file.

    Each entry is a JSON object with timestamp, step, epoch,
    loss values, and learning rate. Written immediately to disk
    so terminal output is preserved even if training crashes.
    """
    entry["timestamp"] = datetime.datetime.now().isoformat()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Train ConceptPyramidBuilder")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--resume", type=str, default="", help="Path to checkpoint to resume from"
    )
    return parser.parse_args()


def _ordering_loss_margin(
    attention_weights: torch.Tensor, margin: float = 1.0
) -> torch.Tensor:
    """Margin-based ordering loss per hybrid-analysis.md Section 5.1.2.

    L_order = Σ_j ReLU(exp_pos[C_j] - exp_pos[C_{j+1}] + margin)
    where exp_pos[C_j] = Σ_t A_j(t) × t

    Args:
        attention_weights: [B, L_k, L] attention weights A_k
        margin: Minimum expected position gap between adjacent concepts

    Returns:
        Scalar ordering loss
    """
    B, Lk, L = attention_weights.shape
    if Lk <= 1:
        return torch.tensor(0.0, device=attention_weights.device)

    positions = torch.arange(L, device=attention_weights.device, dtype=torch.float32)
    # expected_pos: [B, L_k] — expected CoT position for each concept
    expected_pos = (attention_weights * positions.unsqueeze(0).unsqueeze(0)).sum(dim=-1)

    loss = torch.tensor(0.0, device=attention_weights.device)
    for j in range(Lk - 1):
        # Enforce: C_j attends to earlier positions than C_{j+1}
        loss = (
            loss + F.relu(expected_pos[:, j] - expected_pos[:, j + 1] + margin).mean()
        )

    return loss


def _ordering_loss_gaussian(
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Gaussian-target ordering loss (original implementation).

    Encourages each concept's attention to match a Gaussian centered at
    its expected segment position. Soft but does not explicitly enforce
    monotonic ordering.

    Args:
        attention_weights: [B, L_k, L] attention weights A_k

    Returns:
        Scalar ordering loss
    """
    B, Lk, L = attention_weights.shape
    if Lk <= 1:
        return torch.tensor(0.0, device=attention_weights.device)

    centers = torch.linspace(0, L - 1, Lk, device=attention_weights.device)
    positions = torch.arange(L, device=attention_weights.device).float()
    sigma = max(L / Lk / 2, 1.0)
    target = torch.exp(
        -((positions.unsqueeze(0) - centers.unsqueeze(1)) ** 2) / (2 * sigma**2)
    )
    target = target / target.sum(dim=1, keepdim=True)
    attn = attention_weights.mean(dim=0)  # [L_k, L]
    return -(target * torch.log(attn + 1e-8)).sum(dim=1).mean()


def compute_builder_loss(
    pyramid: PyramidOutput,
    loss_weights: dict,
    ordering_loss_type: str = "margin",
) -> tuple[torch.Tensor, dict]:
    """Compute recon + ordering + residual losses.

    Args:
        pyramid: PyramidOutput from builder.forward()
        loss_weights: Dict with recon_loss_weight, concept_loss_weight,
            residual_loss_weight, etc.
        ordering_loss_type: "margin" (design doc spec, mandatory) or
            "gaussian" (original soft target). Can also be "both".

    Returns:
        (total_loss, loss_dict)
    """
    loss_dict = {}
    device = pyramid.projected_hidden.device

    # ── Reconstruction loss ──────────────────────────────────────────
    # Mask out padded positions if attention_mask is provided
    if pyramid.attention_mask is not None:
        mask = pyramid.attention_mask.unsqueeze(-1)  # [B, L, 1]
        recon_diff = (pyramid.reconstructed_hidden - pyramid.projected_hidden) * mask
        recon_loss = (recon_diff**2).sum() / mask.sum()
    else:
        recon_loss = F.mse_loss(pyramid.reconstructed_hidden, pyramid.projected_hidden)
    loss_dict["recon"] = recon_loss.item()

    # ── Ordering loss ────────────────────────────────────────────────
    ordering_loss = torch.tensor(0.0, device=device)
    ordering_margin = loss_weights["ordering_margin"]
    levels_with_ordering = 0

    for lo in pyramid.level_outputs:
        Lk = lo.attention_weights.shape[1]
        if Lk <= 1:
            continue
        levels_with_ordering += 1

        if ordering_loss_type == "margin":
            level_order_loss = _ordering_loss_margin(
                lo.attention_weights, margin=ordering_margin
            )
        elif ordering_loss_type == "gaussian":
            level_order_loss = _ordering_loss_gaussian(lo.attention_weights)
        elif ordering_loss_type == "both":
            level_order_loss = _ordering_loss_margin(
                lo.attention_weights, margin=ordering_margin
            ) + _ordering_loss_gaussian(lo.attention_weights)
        else:
            raise ValueError(f"Unknown ordering_loss_type: {ordering_loss_type}")

        ordering_loss = ordering_loss + level_order_loss

    if levels_with_ordering > 0:
        ordering_loss = ordering_loss / levels_with_ordering
    loss_dict["ordering"] = ordering_loss.item()

    # ── Residual loss ────────────────────────────────────────────────
    if pyramid.attention_mask is not None:
        mask = pyramid.attention_mask.unsqueeze(-1)
        res_loss = (pyramid.residual_hidden.abs() * mask).sum() / mask.sum()
    else:
        res_loss = pyramid.residual_hidden.abs().mean()
    loss_dict["residual"] = res_loss.item()

    # ── Total loss ───────────────────────────────────────────────────
    residual_weight = loss_weights["residual_loss_weight"]
    total_loss = (
        loss_weights["recon_loss_weight"] * recon_loss
        + loss_weights["concept_loss_weight"] * ordering_loss
        + residual_weight * res_loss
    )
    loss_dict["total"] = total_loss.item()

    return total_loss, loss_dict


def save_checkpoint(
    builder: ConceptPyramidBuilder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    step: int,
    loss: float,
    checkpoint_dir: Path,
    is_best: bool = False,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "model_state_dict": builder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    path = checkpoint_dir / (
        "checkpoint_best.pt" if is_best else f"checkpoint-epoch{epoch}-step{step}.pt"
    )
    torch.save(checkpoint, path)
    return path


def load_checkpoint(
    checkpoint_path: Path,
    builder: ConceptPyramidBuilder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> tuple[int, int, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    builder.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"], checkpoint["step"], checkpoint["loss"]


def train_builder(config: dict):
    """Main training loop."""
    # Extract sub-configs to avoid repeated deep lookups
    model_cfg = config["model"]
    reason_cfg = model_cfg["reason_model"]
    pyramid_cfg = model_cfg["pyramid"]
    builder_cfg = model_cfg["builder"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    env_cfg = config["environment"]
    log_cfg = config["log"]
    loss_weights = train_cfg["loss_weights"]

    batch_size = train_cfg["batch_size"]
    learning_rate = train_cfg["learning_rate"]
    weight_decay = train_cfg["weight_decay"]
    num_epochs = train_cfg["num_epochs"]
    warmup_ratio = train_cfg["warmup_ratio"]
    gradient_clip = train_cfg["gradient_clip"]
    log_interval = train_cfg["log_step_interval"]
    checkpoint_interval = train_cfg["checkpoint_step_interval"]
    resume = train_cfg["resume"]
    ordering_loss_type = train_cfg["ordering_loss_type"]

    checkpoint_dir = Path(log_cfg["checkpoint_path"])
    log_dir = Path(log_cfg["log_path"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_cfg["log_level"].upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("train_builder")
    terminal_log_path = log_dir / "terminal_output.jsonl"

    seed = env_cfg["seed"]
    setup_environment({"seed": seed, "device": "auto"})
    device = str(get_device("auto"))
    logger.info(f"Device: {device}")

    logger.info(
        f"Model: {reason_cfg['reason_model_name']} | "
        f"hidden_dim={pyramid_cfg['hidden_dim']} | "
        f"levels={pyramid_cfg['num_levels']} | "
        f"concepts={sum(pyramid_cfg['level_lengths'])}"
    )

    builder = ConceptPyramidBuilder(config)
    builder.to(device)

    trainable_params = [p for p in builder.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in builder.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    logger.info(f"Params: {total_params:,} total, {trainable_count:,} trainable")

    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

    use_reasoning = (
        builder_cfg["use_reasoning_loss"] and loss_weights["ntp_loss_weight"] > 0
    )
    dataloader = NLCPV3DataLoader(
        data_cfg=data_cfg,
        batch_size=batch_size,
        include_solution=use_reasoning,
        shuffle=data_cfg["shuffle"],
        drop_last=data_cfg["drop_last"],
        num_workers=env_cfg["dataloader_num_workers"],
    )
    logger.info(
        f"Dataset: {data_cfg['data_name']} | Batches/epoch: {len(dataloader)} | Batch size: {batch_size}"
    )

    total_steps = len(dataloader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    logger.info(
        f"Total steps: {total_steps} | Warmup: {warmup_steps} | LR: {learning_rate}"
    )

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    history = []

    if resume:
        resume_path = Path(resume)
        if resume_path.exists():
            start_epoch, global_step, best_loss = load_checkpoint(
                resume_path, builder, optimizer, scheduler
            )
            logger.info(f"Resumed from epoch {start_epoch}, step {global_step}")
        else:
            logger.warning(f"Resume path not found: {resume_path}")

    config_save_path = log_dir / "config.json"
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    builder.train()

    for epoch in range(start_epoch, num_epochs):
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in pbar:
            # Encode CoT → hidden states, then build pyramid
            enc_out = builder.encode_cot(batch.cot_answers)
            pyramid = builder(
                enc_out.hidden_states, attention_mask=enc_out.attention_mask
            )
            total_loss, loss_dict = compute_builder_loss(
                pyramid,
                loss_weights,
                ordering_loss_type=ordering_loss_type,
            )

            if use_reasoning:
                # Tokenize questions and solutions for NTP loss
                q_tokens = builder.tokenizer(
                    batch.questions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=pyramid_cfg["max_seq_len"],
                )
                q_ids = q_tokens["input_ids"].to(device)
                q_mask = q_tokens["attention_mask"].to(device)

                sol_tokens = builder.tokenizer(
                    batch.solutions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=pyramid_cfg["max_seq_len"],
                )
                sol_ids = sol_tokens["input_ids"].to(device)

                ntp_loss = builder.compute_reasoning_loss(
                    pyramid, q_ids, q_mask, sol_ids
                )
                total_loss = total_loss + loss_weights["ntp_loss_weight"] * ntp_loss
                loss_dict["ntp"] = ntp_loss.item()
                loss_dict["total"] = total_loss.item()

            total_loss.backward()

            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, gradient_clip)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            epoch_losses.append(loss_dict["total"])
            global_step += 1

            if global_step % log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(
                    {
                        "loss": f"{loss_dict['total']:.4f}",
                        "recon": f"{loss_dict['recon']:.4f}",
                        "order": f"{loss_dict['ordering']:.4f}",
                        "lr": f"{lr:.2e}",
                    }
                )
                log_msg = (
                    f"Step {global_step:>5} | loss={loss_dict['total']:.4f} "
                    f"recon={loss_dict['recon']:.4f} ordering={loss_dict['ordering']:.4f} "
                    f"residual={loss_dict['residual']:.4f} lr={lr:.2e}"
                )
                logger.info(log_msg)
                _log_terminal_entry(
                    terminal_log_path,
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": round(loss_dict["total"], 6),
                        "recon": round(loss_dict["recon"], 6),
                        "ordering": round(loss_dict["ordering"], 6),
                        "residual": round(loss_dict["residual"], 6),
                        "lr": lr,
                    },
                )

            if global_step % checkpoint_interval == 0:
                avg_loss = (
                    sum(epoch_losses[-100:]) / len(epoch_losses[-100:])
                    if epoch_losses
                    else float("inf")
                )
                is_best = avg_loss < best_loss
                if is_best:
                    best_loss = avg_loss
                path = save_checkpoint(
                    builder,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    avg_loss,
                    checkpoint_dir,
                    is_best=is_best,
                )
                logger.info(f"Checkpoint: {path.name}")

            history.append({"step": global_step, "epoch": epoch, **loss_dict})

        avg_epoch_loss = (
            sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("inf")
        )
        logger.info(f"Epoch {epoch+1} avg loss: {avg_epoch_loss:.4f}")
        _log_terminal_entry(
            terminal_log_path,
            {"epoch": epoch, "avg_epoch_loss": round(avg_epoch_loss, 6)},
        )

        path = save_checkpoint(
            builder,
            optimizer,
            scheduler,
            epoch + 1,
            global_step,
            avg_epoch_loss,
            checkpoint_dir,
            is_best=False,
        )
        logger.info(f"Epoch checkpoint: {path.name}")

        with open(log_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2, default=str)

    logger.info("Training complete!")
    logger.info(f"Best checkpoint: {checkpoint_dir / 'checkpoint_best.pt'}")


def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    yaml_config = load_config(str(config_path))

    # Merge resume flag from CLI if not in config
    if args.resume and not yaml_config["training"]["resume"]:
        yaml_config["training"]["resume"] = args.resume

    train_builder(yaml_config)


if __name__ == "__main__":
    main()
