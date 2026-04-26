"""NLCP V3 ConceptPyramidBuilder Training Script.

Train the Builder to extract groundtruth concept pyramids from CoT traces.
The Builder learns to decompose Chain-of-Thought into hierarchical concepts
through end-to-end optimization.

Usage:
    # Single GPU training
    python examples/nlcpV3/train_builder.py \
        -c configs/nlcpV3/GSM8K/train_builder_Qwen2.5-0.5B_6level.yml

    # Multi-GPU training (DDP)
    torchrun --nproc_per_node=2 examples/nlcpV3/train_builder.py \
        -c configs/nlcpV3/GSM8K/train_builder_Qwen2.5-0.5B_6level.yml

Training Losses:
    1. CoT Reconstruction Loss (L_recon, recon_loss_weight):
       MSE between reconstructed_hidden and projected_hidden.
       Ensures the pyramid preserves CoT information — the primary
       supervision signal for the Builder.

    2. Intra-Level Ordering Loss (L_order, concept_loss_weight):
       Encourages concepts within each level to attend to sequential
       CoT positions. NOT a reconstruction loss — it is a structural
       regularizer that enforces ordered concept extraction.

    3. Residual Regularization (L_res, fixed small weight 0.01):
       Minimizes ||H_rest|| to ensure clean residual decomposition.
       All meaningful information should be captured by the pyramid.

    4. NTP / Reasoning Loss (L_ntp, ntp_loss_weight) — optional:
       Enabled when use_reasoning_loss=True in config.
       Given Q + concept pyramid (via back_proj), can the reason_model
       generate the correct solution? This validates that the pyramid
       supports effective reasoning — arguably more important than
       recon_loss alone. Requires back_proj (D → D_encoder) to inject
       concept embeddings into the model's input space.

Output Structure:
    EXPERIMENT/nlcpV3/builder/{experiment_name}/
    ├── checkpoints/
    │   ├── checkpoint-epoch{N}-step{S}.pt
    │   └── checkpoint_best.pt
    └── logs/
        ├── training.log
        ├── config.json
        └── training_history.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

from nlcpV3.config import NLCPV3Config
from nlcpV3.concept_hybrid_builder import ConceptPyramidBuilder, PyramidOutput
from nlcpV3.data_loader import NLCPV3DataLoader
from lmbase.utils.env_tools import get_device
from ram.utils import load_config, setup_environment


def parse_args():
    parser = argparse.ArgumentParser(description="Train ConceptPyramidBuilder")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--resume", type=str, default="", help="Path to checkpoint to resume from"
    )
    return parser.parse_args()


def compute_builder_loss(
    pyramid: PyramidOutput,
    config: NLCPV3Config,
) -> tuple[torch.Tensor, dict]:
    """Compute training losses for ConceptPyramidBuilder.

    Args:
        pyramid: Output from builder.forward()
        config: NLCPV3Config with loss weights

    Returns:
        total_loss: Weighted sum of all losses
        loss_dict: Individual loss components for logging
    """
    loss_dict = {}

    # --- 1. Reconstruction Loss (recon_loss_weight) ---
    # L_recon = MSE(reconstructed_hidden, projected_hidden)
    # Ensures the pyramid captures CoT information.
    # This is the "CoT reconstruction" check: can f_hat_K recover the
    # original CoT hidden states after hierarchical compression?
    recon_loss = F.mse_loss(
        pyramid.reconstructed_hidden,
        pyramid.projected_hidden,
    )
    loss_dict["recon"] = recon_loss.item()

    # --- 2. Intra-Level Ordering Loss (concept_loss_weight) ---
    # Encourages concepts within each level to attend to sequential positions.
    # For level k with L_k concepts and sequence length L:
    #   Concept j should attend most to position ~ j * (L / L_k)
    # We compute a soft target: a Gaussian centered at expected position.
    # This is NOT a reconstruction loss — it is a structural regularizer
    # that enforces ordered concept extraction within each granularity level.
    ordering_loss = torch.tensor(0.0, device=recon_loss.device)
    for k, lo in enumerate(pyramid.level_outputs):
        Lk = lo.attention_weights.shape[1]  # num concepts in this level
        seq_len = lo.attention_weights.shape[2]  # CoT sequence length
        if Lk <= 1:
            continue

        # Expected center position for concept j
        centers = torch.linspace(0, seq_len - 1, Lk, device=lo.attention_weights.device)
        # Create soft target: Gaussian around center
        positions = torch.arange(seq_len, device=lo.attention_weights.device).float()
        sigma = max(seq_len / Lk / 2, 1.0)
        target = torch.exp(
            -((positions.unsqueeze(0) - centers.unsqueeze(1)) ** 2) / (2 * sigma**2)
        )
        target = target / target.sum(dim=1, keepdim=True)  # normalize per concept

        # Cross-entropy between predicted attention and soft target
        # attention_weights: [B, Lk, L]
        attn = lo.attention_weights.mean(dim=0)  # [Lk, L]
        level_order_loss = -(target * torch.log(attn + 1e-8)).sum(dim=1).mean()
        ordering_loss = ordering_loss + level_order_loss

    if len(pyramid.level_outputs) > 0:
        ordering_loss = ordering_loss / len(pyramid.level_outputs)
    loss_dict["ordering"] = ordering_loss.item()

    # --- 3. Residual Regularization (fixed small weight) ---
    # Minimizes ||H_rest|| to ensure clean residual decomposition.
    # H_rest_K should be small: all meaningful information should be
    # captured by the pyramid, leaving only noise in the final residual.
    res_loss = pyramid.residual_hidden.abs().mean()
    loss_dict["residual"] = res_loss.item()

    # --- Total Loss ---
    # Weighted combination of builder objectives.
    # When use_reasoning_loss is enabled and ntp_loss_weight > 0, the NTP loss
    # is added externally (not here) because it requires question/solution data
    # that is not available in this function.
    total_loss = (
        config.recon_loss_weight * recon_loss
        + config.concept_loss_weight * ordering_loss
        + 0.01 * res_loss  # small fixed regularization for clean decomposition
    )
    loss_dict["total"] = total_loss.item()

    return total_loss, loss_dict


def create_dataloader(
    data_cfg: dict,
    batch_size: int,
    num_workers: int,
    include_solution: bool,
    shuffle: bool,
    drop_last: bool,
):
    """Create NLCP V3 dataloader using NLCPV3DataLoader.

    Args:
        data_cfg: Dataset configuration dict for lmbase registry.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        include_solution: If True, also extract groundtruth solutions
            (needed for NTP / reasoning loss).
        shuffle: Whether to shuffle the dataset.
        drop_last: Whether to drop incomplete final batches.
    """
    return NLCPV3DataLoader(
        data_cfg=data_cfg,
        batch_size=batch_size,
        include_solution=include_solution,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


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
    """Save training checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "model_state_dict": builder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }

    if is_best:
        path = checkpoint_dir / "checkpoint_best.pt"
    else:
        path = checkpoint_dir / f"checkpoint-epoch{epoch}-step{step}.pt"

    torch.save(checkpoint, path)
    return path


def load_checkpoint(
    checkpoint_path: Path,
    builder: ConceptPyramidBuilder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> tuple[int, int]:
    """Load training checkpoint and return (epoch, step)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    builder.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"], checkpoint["step"]


def train_builder(config: dict):
    """Main training loop for ConceptPyramidBuilder."""
    # =================================================================
    # Extract config sections
    # =================================================================
    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    env_cfg = config.get("environment", {})
    log_cfg = config.get("log", {})

    # Training hyperparameters
    batch_size = train_cfg["batch_size"]
    learning_rate = train_cfg["learning_rate"]
    weight_decay = train_cfg.get("weight_decay", 0.01)
    num_epochs = train_cfg["num_epochs"]
    warmup_ratio = train_cfg.get("warmup_ratio", 0.1)
    gradient_clip = train_cfg.get("gradient_clip", 1.0)
    log_interval = train_cfg.get("log_step_interval", 10)
    checkpoint_interval = train_cfg.get("checkpoint_step_interval", 500)
    eval_interval = train_cfg.get("eval_step_interval", 0)
    resume = train_cfg.get("resume", "")

    # Output directories
    save_folder = Path(log_cfg.get("save_folder", "EXPERIMENT/nlcpV3/builder"))
    checkpoint_dir = Path(log_cfg.get("checkpoint_path", save_folder / "checkpoints"))
    log_dir = Path(log_cfg.get("log_path", save_folder / "logs"))

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # =================================================================
    # Setup logging
    # =================================================================
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("train_builder")

    logger.info("=" * 60)
    logger.info("ConceptPyramidBuilder Training")
    logger.info("=" * 60)

    # =================================================================
    # Setup environment
    # =================================================================
    seed = env_cfg.get("seed", 42)
    setup_environment({"seed": seed, "device": "auto"})
    device = str(get_device("auto"))
    logger.info(f"Device: {device}")

    # =================================================================
    # Build NLCPV3Config and Builder
    # =================================================================
    logger.info("[1] Building NLCPV3Config...")
    nlcp_config = NLCPV3Config.from_yaml(config)
    logger.info(f"    reason_model: {nlcp_config.reason_model_name}")
    logger.info(f"    hidden_dim: {nlcp_config.hidden_dim}")
    logger.info(f"    num_levels: {nlcp_config.num_levels}")
    logger.info(f"    level_lengths: {nlcp_config.level_lengths}")
    logger.info(f"    total_concepts: {nlcp_config.total_concepts}")

    logger.info("[2] Loading ConceptPyramidBuilder...")
    builder = ConceptPyramidBuilder(nlcp_config)
    builder.to(device)
    logger.info(f"    reason_model_hidden_dim: {builder.reason_model_hidden_dim}")

    # Collect trainable parameters
    trainable_params = []
    for name, param in builder.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            logger.info(f"    [TRAIN] {name}: {list(param.shape)}")

    total_params = sum(p.numel() for p in builder.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    logger.info(f"    Total params: {total_params:,}")
    logger.info(f"    Trainable params: {trainable_count:,}")

    # =================================================================
    # Setup optimizer and scheduler
    # =================================================================
    logger.info("[3] Setting up optimizer...")
    optimizer = AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # =================================================================
    # Load dataset
    # =================================================================
    logger.info("[4] Loading dataset...")
    use_reasoning = nlcp_config.use_reasoning_loss and nlcp_config.ntp_loss_weight > 0
    dataloader = create_dataloader(
        data_cfg,
        batch_size=batch_size,
        num_workers=env_cfg["dataloader_num_workers"],
        include_solution=use_reasoning,
        shuffle=data_cfg["shuffle"],
        drop_last=data_cfg["drop_last"],
    )
    logger.info(f"    Dataset: {data_cfg['data_name']}")
    logger.info(f"    Batches per epoch: {len(dataloader)}")
    logger.info(f"    Batch size: {batch_size}")

    total_steps = len(dataloader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
    )
    logger.info(f"    Total steps: {total_steps}")
    logger.info(f"    Warmup steps: {warmup_steps}")

    # =================================================================
    # Resume from checkpoint
    # =================================================================
    start_epoch = 0
    start_step = 0
    best_loss = float("inf")

    if resume:
        resume_path = Path(resume)
        if resume_path.exists():
            logger.info(f"[5] Resuming from checkpoint: {resume_path}")
            start_epoch, start_step = load_checkpoint(
                resume_path, builder, optimizer, scheduler
            )
            logger.info(f"    Resumed at epoch {start_epoch}, step {start_step}")
        else:
            logger.warning(f"Resume path not found: {resume_path}")

    # =================================================================
    # Save config
    # =================================================================
    config_save_path = log_dir / "config.json"
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    logger.info(f"[6] Config saved to: {config_save_path}")

    # =================================================================
    # Training loop
    # =================================================================
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    builder.train()
    global_step = start_step

    for epoch in range(start_epoch, num_epochs):
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Skip steps before resume point
            if global_step < start_step:
                global_step += 1
                continue

            # Forward: build pyramid from BuilderInput (Q, CoT, Solution)
            # All processing happens internally: tokenize → encode_cot → build pyramid
            pyramid = builder(batch)

            # Compute reconstruction + ordering + residual loss
            total_loss, loss_dict = compute_builder_loss(pyramid, nlcp_config)

            # Compute NTP / reasoning loss if enabled
            if use_reasoning:
                ntp_loss = builder.compute_reasoning_loss(pyramid)
                total_loss = total_loss + nlcp_config.ntp_loss_weight * ntp_loss
                loss_dict["ntp"] = ntp_loss.item()
                loss_dict["total"] = total_loss.item()

            # Backward
            total_loss.backward()

            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, gradient_clip)

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

            # Scheduler step (after warmup)
            if global_step >= warmup_steps:
                scheduler.step()

            # Logging
            epoch_losses.append(loss_dict["total"])
            global_step += 1

            if global_step % log_interval == 0:
                lr = (
                    scheduler.get_last_lr()[0]
                    if scheduler.get_last_lr()
                    else learning_rate
                )
                pbar.set_postfix(
                    {
                        "loss": f"{loss_dict['total']:.4f}",
                        "recon": f"{loss_dict['recon']:.4f}",
                        "order": f"{loss_dict['ordering']:.4f}",
                        "lr": f"{lr:.2e}",
                    }
                )
                logger.info(
                    f"Step {global_step} | loss={loss_dict['total']:.4f} "
                    f"recon={loss_dict['recon']:.4f} ordering={loss_dict['ordering']:.4f} "
                    f"residual={loss_dict['residual']:.4f} lr={lr:.2e}"
                )

            # Checkpointing
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
                logger.info(f"Checkpoint saved: {path}")

        # End of epoch
        avg_epoch_loss = (
            sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("inf")
        )
        logger.info(f"Epoch {epoch+1} completed. Avg loss: {avg_epoch_loss:.4f}")

        # Save epoch checkpoint
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
        logger.info(f"Epoch checkpoint saved: {path}")

    # =================================================================
    # Training complete
    # =================================================================
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best checkpoint: {checkpoint_dir / 'checkpoint_best.pt'}")
    logger.info("=" * 60)


def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    yaml_config = load_config(str(config_path))

    # Merge resume flag from CLI if not in config
    if args.resume and not yaml_config.get("training", {}).get("resume"):
        yaml_config.setdefault("training", {})["resume"] = args.resume

    train_builder(yaml_config)


if __name__ == "__main__":
    main()
