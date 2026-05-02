"""Train NLCP V3 ConceptPyramidBuilder.

Usage:
    python3 examples/nlcpV3/train_builder.py -c configs/nlcpV3/GSM8K/train_builder_Qwen2.5-0.5B_6level.yml
"""

import argparse
import datetime
import json
import logging
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torch.optim import AdamW
from tqdm import tqdm

import swanlab

# Ensure project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

from nlcpV3.concept_hybrid_builder import ConceptPyramidBuilder, PyramidOutput
from nlcpV3.data_loader import NLCPV3DataLoader
from lmbase.utils.env_tools import get_device
from ram.utils import (
    load_config,
    setup_environment,
)  # noqa: F401  (kept for back-compat)


def _seed_single_device(seed: int, device: str) -> None:
    """Seed RNGs for CPU + the chosen CUDA device only.

    Why not ``ram.utils.setup_environment`` (which calls
    ``torch.cuda.manual_seed_all``)?

      ``manual_seed_all`` seeds RNG state on EVERY visible CUDA device.
      To do that, PyTorch must create a full CUDA context on each GPU
      (~300-500 MB each). On a shared cluster, any one of those GPUs
      might be too tight for a new context — the failure is queued as
      an ASYNC error and surfaces later as a misleading "OOM on your
      chosen GPU" when ``builder.to(device)`` runs. Since we only ever
      allocate tensors on ONE device in this script, seeding any other
      device is both wasteful and risky.

    This helper seeds only the chosen device and leaves other GPUs
    untouched, so no spurious context-init failures can be parked.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and device.startswith("cuda:"):
        dev_idx = int(device.split(":")[1])
        with torch.cuda.device(dev_idx):
            torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _log_model_summary(builder: ConceptPyramidBuilder, config: dict, logger):
    """Log a detailed model architecture summary table."""
    reason_cfg = config["model"]["reason_model"]
    pyramid_cfg = config["model"]["pyramid"]
    train_rm_cfg = config["training"]["reason_model"]
    loss_weights = config["training"]["loss_weights"]

    total_params = sum(p.numel() for p in builder.parameters())
    trainable_params = sum(p.numel() for p in builder.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    # Per-module param counts
    def _count(module):
        return sum(p.numel() for p in module.parameters())

    def _count_trainable(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    reason_total = _count(builder.reason_model)
    reason_train = _count_trainable(builder.reason_model)
    proj_params = _count(builder.input_proj) + _count(builder.input_proj_norm)
    query_params = sum(q.numel() for q in builder.concept_queries)
    level_proj_params = _count(builder.level_projs)
    back_proj_params = _count(builder.back_proj)
    temp_params = builder.temperature.numel()

    level_lengths = pyramid_cfg["level_lengths"]
    D = pyramid_cfg["hidden_dim"]
    D_enc = builder.reason_model_hidden_dim

    lines = [
        "",
        "=" * 72,
        "  MODEL ARCHITECTURE SUMMARY",
        "=" * 72,
        "",
        "  Reason Model",
        "  ├─ name              : %s" % reason_cfg["reason_model_name"],
        "  ├─ encoder_dim       : %d" % D_enc,
        "  ├─ vocab_size        : %d" % builder.reason_model.config.vocab_size,
        "  ├─ num_layers        : %d" % builder.reason_model.config.num_hidden_layers,
        "  ├─ freeze            : %s" % train_rm_cfg["freeze"],
        "  ├─ lora              : %s" % (train_rm_cfg["lora"] or "None"),
        "  ├─ params            : %s (trainable: %s)"
        % (f"{reason_total:,}", f"{reason_train:,}"),
        "  ",
        "  Pyramid",
        "  ├─ hidden_dim (D)    : %d" % D,
        "  ├─ num_levels (K)    : %d" % pyramid_cfg["num_levels"],
        "  ├─ level_lengths     : %s  (total: %d)"
        % (level_lengths, sum(level_lengths)),
        "  ├─ max_seq_len       : %d" % pyramid_cfg["max_seq_len"],
        "  ",
        "  Modules                     Shape                 Params",
        "  " + "-" * 68,
        "  input_proj               : [%d, %d] + LN        %s"
        % (D_enc, D, f"{proj_params:,}"),
        "  concept_queries          : %d levels             %s"
        % (len(level_lengths), f"{query_params:,}"),
    ]
    for k, L_k in enumerate(level_lengths):
        lines.append("    level %d               : [%d, %d]" % (k, L_k, D))
    lines += [
        "  temperature              : [1]                   %d" % temp_params,
        "  level_projs              : %d × [%d, %d]        %s"
        % (len(level_lengths), D, D, f"{level_proj_params:,}"),
        "  back_proj                : [%d, %d]              %s"
        % (D, D_enc, f"{back_proj_params:,}"),
        "  ",
        "  Loss Weights",
        "  ├─ recon               : %s" % loss_weights["recon_loss_weight"],
        "  ├─ ordering            : %s" % loss_weights["ordering_loss_weight"],
        "  ├─ residual            : %s" % loss_weights["residual_loss_weight"],
        "  ├─ reasoning           : %s" % loss_weights["reasoning_loss_weight"],
        "  ├─ ordering_margin     : %s" % loss_weights["ordering_margin"],
        "  ",
        "  Parameter Summary",
        "  ├─ total               : %s" % f"{total_params:,}",
        "  ├─ trainable           : %s  (%.2f%%)"
        % (f"{trainable_params:,}", 100.0 * trainable_params / total_params),
        "  └─ frozen              : %s" % f"{frozen_params:,}",
        "=" * 72,
        "",
    ]
    for line in lines:
        logger.info(line)


def _log_terminal_entry(log_path: Path, entry: dict):
    """Append a structured JSON line to the terminal output log file.

    Each entry is a JSON object with timestamp, step, epoch,
    loss values, and learning rate. Written immediately to disk
    so terminal output is preserved even if training crashes.
    """
    entry["timestamp"] = datetime.datetime.now().isoformat()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _log_eval_results(
    eval_losses,
    eval_samples,
    loss_weights,
    eval_type,
    global_step,
    logger,
    terminal_log_path,
    eval_history,
    eval_sample_history,
    log_dir,
    swanlab_prefix,
):
    """Log eval results (raw/weighted) to console, terminal, SwanLab, eval_history.

    Also appends a record to ``eval_sample_history`` documenting exactly
    which samples were consumed by this eval invocation (question text,
    groundtruth solution, and a short stable sample_id), and persists it
    to ``log_dir / eval_sample_history.json`` for repeated verification.
    """
    ew = {
        "recon": eval_losses["recon"] * loss_weights["recon_loss_weight"],
        "ordering": eval_losses["ordering"] * loss_weights["ordering_loss_weight"],
        "residual": eval_losses["residual"] * loss_weights["residual_loss_weight"],
    }
    reasoning_part = ""
    if "reasoning" in eval_losses:
        ew["reasoning"] = (
            eval_losses["reasoning"] * loss_weights["reasoning_loss_weight"]
        )
        reasoning_part = " reasoning=%.4f/%.4f" % (
            eval_losses["reasoning"],
            ew["reasoning"],
        )
    label = "eval(quick)" if eval_type == "quick" else "eval(full) "
    logger.info(
        "  %s | total=%.4f recon=%.4f/%.4f ordering=%.4f/%.4f" " residual=%.4f/%.4f%s",
        label,
        eval_losses["total"],
        eval_losses["recon"],
        ew["recon"],
        eval_losses["ordering"],
        ew["ordering"],
        eval_losses["residual"],
        ew["residual"],
        reasoning_part,
    )
    # SwanLab
    metrics = {
        f"{swanlab_prefix}/total_loss": eval_losses["total"],
        f"{swanlab_prefix}/recon_raw": eval_losses["recon"],
        f"{swanlab_prefix}/recon_weighted": ew["recon"],
        f"{swanlab_prefix}/ordering_raw": eval_losses["ordering"],
        f"{swanlab_prefix}/ordering_weighted": ew["ordering"],
        f"{swanlab_prefix}/residual_raw": eval_losses["residual"],
        f"{swanlab_prefix}/residual_weighted": ew["residual"],
    }
    if "reasoning" in eval_losses:
        metrics[f"{swanlab_prefix}/reasoning_raw"] = eval_losses["reasoning"]
        metrics[f"{swanlab_prefix}/reasoning_weighted"] = ew["reasoning"]
    swanlab.log(metrics, step=global_step)
    # Terminal log
    _log_terminal_entry(
        terminal_log_path,
        {
            "step": global_step,
            "eval_type": eval_type,
            **{f"eval_{k}": round(v, 6) for k, v in eval_losses.items()},
            **{f"eval_{k}_w": round(v, 6) for k, v in ew.items()},
        },
    )
    # Eval history + save immediately (crash-safe)
    eval_history.append(
        {
            "step": global_step,
            "eval_type": eval_type,
            **eval_losses,
            **{f"{k}_w": v for k, v in ew.items()},
        }
    )
    with open(log_dir / "eval_history.json", "w", encoding="utf-8") as f:
        json.dump(eval_history, f, indent=2, default=str)

    # Sample history: one record per eval invocation containing the
    # exact list of samples consumed. Persist alongside eval_history.json
    # so the caller can reconcile loss-history rows with which data was
    # evaluated, and can repeat the check offline without re-running the
    # model.
    eval_sample_history.append(
        {
            "step": global_step,
            "eval_type": eval_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "num_samples": len(eval_samples),
            "samples": eval_samples,
        }
    )
    with open(log_dir / "eval_sample_history.json", "w", encoding="utf-8") as f:
        json.dump(eval_sample_history, f, indent=2, default=str)


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
    attention_weights: torch.Tensor, margin: float
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
    ordering_loss_type: str,
) -> tuple[torch.Tensor, dict]:
    """Compute recon + ordering + residual losses.

    Args:
        pyramid: PyramidOutput from builder.forward()
        loss_weights: Dict with recon_loss_weight, ordering_loss_weight,
            residual_loss_weight, etc.
        ordering_loss_type: "margin" (design doc spec, mandatory) or
            "gaussian" (original soft target). Can also be "both".

    Returns:
        (total_loss, loss_dict)
    """
    loss_dict = {}
    device = pyramid.projected_hidden.device

    # ── Reconstruction loss ──────────────────────────────────────────
    # MSE between back-projected reconstruction and original CoT encodings:
    #   L_recon = ||back_proj(f_hat_K) - H_CoT||^2
    # This measures how well the pyramid preserves the ORIGINAL encoder
    # information, analogous to VAR's reconstruction against frozen encoder output.
    if pyramid.attention_mask is not None:
        mask = pyramid.attention_mask.unsqueeze(-1)  # [B, L, 1]
        recon_diff = (
            pyramid.reconstructed_encoder_hidden - pyramid.encoder_hidden_states
        ) * mask
        num_valid_elements = (
            mask.sum() * pyramid.encoder_hidden_states.shape[-1]
        )  # tokens × D_encoder
        recon_loss = (recon_diff**2).sum() / num_valid_elements
    else:
        recon_loss = F.mse_loss(
            pyramid.reconstructed_encoder_hidden, pyramid.encoder_hidden_states
        )
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
    # L1 averaged over all valid elements (B, L, D), consistent with
    # the per-element mean convention used by reconstruction loss.
    if pyramid.attention_mask is not None:
        mask = pyramid.attention_mask.unsqueeze(-1)
        num_valid_elements = (
            mask.sum() * pyramid.residual_hidden.shape[-1]
        )  # tokens × D
        res_loss = (pyramid.residual_hidden.abs() * mask).sum() / num_valid_elements
    else:
        res_loss = pyramid.residual_hidden.abs().mean()
    loss_dict["residual"] = res_loss.item()

    # ── Total loss ───────────────────────────────────────────────────
    residual_weight = loss_weights["residual_loss_weight"]
    total_loss = (
        loss_weights["recon_loss_weight"] * recon_loss
        + loss_weights["ordering_loss_weight"] * ordering_loss
        + residual_weight * res_loss
    )
    loss_dict["total"] = total_loss.item()

    return total_loss, loss_dict


@torch.no_grad()
def evaluate_builder(
    builder: ConceptPyramidBuilder,
    eval_dataloader: NLCPV3DataLoader,
    loss_weights: dict,
    ordering_loss_type: str,
    device: str,
    pyramid_cfg: dict,
    max_batches: int = 0,
) -> dict:
    """Run evaluation on test data and return averaged loss dict.

    Args:
        builder: The model to evaluate.
        eval_dataloader: DataLoader yielding BuilderInput batches from test set.
        loss_weights: Loss weight configuration.
        ordering_loss_type: "margin", "gaussian", or "both".
        device: Device string.
        pyramid_cfg: Pyramid config (for max_seq_len).
        max_batches: Maximum batches to evaluate. 0 = all batches.

    Returns:
        Tuple ``(avg_loss_dict, samples_list)`` where ``avg_loss_dict`` has
        keys ``total, recon, ordering, residual, reasoning`` (reasoning
        only if the batch had solutions) averaged across consumed
        batches, and ``samples_list`` is a list of per-sample records
        (``batch_idx``, ``pos_in_batch``, ``main_id``, ``question``,
        ``solution``) in the exact order they were evaluated.
    """
    builder.eval()
    all_losses = []
    all_samples = []

    for i, batch in enumerate(eval_dataloader):
        if max_batches > 0 and i >= max_batches:
            break

        enc_out = builder.encode_cot(batch.cot_answers)
        pyramid = builder(enc_out.hidden_states, attention_mask=enc_out.attention_mask)
        _, loss_dict = compute_builder_loss(
            pyramid, loss_weights, ordering_loss_type=ordering_loss_type
        )

        if batch.has_solution:
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

            reasoning_loss = builder.compute_reasoning_loss(
                pyramid, q_ids, q_mask, sol_ids
            )
            loss_dict["reasoning"] = reasoning_loss.item()
            loss_dict["total"] = (
                loss_dict["total"]
                + loss_weights["reasoning_loss_weight"] * reasoning_loss.item()
            )

        all_losses.append(loss_dict)

        # Record per-sample metadata so eval_sample_history.json can
        # reconstruct which inputs were consumed by this eval invocation.
        # ``main_id`` comes straight from the lmbase dataset record
        # (e.g. "ID1"), so rows here align 1:1 with source dataset rows
        # and can be re-looked up without any hashing on our side.
        for j in range(batch.batch_size):
            all_samples.append(
                {
                    "batch_idx": i,
                    "pos_in_batch": j,
                    "main_id": batch.main_ids[j],
                    "question": batch.questions[j],
                    "solution": batch.solutions[j] if batch.has_solution else None,
                }
            )

    builder.train()

    if not all_losses:
        return {"total": 0.0, "recon": 0.0, "ordering": 0.0, "residual": 0.0}, []

    # Average across all batches
    avg = {}
    keys = all_losses[0].keys()
    for k in keys:
        avg[k] = sum(d.get(k, 0.0) for d in all_losses) / len(all_losses)
    return avg, all_samples


def save_checkpoint(
    builder: ConceptPyramidBuilder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    step: int,
    loss: float,
    checkpoint_dir: Path,
    filename: str,
) -> Path:
    """Save full training state to ``checkpoint_dir / filename``."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "model_state_dict": builder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    path = checkpoint_dir / filename
    torch.save(checkpoint, path)
    return path


def purge_best_checkpoints(checkpoint_dir: Path, prefix: str) -> None:
    """Remove previous best checkpoints matching ``{prefix}-*.pt`` or legacy ``{prefix}.pt``.

    Used to preserve the "exactly one best file" invariant when the filename carries
    epoch/step tags, so each new best replaces the old one on disk.
    """
    for old in checkpoint_dir.glob(f"{prefix}-*.pt"):
        try:
            old.unlink()
        except OSError:
            pass
    legacy = checkpoint_dir / f"{prefix}.pt"
    if legacy.exists():
        try:
            legacy.unlink()
        except OSError:
            pass


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


def train_builder(config: dict, config_path: Path):
    """Main training loop."""
    # Extract sub-configs
    pyramid_cfg = config["model"]["pyramid"]
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
    log_interval = log_cfg["log_step_interval"]
    checkpoint_interval = log_cfg["checkpoint_step_interval"]
    checkpoint_clean = log_cfg.get("checkpoint_clean", False)
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
    # Pick the device FIRST (NVML-only probe, no CUDA context init on
    # other GPUs), then seed ONLY that device. Avoids
    # manual_seed_all's multi-GPU context init — see
    # _seed_single_device() above for why that matters.
    device = str(get_device("auto"))
    _seed_single_device(seed, device)
    logger.info("Device: %s | seed: %d", device, seed)

    # ── Load .env and initialize SwanLab ─────────────────────────────
    dotenv_path = env_cfg.get("dotenv_path", ".env")
    load_dotenv(dotenv_path)

    # Derive experiment name from the config file's location under
    # ``configs/nlcpV4/``. All path segments between that root and the
    # file (dataset, and any nested variant such as ``AutoWeighted/``)
    # are joined by ``-`` with the filename stem so the name is unique
    # and self-describing regardless of directory depth:
    #   configs/nlcpV4/GSM8K/train_builder_Qwen2.5-0.5B_6level.yml
    #     -> "GSM8K-train_builder_Qwen2.5-0.5B_6level"
    #   configs/nlcpV4/GSM8K/AutoWeighted/train_builder_Qwen2.5-0.5B_6level.yml
    #     -> "GSM8K-AutoWeighted-train_builder_Qwen2.5-0.5B_6level"
    # Fail-fast: if the config lives outside configs/nlcpV4/, fall back
    # to the legacy single-parent form so out-of-tree configs still run.
    configs_root = PROJECT_ROOT / "configs" / "nlcpV4"
    try:
        rel_parts = config_path.resolve().relative_to(configs_root).parent.parts
    except ValueError:
        rel_parts = (config_path.parent.name,)
    experiment_name = "-".join([*rel_parts, config_path.stem])

    swanlab.init(
        project="ReasoningAR",
        experiment_name=experiment_name,
        config=config,
    )
    logger.info("SwanLab initialized")

    builder = ConceptPyramidBuilder(config)
    builder.to(device)
    _log_model_summary(builder, config, logger)

    trainable_params = [p for p in builder.parameters() if p.requires_grad]

    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

    dataloader = NLCPV3DataLoader(
        data_cfg=data_cfg,
        batch_size=batch_size,
        include_solution=True,
        shuffle=data_cfg["shuffle"],
        drop_last=data_cfg["drop_last"],
        num_workers=env_cfg["dataloader_num_workers"],
    )
    logger.info(
        "Dataset: %s | Batches/epoch: %d | Batch size: %d",
        data_cfg["data_name"],
        len(dataloader),
        batch_size,
    )

    # ── Evaluation setup ─────────────────────────────────────────
    eval_cfg = config["evaluation"]
    eval_interval = eval_cfg["eval_step_interval"]
    eval_enabled = eval_interval > 0
    eval_dataloader = None
    eval_history = []
    eval_sample_history = []
    quick_eval_batches = 0
    full_eval_batches = 0

    if eval_enabled:
        eval_data_cfg = eval_cfg["data"]
        eval_dataloader = NLCPV3DataLoader(
            data_cfg=eval_data_cfg,
            batch_size=batch_size,
            include_solution=True,
            shuffle=True,
            drop_last=False,
            num_workers=env_cfg["dataloader_num_workers"],
        )
        eval_dataset_size = eval_dataloader.dataset_size

        # Resolve log_num_samples: (0,1] = proportion, >1 = exact count
        raw_log_ns = eval_data_cfg["log_num_samples"]
        if 0 < raw_log_ns <= 1.0:
            log_eval_samples = int(eval_dataset_size * raw_log_ns)
        else:
            log_eval_samples = int(raw_log_ns)
        quick_eval_batches = max(1, (log_eval_samples + batch_size - 1) // batch_size)

        # Resolve eval_num_samples: (0,1] = proportion, >1 = exact count
        raw_eval_ns = eval_data_cfg["eval_num_samples"]
        if 0 < raw_eval_ns <= 1.0:
            full_eval_samples = int(eval_dataset_size * raw_eval_ns)
        else:
            full_eval_samples = int(raw_eval_ns)
        full_eval_batches = max(1, (full_eval_samples + batch_size - 1) // batch_size)

        logger.info(
            "Eval: %s (split=%s) | dataset_size=%d | full_eval=%d batches | quick_eval=%d batches",
            eval_data_cfg["data_name"],
            eval_data_cfg["split"],
            eval_dataset_size,
            full_eval_batches,
            quick_eval_batches,
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
        "Total steps: %d | Warmup: %d | LR: %s",
        total_steps,
        warmup_steps,
        learning_rate,
    )

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    best_eval_loss = float("inf")
    history = []

    if resume:
        resume_path = Path(resume)
        if resume_path.exists():
            start_epoch, global_step, best_loss = load_checkpoint(
                resume_path, builder, optimizer, scheduler
            )
            logger.info("Resumed from epoch %d, step %d", start_epoch, global_step)
        else:
            logger.warning("Resume path not found: %s", resume_path)

    config_save_path = log_dir / "config.json"
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)

    builder.train()

    for epoch in range(start_epoch, num_epochs):
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        epoch_num_batches = len(dataloader)
        epoch_mid_batch = epoch_num_batches // 2 if epoch_num_batches > 1 else -1

        for batch_idx, batch in enumerate(pbar):
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

            if batch.has_solution:
                # Tokenize questions and solutions for reasoning loss
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

                reasoning_loss = builder.compute_reasoning_loss(
                    pyramid, q_ids, q_mask, sol_ids
                )
                total_loss = (
                    total_loss + loss_weights["reasoning_loss_weight"] * reasoning_loss
                )
                loss_dict["reasoning"] = reasoning_loss.item()
                loss_dict["total"] = total_loss.item()

            total_loss.backward()

            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, gradient_clip)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            epoch_losses.append(loss_dict["total"])
            global_step += 1

            # Compute weighted individual losses
            w = {
                "recon": loss_dict["recon"] * loss_weights["recon_loss_weight"],
                "ordering": loss_dict["ordering"]
                * loss_weights["ordering_loss_weight"],
                "residual": loss_dict["residual"]
                * loss_weights["residual_loss_weight"],
            }
            if "reasoning" in loss_dict:
                w["reasoning"] = (
                    loss_dict["reasoning"] * loss_weights["reasoning_loss_weight"]
                )

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
                # Console: raw/weighted for each component
                reasoning_part = ""
                if "reasoning" in loss_dict:
                    reasoning_part = " reasoning=%.4f/%.4f" % (
                        loss_dict["reasoning"],
                        w["reasoning"],
                    )
                logger.info(
                    "Step %5d | total=%.4f recon=%.4f/%.4f ordering=%.4f/%.4f"
                    " residual=%.4f/%.4f%s lr=%.2e",
                    global_step,
                    loss_dict["total"],
                    loss_dict["recon"],
                    w["recon"],
                    loss_dict["ordering"],
                    w["ordering"],
                    loss_dict["residual"],
                    w["residual"],
                    reasoning_part,
                    lr,
                )
                # terminal_output.jsonl: raw + weighted
                terminal_entry = {
                    "step": global_step,
                    "epoch": epoch,
                    "total": round(loss_dict["total"], 6),
                    "recon": round(loss_dict["recon"], 6),
                    "recon_w": round(w["recon"], 6),
                    "ordering": round(loss_dict["ordering"], 6),
                    "ordering_w": round(w["ordering"], 6),
                    "residual": round(loss_dict["residual"], 6),
                    "residual_w": round(w["residual"], 6),
                    "lr": lr,
                }
                if "reasoning" in loss_dict:
                    terminal_entry["reasoning"] = round(loss_dict["reasoning"], 6)
                    terminal_entry["reasoning_w"] = round(w["reasoning"], 6)
                _log_terminal_entry(terminal_log_path, terminal_entry)

                # SwanLab: raw + weighted as separate metrics
                swanlab_metrics = {
                    "train/total_loss": loss_dict["total"],
                    "train/recon_raw": loss_dict["recon"],
                    "train/recon_weighted": w["recon"],
                    "train/ordering_raw": loss_dict["ordering"],
                    "train/ordering_weighted": w["ordering"],
                    "train/residual_raw": loss_dict["residual"],
                    "train/residual_weighted": w["residual"],
                    "train/lr": lr,
                }
                if "reasoning" in loss_dict:
                    swanlab_metrics["train/reasoning_raw"] = loss_dict["reasoning"]
                    swanlab_metrics["train/reasoning_weighted"] = w["reasoning"]
                swanlab.log(swanlab_metrics, step=global_step)

                # ── Quick eval (skip when full eval fires at same step) ──
                if eval_enabled and not (global_step % eval_interval == 0):
                    eval_losses, eval_samples = evaluate_builder(
                        builder,
                        eval_dataloader,
                        loss_weights,
                        ordering_loss_type,
                        device,
                        pyramid_cfg,
                        max_batches=quick_eval_batches,
                    )
                    _log_eval_results(
                        eval_losses,
                        eval_samples,
                        loss_weights,
                        "quick",
                        global_step,
                        logger,
                        terminal_log_path,
                        eval_history,
                        eval_sample_history,
                        log_dir,
                        "eval_quick",
                    )

            # ── Checkpoint scheduling ──────────────────────────
            #   checkpoint_clean=True  : save only at epoch-start (batch_idx==0)
            #                            and epoch-mid (batch_idx==epoch_mid_batch).
            #                            checkpoint_step_interval is ignored.
            #   checkpoint_clean=False : save per checkpoint_step_interval (legacy).
            # Best checkpoint is always tracked (overwrite-by-purge).
            save_regular = False
            save_tag = ""
            if checkpoint_clean:
                if batch_idx == 0:
                    save_regular = True
                    save_tag = "epoch-start"
                elif batch_idx == epoch_mid_batch:
                    save_regular = True
                    save_tag = "epoch-mid"
            else:
                if global_step % checkpoint_interval == 0:
                    save_regular = True

            if save_regular:
                avg_loss = (
                    sum(epoch_losses[-100:]) / len(epoch_losses[-100:])
                    if epoch_losses
                    else float("inf")
                )
                # Track best: overwrite any previous best file.
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    purge_best_checkpoints(checkpoint_dir, "checkpoint_best")
                    best_path = save_checkpoint(
                        builder,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        avg_loss,
                        checkpoint_dir,
                        filename=(f"checkpoint_best-epoch{epoch}-step{global_step}.pt"),
                    )
                    logger.info("Best checkpoint: %s", best_path.name)
                tag_part = f"-{save_tag}" if save_tag else ""
                path = save_checkpoint(
                    builder,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    avg_loss,
                    checkpoint_dir,
                    filename=(
                        f"checkpoint{tag_part}-epoch{epoch}-step{global_step}.pt"
                    ),
                )
                logger.info("Checkpoint: %s", path.name)

            # ── Full eval at eval_interval ──────────────────────
            if eval_enabled and global_step % eval_interval == 0:
                eval_losses, eval_samples = evaluate_builder(
                    builder,
                    eval_dataloader,
                    loss_weights,
                    ordering_loss_type,
                    device,
                    pyramid_cfg,
                    max_batches=full_eval_batches,
                )
                _log_eval_results(
                    eval_losses,
                    eval_samples,
                    loss_weights,
                    "full",
                    global_step,
                    logger,
                    terminal_log_path,
                    eval_history,
                    eval_sample_history,
                    log_dir,
                    "eval",
                )
                # Best eval checkpoint (always tracked; overwrite-by-purge).
                if eval_losses["total"] < best_eval_loss:
                    best_eval_loss = eval_losses["total"]
                    purge_best_checkpoints(checkpoint_dir, "checkpoint_best_eval")
                    best_eval_name = (
                        f"checkpoint_best_eval-epoch{epoch}-step{global_step}.pt"
                    )
                    ckpt = {
                        "epoch": epoch,
                        "step": global_step,
                        "eval_loss": eval_losses["total"],
                        "model_state_dict": builder.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    }
                    torch.save(ckpt, checkpoint_dir / best_eval_name)
                    logger.info(
                        "Best eval checkpoint: %s (eval_loss=%.4f)",
                        best_eval_name,
                        eval_losses["total"],
                    )

            # Record step history with raw + weighted losses
            step_record = {"step": global_step, "epoch": epoch, **loss_dict}
            step_record.update({f"{k}_w": v for k, v in w.items()})
            history.append(step_record)

        avg_epoch_loss = (
            sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("inf")
        )
        logger.info("Epoch %d avg loss: %.4f", epoch + 1, avg_epoch_loss)
        _log_terminal_entry(
            terminal_log_path,
            {"epoch": epoch, "avg_epoch_loss": round(avg_epoch_loss, 6)},
        )

        # ── SwanLab epoch-level logging ──────────────────────────
        swanlab.log(
            {"epoch/avg_loss": avg_epoch_loss, "epoch/epoch": epoch + 1},
            step=global_step,
        )

        # Epoch-end checkpoint is saved only in the legacy (non-clean) mode;
        # in clean mode the epoch-start of the next epoch already covers it.
        if not checkpoint_clean:
            path = save_checkpoint(
                builder,
                optimizer,
                scheduler,
                epoch + 1,
                global_step,
                avg_epoch_loss,
                checkpoint_dir,
                filename=f"checkpoint-epoch{epoch+1}-step{global_step}.pt",
            )
            logger.info("Epoch checkpoint: %s", path.name)

        with open(log_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, default=str)

        if eval_history:
            with open(log_dir / "eval_history.json", "w", encoding="utf-8") as f:
                json.dump(eval_history, f, indent=2, default=str)
        if eval_sample_history:
            with open(log_dir / "eval_sample_history.json", "w", encoding="utf-8") as f:
                json.dump(eval_sample_history, f, indent=2, default=str)

    logger.info("Training complete!")
    # By construction there is at most one best-train and one best-eval file
    # (purge_best_checkpoints removes the old one before every new best save).
    best_file = next(checkpoint_dir.glob("checkpoint_best-*.pt"), None)
    best_eval_file = next(checkpoint_dir.glob("checkpoint_best_eval-*.pt"), None)
    if best_file is not None:
        logger.info("Best train checkpoint: %s", best_file)
    if best_eval_file is not None:
        logger.info("Best eval checkpoint:  %s", best_eval_file)

    # ── Finish SwanLab run ────────────────────────────────────────
    swanlab.finish()
    logger.info("SwanLab run finished")


def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    yaml_config = load_config(str(config_path))

    # Merge resume flag from CLI if not in config
    if args.resume and not yaml_config["training"]["resume"]:
        yaml_config["training"]["resume"] = args.resume

    train_builder(yaml_config, config_path=config_path)


if __name__ == "__main__":
    main()
