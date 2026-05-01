"""Visualize Builder training losses and learning rate from training logs.

Usage:
    python3 examples/nlcpV4/builder_training_analysis.py -c configs/nlcpV4/GSM8K/train_builder_Qwen2.5-0.5B_6level.yml

Behavior:
    - Loads training_history.json, terminal_output.jsonl, eval_history.json
    - If eval_history exists: overlays eval curves (quick/full) on each loss subplot
    - If eval_history is empty: loads best checkpoint, runs full evaluation on
      test set, and plots the result as single-point markers on each subplot
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

from lmbase.utils.env_tools import get_device
from nlcpV4.concept_builder import ConceptPyramidBuilder
from nlcpV4.data_loader import NLCPV4DataLoader
from nlcpV4.eval_builder import evaluate_builder
from ram.utils import load_config

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Builder training logs")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to YAML config file"
    )
    return parser.parse_args()


def smooth(values, window):
    """Simple moving average smoothing."""
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def load_training_history(log_dir: Path):
    """Load training_history.json → list of dicts."""
    path = log_dir / "training_history.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_eval_history(log_dir: Path):
    """Load eval_history.json → list of dicts, or empty list if not found."""
    path = log_dir / "eval_history.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_checkpoint_eval(config: dict, project_root: Path) -> dict | None:
    """Load best checkpoint and run full evaluation on the test set.

    Returns averaged loss dict (keys: total, recon, ordering, residual,
    optionally reasoning), or None if no checkpoint is found.
    """
    # Locate best checkpoint
    checkpoint_dir = Path(config["log"]["checkpoint_path"])
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = project_root / checkpoint_dir

    # Prefer best_eval checkpoint, then best training checkpoint
    ckpt_path = None
    for name in ["checkpoint_best_eval.pt", "checkpoint_best.pt"]:
        candidate = checkpoint_dir / name
        if candidate.exists():
            ckpt_path = candidate
            break

    if ckpt_path is None:
        logger.warning("No best checkpoint found in %s — skipping eval", checkpoint_dir)
        return None

    logger.info("Loading checkpoint: %s", ckpt_path)
    device = str(get_device("auto"))

    # Build model from config
    builder = ConceptPyramidBuilder(config)
    builder.to(device)

    # Load checkpoint weights
    ckpt = torch.load(ckpt_path, map_location=device)
    builder.load_state_dict(ckpt["model_state_dict"])
    logger.info(
        "Checkpoint loaded (step=%s, %s=%.4f)",
        ckpt.get("step", "?"),
        "eval_loss" if "eval_loss" in ckpt else "loss",
        ckpt.get("eval_loss", ckpt.get("loss", 0.0)),
    )

    # Set up eval dataloader
    eval_cfg = config["evaluation"]
    eval_data_cfg = eval_cfg["data"]
    train_cfg = config["training"]
    env_cfg = config["environment"]

    eval_dataloader = NLCPV4DataLoader(
        data_cfg=eval_data_cfg,
        batch_size=train_cfg["batch_size"],
        include_solution=True,
        shuffle=False,
        drop_last=False,
        num_workers=env_cfg["dataloader_num_workers"],
    )
    logger.info(
        "Eval dataset: %s (split=%s, size=%d)",
        eval_data_cfg["data_name"],
        eval_data_cfg["split"],
        eval_dataloader.dataset_size,
    )

    # Run full evaluation (all batches)
    loss_weights = train_cfg["loss_weights"]
    ordering_loss_type = train_cfg["ordering_loss_type"]

    eval_losses = evaluate_builder(
        builder=builder,
        eval_dataloader=eval_dataloader,
        loss_weights=loss_weights,
        ordering_loss_type=ordering_loss_type,
        # Evaluate all batches
        max_batches=0,
    )

    logger.info(
        "Checkpoint eval: total=%.4f recon=%.4f ordering=%.4f residual=%.4f%s",
        eval_losses["total"],
        eval_losses["recon"],
        eval_losses["ordering"],
        eval_losses["residual"],
        (
            " reasoning=%.4f" % eval_losses["reasoning"]
            if "reasoning" in eval_losses
            else ""
        ),
    )
    return eval_losses


def load_terminal_output(log_dir: Path):
    """Load terminal_output.jsonl → list of dicts (has lr)."""
    path = log_dir / "terminal_output.jsonl"
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _plot_eval_on_ax(
    ax,
    eval_quick,
    eval_full,
    key,
    weight,
    ckpt_eval,
    last_step,
):
    """Overlay eval data on a training loss subplot.

    If eval_quick/eval_full exist, plot them as scatter points.
    If both are empty but ckpt_eval is available, plot a single
    marker + horizontal dashed line.
    """
    has_history = bool(eval_quick) or bool(eval_full)

    if eval_quick:
        eq_steps = np.array([r["step"] for r in eval_quick])
        eq_vals = np.array([r.get(key, 0.0) * weight for r in eval_quick])
        ax.plot(
            eq_steps,
            eq_vals,
            ".",
            color="tab:cyan",
            markersize=2,
            alpha=0.5,
            label="eval(quick)",
        )
    if eval_full:
        ef_steps = np.array([r["step"] for r in eval_full])
        ef_vals = np.array([r.get(key, 0.0) * weight for r in eval_full])
        ax.plot(
            ef_steps,
            ef_vals,
            "s",
            color="tab:red",
            markersize=4,
            alpha=0.8,
            label="eval(full)",
        )

    # Checkpoint eval fallback: single point when no eval history
    if not has_history and ckpt_eval is not None:
        val = ckpt_eval.get(key, 0.0) * weight
        ax.axhline(y=val, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.plot(
            last_step,
            val,
            "*",
            color="tab:red",
            markersize=12,
            zorder=5,
            label="best ckpt eval",
        )


def _plot_eval_total_on_ax(
    ax,
    eval_quick,
    eval_full,
    ckpt_eval,
    last_step,
):
    """Overlay eval total loss (already weighted) on total-loss subplot."""
    has_history = bool(eval_quick) or bool(eval_full)

    if eval_quick:
        eq_steps = np.array([r["step"] for r in eval_quick])
        eq_total = np.array([r["total"] for r in eval_quick])
        ax.plot(
            eq_steps,
            eq_total,
            ".",
            color="tab:cyan",
            markersize=2,
            alpha=0.5,
            label="eval(quick)",
        )
    if eval_full:
        ef_steps = np.array([r["step"] for r in eval_full])
        ef_total = np.array([r["total"] for r in eval_full])
        ax.plot(
            ef_steps,
            ef_total,
            "s",
            color="tab:red",
            markersize=4,
            alpha=0.8,
            label="eval(full)",
        )

    if not has_history and ckpt_eval is not None:
        val = ckpt_eval["total"]
        ax.axhline(y=val, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.plot(
            last_step,
            val,
            "*",
            color="tab:red",
            markersize=12,
            zorder=5,
            label="best ckpt eval",
        )


def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_config(str(config_path))

    log_dir = Path(config["log"]["log_path"])
    if not log_dir.is_absolute():
        log_dir = PROJECT_ROOT / log_dir

    loss_weights = config["training"]["loss_weights"]
    w_recon = loss_weights["recon_loss_weight"]
    w_ordering = loss_weights["ordering_loss_weight"]
    w_residual = loss_weights["residual_loss_weight"]
    w_reasoning = loss_weights["reasoning_loss_weight"]

    experiment_name = f"{config_path.parent.name}-{config_path.stem}"

    # ── Load data ─────────────────────────────────────────────────
    history = load_training_history(log_dir)
    terminal = load_terminal_output(log_dir)
    eval_hist = load_eval_history(log_dir)

    # Separate quick and full eval histories
    eval_quick = [r for r in eval_hist if r.get("eval_type") == "quick"]
    eval_full = [r for r in eval_hist if r.get("eval_type") == "full"]

    # If no eval history, try checkpoint evaluation fallback
    ckpt_eval = None
    if not eval_hist:
        print("No eval_history.json found — running checkpoint evaluation...")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        ckpt_eval = run_checkpoint_eval(config, PROJECT_ROOT)

    # Auto smoothing window: 1% of total steps, clamped to [10, 500]
    window = max(10, min(500, len(history) // 100))

    steps = np.array([r["step"] for r in history])
    last_step = int(steps[-1]) if len(steps) > 0 else 0
    # Weighted losses
    recon_w = np.array([r["recon"] * w_recon for r in history])
    ordering_w = np.array([r["ordering"] * w_ordering for r in history])
    residual_w = np.array([r["residual"] * w_residual for r in history])
    reasoning_w = np.array([r.get("reasoning", 0.0) * w_reasoning for r in history])
    total = np.array([r["total"] for r in history])

    lr_steps = np.array([r["step"] for r in terminal if "lr" in r])
    lr_values = np.array([r["lr"] for r in terminal if "lr" in r])

    # ── Figure 1: All weighted losses ─────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(f"Builder Training Analysis: {experiment_name}", fontsize=14, y=0.98)

    # Total loss
    ax = axes[0, 0]
    ax.plot(steps, total, alpha=0.15, color="black", linewidth=0.5)
    s = smooth(total, window)
    ax.plot(steps[: len(s)] + window // 2, s, color="black", linewidth=1.5)
    _plot_eval_total_on_ax(ax, eval_quick, eval_full, ckpt_eval, last_step)
    ax.set_title("Total Loss (weighted sum)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    if eval_quick or eval_full or ckpt_eval:
        ax.legend(fontsize=7)

    # Weighted recon loss
    ax = axes[0, 1]
    ax.plot(steps, recon_w, alpha=0.15, color="tab:blue", linewidth=0.5)
    s = smooth(recon_w, window)
    ax.plot(steps[: len(s)] + window // 2, s, color="tab:blue", linewidth=1.5)
    _plot_eval_on_ax(ax, eval_quick, eval_full, "recon", w_recon, ckpt_eval, last_step)
    ax.set_title(f"Recon Loss (×{w_recon})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    if eval_quick or eval_full or ckpt_eval:
        ax.legend(fontsize=7)

    # Weighted ordering loss
    ax = axes[1, 0]
    ax.plot(steps, ordering_w, alpha=0.15, color="tab:orange", linewidth=0.5)
    s = smooth(ordering_w, window)
    ax.plot(steps[: len(s)] + window // 2, s, color="tab:orange", linewidth=1.5)
    _plot_eval_on_ax(
        ax, eval_quick, eval_full, "ordering", w_ordering, ckpt_eval, last_step
    )
    ax.set_title(f"Ordering Loss (×{w_ordering})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    if eval_quick or eval_full or ckpt_eval:
        ax.legend(fontsize=7)

    # Weighted residual loss
    ax = axes[1, 1]
    ax.plot(steps, residual_w, alpha=0.15, color="tab:green", linewidth=0.5)
    s = smooth(residual_w, window)
    ax.plot(steps[: len(s)] + window // 2, s, color="tab:green", linewidth=1.5)
    _plot_eval_on_ax(
        ax, eval_quick, eval_full, "residual", w_residual, ckpt_eval, last_step
    )
    ax.set_title(f"Residual Loss (×{w_residual})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    if eval_quick or eval_full or ckpt_eval:
        ax.legend(fontsize=7)

    # Weighted reasoning loss
    ax = axes[2, 0]
    ax.plot(steps, reasoning_w, alpha=0.15, color="tab:red", linewidth=0.5)
    s = smooth(reasoning_w, window)
    ax.plot(steps[: len(s)] + window // 2, s, color="tab:red", linewidth=1.5)
    _plot_eval_on_ax(
        ax, eval_quick, eval_full, "reasoning", w_reasoning, ckpt_eval, last_step
    )
    ax.set_title(f"Reasoning Loss (×{w_reasoning})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    if eval_quick or eval_full or ckpt_eval:
        ax.legend(fontsize=7)

    # Learning rate
    ax = axes[2, 1]
    ax.plot(lr_steps, lr_values, color="tab:purple", linewidth=1.5)
    ax.set_title("Learning Rate")
    ax.set_xlabel("Step")
    ax.set_ylabel("LR")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    plt.tight_layout()

    # ── Figure 2: All weighted losses overlaid ────────────────────
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    ax2.set_title(f"All Weighted Losses: {experiment_name}")

    for data, label, color in [
        (recon_w, f"recon (×{w_recon})", "tab:blue"),
        (ordering_w, f"ordering (×{w_ordering})", "tab:orange"),
        (residual_w, f"residual (×{w_residual})", "tab:green"),
        (reasoning_w, f"reasoning (×{w_reasoning})", "tab:red"),
        (total, "total", "black"),
    ]:
        ax2.plot(steps, data, alpha=0.08, color=color, linewidth=0.5)
        s = smooth(data, window)
        ax2.plot(
            steps[: len(s)] + window // 2, s, label=label, linewidth=1.5, color=color
        )

    # Add checkpoint eval single-point markers to overlay if applicable
    if not eval_hist and ckpt_eval is not None:
        for key, weight, color in [
            ("recon", w_recon, "tab:blue"),
            ("ordering", w_ordering, "tab:orange"),
            ("residual", w_residual, "tab:green"),
            ("reasoning", w_reasoning, "tab:red"),
            ("total", 1.0, "black"),
        ]:
            val = ckpt_eval.get(key, 0.0) * weight
            ax2.plot(
                last_step,
                val,
                "*",
                color=color,
                markersize=14,
                zorder=5,
            )
            ax2.axhline(
                y=val,
                color=color,
                linestyle="--",
                linewidth=0.8,
                alpha=0.4,
            )

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    fig.savefig(log_dir / "training_losses.png", dpi=150, bbox_inches="tight")
    fig2.savefig(log_dir / "training_losses_overlay.png", dpi=150, bbox_inches="tight")
    print("Saved to %s" % log_dir)


if __name__ == "__main__":
    main()
