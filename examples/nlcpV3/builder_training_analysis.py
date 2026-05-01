"""Visualize Builder training losses and learning rate from training logs.

Usage:
    python3 examples/nlcpV3/builder_training_analysis.py -c configs/nlcpV3/GSM8K/train_builder_Qwen2.5-0.5B_6level.yml
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ram.utils import load_config


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

    # Auto smoothing window: 1% of total steps, clamped to [10, 500]
    window = max(10, min(500, len(history) // 100))

    steps = np.array([r["step"] for r in history])
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
    ax.set_title("Total Loss (weighted sum)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    if eval_quick or eval_full:
        ax.legend(fontsize=7)

    # Weighted recon loss
    ax = axes[0, 1]
    ax.plot(steps, recon_w, alpha=0.15, color="tab:blue", linewidth=0.5)
    s = smooth(recon_w, window)
    ax.plot(steps[: len(s)] + window // 2, s, color="tab:blue", linewidth=1.5)
    if eval_quick:
        eq_steps = np.array([r["step"] for r in eval_quick])
        eq_vals = np.array([r["recon"] * w_recon for r in eval_quick])
        ax.plot(eq_steps, eq_vals, ".", color="tab:cyan", markersize=2, alpha=0.5)
    if eval_full:
        ef_steps = np.array([r["step"] for r in eval_full])
        ef_vals = np.array([r["recon"] * w_recon for r in eval_full])
        ax.plot(ef_steps, ef_vals, "s", color="tab:red", markersize=4, alpha=0.8)
    ax.set_title(f"Recon Loss (×{w_recon})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # Weighted ordering loss
    ax = axes[1, 0]
    ax.plot(steps, ordering_w, alpha=0.15, color="tab:orange", linewidth=0.5)
    s = smooth(ordering_w, window)
    ax.plot(steps[: len(s)] + window // 2, s, color="tab:orange", linewidth=1.5)
    if eval_quick:
        eq_steps = np.array([r["step"] for r in eval_quick])
        eq_vals = np.array([r["ordering"] * w_ordering for r in eval_quick])
        ax.plot(eq_steps, eq_vals, ".", color="tab:cyan", markersize=2, alpha=0.5)
    if eval_full:
        ef_steps = np.array([r["step"] for r in eval_full])
        ef_vals = np.array([r["ordering"] * w_ordering for r in eval_full])
        ax.plot(ef_steps, ef_vals, "s", color="tab:red", markersize=4, alpha=0.8)
    ax.set_title(f"Ordering Loss (×{w_ordering})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # Weighted residual loss
    ax = axes[1, 1]
    ax.plot(steps, residual_w, alpha=0.15, color="tab:green", linewidth=0.5)
    s = smooth(residual_w, window)
    ax.plot(steps[: len(s)] + window // 2, s, color="tab:green", linewidth=1.5)
    if eval_quick:
        eq_steps = np.array([r["step"] for r in eval_quick])
        eq_vals = np.array([r["residual"] * w_residual for r in eval_quick])
        ax.plot(eq_steps, eq_vals, ".", color="tab:cyan", markersize=2, alpha=0.5)
    if eval_full:
        ef_steps = np.array([r["step"] for r in eval_full])
        ef_vals = np.array([r["residual"] * w_residual for r in eval_full])
        ax.plot(ef_steps, ef_vals, "s", color="tab:red", markersize=4, alpha=0.8)
    ax.set_title(f"Residual Loss (×{w_residual})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # Weighted reasoning loss
    ax = axes[2, 0]
    ax.plot(steps, reasoning_w, alpha=0.15, color="tab:red", linewidth=0.5)
    s = smooth(reasoning_w, window)
    ax.plot(steps[: len(s)] + window // 2, s, color="tab:red", linewidth=1.5)
    if eval_quick:
        eq_steps = np.array([r["step"] for r in eval_quick])
        eq_vals = np.array([r.get("reasoning", 0.0) * w_reasoning for r in eval_quick])
        ax.plot(eq_steps, eq_vals, ".", color="tab:cyan", markersize=2, alpha=0.5)
    if eval_full:
        ef_steps = np.array([r["step"] for r in eval_full])
        ef_vals = np.array([r.get("reasoning", 0.0) * w_reasoning for r in eval_full])
        ax.plot(ef_steps, ef_vals, "s", color="tab:red", markersize=4, alpha=0.8)
    ax.set_title(f"Reasoning Loss (×{w_reasoning})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

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
