"""Visualize Builder training losses and learning rate from training logs.

Usage:
    # Single experiment
    python3 examples/nlcpV4/builder_training_analysis.py \
        -m builder -d GSM8K -e Qwen2.5-0.5B_6level

    # All experiments for a module + dataset
    python3 examples/nlcpV4/builder_training_analysis.py \
        -m builder -d GSM8K -e all

Arguments:
    -m / --module      Module name: 'builder' or 'predictor'.
    -d / --dataset     Dataset name (directory under configs/nlcpV4/).
    -e / --experiment  Config stem after 'train_{module}_'
                       (e.g. 'Qwen2.5-0.5B_6level') or 'all' to process
                       every matching config under the dataset.
    -o / --overlap     If true (default), overwrite existing analysis
                       outputs. Use ``--no-overlap`` to skip configs whose
                       outputs already exist.

Behavior:
    For each selected config:
    1. If log directory / training_history.json is missing -> skip with
       [SKIP NO-DATA].
    2. If all analysis outputs already exist AND --no-overlap is set ->
       skip with [SKIP EXISTS]. With the default ``--overlap``, existing
       outputs are overwritten.
    3. Otherwise run the analysis: overlay eval curves (quick/full), or
       fall back to checkpoint eval when eval_history.json is empty.
    A compact status table is printed at the end.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

# Lightweight imports only. Heavy imports (torch, ConceptPyramidBuilder,
# NLCPV4DataLoader, evaluate_builder) are deferred into `run_checkpoint_eval`
# so that analysis with an existing eval_history.json does not require
# torch / transformers / swanlab / nlcpV4 package dependencies.
from ram.utils import load_config

logger = logging.getLogger(__name__)

# --- Batch-mode constants ------------------------------------------
CONFIGS_ROOT = PROJECT_ROOT / "configs" / "nlcpV4"
VALID_MODULES = {"builder", "predictor"}
ALL_KEYWORD = "all"
# The six PNGs produced by a successful analysis run (weighted + raw).
# Presence of ALL six is treated as "already analyzed". They live in
# <experiment>/train_analysis/, not in the logs/ directory.
ANALYSIS_OUTPUT_DIR_NAME = "train_analysis"
ANALYSIS_OUTPUTS = (
    "training_losses.png",
    "training_losses_overlay.png",
    "eval_losses_overlay.png",
    "training_losses_raw.png",
    "training_losses_overlay_raw.png",
    "eval_losses_overlay_raw.png",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze Builder training logs (batch mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--module",
        required=True,
        choices=sorted(VALID_MODULES),
        help="Module name: 'builder' or 'predictor'.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        help="Dataset name (directory under configs/nlcpV4/).",
    )
    parser.add_argument(
        "-e",
        "--experiment",
        required=True,
        help=(
            "Config stem after 'train_{module}_' (e.g. 'Qwen2.5-0.5B_6level'), "
            "or 'all' to process every matching config under the dataset."
        ),
    )
    parser.add_argument(
        "-o",
        "--overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true (default), overwrite existing analysis outputs. "
            "Pass --no-overlap to skip configs whose outputs already exist."
        ),
    )
    return parser.parse_args()


def discover_configs(module: str, dataset: str, experiment: str) -> list[Path]:
    """Resolve (-m, -d, -e) into a list of config paths.

    Config file path convention:
        configs/nlcpV4/{dataset}/train_{module}_{experiment}.yml
    """
    dataset_dir = CONFIGS_ROOT / dataset
    if not dataset_dir.is_dir():
        print(f"[ERROR] Dataset dir not found: {dataset_dir}")
        return []

    prefix = f"train_{module}_"
    if experiment == ALL_KEYWORD:
        return sorted(dataset_dir.glob(f"{prefix}*.yml"))

    p = dataset_dir / f"{prefix}{experiment}.yml"
    if not p.is_file():
        print(f"[ERROR] Config file not found: {p}")
        return []
    return [p]


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
    # Lazy imports: only load heavy deps when actually running inference.
    import torch

    from lmbase.utils.env_tools import get_device
    from nlcpV4.concept_builder import ConceptPyramidBuilder
    from nlcpV4.data_loader import NLCPV4DataLoader
    from nlcpV4.eval_builder import evaluate_builder

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

    # Discard reasoning_texts — not needed for plotting
    eval_losses, _ = evaluate_builder(
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

    If eval_quick/eval_full exist, plot them as line + marker so they
    are visually distinct from the solid training curve.
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
            linestyle="--",
            marker=".",
            color="tab:cyan",
            linewidth=1.0,
            markersize=4,
            alpha=0.8,
            label="eval(quick)",
        )
    if eval_full:
        ef_steps = np.array([r["step"] for r in eval_full])
        ef_vals = np.array([r.get(key, 0.0) * weight for r in eval_full])
        ax.plot(
            ef_steps,
            ef_vals,
            linestyle=":",
            marker="s",
            color="tab:red",
            linewidth=1.2,
            markersize=6,
            alpha=0.9,
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
    total_getter=None,
):
    """Overlay eval total loss on the total-loss subplot.

    ``total_getter(record)`` extracts the per-record total (defaults to
    ``r["total"]`` which is the *weighted* total). For raw mode callers
    pass a getter that sums the raw per-component values. The same getter
    is applied to ``ckpt_eval`` (which shares the record schema).
    """
    if total_getter is None:
        total_getter = lambda r: r["total"]  # noqa: E731
    has_history = bool(eval_quick) or bool(eval_full)

    if eval_quick:
        eq_steps = np.array([r["step"] for r in eval_quick])
        eq_total = np.array([total_getter(r) for r in eval_quick])
        ax.plot(
            eq_steps,
            eq_total,
            linestyle="--",
            marker=".",
            color="tab:cyan",
            linewidth=1.0,
            markersize=4,
            alpha=0.8,
            label="eval(quick)",
        )
    if eval_full:
        ef_steps = np.array([r["step"] for r in eval_full])
        ef_total = np.array([total_getter(r) for r in eval_full])
        ax.plot(
            ef_steps,
            ef_total,
            linestyle=":",
            marker="s",
            color="tab:red",
            linewidth=1.2,
            markersize=6,
            alpha=0.9,
            label="eval(full)",
        )

    if not has_history and ckpt_eval is not None:
        val = total_getter(ckpt_eval)
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


def _build_figures(
    *,
    mode: str,
    output_dir: Path,
    experiment_name: str,
    steps,
    last_step,
    window,
    comp_arrays,
    total_array,
    comp_weights,
    eval_total_getter,
    eval_quick,
    eval_full,
    ckpt_eval,
    lr_steps,
    lr_values,
) -> None:
    """Build the 3 PNGs for one loss mode and save them under ``output_dir``.

    ``mode`` is 'weighted' or 'raw' and controls filename suffix + titles.
    ``comp_arrays`` / ``total_array`` are already in the chosen mode.
    ``comp_weights`` is applied to raw eval per-component values (1.0 in
    raw mode). ``eval_total_getter`` extracts a per-record total in the
    chosen mode.
    """
    if mode == "weighted":
        suffix = ""
        mode_label = "Weighted"

        def comp_title(key):
            return f"{key.capitalize()} Loss (\u00d7{comp_weights[key]})"

        total_subplot_title = "Total Loss (weighted sum)"
        overlay_title = f"All Weighted Losses: {experiment_name}"
        eval_overlay_title = f"All Weighted Eval Losses: {experiment_name}"

        def overlay_label(key):
            if key == "total":
                return "total"
            return f"{key} (\u00d7{comp_weights[key]})"

    elif mode == "raw":
        suffix = "_raw"
        mode_label = "Raw"

        def comp_title(key):
            return f"{key.capitalize()} Loss (raw)"

        total_subplot_title = "Total Loss (raw sum)"
        overlay_title = f"All Raw Losses: {experiment_name}"
        eval_overlay_title = f"All Raw Eval Losses: {experiment_name}"

        def overlay_label(key):
            return key

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ── Figure 1: 3x2 grid of component losses + LR ──────────────
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(f"Builder Training Analysis ({mode_label}): {experiment_name}", y=0.98)

    # Total loss (weighted-sum or raw-sum depending on mode)
    ax = axes[0, 0]
    ax.plot(steps, total_array, alpha=0.15, color="black", linewidth=0.5)
    s = smooth(total_array, window)
    ax.plot(
        steps[: len(s)] + window // 2,
        s,
        color="black",
        linewidth=1.5,
        label="train",
    )
    _plot_eval_total_on_ax(
        ax,
        eval_quick,
        eval_full,
        ckpt_eval,
        last_step,
        total_getter=eval_total_getter,
    )
    ax.set_title(total_subplot_title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    grid_layout = [
        ("recon", axes[0, 1], "tab:blue"),
        ("ordering", axes[1, 0], "tab:orange"),
        ("residual", axes[1, 1], "tab:green"),
        ("reasoning", axes[2, 0], "tab:red"),
    ]
    for key, ax, color in grid_layout:
        data = comp_arrays[key]
        ax.plot(steps, data, alpha=0.15, color=color, linewidth=0.5)
        s = smooth(data, window)
        ax.plot(
            steps[: len(s)] + window // 2,
            s,
            color=color,
            linewidth=1.5,
            label="train",
        )
        _plot_eval_on_ax(
            ax,
            eval_quick,
            eval_full,
            key,
            comp_weights[key],
            ckpt_eval,
            last_step,
        )
        ax.set_title(comp_title(key))
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Learning rate
    ax = axes[2, 1]
    ax.plot(lr_steps, lr_values, color="tab:purple", linewidth=1.5)
    ax.set_title("Learning Rate")
    ax.set_xlabel("Step")
    ax.set_ylabel("LR")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    plt.tight_layout()

    # ── Figure 2: overlay ─────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    ax2.set_title(overlay_title)

    overlay_items = [
        ("recon", comp_arrays["recon"], "tab:blue"),
        ("ordering", comp_arrays["ordering"], "tab:orange"),
        ("residual", comp_arrays["residual"], "tab:green"),
        ("reasoning", comp_arrays["reasoning"], "tab:red"),
        ("total", total_array, "black"),
    ]
    for key, data, color in overlay_items:
        ax2.plot(steps, data, alpha=0.08, color=color, linewidth=0.5)
        s = smooth(data, window)
        ax2.plot(
            steps[: len(s)] + window // 2,
            s,
            label=overlay_label(key),
            linewidth=1.5,
            color=color,
        )

    # Checkpoint-eval markers when no eval history
    if not eval_quick and not eval_full and ckpt_eval is not None:
        for key, _data, color in overlay_items:
            if key == "total":
                val = eval_total_getter(ckpt_eval)
            else:
                val = ckpt_eval.get(key, 0.0) * comp_weights[key]
            ax2.plot(last_step, val, "*", color=color, markersize=14, zorder=5)
            ax2.axhline(y=val, color=color, linestyle="--", linewidth=0.8, alpha=0.4)

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    # ── Figure 3: eval overlay ────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    ax3.set_title(eval_overlay_title)

    eval_plot_items = [
        ("recon", "tab:blue"),
        ("ordering", "tab:orange"),
        ("residual", "tab:green"),
        ("reasoning", "tab:red"),
        ("total", "black"),
    ]
    has_any_eval = bool(eval_quick) or bool(eval_full)

    def _eval_values(records, key):
        if key == "total":
            return np.array([eval_total_getter(r) for r in records])
        return np.array([r.get(key, 0.0) * comp_weights[key] for r in records])

    if eval_quick:
        eq_steps = np.array([r["step"] for r in eval_quick])
        for key, color in eval_plot_items:
            ax3.plot(
                eq_steps,
                _eval_values(eval_quick, key),
                linestyle="--",
                marker=".",
                color=color,
                linewidth=1.0,
                markersize=4,
                alpha=0.8,
                label=f"{overlay_label(key)} [quick]",
            )
    if eval_full:
        ef_steps = np.array([r["step"] for r in eval_full])
        for key, color in eval_plot_items:
            ax3.plot(
                ef_steps,
                _eval_values(eval_full, key),
                linestyle=":",
                marker="s",
                color=color,
                linewidth=1.2,
                markersize=6,
                alpha=0.9,
                label=f"{overlay_label(key)} [full]",
            )

    if not has_any_eval and ckpt_eval is not None:
        for key, color in eval_plot_items:
            if key == "total":
                val = eval_total_getter(ckpt_eval)
            else:
                val = ckpt_eval.get(key, 0.0) * comp_weights[key]
            ax3.plot(
                last_step,
                val,
                "*",
                color=color,
                markersize=14,
                zorder=5,
                label=f"{overlay_label(key)} [best ckpt]",
            )
            ax3.axhline(y=val, color=color, linestyle="--", linewidth=0.8, alpha=0.4)

    ax3.set_xlabel("Step")
    ax3.set_ylabel("Loss")
    if has_any_eval or ckpt_eval is not None:
        ax3.legend(ncol=2)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()

    fig.savefig(
        output_dir / f"training_losses{suffix}.png", dpi=150, bbox_inches="tight"
    )
    fig2.savefig(
        output_dir / f"training_losses_overlay{suffix}.png",
        dpi=150,
        bbox_inches="tight",
    )
    fig3.savefig(
        output_dir / f"eval_losses_overlay{suffix}.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    plt.close(fig2)
    plt.close(fig3)


def _run_builder_analysis(config_path: Path) -> None:
    """Run the full analysis for a single config and write 6 PNGs to
    ``<experiment>/train_analysis/`` (sibling of ``logs/``).

    For each pass (weighted and raw) three figures are produced:
    training_losses, training_losses_overlay, eval_losses_overlay.

    Precondition: caller has already verified that training_history.json
    exists under config.log.log_path.
    """
    config = load_config(str(config_path))

    # --- Plot styling: larger, bold titles/labels/legend/ticks ----
    # Set once via rcParams so every set_title / set_xlabel /
    # set_ylabel / legend call inherits the same typography. Safe to
    # re-apply on each batch iteration (idempotent).
    plt.rcParams.update(
        {
            "font.weight": "bold",
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "axes.labelweight": "bold",
            "figure.titlesize": 18,
            "figure.titleweight": "bold",
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            # Strip top and right border on every Axes (applies to
            # fig's subplots as well as ax2 / ax3 created later).
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

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

    # ── Output directory: sibling of logs/ ───────────────────────
    # Saved under <experiment>/train_analysis/ to keep logs/ lightweight.
    output_dir = log_dir.parent / ANALYSIS_OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Raw component arrays (mirror the weighted arrays) ─────────
    recon_raw = np.array([r["recon"] for r in history])
    ordering_raw = np.array([r["ordering"] for r in history])
    residual_raw = np.array([r["residual"] for r in history])
    reasoning_raw = np.array([r.get("reasoning", 0.0) for r in history])
    # Raw total = element-wise sum of raw per-component values.
    total_raw = recon_raw + ordering_raw + residual_raw + reasoning_raw

    _raw_comps = ("recon", "ordering", "residual", "reasoning")

    # ── Weighted pass ─────────────────────────────────────────
    _build_figures(
        mode="weighted",
        output_dir=output_dir,
        experiment_name=experiment_name,
        steps=steps,
        last_step=last_step,
        window=window,
        comp_arrays={
            "recon": recon_w,
            "ordering": ordering_w,
            "residual": residual_w,
            "reasoning": reasoning_w,
        },
        total_array=total,
        comp_weights={
            "recon": w_recon,
            "ordering": w_ordering,
            "residual": w_residual,
            "reasoning": w_reasoning,
        },
        eval_total_getter=lambda r: r["total"],
        eval_quick=eval_quick,
        eval_full=eval_full,
        ckpt_eval=ckpt_eval,
        lr_steps=lr_steps,
        lr_values=lr_values,
    )

    # ── Raw pass ─────────────────────────────────────────────
    _build_figures(
        mode="raw",
        output_dir=output_dir,
        experiment_name=experiment_name,
        steps=steps,
        last_step=last_step,
        window=window,
        comp_arrays={
            "recon": recon_raw,
            "ordering": ordering_raw,
            "residual": residual_raw,
            "reasoning": reasoning_raw,
        },
        total_array=total_raw,
        comp_weights={k: 1.0 for k in _raw_comps},
        eval_total_getter=lambda r: sum(r.get(k, 0.0) for k in _raw_comps),
        eval_quick=eval_quick,
        eval_full=eval_full,
        ckpt_eval=ckpt_eval,
        lr_steps=lr_steps,
        lr_values=lr_values,
    )

    print("Saved to %s" % output_dir)


def analyze_one(config_path: Path, overlap: bool = True) -> tuple[str, str]:
    """Analyze a single config. Returns (status, detail) tuple.

    Args:
        config_path: YAML config file to analyze.
        overlap: If True (default), always re-run even when outputs
            already exist (overwriting them). If False, skip configs
            whose outputs already exist.

    status is one of:
      - 'analyzed'        : analysis ran and PNGs were written.
      - 'skip_no_data'    : training_history.json missing (training not started
                            or log_dir absent); skipped.
      - 'skip_exists'     : all output PNGs already exist AND overlap=False;
                            skipped.
      - 'error'           : unexpected error (detail contains the message).
    """
    try:
        config = load_config(str(config_path))
    except Exception as exc:  # noqa: BLE001
        return "error", f"load_config failed: {exc}"

    log_dir = Path(config["log"]["log_path"])
    if not log_dir.is_absolute():
        log_dir = PROJECT_ROOT / log_dir

    training_history = log_dir / "training_history.json"
    if not training_history.is_file():
        return "skip_no_data", f"training_history.json missing at {log_dir}"

    output_dir = log_dir.parent / ANALYSIS_OUTPUT_DIR_NAME
    if not overlap and all((output_dir / name).is_file() for name in ANALYSIS_OUTPUTS):
        return "skip_exists", f"all outputs already exist at {output_dir}"
    try:
        _run_builder_analysis(config_path)
    except Exception as exc:  # noqa: BLE001
        # Close any half-drawn figures to avoid leaks in batch mode.
        plt.close("all")
        return "error", f"{type(exc).__name__}: {exc}"

    return "analyzed", f"wrote {len(ANALYSIS_OUTPUTS)} PNGs to {output_dir}"


def _print_summary(rows: list[tuple[str, str, str]]) -> None:
    """Print a compact status table. rows = [(status, config_stem, detail), ...]."""
    if not rows:
        print("[SUMMARY] No configs processed.")
        return

    status_w = max(len("Status"), *(len(r[0]) for r in rows))
    name_w = max(len("Config"), *(len(r[1]) for r in rows))

    sep = "-" * (status_w + name_w + 5)
    print("=" * (status_w + name_w + 5))
    print("Summary")
    print("=" * (status_w + name_w + 5))
    print(f"{'Status':<{status_w}} | {'Config':<{name_w}}")
    print(sep)
    for status, name, _detail in rows:
        print(f"{status:<{status_w}} | {name:<{name_w}}")
    print(sep)

    counts: dict[str, int] = {}
    for status, _name, _detail in rows:
        counts[status] = counts.get(status, 0) + 1
    parts = [f"{k}={v}" for k, v in sorted(counts.items())]
    print("Totals: " + "  ".join(parts))


def main():
    args = parse_args()
    module: str = args.module
    dataset: str = args.dataset
    experiment: str = args.experiment
    overlap: bool = args.overlap

    configs = discover_configs(module, dataset, experiment)
    if not configs:
        return 1

    print(
        f"[ANALYZE] module={module} dataset={dataset} "
        f"experiment={experiment} overlap={overlap}  "
        f"({len(configs)} config file(s))"
    )
    for p in configs:
        print(
            f"  - {p.relative_to(PROJECT_ROOT) if p.is_relative_to(PROJECT_ROOT) else p}"
        )
    print()

    rows: list[tuple[str, str, str]] = []
    for cfg_path in configs:
        print("=" * 70)
        print(f"[CONFIG] {cfg_path.name}")
        print("=" * 70)
        status, detail = analyze_one(cfg_path, overlap=overlap)
        if status == "analyzed":
            print(f"[OK]   {detail}")
        elif status == "skip_no_data":
            print(f"[SKIP NO-DATA] {detail}")
        elif status == "skip_exists":
            print(f"[SKIP EXISTS]  {detail}")
        else:
            print(f"[ERROR] {detail}")
        rows.append((status, cfg_path.stem, detail))
        print()

    _print_summary(rows)
    # Non-zero exit if any config errored out, so CI can catch it.
    any_error = any(r[0] == "error" for r in rows)
    return 1 if any_error else 0


if __name__ == "__main__":
    sys.exit(main())
