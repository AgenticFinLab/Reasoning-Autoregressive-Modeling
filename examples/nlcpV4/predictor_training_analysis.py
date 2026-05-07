"""Visualize Predictor training losses and learning rate from training logs.

Usage:
    # Single experiment — reads artifacts from the project-local EXPERIMENT/ tree.
    python3 examples/nlcpV4/predictor_training_analysis.py -m predictor -d GSM8K -e Qwen2.5-0.5B_2level_shared

    # All experiments for a module + dataset (baseline + nested variants).
    python3 examples/nlcpV4/predictor_training_analysis.py -m predictor -d GSM8K -e all

    # Read training artifacts from a non-default storage root.
    python3 examples/nlcpV4/predictor_training_analysis.py -m predictor -d GSM8K -e all -s /Data/<proj>

    # Skip already-analyzed configs (by default existing PNGs are overwritten).
    python3 examples/nlcpV4/predictor_training_analysis.py -m predictor -d GSM8K -e all --no-overlap

Arguments:
    -s / --storage-root   Prefix prepended to RELATIVE log paths in the
                          YAML.  MUST match the value used at training time.
                          Default is ``./`` (current working directory).
    -m / --module         Module name: 'builder' or 'predictor'.
    -d / --dataset        Dataset name (directory under configs/nlcpV4/).
    -e / --experiment     Config stem after 'train_{module}_'
                          (e.g. 'Qwen2.5-0.5B_2level_shared') or 'all'.
    -o / --overlap        If true (default), overwrite existing analysis
                          outputs. Use ``--no-overlap`` to skip.

Behavior:
    For each selected config:
    1. If log directory / training_history.json is missing -> skip with
       [SKIP NO-DATA].
    2. If all analysis outputs already exist AND --no-overlap is set ->
       skip with [SKIP EXISTS].
    3. Otherwise run the analysis: overlay eval curves (quick/full), or
       fall back to checkpoint eval when eval_history.json is empty.
    4. Y-limits on every loss subplot are computed from the 99.5th
       percentile so rare spikes do not compress the visible scale.
    A compact status table is printed at the end.

Differences from builder_training_analysis.py:
    - Predictor has 2 loss components (concept, reasoning) instead of 4.
    - Predictor eval_history.json contains ``concept_per_level`` (list of
      per-pyramid-level losses); an extra figure is produced.
    - Loss weights: ``concept_loss_weight``, ``reasoning_loss_weight``.
    - Subplot layout: 2×2 grid (total, concept, reasoning, LR).
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

from ram.utils import apply_storage_root, load_config, print_storage_paths

logger = logging.getLogger(__name__)

# --- Batch-mode constants ------------------------------------------
CONFIGS_ROOT = PROJECT_ROOT / "configs" / "nlcpV4"
VALID_MODULES = {"builder", "predictor"}
ALL_KEYWORD = "all"
ANALYSIS_OUTPUT_DIR_NAME = "train_analysis"

# Configs whose eval_losses_overlay figures should suppress the legend
# (e.g. too many overlapping entries make the legend unreadable).
IGNORE_LEGEND_LIST = [
    # "GSM8K_Qwen2.5-0.5B_4level_independent_AutoWeighted",
    # "GSM8K_Qwen2.5-0.5B_6level_independent_AutoWeighted",
]

# Configs whose plot titles should replace the model-name substring
# (e.g. ``Qwen2.5-0.5B_``) with a different label for presentation.
# Matched against ``log_dir.parent.name`` — same key as IGNORE_LEGEND_LIST.
STRIP_MODEL_IN_TITLE_LIST = [
    "GSM8K_Qwen2.5-0.5B_2level_independent_AutoWeighted",
    "GSM8K_Qwen2.5-0.5B_4level_independent_AutoWeighted",
    "GSM8K_Qwen2.5-0.5B_6level_independent_AutoWeighted",
]

# Matches model-name tokens like ``Qwen2.5-0.5B_`` / ``Qwen3-8B_``
# that appear in experiment names; used to rename them when the
# parent dir is listed in STRIP_MODEL_IN_TITLE_LIST.
_MODEL_NAME_RE = re.compile(r"Qwen[\d.]+-[\d.]+B_")

# Replacement label for titles matched by STRIP_MODEL_IN_TITLE_LIST
# (set to empty string to strip instead of rename).
TITLE_MODEL_REPLACEMENT = "Llama-2-7B-chat_"

ANALYSIS_OUTPUTS = (
    "training_losses.png",
    "training_losses_overlay.png",
    "eval_losses_overlay.png",
    "training_losses_raw.png",
    "training_losses_overlay_raw.png",
    "eval_losses_overlay_raw.png",
    "eval_reasoning_accuracy.png",
    "concept_per_level.png",
)


def parse_args():
    """Parse CLI arguments for the training-analysis script."""
    parser = argparse.ArgumentParser(
        description="Analyze Predictor training logs (batch mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-s",
        "--storage-root",
        type=str,
        default="./",
        help=(
            "Prefix to prepend to every relative output path in "
            "config.log. MUST match the value used when training "
            "produced the artifacts this script reads. Default './'."
        ),
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
            "Config stem after 'train_{module}_' "
            "(e.g. 'Qwen2.5-0.5B_2level_shared'), or 'all'."
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
    parser.add_argument(
        "--cut-step",
        type=int,
        default=None,
        help=(
            "If set, only plot data up to this training step. "
            "Useful for zooming into early training."
        ),
    )
    return parser.parse_args()


def discover_configs(module: str, dataset: str, experiment: str) -> list[Path]:
    """Resolve (-m, -d, -e) into a list of config paths.

    Uses recursive glob so configs nested in subdirectories (e.g.
    ``GSM8K/AutoWeighted/train_predictor_*.yml``) are also discovered.
    ``dataset`` may itself contain a subpath (e.g. ``GSM8K/AutoWeighted``)
    to restrict discovery to that subtree.
    """
    dataset_dir = CONFIGS_ROOT / dataset
    if not dataset_dir.is_dir():
        print(f"[ERROR] Dataset dir not found: {dataset_dir}")
        return []

    prefix = f"train_{module}_"
    if experiment == ALL_KEYWORD:
        # Recurse into subdirectories (e.g. AutoWeighted/)
        return sorted(dataset_dir.rglob(f"{prefix}*.yml"))

    # Try flat first, then recursive fallback
    p = dataset_dir / f"{prefix}{experiment}.yml"
    if p.is_file():
        return [p]
    # Search subdirectories
    matches = sorted(dataset_dir.rglob(f"{prefix}{experiment}.yml"))
    if matches:
        return matches
    print(f"[ERROR] Config file not found: {p}")
    return []


def _derive_experiment_name(config_path: Path) -> str:
    """Self-describing experiment name from config path under CONFIGS_ROOT."""
    try:
        rel_parts = config_path.resolve().relative_to(CONFIGS_ROOT).parent.parts
    except ValueError:
        rel_parts = (config_path.parent.name,)
    return "-".join([*rel_parts, config_path.stem])


def smooth(values, window):
    """Simple moving average smoothing."""
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def _robust_ylim(*arrays, upper_percentile: float = 99.5, pad_frac: float = 0.08):
    """Compute a y-axis limit that ignores outlier spikes."""
    collected = []
    for arr in arrays:
        if arr is None:
            continue
        a = np.asarray(arr, dtype=float).ravel()
        a = a[np.isfinite(a)]
        if a.size:
            collected.append(a)
    if not collected:
        return None
    data = np.concatenate(collected)
    lo = float(data.min())
    hi = float(np.percentile(data, upper_percentile))
    if hi <= lo:
        hi = float(data.max())
        if hi <= lo:
            return None
    span = hi - lo
    return (lo - pad_frac * span, hi + pad_frac * span)


def _apply_robust_ylim(ax, *arrays, **kwargs) -> None:
    """Apply _robust_ylim to ``ax``; no-op if no finite data."""
    lim = _robust_ylim(*arrays, **kwargs)
    if lim is not None:
        ax.set_ylim(*lim)


def load_training_history(log_dir: Path):
    """Load training_history.json -> list of dicts."""
    path = log_dir / "training_history.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_eval_history(log_dir: Path):
    """Load eval_history.json -> list of dicts, or empty list if not found."""
    path = log_dir / "eval_history.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _backfill_reasoning_accuracy(eval_hist: list, log_dir: Path) -> None:
    """Compute reasoning_accuracy from log files and inject into eval records."""
    texts_path = log_dir / "eval_reasoning_texts.jsonl"
    if not texts_path.exists():
        return
    texts_by_key: dict[tuple[int, str], list[str]] = {}
    with open(texts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            text_type = entry.get("type", "teacher_forced")
            if text_type != "teacher_forced":
                continue
            key = (entry["step"], entry.get("eval_type", "full"))
            texts_by_key[key] = entry["texts"]

    samples_path = log_dir / "eval_sample_history.json"
    if not samples_path.exists():
        return
    with open(samples_path, "r", encoding="utf-8") as f:
        sample_history = json.load(f)
    solutions_by_key: dict[tuple[int, str], list[str]] = {}
    for record in sample_history:
        key = (record["step"], record.get("eval_type", "full"))
        solutions_by_key[key] = [s.get("solution") for s in record.get("samples", [])]

    num_filled = 0
    for r in eval_hist:
        key = (r["step"], r.get("eval_type", "full"))
        texts = texts_by_key.get(key)
        solutions = solutions_by_key.get(key)
        if texts and solutions:
            from nlcpV4.eval_builder import compute_reasoning_accuracy

            acc_result = compute_reasoning_accuracy(texts, solutions)
            r["reasoning_accuracy"] = acc_result["accuracy"]
            num_filled += 1

    if num_filled > 0:
        logger.info(
            "Computed reasoning_accuracy for %d eval records from logs",
            num_filled,
        )


def load_terminal_output(log_dir: Path):
    """Load terminal_output.jsonl -> list of dicts (has lr)."""
    path = log_dir / "terminal_output.jsonl"
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_checkpoint_eval(config: dict) -> dict | None:
    """Load best checkpoint and run full evaluation on the test set.

    Returns averaged loss dict (keys: total, concept, reasoning,
    concept_per_level), or None if no checkpoint is found.
    """
    import torch

    from lmbase.utils.env_tools import get_device
    from nlcpV4.concept_builder import ConceptPyramidBuilder
    from nlcpV4.concept_predictor import ConceptPredictor
    from nlcpV4.data_loader import NLCPV4DataLoader
    from nlcpV4.eval_builder import evaluate_predictor

    checkpoint_dir = Path(config["log"]["checkpoint_path"]).expanduser()

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

    # Build predictor from config
    predictor = ConceptPredictor(config)
    predictor.to(device)

    # Also need a builder for evaluation (predictor eval uses builder
    # to produce ground-truth concepts)
    builder = ConceptPyramidBuilder(config)
    builder.to(device)
    builder.eval()

    # Load checkpoint weights
    ckpt = torch.load(ckpt_path, map_location=device)
    predictor.load_state_dict(ckpt["model_state_dict"])
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

    loss_weights = train_cfg["loss_weights"]
    max_length = config["model"]["max_length"]

    eval_losses, _, _ = evaluate_predictor(
        predictor=predictor,
        builder=builder,
        eval_dataloader=eval_dataloader,
        loss_weights=loss_weights,
        max_length=max_length,
        device=device,
        max_batches=0,
    )

    logger.info(
        "Checkpoint eval: total=%.4f concept=%.4f reasoning=%.4f",
        eval_losses["total"],
        eval_losses["concept"],
        eval_losses.get("reasoning", 0.0),
    )
    return eval_losses


# ── Eval overlay helpers ─────────────────────────────────────────


def _plot_eval_on_ax(ax, eval_quick, eval_full, key, weight, ckpt_eval, last_step):
    """Overlay eval data on a training loss subplot."""
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
            label="eval(batch)",
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
    ax, eval_quick, eval_full, ckpt_eval, last_step, total_getter=None
):
    """Overlay eval total loss on the total-loss subplot."""
    if total_getter is None:

        def total_getter(r):
            return r["total"]

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
            label="eval(batch)",
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


# ── Figure builders ──────────────────────────────────────────────


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
    suppress_eval_legend: bool = False,
) -> None:
    """Build 3 PNGs for one loss mode (weighted / raw).

    Predictor layout: 2x2 grid — total, concept, reasoning, LR.
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

    # ── Figure 1: 2x2 grid (total, concept, reasoning, LR) ──────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"{experiment_name}", y=0.98, fontsize=22)

    # Total loss
    ax = axes[0, 0]
    ax.plot(steps, total_array, alpha=0.15, color="black", linewidth=0.5)
    s = smooth(total_array, window)
    ax.plot(
        steps[: len(s)] + window // 2, s, color="black", linewidth=1.5, label="train"
    )
    _plot_eval_total_on_ax(
        ax, eval_quick, eval_full, ckpt_eval, last_step, total_getter=eval_total_getter
    )
    ax.set_title(total_subplot_title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _apply_robust_ylim(ax, total_array)

    grid_layout = [
        ("concept", axes[0, 1], "tab:blue"),
        ("reasoning", axes[1, 0], "tab:red"),
    ]
    for key, ax, color in grid_layout:
        data = comp_arrays[key]
        ax.plot(steps, data, alpha=0.15, color=color, linewidth=0.5)
        s = smooth(data, window)
        ax.plot(
            steps[: len(s)] + window // 2, s, color=color, linewidth=1.5, label="train"
        )
        _plot_eval_on_ax(
            ax, eval_quick, eval_full, key, comp_weights[key], ckpt_eval, last_step
        )
        ax.set_title(comp_title(key))
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _apply_robust_ylim(ax, data)

    # Learning rate
    ax = axes[1, 1]
    ax.plot(lr_steps, lr_values, color="tab:purple", linewidth=1.5)
    ax.set_title("Learning Rate")
    ax.set_xlabel("Step")
    ax.set_ylabel("LR")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    plt.tight_layout()

    # ── Figure 2: overlay ────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    ax2.set_title(experiment_name)

    overlay_items = [
        ("concept", comp_arrays["concept"], "tab:blue"),
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
    _apply_robust_ylim(
        ax2, comp_arrays["concept"], comp_arrays["reasoning"], total_array
    )
    plt.tight_layout()

    # ── Figure 3: eval overlay ───────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    ax3.set_title(experiment_name)

    eval_plot_items = [
        ("concept", "tab:blue"),
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
                label=f"{overlay_label(key)} [batch]",
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
        if not suppress_eval_legend:
            ax3.legend(ncol=1)
    ax3.grid(True, alpha=0.3)
    _eval_ylim_arrays: list = []
    for _records in (eval_quick, eval_full):
        if not _records:
            continue
        for _k, _c in eval_plot_items:
            _eval_ylim_arrays.append(_eval_values(_records, _k))
    if not has_any_eval and ckpt_eval is not None:
        for _k, _c in eval_plot_items:
            if _k == "total":
                _eval_ylim_arrays.append(np.array([eval_total_getter(ckpt_eval)]))
            else:
                _eval_ylim_arrays.append(
                    np.array([ckpt_eval.get(_k, 0.0) * comp_weights[_k]])
                )
    _apply_robust_ylim(ax3, *_eval_ylim_arrays)
    plt.tight_layout()

    fig.savefig(
        output_dir / f"training_losses{suffix}.png", dpi=150, bbox_inches="tight"
    )
    fig.savefig(output_dir / f"training_losses{suffix}.pdf", bbox_inches="tight")
    fig2.savefig(
        output_dir / f"training_losses_overlay{suffix}.png",
        dpi=150,
        bbox_inches="tight",
    )
    fig2.savefig(
        output_dir / f"training_losses_overlay{suffix}.pdf",
        bbox_inches="tight",
    )
    fig3.savefig(
        output_dir / f"eval_losses_overlay{suffix}.png", dpi=150, bbox_inches="tight"
    )
    fig3.savefig(output_dir / f"eval_losses_overlay{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)
    plt.close(fig2)
    plt.close(fig3)


def _build_concept_per_level_figure(
    *,
    output_dir: Path,
    experiment_name: str,
    eval_quick: list,
    eval_full: list,
) -> None:
    """Build a figure showing concept loss per pyramid level from eval history.

    Each eval record has ``concept_per_level`` — a list of per-level MSE
    losses. This figure overlays each level as a separate curve so the
    user can spot levels that converge slowly or diverge.
    """
    # Determine number of levels from the first record that has the field
    all_records = eval_quick + eval_full
    all_records = [
        r for r in all_records if "concept_per_level" in r and r["concept_per_level"]
    ]
    if not all_records:
        return  # nothing to plot

    n_levels = len(all_records[0]["concept_per_level"])
    cmap = plt.get_cmap("viridis")
    level_colors = cmap(np.linspace(0.15, 0.85, n_levels))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title(experiment_name)

    if eval_quick:
        eq_with = [
            r for r in eval_quick if "concept_per_level" in r and r["concept_per_level"]
        ]
        if eq_with:
            eq_steps = np.array([r["step"] for r in eq_with])
            for lvl in range(n_levels):
                vals = np.array([r["concept_per_level"][lvl] for r in eq_with])
                ax.plot(
                    eq_steps,
                    vals,
                    linestyle="--",
                    marker=".",
                    color=level_colors[lvl],
                    linewidth=1.0,
                    markersize=4,
                    alpha=0.8,
                    label=f"L{lvl} [batch]",
                )

    if eval_full:
        ef_with = [
            r for r in eval_full if "concept_per_level" in r and r["concept_per_level"]
        ]
        if ef_with:
            ef_steps = np.array([r["step"] for r in ef_with])
            for lvl in range(n_levels):
                vals = np.array([r["concept_per_level"][lvl] for r in ef_with])
                ax.plot(
                    ef_steps,
                    vals,
                    linestyle="-",
                    marker="s",
                    color=level_colors[lvl],
                    linewidth=1.2,
                    markersize=5,
                    alpha=0.9,
                    label=f"L{lvl} [full]",
                )

    ax.set_xlabel("Step")
    ax.set_ylabel("Concept Loss (MSE)")
    ax.legend(ncol=1)
    ax.grid(True, alpha=0.3)

    # Robust ylim across all level curves
    ylim_arrays = []
    for r in all_records:
        ylim_arrays.append(np.array(r["concept_per_level"]))
    _apply_robust_ylim(ax, *ylim_arrays)

    plt.tight_layout()
    fig.savefig(output_dir / "concept_per_level.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "concept_per_level.pdf", bbox_inches="tight")
    plt.close(fig)


def _build_accuracy_figure(
    *,
    output_dir: Path,
    experiment_name: str,
    eval_quick: list,
    eval_full: list,
    ckpt_eval: dict | None,
    last_step: int,
) -> None:
    """Build the reasoning accuracy PNG."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title(experiment_name)

    has_data = False

    acc_quick = [
        (r["step"], r["reasoning_accuracy"])
        for r in eval_quick
        if "reasoning_accuracy" in r
    ]
    if acc_quick:
        steps_q, vals_q = zip(*acc_quick)
        ax.plot(
            np.array(steps_q),
            np.array(vals_q),
            linestyle="--",
            marker=".",
            color="tab:cyan",
            linewidth=1.2,
            markersize=5,
            alpha=0.9,
            label="eval(batch)",
        )
        has_data = True

    acc_full = [
        (r["step"], r["reasoning_accuracy"])
        for r in eval_full
        if "reasoning_accuracy" in r
    ]
    if acc_full:
        steps_f, vals_f = zip(*acc_full)
        ax.plot(
            np.array(steps_f),
            np.array(vals_f),
            linestyle="-",
            marker="s",
            color="tab:red",
            linewidth=1.5,
            markersize=6,
            alpha=0.9,
            label="eval(full)",
        )
        has_data = True

    if not has_data and ckpt_eval is not None and "reasoning_accuracy" in ckpt_eval:
        val = ckpt_eval["reasoning_accuracy"]
        ax.axhline(y=val, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.plot(
            last_step,
            val,
            "*",
            color="tab:red",
            markersize=14,
            zorder=5,
            label=f"best ckpt eval ({val:.1%})",
        )
        has_data = True

    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(True, alpha=0.3)
    if has_data:
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "No reasoning_accuracy data in eval_history\n"
            "(run eval with updated eval_builder to populate)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
        )

    plt.tight_layout()
    fig.savefig(
        output_dir / "eval_reasoning_accuracy.png", dpi=150, bbox_inches="tight"
    )
    fig.savefig(output_dir / "eval_reasoning_accuracy.pdf", bbox_inches="tight")
    plt.close(fig)


# ── Main analysis runner ─────────────────────────────────────────


def _run_predictor_analysis(
    config_path: Path, storage_root: str, cut_step: int | None = None
) -> None:
    """Run the full analysis for a single predictor config and write PNGs."""
    config = load_config(str(config_path))
    apply_storage_root(config, storage_root)

    plt.rcParams.update(
        {
            "font.weight": "bold",
            "axes.titlesize": 28,
            "axes.titleweight": "bold",
            "axes.labelsize": 22,
            "axes.labelweight": "bold",
            "figure.titlesize": 30,
            "figure.titleweight": "bold",
            "legend.fontsize": 28,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    log_dir = Path(config["log"]["log_path"]).expanduser()

    loss_weights = config["training"]["loss_weights"]
    w_concept = loss_weights["concept_loss_weight"]
    w_reasoning = loss_weights["reasoning_loss_weight"]

    # Title: strip "train_" prefix, keep "predictor_..." onward
    _stem = config_path.stem
    if _stem.startswith("train_"):
        _stem = _stem[len("train_") :]
    experiment_name = _stem

    # Optionally rename the model-name token (e.g. ``Qwen2.5-0.5B_`` ->
    # ``Llama-2-7B-chat_``) in the title for configs listed in
    # STRIP_MODEL_IN_TITLE_LIST.
    if log_dir.parent.name in STRIP_MODEL_IN_TITLE_LIST:
        experiment_name = _MODEL_NAME_RE.sub(TITLE_MODEL_REPLACEMENT, experiment_name)

    # ── Load data ─────────────────────────────────────────────────
    history = load_training_history(log_dir)
    terminal = load_terminal_output(log_dir)
    eval_hist = load_eval_history(log_dir)

    # Apply --cut-step filter
    if cut_step is not None:
        history = [r for r in history if r["step"] <= cut_step]
        terminal = [r for r in terminal if r.get("step", 0) <= cut_step]
        eval_hist = [r for r in eval_hist if r["step"] <= cut_step]

    if not history:
        print(f"[SKIP] No training data within cut_step={cut_step}")
        return

    _backfill_reasoning_accuracy(eval_hist, log_dir)

    eval_quick = [r for r in eval_hist if r.get("eval_type") == "quick"]
    eval_full = [r for r in eval_hist if r.get("eval_type") == "full"]

    ckpt_eval = None
    if not eval_hist:
        print("No eval_history.json found — running checkpoint evaluation...")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        ckpt_eval = run_checkpoint_eval(config)

    window = max(10, min(500, len(history) // 100))

    steps = np.array([r["step"] for r in history])
    last_step = int(steps[-1]) if len(steps) > 0 else 0

    # Weighted losses
    concept_w = np.array([r["concept"] * w_concept for r in history])
    reasoning_w = np.array([r.get("reasoning", 0.0) * w_reasoning for r in history])
    total = np.array([r["total"] for r in history])

    lr_steps = np.array([r["step"] for r in terminal if "lr" in r])
    lr_values = np.array([r["lr"] for r in terminal if "lr" in r])

    # ── Output directory ──────────────────────────────────────────
    output_dir = log_dir.parent / ANALYSIS_OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if this config should suppress eval overlay legend
    suppress_eval_legend = log_dir.parent.name in IGNORE_LEGEND_LIST

    # ── Raw component arrays ──────────────────────────────────────
    concept_raw = np.array([r["concept"] for r in history])
    reasoning_raw = np.array([r.get("reasoning", 0.0) for r in history])
    total_raw = concept_raw + reasoning_raw

    _raw_comps = ("concept", "reasoning")

    # ── Weighted pass ─────────────────────────────────────────
    _build_figures(
        mode="weighted",
        output_dir=output_dir,
        experiment_name=experiment_name,
        steps=steps,
        last_step=last_step,
        window=window,
        comp_arrays={"concept": concept_w, "reasoning": reasoning_w},
        total_array=total,
        comp_weights={"concept": w_concept, "reasoning": w_reasoning},
        eval_total_getter=lambda r: r["total"],
        eval_quick=eval_quick,
        eval_full=eval_full,
        ckpt_eval=ckpt_eval,
        lr_steps=lr_steps,
        lr_values=lr_values,
        suppress_eval_legend=suppress_eval_legend,
    )

    # ── Raw pass ──────────────────────────────────────────────
    _build_figures(
        mode="raw",
        output_dir=output_dir,
        experiment_name=experiment_name,
        steps=steps,
        last_step=last_step,
        window=window,
        comp_arrays={"concept": concept_raw, "reasoning": reasoning_raw},
        total_array=total_raw,
        comp_weights={k: 1.0 for k in _raw_comps},
        eval_total_getter=lambda r: sum(r.get(k, 0.0) for k in _raw_comps),
        eval_quick=eval_quick,
        eval_full=eval_full,
        ckpt_eval=ckpt_eval,
        lr_steps=lr_steps,
        lr_values=lr_values,
        suppress_eval_legend=suppress_eval_legend,
    )

    # ── Concept per-level figure ──────────────────────────────
    _build_concept_per_level_figure(
        output_dir=output_dir,
        experiment_name=experiment_name,
        eval_quick=eval_quick,
        eval_full=eval_full,
    )

    # ── Reasoning accuracy figure ─────────────────────────────
    _build_accuracy_figure(
        output_dir=output_dir,
        experiment_name=experiment_name,
        eval_quick=eval_quick,
        eval_full=eval_full,
        ckpt_eval=ckpt_eval,
        last_step=last_step,
    )

    print("Saved to %s" % output_dir)


def analyze_one(
    config_path: Path,
    *,
    overlap: bool,
    storage_root: str,
    cut_step: int | None = None,
) -> tuple[str, str]:
    """Analyze a single config. Returns (status, detail) tuple."""
    try:
        config = load_config(str(config_path))
    except Exception as exc:  # noqa: BLE001
        return "error", f"load_config failed: {exc}"

    apply_storage_root(config, storage_root)
    print_storage_paths(config, storage_root)

    log_dir = Path(config["log"]["log_path"]).expanduser()

    training_history = log_dir / "training_history.json"
    if not training_history.is_file():
        return "skip_no_data", f"training_history.json missing at {log_dir}"

    output_dir = log_dir.parent / ANALYSIS_OUTPUT_DIR_NAME
    if not overlap and all((output_dir / name).is_file() for name in ANALYSIS_OUTPUTS):
        return "skip_exists", f"all outputs already exist at {output_dir}"
    try:
        _run_predictor_analysis(
            config_path, storage_root=storage_root, cut_step=cut_step
        )
    except Exception as exc:  # noqa: BLE001
        plt.close("all")
        return "error", f"{type(exc).__name__}: {exc}"

    return "analyzed", f"wrote {len(ANALYSIS_OUTPUTS)} PNGs to {output_dir}"


def _print_summary(rows: list[tuple[str, str, str]]) -> None:
    """Print a compact status table."""
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
    """CLI entry point."""
    args = parse_args()
    module: str = args.module
    dataset: str = args.dataset
    experiment: str = args.experiment
    overlap: bool = args.overlap
    storage_root: str = args.storage_root

    cut_step: int | None = args.cut_step

    configs = discover_configs(module, dataset, experiment)
    if not configs:
        return 1

    print(
        f"[ANALYZE] module={module} dataset={dataset} "
        f"experiment={experiment} overlap={overlap} cut_step={cut_step}  "
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
        status, detail = analyze_one(
            cfg_path, overlap=overlap, storage_root=storage_root, cut_step=cut_step
        )
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
    any_error = any(r[0] == "error" for r in rows)
    return 1 if any_error else 0


if __name__ == "__main__":
    sys.exit(main())
