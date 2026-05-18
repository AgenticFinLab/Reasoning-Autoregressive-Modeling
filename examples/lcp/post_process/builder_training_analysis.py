"""Visualize Builder training losses and learning rate from training logs.

Usage:
    # Single experiment — reads artifacts from the project-local EXPERIMENT/ tree.
    python3 examples/lcp/builder_training_analysis.py -m builder -d GSM8K -e Qwen2.5-0.5B_6level

    # All experiments for a module + dataset (baseline + nested variants).
    python3 examples/lcp/builder_training_analysis.py -m builder -d GSM8K -e all

    # Variant subtree (e.g. AutoWeighted/). Note the path has to be
    # given to -d exactly as it appears under configs/lcp/.
    python3 examples/lcp/builder_training_analysis.py -m builder -d GSM8K/AutoWeighted -e all

    # Read training artifacts from a non-default storage root.
    # MUST match the -s value that train_builder.py / run_experiments.py
    # was launched with — otherwise this script looks at the wrong
    # checkpoints / log_path directories and reports [SKIP NO-DATA].
    python3 examples/lcp/builder_training_analysis.py -m builder -d GSM8K -e all -s /Data/<proj>

    # Skip already-analyzed configs (by default existing PNGs are overwritten).
    python3 examples/lcp/builder_training_analysis.py -m builder -d GSM8K -e all --no-overlap

Arguments:
    -s / --storage-root   Prefix prepended to RELATIVE log paths in the
                          YAML (save_folder/checkpoint_path/log_path).
                          Listed FIRST because it controls every output
                          path this script reads. MUST match the value
                          used at training time so this script reads the
                          artifacts actually on disk. Default is ``./``
                          (current working directory) — never an implicit
                          project root. The resolved paths are printed
                          as a ``[STORAGE]`` block per config at startup.
    -m / --module         Module name: 'builder' or 'predictor'.
    -d / --dataset        Dataset name (directory under configs/lcp/).
                          May be a nested path like 'GSM8K/AutoWeighted'.
    -e / --experiment     Config stem after 'train_{module}_'
                          (e.g. 'Qwen2.5-0.5B_6level') or 'all' to process
                          every matching config under the dataset.
    -o / --overlap        If true (default), overwrite existing analysis
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
    4. Y-limits on every loss subplot / overlay are computed from the
       99.5th percentile of the plotted data so a rare spike step
       (e.g. a single loss value in the hundreds while the normal
       regime is in single digits) does not compress the visible
       scale. The spike point is still drawn — it simply clips
       outside the visible axis box.
    A compact status table is printed at the end.
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
from lcp.concept_builder import ConceptPyramidBuilder
from lcp.data_loader import LCPDataLoader
from lcp.eval_builder import (
    MODE_TEACHER_FORCED,
    compute_reasoning_accuracy,
    evaluate_builder,
)
from ram.utils import apply_storage_root, load_config, print_storage_paths

logger = logging.getLogger(__name__)

# --- Batch-mode constants ------------------------------------------
CONFIGS_ROOT = PROJECT_ROOT / "configs" / "lcp"
VALID_MODULES = {"builder", "predictor"}
ALL_KEYWORD = "all"
# The six PNGs produced by a successful analysis run (weighted + raw).
# Presence of ALL six is treated as "already analyzed". They live in
# <experiment>/train_analysis/, not in the logs/ directory.
ANALYSIS_OUTPUT_DIR_NAME = "train_analysis"

# Configs whose eval_losses_overlay figures should suppress the legend
# (e.g. too many overlapping entries make the legend unreadable).
IGNORE_LEGEND_LIST: list[str] = [
    # "GSM8K_Qwen2.5-0.5B_6level_AutoWeighted",
]

ANALYSIS_OUTPUTS = (
    "training_losses.png",
    "training_losses_overlay.png",
    "eval_losses_overlay.png",
    "training_losses_raw.png",
    "training_losses_overlay_raw.png",
    "eval_losses_overlay_raw.png",
    "eval_reasoning_accuracy.png",
)


def parse_args():
    """Parse CLI arguments for the training-analysis script."""
    parser = argparse.ArgumentParser(
        description="Analyze Builder training logs (batch mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-s",
        "--storage-root",
        type=str,
        default="./",
        help=(
            "Prefix to prepend to every relative output path in "
            "config.log (save_folder / checkpoint_path / log_path). "
            "MUST match the value used when training produced the "
            "artifacts this script reads, otherwise this tool will "
            "look at the wrong log directory and report "
            "``[SKIP NO-DATA]``. Absolute paths in YAML are preserved. "
            "Default is './' (current working directory) — no silent "
            "project-root fallback. The resolved paths are printed "
            "per-config as a ``[STORAGE]`` block so you can verify."
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
        help="Dataset name (directory under configs/lcp/).",
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

    Config file path convention:
        configs/lcp/{dataset}/train_{module}_{experiment}.yml

    ``dataset`` may include nested subdirectories (e.g. ``GSM8K/AutoWeighted``)
    to discover configs under a variant folder — ``Path`` division handles
    the extra separator transparently.
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
    """Self-describing experiment name from the config's path under ``CONFIGS_ROOT``.

    Joins all path segments between ``configs/lcp/`` and the YAML
    file with ``-`` plus the filename stem, so nested variants keep
    the dataset name visible in SwanLab and plot titles:

        configs/lcp/GSM8K/train_builder_Qwen2.5-0.5B_6level.yml
          -> "GSM8K-train_builder_Qwen2.5-0.5B_6level"
        configs/lcp/GSM8K/AutoWeighted/train_builder_Qwen2.5-0.5B_6level.yml
          -> "GSM8K-AutoWeighted-train_builder_Qwen2.5-0.5B_6level"

    Falls back to the legacy single-parent form when the file lives
    outside ``CONFIGS_ROOT`` (keeps back-compat for out-of-tree configs).
    """
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
    """Compute a y-axis limit that ignores outlier spikes.

    A single anomalous training step can push a loss value into the
    hundreds while the normal regime lives in single digits; letting
    matplotlib auto-scale around that spike compresses the rest of
    the curve into a flat line at the bottom of the axis. Here we
    use the ``upper_percentile``-th percentile of the pooled data as
    the upper bound, so outlier points above that percentile still
    get drawn but clip outside the visible axis box (matplotlib's
    default behaviour).

    Args:
        *arrays: Any number of array-likes (train loss curve, eval
            curves, etc.). ``None`` entries are skipped; non-finite
            values are filtered.
        upper_percentile: Percentile used as the upper y-bound.
            ``99.5`` keeps 99.5% of points in view and clips only
            the top 0.5% — plenty of headroom for ordinary training
            noise, aggressive enough to neutralise single-step
            spikes.
        pad_frac: Extra vertical padding as a fraction of the span,
            applied at both ends.

    Returns:
        ``(ymin, ymax)`` tuple, or ``None`` when no finite data is
        available (caller should then skip ``set_ylim``).
    """
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
    """Apply :func:`_robust_ylim` to ``ax``; no-op if no finite data."""
    lim = _robust_ylim(*arrays, **kwargs)
    if lim is not None:
        ax.set_ylim(*lim)


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


def _backfill_reasoning_accuracy(eval_hist: list, log_dir: Path) -> None:
    """Compute reasoning_accuracy from log files and inject into eval records.

    Loads ``eval_reasoning_texts.jsonl`` (predicted texts per step/eval_type)
    and ``eval_sample_history.json`` (ground-truth solutions per step/eval_type),
    matches them by (step, eval_type), runs ``compute_reasoning_accuracy``,
    and injects the result into the corresponding eval_history records in-place.

    Handles both old format (no "type" field) and new format with
    "type": "teacher_forced" / "generation".  Only teacher-forced texts
    are used for accuracy computation.
    """
    # Load reasoning texts (predictions)
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
            # New format has "type" field; only use teacher_forced for accuracy.
            # Old format has no "type" field — treat as teacher_forced.
            text_type = entry.get("type", "teacher_forced")
            if text_type != "teacher_forced":
                continue
            key = (entry["step"], entry.get("eval_type", "full"))
            texts_by_key[key] = entry["texts"]

    # Load sample history (ground-truth solutions)
    samples_path = log_dir / "eval_sample_history.json"
    if not samples_path.exists():
        return
    with open(samples_path, "r", encoding="utf-8") as f:
        sample_history = json.load(f)
    solutions_by_key: dict[tuple[int, str], list[str]] = {}
    for record in sample_history:
        key = (record["step"], record.get("eval_type", "full"))
        solutions_by_key[key] = [s.get("solution") for s in record.get("samples", [])]

    # Compute accuracy for each eval record
    num_filled = 0
    for r in eval_hist:
        key = (r["step"], r.get("eval_type", "full"))
        texts = texts_by_key.get(key)
        solutions = solutions_by_key.get(key)
        if texts and solutions:
            acc_result = compute_reasoning_accuracy(texts, solutions)
            r["reasoning_accuracy"] = acc_result["accuracy"]
            num_filled += 1

    if num_filled > 0:
        logger.info(
            "Computed reasoning_accuracy for %d eval records from logs",
            num_filled,
        )


def run_checkpoint_eval(config: dict) -> dict | None:
    """Load best checkpoint and run full evaluation on the test set.

    Returns averaged loss dict (keys: total, recon, ordering, residual,
    optionally reasoning), or None if no checkpoint is found.

    Paths are resolved strictly from ``config['log']['checkpoint_path']``
    as-is — absolute values are used verbatim; relative values are
    resolved against the CURRENT WORKING DIRECTORY (no silent
    project-root fallback). The user controls this via ``-s``.
    """
    # Locate best checkpoint.
    checkpoint_dir = Path(config["log"]["checkpoint_path"]).expanduser()
    # NOTE: relative paths intentionally resolve against CWD — every
    # caller now runs after ``apply_storage_root`` with a mandatory
    # ``-s`` (default ``./``) so the path shown in ``[STORAGE]`` is
    # exactly what's opened here.

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

    eval_dataloader = LCPDataLoader(
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

    # Build generation_kwargs from YAML even though teacher-forced mode
    # below will not consume them — keeps the contract that all HF
    # ``.generate()`` knobs come from YAML, never from in-code defaults.
    generation_kwargs = {
        "max_new_tokens": eval_cfg["generation_max_tokens"],
        "do_sample": eval_cfg["do_sample"],
        "temperature": eval_cfg["temperature"],
        "top_k": eval_cfg["top_k"],
        "top_p": eval_cfg["top_p"],
    }

    # Discard reasoning_texts and samples list — not needed for plotting.
    eval_losses, _, _ = evaluate_builder(
        builder=builder,
        eval_dataloader=eval_dataloader,
        loss_weights=loss_weights,
        ordering_loss_type=ordering_loss_type,
        # Evaluate all batches
        max_batches=0,
        mode=MODE_TEACHER_FORCED,
        generation_kwargs=generation_kwargs,
        output_root=None,
        dump_artifacts=False,
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
    suppress_eval_legend: bool = False,
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
    fig.suptitle(f"{experiment_name}", y=0.98, fontsize=22)

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
    # Keep the axis scale anchored to the normal regime: a rare spike
    # step will draw outside the axis box rather than compressing the
    # rest of the curve. See ``_robust_ylim`` for details.
    _apply_robust_ylim(ax, total_array)

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
        _apply_robust_ylim(ax, data)

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
    ax2.set_title(experiment_name)

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
    # Robust ylim across all five curves (4 components + total).
    _apply_robust_ylim(
        ax2,
        comp_arrays["recon"],
        comp_arrays["ordering"],
        comp_arrays["residual"],
        comp_arrays["reasoning"],
        total_array,
    )
    plt.tight_layout()

    # ── Figure 3: eval overlay ────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    ax3.set_title(experiment_name)

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
        if not suppress_eval_legend:
            ax3.legend(ncol=2)
    ax3.grid(True, alpha=0.3)
    # Robust ylim based on the exact same values drawn on ax3
    # (quick / full eval per-component and per-total, plus the ckpt
    # fallback markers when eval history is empty). Outlier eval
    # values will clip outside the axis rather than squash the rest.
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


def _build_accuracy_figure(
    *,
    output_dir: Path,
    experiment_name: str,
    eval_quick: list,
    eval_full: list,
    ckpt_eval: dict | None,
    last_step: int,
) -> None:
    """Build the reasoning accuracy PNG and save to ``output_dir``.

    Plots accuracy (exact-match on final answer) from eval_history
    records that contain a ``reasoning_accuracy`` field. If no eval
    records have this field (e.g. older runs), the figure is still
    created but will be empty with a note.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title(experiment_name)

    has_data = False

    # Quick eval accuracy
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
            label="eval(quick)",
        )
        has_data = True

    # Full eval accuracy
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

    # Checkpoint eval fallback
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


def _run_builder_analysis(
    config_path: Path, storage_root: str, cut_step: int | None = None
) -> None:
    """Run the full analysis for a single config and write 6 PNGs to
    ``<experiment>/train_analysis/`` (sibling of ``logs/``).

    For each pass (weighted and raw) three figures are produced:
    training_losses, training_losses_overlay, eval_losses_overlay.

    Precondition: caller has already verified that training_history.json
    exists under config.log.log_path.
    """
    config = load_config(str(config_path))
    apply_storage_root(config, storage_root)

    # --- Plot styling: larger, bold titles/labels/legend/ticks ----
    # Set once via rcParams so every set_title / set_xlabel /
    # set_ylabel / legend call inherits the same typography. Safe to
    # re-apply on each batch iteration (idempotent).
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
            # Strip top and right border on every Axes (applies to
            # fig's subplots as well as ax2 / ax3 created later).
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    log_dir = Path(config["log"]["log_path"]).expanduser()
    # Relative paths resolve against CWD — never PROJECT_ROOT. The
    # ``-s`` default of ``./`` plus the ``[STORAGE]`` print above make
    # the exact location visible.

    loss_weights = config["training"]["loss_weights"]
    w_recon = loss_weights["recon_loss_weight"]
    w_ordering = loss_weights["ordering_loss_weight"]
    w_residual = loss_weights["residual_loss_weight"]
    w_reasoning = loss_weights["reasoning_loss_weight"]

    experiment_name = _derive_experiment_name(config_path)

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

    # Backfill reasoning_accuracy from reasoning texts + sample history
    # for older runs that didn't compute it during training.
    _backfill_reasoning_accuracy(eval_hist, log_dir)

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
        ckpt_eval = run_checkpoint_eval(config)

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

    # Check if this config should suppress eval overlay legend
    suppress_eval_legend = log_dir.parent.name in IGNORE_LEGEND_LIST

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
        suppress_eval_legend=suppress_eval_legend,
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
        suppress_eval_legend=suppress_eval_legend,
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

    apply_storage_root(config, storage_root)
    # Surface the resolved log paths per config so downstream failures
    # (``[SKIP NO-DATA]`` / ``[ERROR]``) are easy to diagnose.
    print_storage_paths(config, storage_root)

    log_dir = Path(config["log"]["log_path"]).expanduser()
    # Relative paths resolve against CWD (matching run_checkpoint_eval
    # above). No silent project-root fallback.

    training_history = log_dir / "training_history.json"
    if not training_history.is_file():
        return "skip_no_data", f"training_history.json missing at {log_dir}"

    output_dir = log_dir.parent / ANALYSIS_OUTPUT_DIR_NAME
    if not overlap and all((output_dir / name).is_file() for name in ANALYSIS_OUTPUTS):
        return "skip_exists", f"all outputs already exist at {output_dir}"
    try:
        _run_builder_analysis(config_path, storage_root=storage_root, cut_step=cut_step)
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
    """CLI entry point: iterate matching configs and emit analysis plots."""
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
    # Non-zero exit if any config errored out, so CI can catch it.
    any_error = any(r[0] == "error" for r in rows)
    return 1 if any_error else 0


if __name__ == "__main__":
    sys.exit(main())
