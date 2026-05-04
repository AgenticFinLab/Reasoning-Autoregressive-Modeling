"""Analyze Loss_prepare.json and recommend per-config loss weights.

Reads ``EXPERIMENT/nlcpV4/builder/Loss_prepare.json`` (produced by
``loss_prepare.py``) and generates comparative plots + a recommended
loss-weight table, to help choose ``training.loss_weights`` per config.

Key-format contract (one entry per config)::

    "<dataset>/train_<module>_<model>_<level>level": {
        "batch_size": int,
        "num_batches": int,
        "weights":  {"recon","ordering","residual","reasoning"},
        "per_batch": [ { "raw": {...}, "weighted": {...},
                         "total_weighted": float } ],
        "stats": {
            "raw":            {<comp>: {mean,std,min,max}},
            "weighted":       {<comp>: {mean,std,min,max}},
            "total_weighted": {mean,std,min,max},
        },
    }

Generated outputs (under ``EXPERIMENT/nlcpV4/builder/training_prepare/``)::

    loss_by_level_per_model_raw.png        # same model × different levels
    loss_by_level_per_model_weighted.png
    loss_by_model_per_level_raw.png        # same level × different models
    loss_by_model_per_level_weighted.png
    heatmap_raw.png                        # 4 subplots, (model × level)
    heatmap_weighted.png
    recommended_weights_heatmap.png        # recommended w per comp & config
    recommended_weights.csv
    weights_summary.txt                    # human-readable recommendation

Weight-recommendation rule (default)::

    target_value  = --target  (default 0.8, ≈ median residual)
    rec_w[comp]   = target_value / raw_mean[comp]

so each weighted component contributes ≈ ``target_value`` to the total.
Residual (the most stable and smallest raw value) gets a recommended
weight ≈ 1.0, matching its current setting as the reference scale.

Usage::

    # Use default paths (require --dataset to resolve the per-dataset filename):
    #   -i: EXPERIMENT/nlcpV4/builder/<dataset>_Loss_prepare.json
    #   -o: EXPERIMENT/nlcpV4/builder/training_prepare/
    python3 examples/nlcpV4/builder_training_prepare.py --dataset GSM8K

    # Tune the recommendation target (median weighted contribution per comp).
    python3 examples/nlcpV4/builder_training_prepare.py --dataset GSM8K --target 1.0

    # Override -i / -o explicitly when the file lives outside the default tree.
    python3 examples/nlcpV4/builder_training_prepare.py \\
        -i path/to/<dataset>_Loss_prepare.json -o path/to/output_dir

    # Pull the same default paths but rebased under a storage root.
    # MUST match the ``-s`` that ``loss_prepare.py`` was launched with
    # so this tool reads the exact JSON just written.
    # Resolved paths:
    #   -i: /Data/<proj>/EXPERIMENT/nlcpV4/builder/GSM8K_Loss_prepare.json
    #   -o: /Data/<proj>/EXPERIMENT/nlcpV4/builder/training_prepare/
    python3 examples/nlcpV4/builder_training_prepare.py -s /Data/<proj> --dataset GSM8K

    # Filter to a single module subset (still reads the dataset-specific
    # JSON written by loss_prepare.py).
    python3 examples/nlcpV4/builder_training_prepare.py \\
        --dataset GSM8K --module builder -s /Data/<proj>

Arguments:
    -s / --storage-root   Prefix used to compute the DEFAULT ``-i`` /
                          ``-o`` paths. Listed FIRST because it controls
                          every default input/output path this script
                          reads and writes. Ignored for explicit ``-i`` /
                          ``-o`` values. Default is ``./`` (current
                          working directory) — NEVER an implicit
                          project root. The resolved paths are printed
                          as a ``[STORAGE]`` block at startup. MUST
                          match the ``-s`` used when running
                          ``loss_prepare.py`` so this tool reads the
                          JSON just written by it.
    -i / --input          Path to <dataset>_Loss_prepare.json. When
                          omitted the default is computed from ``-s``
                          and ``--dataset``:
                            <storage_root>/EXPERIMENT/nlcpV4/builder/
                            <dataset>_Loss_prepare.json
    -o / --output-dir     Output directory for plots + CSV. Default:
                            <storage_root>/EXPERIMENT/nlcpV4/builder/
                            training_prepare/
    --target              Target weighted-contribution per component for
                          the recommender (default: 0.8, ≈ median
                          residual raw so residual weight ≈ 1.0).
    --dataset             Optional in-memory key filter — only analyze
                          entries whose dataset matches this name.
    --module              Optional in-memory key filter — only analyze
                          entries whose module matches this name.
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Components in fixed display order (must match training.loss_weights keys).
COMPONENTS = ("recon", "ordering", "residual", "reasoning")
COMPONENT_COLORS = {
    "recon": "tab:blue",
    "ordering": "tab:orange",
    "residual": "tab:green",
    "reasoning": "tab:red",
}

# Key layout: "<dataset>/train_<module>_<model>_<level>level".
# Model is greedy up to the final "_<digits>level" suffix.
KEY_PATTERN = re.compile(
    r"^(?P<dataset>[^/]+)/train_(?P<module>[A-Za-z0-9]+)_"
    r"(?P<model>.+)_(?P<level>\d+)level$"
)


# -----------------------------------------------------------------------------
# Typography — inherit the same bold + enlarged look as builder_training_analysis.
# -----------------------------------------------------------------------------
def _apply_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.weight": "bold",
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "axes.labelweight": "bold",
            "figure.titlesize": 17,
            "figure.titleweight": "bold",
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def parse_key(key: str) -> dict | None:
    """Split a ``<dataset>/train_<module>_<model>_<level>level`` key into parts.

    Returns a dict with ``dataset``, ``module``, ``model``, ``level`` (int),
    or ``None`` if the key does not match the expected pattern.
    """
    m = KEY_PATTERN.match(key)
    if m is None:
        return None
    return {
        "dataset": m.group("dataset"),
        "module": m.group("module"),
        "model": m.group("model"),
        "level": int(m.group("level")),
    }


def load_entries(path: Path) -> list[dict]:
    """Flatten the JSON into a list of per-config records."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries: list[dict] = []
    for key, v in data.items():
        meta = parse_key(key)
        if meta is None:
            print(f"[WARN] Unparseable key skipped: {key}")
            continue
        raw = {c: v["stats"]["raw"][c]["mean"] for c in COMPONENTS}
        weighted = {c: v["stats"]["weighted"][c]["mean"] for c in COMPONENTS}
        entries.append(
            {
                "key": key,
                **meta,
                "batch_size": v["batch_size"],
                "num_batches": v["num_batches"],
                "weights": v["weights"],
                "raw": raw,
                "weighted": weighted,
                "total_weighted": v["stats"]["total_weighted"]["mean"],
            }
        )
    return entries


def _sorted_unique(values, key=lambda x: x):
    return sorted(set(values), key=key)


def _model_sort_key(model: str):
    """Sort models by family then size: Qwen2.5-0.5B < Qwen2.5-1.5B < Qwen3-0.6B ..."""
    fam_match = re.match(r"^(Qwen[\d.]*)-([\d.]+)B", model)
    if fam_match:
        fam = fam_match.group(1)
        size = float(fam_match.group(2))
        return (fam, size, model)
    return (model, 0.0, model)


# -----------------------------------------------------------------------------
# Matrix helpers (model × level)
# -----------------------------------------------------------------------------
def build_matrix(
    entries: list[dict], models: list[str], levels: list[int], field: str, comp: str
) -> np.ndarray:
    """Return a (len(models), len(levels)) matrix for field[comp].

    Missing (model, level) cells are filled with NaN.
    ``field`` is "raw" or "weighted" (both are dict-by-component),
    or "recommended" (caller precomputes and stores in entry["recommended"]).
    """
    lookup = {(e["model"], e["level"]): e for e in entries}
    mat = np.full((len(models), len(levels)), np.nan, dtype=float)
    for i, mdl in enumerate(models):
        for j, lvl in enumerate(levels):
            e = lookup.get((mdl, lvl))
            if e is None:
                continue
            mat[i, j] = e[field][comp]
    return mat


# -----------------------------------------------------------------------------
# Recommendation
# -----------------------------------------------------------------------------
def recommend_weights(entries: list[dict], target: float) -> None:
    """Attach ``recommended`` + ``rec_total_weighted`` to each entry (in place)."""
    for e in entries:
        rec = {}
        for c in COMPONENTS:
            raw = e["raw"][c]
            rec[c] = target / raw if raw > 0 else float("nan")
        e["recommended"] = rec
        e["rec_total_weighted"] = sum(rec[c] * e["raw"][c] for c in COMPONENTS)


def summarize_recommendations(entries: list[dict]) -> dict:
    """Global aggregate: per-component median / mean / range of recommended weights."""
    agg: dict = {}
    for c in COMPONENTS:
        vals = np.array([e["recommended"][c] for e in entries], dtype=float)
        vals = vals[np.isfinite(vals)]
        agg[c] = {
            "median": float(np.median(vals)),
            "mean": float(np.mean(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "std": float(np.std(vals)),
        }
    return agg


# -----------------------------------------------------------------------------
# Plot: per-model curves (x=level, y=loss, one subplot per model)
# -----------------------------------------------------------------------------
def plot_per_model(
    entries: list[dict],
    models: list[str],
    levels: list[int],
    output_path: Path,
    *,
    field: str,
    title_suffix: str,
    log_y: bool,
) -> None:
    """One subplot per model: curves of 4 components vs level."""
    n = len(models)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7.5 * ncols, 5.0 * nrows), squeeze=False
    )
    fig.suptitle(f"Loss vs Level, per model — {title_suffix}", y=0.995)

    for idx, mdl in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        sub = [e for e in entries if e["model"] == mdl]
        sub.sort(key=lambda e: e["level"])
        xs = [e["level"] for e in sub]
        bs = sub[0]["batch_size"] if sub else "?"
        for c in COMPONENTS:
            ys = [e[field][c] for e in sub]
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2.0,
                color=COMPONENT_COLORS[c],
                label=c,
            )
        ax.set_title(f"{mdl}  (bs={bs})")
        ax.set_xlabel("Level")
        ax.set_ylabel("Loss (log)" if log_y else "Loss")
        ax.set_xticks(levels)
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(ncol=2)

    # hide unused axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Plot: per-level grouped bars (x=model, bars per component)
# -----------------------------------------------------------------------------
def plot_per_level(
    entries: list[dict],
    models: list[str],
    levels: list[int],
    output_path: Path,
    *,
    field: str,
    title_suffix: str,
    log_y: bool,
) -> None:
    """One subplot per level: grouped bars (model × component)."""
    n = len(levels)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7.0 * ncols, 4.8 * nrows), squeeze=False
    )
    fig.suptitle(f"Loss per model, per level — {title_suffix}", y=0.995)

    lookup = {(e["model"], e["level"]): e for e in entries}
    x_positions = np.arange(len(models))
    width = 0.2

    for idx, lvl in enumerate(levels):
        ax = axes[idx // ncols][idx % ncols]
        for j, c in enumerate(COMPONENTS):
            heights = [
                lookup[(m, lvl)][field][c] if (m, lvl) in lookup else 0.0
                for m in models
            ]
            offsets = x_positions + (j - 1.5) * width
            ax.bar(
                offsets,
                heights,
                width=width,
                color=COMPONENT_COLORS[c],
                label=c,
                edgecolor="black",
                linewidth=0.5,
            )
        ax.set_title(f"Level {lvl}")
        ax.set_xlabel("Model")
        ax.set_ylabel("Loss (log)" if log_y else "Loss")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(models, rotation=25, ha="right")
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y", which="both")
        ax.legend(ncol=2)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Plot: heatmaps (model × level), one subplot per component
# -----------------------------------------------------------------------------
def plot_heatmap_grid(
    entries: list[dict],
    models: list[str],
    levels: list[int],
    output_path: Path,
    *,
    field: str,
    title_suffix: str,
    value_fmt: str = "{:.2f}",
) -> None:
    """2×2 grid of heatmaps, one per component (model × level).

    Y-axis labels embed each model's batch_size as ``{model} (bs={bs})``
    so the reader can see the sampling context directly next to the cells.
    """
    # Map model -> batch_size (first entry wins; bs is constant per model).
    bs_by_model: dict[str, int] = {}
    for e in entries:
        bs_by_model.setdefault(e["model"], e["batch_size"])
    model_labels = [f"{m} (bs={bs_by_model.get(m, '?')})" for m in models]

    # Compact bs string for the suptitle, e.g. "bs=2|4".
    bs_set = sorted({bs for bs in bs_by_model.values()})
    bs_tag = "bs=" + "|".join(str(b) for b in bs_set)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{title_suffix}  (rows=model, cols=level, {bs_tag})", y=0.995)

    for idx, c in enumerate(COMPONENTS):
        ax = axes[idx // 2][idx % 2]
        mat = build_matrix(entries, models, levels, field, c)
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_title(f"{c}")
        ax.set_xticks(np.arange(len(levels)))
        ax.set_xticklabels([str(v) for v in levels])
        ax.set_yticks(np.arange(len(models)))
        ax.set_yticklabels(model_labels)
        ax.set_xlabel("Level")
        ax.set_ylabel("Model (batch_size)")
        # Annotate each cell
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                # Choose text color by luminance
                cmap_val = (v - np.nanmin(mat)) / max(
                    np.nanmax(mat) - np.nanmin(mat), 1e-9
                )
                color = "white" if cmap_val < 0.5 else "black"
                ax.text(
                    j,
                    i,
                    value_fmt.format(v),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                    fontweight="bold",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# CSV + summary text
# -----------------------------------------------------------------------------
def write_csv(entries: list[dict], output_path: Path) -> None:
    """Dump recommended per-config weights to a CSV side-car file."""
    header = [
        "key",
        "dataset",
        "model",
        "level",
        "batch_size",
    ]
    for c in COMPONENTS:
        header.append(f"raw_{c}")
    for c in COMPONENTS:
        header.append(f"cur_w_{c}")
    header.append("cur_total_weighted")
    for c in COMPONENTS:
        header.append(f"rec_w_{c}")
    header.append("rec_total_weighted")

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for e in entries:
            row = [
                e["key"],
                e["dataset"],
                e["model"],
                e["level"],
                e["batch_size"],
            ]
            row += [f"{e['raw'][c]:.6f}" for c in COMPONENTS]
            row += [f"{e['weights'][c]:.6f}" for c in COMPONENTS]
            row.append(f"{e['total_weighted']:.6f}")
            row += [f"{e['recommended'][c]:.6f}" for c in COMPONENTS]
            row.append(f"{e['rec_total_weighted']:.6f}")
            writer.writerow(row)


def write_summary(
    entries: list[dict],
    aggregate: dict,
    target: float,
    output_path: Path,
) -> None:
    """Write a human-readable text summary of the weight recommendation."""
    lines: list[str] = []
    lines.append("Builder Loss-Weight Recommendation Summary")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Target weighted-contribution per component: {target}")
    lines.append(
        "Rule: rec_w[comp] = target / raw_mean[comp]  " "=> weighted_rec[comp] ≈ target"
    )
    lines.append(f"Expected rec_total_weighted ≈ {target * len(COMPONENTS):.3f}")
    lines.append("")
    lines.append("--- Per-component aggregate of recommended weights ---")
    lines.append(
        f"{'component':<12} {'median':>10} {'mean':>10} "
        f"{'min':>10} {'max':>10} {'std':>10}"
    )
    for c in COMPONENTS:
        a = aggregate[c]
        lines.append(
            f"{c:<12} {a['median']:>10.4f} {a['mean']:>10.4f} "
            f"{a['min']:>10.4f} {a['max']:>10.4f} {a['std']:>10.4f}"
        )
    lines.append("")
    lines.append("--- Per-config recommendations (sorted by model, level) ---")
    header = (
        f"{'model':<22} {'lvl':>3} {'bs':>3} "
        + " ".join(f"{'raw_' + c:>11}" for c in COMPONENTS)
        + "   "
        + " ".join(f"{'rec_w_' + c:>10}" for c in COMPONENTS)
        + f"   {'rec_total':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    entries_sorted = sorted(
        entries, key=lambda e: (_model_sort_key(e["model"]), e["level"])
    )
    for e in entries_sorted:
        raw_vals = " ".join(f"{e['raw'][c]:>11.4f}" for c in COMPONENTS)
        rec_vals = " ".join(f"{e['recommended'][c]:>10.4f}" for c in COMPONENTS)
        lines.append(
            f"{e['model']:<22} {e['level']:>3} {e['batch_size']:>3} "
            f"{raw_vals}   {rec_vals}   {e['rec_total_weighted']:>10.4f}"
        )
    lines.append("")
    lines.append("--- Interpretation notes ---")
    lines.append(
        "- recon dominates raw loss (≈1-80); its weight must be model-specific."
    )
    lines.append(
        "- ordering grows rapidly with level (≈1 → 35); needs level-adaptive weight."
    )
    lines.append(
        "- residual is stable (~0.8); keep weight ≈ 1/raw (≈ 1.0 is already close)."
    )
    lines.append(
        "- reasoning varies 4-12; tune weight per (model, level) or use median."
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    """Parse CLI arguments for the weight-recommendation script."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Loss_prepare.json: compare losses across models/levels "
            "and recommend per-config loss weights."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-s",
        "--storage-root",
        type=str,
        default="./",
        help=(
            "Prefix used to compute the default -i / -o paths. "
            "Ignored for paths passed explicitly via -i or -o. "
            "Default is './' (current working directory) — NO silent "
            "project-root fallback. Use to match the -s value that "
            "loss_prepare.py was launched with."
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        default=None,
        help=(
            "Path to <dataset>_Loss_prepare.json. Default: "
            "<storage_root>/EXPERIMENT/nlcpV4/builder/"
            "<dataset>_Loss_prepare.json, where <storage_root> is -s "
            "and <dataset> is --dataset (required when -i is omitted)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help=(
            "Output directory. Default: "
            "<storage_root>/EXPERIMENT/nlcpV4/builder/training_prepare/."
        ),
    )
    parser.add_argument(
        "--target",
        type=float,
        default=0.8,
        help=(
            "Target weighted-contribution per component for the recommender "
            "(default: 0.8, ≈ median residual raw, so residual weight ≈ 1.0)."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional: only analyze configs whose dataset matches this name.",
    )
    parser.add_argument(
        "--module",
        default=None,
        help="Optional: only analyze configs whose module matches this name.",
    )
    return parser.parse_args()


def main():
    """CLI entry point: analyse Loss_prepare.json and emit plots + CSV."""
    args = parse_args()
    storage_root: str = args.storage_root
    base = Path(storage_root)
    if args.input:
        input_path = Path(args.input)
    else:
        if args.dataset is None:
            print(
                "[ERROR] --dataset is required when -i/--input is omitted, "
                "because the default Loss_prepare.json filename is now "
                "<dataset>_Loss_prepare.json."
            )
            return 1
        default_input = (
            base
            / "EXPERIMENT"
            / "nlcpV4"
            / "builder"
            / f"{args.dataset}_Loss_prepare.json"
        )
        input_path = default_input
    default_output_dir = base / "EXPERIMENT" / "nlcpV4" / "builder" / "training_prepare"
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir

    # Surface the resolved storage paths up front. No silent
    # PROJECT_ROOT fallback — the CLI default ``./`` (current working
    # directory) is announced explicitly so the user can verify.
    cwd = Path.cwd().resolve()

    def _abs(p: Path) -> str:
        p2 = p.expanduser()
        if not p2.is_absolute():
            p2 = (cwd / p2).resolve()
        return str(p2)

    print(f"[STORAGE] storage_root = {storage_root!r} (cwd={cwd})")
    print(f"[STORAGE]   input  Loss_prepare.json = {input_path}")
    print(f"[STORAGE]                              (absolute: {_abs(input_path)})")
    print(f"[STORAGE]   output training_prepare/  = {output_dir}")
    print(f"[STORAGE]                              (absolute: {_abs(output_dir)})")

    if not input_path.is_file():
        print(f"[ERROR] Input file not found: {input_path}")
        return 1

    entries = load_entries(input_path)
    if args.dataset is not None:
        entries = [e for e in entries if e["dataset"] == args.dataset]
    if args.module is not None:
        entries = [e for e in entries if e["module"] == args.module]
    if not entries:
        print("[ERROR] No entries to analyze (check --dataset / --module filters).")
        return 1

    datasets = _sorted_unique(e["dataset"] for e in entries)
    modules = _sorted_unique(e["module"] for e in entries)
    models = sorted({e["model"] for e in entries}, key=_model_sort_key)
    levels = _sorted_unique((e["level"] for e in entries), key=int)

    print(
        f"[PREPARE] input={input_path}  entries={len(entries)}  "
        f"datasets={datasets}  modules={modules}  "
        f"models={len(models)}  levels={levels}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _apply_rcparams()

    recommend_weights(entries, target=args.target)
    aggregate = summarize_recommendations(entries)

    # --- Per-model (same model, different levels) ---
    plot_per_model(
        entries,
        models,
        levels,
        output_dir / "loss_by_level_per_model_raw.png",
        field="raw",
        title_suffix="raw losses",
        log_y=True,
    )
    plot_per_model(
        entries,
        models,
        levels,
        output_dir / "loss_by_level_per_model_weighted.png",
        field="weighted",
        title_suffix="weighted losses (current weights)",
        log_y=False,
    )

    # --- Per-level (same level, different models) ---
    plot_per_level(
        entries,
        models,
        levels,
        output_dir / "loss_by_model_per_level_raw.png",
        field="raw",
        title_suffix="raw losses",
        log_y=True,
    )
    plot_per_level(
        entries,
        models,
        levels,
        output_dir / "loss_by_model_per_level_weighted.png",
        field="weighted",
        title_suffix="weighted losses (current weights)",
        log_y=False,
    )

    # --- Heatmaps ---
    plot_heatmap_grid(
        entries,
        models,
        levels,
        output_dir / "heatmap_raw.png",
        field="raw",
        title_suffix="Raw losses by model × level",
        value_fmt="{:.2f}",
    )
    plot_heatmap_grid(
        entries,
        models,
        levels,
        output_dir / "heatmap_weighted.png",
        field="weighted",
        title_suffix="Weighted losses (current weights) by model × level",
        value_fmt="{:.3f}",
    )

    # --- Recommended weights heatmap ---
    plot_heatmap_grid(
        entries,
        models,
        levels,
        output_dir / "recommended_weights_heatmap.png",
        field="recommended",
        title_suffix=(f"Recommended weights (target={args.target}) by model × level"),
        value_fmt="{:.3f}",
    )

    # --- CSV + Summary ---
    write_csv(entries, output_dir / "recommended_weights.csv")
    write_summary(entries, aggregate, args.target, output_dir / "weights_summary.txt")

    print(f"[OK] wrote outputs to {output_dir}")
    print("     plots:")
    for name in sorted(
        p.name for p in output_dir.iterdir() if p.suffix in {".png", ".csv", ".txt"}
    ):
        print(f"       - {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
