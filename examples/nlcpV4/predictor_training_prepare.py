"""Analyze <dataset>_Loss_prepare_<mode>.json (PREDICTOR module) and recommend
per-config loss weights.

Predictor-specialised twin of ``builder_training_prepare.py``. Reuses the
truly generic helpers from the builder variant (matplotlib rc setup,
model-size sort key, sorted-unique, heatmap-matrix builder) and
overrides everything that depends on the component set (builder has 4:
recon/ordering/residual/reasoning; predictor has 2: concept/reasoning)
or the mode split (SHARED vs INDEPENDENT, each in its own JSON).

Key-format contract (one entry per config)::

    "<dataset>/train_<module>_<model>_<level>level_<mode>": {
        "batch_size": int,
        "num_batches": int,
        "weights":  {"concept","reasoning"},
        "per_batch": [ { "raw": {...}, "weighted": {...},
                         "total_weighted": float } ],
        "stats": {
            "raw":            {<comp>: {mean,std,min,max}},
            "weighted":       {<comp>: {mean,std,min,max}},
            "total_weighted": {mean,std,min,max},
        },
    }

The ``_<mode>`` suffix on the key is optional — if absent, the mode is
taken from ``-M/--mode``. ``loss_predictor_prepare.py`` always routes
each mode into its own JSON file, so a single input never mixes modes.

Generated outputs (under
``<storage>/EXPERIMENT/nlcpV4/predictor/<dataset>_<mode>_training_prepare/``)::

    loss_by_level_per_model_raw.png        # same model x different levels
    loss_by_level_per_model_weighted.png
    loss_by_model_per_level_raw.png        # same level x different models
    loss_by_model_per_level_weighted.png
    heatmap_raw.png                        # 1x2 subplots, (model x level)
    heatmap_weighted.png
    recommended_weights_heatmap.png        # recommended w per comp & config
    recommended_weights.csv
    weights_summary.txt                    # human-readable recommendation

Weight-recommendation rule (default)::

    target_value  = --target  (default 1.0)
    rec_w[comp]   = target_value / raw_mean[comp]

so each weighted component contributes ~= ``target_value`` to the total.
Anchoring ``--target`` at the typical reasoning CE magnitude (a few units)
keeps ``reasoning_loss_weight ~= 1/raw_reasoning``, which is a sensible
starting point for predictor tuning. ``concept`` (MSE) then gets a
matching weight that brings it to the same target contribution.

Usage::

    # SHARED mode (default paths — only --dataset and -M required):
    #   -i: <storage>/EXPERIMENT/nlcpV4/predictor/GSM8K_Loss_prepare_shared.json
    #   -o: <storage>/EXPERIMENT/nlcpV4/predictor/GSM8K_shared_training_prepare/
    python3 examples/nlcpV4/predictor_training_prepare.py --dataset GSM8K -M shared

    # INDEPENDENT mode:
    python3 examples/nlcpV4/predictor_training_prepare.py --dataset GSM8K -M independent

    # Tune the recommendation target (target weighted contribution per comp).
    python3 examples/nlcpV4/predictor_training_prepare.py --dataset GSM8K -M shared --target 2.0

    # Override -i / -o explicitly when the file lives outside the default tree.
    python3 examples/nlcpV4/predictor_training_prepare.py -M independent -i path/to/GSM8K_Loss_prepare_independent.json -o path/to/output_dir

    # Pull the same default paths but rebased under a storage root. MUST
    # match the ``-s`` that ``loss_predictor_prepare.py`` was launched with
    # so this tool reads the exact JSON just written.
    python3 examples/nlcpV4/predictor_training_prepare.py -s /Data/ReasoningNLCP --dataset GSM8K -M shared
    python3 examples/nlcpV4/predictor_training_prepare.py -s /Data/ReasoningNLCP --dataset GSM8K -M independent

Arguments:
    -s / --storage-root   Prefix used to compute the DEFAULT ``-i`` /
                          ``-o`` paths. Ignored when ``-i`` / ``-o`` are
                          explicit. Default is ``./`` (current working
                          directory) -- NEVER an implicit project root.
                          The resolved paths are printed as a
                          ``[STORAGE]`` block at startup. MUST match the
                          ``-s`` used when running
                          ``loss_predictor_prepare.py``.
    -i / --input          Path to <dataset>_Loss_prepare_<mode>.json.
                          Default: <storage_root>/EXPERIMENT/nlcpV4/
                          predictor/<dataset>_Loss_prepare_<mode>.json.
    -o / --output-dir     Output directory for plots + CSV. Default:
                          <storage_root>/EXPERIMENT/nlcpV4/predictor/
                          <dataset>_<mode>_training_prepare/.
    --target              Target weighted-contribution per component for
                          the recommender (default: 1.0).
    --dataset             Dataset name (drives default path resolution
                          and in-memory filtering).
    -M / --mode           Predictor mode filter: ``shared`` or
                          ``independent``. SHARED and INDEPENDENT
                          results must never be co-mingled -- each mode
                          lives in its own input JSON and its own
                          output folder.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Reuse the component-agnostic helpers from the builder variant so this
# file stays a thin predictor-specific wrapper (DRY: typography, sorting,
# matrix builder).
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from builder_training_prepare import (  # noqa: E402
    _apply_rcparams,
    _model_sort_key,
    _sorted_unique,
    build_matrix,
)

# Predictor components in fixed display order — MUST match the keys
# used in ``training.loss_weights`` and produced by
# ``compute_predictor_loss``. Order matters for CSV columns and plot
# legend entries.
COMPONENTS = ("concept", "reasoning")
COMPONENT_COLORS = {
    "concept": "tab:blue",
    "reasoning": "tab:red",
}

VALID_MODES = ("shared", "independent")

# Key layout: "<dataset>/train_<module>_<model>_<level>level_<mode>".
# The mode suffix is optional — when absent we fall back to ``args.mode``
# (the file has already been routed per-mode upstream by
# ``loss_predictor_prepare.py``, so every entry in one file belongs to
# the same concrete mode).
KEY_PATTERN = re.compile(
    r"^(?P<dataset>[^/]+)/train_(?P<module>[A-Za-z0-9]+)_"
    r"(?P<model>.+)_(?P<level>\d+)level"
    r"(?:_(?P<mode>shared|independent))?$"
)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def parse_key(key: str) -> dict | None:
    """Split a predictor Loss_prepare JSON key into its structured parts.

    Returns a dict with ``dataset``, ``module``, ``model``, ``level`` (int),
    ``mode`` (``shared`` / ``independent`` / ``None``), or ``None`` when
    the key does not match the expected pattern.
    """
    m = KEY_PATTERN.match(key)
    if m is None:
        return None
    return {
        "dataset": m.group("dataset"),
        "module": m.group("module"),
        "model": m.group("model"),
        "level": int(m.group("level")),
        "mode": m.group("mode"),  # may be None; caller fills from --mode
    }


def load_entries(path: Path, default_mode: str) -> list[dict]:
    """Flatten the JSON into a list of per-config records.

    ``default_mode`` is used when a key lacks the ``_shared`` /
    ``_independent`` suffix (predictor JSONs are always mode-scoped, so
    this fallback is safe and rarely triggers).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries: list[dict] = []
    for key, v in data.items():
        meta = parse_key(key)
        if meta is None:
            print(f"[WARN] Unparseable key skipped: {key}")
            continue
        if meta["mode"] is None:
            meta["mode"] = default_mode
        # Strict access: missing components fail-fast at KeyError.
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


# -----------------------------------------------------------------------------
# Recommendation
# -----------------------------------------------------------------------------
def recommend_weights(entries: list[dict], target: float) -> None:
    """Attach ``recommended`` + ``rec_total_weighted`` to each entry (in place)."""
    for e in entries:
        rec: dict = {}
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
    """One subplot per model: curves of the 2 predictor components vs level."""
    n = len(models)
    ncols = min(n, 2) if n > 0 else 1
    nrows = (n + ncols - 1) // ncols if n > 0 else 1
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7.5 * ncols, 5.0 * nrows), squeeze=False
    )
    fig.suptitle(f"Predictor loss vs Level, per model — {title_suffix}", y=0.995)

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
    """One subplot per level: grouped bars (model x component).

    With 2 components the bars are wider and centered symmetrically
    around each model's tick, giving a cleaner read than the 4-bar
    builder layout.
    """
    n = len(levels)
    ncols = min(n, 3) if n > 0 else 1
    nrows = (n + ncols - 1) // ncols if n > 0 else 1
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7.0 * ncols, 4.8 * nrows), squeeze=False
    )
    fig.suptitle(f"Predictor loss per model, per level — {title_suffix}", y=0.995)

    lookup = {(e["model"], e["level"]): e for e in entries}
    x_positions = np.arange(len(models))
    width = 0.35
    offsets_per_c = [(j - 0.5) * width for j in range(len(COMPONENTS))]

    for idx, lvl in enumerate(levels):
        ax = axes[idx // ncols][idx % ncols]
        for j, c in enumerate(COMPONENTS):
            heights = [
                lookup[(m, lvl)][field][c] if (m, lvl) in lookup else 0.0
                for m in models
            ]
            offsets = x_positions + offsets_per_c[j]
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
# Plot: heatmaps (model x level), one subplot per component
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
    """1x2 grid of heatmaps, one per component (model x level).

    The builder variant uses a 2x2 grid for its 4 components; for 2
    predictor components a 1x2 strip reads more naturally.
    """
    bs_by_model: dict[str, int] = {}
    for e in entries:
        bs_by_model.setdefault(e["model"], e["batch_size"])
    model_labels = [f"{m} (bs={bs_by_model.get(m, '?')})" for m in models]

    bs_set = sorted({bs for bs in bs_by_model.values()})
    bs_tag = "bs=" + "|".join(str(b) for b in bs_set)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), squeeze=False)
    fig.suptitle(f"{title_suffix}  (rows=model, cols=level, {bs_tag})", y=0.995)

    for idx, c in enumerate(COMPONENTS):
        ax = axes[0][idx]
        mat = build_matrix(entries, models, levels, field, c)
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_title(f"{c}")
        ax.set_xticks(np.arange(len(levels)))
        ax.set_xticklabels([str(v) for v in levels])
        ax.set_yticks(np.arange(len(models)))
        ax.set_yticklabels(model_labels)
        ax.set_xlabel("Level")
        ax.set_ylabel("Model (batch_size)")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    continue
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
def write_csv(entries: list[dict], output_path: Path, mode: str) -> None:
    """Dump recommended per-config weights to a CSV sidecar file.

    Columns are a superset of the builder CSV: an extra ``mode`` column
    is inserted right after ``module`` so downstream tooling can merge
    SHARED / INDEPENDENT results when needed without losing provenance.
    """
    header = ["key", "dataset", "module", "mode", "model", "level", "batch_size"]
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
                e["module"],
                e.get("mode", mode),
                e["model"],
                e["level"],
                e["batch_size"],
            ]
            row += [f"{e['raw'][c]:.6f}" for c in COMPONENTS]
            row += [
                f"{float(e['weights'].get(c, float('nan'))):.6f}" for c in COMPONENTS
            ]
            row.append(f"{e['total_weighted']:.6f}")
            row += [f"{e['recommended'][c]:.6f}" for c in COMPONENTS]
            row.append(f"{e['rec_total_weighted']:.6f}")
            writer.writerow(row)


def write_summary(
    entries: list[dict],
    aggregate: dict,
    target: float,
    mode: str,
    output_path: Path,
) -> None:
    """Write a human-readable text summary of the weight recommendation."""
    lines: list[str] = []
    lines.append(f"Predictor Loss-Weight Recommendation Summary  (mode={mode})")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Target weighted-contribution per component: {target}")
    lines.append(
        "Rule: rec_w[comp] = target / raw_mean[comp]  "
        "=> weighted_rec[comp] ~= target"
    )
    lines.append(f"Expected rec_total_weighted ~= {target * len(COMPONENTS):.3f}")
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
    lines.append("--- Interpretation notes (predictor) ---")
    lines.append(
        "- concept (MSE on ground-truth concepts): per-level target, scale "
        "varies with encoder depth and K. Often dominates raw loss; cap via "
        "target/raw to keep parity with reasoning."
    )
    lines.append(
        "- reasoning (CE over solution tokens): stable across models (a few "
        "units), a natural anchor for the target. Keeping reasoning weight "
        "~= 1/raw_reasoning is a sensible starting point."
    )
    lines.append(
        f"- mode={mode!r}: SHARED reuses the builder's reason_model + "
        "back_proj (no LoRA allowed); INDEPENDENT has its own backbone "
        "(LoRA optional). The two modes MUST NOT be co-mingled — each "
        "has its own Loss_prepare_<mode>.json and its own training_prepare/ "
        "output folder."
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the predictor weight-recommendation script."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze <dataset>_Loss_prepare_<mode>.json (produced by "
            "loss_predictor_prepare.py): compare concept/reasoning losses "
            "across models x levels and recommend per-config weights."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-s",
        "--storage-root",
        type=str,
        default="./",
        help=(
            "Prefix used to compute the default -i / -o paths. Ignored "
            "when -i / -o are explicit. Default is './' (current working "
            "directory) — NO silent project-root fallback. Must match "
            "the -s used when loss_predictor_prepare.py was launched."
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        default=None,
        help=(
            "Path to <dataset>_Loss_prepare_<mode>.json. Default: "
            "<storage_root>/EXPERIMENT/nlcpV4/predictor/"
            "<dataset>_Loss_prepare_<mode>.json. --dataset and -M are "
            "required when -i is omitted."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help=(
            "Output directory. Default: "
            "<storage_root>/EXPERIMENT/nlcpV4/predictor/"
            "<dataset>_<mode>_training_prepare/. --dataset and -M are "
            "required when -o is omitted."
        ),
    )
    parser.add_argument(
        "--target",
        type=float,
        default=1.0,
        help=(
            "Target weighted-contribution per component for the recommender "
            "(default: 1.0). For predictor, anchoring around the reasoning "
            "CE magnitude is typical so reasoning_loss_weight ~= "
            "1/raw_reasoning."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Dataset name. Drives default path resolution and in-memory "
            "filtering. Required when -i / -o are omitted."
        ),
    )
    parser.add_argument(
        "-M",
        "--mode",
        default=None,
        choices=sorted(VALID_MODES),
        help=(
            "Predictor mode: 'shared' (backbone reused from builder; "
            "LoRA forbidden) or 'independent' (own backbone; LoRA "
            "optional). SHARED and INDEPENDENT results MUST NOT be "
            "co-mingled — each mode has its own "
            "<dataset>_Loss_prepare_<mode>.json and its own output folder."
        ),
    )
    return parser.parse_args()


def _infer_mode_from_filename(path: Path) -> str | None:
    """Return 'shared' / 'independent' if the filename ends with that suffix."""
    stem = path.stem
    for m in VALID_MODES:
        if stem.endswith(f"_{m}"):
            return m
    return None


def main() -> int:
    """CLI entry point: analyse predictor Loss_prepare JSON and emit plots + CSV."""
    args = parse_args()
    storage_root: str = args.storage_root
    base = Path(storage_root)

    # --- Resolve input path + mode -----------------------------------------
    if args.input:
        input_path = Path(args.input)
        if args.mode is None:
            args.mode = _infer_mode_from_filename(input_path)
            if args.mode is None:
                print(
                    "[ERROR] Could not infer mode from -i filename "
                    f"{input_path.name!r}. Pass -M/--mode explicitly."
                )
                return 1
    else:
        if args.dataset is None or args.mode is None:
            print(
                "[ERROR] --dataset and -M/--mode are required when -i is "
                "omitted. Default input is "
                "<storage>/EXPERIMENT/nlcpV4/predictor/"
                "<dataset>_Loss_prepare_<mode>.json."
            )
            return 1
        input_path = (
            base
            / "EXPERIMENT"
            / "nlcpV4"
            / "predictor"
            / f"{args.dataset}_Loss_prepare_{args.mode}.json"
        )

    # --- Resolve output directory ------------------------------------------
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if args.dataset is None or args.mode is None:
            print(
                "[ERROR] --dataset and -M/--mode are required when "
                "-o/--output-dir is omitted."
            )
            return 1
        output_dir = (
            base
            / "EXPERIMENT"
            / "nlcpV4"
            / "predictor"
            / f"{args.dataset}_{args.mode}_training_prepare"
        )

    # Surface the resolved storage paths up front (no silent fallback).
    cwd = Path.cwd().resolve()

    def _abs(p: Path) -> str:
        p2 = p.expanduser()
        if not p2.is_absolute():
            p2 = (cwd / p2).resolve()
        return str(p2)

    print(f"[STORAGE] storage_root = {storage_root!r} (cwd={cwd})")
    print(f"[STORAGE]   mode                     = {args.mode}")
    print(f"[STORAGE]   input  Loss_prepare.json = {input_path}")
    print(f"[STORAGE]                              (absolute: {_abs(input_path)})")
    print(f"[STORAGE]   output training_prepare/ = {output_dir}")
    print(f"[STORAGE]                              (absolute: {_abs(output_dir)})")

    if not input_path.is_file():
        print(f"[ERROR] Input file not found: {input_path}")
        return 1

    entries = load_entries(input_path, default_mode=args.mode)
    if args.dataset is not None:
        entries = [e for e in entries if e["dataset"] == args.dataset]
    # Defensive mode filter — loss_predictor_prepare.py already routes
    # per mode, but guard against ad-hoc / hand-merged JSONs that mix.
    entries = [e for e in entries if e["mode"] == args.mode]
    if not entries:
        print(
            "[ERROR] No entries to analyze (check --dataset / -M filters "
            "and the input file)."
        )
        return 1

    datasets = _sorted_unique(e["dataset"] for e in entries)
    models = sorted({e["model"] for e in entries}, key=_model_sort_key)
    levels = _sorted_unique((e["level"] for e in entries), key=int)

    print(
        f"[PREPARE] input={input_path}  entries={len(entries)}  "
        f"datasets={datasets}  mode={args.mode}  "
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
        title_suffix=f"raw losses (mode={args.mode})",
        log_y=True,
    )
    plot_per_model(
        entries,
        models,
        levels,
        output_dir / "loss_by_level_per_model_weighted.png",
        field="weighted",
        title_suffix=f"weighted losses (current weights, mode={args.mode})",
        log_y=False,
    )

    # --- Per-level (same level, different models) ---
    plot_per_level(
        entries,
        models,
        levels,
        output_dir / "loss_by_model_per_level_raw.png",
        field="raw",
        title_suffix=f"raw losses (mode={args.mode})",
        log_y=True,
    )
    plot_per_level(
        entries,
        models,
        levels,
        output_dir / "loss_by_model_per_level_weighted.png",
        field="weighted",
        title_suffix=f"weighted losses (current weights, mode={args.mode})",
        log_y=False,
    )

    # --- Heatmaps ---
    plot_heatmap_grid(
        entries,
        models,
        levels,
        output_dir / "heatmap_raw.png",
        field="raw",
        title_suffix=f"Raw losses by model x level (mode={args.mode})",
        value_fmt="{:.3f}",
    )
    plot_heatmap_grid(
        entries,
        models,
        levels,
        output_dir / "heatmap_weighted.png",
        field="weighted",
        title_suffix=(
            f"Weighted losses (current weights) by model x level " f"(mode={args.mode})"
        ),
        value_fmt="{:.3f}",
    )

    # --- Recommended weights heatmap ---
    plot_heatmap_grid(
        entries,
        models,
        levels,
        output_dir / "recommended_weights_heatmap.png",
        field="recommended",
        title_suffix=(
            f"Recommended weights (target={args.target}, mode={args.mode}) "
            f"by model x level"
        ),
        value_fmt="{:.3f}",
    )

    # --- CSV + Summary ---
    write_csv(entries, output_dir / "recommended_weights.csv", mode=args.mode)
    write_summary(
        entries, aggregate, args.target, args.mode, output_dir / "weights_summary.txt"
    )

    print(f"[OK] wrote outputs to {output_dir}")
    print("     plots:")
    for name in sorted(
        p.name for p in output_dir.iterdir() if p.suffix in {".png", ".csv", ".txt"}
    ):
        print(f"       - {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
