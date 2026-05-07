"""Render concept-pyramid diagnostic figures from ``eval_builder.py`` dumps.

This tool consumes the per-sample tensors that ``eval_builder.py`` already
serialised to disk and renders the full suite of concept-pyramid diagnostic
figures INTO those very same sample folders. No checkpoint load, no forward
pass — the heavy lifting was already done by eval_builder; this script just
reads ``pyramid.pt`` + ``input.json`` and plots.

Storage layout
==============
Given a builder config (e.g. the AutoWeighted 4-level GSM8K run), the
resolved tree is::

    <storage_root>/EXPERIMENT/nlcpV4/builder/
        GSM8K_Qwen2.5-0.5B_4level_AutoWeighted/logs/eval_builder/<mode>/
            # ----- aggregate figures pooling ALL samples (written here) -----
            aggregate_concept_pca_level{k}.{png,pdf}                    # per level k
            sample_<main_id_1>/
                input.json           # written by eval_builder
                pyramid.pt           # written by eval_builder (requires -v 1)
                reasoning.json       # written by eval_builder
                timing.json          # written by eval_builder
                losses.json          # written by eval_builder
                # ----- figures produced by THIS script (written in place) -----
                # Every plot is a SEPARATE file; multi-level analyses are split
                # into per-level (and per-slot) files — subplots are avoided so
                # figures stay legible at any pyramid depth.
                attention_heatmap_level{k}.{png,pdf}                    # per level k
                attention_text_overlay_level{k}.{png,pdf}               # per level k
                attention_text_per_concept_level{k}_slot{j}.{png,pdf}   # per (k, slot j)
                concept_norms_level{k}.{png,pdf}                        # per level k
                intra_level_similarity_level{k}.{png,pdf}               # per level k
                concept_pca_level{k}.{png,pdf}                          # per level k
                # Single-axis plots (cross-level comparison lives on one axis):
                residual_decomposition.{png,pdf}
                inter_level_similarity.{png,pdf}
                attention_entropy.{png,pdf}
                token_coverage.{png,pdf}
                concept_pca.{png,pdf}
            sample_<main_id_2>/
                ...

Every sample folder becomes self-contained: raw Q/CoT in ``input.json``,
intermediate tensors in ``pyramid.pt``, reasoning output in
``reasoning.json``, and every derived figure beside them. Open one folder
→ see everything about that sample.

Prerequisites
=============
You MUST have run ``eval_builder.py`` for this config first, with
``-v 1`` (i.e. ``intermediate_vector_save: 1`` in the YAML) so that
``pyramid.pt`` files exist. Example::

    python3 examples/nlcpV4/eval_builder.py -c configs/nlcpV4/GSM8K/AutoWeighted/train_builder_Qwen2.5-0.5B_4level.yml -s /Data/ReasoningNLCP --mode teacher_forced --max-samples 50 -v 1

WHAT IS ANALYZED
================
The builder's :class:`PyramidOutput` carries two first-class tensors
per level, both of which eval_builder serialises into ``pyramid.pt``:

    * ``C_k`` (concept encodings)   : [1, L_k, D]
    * ``A_k`` (soft attention map)  : [1, L_k, L]

This script reads them back (plus ``projected_hidden``,
``reconstruction``, ``attention_mask``, …) and runs the analyses below.

AVAILABLE ANALYSES (select via ``--analyses``)
==============================================
    ``attention_heatmaps``      — per-level attention heatmap (ONE file per
                                  level). Shows which CoT tokens each
                                  concept slot attends to.
    ``attention_text_overlay``  — per-level attention on CoT text (ONE
                                  file per level). Box colours encode
                                  slot-summed attention per token.
    ``attention_text_per_concept`` — per-level, per-concept attention on
                                     text (ONE file per (level, slot)).
                                     Isolates what each concept slot
                                     reads individually.
    ``concept_norms``           — per-level histogram of ``||C_k[j]||``
                                  (ONE file per level).
    ``residual_decomposition``  — single-axis line plot of ``||H_rest_k||``,
                                  ``||R_k||``, ``||H_hat_k||`` vs level.
    ``intra_level_similarity``  — slot-to-slot cosine similarity heatmap
                                  (ONE file per level).
    ``inter_level_similarity``  — single K×K cosine-similarity matrix over
                                  slot-averaged per-level concepts.
    ``attention_entropy``       — single-axis per-level box plots of
                                  ``H(A_{k,j})``.
    ``token_coverage``          — single-axis per-level curves of column-
                                  summed attention over normalised token
                                  position.
    ``concept_pca``             — single 2-D PCA scatter of ALL concepts
                                  coloured by level (cross-level view).
    ``concept_pca_per_level``   — per-level 2-D PCA scatter (ONE file per
                                  level), using that level's own top-2
                                  principal axes. Reveals intra-level
                                  geometry.
    ``aggregate_concept_pca_per_level`` — AGGREGATE per-level 2-D PCA
                                          across ALL samples (ONE file
                                          per level, written to
                                          eval_root beside the sample
                                          folders). Slot index encoded
                                          by marker shape; sample index
                                          encoded by colour. Lets you
                                          compare slot geometry across
                                          the whole dataset at once.
    ``all`` (default)           — run every analysis above.

Usage
-----
Single config (reads every ``sample_*/pyramid.pt`` under the eval_builder
tree for the chosen mode, plots in place)::

    python3 examples/nlcpV4/builder_concept_pyramid_analysis.py -c configs/nlcpV4/GSM8K/AutoWeighted/train_builder_Qwen2.5-0.5B_4level.yml --mode teacher_forced

Batch over an entire dataset dir (every ``train_builder_*.yml`` under it)::

    python3 examples/nlcpV4/builder_concept_pyramid_analysis.py -d GSM8K/AutoWeighted -e all --mode teacher_forced

Select a subset of analyses::

    python3 examples/nlcpV4/builder_concept_pyramid_analysis.py -c configs/nlcpV4/GSM8K/AutoWeighted/train_builder_Qwen2.5-0.5B_4level.yml --analyses attention_heatmaps,attention_text_overlay,concept_norms

Arguments
---------
    -s / --storage-root   Prefix prepended to relative log paths. MUST match
                          the value used when ``eval_builder`` ran.
                          Default: ``./`` (current working directory).
    -c / --config         Direct path to a single ``train_builder_*.yml``.
                          Mutually exclusive with ``-d``/``-e``.
    -d / --dataset        Dataset dir under ``configs/nlcpV4/`` (may be
                          nested, e.g. ``GSM8K/AutoWeighted``). Required when
                          ``-c`` is not given.
    -e / --experiment     Config stem after ``train_builder_`` or ``all`` to
                          process every matching config. Required when ``-c``
                          is not given.
    --mode                Which ``eval_builder/<mode>/`` subdirectory to read.
                          One of ``teacher_forced`` (default), ``free_generation``,
                          ``both``. MUST match the ``--mode`` passed to
                          ``eval_builder``.
    -o / --overlap        If true (default), overwrite existing figures.
                          ``--no-overlap`` skips configs whose per-sample
                          figures are already present.
    --analyses            Comma-separated subset of analysis keys or ``all``
                          (default).
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

from ram.utils import apply_storage_root, load_config, print_storage_paths

logger = logging.getLogger(__name__)

# --- Batch-mode constants ------------------------------------------
CONFIGS_ROOT = PROJECT_ROOT / "configs" / "nlcpV4"
ALL_KEYWORD = "all"
# eval_builder.py writes per-sample dumps under
#   <log_path>/eval_builder/<mode>/sample_<main_id>/{input,pyramid,reasoning,\u2026}.
# We plot directly into those sample folders (no separate output tree) so the
# raw inputs, intermediate tensors and diagnostic figures sit side-by-side.
EVAL_BUILDER_DIR_NAME = "eval_builder"
VALID_EVAL_MODES = ("teacher_forced", "free_generation", "both")
DEFAULT_EVAL_MODE = "teacher_forced"
PYRAMID_DUMP_NAME = "pyramid.pt"
INPUT_JSON_NAME = "input.json"
# Canonical analysis keys — must match the dispatch table in _run_concept_analysis.
# Every key produces a single file per sample named ``<key>.{png,pdf}`` that
# lives inside the sample's own folder, so the output layout is flat and
# self-describing: open a sample folder → see everything about that sample.
ALL_ANALYSES = (
    "attention_heatmaps",
    "attention_text_overlay",
    "attention_text_per_concept",
    "concept_norms",
    "residual_decomposition",
    "intra_level_similarity",
    "inter_level_similarity",
    "attention_entropy",
    "token_coverage",
    "concept_pca",
    "concept_pca_per_level",
    "aggregate_concept_pca_per_level",
)
# Per-sample SENTINEL filename stem — used by ``_existing_outputs`` to
# decide whether a sample's figures were already produced. Multi-level
# analyses write many files, but the level-0 (and slot-0) file is always
# present when the run completed, so it's a safe existence probe. Single
# plots map to their exact filename. These files live INSIDE each
# ``sample_*/`` directory.
_PER_SAMPLE_ANALYSIS_SENTINEL_STEMS: Dict[str, str] = {
    "attention_heatmaps": "attention_heatmap_level0",
    "attention_text_overlay": "attention_text_overlay_level0",
    "attention_text_per_concept": "attention_text_per_concept_level0_slot0",
    "concept_norms": "concept_norms_level0",
    "residual_decomposition": "residual_decomposition",
    "intra_level_similarity": "intra_level_similarity_level0",
    "inter_level_similarity": "inter_level_similarity",
    "attention_entropy": "attention_entropy",
    "token_coverage": "token_coverage",
    "concept_pca": "concept_pca",
    "concept_pca_per_level": "concept_pca_level0",
}
# Aggregate SENTINEL filename stem — these files live at the eval_root
# level (beside the ``sample_*/`` folders), NOT inside them, because they
# pool data from every sample.
_AGGREGATE_ANALYSIS_SENTINEL_STEMS: Dict[str, str] = {
    "aggregate_concept_pca_per_level": "aggregate_concept_pca_level0",
}


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    The script auto-discovers ``sample_*/pyramid.pt`` under
    ``<log_path>/eval_builder/<mode>/`` (resolved from the YAML via
    ``apply_storage_root``) and renders figures in place, so the CLI
    surface is intentionally small: choose a config (or a dataset batch),
    choose a mode, choose an analysis subset — that's it.
    """
    parser = argparse.ArgumentParser(
        description="Render concept-pyramid figures from eval_builder dumps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-s",
        "--storage-root",
        type=str,
        default="./",
        help=(
            "Prefix for relative config.log paths. MUST match the value "
            "used when eval_builder ran so this script reads the same "
            "eval_builder/<mode>/ tree. Default './' (current working directory)."
        ),
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help=(
            "Direct path to a single train_builder_*.yml. Mutually "
            "exclusive with -d/-e; when given, dataset/experiment "
            "discovery is bypassed and this config drives the "
            "eval_builder output-tree lookup."
        ),
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default=None,
        help=(
            "Dataset dir under configs/nlcpV4/ (may be nested, e.g. "
            "'GSM8K/AutoWeighted'). Required when -c is not given."
        ),
    )
    parser.add_argument(
        "-e",
        "--experiment",
        default=None,
        help=(
            "Config stem after 'train_builder_' or 'all' to process every "
            "matching config under the dataset. Required when -c is not given."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=list(VALID_EVAL_MODES),
        default=DEFAULT_EVAL_MODE,
        help=(
            "Which eval_builder/<mode>/ subdirectory to read. MUST match "
            "the --mode passed to eval_builder. One of "
            f"{', '.join(VALID_EVAL_MODES)}. Default '{DEFAULT_EVAL_MODE}'."
        ),
    )
    parser.add_argument(
        "-o",
        "--overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true (default), overwrite existing figures. --no-overlap "
            "skips configs whose per-sample figures already exist."
        ),
    )
    parser.add_argument(
        "--analyses",
        type=str,
        default=ALL_KEYWORD,
        help=(
            "Comma-separated subset of analysis keys or 'all' (default). "
            f"Valid keys: {', '.join(ALL_ANALYSES)}."
        ),
    )
    return parser.parse_args()


def _parse_analyses(arg: str) -> Tuple[str, ...]:
    """Validate and normalise the ``--analyses`` argument."""
    if arg.strip() == ALL_KEYWORD:
        return ALL_ANALYSES
    keys = tuple(k.strip() for k in arg.split(",") if k.strip())
    bad = [k for k in keys if k not in ALL_ANALYSES]
    if bad:
        raise SystemExit(
            f"[ERROR] Unknown --analyses keys: {bad}. "
            f"Valid keys: {', '.join(ALL_ANALYSES)} (or 'all')."
        )
    return keys


def discover_configs(dataset: str, experiment: str) -> List[Path]:
    """Resolve (-d, -e) to a list of ``train_builder_*.yml`` paths.

    Shares the layout convention of :mod:`builder_training_analysis`:
    ``configs/nlcpV4/{dataset}/train_builder_{experiment}.yml``; ``dataset``
    may be nested and ``experiment`` may be ``'all'`` for recursive match.
    """
    dataset_dir = CONFIGS_ROOT / dataset
    if not dataset_dir.is_dir():
        print(f"[ERROR] Dataset dir not found: {dataset_dir}")
        return []

    prefix = "train_builder_"
    if experiment == ALL_KEYWORD:
        return sorted(dataset_dir.rglob(f"{prefix}*.yml"))

    p = dataset_dir / f"{prefix}{experiment}.yml"
    if p.is_file():
        return [p]
    matches = sorted(dataset_dir.rglob(f"{prefix}{experiment}.yml"))
    if matches:
        return matches
    print(f"[ERROR] Config file not found: {p}")
    return []


# =============================================================================
# Per-sample disk loading (from eval_builder dumps)
# =============================================================================


def _load_pyramid_from_disk(pt_path: Path) -> SimpleNamespace:
    """Reconstruct a PyramidOutput-shaped namespace from eval_builder's pyramid.pt.

    eval_builder's ``_pyramid_to_dump_dict`` serialises::

        concepts:             list[Tensor[1, L_k, D]]   (K entries)
        attention_weights:    list[Tensor[1, L_k, L]]   (K entries)
        reconstruction:       list[Tensor[1, L, D]]     (K entries)
        encoder_hidden_states: Tensor[1, L, D_enc]
        projected_hidden:     Tensor[1, L, D]
        reconstructed_hidden: Tensor[1, L, D]
        residual_hidden:      Tensor[1, L, D]
        attention_mask:       Tensor[1, L]

    The plotting functions access a strict subset of PyramidOutput's
    surface — ``num_levels``, ``level_lengths``, ``concepts``,
    ``level_outputs[k].attention_weights / .reconstruction``,
    ``projected_hidden``, ``attention_mask`` — so a SimpleNamespace with
    a matching level_outputs list of inner namespaces is sufficient. No
    downstream plotting code needs to change.
    """
    data = torch.load(pt_path, map_location="cpu")
    concepts = list(data["concepts"])
    num_levels = len(concepts)
    level_lengths = [int(c.shape[1]) for c in concepts]
    level_outputs = [
        SimpleNamespace(
            attention_weights=data["attention_weights"][k],
            reconstruction=data["reconstruction"][k],
        )
        for k in range(num_levels)
    ]
    return SimpleNamespace(
        num_levels=num_levels,
        level_lengths=level_lengths,
        concepts=concepts,
        level_outputs=level_outputs,
        encoder_hidden_states=data["encoder_hidden_states"],
        projected_hidden=data["projected_hidden"],
        reconstructed_hidden=data["reconstructed_hidden"],
        residual_hidden=data["residual_hidden"],
        attention_mask=data["attention_mask"],
    )


def _load_sample_input(input_path: Path) -> Dict[str, Any]:
    """Read ``input.json`` (main_id, question, cot_answer, solution…).

    eval_builder.py writes this via ``_dump_builder_sample``; field names
    are stable (see ``BuilderInput`` contract). Missing fields fall back
    to empty strings in the caller.
    """
    with input_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_tokenizer_from_config(config: Dict[str, Any]):
    """Instantiate the HuggingFace tokenizer used by the builder at eval time.

    The builder tokenises CoT with this tokenizer and
    ``max_length = config['model']['pyramid']['max_seq_len']`` (see
    ``concept_builder.py``). Reloading the same tokenizer here lets us
    produce column labels whose order is identical to the saved
    ``attention_weights[k]`` tensor's token dimension on the valid-length
    prefix — no label drift between figure and tensor.
    """
    model_name = config["model"]["reason_model"]["reason_model_name"]
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _retokenize_cot(tokenizer, cot_text: str, max_length: int) -> List[str]:
    """Retokenise a CoT string into a list of decoded token strings.

    Mirrors the builder's tokenisation parameters (``truncation=True,
    max_length=max_seq_len``). ``padding=False`` is safe here because
    eval_builder runs with batch_size=1 — the saved attention tensors'
    token dimension already equals the un-padded valid length.
    """
    enc = tokenizer(
        cot_text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_length,
    )
    ids = enc["input_ids"][0]
    # One decode-per-id keeps tokens aligned with attention columns.
    return [tokenizer.decode([int(t)]) for t in ids.tolist()]


# =============================================================================
# Figure helpers
# =============================================================================


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """Save ``fig`` as both ``<stem>.png`` (dpi=150) and ``<stem>.pdf``."""
    fig.savefig(output_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _apply_plot_style() -> None:
    """Uniform styling across all figures (matches predictor_training_analysis)."""
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "axes.labelweight": "bold",
            "legend.fontsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.titlesize": 17,
            "figure.titleweight": "bold",
        }
    )


def _short(s: str, n: int = 8) -> str:
    """Truncate a token string for compact tick labels."""
    s = s.replace("\n", "⏎").replace("\t", "→")
    return s if len(s) <= n else s[: n - 1] + "…"


def _tick_positions(length: int, max_ticks: int = 20) -> np.ndarray:
    """Evenly sample up to ``max_ticks`` positions from ``[0, length)``."""
    if length <= max_ticks:
        return np.arange(length)
    return np.linspace(0, length - 1, max_ticks).round().astype(int)


def _per_row_valid_length(pyramid: SimpleNamespace) -> np.ndarray:
    """Return the number of valid (non-pad) tokens per row as ``[B]`` array."""
    if pyramid.attention_mask is None:
        # Unmasked case — every position counts.
        B, L, _ = pyramid.projected_hidden.shape
        return np.full((B,), L, dtype=np.int64)
    return pyramid.attention_mask.sum(dim=1).cpu().numpy().astype(np.int64)


# =============================================================================
# Analyses
# =============================================================================


def _plot_attention_heatmaps(
    pyramid: SimpleNamespace,
    tokens_per_row: Sequence[Sequence[str]],
    output_dir: Path,
    experiment_name: str,
    *,
    file_stem: str = "attention_heatmap",
) -> None:
    """Per-level attention heatmap — ONE standalone file per level.

    Writes ``<file_stem>_level{k}.{png,pdf}`` for every ``k \u2208 [0, K)``.
    Each figure has a single axes: rows = concept slots ``j \u2208 [0, L_k)``,
    cols = valid CoT tokens (up to 20 tick labels) for sample 0.
    """
    K = pyramid.num_levels
    level_lengths = pyramid.level_lengths
    B = pyramid.projected_hidden.shape[0]
    if B == 0:
        return
    valid_lens = _per_row_valid_length(pyramid)
    i = 0  # batch_size = 1 by design
    L_i = int(valid_lens[i])
    tokens_i = list(tokens_per_row[i])[:L_i]

    cot_preview = "".join(tokens_per_row[i][:30]).replace("\n", " ").strip()
    cot_preview = (
        cot_preview if len(cot_preview) <= 110 else cot_preview[:107] + "\u2026"
    )

    for k in range(K):
        A_k = pyramid.level_outputs[k].attention_weights[i, :, :L_i]
        mat = A_k.detach().cpu().float().numpy()
        L_k = level_lengths[k]
        fig_h = max(2.4, 0.35 * L_k + 2.0)
        fig, ax = plt.subplots(1, 1, figsize=(16, fig_h))
        im = ax.imshow(
            mat,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            vmin=0.0,
        )
        ax.set_ylabel(f"slot j  (L_k={L_k})")
        ax.set_yticks(np.arange(L_k))
        ax.set_yticklabels([str(j) for j in range(L_k)])
        xticks = _tick_positions(L_i, max_ticks=20)
        ax.set_xticks(xticks)
        ax.set_xticklabels(
            [_short(tokens_i[t], n=8) for t in xticks],
            rotation=60,
            ha="right",
        )
        ax.set_xlabel("CoT token")
        fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
        ax.set_title(
            f"{experiment_name} \u2014 attention heatmap (level {k})\n{cot_preview}"
        )
        fig.tight_layout()
        _save_figure(fig, output_dir, f"{file_stem}_level{k}")


def _plot_attention_text_overlay(
    pyramid: SimpleNamespace,
    tokens_per_row: Sequence[Sequence[str]],
    output_dir: Path,
    experiment_name: str,
    *,
    file_stem: str = "attention_text_overlay",
    tokens_per_line: int = 18,
) -> None:
    """Per-level attention-on-text overlay — ONE standalone file per level.

    For sample 0 (B==1) we aggregate each level's attention across concept
    slots (``a_k[t] = sum_j A_k[0, j, t]``), normalise per level, then
    draw the CoT token sequence as a grid of rounded boxes wrapping every
    ``tokens_per_line`` tokens. Box background colour (YlOrRd) encodes
    ``a_k[t]``. ONE figure per level; written as
    ``<file_stem>_level{k}.{png,pdf}``.
    """
    K = pyramid.num_levels
    B = pyramid.projected_hidden.shape[0]
    if B == 0:
        return
    valid_lens = _per_row_valid_length(pyramid)
    cmap = plt.get_cmap("YlOrRd")
    i = 0  # batch_size = 1 by design

    L_i = int(valid_lens[i])
    if L_i == 0:
        return
    tokens_i = list(tokens_per_row[i])[:L_i]
    n_lines = (L_i + tokens_per_line - 1) // tokens_per_line

    fig_w = min(20.0, 1.05 * tokens_per_line + 2.0)
    fig_h = max(2.4, 0.45 * n_lines + 1.4)

    preview = "".join(tokens_i[:30]).replace("\n", " ").strip()
    preview = preview if len(preview) <= 110 else preview[:107] + "\u2026"

    for k in range(K):
        A_k = (
            pyramid.level_outputs[k]
            .attention_weights[i, :, :L_i]
            .detach()
            .cpu()
            .float()
            .numpy()
        )
        attn = A_k.sum(axis=0)  # [L_i]
        a_max = float(attn.max()) if attn.size and float(attn.max()) > 0 else 1.0
        norm_attn = attn / a_max

        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
        ax.set_xlim(-0.55, tokens_per_line - 0.45)
        ax.set_ylim(-(n_lines - 1) - 0.6, 0.6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        for t_idx, tok in enumerate(tokens_i):
            row = t_idx // tokens_per_line
            col = t_idx % tokens_per_line
            weight = float(norm_attn[t_idx])
            face = cmap(weight)
            luminance = 0.299 * face[0] + 0.587 * face[1] + 0.114 * face[2]
            txt_color = "white" if luminance < 0.5 else "black"
            ax.text(
                col,
                -row,
                _short(tok, n=10),
                ha="center",
                va="center",
                fontsize=8,
                color=txt_color,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor=face,
                    edgecolor="0.6",
                    linewidth=0.3,
                ),
            )

        import matplotlib as _mpl

        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=_mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        )
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.012, pad=0.01)
        ax.set_title(
            f"{experiment_name} \u2014 attention-on-text (level {k}, "
            f"L_k={pyramid.level_lengths[k]}, max_w={a_max:.3f})\n{preview}",
            loc="left",
            fontsize=11,
        )
        fig.tight_layout()
        _save_figure(fig, output_dir, f"{file_stem}_level{k}")


def _plot_attention_text_per_concept(
    pyramid: SimpleNamespace,
    tokens_per_row: Sequence[Sequence[str]],
    output_dir: Path,
    experiment_name: str,
    *,
    file_stem: str = "attention_text_per_concept",
    tokens_per_line: int = 18,
) -> None:
    """Per-level, per-concept attention on text — ONE file per (level, slot).

    For EVERY level k AND EVERY concept slot ``j \u2208 [0, L_k)`` this
    function writes ``<file_stem>_level{k}_slot{j}.{png,pdf}``. The CoT
    is rendered as wrapped rows of rounded token boxes whose background
    colour (YlOrRd) encodes ``A_k[0, j, t]`` normalised by that slot's
    own maximum, so every file uses the full [0, 1] colour range.

    Keeping each concept slot in its own file lets the reader scroll
    through them independently at any pyramid depth without subplots
    shrinking each row to illegibility.
    """
    K = pyramid.num_levels
    B = pyramid.projected_hidden.shape[0]
    if B == 0:
        return
    valid_lens = _per_row_valid_length(pyramid)
    cmap = plt.get_cmap("YlOrRd")
    i = 0  # batch_size = 1 by design

    L_i = int(valid_lens[i])
    if L_i == 0:
        return
    tokens_i = list(tokens_per_row[i])[:L_i]
    n_lines = (L_i + tokens_per_line - 1) // tokens_per_line

    fig_w = min(20.0, 1.05 * tokens_per_line + 2.0)
    fig_h = max(2.4, 0.45 * n_lines + 1.4)

    preview = "".join(tokens_i[:30]).replace("\n", " ").strip()
    preview = preview if len(preview) <= 110 else preview[:107] + "\u2026"

    for k in range(K):
        A_k = (
            pyramid.level_outputs[k]
            .attention_weights[i, :, :L_i]
            .detach()
            .cpu()
            .float()
            .numpy()
        )  # [L_k, L_i]
        L_k = pyramid.level_lengths[k]
        for j in range(L_k):
            row = A_k[j]
            r_max = float(row.max()) if row.size and float(row.max()) > 0 else 1.0
            norm_row = row / r_max

            fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
            ax.set_xlim(-0.55, tokens_per_line - 0.45)
            ax.set_ylim(-(n_lines - 1) - 0.6, 0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            for t_idx, tok in enumerate(tokens_i):
                r = t_idx // tokens_per_line
                c = t_idx % tokens_per_line
                weight = float(norm_row[t_idx])
                face = cmap(weight)
                luminance = 0.299 * face[0] + 0.587 * face[1] + 0.114 * face[2]
                txt_color = "white" if luminance < 0.5 else "black"
                ax.text(
                    c,
                    -r,
                    _short(tok, n=10),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=txt_color,
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor=face,
                        edgecolor="0.6",
                        linewidth=0.3,
                    ),
                )

            import matplotlib as _mpl

            sm = plt.cm.ScalarMappable(
                cmap=cmap, norm=_mpl.colors.Normalize(vmin=0.0, vmax=1.0)
            )
            sm.set_array([])
            fig.colorbar(sm, ax=ax, fraction=0.012, pad=0.01)
            ax.set_title(
                f"{experiment_name} \u2014 attention-on-text "
                f"(level {k}, slot {j}, L_k={L_k}, max_w={r_max:.3f})\n{preview}",
                loc="left",
                fontsize=11,
            )
            fig.tight_layout()
            _save_figure(fig, output_dir, f"{file_stem}_level{k}_slot{j}")


def _plot_concept_norms(
    pyramid: SimpleNamespace,
    output_dir: Path,
    experiment_name: str,
) -> None:
    """Per-level histograms of ``||C_k[b,j]||`` — ONE file per level.

    Writes ``concept_norms_level{k}.{png,pdf}`` for every level. Sharing
    the x-axis across all files (via a global max) makes the trend
    towards smaller norms at deeper levels visible when flipping through
    the files.
    """
    K = pyramid.num_levels
    norms_per_level: List[np.ndarray] = []
    for k in range(K):
        C_k = pyramid.concepts[k]  # [B, L_k, D]
        n_k = torch.linalg.vector_norm(C_k, dim=-1).detach().cpu().float().numpy()
        norms_per_level.append(n_k.reshape(-1))

    global_max = (
        float(max(n.max() for n in norms_per_level)) if norms_per_level else 1.0
    )

    for k in range(K):
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))
        ax.hist(norms_per_level[k], bins=30, color="tab:blue", alpha=0.75)
        ax.axvline(
            float(np.mean(norms_per_level[k])),
            color="red",
            linestyle="--",
            linewidth=1.2,
            label="mean",
        )
        ax.set_xlabel(r"$\|C_k\|_2$")
        ax.set_ylabel("count")
        ax.set_xlim(0, global_max * 1.05)
        ax.legend(loc="upper right")
        ax.set_title(
            f"{experiment_name} \u2014 concept-norm distribution "
            f"(level {k}, L_k={pyramid.level_lengths[k]})"
        )
        fig.tight_layout()
        _save_figure(fig, output_dir, f"concept_norms_level{k}")


def _plot_residual_decomposition(
    pyramid: SimpleNamespace,
    output_dir: Path,
    experiment_name: str,
) -> None:
    """Line plot of masked mean norms vs level index.

    Shows three interrelated curves:
        * ``||H_proj||`` — flat reference (level 0 input)
        * ``||R_k||`` — per-level reconstruction contribution
        * ``||H_hat_k||`` — cumulative reconstruction
        * ``||H_rest_k||`` — residual after level k (should approach 0)

    Norms are averaged over valid token positions only, guarded by
    ``pyramid.attention_mask`` — padded positions would inflate every
    curve with zeros otherwise.
    """
    K = pyramid.num_levels
    mask = pyramid.attention_mask  # [B, L] or None
    if mask is None:
        mask_f = torch.ones_like(pyramid.projected_hidden[..., 0])
    else:
        mask_f = mask.to(pyramid.projected_hidden.dtype)

    def _masked_mean_norm(t: torch.Tensor) -> float:
        # ||t[b, l]|| averaged over valid (b, l).
        n = torch.linalg.vector_norm(t, dim=-1)  # [B, L]
        return float((n * mask_f).sum() / mask_f.sum().clamp_min(1.0))

    H_proj_n = _masked_mean_norm(pyramid.projected_hidden)

    R_norms: List[float] = [
        _masked_mean_norm(pyramid.level_outputs[k].reconstruction) for k in range(K)
    ]

    # Recompute cumulative + residual from R_k so this plot is self-consistent
    # (matches the builder's internal accumulator step for step).
    H_hat_cumul = torch.zeros_like(pyramid.projected_hidden)
    hat_norms: List[float] = []
    rest_norms: List[float] = []
    for k in range(K):
        H_hat_cumul = H_hat_cumul + pyramid.level_outputs[k].reconstruction
        hat_norms.append(_masked_mean_norm(H_hat_cumul))
        rest_norms.append(_masked_mean_norm(pyramid.projected_hidden - H_hat_cumul))

    xs = np.arange(K)
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.0))
    ax.axhline(
        H_proj_n,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label=r"$\|H_{\mathrm{proj}}\|$ (input)",
    )
    ax.plot(xs, R_norms, marker="o", linewidth=2.0, label=r"$\|R_k\|$ (per-level)")
    ax.plot(
        xs,
        hat_norms,
        marker="s",
        linewidth=2.0,
        label=r"$\|H_{\hat{}_k}\|$ (cumulative)",
    )
    ax.plot(
        xs,
        rest_norms,
        marker="^",
        linewidth=2.0,
        label=r"$\|H_{\mathrm{rest}_k}\|$ (residual)",
    )
    ax.set_xlabel("pyramid level $k$")
    ax.set_ylabel("masked mean norm")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{k}\nL={pyramid.level_lengths[k]}" for k in xs])
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best")
    ax.set_title(f"{experiment_name} — residual decomposition")
    fig.tight_layout()
    _save_figure(fig, output_dir, "residual_decomposition")


def _plot_intra_level_similarity(
    pyramid: SimpleNamespace,
    output_dir: Path,
    experiment_name: str,
) -> None:
    """Slot-to-slot cosine-similarity heatmap — ONE file per level.

    Writes ``intra_level_similarity_level{k}.{png,pdf}``. For level k the
    matrix has shape ``[L_k, L_k]`` (averaged over batch). A diffuse
    matrix means healthy orthogonal slots; a near-uniform matrix signals
    slot collapse.
    """
    K = pyramid.num_levels
    for k in range(K):
        C_k = pyramid.concepts[k]  # [B, L_k, D]
        eps = 1e-8
        C_unit = C_k / (torch.linalg.vector_norm(C_k, dim=-1, keepdim=True) + eps)
        sim = torch.bmm(C_unit, C_unit.transpose(1, 2))  # [B, L_k, L_k]
        mat = sim.mean(dim=0).detach().cpu().float().numpy()

        L_k = pyramid.level_lengths[k]
        fig_size = max(4.0, 0.25 * L_k + 3.5)
        fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
        im = ax.imshow(
            mat,
            aspect="auto",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax.set_xlabel("slot j")
        ax.set_ylabel("slot i")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        ax.set_title(
            f"{experiment_name} \u2014 intra-level cosine similarity "
            f"(level {k}, L_k={L_k})"
        )
        fig.tight_layout()
        _save_figure(fig, output_dir, f"intra_level_similarity_level{k}")


def _plot_inter_level_similarity(
    pyramid: SimpleNamespace,
    output_dir: Path,
    experiment_name: str,
) -> None:
    """K×K mean cosine similarity across level-averaged concept vectors.

    Procedure: for each sample average C_k over slots to get one vector
    per level ``[B, D]``, unit-normalise, take the per-sample KxK Gram
    matrix, then average across the batch. High off-diagonal entries
    mean levels are redundant.
    """
    K = pyramid.num_levels
    level_vecs = []
    for k in range(K):
        # Slot-mean per sample: [B, D].
        level_vecs.append(pyramid.concepts[k].mean(dim=1))
    stacked = torch.stack(level_vecs, dim=1)  # [B, K, D]
    eps = 1e-8
    stacked_unit = stacked / (
        torch.linalg.vector_norm(stacked, dim=-1, keepdim=True) + eps
    )
    sim = torch.bmm(stacked_unit, stacked_unit.transpose(1, 2))  # [B, K, K]
    mat = sim.mean(dim=0).detach().cpu().float().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.0))
    im = ax.imshow(
        mat,
        aspect="auto",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(K))
    ax.set_yticks(np.arange(K))
    ax.set_xticklabels([f"k={i}" for i in range(K)])
    ax.set_yticklabels([f"k={i}" for i in range(K)])
    for i in range(K):
        for j in range(K):
            ax.text(
                j,
                i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if abs(mat[i, j]) > 0.5 else "black",
                fontsize=10,
            )
    ax.set_title(f"{experiment_name} — inter-level mean cosine similarity")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    _save_figure(fig, output_dir, "inter_level_similarity")


def _plot_attention_entropy(
    pyramid: SimpleNamespace,
    output_dir: Path,
    experiment_name: str,
) -> None:
    """Per-level box plots of H(A_{k,j}) with log(L_valid) reference.

    Sharp attention (single peak) has low entropy; diffuse attention
    approaches ``log(L_valid)``. Plotting the median reference line as a
    dashed horizontal guides interpretation.
    """
    K = pyramid.num_levels
    valid_lens = _per_row_valid_length(pyramid)  # [B]
    ref_entropy = float(np.mean(np.log(np.clip(valid_lens, 1, None))))

    entropies_per_level: List[np.ndarray] = []
    for k in range(K):
        A_k = pyramid.level_outputs[k].attention_weights.detach().cpu().float()
        # H = -Σ p log p along token dim, per (b, j). Tiny epsilon avoids
        # ``log(0)`` on the many zeros that a sharply peaked softmax emits.
        eps = 1e-12
        H = -(A_k * (A_k.clamp_min(eps).log())).sum(dim=-1)  # [B, L_k]
        entropies_per_level.append(H.numpy().reshape(-1))

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.8))
    ax.boxplot(
        entropies_per_level,
        labels=[f"k={k}" for k in range(K)],
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="tab:cyan", alpha=0.6),
    )
    ax.axhline(
        ref_entropy,
        color="red",
        linestyle="--",
        linewidth=1.3,
        label=rf"$\log L_{{\mathrm{{valid}}}}\approx{ref_entropy:.2f}$ (uniform)",
    )
    ax.axhline(0.0, color="grey", linestyle=":", linewidth=1.0, label="0 (single-peak)")
    ax.set_ylabel(r"attention entropy $H(A_{k,j})$ [nats]")
    ax.set_xlabel("pyramid level")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left")
    ax.set_title(f"{experiment_name} — per-level attention entropy")
    fig.tight_layout()
    _save_figure(fig, output_dir, "attention_entropy")


def _plot_token_coverage(
    pyramid: SimpleNamespace,
    output_dir: Path,
    experiment_name: str,
) -> None:
    """Token-level attention-received curves, one line per pyramid level.

    Each row's A_k is slot-summed (Σ_j A_k[:, j, :]) to yield a token
    importance vector. We normalise token positions to ``[0, 1]`` by
    each row's valid length so curves from different-length rows can be
    averaged coherently. Resampling onto a fixed grid (100 bins) makes
    this trivial with linear interpolation.
    """
    K = pyramid.num_levels
    B = pyramid.projected_hidden.shape[0]
    valid_lens = _per_row_valid_length(pyramid)
    n_bins = 100
    grid = np.linspace(0.0, 1.0, n_bins)

    curves_per_level: List[np.ndarray] = []
    for k in range(K):
        A_k = pyramid.level_outputs[k].attention_weights.detach().cpu().float().numpy()
        # Slot-summed importance per (b, l).
        token_imp = A_k.sum(axis=1)  # [B, L_full]

        resampled = np.zeros((B, n_bins), dtype=np.float64)
        for b in range(B):
            L_b = int(valid_lens[b])
            if L_b <= 1:
                continue
            xs_src = np.linspace(0.0, 1.0, L_b)
            resampled[b] = np.interp(grid, xs_src, token_imp[b, :L_b])
        curves_per_level.append(resampled.mean(axis=0))

    fig, ax = plt.subplots(1, 1, figsize=(9.0, 5.0))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, K))
    for k in range(K):
        ax.plot(
            grid,
            curves_per_level[k],
            color=colors[k],
            linewidth=2.0,
            label=f"level {k} (L_k={pyramid.level_lengths[k]})",
        )
    ax.set_xlabel("normalised CoT token position")
    ax.set_ylabel("mean attention received")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best")
    ax.set_title(f"{experiment_name} — per-level token coverage")
    fig.tight_layout()
    _save_figure(fig, output_dir, "token_coverage")


def _plot_concept_pca(
    pyramid: SimpleNamespace,
    output_dir: Path,
    experiment_name: str,
) -> None:
    """2-D PCA scatter of all concepts coloured by level.

    Uses ``torch.svd_lowrank`` to avoid an sklearn dependency. We mean-
    centre the stacked matrix and keep the top 2 right singular vectors.
    """
    K = pyramid.num_levels
    # Stack concepts into one matrix with a matching level label array.
    rows: List[torch.Tensor] = []
    labels: List[np.ndarray] = []
    for k in range(K):
        C_k = pyramid.concepts[k].detach().cpu().float()  # [B, L_k, D]
        B_k, Lk, _ = C_k.shape
        rows.append(C_k.reshape(B_k * Lk, -1))
        labels.append(np.full((B_k * Lk,), k, dtype=np.int64))
    X = torch.cat(rows, dim=0)  # [N, D]
    y = np.concatenate(labels, axis=0)  # [N]

    # Mean-centre before SVD for a PCA-equivalent result.
    X_centred = X - X.mean(dim=0, keepdim=True)
    # ``q=3`` gives a small buffer over the 2 components we actually use.
    _, _, V = torch.svd_lowrank(X_centred, q=3)
    pcs = (X_centred @ V[:, :2]).numpy()  # [N, 2]

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6.0))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, K))
    for k in range(K):
        sel = y == k
        ax.scatter(
            pcs[sel, 0],
            pcs[sel, 1],
            s=22,
            alpha=0.55,
            color=colors[k],
            label=f"level {k} (L_k={pyramid.level_lengths[k]})",
            edgecolors="none",
        )
    ax.axhline(0.0, color="grey", linestyle=":", linewidth=0.7)
    ax.axvline(0.0, color="grey", linestyle=":", linewidth=0.7)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="best")
    ax.set_title(f"{experiment_name} — concept PCA (2-D)")
    fig.tight_layout()
    _save_figure(fig, output_dir, "concept_pca")


def _plot_concept_pca_per_level(
    pyramid: SimpleNamespace,
    output_dir: Path,
    experiment_name: str,
) -> None:
    """Per-level 2-D PCA of concepts — ONE standalone scatter per level.

    Writes ``concept_pca_level{k}.{png,pdf}``. Unlike ``concept_pca``
    (all levels in one shared axis), here each level's concepts are
    mean-centred and projected onto their OWN top-2 principal axes via
    ``torch.svd_lowrank`` — so the scatter reflects the intra-level
    structure (spread, clusters) directly without cross-level noise.
    Each point carries its slot index ``j`` as a text annotation so the
    reader can correlate geometry with the other per-slot analyses.
    """
    K = pyramid.num_levels
    for k in range(K):
        C_k = pyramid.concepts[k].detach().cpu().float()  # [B, L_k, D]
        B_k, L_k, _ = C_k.shape
        X = C_k.reshape(B_k * L_k, -1)
        if X.shape[0] < 2:
            # PCA on a single point is meaningless; skip the figure.
            continue
        X_centred = X - X.mean(dim=0, keepdim=True)
        q = min(3, X_centred.shape[0], X_centred.shape[1])
        _, _, V = torch.svd_lowrank(X_centred, q=q)
        pcs = (X_centred @ V[:, :2]).numpy()  # [N, 2]

        # Slot labels repeat every L_k rows across the batch dimension.
        slots = np.tile(np.arange(L_k), B_k)

        fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
        sc = ax.scatter(
            pcs[:, 0],
            pcs[:, 1],
            c=slots,
            cmap="viridis",
            s=36,
            alpha=0.85,
            edgecolors="none",
        )
        # Annotate first batch's slots (batch_size is 1 at eval time, so
        # this labels every point for the typical use case).
        for n in range(min(L_k, pcs.shape[0])):
            ax.annotate(
                f"{slots[n]}",
                (pcs[n, 0], pcs[n, 1]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=9,
                color="black",
            )
        ax.axhline(0.0, color="grey", linestyle=":", linewidth=0.7)
        ax.axvline(0.0, color="grey", linestyle=":", linewidth=0.7)
        ax.set_xlabel("PC 1 (level-local)")
        ax.set_ylabel("PC 2 (level-local)")
        ax.grid(True, linestyle=":", alpha=0.4)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("slot j")
        ax.set_title(f"{experiment_name} — concept PCA (level {k}, L_k={L_k})")
        fig.tight_layout()
        _save_figure(fig, output_dir, f"concept_pca_level{k}")


# =============================================================================
# Orchestration
# =============================================================================


def _derive_experiment_name(config_path: Path) -> str:
    """Strip ``train_builder_`` prefix + ``.yml`` suffix for titles/filenames."""
    stem = config_path.stem
    prefix = "train_builder_"
    return stem[len(prefix) :] if stem.startswith(prefix) else stem


def _safe_id(main_id: Any) -> str:
    """Sanitise an arbitrary main_id to a filesystem-friendly folder name."""
    s = str(main_id).strip().replace("/", "_").replace("\\", "_")
    keep = (c if (c.isalnum() or c in "-_.") else "_" for c in s)
    safe = "".join(keep)
    return safe or "unnamed"


def _resolve_eval_root(config: Dict[str, Any], mode: str) -> Path:
    """Return ``<log_path>/eval_builder/<mode>/`` (after ``apply_storage_root``).

    This is the directory eval_builder writes its per-sample dumps into,
    and the directory this script reads from AND writes figures into.
    """
    log_dir = Path(config["log"]["log_path"]).expanduser()
    return log_dir / EVAL_BUILDER_DIR_NAME / mode


def _existing_outputs(eval_root: Path, analyses: Sequence[str]) -> bool:
    """Return True iff every ``sample_*/pyramid.pt`` already has every figure.

    Layout: ``<eval_root>/sample_*/<file_stem>.png``. Both PNG and PDF
    are written together so checking the PNG is sufficient. Sample
    folders missing ``pyramid.pt`` (e.g. eval_builder ran with ``-v 0``)
    are skipped — they'd never get figures anyway.
    """
    if not eval_root.is_dir():
        return False
    sample_dirs = sorted(
        p for p in eval_root.iterdir() if p.is_dir() and p.name.startswith("sample_")
    )
    if not sample_dirs:
        return False
    needed_stems = [
        _ANALYSIS_SENTINEL_STEMS[k] for k in analyses if k in _ANALYSIS_SENTINEL_STEMS
    ]
    any_with_dump = False
    for sd in sample_dirs:
        if not (sd / PYRAMID_DUMP_NAME).is_file():
            continue
        any_with_dump = True
        for stem in needed_stems:
            if not (sd / f"{stem}.png").is_file():
                return False
    return any_with_dump


def _run_concept_analysis(
    config_path: Path,
    *,
    storage_root: str,
    mode: str,
    analyses: Sequence[str],
) -> int:
    """Iterate every ``sample_*/pyramid.pt`` under eval_root and render figures.

    Figures land INSIDE each sample folder (next to ``input.json`` /
    ``pyramid.pt`` / ``reasoning.json``), so the layout is flat and
    self-describing: open a sample folder → everything about that sample.

    The HF tokenizer is lazy-instantiated on first use (zero cost when
    there are no samples to plot) so that repeatedly invoking this on
    skip_no_data configs does not trigger tokenizer downloads.

    Returns the number of sample folders that received figures.
    """
    config = load_config(str(config_path))
    apply_storage_root(config, storage_root)
    _apply_plot_style()

    experiment_name = _derive_experiment_name(config_path)
    eval_root = _resolve_eval_root(config, mode)

    sample_dirs = sorted(
        p for p in eval_root.iterdir() if p.is_dir() and p.name.startswith("sample_")
    )

    tokenizer = None
    max_seq_len = int(config["model"]["pyramid"]["max_seq_len"])
    analyses_set = set(analyses)
    n_rendered = 0

    for sample_dir in sample_dirs:
        pt_path = sample_dir / PYRAMID_DUMP_NAME
        if not pt_path.is_file():
            logger.info("  skip (no pyramid.pt): %s", sample_dir.name)
            continue
        try:
            pyramid = _load_pyramid_from_disk(pt_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "  load pyramid.pt failed (%s): %s", type(exc).__name__, pt_path
            )
            continue

        # Retokenise CoT from input.json so attention heatmaps / overlays
        # carry meaningful column labels; fall back to integer indices if
        # input.json is absent (should be rare — eval_builder always writes
        # it unless artifacts.input was explicitly disabled).
        input_json = sample_dir / INPUT_JSON_NAME
        if input_json.is_file():
            if tokenizer is None:
                tokenizer = _load_tokenizer_from_config(config)
            sample_input = _load_sample_input(input_json)
            cot_text = sample_input.get("cot_answer", "") or ""
            tokens_per_row = [_retokenize_cot(tokenizer, cot_text, max_seq_len)]
        else:
            L = int(pyramid.projected_hidden.shape[1])
            tokens_per_row = [[str(i) for i in range(L)]]

        # Dispatch every requested analysis into THIS sample's folder.
        # Multi-level analyses write per-level (or per-slot) files named
        # with a suffix; single-axis analyses write a single file. See
        # ``_ANALYSIS_SENTINEL_STEMS`` for the existence-probe stems.
        if "attention_heatmaps" in analyses_set:
            _plot_attention_heatmaps(
                pyramid,
                tokens_per_row,
                sample_dir,
                experiment_name,
                file_stem="attention_heatmap",
            )
        if "attention_text_overlay" in analyses_set:
            _plot_attention_text_overlay(
                pyramid,
                tokens_per_row,
                sample_dir,
                experiment_name,
                file_stem="attention_text_overlay",
            )
        if "attention_text_per_concept" in analyses_set:
            _plot_attention_text_per_concept(
                pyramid,
                tokens_per_row,
                sample_dir,
                experiment_name,
                file_stem="attention_text_per_concept",
            )
        if "concept_norms" in analyses_set:
            _plot_concept_norms(pyramid, sample_dir, experiment_name)
        if "residual_decomposition" in analyses_set:
            _plot_residual_decomposition(pyramid, sample_dir, experiment_name)
        if "intra_level_similarity" in analyses_set:
            _plot_intra_level_similarity(pyramid, sample_dir, experiment_name)
        if "inter_level_similarity" in analyses_set:
            _plot_inter_level_similarity(pyramid, sample_dir, experiment_name)
        if "attention_entropy" in analyses_set:
            _plot_attention_entropy(pyramid, sample_dir, experiment_name)
        if "token_coverage" in analyses_set:
            _plot_token_coverage(pyramid, sample_dir, experiment_name)
        if "concept_pca" in analyses_set:
            _plot_concept_pca(pyramid, sample_dir, experiment_name)
        if "concept_pca_per_level" in analyses_set:
            _plot_concept_pca_per_level(pyramid, sample_dir, experiment_name)

        n_rendered += 1
        logger.info(
            "  %s: wrote %d analyses -> %s",
            sample_dir.name,
            len(analyses_set),
            sample_dir,
        )

    print(f"Rendered figures into {n_rendered} sample folder(s) under {eval_root}")
    return n_rendered


def analyze_one(
    config_path: Path,
    *,
    overlap: bool,
    storage_root: str,
    mode: str,
    analyses: Sequence[str],
) -> Tuple[str, str]:
    """Per-config status tracker: resolve eval_root, skip or render.

    A config is ``skip_no_data`` when the eval_builder dump directory
    doesn't exist or has no ``sample_*/pyramid.pt`` files (you need to
    run ``eval_builder.py ... -v 1 --mode <mode>`` first). It is
    ``skip_exists`` when every expected figure is already on disk and
    ``--no-overlap`` was requested.
    """
    try:
        config = load_config(str(config_path))
    except Exception as exc:  # noqa: BLE001
        return "error", f"load_config failed: {exc}"

    apply_storage_root(config, storage_root)
    print_storage_paths(config, storage_root)

    eval_root = _resolve_eval_root(config, mode)
    if not eval_root.is_dir():
        return "skip_no_data", f"no eval dump at {eval_root}"

    sample_dumps = [
        p
        for p in eval_root.iterdir()
        if p.is_dir()
        and p.name.startswith("sample_")
        and (p / PYRAMID_DUMP_NAME).is_file()
    ]
    if not sample_dumps:
        return "skip_no_data", f"no sample_*/pyramid.pt under {eval_root}"

    if not overlap and _existing_outputs(eval_root, analyses):
        return "skip_exists", f"figures already present under {eval_root}"

    try:
        n = _run_concept_analysis(
            config_path,
            storage_root=storage_root,
            mode=mode,
            analyses=analyses,
        )
    except Exception as exc:  # noqa: BLE001
        plt.close("all")
        return "error", f"{type(exc).__name__}: {exc}"

    return "analyzed", f"rendered figures in {n} sample folder(s) under {eval_root}"


def _print_summary(rows: List[Tuple[str, str, str]]) -> None:
    """Compact status table — identical layout to builder_training_analysis."""
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
    counts: Dict[str, int] = {}
    for status, _name, _detail in rows:
        counts[status] = counts.get(status, 0) + 1
    parts = [f"{k}={v}" for k, v in sorted(counts.items())]
    print("Totals: " + "  ".join(parts))


def _resolve_configs_from_args(args: argparse.Namespace) -> List[Path]:
    """Resolve --config (preferred) or --dataset/--experiment to a path list."""
    if args.config is not None:
        if args.dataset is not None or args.experiment is not None:
            raise SystemExit("[ERROR] -c/--config is mutually exclusive with -d/-e.")
        p = Path(args.config).expanduser().resolve()
        if not p.is_file():
            print(f"[ERROR] Config file not found: {p}")
            return []
        return [p]
    if args.dataset is None or args.experiment is None:
        raise SystemExit(
            "[ERROR] Provide either -c/--config OR both -d/--dataset and "
            "-e/--experiment."
        )
    return discover_configs(args.dataset, args.experiment)


def main() -> int:
    """CLI entry — iterate matching configs and render concept-analysis figures.

    The script consumes ``eval_builder.py`` dumps, so it expects the
    eval_builder output tree to exist. Configs without a dump are marked
    ``skip_no_data`` rather than erroring — useful when batch-processing
    a dataset where only a subset has been evaluated.
    """
    args = parse_args()
    overlap: bool = args.overlap
    storage_root: str = args.storage_root
    mode: str = args.mode
    analyses = _parse_analyses(args.analyses)

    configs = _resolve_configs_from_args(args)
    if not configs:
        return 1

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    source_desc = (
        f"config={args.config}"
        if args.config is not None
        else f"dataset={args.dataset}  experiment={args.experiment}"
    )
    print(
        f"[ANALYZE] concept pyramid  {source_desc}  mode={mode}  "
        f"analyses={','.join(analyses)}  overlap={overlap}  "
        f"({len(configs)} config file(s))"
    )
    for p in configs:
        rel = p.relative_to(PROJECT_ROOT) if p.is_relative_to(PROJECT_ROOT) else p
        print(f"  - {rel}")
    print()

    rows: List[Tuple[str, str, str]] = []
    for cfg_path in configs:
        print("=" * 70)
        print(f"[CONFIG] {cfg_path.name}")
        print("=" * 70)
        status, detail = analyze_one(
            cfg_path,
            overlap=overlap,
            storage_root=storage_root,
            mode=mode,
            analyses=analyses,
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
    return 0 if all(r[0] != "error" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
