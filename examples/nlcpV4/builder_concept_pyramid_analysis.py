"""Inspect the builder's concept pyramid: encodings and attention.

This tool loads a trained ``ConceptPyramidBuilder`` checkpoint and,
for EACH sample in a deterministic slice of ``evaluation.data``, runs
one forward pass (``batch_size=1``) and emits the complete set of
per-sample diagnostic figures into that sample's OWN folder.

Storage layout
==============
Outputs land next to the model, under
``<checkpoint_dir>.parent / concept_pyramid_analysis/``. Concretely:

    concept_pyramid_analysis/
    ├── sample_<main_id_1>/
    │   ├── input.txt                       # raw Q + CoT (and solution)
    │   ├── attention_heatmap.{png,pdf}
    │   ├── attention_text_overlay.{png,pdf}
    │   ├── concept_norms.{png,pdf}
    │   ├── residual_decomposition.{png,pdf}
    │   ├── intra_level_similarity.{png,pdf}
    │   ├── inter_level_similarity.{png,pdf}
    │   ├── attention_entropy.{png,pdf}
    │   ├── token_coverage.{png,pdf}
    │   └── concept_pca.{png,pdf}
    ├── sample_<main_id_2>/
    │   └── ...

One folder per sample keeps the raw source alongside every derived
figure — you can open ``input.txt`` in any viewer and cross-reference
its contents against the attention-on-text overlay without hunting
across directories.

WHAT IS ANALYZED
================
The builder's :class:`PyramidOutput` carries two first-class tensors
per level:

    * ``C_k`` (concept encodings)   : [B, L_k, D]
    * ``A_k`` (soft attention map)  : [B, L_k, L]

This script inspects both — concept geometry *and* how each concept slot
draws attention over CoT tokens. No training happens; the builder runs in
``eval()`` under ``torch.no_grad()``.

AVAILABLE ANALYSES (select via ``--analyses``)
==============================================
    ``attention_heatmaps``   — per-sample attention heatmap (K subplots).
                               Shows which CoT tokens every concept slot
                               attends to at each pyramid level.
    ``attention_text_overlay``— per-sample "attention-on-text" overlay.
                               The CoT is rendered as wrapped rows of
                               token boxes; each box's background colour
                               encodes the per-level aggregated attention
                               weight (deeper colour = larger weight).
                               The most intuitive diagnostic — you see
                               exactly WHICH substrings each level reads.
    ``concept_norms``         — per-level histograms of ``||C_k[j]||``.
    ``residual_decomposition``— line plot of ``||H_rest_k||``, ``||R_k||``,
                               ``||H_hat_k||`` vs level.
    ``intra_level_similarity``— K cosine-sim heatmaps, one per level.
    ``inter_level_similarity``— single K×K cosine-sim matrix over
                               slot-averaged per-level concepts.
    ``attention_entropy``     — per-level box plots of ``H(A_{k,j})``.
    ``token_coverage``        — per-level curves of column-summed attention
                               over normalised token position.
    ``concept_pca``           — 2-D PCA scatter of all concepts
                               coloured by level.
    ``all`` (default)         — run every analysis above.

Usage
-----
Direct config path (sample 4 deterministic rows from the test split)::

    python3 examples/nlcpV4/builder_concept_pyramid_analysis.py \\
        -c configs/nlcpV4/GSM8K/train_builder_Qwen2.5-0.5B_6level.yml

Override the checkpoint file (use evaluation.data from the config
but load this exact ``.pt`` directly — output goes next to it)::

    python3 examples/nlcpV4/builder_concept_pyramid_analysis.py \\
        -c configs/nlcpV4/GSM8K/train_builder_Qwen2.5-0.5B_6level.yml \\
        -p /Data/proj/logs/...builder.../checkpoints/checkpoint_best.pt

Discovery by dataset + experiment::

    python3 examples/nlcpV4/builder_concept_pyramid_analysis.py \\
        -d GSM8K -e Qwen2.5-0.5B_6level

Select a subset of analyses::

    python3 examples/nlcpV4/builder_concept_pyramid_analysis.py \\
        -c configs/nlcpV4/GSM8K/train_builder_Qwen2.5-0.5B_6level.yml \\
        --analyses attention_heatmaps,attention_text_overlay,concept_norms \\
        --num-samples 2

Arguments
---------
    -s / --storage-root     Prefix prepended to relative log paths.
                            MUST match the value used at training time.
    -c / --config           Direct path to a single ``train_builder_*.yml``.
                            Mutually exclusive with ``-d``/``-e``.
    -d / --dataset          Dataset directory under ``configs/nlcpV4/``.
                            Required when ``-c`` is not given.
    -e / --experiment       Config stem after ``train_builder_`` or ``all``.
                            Required when ``-c`` is not given.
    -p / --checkpoint       Override the checkpoint file. When set, the
                            config's ``evaluation`` section still drives
                            data + model structure, but THIS exact ``.pt``
                            is loaded, and outputs go to
                            ``<parent-of-p>.parent / concept_pyramid_analysis/``.
    -o / --overlap          If true (default), overwrite existing figures.
                            ``--no-overlap`` skips samples already written.
    --split                 Data split: ``test`` (default) or ``train``.
    --num-samples           How many samples to analyze (each gets its
                            own folder). Default 4. The forward pass
                            always runs with ``batch_size=1``.
    --analyses              Comma-separated subset of analysis keys or
                            ``all`` (default).
    --seed                  RNG seed for deterministic batch sampling.
"""

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

from lmbase.utils.env_tools import get_device
from nlcpV4.concept_builder import ConceptPyramidBuilder, PyramidOutput
from nlcpV4.data_loader import BuilderInput, NLCPV4DataLoader
from ram.utils import apply_storage_root, load_config, print_storage_paths

logger = logging.getLogger(__name__)

# --- Batch-mode constants ------------------------------------------
CONFIGS_ROOT = PROJECT_ROOT / "configs" / "nlcpV4"
ALL_KEYWORD = "all"
# Outputs land under <experiment>/concept_pyramid_analysis/ (sibling of logs/
# and checkpoints/). The name matches the script stem for discoverability.
ANALYSIS_OUTPUT_DIR_NAME = "concept_pyramid_analysis"
# Canonical analysis keys — must match the dispatch table in _run_concept_analysis.
# Every key produces a single file per sample named ``<key>.{png,pdf}`` that
# lives inside the sample's own folder, so the output layout is flat and
# self-describing: open a sample folder → see everything about that sample.
ALL_ANALYSES = (
    "attention_heatmaps",
    "attention_text_overlay",
    "concept_norms",
    "residual_decomposition",
    "intra_level_similarity",
    "inter_level_similarity",
    "attention_entropy",
    "token_coverage",
    "concept_pca",
)
# Per-analysis filename stem written into each sample folder. We intentionally
# drop the plural-heatmaps and overlay suffixes so the filenames read cleanly
# inside ``sample_<id>/``.
_ANALYSIS_FILE_STEMS: Dict[str, str] = {
    "attention_heatmaps": "attention_heatmap",
    "attention_text_overlay": "attention_text_overlay",
    "concept_norms": "concept_norms",
    "residual_decomposition": "residual_decomposition",
    "intra_level_similarity": "intra_level_similarity",
    "inter_level_similarity": "inter_level_similarity",
    "attention_entropy": "attention_entropy",
    "token_coverage": "token_coverage",
    "concept_pca": "concept_pca",
}


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments. Mirrors ``builder_training_analysis``."""
    parser = argparse.ArgumentParser(
        description="Analyze concept encodings + attention from a trained builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-s",
        "--storage-root",
        type=str,
        default="./",
        help=(
            "Prefix for relative config.log paths. MUST match the value "
            "used at training time so this script reads the same "
            "checkpoint / eval data. Default './'."
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
            "discovery is bypassed and this config drives model, "
            "checkpoint and evaluation-data retrieval."
        ),
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default=None,
        help=(
            "Dataset dir under configs/nlcpV4/ (may be nested). "
            "Required when -c is not given."
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
        "-p",
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Override the checkpoint .pt file. When set, this exact file "
            "is loaded (the config's evaluation section still drives data "
            "loading & model structure). Outputs go to the parent of -p's "
            "parent dir / concept_pyramid_analysis/."
        ),
    )
    parser.add_argument(
        "-o",
        "--overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true (default), overwrite existing figures. --no-overlap "
            "skips configs whose outputs already exist."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="test",
        help="Data split for the analysis batch. Default 'test'.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help=(
            "How many samples to analyze (each gets its own folder). "
            "Default 4. The forward pass always runs with batch_size=1, "
            "ignoring any value set in the config."
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for deterministic batch sampling. Default 42.",
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
# Builder + data loading
# =============================================================================


def _locate_best_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Return the preferred ``.pt`` file, or None when absent."""
    for name in ("checkpoint_best_eval.pt", "checkpoint_best.pt"):
        p = checkpoint_dir / name
        if p.is_file():
            return p
    return None


def load_builder(
    config: Dict[str, Any],
    *,
    checkpoint_override: Optional[Path] = None,
) -> Tuple[ConceptPyramidBuilder, torch.device, Path]:
    """Instantiate a builder and load the best (or overridden) checkpoint.

    When ``checkpoint_override`` is provided we load THAT exact ``.pt``
    file (the user passed ``-p``); otherwise we discover the best file
    under ``config['log']['checkpoint_path']``.

    Raises ``FileNotFoundError`` when no checkpoint can be located, so
    the caller can mark the config as ``skip_no_data`` cleanly.
    """
    if checkpoint_override is not None:
        ckpt_path = checkpoint_override
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    else:
        checkpoint_dir = Path(config["log"]["checkpoint_path"]).expanduser()
        ckpt_path = _locate_best_checkpoint(checkpoint_dir)
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoint_best*.pt under {checkpoint_dir}")

    device = torch.device(str(get_device("auto")))
    builder = ConceptPyramidBuilder(config)
    builder.to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    builder.load_state_dict(ckpt["model_state_dict"])
    builder.eval()
    logger.info(
        "Loaded checkpoint: %s (step=%s, loss=%.4f)",
        ckpt_path,
        ckpt.get("step", "?"),
        ckpt.get("eval_loss", ckpt.get("loss", 0.0)),
    )
    return builder, device, ckpt_path


def collect_batch(
    config: Dict[str, Any],
    *,
    split: str,
    num_samples: int,
    batch_size: int,
    seed: int,
) -> BuilderInput:
    """Gather ``num_samples`` rows from ``evaluation.data`` deterministically.

    We hijack the evaluation data config and override its ``split`` so the
    user can flip between ``test`` and ``train`` from the CLI without
    editing YAML. ``include_solution`` is set to ``False`` — this bypasses
    the builder's reasoning branch and keeps the forward pass cheap.
    """
    data_cfg = dict(config["evaluation"]["data"])
    data_cfg["split"] = split

    env_cfg = config["environment"]
    loader = NLCPV4DataLoader(
        data_cfg=data_cfg,
        batch_size=batch_size,
        include_solution=False,
        shuffle=False,
        drop_last=False,
        num_workers=env_cfg["dataloader_num_workers"],
    )

    # Seed so any shuffle-like behaviour downstream is repeatable.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    questions: List[str] = []
    cot_answers: List[str] = []
    main_ids: List[str] = []
    for batch in loader:
        take = min(num_samples - len(questions), batch.batch_size)
        questions.extend(batch.questions[:take])
        cot_answers.extend(batch.cot_answers[:take])
        main_ids.extend(batch.main_ids[:take])
        if len(questions) >= num_samples:
            break

    if not questions:
        raise RuntimeError(
            f"Empty batch from split='{split}' dataset_size={loader.dataset_size}"
        )
    return BuilderInput(
        questions=questions,
        cot_answers=cot_answers,
        solutions=[],
        main_ids=main_ids,
    )


@torch.no_grad()
def run_forward(
    builder: ConceptPyramidBuilder, batch: BuilderInput
) -> Tuple[PyramidOutput, List[List[str]]]:
    """Run the builder and also return per-row decoded token strings.

    The builder's ``forward`` tokenizes CoT internally but does not expose
    the token ids. To label attention-heatmap x-axes we retokenize the
    same inputs with the same tokenizer and max length; the token order
    is stable across calls so the labels align with ``A_k`` columns.
    """
    pyramid = builder(batch)

    max_length = builder.pyramid_cfg["max_seq_len"]
    tokenized = builder.tokenizer(
        batch.cot_answers,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    ids = tokenized["input_ids"]
    mask = tokenized["attention_mask"]

    tokens_per_row: List[List[str]] = []
    for row_ids, row_mask in zip(ids, mask):
        valid = row_ids[row_mask.bool()]
        # One decode-per-id keeps tokens aligned with columns of A_k (the
        # builder runs attention over the full padded length — here we
        # report only the non-pad prefix).
        tokens_per_row.append(
            [builder.tokenizer.decode([int(t)]) for t in valid.tolist()]
        )
    return pyramid, tokens_per_row


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


def _per_row_valid_length(pyramid: PyramidOutput) -> np.ndarray:
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
    pyramid: PyramidOutput,
    tokens_per_row: Sequence[Sequence[str]],
    output_dir: Path,
    experiment_name: str,
    *,
    file_stem: str = "attention_heatmap",
) -> None:
    """Single-sample attention heatmap with K subplots.

    Renders ``A_k[0]`` for each level (the input pyramid is expected to
    carry ``B == 1``) with shape ``[L_k, valid_len]``. x-axis shows at
    most 20 decoded token strings; y-axis shows the concept-slot index
    ``j ∈ [0, L_k)``. Saved as ``<file_stem>.{png,pdf}`` in ``output_dir``.
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
    fig, axes = plt.subplots(
        K,
        1,
        figsize=(16, 1.6 * K + 1.2),
        squeeze=False,
    )
    axes = axes.flatten()

    for k, ax in enumerate(axes):
        A_k = pyramid.level_outputs[k].attention_weights[i, :, :L_i]
        mat = A_k.detach().cpu().float().numpy()
        im = ax.imshow(
            mat,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            vmin=0.0,
        )
        ax.set_ylabel(f"level {k}\n(L_k={level_lengths[k]})")
        ax.set_yticks(np.arange(level_lengths[k]))
        ax.set_yticklabels([str(j) for j in range(level_lengths[k])])
        xticks = _tick_positions(L_i, max_ticks=20)
        ax.set_xticks(xticks)
        if k == K - 1:
            ax.set_xticklabels(
                [_short(tokens_i[t], n=8) for t in xticks],
                rotation=60,
                ha="right",
            )
            ax.set_xlabel("CoT token")
        else:
            ax.set_xticklabels([])
        fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)

    cot_preview = "".join(tokens_per_row[i][:30]).replace("\n", " ").strip()
    cot_preview = cot_preview if len(cot_preview) <= 110 else cot_preview[:107] + "…"
    fig.suptitle(
        f"{experiment_name} — attention pyramid\n{cot_preview}",
        y=1.02,
    )
    fig.tight_layout()
    _save_figure(fig, output_dir, file_stem)


def _plot_attention_text_overlay(
    pyramid: PyramidOutput,
    tokens_per_row: Sequence[Sequence[str]],
    output_dir: Path,
    experiment_name: str,
    *,
    file_stem: str = "attention_text_overlay",
    tokens_per_line: int = 18,
) -> None:
    """Single-sample CoT-token coloured-box overlay, one row per level.

    For sample 0 (input pyramid is expected to carry ``B == 1``) we
    aggregate each level's attention across concept slots
    (``a_k[t] = sum_j A_k[0, j, t]``), normalise per level, then draw the
    CoT token sequence as a grid of rounded boxes wrapping every
    ``tokens_per_line`` tokens. Box background colour (YlOrRd) encodes
    ``a_k[t]``; one subplot per level, all stacked vertically.

    Saved as ``<file_stem>.{png,pdf}`` in ``output_dir``.
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
    fig_h = max(2.2, K * (0.45 * n_lines + 0.9))
    fig, axes = plt.subplots(K, 1, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes[:, 0]

    for k in range(K):
        ax = axes[k]
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

        ax.set_title(
            f"Level {k}  (L_k={pyramid.level_lengths[k]}, max_w={a_max:.3f})",
            loc="left",
            fontsize=11,
        )
        import matplotlib as _mpl

        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=_mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        )
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.012, pad=0.01)

    preview = "".join(tokens_i[:30]).replace("\n", " ").strip()
    preview = preview if len(preview) <= 110 else preview[:107] + "…"
    fig.suptitle(
        f"{experiment_name} — attention-on-text overlay\n{preview}",
        y=1.0,
    )
    fig.tight_layout()
    _save_figure(fig, output_dir, file_stem)


def _plot_concept_norms(
    pyramid: PyramidOutput,
    output_dir: Path,
    experiment_name: str,
) -> None:
    """Per-level histograms of ``||C_k[b,j]||``.

    Coarse-to-fine decomposition predicts that deeper levels operate on
    a shrinking residual, so concept norms should *decrease* with ``k``.
    Plotting all K histograms with a shared x-axis makes the trend legible.
    """
    K = pyramid.num_levels
    # Pool norms per level (across batch + slots) onto one row of subplots.
    norms_per_level: List[np.ndarray] = []
    for k in range(K):
        C_k = pyramid.concepts[k]  # [B, L_k, D]
        n_k = torch.linalg.vector_norm(C_k, dim=-1).detach().cpu().float().numpy()
        norms_per_level.append(n_k.reshape(-1))

    global_max = (
        float(max(n.max() for n in norms_per_level)) if norms_per_level else 1.0
    )

    cols = min(K, 3)
    rows = int(np.ceil(K / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.5 * cols, 3.2 * rows), squeeze=False
    )
    axes_flat = axes.flatten()
    for k in range(K):
        ax = axes_flat[k]
        ax.hist(norms_per_level[k], bins=30, color="tab:blue", alpha=0.75)
        ax.axvline(
            float(np.mean(norms_per_level[k])),
            color="red",
            linestyle="--",
            linewidth=1.2,
            label="mean",
        )
        ax.set_title(f"level {k} (L_k={pyramid.level_lengths[k]})")
        ax.set_xlabel(r"$\|C_k\|_2$")
        ax.set_ylabel("count")
        ax.set_xlim(0, global_max * 1.05)
        ax.legend(loc="upper right")
    for idx in range(K, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    fig.suptitle(f"{experiment_name} — concept-norm distribution per level")
    fig.tight_layout()
    _save_figure(fig, output_dir, "concept_norms")


def _plot_residual_decomposition(
    pyramid: PyramidOutput,
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
    pyramid: PyramidOutput,
    output_dir: Path,
    experiment_name: str,
) -> None:
    """K cosine-similarity heatmaps, one per level (averaged over batch).

    For level k the slot-to-slot matrix has shape ``[L_k, L_k]``. A diffuse
    matrix indicates orthogonal slots (healthy); a near-uniform matrix
    signals slot collapse.
    """
    K = pyramid.num_levels
    mats: List[np.ndarray] = []
    for k in range(K):
        C_k = pyramid.concepts[k]  # [B, L_k, D]
        # Unit-normalise along D, then outer product per-batch, average.
        eps = 1e-8
        C_unit = C_k / (torch.linalg.vector_norm(C_k, dim=-1, keepdim=True) + eps)
        sim = torch.bmm(C_unit, C_unit.transpose(1, 2))  # [B, L_k, L_k]
        mats.append(sim.mean(dim=0).detach().cpu().float().numpy())

    cols = min(K, 3)
    rows = int(np.ceil(K / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.0 * cols, 3.6 * rows), squeeze=False
    )
    axes_flat = axes.flatten()
    for k in range(K):
        ax = axes_flat[k]
        im = ax.imshow(
            mats[k],
            aspect="auto",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax.set_title(f"level {k} (L_k={pyramid.level_lengths[k]})")
        ax.set_xlabel("slot j")
        ax.set_ylabel("slot i")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    for idx in range(K, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    fig.suptitle(f"{experiment_name} — intra-level concept cosine similarity")
    fig.tight_layout()
    _save_figure(fig, output_dir, "intra_level_similarity")


def _plot_inter_level_similarity(
    pyramid: PyramidOutput,
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
    pyramid: PyramidOutput,
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
    pyramid: PyramidOutput,
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
    pyramid: PyramidOutput,
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


def _resolve_output_root(
    config: Dict[str, Any],
    checkpoint_override: Optional[Path],
) -> Path:
    """Where to write ``concept_pyramid_analysis/`` for this run.

    * Without ``-p`` → next to the model's logs (``<log>.parent``).
    * With ``-p`` → ``Path(-p).parent.parent``. This co-locates the
      figures with the experiment that owns the override checkpoint.
    """
    if checkpoint_override is not None:
        return checkpoint_override.parent.parent / ANALYSIS_OUTPUT_DIR_NAME
    log_dir = Path(config["log"]["log_path"]).expanduser()
    return log_dir.parent / ANALYSIS_OUTPUT_DIR_NAME


def _save_sample_input(batch: BuilderInput, output_dir: Path, sample_id: str) -> None:
    """Write the sample's raw Q + CoT (and solution if any) to ``input.txt``.

    Co-locating the source text with every figure means the reader can
    open one folder and immediately verify which substrings drive the
    attention overlay — no need to reproduce the deterministic batch.
    """
    lines: List[str] = [
        f"sample_id: {sample_id}",
        "=" * 70,
        "## QUESTION",
        batch.questions[0] if batch.questions else "",
        "",
        "## CHAIN-OF-THOUGHT",
        batch.cot_answers[0] if batch.cot_answers else "",
    ]
    if batch.solutions:
        lines.extend(["", "## SOLUTION", batch.solutions[0]])
    (output_dir / "input.txt").write_text("\n".join(lines), encoding="utf-8")


def _existing_outputs(
    output_dir: Path, analyses: Sequence[str], num_samples: int
) -> bool:
    """Return True iff ``num_samples`` sample folders already carry every analysis.

    Layout: ``<output_dir>/sample_*/<file_stem>.png``. Both PNG and PDF
    are written together so checking the PNG is sufficient.
    """
    if not output_dir.is_dir():
        return False
    sample_dirs = sorted(
        p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("sample_")
    )
    if len(sample_dirs) < num_samples:
        return False
    needed_stems = [
        _ANALYSIS_FILE_STEMS[k] for k in analyses if k in _ANALYSIS_FILE_STEMS
    ]
    for sd in sample_dirs[:num_samples]:
        for stem in needed_stems:
            if not (sd / f"{stem}.png").is_file():
                return False
    return True


def _run_concept_analysis(
    config_path: Path,
    *,
    storage_root: str,
    split: str,
    num_samples: int,
    analyses: Sequence[str],
    seed: int,
    checkpoint_override: Optional[Path],
) -> None:
    """Full pipeline for a single config: load -> per-sample forward -> dispatch.

    Forward pass uses ``batch_size=1`` by design. Each sample lands in
    ``<output_root>/sample_<main_id>/`` together with ``input.txt`` and
    every requested analysis figure. Aggregate-style plots (histograms,
    PCA, similarity matrices) still render meaningfully on a B=1 slice
    because they pool over the per-sample slot dimension.
    """
    config = load_config(str(config_path))
    apply_storage_root(config, storage_root)
    _apply_plot_style()

    experiment_name = _derive_experiment_name(config_path)
    output_root = _resolve_output_root(config, checkpoint_override)
    output_root.mkdir(parents=True, exist_ok=True)

    builder, device, _ckpt_path = load_builder(
        config, checkpoint_override=checkpoint_override
    )

    # Collect num_samples deterministic rows; loader uses bs=1 so the
    # first num_samples rows are always the same regardless of split.
    batch = collect_batch(
        config,
        split=split,
        num_samples=num_samples,
        batch_size=1,
        seed=seed,
    )
    logger.info(
        "Per-sample analysis: split=%s  N=%d  device=%s",
        split,
        batch.batch_size,
        device,
    )

    analyses_set = set(analyses)
    actual = min(num_samples, batch.batch_size)
    for i in range(actual):
        sample_id = _safe_id(batch.main_ids[i])
        sample_dir = output_root / f"sample_{sample_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Single-sample BuilderInput -> forward yields a B=1 pyramid.
        single = BuilderInput(
            questions=[batch.questions[i]],
            cot_answers=[batch.cot_answers[i]],
            solutions=[batch.solutions[i]] if batch.solutions else [],
            main_ids=[batch.main_ids[i]],
        )
        _save_sample_input(single, sample_dir, sample_id)
        pyramid, tokens_per_row = run_forward(builder, single)

        # Dispatch every requested analysis into THIS sample's folder.
        if "attention_heatmaps" in analyses_set:
            _plot_attention_heatmaps(
                pyramid,
                tokens_per_row,
                sample_dir,
                experiment_name,
                file_stem=_ANALYSIS_FILE_STEMS["attention_heatmaps"],
            )
        if "attention_text_overlay" in analyses_set:
            _plot_attention_text_overlay(
                pyramid,
                tokens_per_row,
                sample_dir,
                experiment_name,
                file_stem=_ANALYSIS_FILE_STEMS["attention_text_overlay"],
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

        logger.info(
            "sample_%s: wrote %d analyses -> %s",
            sample_id,
            len(analyses_set),
            sample_dir,
        )

    print(f"Saved {actual} sample folder(s) to {output_root}")


def analyze_one(
    config_path: Path,
    *,
    overlap: bool,
    storage_root: str,
    split: str,
    num_samples: int,
    analyses: Sequence[str],
    seed: int,
    checkpoint_override: Optional[Path],
) -> Tuple[str, str]:
    """Per-config status tracker — mirrors the pattern of ``builder_training_analysis``."""
    try:
        config = load_config(str(config_path))
    except Exception as exc:  # noqa: BLE001
        return "error", f"load_config failed: {exc}"

    apply_storage_root(config, storage_root)
    print_storage_paths(config, storage_root)

    # Fast skip: with -p we just check the override file exists; without -p
    # we look for any checkpoint_best*.pt under the config-resolved dir.
    if checkpoint_override is None:
        checkpoint_dir = Path(config["log"]["checkpoint_path"]).expanduser()
        if _locate_best_checkpoint(checkpoint_dir) is None:
            return "skip_no_data", f"no checkpoint under {checkpoint_dir}"
    else:
        if not checkpoint_override.is_file():
            return "skip_no_data", f"-p file not found: {checkpoint_override}"

    output_root = _resolve_output_root(config, checkpoint_override)
    if not overlap and _existing_outputs(output_root, analyses, num_samples):
        return "skip_exists", f"outputs exist at {output_root}"

    try:
        _run_concept_analysis(
            config_path,
            storage_root=storage_root,
            split=split,
            num_samples=num_samples,
            analyses=analyses,
            seed=seed,
            checkpoint_override=checkpoint_override,
        )
    except Exception as exc:  # noqa: BLE001
        plt.close("all")
        return "error", f"{type(exc).__name__}: {exc}"

    return "analyzed", f"wrote figures to {output_root}"


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
    """CLI entry — iterate matching configs and emit concept-analysis figures."""
    args = parse_args()
    overlap: bool = args.overlap
    storage_root: str = args.storage_root
    split: str = args.split
    num_samples: int = args.num_samples
    analyses = _parse_analyses(args.analyses)
    seed: int = args.seed
    checkpoint_override: Optional[Path] = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint is not None
        else None
    )

    if num_samples <= 0:
        raise SystemExit("[ERROR] --num-samples must be > 0")
    if checkpoint_override is not None and not checkpoint_override.is_file():
        raise SystemExit(f"[ERROR] -p/--checkpoint not found: {checkpoint_override}")

    configs = _resolve_configs_from_args(args)
    if not configs:
        return 1
    if checkpoint_override is not None and len(configs) != 1:
        raise SystemExit(
            "[ERROR] -p/--checkpoint requires exactly one config (use -c)."
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    source_desc = (
        f"config={args.config}"
        if args.config is not None
        else f"dataset={args.dataset}  experiment={args.experiment}"
    )
    ckpt_desc = (
        f"  checkpoint={checkpoint_override}" if checkpoint_override is not None else ""
    )
    print(
        f"[ANALYZE] concept pyramid  {source_desc}  split={split}  "
        f"num_samples={num_samples}  bs=1  "
        f"analyses={','.join(analyses)}  overlap={overlap}{ckpt_desc}  "
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
            split=split,
            num_samples=num_samples,
            analyses=analyses,
            seed=seed,
            checkpoint_override=checkpoint_override,
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
