"""Inspect a trained predictor's concept pyramid: predicted vs ground-truth.

Stage-2 sibling of ``builder_concept_pyramid_analysis.py``. Loads a
paired (builder, predictor) pair driven by a predictor YAML, runs ONE
teacher-forced forward pass per sample on a deterministic batch drawn
from ``evaluation.data``, and emits a folder of figures *per sample*.

WHAT IS ANALYZED
================
The predictor's :class:`PredictorOutput` carries per level:

    * ``C_hat_k`` (predicted concept encodings) : [B, L_k, D]
    * ``C_gt_k``  (ground-truth concepts, optional) : [B, L_k, D]

Unlike the builder, the predictor does NOT expose attention weights
(Option X uses the native LLM causal mask; Option Y explicitly discards
cross-attention weights via ``need_weights=False``). The analyses here
therefore focus on concept geometry and on the predicted-vs-gt
discrepancy.

AVAILABLE ANALYSES (select via ``--analyses``)
==============================================
    ``predicted_vs_gt_mse``    — per-level MSE between ``C_hat_k`` and
                                 ``C_gt_k`` (mean over feature dim;
                                 spread is taken across slots).
    ``predicted_vs_gt_cosine`` — per-level slot-wise cosine similarity.
    ``per_slot_cosine_heatmap``— K subplots, each a [1, L_k] cosine
                                 strip for THIS sample.
    ``norm_drift``             — per-level side-by-side box plots of
                                 ``||C_hat_k[·,j]||`` vs ``||C_gt_k[·,j]||``.
    ``concept_norms``          — per-level histograms of
                                 ``||C_hat_k[·,j]||`` (predicted only).
    ``intra_level_similarity`` — K cosine-sim heatmaps over predicted
                                 slots within each level.
    ``inter_level_similarity`` — single K×K cosine matrix over
                                 slot-averaged predicted concepts.
    ``joint_concept_pca``      — 2-D PCA scatter of predicted + gt on
                                 shared axes, colour by level, shape
                                 by source.
    ``all`` (default)          — run every analysis above.

Usage
-----
Direct config path::

    python3 examples/nlcpV4/predictor_concept_pyramid_analysis.py \\
        -c configs/nlcpV4/GSM8K/train_predictor_Qwen2.5-0.5B_6level_independent.yml

Discovery::

    python3 examples/nlcpV4/predictor_concept_pyramid_analysis.py \\
        -d GSM8K -e Qwen2.5-0.5B_6level_independent

Override checkpoint (only ``evaluation.data`` is read from the YAML)::

    python3 examples/nlcpV4/predictor_concept_pyramid_analysis.py \\
        -c <yaml> -p /path/to/checkpoint.pt

Arguments
---------
    -s / --storage-root  Prefix for relative ``log.*`` paths. MUST match
                         the value used at training time.
    -c / --config        Direct path to a ``train_predictor_*.yml``.
                         Mutually exclusive with ``-d``/``-e``.
    -d / --dataset       Dataset dir under ``configs/nlcpV4/``.
    -e / --experiment    Config stem after ``train_predictor_`` or ``all``.
    -p / --checkpoint    Override the predictor checkpoint .pt file. When
                         set, this exact file is loaded; the rest of the
                         config still drives builder + evaluation data.
                         Outputs go next to ``-p``'s grandparent dir.
    -o / --overlap       If true (default), overwrite existing figures.
    --split              ``test`` (default) or ``train``.
    --num-samples        How many samples to analyze (each gets its own
                         folder). Default 4. Forward batch size is
                         always 1, ignoring any value in the config.
    --analyses           Comma-separated subset or ``all`` (default).
    --seed               RNG seed for deterministic batch sampling.

Storage layout
--------------
All output is grouped per sample::

    <output_root>/concept_pyramid_analysis/
    ├── sample_<safe_main_id_0>/
    │   ├── input.txt
    │   ├── predicted_vs_gt_mse.{png,pdf}
    │   ├── predicted_vs_gt_cosine.{png,pdf}
    │   ├── per_slot_cosine_heatmap.{png,pdf}
    │   ├── norm_drift.{png,pdf}
    │   ├── concept_norms.{png,pdf}
    │   ├── intra_level_similarity.{png,pdf}
    │   ├── inter_level_similarity.{png,pdf}
    │   └── joint_concept_pca.{png,pdf}
    └── sample_<safe_main_id_1>/
        └── ...

``<output_root>`` resolves to ``<log_path>.parent`` from the YAML, or
to ``Path(-p).parent.parent`` when an override checkpoint is given.
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
from nlcpV4.concept_builder import ConceptPyramidBuilder
from nlcpV4.concept_predictor import ConceptPredictor, PredictorOutput
from nlcpV4.data_loader import BuilderInput, NLCPV4DataLoader
from nlcpV4.eval_builder import _strip_solutions, _tokenize_qs
from nlcpV4.train_predictor import (
    _inherit_pyramid_from_builder,
    _load_frozen_builder,
    _resolve_builder_checkpoint_path,
    _resolve_config_path,
)
from ram.utils import apply_storage_root, load_config, print_storage_paths

logger = logging.getLogger(__name__)

# --- Batch-mode constants ------------------------------------------
CONFIGS_ROOT = PROJECT_ROOT / "configs" / "nlcpV4"
ALL_KEYWORD = "all"
ANALYSIS_OUTPUT_DIR_NAME = "concept_pyramid_analysis"
ALL_ANALYSES = (
    "predicted_vs_gt_mse",
    "predicted_vs_gt_cosine",
    "per_slot_cosine_heatmap",
    "norm_drift",
    "concept_norms",
    "intra_level_similarity",
    "inter_level_similarity",
    "joint_concept_pca",
)
# Single source of truth for the filename stem each analysis writes
# into a per-sample folder (``<sample_dir>/<stem>.{png,pdf}``).
_ANALYSIS_FILE_STEMS: Dict[str, str] = {
    "predicted_vs_gt_mse": "predicted_vs_gt_mse",
    "predicted_vs_gt_cosine": "predicted_vs_gt_cosine",
    "per_slot_cosine_heatmap": "per_slot_cosine_heatmap",
    "norm_drift": "norm_drift",
    "concept_norms": "concept_norms",
    "intra_level_similarity": "intra_level_similarity",
    "inter_level_similarity": "inter_level_similarity",
    "joint_concept_pca": "joint_concept_pca",
}


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments. Mirrors ``builder_concept_encodings_analysis``."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze predicted vs ground-truth concept encodings from a "
            "trained predictor"
        ),
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
            "Direct path to a single train_predictor_*.yml. Mutually "
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
            "Config stem after 'train_predictor_' or 'all' to process "
            "every matching config. Required when -c is not given."
        ),
    )
    parser.add_argument(
        "-p",
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Override the predictor checkpoint .pt file. When set, this "
            "exact file is loaded; the rest of the config still drives "
            "builder + evaluation data. Outputs go to the parent of "
            "-p's parent dir / concept_pyramid_analysis/."
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
    """Resolve (-d, -e) to a list of ``train_predictor_*.yml`` paths."""
    dataset_dir = CONFIGS_ROOT / dataset
    if not dataset_dir.is_dir():
        print(f"[ERROR] Dataset dir not found: {dataset_dir}")
        return []

    prefix = "train_predictor_"
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


# =============================================================================
# Predictor + builder + data loading
# =============================================================================


def _locate_best_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Return the preferred predictor ``.pt``, or None when absent.

    Mirrors the builder-side helper but also handles the epoch/step
    suffixed filenames (``checkpoint_best_eval-epoch9-step18500.pt``).
    """
    for stem in ("checkpoint_best_eval", "checkpoint_best"):
        exact = checkpoint_dir / f"{stem}.pt"
        if exact.is_file():
            return exact
        matches = sorted(checkpoint_dir.glob(f"{stem}*.pt"))
        if matches:
            # Latest by step, falling back to lexical order.
            def _step(p: Path) -> int:
                import re as _re

                m = _re.search(r"-step(\d+)", p.name)
                return int(m.group(1)) if m else 0

            matches.sort(key=_step, reverse=True)
            return matches[0]
    return None


def load_predictor(
    predictor_config_path: Path,
    storage_root: str,
    *,
    checkpoint_override: Optional[Path] = None,
) -> Tuple[ConceptPredictor, ConceptPyramidBuilder, dict, torch.device, Path]:
    """Load a frozen builder + predictor pair from a predictor YAML.

    Replicates the exact config-driven flow used by
    :func:`train_predictor.train_predictor`. When ``checkpoint_override``
    is given, that exact .pt file is loaded for the predictor; the
    builder, evaluation data and pyramid hyper-parameters still come
    from the YAML(s).

    Raises ``FileNotFoundError`` when a required checkpoint is missing
    so the caller can classify the config as ``skip_no_data``.
    """
    # --- Predictor config ---
    config = load_config(str(predictor_config_path))
    apply_storage_root(config, storage_root)

    # --- Paired builder config ---
    builder_ref = config["model"]["builder"]
    builder_cfg_path = _resolve_config_path(
        builder_ref["config_path"], predictor_config_path.parent
    )
    if not builder_cfg_path.is_file():
        raise FileNotFoundError(f"Paired builder config not found: {builder_cfg_path}")
    builder_config = load_config(str(builder_cfg_path))
    apply_storage_root(builder_config, storage_root)
    _inherit_pyramid_from_builder(config, builder_config)

    # --- Device ---
    device = torch.device(str(get_device("auto")))

    # --- Frozen builder ---
    builder_ckpt_path = _resolve_builder_checkpoint_path(
        builder_ref["checkpoint_path"], storage_root
    )
    strict = bool(builder_ref.get("strict_load", False))
    builder = _load_frozen_builder(
        builder_config, builder_ckpt_path, strict, str(device), logger
    )

    # --- Predictor ---
    predictor = ConceptPredictor(config, builder=builder)
    predictor.to(device)

    # --- Predictor checkpoint ---
    if checkpoint_override is not None:
        ckpt_path: Optional[Path] = checkpoint_override
        if ckpt_path is None or not ckpt_path.is_file():
            raise FileNotFoundError(f"-p/--checkpoint not found: {checkpoint_override}")
    else:
        checkpoint_dir = Path(config["log"]["checkpoint_path"]).expanduser()
        ckpt_path = _locate_best_checkpoint(checkpoint_dir)
        if ckpt_path is None:
            raise FileNotFoundError(
                f"No checkpoint_best*.pt for predictor under {checkpoint_dir}"
            )
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"]
    missing, unexpected = predictor.load_state_dict(state, strict=False)
    if missing or unexpected:
        logger.warning(
            "Predictor loaded with strict=False | missing=%d unexpected=%d",
            len(missing),
            len(unexpected),
        )
    predictor.eval()
    logger.info(
        "Predictor checkpoint: %s (step=%s, loss=%.4f)",
        ckpt_path,
        ckpt.get("step", "?"),
        ckpt.get("eval_loss", ckpt.get("loss", 0.0)),
    )

    return predictor, builder, config, device, ckpt_path


def collect_batch(
    config: Dict[str, Any],
    *,
    split: str,
    num_samples: int,
    batch_size: int,
    seed: int,
) -> BuilderInput:
    """Gather ``num_samples`` rows from ``evaluation.data`` deterministically.

    Identical semantics to the builder variant: we hijack
    ``evaluation.data``'s split and keep ``include_solution=True`` so
    the unified teacher-forced forward can optionally use solution_ids
    (we read but do not require it).
    """
    data_cfg = dict(config["evaluation"]["data"])
    data_cfg["split"] = split

    env_cfg = config["environment"]
    loader = NLCPV4DataLoader(
        data_cfg=data_cfg,
        batch_size=batch_size,
        include_solution=True,
        shuffle=False,
        drop_last=False,
        num_workers=env_cfg["dataloader_num_workers"],
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    questions: List[str] = []
    cot_answers: List[str] = []
    solutions: List[str] = []
    main_ids: List[str] = []
    for batch in loader:
        take = min(num_samples - len(questions), batch.batch_size)
        questions.extend(batch.questions[:take])
        cot_answers.extend(batch.cot_answers[:take])
        solutions.extend(
            list(batch.solutions)[:take] if batch.solutions else [""] * take
        )
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
        solutions=solutions,
        main_ids=main_ids,
    )


@torch.no_grad()
def run_forward(
    predictor: ConceptPredictor,
    builder: ConceptPyramidBuilder,
    batch: BuilderInput,
    max_length: int,
    device: torch.device,
) -> PredictorOutput:
    """Run the frozen builder for gt_concepts, then the predictor teacher-forced.

    We purposefully DROP ``solution_ids`` from the predictor call.
    The goal of this analysis is to inspect predicted concept geometry
    versus the ground truth — adding reasoning CE would waste compute
    and clutter the forward without changing ``predicted_concepts``.
    """
    # Builder: frozen, no_grad, solutions stripped.
    pyramid = builder(_strip_solutions(batch))
    gt_concepts = [c.detach() for c in pyramid.concepts]

    # Tokenize Q with the builder's tokenizer (ids live on device already).
    q_ids, q_mask, _s_ids, _s_mask = _tokenize_qs(
        builder, batch, max_length, str(device)
    )

    # Predictor: teacher-forced (gt_concepts supplied), no solution CE.
    output = predictor(
        question_ids=q_ids,
        question_attention_mask=q_mask,
        gt_concepts=gt_concepts,
    )
    # _forward_training always echoes gt_concepts on the output, but be
    # defensive in case the implementation changes.
    if output.gt_concepts is None:
        output = PredictorOutput(
            predicted_concepts=output.predicted_concepts,
            gt_concepts=gt_concepts,
            num_levels=output.num_levels,
            level_lengths=output.level_lengths,
            reasoning_logits=output.reasoning_logits,
            reasoning_target_ids=output.reasoning_target_ids,
            reasoning_texts=output.reasoning_texts,
            generation_texts=output.generation_texts,
        )
    return output


# =============================================================================
# Figure helpers
# =============================================================================


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """Save ``fig`` as both ``<stem>.png`` (dpi=150) and ``<stem>.pdf``."""
    fig.savefig(output_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _apply_plot_style() -> None:
    """Uniform styling across all figures."""
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


def _concept_pairs(
    output: PredictorOutput,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Return list of (C_hat_k, C_gt_k) level tensors on CPU, detached.

    Raises when gt_concepts is missing — which cannot happen along the
    teacher-forced path we drive, but we defend against it anyway.
    """
    if output.gt_concepts is None:
        raise RuntimeError("PredictorOutput.gt_concepts is None; cannot run analysis.")
    pairs = []
    for c_hat, c_gt in zip(output.predicted_concepts, output.gt_concepts):
        pairs.append((c_hat.detach().cpu().float(), c_gt.detach().cpu().float()))
    return pairs


# =============================================================================
# Analyses
# =============================================================================


def _plot_predicted_vs_gt_mse(
    output: PredictorOutput,
    output_dir: Path,
    experiment_name: str,
) -> None:
    """Per-level mean squared error ``E[||C_hat - C_gt||^2]``."""
    pairs = _concept_pairs(output)
    K = len(pairs)
    mse_per_level: List[float] = []
    std_per_level: List[float] = []
    for c_hat, c_gt in pairs:
        # [B, L_k] — mean over feature dim
        sq = (c_hat - c_gt).pow(2).mean(dim=-1)
        mse_per_level.append(float(sq.mean().item()))
        std_per_level.append(float(sq.std().item()))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    levels = np.arange(K)
    ax.errorbar(
        levels,
        mse_per_level,
        yerr=std_per_level,
        marker="o",
        capsize=3,
        linewidth=2,
    )
    ax.set_xlabel("Pyramid level k")
    ax.set_ylabel(r"$\mathbb{E}\,\|\hat{C}_k - C_k\|^2$ per dim")
    ax.set_title(f"{experiment_name}: Predicted vs GT — MSE per level")
    ax.set_xticks(levels)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, output_dir, "predicted_vs_gt_mse")


def _plot_predicted_vs_gt_cosine(
    output: PredictorOutput,
    output_dir: Path,
    experiment_name: str,
) -> None:
    """Per-level mean slot-wise cosine similarity between predicted and gt."""
    pairs = _concept_pairs(output)
    K = len(pairs)
    mean_cos: List[float] = []
    std_cos: List[float] = []
    for c_hat, c_gt in pairs:
        # [B, L_k]
        cos = torch.nn.functional.cosine_similarity(c_hat, c_gt, dim=-1)
        mean_cos.append(float(cos.mean().item()))
        std_cos.append(float(cos.std().item()))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    levels = np.arange(K)
    ax.errorbar(
        levels,
        mean_cos,
        yerr=std_cos,
        marker="s",
        capsize=3,
        linewidth=2,
        color="tab:green",
    )
    ax.axhline(1.0, color="k", linestyle=":", linewidth=1, label="perfect match")
    ax.axhline(0.0, color="r", linestyle=":", linewidth=1, label="orthogonal")
    ax.set_xlabel("Pyramid level k")
    ax.set_ylabel(r"$\mathbb{E}\,\cos(\hat{C}_{k,j}, C_{k,j})$")
    ax.set_title(f"{experiment_name}: Predicted vs GT — cosine per level")
    ax.set_xticks(levels)
    ax.set_ylim(-0.1, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, output_dir, "predicted_vs_gt_cosine")


def _plot_per_slot_cosine_heatmap(
    output: PredictorOutput,
    output_dir: Path,
    experiment_name: str,
    *,
    file_stem: str = "per_slot_cosine_heatmap",
) -> None:
    """Single figure: K subplots of a [1, L_k] cosine strip per level.

    Always renders the FIRST sample in the (B=1) batch. The caller is
    expected to feed one row at a time so each figure lands in its
    own per-sample folder.
    """
    pairs = _concept_pairs(output)
    K = len(pairs)
    B = pairs[0][0].shape[0]
    if B == 0:
        return
    i = 0

    fig, axes = plt.subplots(K, 1, figsize=(10, 1.1 * K + 1.5), sharex=False)
    if K == 1:
        axes = [axes]
    for k, ((c_hat, c_gt), ax) in enumerate(zip(pairs, axes)):
        cos = torch.nn.functional.cosine_similarity(
            c_hat[i : i + 1], c_gt[i : i + 1], dim=-1
        ).numpy()  # [1, L_k]
        im = ax.imshow(cos, aspect="auto", cmap="RdYlGn", vmin=-1.0, vmax=1.0)
        ax.set_ylabel(f"L{k}\n(L={cos.shape[1]})", rotation=0, labelpad=22, va="center")
        ax.set_yticks([])
        if k == K - 1:
            ax.set_xlabel("Slot index j")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    fig.suptitle(
        f"{experiment_name}: Slot-wise cos(pred, gt)",
        y=1.02,
    )
    fig.tight_layout()
    _save_figure(fig, output_dir, file_stem)


def _plot_norm_drift(
    output: PredictorOutput,
    output_dir: Path,
    experiment_name: str,
) -> None:
    """Per-level side-by-side box plots of ||C_hat|| vs ||C_gt||."""
    pairs = _concept_pairs(output)
    K = len(pairs)

    data_pred: List[np.ndarray] = []
    data_gt: List[np.ndarray] = []
    for c_hat, c_gt in pairs:
        data_pred.append(c_hat.norm(dim=-1).flatten().numpy())
        data_gt.append(c_gt.norm(dim=-1).flatten().numpy())

    fig, ax = plt.subplots(figsize=(max(7.0, 1.1 * K + 2), 4.5))
    positions = np.arange(K)
    width = 0.35
    bp_pred = ax.boxplot(
        data_pred,
        positions=positions - width / 2,
        widths=width,
        patch_artist=True,
        showfliers=False,
    )
    bp_gt = ax.boxplot(
        data_gt,
        positions=positions + width / 2,
        widths=width,
        patch_artist=True,
        showfliers=False,
    )
    for b in bp_pred["boxes"]:
        b.set_facecolor("tab:blue")
        b.set_alpha(0.55)
    for b in bp_gt["boxes"]:
        b.set_facecolor("tab:orange")
        b.set_alpha(0.55)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"L{k}" for k in range(K)])
    ax.set_xlabel("Pyramid level k")
    ax.set_ylabel(r"$\|C_{k,j}\|$")
    ax.set_title(f"{experiment_name}: Norm drift — predicted vs GT")
    ax.grid(True, alpha=0.3, axis="y")
    from matplotlib.patches import Patch

    legend_items = [
        Patch(facecolor="tab:blue", alpha=0.55, label="predicted"),
        Patch(facecolor="tab:orange", alpha=0.55, label="ground truth"),
    ]
    ax.legend(handles=legend_items, loc="best")
    fig.tight_layout()
    _save_figure(fig, output_dir, "norm_drift")


def _plot_concept_norms(
    output: PredictorOutput,
    output_dir: Path,
    experiment_name: str,
) -> None:
    """Per-level histograms of ``||C_hat_k[b,j]||`` (predicted only)."""
    pairs = _concept_pairs(output)
    K = len(pairs)
    norms: List[np.ndarray] = []
    for c_hat, _ in pairs:
        norms.append(c_hat.norm(dim=-1).flatten().numpy())
    all_vals = np.concatenate(norms) if norms else np.array([0.0])
    x_min, x_max = float(all_vals.min()), float(all_vals.max())

    cols = min(K, 3)
    rows = (K + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.5 * cols, 3.2 * rows), squeeze=False
    )
    for k in range(K):
        r, c = divmod(k, cols)
        ax = axes[r][c]
        ax.hist(norms[k], bins=40, color="tab:blue", alpha=0.8)
        mu = float(norms[k].mean())
        ax.axvline(mu, color="k", linestyle="--", linewidth=1, label=f"μ={mu:.2f}")
        ax.set_title(f"Level {k}  (L={pairs[k][0].shape[1]})")
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(r"$\|\hat{C}_{k,j}\|$")
        ax.set_ylabel("count")
        ax.legend()
    for k in range(K, rows * cols):
        r, c = divmod(k, cols)
        axes[r][c].axis("off")
    fig.suptitle(f"{experiment_name}: Predicted concept-norm distributions", y=1.0)
    fig.tight_layout()
    _save_figure(fig, output_dir, "concept_norms")


def _plot_intra_level_similarity(
    output: PredictorOutput,
    output_dir: Path,
    experiment_name: str,
) -> None:
    """K cosine-sim heatmaps, one per level, on PREDICTED concepts."""
    pairs = _concept_pairs(output)
    K = len(pairs)
    cols = min(K, 3)
    rows = (K + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False
    )
    for k, (c_hat, _) in enumerate(pairs):
        r, c = divmod(k, cols)
        ax = axes[r][c]
        # c_hat: [B, L_k, D]
        x = torch.nn.functional.normalize(c_hat, dim=-1)
        sim = torch.matmul(x, x.transpose(-1, -2))  # [B, L_k, L_k]
        sim_mean = sim.mean(dim=0).numpy()
        im = ax.imshow(sim_mean, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax.set_title(f"Level {k}  (L={sim_mean.shape[0]})")
        ax.set_xlabel("slot j")
        ax.set_ylabel("slot i")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for k in range(K, rows * cols):
        r, c = divmod(k, cols)
        axes[r][c].axis("off")
    fig.suptitle(f"{experiment_name}: Intra-level cosine similarity (predicted)", y=1.0)
    fig.tight_layout()
    _save_figure(fig, output_dir, "intra_level_similarity")


def _plot_inter_level_similarity(
    output: PredictorOutput,
    output_dir: Path,
    experiment_name: str,
) -> None:
    """Single K×K cosine matrix over slot-averaged predicted concepts."""
    pairs = _concept_pairs(output)
    K = len(pairs)
    # Per-level mean over (B, L_k), then normalise → [K, D]
    means = []
    for c_hat, _ in pairs:
        means.append(c_hat.mean(dim=(0, 1)))
    stacked = torch.stack(means, dim=0)  # [K, D]
    stacked = torch.nn.functional.normalize(stacked, dim=-1)
    sim = torch.matmul(stacked, stacked.t()).numpy()

    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    im = ax.imshow(sim, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    for i in range(K):
        for j in range(K):
            ax.text(
                j,
                i,
                f"{sim[i,j]:.2f}",
                ha="center",
                va="center",
                color="k",
                fontsize=9,
            )
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([f"L{k}" for k in range(K)])
    ax.set_yticklabels([f"L{k}" for k in range(K)])
    ax.set_title(f"{experiment_name}: Inter-level cosine (predicted, slot-mean)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save_figure(fig, output_dir, "inter_level_similarity")


def _plot_joint_concept_pca(
    output: PredictorOutput,
    output_dir: Path,
    experiment_name: str,
) -> None:
    """2-D joint PCA of predicted + gt concepts on shared axes.

    All (B × Σ L_k) predicted points + same many gt points are projected
    onto the top-2 principal directions computed from their union.
    Points are coloured by level and shaped by source (predicted=o, gt=x).
    """
    pairs = _concept_pairs(output)
    K = len(pairs)

    all_vecs: List[torch.Tensor] = []
    level_ids: List[np.ndarray] = []
    source_ids: List[np.ndarray] = []  # 0=pred, 1=gt
    for k, (c_hat, c_gt) in enumerate(pairs):
        B, L_k, _ = c_hat.shape
        all_vecs.append(c_hat.reshape(B * L_k, -1))
        level_ids.append(np.full(B * L_k, k, dtype=np.int64))
        source_ids.append(np.zeros(B * L_k, dtype=np.int64))
        all_vecs.append(c_gt.reshape(B * L_k, -1))
        level_ids.append(np.full(B * L_k, k, dtype=np.int64))
        source_ids.append(np.ones(B * L_k, dtype=np.int64))
    X = torch.cat(all_vecs, dim=0)  # [N, D]
    levels = np.concatenate(level_ids)
    sources = np.concatenate(source_ids)

    X_centered = X - X.mean(dim=0, keepdim=True)
    try:
        _u, _s, v = torch.svd_lowrank(X_centered, q=3)
    except Exception:  # fall back to full SVD for tiny matrices
        _u, _s, v = torch.linalg.svd(X_centered, full_matrices=False)
        v = v.t()
    proj = (X_centered @ v[:, :2]).numpy()  # [N, 2]

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    cmap = plt.get_cmap("viridis", K)
    for k in range(K):
        for src, marker, alpha in ((0, "o", 0.55), (1, "x", 0.85)):
            m = (levels == k) & (sources == src)
            if not m.any():
                continue
            label = f"L{k} pred" if src == 0 else f"L{k} gt"
            ax.scatter(
                proj[m, 0],
                proj[m, 1],
                s=14 if src == 0 else 18,
                color=cmap(k),
                marker=marker,
                alpha=alpha,
                edgecolors="none" if src == 0 else cmap(k),
                linewidths=0.7,
                label=label,
            )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"{experiment_name}: Joint PCA of predicted + GT concepts")
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        markerscale=1.2,
        ncol=1,
    )
    fig.tight_layout()
    _save_figure(fig, output_dir, "joint_concept_pca")


# =============================================================================
# Orchestration
# =============================================================================


def _derive_experiment_name(config_path: Path) -> str:
    """Strip ``train_predictor_`` prefix + ``.yml`` suffix for titles/filenames."""
    stem = config_path.stem
    prefix = "train_predictor_"
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

    * Without ``-p`` → ``Path(<log_path>).parent``.
    * With ``-p`` → ``Path(-p).parent.parent``, co-locating the figures
      with the experiment that owns the override checkpoint.
    """
    if checkpoint_override is not None:
        return checkpoint_override.parent.parent / ANALYSIS_OUTPUT_DIR_NAME
    log_dir = Path(config["log"]["log_path"]).expanduser()
    return log_dir.parent / ANALYSIS_OUTPUT_DIR_NAME


def _save_sample_input(batch: BuilderInput, output_dir: Path, sample_id: str) -> None:
    """Write the sample's raw Q + CoT (and solution if any) to ``input.txt``."""
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
    """True iff ``num_samples`` sample folders already carry every analysis."""
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
    """Full pipeline for a single predictor config.

    Output layout under ``<output_root>/concept_pyramid_analysis/``::

        sample_<id>/
            input.txt
            <analysis>.{png,pdf}   for every requested analysis

    The forward pass always runs with batch_size=1; this keeps each
    sample's figures self-contained and lets the user open one folder
    to read both the source text and every diagnostic for that row.
    """
    _apply_plot_style()

    experiment_name = _derive_experiment_name(config_path)

    # Load predictor (this also resolves + applies storage-root to config).
    predictor, builder, config, device, _ckpt = load_predictor(
        config_path, storage_root, checkpoint_override=checkpoint_override
    )
    output_root = _resolve_output_root(config, checkpoint_override)
    output_root.mkdir(parents=True, exist_ok=True)

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

    max_length = config["model"]["pyramid"]["max_seq_len"]
    analyses_set = set(analyses)
    actual = min(num_samples, batch.batch_size)
    for i in range(actual):
        sample_id = _safe_id(batch.main_ids[i])
        sample_dir = output_root / f"sample_{sample_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Single-sample BuilderInput → forward yields a B=1 PredictorOutput.
        single = BuilderInput(
            questions=[batch.questions[i]],
            cot_answers=[batch.cot_answers[i]],
            solutions=[batch.solutions[i]] if batch.solutions else [],
            main_ids=[batch.main_ids[i]],
        )
        _save_sample_input(single, sample_dir, sample_id)
        output = run_forward(predictor, builder, single, max_length, device)

        # Dispatch every requested analysis into THIS sample's folder.
        if "predicted_vs_gt_mse" in analyses_set:
            _plot_predicted_vs_gt_mse(output, sample_dir, experiment_name)
        if "predicted_vs_gt_cosine" in analyses_set:
            _plot_predicted_vs_gt_cosine(output, sample_dir, experiment_name)
        if "per_slot_cosine_heatmap" in analyses_set:
            _plot_per_slot_cosine_heatmap(
                output,
                sample_dir,
                experiment_name,
                file_stem=_ANALYSIS_FILE_STEMS["per_slot_cosine_heatmap"],
            )
        if "norm_drift" in analyses_set:
            _plot_norm_drift(output, sample_dir, experiment_name)
        if "concept_norms" in analyses_set:
            _plot_concept_norms(output, sample_dir, experiment_name)
        if "intra_level_similarity" in analyses_set:
            _plot_intra_level_similarity(output, sample_dir, experiment_name)
        if "inter_level_similarity" in analyses_set:
            _plot_inter_level_similarity(output, sample_dir, experiment_name)
        if "joint_concept_pca" in analyses_set:
            _plot_joint_concept_pca(output, sample_dir, experiment_name)

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
    """Per-config status tracker — mirrors the builder variant."""
    try:
        config = load_config(str(config_path))
    except Exception as exc:  # noqa: BLE001
        return "error", f"load_config failed: {exc}"

    apply_storage_root(config, storage_root)
    print_storage_paths(config, storage_root)

    # Fast skip: with -p we just check the override exists; without -p
    # we look for any predictor checkpoint_best*.pt under the resolved dir.
    if checkpoint_override is None:
        checkpoint_dir = Path(config["log"]["checkpoint_path"]).expanduser()
        if _locate_best_checkpoint(checkpoint_dir) is None:
            return "skip_no_data", f"no predictor checkpoint under {checkpoint_dir}"
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
    except FileNotFoundError as exc:
        plt.close("all")
        return "skip_no_data", f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        plt.close("all")
        return "error", f"{type(exc).__name__}: {exc}"

    return "analyzed", f"wrote figures to {output_root}"


def _print_summary(rows: List[Tuple[str, str, str]]) -> None:
    """Compact status table — identical layout to the builder variant."""
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
        f"[ANALYZE] predictor concept pyramid  {source_desc}  split={split}  "
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
