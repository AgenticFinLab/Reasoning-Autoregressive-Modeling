"""lcp Builder evaluation: library + standalone CLI.

================================================================================
  OVERVIEW
================================================================================
This module is the single source of truth for *Builder* evaluation. It is
dual-purpose:

    1. LIBRARY  -- ``train_builder.py`` (and analysis scripts) import
       ``evaluate_builder``, ``log_eval_results``, ``log_terminal_entry``,
       ``compute_reasoning_accuracy``, ``BuilderEvalArtifacts`` and the
       ``MODE_*`` constants directly. No duplication; the same code path
       runs at every training-time eval and at every standalone run.

    2. CLI      -- ``python examples/lcp/eval_builder.py -c <yaml>``
       loads a checkpoint, runs eval on the configured eval split, and
       writes a self-contained ``sample_<id>/`` folder per row plus an
       aggregate ``eval_summary.json``.

The predictor counterpart lives in ``eval_predictor.py``; neither file
imports from the other.

================================================================================
  LIBRARY USAGE  (called from training / analysis code)
================================================================================

    from lcp.eval_builder import (
        evaluate_builder,            # main eval loop
        log_eval_results,            # console + SwanLab + JSON history
        log_terminal_entry,          # one JSONL row
        compute_reasoning_accuracy,  # exact-match scorer
        BuilderEvalArtifacts,        # per-file dump toggle dataclass
        MODE_TEACHER_FORCED, MODE_FREE_GENERATION, MODE_BOTH, VALID_MODES,
    )

    # Training-time eval: NO per-sample folders (dump_artifacts=False).
    avg_losses, texts_dict, samples = evaluate_builder(
        builder=builder,
        eval_dataloader=eval_dataloader,
        loss_weights=loss_weights,
        ordering_loss_type=ordering_loss_type,
        max_batches=quick_eval_batches,
        mode=eval_mode,                  # YAML evaluation.mode
        generation_max_tokens=gen_max_tokens,
        output_root=None,                # ignored when dump_artifacts=False
        dump_artifacts=False,
    )

    # CLI-style eval: per-sample folders, with artifact toggle.
    # AutoWeighted example — resolved checkpoint/log paths carry the
    # ``_AutoWeighted`` suffix but no code change is needed here.
    artifacts = BuilderEvalArtifacts.from_config(config["evaluation"])
    avg_losses, texts_dict, samples = evaluate_builder(
        ...,
        output_root=Path(
            "/Data/ReasoningNLCP/EXPERIMENT/lcp/builder/"
            "GSM8K_Qwen2.5-0.5B_3level_AutoWeighted/logs/eval_builder/teacher_forced"
        ),
        dump_artifacts=True,
        artifacts=artifacts,
    )

================================================================================
  CLI TEMPLATE
================================================================================

    python examples/lcp/eval_builder.py -c <path/to/train_builder_*.yml> -s <storage-root> --mode <teacher_forced | free_generation | both> --max-samples <N> [--ckpt <explicit/checkpoint.pt>] [-o <custom/output/dir>] [-v <0 | 1>]

Flag reference
--------------
    -c / --config                 (REQUIRED) YAML config path. Builder
                                  reconstruction is configured from
                                  ``model.*`` and ``training.*``.
    -s / --storage-root           Prefix prepended to RELATIVE log /
                                  checkpoint paths in the YAML. MUST
                                  match the value used at training time.
                                  Default: ``/Data/ReasoningNLCP``
    --ckpt                        Explicit checkpoint path. When set it
                                  OVERRIDES auto-discovery from
                                  ``log.checkpoint_path``. Auto-discovery
                                  prefers ``checkpoint_best_eval*.pt`` >
                                  ``checkpoint_best*.pt`` >
                                  ``checkpoint*.pt`` (highest step wins).
    --mode                        (REQUIRED) Which reasoning path to run
                                  per sample. ``teacher_forced`` decodes
                                  with ground-truth solution context;
                                  ``free_generation`` decodes from
                                  [Q, Concepts] only; ``both`` runs both
                                  paths sequentially.
    --max-samples                 (REQUIRED) Cap on rows processed (0 =
                                  all). batch_size is hard-coded to 1.
    -o / --output-dir             Override the per-sample folder root.
                                  Default: ``<log_path>/eval_builder/<mode>/``
    -v / --intermediate-vector-save  Override
                                  ``evaluation.intermediate_vector_save``.
                                  ``0`` skips ``pyramid.pt`` (the only
                                  large file); ``1`` writes it. When
                                  omitted, the YAML value is used.

================================================================================
  USAGE SCENARIOS
================================================================================

All examples below use ``AutoWeighted`` configs (auto-generated from
``Loss_prepare_weights.csv``) — the standard training variant. The
schema is identical to the plain configs, so ``eval_builder.py``
handles both without modification; the only difference is that
AutoWeighted log / checkpoint paths carry the ``_AutoWeighted``
suffix and ``training.loss_weights`` values are auto-tuned.

A. Quick check on 10 samples, teacher-forced, defaults from YAML
   (vector dump comes from ``evaluation.intermediate_vector_save``):

       python examples/lcp/eval_builder.py -c configs/lcp/GSM8K/AutoWeighted/train_builder_Qwen2.5-0.5B_3level.yml -s /Data/ReasoningNLCP --mode teacher_forced --max-samples 10

B. Free-generation only, full eval split, vectors disabled to save disk:

       python examples/lcp/eval_builder.py -c configs/lcp/GSM8K/AutoWeighted/train_builder_Qwen2.5-0.5B_3level.yml -s /Data/ReasoningNLCP --mode free_generation --max-samples 0 -v 0

C. Run BOTH paths and compare, with explicit checkpoint and custom output:

       python examples/lcp/eval_builder.py -c configs/lcp/GSM8K/AutoWeighted/train_builder_Qwen2.5-0.5B_3level.yml -s /Data/ReasoningNLCP --mode both --max-samples 200 --ckpt /Data/ReasoningNLCP/EXPERIMENT/lcp/builder/GSM8K_Qwen2.5-0.5B_3level_AutoWeighted/checkpoints/checkpoint_best_eval-step5000.pt -o /tmp/eval_out -v 1

D. Reuse training output tree (no -o): writes to
   ``<log_path>/eval_builder/<mode>/sample_<id>/`` plus ``eval.log`` and
   ``eval_summary.json`` at that root. For the AutoWeighted 3-level
   config above, that resolves to
   ``/Data/ReasoningNLCP/EXPERIMENT/lcp/builder/GSM8K_Qwen2.5-0.5B_3level_AutoWeighted/logs/eval_builder/<mode>/``.

================================================================================
  OUTPUT LAYOUT
================================================================================

``<output_root>/``
    eval.log                       (CLI logging mirror)
    eval_summary.json              (avg losses, accuracies, artifact dict)
    sample_<safe_main_id>/
        input.json                 main_id, question, cot_answer, solution
        pyramid.pt                 PyramidOutput tensors
                                   (skipped when intermediate_vector_save=0)
        reasoning.json             mode + teacher-forced / free-generation texts
                                   + groundtruth_solution + per-sample accuracy
        timing.json                pyramid_ms, reasoning_gen_ms, total_ms
        losses.json                per-sample loss decomposition

Artifact toggles are driven by the ``BuilderEvalArtifacts`` dataclass,
built from ``evaluation.intermediate_vector_save`` in the YAML and
optionally overridden by the CLI flag ``-v / --intermediate-vector-save``.
"""

import argparse
import datetime
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

import swanlab
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

from lmbase.utils.env_tools import get_device
from lcp.concept_builder import ConceptPyramidBuilder
from lcp.data_loader import BuilderInput, LCPDataLoader
from lcp.losses import compute_builder_loss
from ram.utils import apply_storage_root, load_config

logger = logging.getLogger(__name__)


# =============================================================================
# Mode constants
# =============================================================================

MODE_TEACHER_FORCED = "teacher_forced"
MODE_FREE_GENERATION = "free_generation"
MODE_BOTH = "both"
VALID_MODES = (MODE_TEACHER_FORCED, MODE_FREE_GENERATION, MODE_BOTH)


def _mode_runs_teacher_forced(mode: str) -> bool:
    """Return True when ``mode`` requires the teacher-forced forward pass."""
    return mode in (MODE_TEACHER_FORCED, MODE_BOTH)


def _mode_runs_free_generation(mode: str) -> bool:
    """Return True when ``mode`` requires the free-generation decode."""
    return mode in (MODE_FREE_GENERATION, MODE_BOTH)


# =============================================================================
# Per-sample artifact toggle dataclass (shared with eval_predictor.py)
# =============================================================================


@dataclass
class BuilderEvalArtifacts:
    """Per-sample dump toggles for Builder eval.

    Each boolean field corresponds to one artifact written under
    ``sample_<safe_main_id>/`` by ``_dump_builder_sample``:

        input               -> input.json
        reasoning           -> reasoning.json
        timing              -> timing.json
        losses              -> losses.json
        intermediate_vectors-> pyramid.pt (large tensor file)

    The Predictor variant (``PredictorEvalArtifacts`` in
    ``eval_predictor.py``) mirrors this layout so both CLIs share a
    uniform on/off vocabulary. Pass ``artifacts=None`` to
    ``evaluate_builder`` to disable per-sample dumping entirely (used
    by training-time eval).
    """

    input: bool = True
    reasoning: bool = True
    timing: bool = True
    losses: bool = True
    intermediate_vectors: bool = True

    @classmethod
    def from_config(cls, eval_cfg: dict) -> "BuilderEvalArtifacts":
        """Build from the ``evaluation`` block of a YAML config.

        Reads ``evaluation.intermediate_vector_save`` (0 or 1) and
        leaves all other artifact flags at their defaults (on).
        """
        save_vec = int(eval_cfg["intermediate_vector_save"])
        return cls(intermediate_vectors=bool(save_vec))

    def any_enabled(self) -> bool:
        """True when at least one artifact will be written."""
        return any(getattr(self, f.name) for f in fields(self))


# =============================================================================
# Reasoning accuracy utilities
# =============================================================================


def _extract_final_number(text: str) -> Optional[str]:
    """Extract the final numerical answer from a generated text.

    Tries three strategies in order:
      1. GSM8K ``#### <number>`` marker.
      2. LaTeX ``\\boxed{<answer>}`` marker (MATH).
      3. Last number-like token in the string.

    Returns the extracted answer (stripped) or None when no number can
    be recovered.
    """
    if not text or not text.strip():
        return None

    # Strategy 1: GSM8K answer marker.
    m = re.search(r"####\s*(.+)", text)
    if m:
        return m.group(1).strip()

    # Strategy 2: LaTeX boxed answer.
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()

    # Strategy 3: fallback to last numeric substring.
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()

    return None


def _normalize_answer(answer: str) -> str:
    """Normalize an answer string for equality comparison.

    Handles thousands separators, trailing ``.0`` on integers, and
    converts ``\\frac{a}{b}`` to ``a/b``.
    """
    s = answer.strip()
    s = s.replace(",", "")
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", s)
    if re.match(r"^-?\d+\.0+$", s):
        s = s.split(".")[0]
    return s


def compute_reasoning_accuracy(
    reasoning_texts: list[str],
    solutions: list[str],
) -> dict:
    """Compute exact-match accuracy between predicted texts and GT solutions.

    Args:
        reasoning_texts: Decoded strings from the model. Length N.
        solutions: Ground-truth answer strings from the dataset. Length
            N. Entries may be None when no solution was available.

    Returns:
        Dict with keys ``accuracy`` (float in [0, 1]), ``num_correct``,
        ``num_total``, and ``num_extracted``.
    """
    if not reasoning_texts or not solutions:
        return {"accuracy": 0.0, "num_correct": 0, "num_total": 0, "num_extracted": 0}

    num_correct = 0
    num_total = 0
    num_extracted = 0

    for pred_text, gt_sol in zip(reasoning_texts, solutions):
        if gt_sol is None:
            continue
        num_total += 1

        extracted = _extract_final_number(pred_text)
        if extracted is None:
            continue
        num_extracted += 1

        if _normalize_answer(extracted) == _normalize_answer(gt_sol):
            num_correct += 1

    accuracy = num_correct / num_total if num_total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "num_correct": num_correct,
        "num_total": num_total,
        "num_extracted": num_extracted,
    }


# =============================================================================
# Terminal / file logging utilities
# =============================================================================


def log_terminal_entry(log_path: Path, entry: dict) -> None:
    """Append a JSON line to the terminal output log file.

    Each entry gets a timestamp and is written immediately so partial
    progress survives a crash.
    """
    entry["timestamp"] = datetime.datetime.now().isoformat()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def log_eval_results(
    eval_losses,
    loss_weights,
    eval_type,
    global_step,
    terminal_log_path,
    eval_history,
    log_dir,
    swanlab_prefix,
    reasoning_texts_dict,
    eval_samples,
    eval_sample_history,
) -> None:
    """Emit eval results to console, SwanLab, terminal log, and history JSON.

    The function is a pure logger; it does not dump per-sample folders
    (that is ``_dump_builder_sample``'s job inside ``evaluate_builder``).
    """
    ew = {
        "recon": eval_losses["recon"] * loss_weights["recon_loss_weight"],
        "ordering": eval_losses["ordering"] * loss_weights["ordering_loss_weight"],
        "residual": eval_losses["residual"] * loss_weights["residual_loss_weight"],
    }
    reasoning_part = ""
    if "reasoning" in eval_losses:
        ew["reasoning"] = (
            eval_losses["reasoning"] * loss_weights["reasoning_loss_weight"]
        )
        reasoning_part = " reasoning=%.4f/%.4f" % (
            eval_losses["reasoning"],
            ew["reasoning"],
        )
    label = "eval(quick)" if eval_type == "quick" else "eval(full) "
    logger.info(
        "  %s | total=%.4f recon=%.4f/%.4f ordering=%.4f/%.4f residual=%.4f/%.4f%s",
        label,
        eval_losses["total"],
        eval_losses["recon"],
        ew["recon"],
        eval_losses["ordering"],
        ew["ordering"],
        eval_losses["residual"],
        ew["residual"],
        reasoning_part,
    )

    # SwanLab metrics.
    metrics = {
        f"{swanlab_prefix}/total_loss": eval_losses["total"],
        f"{swanlab_prefix}/recon_raw": eval_losses["recon"],
        f"{swanlab_prefix}/recon_weighted": ew["recon"],
        f"{swanlab_prefix}/ordering_raw": eval_losses["ordering"],
        f"{swanlab_prefix}/ordering_weighted": ew["ordering"],
        f"{swanlab_prefix}/residual_raw": eval_losses["residual"],
        f"{swanlab_prefix}/residual_weighted": ew["residual"],
    }
    if "reasoning" in eval_losses:
        metrics[f"{swanlab_prefix}/reasoning_raw"] = eval_losses["reasoning"]
        metrics[f"{swanlab_prefix}/reasoning_weighted"] = ew["reasoning"]
    swanlab.log(metrics, step=global_step)

    # Terminal log row.
    term_data = {
        "step": global_step,
        "eval_type": eval_type,
        **{f"eval_{k}": round(v, 6) for k, v in eval_losses.items() if k != "_timing"},
        **{f"eval_{k}_w": round(v, 6) for k, v in ew.items()},
    }
    log_terminal_entry(terminal_log_path, term_data)

    # Eval history (crash-safe rewrite per eval).
    eval_history.append(
        {
            "step": global_step,
            "eval_type": eval_type,
            **eval_losses,
            **{f"{k}_w": v for k, v in ew.items()},
        }
    )
    with open(log_dir / "eval_history.json", "w", encoding="utf-8") as f:
        json.dump(eval_history, f, indent=2, default=str)

    # Reasoning decoded texts (append per eval, per text type).
    for text_type, texts in reasoning_texts_dict.items():
        if texts:
            entry = {
                "step": global_step,
                "eval_type": eval_type,
                "type": text_type,
                "texts": texts,
            }
            with open(
                log_dir / "eval_reasoning_texts.jsonl", "a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(entry, default=str) + "\n")

    # Sample history: one record per eval invocation.
    eval_sample_history.append(
        {
            "step": global_step,
            "eval_type": eval_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "num_samples": len(eval_samples),
            "samples": eval_samples,
        }
    )
    with open(log_dir / "eval_sample_history.json", "w", encoding="utf-8") as f:
        json.dump(eval_sample_history, f, indent=2, default=str)


# =============================================================================
# Per-sample folder dumping
# =============================================================================


_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_main_id(main_id: str) -> str:
    """Collapse filesystem-unsafe characters in ``main_id`` to underscores.

    Used to turn the dataset's ``main_id`` (which may contain slashes,
    spaces, or non-printables) into a safe directory name.
    """
    safe = _SAFE_ID_RE.sub("_", str(main_id))
    return safe if safe else "unknown"


def _tensor_to_cpu(value):
    """Detach and move a tensor (or list of tensors) to CPU for saving."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return [_tensor_to_cpu(v) for v in value]
    return value


def _pyramid_to_dump_dict(pyramid) -> dict:
    """Extract a serializable dict of tensors from a ``PyramidOutput``.

    Shapes (single sample, batch_size=1):
        concepts[k]:            [1, L_k, D]
        attention_weights[k]:   [1, L_k, L]
        reconstruction[k]:      [1, L, D]
        encoder_hidden_states:  [1, L, D_encoder]
        projected_hidden:       [1, L, D]
        reconstructed_hidden:   [1, L, D]
        residual_hidden:        [1, L, D]
    """
    return {
        "concepts": [_tensor_to_cpu(c) for c in pyramid.concepts],
        "attention_weights": [
            _tensor_to_cpu(lo.attention_weights) for lo in pyramid.level_outputs
        ],
        "reconstruction": [
            _tensor_to_cpu(lo.reconstruction) for lo in pyramid.level_outputs
        ],
        "encoder_hidden_states": _tensor_to_cpu(pyramid.encoder_hidden_states),
        "projected_hidden": _tensor_to_cpu(pyramid.projected_hidden),
        "reconstructed_hidden": _tensor_to_cpu(pyramid.reconstructed_hidden),
        "residual_hidden": _tensor_to_cpu(pyramid.residual_hidden),
        "attention_mask": _tensor_to_cpu(pyramid.attention_mask),
    }


def _loss_dict_to_json(loss_dict: dict) -> dict:
    """Convert any torch scalars/lists inside a loss dict to plain Python."""
    out = {}
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            out[key] = float(value.item())
        elif isinstance(value, list):
            out[key] = [float(v) if hasattr(v, "item") else v for v in value]
        else:
            out[key] = value
    return out


def _dump_builder_sample(
    sample_dir: Path,
    main_id: str,
    question: str,
    cot_answer: str,
    solution: Optional[str],
    pyramid,
    mode: str,
    timing: dict,
    losses: dict,
    reasoning_tf_text: Optional[str],
    reasoning_free_text: Optional[str],
    reasoning_accuracy: Optional[dict],
    artifacts: BuilderEvalArtifacts,
) -> None:
    """Write a self-contained folder for one Builder eval sample.

    Args:
        sample_dir: Target folder path. Will be created if missing.
        main_id: Raw dataset main_id.
        question, cot_answer, solution: Source row strings.
        pyramid: ``PyramidOutput`` for this single sample.
        mode: One of ``VALID_MODES``.
        timing: Dict of stage durations in ms.
        losses: Per-sample loss decomposition.
        reasoning_tf_text: Teacher-forced decoded text, or None.
        reasoning_free_text: Free-generation decoded text, or None.
        reasoning_accuracy: Optional dict from ``compute_reasoning_accuracy``.
        artifacts: Per-file toggle dataclass. Files whose flag is False
            are skipped; ``intermediate_vectors=False`` skips the
            pyramid.pt tensor dump (the only large file).
    """
    sample_dir.mkdir(parents=True, exist_ok=True)

    if artifacts.input:
        input_payload = {
            "main_id": main_id,
            "question": question,
            "cot_answer": cot_answer,
            "solution": solution,
        }
        with open(sample_dir / "input.json", "w", encoding="utf-8") as f:
            json.dump(input_payload, f, indent=2, ensure_ascii=False)

    if artifacts.intermediate_vectors:
        torch.save(_pyramid_to_dump_dict(pyramid), sample_dir / "pyramid.pt")

    if artifacts.reasoning:
        reasoning_payload = {
            "mode": mode,
            "teacher_forced_text": reasoning_tf_text,
            "free_generation_text": reasoning_free_text,
            "groundtruth_solution": solution,
            "reasoning_accuracy": reasoning_accuracy,
        }
        with open(sample_dir / "reasoning.json", "w", encoding="utf-8") as f:
            json.dump(reasoning_payload, f, indent=2, ensure_ascii=False)

    if artifacts.timing:
        with open(sample_dir / "timing.json", "w", encoding="utf-8") as f:
            json.dump(timing, f, indent=2)

    if artifacts.losses:
        with open(sample_dir / "losses.json", "w", encoding="utf-8") as f:
            json.dump(_loss_dict_to_json(losses), f, indent=2)


# =============================================================================
# Builder evaluation loop
# =============================================================================


@torch.no_grad()
def evaluate_builder(
    builder: ConceptPyramidBuilder,
    eval_dataloader: LCPDataLoader,
    loss_weights: dict,
    ordering_loss_type: str,
    max_batches: int,
    mode: str,
    generation_max_tokens: int,
    output_root: Optional[Path],
    dump_artifacts: bool,
    artifacts: Optional[BuilderEvalArtifacts] = None,
) -> tuple[dict, dict[str, list[str]], list[dict]]:
    """Run Builder evaluation over the eval dataloader.

    Args:
        builder: Builder module to evaluate.
        eval_dataloader: Yields ``BuilderInput`` batches (expected
            ``batch_size=1`` for per-sample dumps).
        loss_weights: Loss weight configuration.
        ordering_loss_type: ``"margin" | "gaussian" | "both"``.
        max_batches: Max batches to consume (0 = all).
        mode: One of ``VALID_MODES``. Controls which reasoning path(s)
            are exercised per sample.
        generation_max_tokens: Max new tokens for free generation.
        output_root: Folder under which ``sample_<id>/`` directories are
            written when ``dump_artifacts`` is True. Ignored otherwise.
        dump_artifacts: Master switch for per-sample folder writes. The
            train loop passes False to avoid disk churn every N steps.
        artifacts: Optional fine-grained per-file toggle. When None and
            ``dump_artifacts`` is True, defaults to all-on (legacy
            behaviour). When ``dump_artifacts`` is False this is
            ignored.

    Returns:
        Tuple ``(averaged_loss_dict, reasoning_texts_dict, samples)``.
        - ``averaged_loss_dict`` contains ``total, recon, ordering,
          residual`` plus ``reasoning`` when solutions are present and a
          ``_timing`` sub-dict.
        - ``reasoning_texts_dict`` has keys ``"teacher_forced"`` and
          ``"generation"``, each a flat list of decoded strings.
        - ``samples`` is the per-row metadata list used by
          ``eval_sample_history.json``.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode={mode!r}. Expected one of {VALID_MODES}.")
    if dump_artifacts and output_root is None:
        raise ValueError("dump_artifacts=True requires output_root to be set.")
    if dump_artifacts and artifacts is None:
        artifacts = BuilderEvalArtifacts()

    builder.eval()
    all_losses: list[dict] = []
    all_texts_tf: list[str] = []
    all_texts_gen: list[str] = []
    all_samples: list[dict] = []
    batch_times_ms: list[float] = []

    want_tf = _mode_runs_teacher_forced(mode)
    want_free = _mode_runs_free_generation(mode)

    if dump_artifacts:
        output_root.mkdir(parents=True, exist_ok=True)

    eval_start = time.perf_counter()
    for i, batch in enumerate(eval_dataloader):
        if max_batches > 0 and i >= max_batches:
            break

        # Forward pass: batch -> pyramid (encode + build + reasoning).
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        pyramid = builder(batch)
        _, loss_dict = compute_builder_loss(
            pyramid, loss_weights, ordering_loss_type=ordering_loss_type
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        pyramid_ms = (time.perf_counter() - t0) * 1000.0
        batch_times_ms.append(pyramid_ms)

        all_losses.append(loss_dict)

        # Teacher-forced reasoning text (already produced by builder.forward
        # when solutions exist, regardless of mode). We only record it when
        # the requested mode includes teacher-forced.
        tf_text = None
        if want_tf and pyramid.reasoning_texts is not None:
            tf_text = pyramid.reasoning_texts[0]
            all_texts_tf.append(tf_text)

        # Free generation: decode solution from [Q, Concepts] only.
        free_text = None
        reasoning_gen_ms = 0.0
        if want_free and batch.has_solution:
            device = next(builder.parameters()).device
            max_length = builder.pyramid_cfg["max_seq_len"]
            q_tokens = builder.tokenizer(
                batch.questions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            q_ids = q_tokens["input_ids"].to(device)
            q_mask = q_tokens["attention_mask"].to(device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_gen = time.perf_counter()

            gen_texts = builder.generate_solution(
                pyramid,
                q_ids,
                q_mask,
                max_new_tokens=generation_max_tokens,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            reasoning_gen_ms = (time.perf_counter() - t_gen) * 1000.0

            free_text = gen_texts[0] if gen_texts else None
            if free_text is not None:
                all_texts_gen.append(free_text)

        # Per-sample metadata.
        for j in range(batch.batch_size):
            all_samples.append(
                {
                    "batch_idx": i,
                    "pos_in_batch": j,
                    "main_id": batch.main_ids[j],
                    "question": batch.questions[j],
                    "solution": batch.solutions[j] if batch.has_solution else None,
                }
            )

        # Per-sample folder dump (batch_size=1 contract: first and only row).
        if dump_artifacts:
            main_id = batch.main_ids[0]
            sample_dir = output_root / f"sample_{_safe_main_id(main_id)}"
            reasoning_acc = None
            if tf_text is not None and batch.has_solution:
                reasoning_acc = compute_reasoning_accuracy(
                    [tf_text], [batch.solutions[0]]
                )
            timing = {
                "pyramid_ms": round(pyramid_ms, 3),
                "reasoning_gen_ms": round(reasoning_gen_ms, 3),
                "total_ms": round(pyramid_ms + reasoning_gen_ms, 3),
            }
            _dump_builder_sample(
                sample_dir=sample_dir,
                main_id=main_id,
                question=batch.questions[0],
                cot_answer=batch.cot_answers[0],
                solution=batch.solutions[0] if batch.has_solution else None,
                pyramid=pyramid,
                mode=mode,
                timing=timing,
                losses=loss_dict,
                reasoning_tf_text=tf_text,
                reasoning_free_text=free_text,
                reasoning_accuracy=reasoning_acc,
                artifacts=artifacts,
            )

    eval_elapsed_s = time.perf_counter() - eval_start
    builder.train()

    if not all_losses:
        return (
            {"total": 0.0, "recon": 0.0, "ordering": 0.0, "residual": 0.0},
            {"teacher_forced": [], "generation": []},
            [],
        )

    # Average scalar keys. Reasoning is present iff solutions are, so
    # keys are consistent across batches in a single invocation.
    avg = {}
    keys = all_losses[0].keys()
    for k in keys:
        avg[k] = sum(d[k] for d in all_losses) / len(all_losses)

    num_batches = len(batch_times_ms)
    avg["_timing"] = {
        "eval_total_s": round(eval_elapsed_s, 3),
        "num_batches": num_batches,
        "batch_mean_ms": round(sum(batch_times_ms) / num_batches, 2),
        "batch_min_ms": round(min(batch_times_ms), 2),
        "batch_max_ms": round(max(batch_times_ms), 2),
    }

    texts_dict = {
        "teacher_forced": all_texts_tf,
        "generation": all_texts_gen,
    }

    return avg, texts_dict, all_samples


# =============================================================================
# Standalone CLI: checkpoint resolution
# =============================================================================


def _extract_step(filename: str) -> int:
    """Extract ``step`` number from a checkpoint filename (``0`` on miss)."""
    m = re.search(r"-step(\d+)", filename)
    return int(m.group(1)) if m else 0


def _resolve_checkpoint_path(raw: str, storage_root: str) -> Path:
    """Resolve a checkpoint path with prefix-glob fallback.

    If the literal path does not exist, the function globs
    ``<stem>*.pt`` in the parent directory and returns the match with
    the highest step number. Mirrors the training-time resolution so a
    single YAML value (e.g. ``checkpoint_best_eval.pt``) works across
    runs without manual epoch/step edits.
    """
    p = Path(raw)
    resolved = p if p.is_absolute() else Path(storage_root) / p

    if resolved.is_file():
        return resolved

    parent = resolved.parent
    stem_prefix = resolved.stem
    if parent.is_dir():
        candidates = sorted(
            parent.glob(f"{stem_prefix}*.pt"),
            key=lambda f: _extract_step(f.name),
            reverse=True,
        )
        if candidates:
            return candidates[0]
    return resolved


def _find_best_builder_checkpoint(checkpoint_dir: Path) -> Path:
    """Pick the best available builder checkpoint in ``checkpoint_dir``.

    Preference order: ``checkpoint_best_eval*.pt`` >
    ``checkpoint_best*.pt`` > any ``checkpoint*.pt``. Highest step wins
    within each tier.
    """
    for pattern in (
        "checkpoint_best_eval*.pt",
        "checkpoint_best*.pt",
        "checkpoint*.pt",
    ):
        candidates = sorted(
            checkpoint_dir.glob(pattern),
            key=lambda f: _extract_step(f.name),
            reverse=True,
        )
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"No builder checkpoint found in {checkpoint_dir}.")


def _load_builder(
    builder_config: dict,
    checkpoint_path: Path,
    device: str,
    cli_logger: logging.Logger,
) -> ConceptPyramidBuilder:
    """Instantiate the Builder and load ``checkpoint_path`` into it.

    Unlike ``train_predictor._load_frozen_builder``, this keeps the
    module in ``eval()`` mode but does not force ``requires_grad=False``
    on every parameter because the CLI caller does not train it.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Builder checkpoint not found: {checkpoint_path}")
    builder = ConceptPyramidBuilder(builder_config)
    builder.to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state_dict"]
    missing, unexpected = builder.load_state_dict(state, strict=False)
    if missing or unexpected:
        cli_logger.warning(
            "Builder loaded with strict=False | missing=%d unexpected=%d",
            len(missing),
            len(unexpected),
        )
    builder.eval()
    cli_logger.info(
        "Builder loaded (epoch=%s step=%s) from %s",
        ckpt.get("epoch", "?"),
        ckpt.get("step", "?"),
        checkpoint_path,
    )
    return builder


# =============================================================================
# Standalone CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """CLI flags for standalone Builder evaluation."""
    parser = argparse.ArgumentParser(description="lcp Builder Evaluation")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to a builder YAML config.",
    )
    parser.add_argument(
        "-s",
        "--storage-root",
        type=str,
        default="/Data/ReasoningNLCP",
        help=(
            "Prefix prepended to RELATIVE log/checkpoint paths in the "
            "config. MUST match the value used at training time. "
            "Default: /Data/ReasoningNLCP"
        ),
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help=(
            "Explicit builder checkpoint path. When set, OVERRIDES "
            "auto-discovery from config's log.checkpoint_path."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=list(VALID_MODES),
        help="Which reasoning path(s) to exercise per sample.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        required=True,
        help="Max number of eval samples to process.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="",
        help=("Output directory override. Default: " "<log_path>/eval_builder/<mode>/"),
    )
    parser.add_argument(
        "-v",
        "--intermediate-vector-save",
        type=int,
        choices=[0, 1],
        default=None,
        help=(
            "Override evaluation.intermediate_vector_save from config. "
            "0 = skip pyramid.pt vector dump, 1 = save it. "
            "When omitted, the YAML value is used."
        ),
    )
    return parser.parse_args()


def _build_eval_dataloader(config: dict) -> LCPDataLoader:
    """Construct the eval-split dataloader with ``batch_size=1``.

    batch_size is hard-coded here because the per-sample folder layout
    assumes exactly one sample per iteration; see the plan's output
    spec for details.
    """
    eval_cfg = config["evaluation"]
    eval_data_cfg = eval_cfg["data"]
    return LCPDataLoader(
        data_cfg=eval_data_cfg,
        batch_size=1,
        include_solution=True,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )


def main() -> None:
    """Entry point for ``python examples/lcp/eval_builder.py``."""
    args = parse_args()

    # Resolve config.
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_config(str(config_path))
    apply_storage_root(config, args.storage_root)

    # Resolve output root: <log_path>/eval_builder/<mode>/ unless overridden.
    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        output_root = Path(config["log"]["log_path"]) / "eval_builder" / args.mode
    output_root.mkdir(parents=True, exist_ok=True)

    # Logging.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(output_root / "eval.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    cli_logger = logging.getLogger("eval_builder")

    cli_logger.info("=" * 72)
    cli_logger.info("  lcp Builder Evaluation")
    cli_logger.info("=" * 72)
    cli_logger.info("Config      : %s", config_path)
    cli_logger.info("Storage root: %s", args.storage_root)
    cli_logger.info("Mode        : %s", args.mode)
    cli_logger.info("Output root : %s", output_root)

    # Resolve checkpoint.
    if args.ckpt:
        ckpt_path = Path(args.ckpt).resolve()
        cli_logger.info("[OVERRIDE] Builder checkpoint (--ckpt): %s", ckpt_path)
    else:
        checkpoint_dir = Path(config["log"]["checkpoint_path"])
        ckpt_path = _find_best_builder_checkpoint(checkpoint_dir)
        cli_logger.info("[AUTO]     Builder checkpoint: %s", ckpt_path)

    # Device + model.
    device = str(get_device("auto"))
    cli_logger.info("Device      : %s", device)
    builder = _load_builder(config, ckpt_path, device, cli_logger)

    # Dataloader.
    dataloader = _build_eval_dataloader(config)

    # Loss config.
    loss_weights = config["training"]["loss_weights"]
    ordering_loss_type = config["training"]["ordering_loss_type"]
    generation_max_tokens = config["evaluation"]["generation_max_tokens"]

    # Per-sample artifact toggles (config + optional CLI override).
    artifacts = BuilderEvalArtifacts.from_config(config["evaluation"])
    if args.intermediate_vector_save is not None:
        artifacts.intermediate_vectors = bool(args.intermediate_vector_save)
    cli_logger.info("Artifacts   : %s", asdict(artifacts))

    # Mode requires solutions?
    if _mode_runs_teacher_forced(args.mode) or _mode_runs_free_generation(args.mode):
        # Both paths benefit from solutions; free generation is skipped
        # silently when solutions are absent, but teacher-forced text is
        # only produced when builder.forward sees solutions. We simply
        # trust the config's eval split to include solutions.
        pass

    cli_logger.info("")
    cli_logger.info("Starting evaluation (max_samples=%d)...", args.max_samples)
    cli_logger.info("-" * 72)

    run_start = time.perf_counter()
    avg_losses, reasoning_texts, samples = evaluate_builder(
        builder=builder,
        eval_dataloader=dataloader,
        loss_weights=loss_weights,
        ordering_loss_type=ordering_loss_type,
        max_batches=args.max_samples,
        mode=args.mode,
        generation_max_tokens=generation_max_tokens,
        output_root=output_root,
        dump_artifacts=True,
        artifacts=artifacts,
    )
    total_elapsed = time.perf_counter() - run_start

    # Aggregate reasoning accuracy over the produced texts.
    solution_list = [s["solution"] for s in samples]
    acc_tf = compute_reasoning_accuracy(
        reasoning_texts["teacher_forced"], solution_list
    )
    acc_gen = compute_reasoning_accuracy(reasoning_texts["generation"], solution_list)

    # Summary.
    cli_logger.info("")
    cli_logger.info("=" * 72)
    cli_logger.info("  EVALUATION COMPLETE")
    cli_logger.info("=" * 72)
    cli_logger.info("Samples processed     : %d", len(samples))
    cli_logger.info("Total wall-clock (s)  : %.2f", total_elapsed)
    cli_logger.info("Avg total loss        : %.4f", avg_losses["total"])
    if "reasoning" in avg_losses:
        cli_logger.info("Avg reasoning loss    : %.4f", avg_losses["reasoning"])
    if acc_tf["num_total"] > 0:
        cli_logger.info(
            "Teacher-forced acc    : %.4f (%d/%d)",
            acc_tf["accuracy"],
            acc_tf["num_correct"],
            acc_tf["num_total"],
        )
    if acc_gen["num_total"] > 0:
        cli_logger.info(
            "Free-generation acc   : %.4f (%d/%d)",
            acc_gen["accuracy"],
            acc_gen["num_correct"],
            acc_gen["num_total"],
        )

    # eval_summary.json at the eval root.
    summary = {
        "config_path": str(config_path),
        "builder_checkpoint": str(ckpt_path),
        "mode": args.mode,
        "device": device,
        "num_samples": len(samples),
        "total_time_s": round(total_elapsed, 3),
        "artifacts": asdict(artifacts),
        "avg_losses": _loss_dict_to_json(avg_losses),
        "reasoning_accuracy": {
            "teacher_forced": acc_tf,
            "free_generation": acc_gen,
        },
        "main_ids": [s["main_id"] for s in samples],
    }
    with open(output_root / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
    cli_logger.info("Summary saved to: %s", output_root / "eval_summary.json")
    cli_logger.info("Done.")


if __name__ == "__main__":
    main()
