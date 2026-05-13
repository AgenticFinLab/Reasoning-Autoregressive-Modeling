"""NLCP V4 Predictor evaluation module and standalone CLI.

This module owns everything Predictor-specific:
  - Library entry point ``evaluate_predictor`` used by both training-time
    eval (called from ``train_predictor.py``) and the standalone CLI
    below.
  - Low-level per-step helpers (``_strip_solutions``, ``_tokenize_qs``,
    ``_run_predictor_step``) that the predictor loop uses and that a
    few analysis scripts import directly.
  - Predictor-flavoured logger (``log_eval_results_predictor``).
  - Per-sample artifact dumper (``_dump_predictor_sample``) that writes
    a self-contained folder per eval sample.
  - ``main()`` entry point so the module can be invoked directly:
        python examples/nlcpV4/eval_predictor.py -c <config> -s <root> --mode <mode>

Mode constants and generic helpers (``_safe_main_id``, ``_tensor_to_cpu``,
``_loss_dict_to_json``, ``log_terminal_entry``, reasoning accuracy) are
imported from ``eval_builder`` to keep a single source of truth.

Per-sample folder layout (written under
``<output_root>/sample_<safe_main_id>/``):

    input.json        # main_id, question, cot_answer, solution
    concepts.pt       # predicted_concepts + gt_concepts (lists of K tensors)
    reasoning.json    # mode + teacher-forced / free-generation texts
    timing.json       # per-stage ms
    losses.json       # per-sample loss decomposition

The builder counterpart lives in ``eval_builder.py``; neither file
imports predictor logic from the other.
"""

import argparse
import datetime
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional

import swanlab
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

from lmbase.utils.env_tools import get_device
from nlcpV4.concept_builder import ConceptPyramidBuilder
from nlcpV4.concept_predictor import ConceptPredictor
from nlcpV4.data_loader import BuilderInput, NLCPV4DataLoader
from nlcpV4.eval_builder import (
    MODE_BOTH,
    MODE_FREE_GENERATION,
    MODE_TEACHER_FORCED,
    VALID_MODES,
    _loss_dict_to_json,
    _mode_runs_free_generation,
    _mode_runs_teacher_forced,
    _safe_main_id,
    _tensor_to_cpu,
    compute_reasoning_accuracy,
    log_terminal_entry,
)
from nlcpV4.losses import compute_predictor_loss
from ram.utils import apply_storage_root, load_config

logger = logging.getLogger(__name__)


# =============================================================================
# Low-level per-step helpers (also consumed by analysis scripts)
# =============================================================================


def _strip_solutions(batch: BuilderInput) -> BuilderInput:
    """Return a clone of ``batch`` with ``solutions=[]``.

    The frozen Builder's ``forward`` branches on whether solutions are
    present; the predictor only needs concepts from that pass, so we
    strip solutions to skip the Builder's reasoning branch entirely.
    """
    return BuilderInput(
        questions=list(batch.questions),
        cot_answers=list(batch.cot_answers),
        solutions=[],
        main_ids=list(batch.main_ids),
    )


def _tokenize_qs(
    builder: ConceptPyramidBuilder,
    batch: BuilderInput,
    max_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize questions and solutions with the Builder's tokenizer.

    Returns ``(q_ids, q_mask, s_ids, s_mask)`` all on ``device``.
    """
    tokenizer = builder.tokenizer
    q = tokenizer(
        batch.questions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    s = tokenizer(
        batch.solutions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return (
        q["input_ids"].to(device),
        q["attention_mask"].to(device),
        s["input_ids"].to(device),
        s["attention_mask"].to(device),
    )


def _run_predictor_step(
    predictor: ConceptPredictor,
    builder: ConceptPyramidBuilder,
    batch: BuilderInput,
    max_length: int,
    device: str,
):
    """Build gt_concepts (frozen builder), tokenize, call predictor.

    Returns the full PredictorOutput; the caller computes the loss.
    """
    with torch.no_grad():
        pyramid = builder(_strip_solutions(batch))
        gt_concepts = [c.detach() for c in pyramid.concepts]

    q_ids, q_mask, s_ids, s_mask = _tokenize_qs(builder, batch, max_length, device)

    output = predictor(
        question_ids=q_ids,
        question_attention_mask=q_mask,
        gt_concepts=gt_concepts,
        solution_ids=s_ids,
        solution_attention_mask=s_mask,
    )
    return output


# =============================================================================
# Per-sample folder dumping
# =============================================================================


def _concepts_to_dump_dict(output) -> dict:
    """Extract serializable concept tensors from a ``PredictorOutput``.

    Shapes (batch_size=1):
        predicted_concepts[k]: [1, L_k, D]
        gt_concepts[k]:        [1, L_k, D]
    """
    gt = output.gt_concepts
    return {
        "predicted_concepts": [_tensor_to_cpu(c) for c in output.predicted_concepts],
        "gt_concepts": [_tensor_to_cpu(c) for c in gt] if gt is not None else None,
        "level_lengths": list(output.level_lengths),
        "num_levels": output.num_levels,
    }


def _dump_predictor_sample(
    sample_dir: Path,
    main_id: str,
    question: str,
    cot_answer: str,
    solution: Optional[str],
    output,
    mode: str,
    timing: dict,
    losses: dict,
    reasoning_tf_text: Optional[str],
    reasoning_free_text: Optional[str],
    reasoning_accuracy: Optional[dict],
) -> None:
    """Write a self-contained folder for one Predictor eval sample.

    Args:
        sample_dir: Target folder path. Will be created if missing.
        main_id: Raw dataset main_id.
        question, cot_answer, solution: Source row strings.
        output: ``PredictorOutput`` for this single sample.
        mode: One of ``VALID_MODES``.
        timing: Dict of stage durations in ms.
        losses: Per-sample loss decomposition.
        reasoning_tf_text: Teacher-forced decoded text, or None.
        reasoning_free_text: Free-generation decoded text, or None.
        reasoning_accuracy: Optional dict from ``compute_reasoning_accuracy``.
    """
    sample_dir.mkdir(parents=True, exist_ok=True)

    input_payload = {
        "main_id": main_id,
        "question": question,
        "cot_answer": cot_answer,
        "solution": solution,
    }
    with open(sample_dir / "input.json", "w", encoding="utf-8") as f:
        json.dump(input_payload, f, indent=2, ensure_ascii=False)

    torch.save(_concepts_to_dump_dict(output), sample_dir / "concepts.pt")

    reasoning_payload = {
        "mode": mode,
        "teacher_forced_text": reasoning_tf_text,
        "free_generation_text": reasoning_free_text,
        "groundtruth_solution": solution,
        "reasoning_accuracy": reasoning_accuracy,
    }
    with open(sample_dir / "reasoning.json", "w", encoding="utf-8") as f:
        json.dump(reasoning_payload, f, indent=2, ensure_ascii=False)

    with open(sample_dir / "timing.json", "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)

    with open(sample_dir / "losses.json", "w", encoding="utf-8") as f:
        json.dump(_loss_dict_to_json(losses), f, indent=2)


# =============================================================================
# Predictor evaluation loop
# =============================================================================


@torch.no_grad()
def evaluate_predictor(
    predictor: ConceptPredictor,
    builder: ConceptPyramidBuilder,
    eval_dataloader: NLCPV4DataLoader,
    loss_weights: dict,
    max_length: int,
    device: str,
    max_batches: int,
    mode: str,
    generation_max_tokens: int,
    output_root: Optional[Path],
    dump_artifacts: bool,
) -> tuple[dict, dict[str, list[str]], list[dict]]:
    """Run Predictor evaluation over the eval dataloader.

    Args:
        predictor: Predictor module to evaluate.
        builder: Frozen Stage-1 builder providing gt_concepts.
        eval_dataloader: Yields ``BuilderInput`` batches (expected
            ``batch_size=1`` for per-sample dumps).
        loss_weights: Loss weight configuration.
        max_length: Tokenizer truncation length (typically
            ``pyramid.max_seq_len``).
        device: Target device string.
        max_batches: Max batches to consume (0 = all).
        mode: One of ``VALID_MODES``. Controls which reasoning path(s)
            are exercised per sample.
        generation_max_tokens: Max new tokens for free generation.
        output_root: Folder under which ``sample_<id>/`` directories are
            written when ``dump_artifacts`` is True. Ignored otherwise.
        dump_artifacts: Master switch for per-sample folder writes. The
            train loop passes False to avoid disk churn every N steps.

    Returns:
        Tuple ``(averaged_loss_dict, reasoning_texts_dict, samples)``.
        - ``averaged_loss_dict`` contains ``total, concept`` plus
          ``reasoning`` when solutions are present and
          ``concept_per_level`` (list[float]) and a ``_timing`` sub-dict.
        - ``reasoning_texts_dict`` has keys ``"teacher_forced"`` and
          ``"generation"``, each a flat list of decoded strings.
        - ``samples`` is the per-row metadata list used by
          ``eval_sample_history.json``.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode={mode!r}. Expected one of {VALID_MODES}.")
    if dump_artifacts and output_root is None:
        raise ValueError("dump_artifacts=True requires output_root to be set.")

    predictor.eval()
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

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        output = _run_predictor_step(predictor, builder, batch, max_length, device)
        _, loss_dict = compute_predictor_loss(
            output, loss_weights, concept_loss_type="mse"
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        predictor_forward_ms = (time.perf_counter() - t0) * 1000.0
        batch_times_ms.append(predictor_forward_ms)

        all_losses.append(loss_dict)

        tf_text = None
        if want_tf and output.reasoning_texts is not None:
            tf_text = output.reasoning_texts[0]
            all_texts_tf.append(tf_text)

        free_text = None
        reasoning_gen_ms = 0.0
        if want_free and batch.has_solution:
            q_ids, q_mask, _, _ = _tokenize_qs(builder, batch, max_length, device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_gen = time.perf_counter()

            gen_texts = predictor.generate_solution(
                output.predicted_concepts,
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

        if dump_artifacts:
            main_id = batch.main_ids[0]
            sample_dir = output_root / f"sample_{_safe_main_id(main_id)}"
            reasoning_acc = None
            if tf_text is not None and batch.has_solution:
                reasoning_acc = compute_reasoning_accuracy(
                    [tf_text], [batch.solutions[0]]
                )
            timing = {
                "predictor_forward_ms": round(predictor_forward_ms, 3),
                "reasoning_gen_ms": round(reasoning_gen_ms, 3),
                "total_ms": round(predictor_forward_ms + reasoning_gen_ms, 3),
            }
            _dump_predictor_sample(
                sample_dir=sample_dir,
                main_id=main_id,
                question=batch.questions[0],
                cot_answer=batch.cot_answers[0],
                solution=batch.solutions[0] if batch.has_solution else None,
                output=output,
                mode=mode,
                timing=timing,
                losses=loss_dict,
                reasoning_tf_text=tf_text,
                reasoning_free_text=free_text,
                reasoning_accuracy=reasoning_acc,
            )

    predictor.train()
    if hasattr(predictor, "reason_model"):
        predictor.reason_model.eval()

    eval_elapsed_s = time.perf_counter() - eval_start

    if not all_losses:
        return (
            {"total": 0.0, "concept": 0.0},
            {"teacher_forced": [], "generation": []},
            [],
        )

    scalar_keys = [k for k in all_losses[0].keys() if k != "concept_per_level"]
    avg = {k: sum(d[k] for d in all_losses) / len(all_losses) for k in scalar_keys}

    if "concept_per_level" in all_losses[0]:
        per_level = list(zip(*[d["concept_per_level"] for d in all_losses]))
        avg["concept_per_level"] = [sum(col) / len(col) for col in per_level]

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
# Predictor-flavoured eval logger
# =============================================================================


def log_eval_results_predictor(
    eval_losses: dict,
    loss_weights: dict,
    eval_type: str,
    global_step: int,
    terminal_log_path: Path,
    eval_history: list,
    log_dir: Path,
    swanlab_prefix: str,
    reasoning_texts_dict: dict[str, list[str]],
    eval_samples: list[dict],
    eval_sample_history: list,
) -> None:
    """Console + SwanLab + eval_history + sample-history writer for predictor eval.

    Mirrors ``log_eval_results`` in shape, but for the predictor's
    two-component loss schema (concept + reasoning).
    """
    w_concept = eval_losses["concept"] * loss_weights["concept_loss_weight"]
    ew = {"concept": w_concept}
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
        "  %s | total=%.4f concept=%.4f/%.4f%s",
        label,
        eval_losses["total"],
        eval_losses["concept"],
        ew["concept"],
        reasoning_part,
    )

    metrics = {
        f"{swanlab_prefix}/total_loss": eval_losses["total"],
        f"{swanlab_prefix}/concept_raw": eval_losses["concept"],
        f"{swanlab_prefix}/concept_weighted": ew["concept"],
    }
    if "reasoning" in eval_losses:
        metrics[f"{swanlab_prefix}/reasoning_raw"] = eval_losses["reasoning"]
        metrics[f"{swanlab_prefix}/reasoning_weighted"] = ew["reasoning"]
    if "concept_per_level" in eval_losses:
        for k, v in enumerate(eval_losses["concept_per_level"]):
            metrics[f"{swanlab_prefix}/concept_level{k}"] = v
    swanlab.log(metrics, step=global_step)

    term_entry = {
        "step": global_step,
        "eval_type": eval_type,
        **{
            f"eval_{k}": round(v, 6)
            for k, v in eval_losses.items()
            if k != "concept_per_level" and k != "_timing"
        },
        **{f"eval_{k}_w": round(v, 6) for k, v in ew.items()},
    }
    if "concept_per_level" in eval_losses:
        term_entry["eval_concept_per_level"] = [
            round(v, 6) for v in eval_losses["concept_per_level"]
        ]
    log_terminal_entry(terminal_log_path, term_entry)

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

    if reasoning_texts_dict:
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
# Standalone CLI: checkpoint + config resolution
# =============================================================================


def _extract_step(filename: str) -> int:
    """Extract ``step`` number from a checkpoint filename (``0`` on miss)."""
    m = re.search(r"-step(\d+)", filename)
    return int(m.group(1)) if m else 0


def _resolve_checkpoint_path(raw: str, storage_root: str) -> Path:
    """Resolve a checkpoint path with prefix-glob fallback.

    If the literal path does not exist, globs ``<stem>*.pt`` in the
    parent directory and returns the match with the highest step number.
    Mirrors training-time resolution so a single YAML value works across
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


def _resolve_config_path(raw: str, base: Path) -> Path:
    """Resolve a YAML path: absolute stays, relative joins ``base``."""
    p = Path(raw)
    return p if p.is_absolute() else base / p


def _inherit_pyramid_from_builder(predictor_config: dict, builder_config: dict) -> None:
    """Copy ``model.pyramid`` from the builder config into the predictor config.

    The predictor YAML intentionally does NOT re-declare ``model.pyramid``
    (geometry drift between Stage 1 and Stage 2 would silently corrupt
    training / eval). This helper is the single place the inheritance
    happens. Fail-fast if the predictor config already has a
    ``model.pyramid`` block — that would defeat the inheritance guarantee.
    """
    if "model" not in predictor_config:
        raise ValueError("Predictor config missing top-level 'model' block.")
    if "pyramid" in predictor_config["model"]:
        raise ValueError(
            "Predictor config must NOT declare 'model.pyramid' directly; "
            "it is inherited from the builder config pointed to by "
            "'model.builder.config_path'. Remove the pyramid block "
            "and let the evaluator inject it."
        )
    if "model" not in builder_config or "pyramid" not in builder_config["model"]:
        raise ValueError(
            "Builder config does not expose 'model.pyramid'; predictor "
            "inheritance is broken. Check the paired builder YAML."
        )
    predictor_config["model"]["pyramid"] = builder_config["model"]["pyramid"]


def _find_best_predictor_checkpoint(checkpoint_dir: Path) -> Path:
    """Pick the best available predictor checkpoint in ``checkpoint_dir``.

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
    raise FileNotFoundError(f"No predictor checkpoint found in {checkpoint_dir}.")


def _load_frozen_builder(
    builder_config: dict,
    checkpoint_path: Path,
    device: str,
    cli_logger: logging.Logger,
) -> ConceptPyramidBuilder:
    """Construct the Builder, load its checkpoint, freeze everything.

    Replica of ``train_predictor._load_frozen_builder`` adjusted for CLI
    use (no strict-load knob; we warn instead of raising).
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
    for p in builder.parameters():
        p.requires_grad = False
    builder.eval()
    cli_logger.info(
        "Builder loaded (epoch=%s step=%s) from %s",
        ckpt.get("epoch", "?"),
        ckpt.get("step", "?"),
        checkpoint_path,
    )
    return builder


def _load_predictor(
    config: dict,
    builder: ConceptPyramidBuilder,
    checkpoint_path: Path,
    device: str,
    cli_logger: logging.Logger,
) -> ConceptPredictor:
    """Instantiate the Predictor (with frozen Builder) and load weights."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Predictor checkpoint not found: {checkpoint_path}")
    predictor = ConceptPredictor(config, builder=builder)
    predictor.to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state_dict"]
    missing, unexpected = predictor.load_state_dict(state, strict=False)
    if missing or unexpected:
        cli_logger.warning(
            "Predictor loaded with strict=False | missing=%d unexpected=%d",
            len(missing),
            len(unexpected),
        )
    predictor.eval()
    if config["model"]["predictor"]["use_shared_model"]:
        predictor.reason_model.eval()
    cli_logger.info(
        "Predictor loaded (epoch=%s step=%s) from %s",
        ckpt.get("epoch", "?"),
        ckpt.get("step", "?"),
        checkpoint_path,
    )
    return predictor


# =============================================================================
# Standalone CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """CLI flags for standalone Predictor evaluation."""
    parser = argparse.ArgumentParser(description="NLCP V4 Predictor Evaluation")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to a predictor YAML config.",
    )
    parser.add_argument(
        "-s",
        "--storage-root",
        type=str,
        default="./",
        help="Prefix for relative paths in the config.",
    )
    parser.add_argument(
        "-p",
        "--predictor-ckpt",
        type=str,
        default="",
        help=(
            "Explicit predictor checkpoint path. When set, OVERRIDES "
            "auto-discovery from config's log.checkpoint_path."
        ),
    )
    parser.add_argument(
        "-q",
        "--builder-ckpt",
        type=str,
        default="",
        help=(
            "Explicit builder (Stage-1) checkpoint path. When set, "
            "OVERRIDES config's model.builder.checkpoint_path."
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
        help=(
            "Output directory override. Default: " "<log_path>/eval_predictor/<mode>/"
        ),
    )
    return parser.parse_args()


def _build_eval_dataloader(config: dict) -> NLCPV4DataLoader:
    """Construct the eval-split dataloader with ``batch_size=1``.

    batch_size is hard-coded here because the per-sample folder layout
    assumes exactly one sample per iteration; see the plan's output
    spec for details.
    """
    eval_cfg = config["evaluation"]
    eval_data_cfg = eval_cfg["data"]
    return NLCPV4DataLoader(
        data_cfg=eval_data_cfg,
        batch_size=1,
        include_solution=True,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )


def main() -> None:
    """Entry point for ``python examples/nlcpV4/eval_predictor.py``."""
    args = parse_args()

    predictor_config_path = Path(args.config)
    if not predictor_config_path.is_absolute():
        predictor_config_path = PROJECT_ROOT / predictor_config_path
    predictor_config = load_config(str(predictor_config_path))
    apply_storage_root(predictor_config, args.storage_root)

    builder_config_raw = predictor_config["model"]["builder"]["config_path"]
    builder_config_path = _resolve_config_path(builder_config_raw, PROJECT_ROOT)
    builder_config = load_config(str(builder_config_path))
    apply_storage_root(builder_config, args.storage_root)

    _inherit_pyramid_from_builder(predictor_config, builder_config)

    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        output_root = (
            Path(predictor_config["log"]["log_path"]) / "eval_predictor" / args.mode
        )
    output_root.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(output_root / "eval.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    cli_logger = logging.getLogger("eval_predictor")

    cli_logger.info("=" * 72)
    cli_logger.info("  NLCP V4 Predictor Evaluation")
    cli_logger.info("=" * 72)
    cli_logger.info("Predictor config : %s", predictor_config_path)
    cli_logger.info("Builder config   : %s", builder_config_path)
    cli_logger.info("Storage root     : %s", args.storage_root)
    cli_logger.info("Mode             : %s", args.mode)
    cli_logger.info("Output root      : %s", output_root)

    if args.builder_ckpt:
        builder_ckpt_path = Path(args.builder_ckpt).resolve()
        cli_logger.info("[OVERRIDE] Builder checkpoint (-q): %s", builder_ckpt_path)
    else:
        builder_ckpt_raw = predictor_config["model"]["builder"]["checkpoint_path"]
        builder_ckpt_path = _resolve_checkpoint_path(
            builder_ckpt_raw, args.storage_root
        ).resolve()
        cli_logger.info("[AUTO]     Builder checkpoint: %s", builder_ckpt_path)

    if args.predictor_ckpt:
        predictor_ckpt_path = Path(args.predictor_ckpt).resolve()
        cli_logger.info("[OVERRIDE] Predictor checkpoint (-p): %s", predictor_ckpt_path)
    else:
        checkpoint_dir = Path(predictor_config["log"]["checkpoint_path"])
        predictor_ckpt_path = _find_best_predictor_checkpoint(checkpoint_dir)
        cli_logger.info("[AUTO]     Predictor checkpoint: %s", predictor_ckpt_path)

    device = str(get_device("auto"))
    cli_logger.info("Device           : %s", device)

    builder = _load_frozen_builder(
        builder_config, builder_ckpt_path, device, cli_logger
    )
    predictor = _load_predictor(
        predictor_config, builder, predictor_ckpt_path, device, cli_logger
    )

    dataloader = _build_eval_dataloader(predictor_config)

    loss_weights = predictor_config["training"]["loss_weights"]
    max_length = predictor_config["model"]["pyramid"]["max_seq_len"]
    generation_max_tokens = predictor_config["evaluation"]["generation_max_tokens"]

    cli_logger.info("")
    cli_logger.info("Starting evaluation (max_samples=%d)...", args.max_samples)
    cli_logger.info("-" * 72)

    run_start = time.perf_counter()
    avg_losses, reasoning_texts, samples = evaluate_predictor(
        predictor=predictor,
        builder=builder,
        eval_dataloader=dataloader,
        loss_weights=loss_weights,
        max_length=max_length,
        device=device,
        max_batches=args.max_samples,
        mode=args.mode,
        generation_max_tokens=generation_max_tokens,
        output_root=output_root,
        dump_artifacts=True,
    )
    total_elapsed = time.perf_counter() - run_start

    solution_list = [s["solution"] for s in samples]
    acc_tf = compute_reasoning_accuracy(
        reasoning_texts["teacher_forced"], solution_list
    )
    acc_gen = compute_reasoning_accuracy(reasoning_texts["generation"], solution_list)

    cli_logger.info("")
    cli_logger.info("=" * 72)
    cli_logger.info("  EVALUATION COMPLETE")
    cli_logger.info("=" * 72)
    cli_logger.info("Samples processed    : %d", len(samples))
    cli_logger.info("Total wall-clock (s) : %.2f", total_elapsed)
    cli_logger.info("Avg total loss       : %.4f", avg_losses["total"])
    cli_logger.info("Avg concept loss     : %.4f", avg_losses["concept"])
    if "reasoning" in avg_losses:
        cli_logger.info("Avg reasoning loss   : %.4f", avg_losses["reasoning"])
    if acc_tf["num_total"] > 0:
        cli_logger.info(
            "Teacher-forced acc   : %.4f (%d/%d)",
            acc_tf["accuracy"],
            acc_tf["num_correct"],
            acc_tf["num_total"],
        )
    if acc_gen["num_total"] > 0:
        cli_logger.info(
            "Free-generation acc  : %.4f (%d/%d)",
            acc_gen["accuracy"],
            acc_gen["num_correct"],
            acc_gen["num_total"],
        )

    summary = {
        "predictor_config_path": str(predictor_config_path),
        "builder_config_path": str(builder_config_path),
        "predictor_checkpoint": str(predictor_ckpt_path),
        "builder_checkpoint": str(builder_ckpt_path),
        "mode": args.mode,
        "device": device,
        "num_samples": len(samples),
        "total_time_s": round(total_elapsed, 3),
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
