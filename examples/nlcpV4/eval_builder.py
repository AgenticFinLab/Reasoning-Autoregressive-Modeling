"""Evaluation and logging for NLCP V4 Builder and Predictor.

This module provides:
  - Builder evaluation loop (evaluate_builder)
  - Predictor evaluation loop (evaluate_predictor)
  - Builder eval result logging (log_eval_results)
  - Predictor eval result logging (log_eval_results_predictor)
  - Reasoning accuracy computation (compute_reasoning_accuracy)
  - Terminal log utility (log_terminal_entry)

Loss computation is in losses.py; this module imports compute_builder_loss
and compute_predictor_loss from there.

Used by:
    examples/nlcpV4/train_builder.py  (imports evaluate_builder,
        log_eval_results, log_terminal_entry)
    examples/nlcpV4/train_predictor.py  (imports evaluate_predictor,
        log_eval_results_predictor, log_terminal_entry)
    examples/nlcpV4/builder_training_analysis.py  (imports
        compute_reasoning_accuracy)
"""

import datetime
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import swanlab
import torch

from nlcpV4.concept_builder import ConceptPyramidBuilder
from nlcpV4.concept_predictor import ConceptPredictor
from nlcpV4.data_loader import BuilderInput, NLCPV4DataLoader
from nlcpV4.losses import compute_builder_loss, compute_predictor_loss

logger = logging.getLogger(__name__)


# ── Reasoning accuracy utilities ─────────────────────────────────────


def _extract_final_number(text: str) -> Optional[str]:
    """Extract the final numerical answer from a generated text.

    Tries multiple extraction strategies in order:
      1. GSM8K "#### <number>" pattern
      2. LaTeX \\boxed{<answer>} pattern (MATH)
      3. Last number-like token in the string

    Returns the extracted answer string (stripped), or None if
    no number can be found.
    """
    if not text or not text.strip():
        return None

    # Strategy 1: GSM8K "#### <answer>" pattern
    m = re.search(r"####\s*(.+)", text)
    if m:
        return m.group(1).strip()

    # Strategy 2: LaTeX \boxed{<answer>}
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()

    # Strategy 3: last number (int or decimal, possibly negative)
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()

    return None


def _normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison.

    Handles:
      - Leading/trailing whitespace
      - Comma-separated thousands (e.g. "1,234" -> "1234")
      - Trailing ".0" removal for integers
      - Fraction normalization: "\\frac{a}{b}" -> "a/b"
    """
    s = answer.strip()
    # Remove commas in numbers
    s = s.replace(",", "")
    # \frac{a}{b} -> a/b
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", s)
    # Remove trailing .0 ("42.0" -> "42")
    if re.match(r"^-?\d+\.0+$", s):
        s = s.split(".")[0]
    return s


def compute_reasoning_accuracy(
    reasoning_texts: list[str],
    solutions: list[str],
) -> dict:
    """Compute exact-match accuracy between predicted texts and GT solutions.

    Extracts the final numerical answer from each reasoning text and
    compares (after normalization) against the corresponding ground-truth
    solution string.

    Args:
        reasoning_texts: Model-generated decoded strings (teacher-forced
            argmax). Length N.
        solutions: Ground-truth answer strings from the dataset
            (e.g. "3430", "200"). Length N. Entries may be None if
            solution was not available.

    Returns:
        Dict with keys:
          - "accuracy": float in [0, 1]. Fraction of exact matches.
          - "num_correct": int. Number of exact matches.
          - "num_total": int. Number of samples compared (excludes
            None solutions).
          - "num_extracted": int. Number of predictions where a final
            answer could be extracted (even if wrong).
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

        pred_norm = _normalize_answer(extracted)
        gt_norm = _normalize_answer(gt_sol)
        if pred_norm == gt_norm:
            num_correct += 1

    accuracy = num_correct / num_total if num_total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "num_correct": num_correct,
        "num_total": num_total,
        "num_extracted": num_extracted,
    }


# ── Terminal / file logging utilities ────────────────────────────────


def log_terminal_entry(log_path: Path, entry: dict):
    """Append a structured JSON line to the terminal output log file.

    Each entry is a JSON object with timestamp, step, epoch,
    loss values, and learning rate. Written immediately to disk
    so terminal output is preserved even if training crashes.
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
):
    """Log eval results (raw/weighted) to console, terminal, SwanLab, eval_history.

    Args:
        reasoning_texts_dict: Dict with keys "teacher_forced" and "generation",
            each mapping to a list of decoded strings. Empty lists are skipped.

    Also appends a record to ``eval_sample_history`` documenting which
    samples were consumed by this eval invocation.
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
        "  %s | total=%.4f recon=%.4f/%.4f ordering=%.4f/%.4f" " residual=%.4f/%.4f%s",
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
    timing = eval_losses.get("_timing")
    # SwanLab
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
    # Terminal log
    term_data = {
        "step": global_step,
        "eval_type": eval_type,
        **{f"eval_{k}": round(v, 6) for k, v in eval_losses.items() if k != "_timing"},
        **{f"eval_{k}_w": round(v, 6) for k, v in ew.items()},
    }
    log_terminal_entry(
        terminal_log_path,
        term_data,
    )
    # Eval history + save immediately (crash-safe)
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

    # Save reasoning decoded texts (crash-safe, append per eval)
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

    # Sample history: one record per eval invocation listing the exact
    # rows consumed so the caller can re-verify which inputs produced
    # the losses above without re-running the model.
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


# ── Evaluation loop ──────────────────────────────────────────────────


@torch.no_grad()
def evaluate_builder(
    builder: ConceptPyramidBuilder,
    eval_dataloader: NLCPV4DataLoader,
    loss_weights: dict,
    ordering_loss_type: str,
    max_batches: int,
    teacher_force_reasoning: bool = True,
    generation_reasoning: bool = False,
    generation_max_tokens: int = 256,
) -> tuple[dict, dict[str, list[str]], list[dict]]:
    """Run evaluation on test data and return averaged loss + texts + samples.

    Args:
        builder: The model to evaluate.
        eval_dataloader: DataLoader yielding BuilderInput batches from test set.
        loss_weights: Loss weight configuration.
        ordering_loss_type: "margin", "gaussian", or "both".
        max_batches: Maximum batches to evaluate. 0 = all batches.
        teacher_force_reasoning: If True, record teacher-forced argmax texts.
        generation_reasoning: If True, record free autoregressive generation texts.
        generation_max_tokens: Max new tokens for free generation.

    Returns:
        Tuple ``(averaged_loss_dict, reasoning_texts_dict, samples)``.
        - ``averaged_loss_dict`` has keys: total, recon, ordering, residual, reasoning.
        - ``reasoning_texts_dict`` has keys "teacher_forced" and/or "generation",
          each mapping to a flat list of decoded strings from all batches.
        - ``samples`` is a list of per-sample records
          ``{batch_idx, pos_in_batch, main_id, question, solution}``.
    """
    builder.eval()
    all_losses = []
    all_reasoning_texts_tf: list[str] = []
    all_reasoning_texts_gen: list[str] = []
    all_samples: list[dict] = []
    batch_times_ms: list[float] = []

    # No progress bar here: eval runs silently and the single summary
    # line printed by ``log_eval_results`` after this returns is the
    # only eval output. This keeps the training log clean — a tqdm
    # bar's per-iter ``\r`` refresh would otherwise become a separate
    # line in a tee'd log file.
    eval_start = time.perf_counter()
    for i, batch in enumerate(eval_dataloader):
        if max_batches > 0 and i >= max_batches:
            break

        # Forward pass: batch -> pyramid (encode + build + reasoning)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        pyramid = builder(batch)

        _, loss_dict = compute_builder_loss(
            pyramid, loss_weights, ordering_loss_type=ordering_loss_type
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_times_ms.append((time.perf_counter() - t0) * 1000.0)

        all_losses.append(loss_dict)

        if teacher_force_reasoning and pyramid.reasoning_texts is not None:
            all_reasoning_texts_tf.extend(pyramid.reasoning_texts)

        # Free generation: generate solution from [Q, Concepts] only
        if generation_reasoning and batch.has_solution:
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
            gen_texts = builder.generate_solution(
                pyramid,
                q_ids,
                q_mask,
                max_new_tokens=generation_max_tokens,
            )
            all_reasoning_texts_gen.extend(gen_texts)

        # Record per-sample metadata so eval_sample_history.json can
        # reconstruct which inputs were consumed by this eval invocation.
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

    eval_elapsed_s = time.perf_counter() - eval_start
    builder.train()

    if not all_losses:
        return (
            {"total": 0.0, "recon": 0.0, "ordering": 0.0, "residual": 0.0},
            {"teacher_forced": [], "generation": []},
            [],
        )

    # Average across all batches. Keys are consistent across batches
    # within a single eval run (reasoning is present iff solutions are),
    # so direct ``d[k]`` access is safe here.
    avg = {}
    keys = all_losses[0].keys()
    for k in keys:
        avg[k] = sum(d[k] for d in all_losses) / len(all_losses)

    # Timing metadata: total eval wall-clock, per-batch mean/min/max.
    num_batches = len(batch_times_ms)
    avg["_timing"] = {
        "eval_total_s": round(eval_elapsed_s, 3),
        "num_batches": num_batches,
        "batch_mean_ms": round(sum(batch_times_ms) / num_batches, 2),
        "batch_min_ms": round(min(batch_times_ms), 2),
        "batch_max_ms": round(max(batch_times_ms), 2),
    }

    texts_dict = {
        "teacher_forced": all_reasoning_texts_tf,
        "generation": all_reasoning_texts_gen,
    }

    return avg, texts_dict, all_samples


# =============================================================================
# Predictor evaluation
# =============================================================================


def _strip_solutions(batch: BuilderInput) -> BuilderInput:
    """Return a clone of ``batch`` with ``solutions=[]`` so the Builder
    forward skips its own ``_prepare_reasoning`` path."""
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
    """Tokenize questions and solutions with the Builder's tokenizer."""
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


@torch.no_grad()
def evaluate_predictor(
    predictor: ConceptPredictor,
    builder: ConceptPyramidBuilder,
    eval_dataloader: NLCPV4DataLoader,
    loss_weights: dict,
    max_length: int,
    device: str,
    max_batches: int,
    teacher_force_reasoning: bool = True,
    generation_reasoning: bool = False,
    generation_max_tokens: int = 256,
) -> tuple[dict, dict[str, list[str]], list[dict]]:
    """Run eval on ``max_batches`` of test data; return averaged losses.

    Returns:
        (averaged_loss_dict, reasoning_texts_dict, samples)
          - averaged_loss_dict keys: total, concept, reasoning (optional),
            concept_per_level (list[float], averaged elementwise).
            Also contains ``_timing`` dict with eval timing metadata.
          - reasoning_texts_dict: {"teacher_forced": [...], "generation": [...]}.
          - samples: per-row metadata for eval_sample_history.json.
    """
    predictor.eval()
    all_losses: list[dict] = []
    all_texts_tf: list[str] = []
    all_texts_gen: list[str] = []
    all_samples: list[dict] = []
    batch_times_ms: list[float] = []

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
        batch_times_ms.append((time.perf_counter() - t0) * 1000.0)

        all_losses.append(loss_dict)

        if teacher_force_reasoning and output.reasoning_texts is not None:
            all_texts_tf.extend(output.reasoning_texts)

        # Free generation from predicted concepts
        if generation_reasoning and batch.has_solution:
            q_ids, q_mask, _, _ = _tokenize_qs(builder, batch, max_length, device)
            gen_texts = predictor.generate_solution(
                output.predicted_concepts,
                q_ids,
                q_mask,
                max_new_tokens=generation_max_tokens,
            )
            all_texts_gen.extend(gen_texts)

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

    # Timing metadata
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

    # SwanLab metrics
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

    # terminal_output.jsonl row
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

    # eval_history.json (crash-safe rewrite per eval)
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
