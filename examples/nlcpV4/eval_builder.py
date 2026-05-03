"""Evaluation and logging for NLCP V4 ConceptPyramidBuilder.

This module provides:
  - Full evaluation loop (evaluate_builder)
  - Eval result logging (console, terminal, SwanLab, eval_history)
  - Terminal log utility (log_terminal_entry)

Loss computation is in losses.py; this module imports compute_builder_loss
from there.

Used by:
    examples/nlcpV4/train_builder.py  (imports evaluate_builder,
        log_eval_results, log_terminal_entry)
"""

import datetime
import json
import logging
from pathlib import Path

import swanlab
import torch

from nlcpV4.concept_builder import ConceptPyramidBuilder
from nlcpV4.data_loader import NLCPV4DataLoader
from nlcpV4.losses import compute_builder_loss

logger = logging.getLogger(__name__)


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
    reasoning_texts,
    eval_samples,
    eval_sample_history,
):
    """Log eval results (raw/weighted) to console, terminal, SwanLab, eval_history.

    Also appends a record to ``eval_sample_history`` documenting which
    samples were consumed by this eval invocation (``main_id`` from the
    lmbase source row, question, groundtruth solution), and persists it
    to ``log_dir / eval_sample_history.json`` so loss-history rows can
    be reconciled with the exact data that produced them.
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
    log_terminal_entry(
        terminal_log_path,
        {
            "step": global_step,
            "eval_type": eval_type,
            **{f"eval_{k}": round(v, 6) for k, v in eval_losses.items()},
            **{f"eval_{k}_w": round(v, 6) for k, v in ew.items()},
        },
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
    if reasoning_texts:
        entry = {
            "step": global_step,
            "eval_type": eval_type,
            "texts": reasoning_texts,
        }
        with open(log_dir / "eval_reasoning_texts.jsonl", "a", encoding="utf-8") as f:
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
) -> tuple[dict, list[str], list[dict]]:
    """Run evaluation on test data and return averaged loss + texts + samples.

    Args:
        builder: The model to evaluate.
        eval_dataloader: DataLoader yielding BuilderInput batches from test set.
        loss_weights: Loss weight configuration.
        ordering_loss_type: "margin", "gaussian", or "both".
        max_batches: Maximum batches to evaluate. 0 = all batches.

    Returns:
        Tuple ``(averaged_loss_dict, reasoning_texts, samples)``.
        - ``averaged_loss_dict`` has keys: total, recon, ordering, residual, reasoning.
        - ``reasoning_texts`` is a flat list of teacher-forced decoded
          strings from all batches (empty if no solutions).
        - ``samples`` is a list of per-sample records
          ``{batch_idx, pos_in_batch, main_id, question, solution}`` in
          the exact order they were consumed. ``main_id`` comes from
          the lmbase source row, so rows align 1:1 with the dataset.
    """
    builder.eval()
    all_losses = []
    all_reasoning_texts = []
    all_samples: list[dict] = []

    # No progress bar here: eval runs silently and the single summary
    # line printed by ``log_eval_results`` after this returns is the
    # only eval output. This keeps the training log clean — a tqdm
    # bar's per-iter ``\r`` refresh would otherwise become a separate
    # line in a tee'd log file.
    for i, batch in enumerate(eval_dataloader):
        if max_batches > 0 and i >= max_batches:
            break

        # Forward pass: batch -> pyramid (encode + build + reasoning)
        pyramid = builder(batch)

        _, loss_dict = compute_builder_loss(
            pyramid, loss_weights, ordering_loss_type=ordering_loss_type
        )

        all_losses.append(loss_dict)

        if pyramid.reasoning_texts is not None:
            all_reasoning_texts.extend(pyramid.reasoning_texts)

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

    builder.train()

    if not all_losses:
        return (
            {"total": 0.0, "recon": 0.0, "ordering": 0.0, "residual": 0.0},
            [],
            [],
        )

    # Average across all batches. Keys are consistent across batches
    # within a single eval run (reasoning is present iff solutions are),
    # so direct ``d[k]`` access is safe here.
    avg = {}
    keys = all_losses[0].keys()
    for k in keys:
        avg[k] = sum(d[k] for d in all_losses) / len(all_losses)
    return avg, all_reasoning_texts, all_samples
