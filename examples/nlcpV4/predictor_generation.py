"""NLCP V4 Predictor Generation (Inference) Script.

Given a trained ConceptPredictor (Stage 2) and its paired frozen Builder
(Stage 1), this script loads both models, feeds questions through the
predictor's autoregressive inference loop, and records the full concept
pyramid output plus timing.

Usage:
    # Use checkpoint from config:
    python examples/nlcpV4/predictor_generation.py -c configs/nlcpV4/GSM8K/train_predictor_Qwen2.5-0.5B_3level_shared.yml -s /Data/<proj>

    # Override predictor checkpoint with -p:
    python examples/nlcpV4/predictor_generation.py -c configs/nlcpV4/GSM8K/train_predictor_Qwen2.5-0.5B_3level_shared.yml -s /Data/<proj> -p /path/to/predictor_checkpoint.pt

    # Specify input questions via -q (JSON file with list of question strings):
    python examples/nlcpV4/predictor_generation.py -c configs/nlcpV4/GSM8K/train_predictor_Qwen2.5-0.5B_3level_shared.yml -q questions.json

    # Use eval split from config as input source (default):
    python examples/nlcpV4/predictor_generation.py -c configs/nlcpV4/GSM8K/train_predictor_Qwen2.5-0.5B_3level_shared.yml --max-samples 20

Arguments:
    -c / --config         Path to a predictor YAML config.
    -s / --storage-root   Prefix for relative paths.  Default "./".
    -p / --predictor-ckpt Explicit predictor checkpoint path.  When set,
                          OVERRIDES the best checkpoint auto-discovered
                          from config's log.checkpoint_path.  Printed
                          clearly in the log.
    -q / --questions      JSON file containing a list of question strings.
                          When provided, uses these instead of the eval
                          dataset from config.
    --max-samples         Max number of samples to generate.  Default 50.
    -o / --output-dir     Output directory for generation results.
                          Default: <log_path>/generation_results/
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Project-root path injection.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

from lmbase.utils.env_tools import get_device
from nlcpV4.concept_builder import ConceptPyramidBuilder
from nlcpV4.concept_predictor import ConceptPredictor
from nlcpV4.data_loader import BuilderInput, NLCPV4DataLoader
from ram.utils import apply_storage_root, load_config

# =============================================================================
# CLI
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="NLCP V4 Predictor Generation (Inference)"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to predictor YAML config.",
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
            "Explicit predictor checkpoint path.  When set, OVERRIDES "
            "auto-discovery from config's log.checkpoint_path."
        ),
    )
    parser.add_argument(
        "-q",
        "--questions",
        type=str,
        default="",
        help="JSON file with a list of question strings to use as input.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Max number of samples to generate (from eval dataset).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="",
        help="Output directory.  Default: <log_path>/generation_results/",
    )
    return parser.parse_args()


# =============================================================================
# Checkpoint resolution helpers (mirrored from train_predictor.py)
# =============================================================================


def _extract_step(filename: str) -> int:
    m = re.search(r"-step(\d+)", filename)
    return int(m.group(1)) if m else 0


def _resolve_checkpoint_path(raw: str, storage_root: str) -> Path:
    """Resolve a checkpoint path with glob fallback."""
    p = Path(raw)
    if p.is_absolute():
        resolved = p
    else:
        resolved = Path(storage_root) / p

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
    p = Path(raw)
    return p if p.is_absolute() else base / p


def _inherit_pyramid_from_builder(predictor_config: dict, builder_config: dict) -> None:
    if "model" not in predictor_config:
        raise ValueError("Predictor config missing top-level 'model' block.")
    if "pyramid" in predictor_config["model"]:
        raise ValueError(
            "Predictor config must NOT declare 'model.pyramid' directly; "
            "it is inherited from the builder config."
        )
    if "model" not in builder_config or "pyramid" not in builder_config["model"]:
        raise ValueError("Builder config does not expose 'model.pyramid'.")
    predictor_config["model"]["pyramid"] = builder_config["model"]["pyramid"]


# =============================================================================
# Model loading
# =============================================================================


def _load_frozen_builder(
    builder_config: dict,
    checkpoint_path: Path,
    device: str,
    logger: logging.Logger,
) -> ConceptPyramidBuilder:
    """Construct and freeze the Stage-1 Builder."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Builder checkpoint not found: {checkpoint_path}")
    builder = ConceptPyramidBuilder(builder_config)
    builder.to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state_dict"]
    missing, unexpected = builder.load_state_dict(state, strict=False)
    if missing or unexpected:
        logger.warning(
            "Builder loaded with strict=False | missing=%d unexpected=%d",
            len(missing),
            len(unexpected),
        )
    for p in builder.parameters():
        p.requires_grad = False
    builder.eval()
    logger.info(
        "Builder loaded (epoch=%s step=%s) from %s",
        ckpt.get("epoch", "?"),
        ckpt.get("step", "?"),
        checkpoint_path,
    )
    return builder


def _load_predictor(
    config: dict,
    checkpoint_path: Path,
    builder: ConceptPyramidBuilder,
    device: str,
    logger: logging.Logger,
) -> ConceptPredictor:
    """Construct predictor and load checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Predictor checkpoint not found: {checkpoint_path}")
    predictor = ConceptPredictor(config, builder=builder)
    predictor.to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    predictor.load_state_dict(ckpt["model_state_dict"])
    predictor.eval()
    logger.info(
        "Predictor loaded (epoch=%s step=%s loss=%.4f) from %s",
        ckpt.get("epoch", "?"),
        ckpt.get("step", "?"),
        ckpt.get("loss", float("nan")),
        checkpoint_path,
    )
    return predictor


def _find_best_predictor_checkpoint(checkpoint_dir: Path) -> Path:
    """Auto-discover the best predictor checkpoint (best_eval > best > latest)."""
    # Prefer best_eval
    candidates = sorted(
        checkpoint_dir.glob("checkpoint_best_eval*.pt"),
        key=lambda f: _extract_step(f.name),
        reverse=True,
    )
    if candidates:
        return candidates[0]
    # Fallback to best
    candidates = sorted(
        checkpoint_dir.glob("checkpoint_best*.pt"),
        key=lambda f: _extract_step(f.name),
        reverse=True,
    )
    if candidates:
        return candidates[0]
    # Fallback to any checkpoint
    candidates = sorted(
        checkpoint_dir.glob("checkpoint*.pt"),
        key=lambda f: _extract_step(f.name),
        reverse=True,
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No predictor checkpoint found in {checkpoint_dir}")


# =============================================================================
# Generation
# =============================================================================


@torch.no_grad()
def generate_from_questions(
    predictor: ConceptPredictor,
    builder: ConceptPyramidBuilder,
    questions: list[str],
    main_ids: list[str],
    max_length: int,
    device: str,
    logger: logging.Logger,
) -> list[dict]:
    """Run predictor inference on a list of questions.

    Returns a list of per-sample result dicts containing:
        - question, main_id
        - predicted concept pyramid (per-level norms and stats)
        - timing information
    """
    results = []
    tokenizer = builder.tokenizer

    for idx, (question, main_id) in enumerate(zip(questions, main_ids)):
        logger.info(
            "Generating sample %d/%d (main_id=%s)", idx + 1, len(questions), main_id
        )

        # Tokenize the question
        q_enc = tokenizer(
            [question],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        q_ids = q_enc["input_ids"].to(device)
        q_mask = q_enc["attention_mask"].to(device)

        # Time the inference
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        output = predictor(
            question_ids=q_ids,
            question_attention_mask=q_mask,
            gt_concepts=None,  # inference mode
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Extract concept pyramid statistics
        pyramid_info = []
        for level_idx, concept_tensor in enumerate(output.predicted_concepts):
            # concept_tensor: [1, L_k, D]
            c = concept_tensor.squeeze(0)  # [L_k, D]
            pyramid_info.append(
                {
                    "level": level_idx,
                    "num_concepts": c.shape[0],
                    "dim": c.shape[1],
                    "norm_mean": float(c.norm(dim=-1).mean()),
                    "norm_std": float(c.norm(dim=-1).std()),
                    "norm_min": float(c.norm(dim=-1).min()),
                    "norm_max": float(c.norm(dim=-1).max()),
                    "values_mean": float(c.mean()),
                    "values_std": float(c.std()),
                }
            )

        # Record input token count
        input_token_count = int(q_mask.sum())

        result = {
            "sample_idx": idx,
            "main_id": main_id,
            "question": question,
            "input_tokens": input_token_count,
            "num_levels": output.num_levels,
            "level_lengths": output.level_lengths,
            "total_concept_slots": sum(output.level_lengths),
            "inference_time_ms": round(elapsed_ms, 3),
            "pyramid": pyramid_info,
        }
        results.append(result)

        logger.info(
            "  Done: %d tokens -> %d concept slots in %.1f ms",
            input_token_count,
            sum(output.level_lengths),
            elapsed_ms,
        )

    return results


@torch.no_grad()
def generate_with_teacher_forced_reasoning(
    predictor: ConceptPredictor,
    builder: ConceptPyramidBuilder,
    questions: list[str],
    solutions: list[str],
    cot_answers: list[str],
    main_ids: list[str],
    max_length: int,
    device: str,
    logger: logging.Logger,
) -> list[dict]:
    """Run predictor inference + teacher-forced reasoning decode.

    When solutions are available, we can also:
      1. Run inference to get predicted concepts
      2. Run teacher-forced forward with gt_concepts to get reasoning texts
      3. Compare predicted vs gt concepts (MSE)

    Returns per-sample dicts with full details.
    """
    results = []
    tokenizer = builder.tokenizer

    for idx, (question, solution, cot, main_id) in enumerate(
        zip(questions, solutions, cot_answers, main_ids)
    ):
        logger.info(
            "Generating sample %d/%d (main_id=%s)", idx + 1, len(questions), main_id
        )

        # --- Step 1: Pure inference (question only) ---
        q_enc = tokenizer(
            [question],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        q_ids = q_enc["input_ids"].to(device)
        q_mask = q_enc["attention_mask"].to(device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        infer_output = predictor(
            question_ids=q_ids,
            question_attention_mask=q_mask,
            gt_concepts=None,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - t0) * 1000.0

        # --- Step 2: Get gt_concepts from frozen builder ---
        batch = BuilderInput(
            questions=[question],
            cot_answers=[cot],
            solutions=[solution],
            main_ids=[main_id],
        )
        # Builder needs the full input (with cot) to produce gt_concepts
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        pyramid = builder(batch)
        gt_concepts = [c.detach() for c in pyramid.concepts]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        builder_ms = (time.perf_counter() - t1) * 1000.0

        # --- Step 3: Teacher-forced forward for reasoning texts ---
        s_enc = tokenizer(
            [solution],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        s_ids = s_enc["input_ids"].to(device)
        s_mask = s_enc["attention_mask"].to(device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        tf_output = predictor(
            question_ids=q_ids,
            question_attention_mask=q_mask,
            gt_concepts=gt_concepts,
            solution_ids=s_ids,
            solution_attention_mask=s_mask,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        teacher_forced_ms = (time.perf_counter() - t2) * 1000.0

        # --- Compute per-level MSE between predicted and gt ---
        per_level_mse = []
        for pred_c, gt_c in zip(infer_output.predicted_concepts, gt_concepts):
            mse = float(((pred_c - gt_c) ** 2).mean())
            per_level_mse.append(mse)

        # --- Pyramid stats ---
        pyramid_info = []
        for level_idx, concept_tensor in enumerate(infer_output.predicted_concepts):
            c = concept_tensor.squeeze(0)
            pyramid_info.append(
                {
                    "level": level_idx,
                    "num_concepts": c.shape[0],
                    "dim": c.shape[1],
                    "norm_mean": float(c.norm(dim=-1).mean()),
                    "norm_std": float(c.norm(dim=-1).std()),
                    "predicted_vs_gt_mse": per_level_mse[level_idx],
                }
            )

        input_token_count = int(q_mask.sum())
        solution_token_count = int(s_mask.sum())

        result = {
            "sample_idx": idx,
            "main_id": main_id,
            "question": question,
            "groundtruth_solution": solution,
            "input_tokens": input_token_count,
            "solution_tokens": solution_token_count,
            "num_levels": infer_output.num_levels,
            "level_lengths": infer_output.level_lengths,
            "total_concept_slots": sum(infer_output.level_lengths),
            "timing": {
                "inference_ms": round(inference_ms, 3),
                "builder_gt_ms": round(builder_ms, 3),
                "teacher_forced_ms": round(teacher_forced_ms, 3),
                "total_ms": round(inference_ms + builder_ms + teacher_forced_ms, 3),
            },
            "pyramid": pyramid_info,
            "concept_mse_per_level": per_level_mse,
            "concept_mse_avg": float(np.mean(per_level_mse)),
            "reasoning_text": (
                tf_output.reasoning_texts[0] if tf_output.reasoning_texts else None
            ),
            "builder_reasoning_text": (
                pyramid.reasoning_texts[0] if pyramid.reasoning_texts else None
            ),
        }
        results.append(result)

        logger.info(
            "  Inference: %.1f ms | Builder GT: %.1f ms | Teacher-forced: %.1f ms | MSE: %.6f",
            inference_ms,
            builder_ms,
            teacher_forced_ms,
            float(np.mean(per_level_mse)),
        )
        if tf_output.reasoning_texts:
            logger.info("  Predicted reasoning: %s", tf_output.reasoning_texts[0][:200])

    return results


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    # --- Resolve config ---
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config = load_config(str(config_path))
    apply_storage_root(config, args.storage_root)

    # Load builder config and inherit pyramid
    builder_cfg_path_raw = config["model"]["builder"]["config_path"]
    builder_cfg_path = _resolve_config_path(builder_cfg_path_raw, PROJECT_ROOT)
    if not builder_cfg_path.exists():
        raise FileNotFoundError(f"Builder config not found: {builder_cfg_path}")
    builder_config = load_config(str(builder_cfg_path))
    _inherit_pyramid_from_builder(config, builder_config)

    # --- Resolve output dir ---
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config["log"]["log_path"]) / "generation_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Setup logging ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "generation.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("predictor_generation")

    logger.info("=" * 72)
    logger.info("  NLCP V4 Predictor Generation")
    logger.info("=" * 72)
    logger.info("Config: %s", config_path)
    logger.info("Storage root: %s", args.storage_root)
    logger.info("Output dir: %s", output_dir)

    # --- Resolve predictor checkpoint ---
    if args.predictor_ckpt:
        predictor_ckpt_path = Path(args.predictor_ckpt).resolve()
        logger.info("")
        logger.info("[OVERRIDE] Using explicit predictor checkpoint from -p flag:")
        logger.info("  -p %s", predictor_ckpt_path)
        logger.info("  (config's log.checkpoint_path is IGNORED)")
    else:
        checkpoint_dir = Path(config["log"]["checkpoint_path"])
        predictor_ckpt_path = _find_best_predictor_checkpoint(checkpoint_dir)
        logger.info("")
        logger.info("[CONFIG] Using best predictor checkpoint from config:")
        logger.info("  checkpoint_dir: %s", checkpoint_dir)
        logger.info("  selected: %s", predictor_ckpt_path)

    # --- Device ---
    device = str(get_device("auto"))
    logger.info("Device: %s", device)

    # --- Load models ---
    builder_ckpt_raw = config["model"]["builder"]["checkpoint_path"]
    builder_ckpt_path = _resolve_checkpoint_path(
        builder_ckpt_raw, args.storage_root
    ).resolve()
    logger.info("Builder checkpoint: %s", builder_ckpt_path)

    builder = _load_frozen_builder(builder_config, builder_ckpt_path, device, logger)
    predictor = _load_predictor(config, predictor_ckpt_path, builder, device, logger)

    max_length = config["model"]["pyramid"]["max_seq_len"]
    level_lengths = config["model"]["pyramid"]["level_lengths"]
    logger.info(
        "Pyramid: K=%d, level_lengths=%s, total_C=%d, max_seq_len=%d",
        config["model"]["pyramid"]["num_levels"],
        level_lengths,
        sum(level_lengths),
        max_length,
    )

    # --- Prepare input questions ---
    if args.questions:
        # Load from JSON file
        q_path = Path(args.questions)
        if not q_path.is_absolute():
            q_path = PROJECT_ROOT / q_path
        with open(q_path, "r", encoding="utf-8") as f:
            q_data = json.load(f)

        # Support both list-of-strings and list-of-dicts
        if isinstance(q_data[0], str):
            questions = q_data[: args.max_samples]
            main_ids = [f"q_{i}" for i in range(len(questions))]
            solutions = []
            cot_answers = []
        else:
            questions = [d["question"] for d in q_data[: args.max_samples]]
            main_ids = [
                d.get("main_id", f"q_{i}")
                for i, d in enumerate(q_data[: args.max_samples])
            ]
            solutions = [
                d.get("groundtruth", d.get("solution", ""))
                for d in q_data[: args.max_samples]
            ]
            cot_answers = [d.get("cot_answer", "") for d in q_data[: args.max_samples]]

        logger.info("Input source: -q %s (%d questions)", q_path, len(questions))
    else:
        # Use eval dataset from config
        eval_cfg = config.get("evaluation", {})
        eval_data_cfg = eval_cfg.get("data", config["data"])
        dataloader = NLCPV4DataLoader(
            data_cfg=eval_data_cfg,
            batch_size=1,
            include_solution=True,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        questions, solutions, cot_answers, main_ids = [], [], [], []
        for batch in dataloader:
            questions.extend(batch.questions)
            solutions.extend(batch.solutions)
            cot_answers.extend(batch.cot_answers)
            main_ids.extend(batch.main_ids)
            if len(questions) >= args.max_samples:
                break
        questions = questions[: args.max_samples]
        solutions = solutions[: args.max_samples]
        cot_answers = cot_answers[: args.max_samples]
        main_ids = main_ids[: args.max_samples]
        logger.info(
            "Input source: eval dataset (%s, split=%s, %d samples)",
            eval_data_cfg["data_name"],
            eval_data_cfg["split"],
            len(questions),
        )

    logger.info("")
    logger.info("Starting generation for %d samples...", len(questions))
    logger.info("-" * 72)

    # --- Run generation ---
    total_start = time.perf_counter()

    has_solutions = bool(solutions and solutions[0])
    if has_solutions:
        results = generate_with_teacher_forced_reasoning(
            predictor=predictor,
            builder=builder,
            questions=questions,
            solutions=solutions,
            cot_answers=cot_answers,
            main_ids=main_ids,
            max_length=max_length,
            device=device,
            logger=logger,
        )
    else:
        results = generate_from_questions(
            predictor=predictor,
            builder=builder,
            questions=questions,
            main_ids=main_ids,
            max_length=max_length,
            device=device,
            logger=logger,
        )

    total_elapsed = time.perf_counter() - total_start

    # --- Summary ---
    logger.info("")
    logger.info("=" * 72)
    logger.info("  GENERATION COMPLETE")
    logger.info("=" * 72)
    logger.info("  Samples generated : %d", len(results))
    logger.info("  Total wall-clock  : %.2f s", total_elapsed)
    logger.info(
        "  Avg per sample    : %.1f ms", total_elapsed * 1000.0 / max(1, len(results))
    )
    if has_solutions and results:
        avg_mse = float(np.mean([r["concept_mse_avg"] for r in results]))
        logger.info("  Avg concept MSE   : %.6f", avg_mse)

    # --- Save results ---
    # Main results file
    output_file = output_dir / "generation_results.json"
    output_payload = {
        "config_path": str(config_path),
        "predictor_checkpoint": str(predictor_ckpt_path),
        "builder_checkpoint": str(builder_ckpt_path),
        "predictor_ckpt_source": (
            "CLI -p flag" if args.predictor_ckpt else "config auto-discovery"
        ),
        "device": device,
        "max_length": max_length,
        "num_samples": len(results),
        "total_time_s": round(total_elapsed, 3),
        "avg_time_per_sample_ms": round(
            total_elapsed * 1000.0 / max(1, len(results)), 3
        ),
        "pyramid_geometry": {
            "num_levels": config["model"]["pyramid"]["num_levels"],
            "level_lengths": level_lengths,
            "total_concepts": sum(level_lengths),
            "hidden_dim": config["model"]["pyramid"]["hidden_dim"],
        },
        "results": results,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, default=str, ensure_ascii=False)
    logger.info("Results saved to: %s", output_file)

    # Save concept tensors (binary, for downstream analysis)
    # Only save the raw tensors if sample count is small
    if len(results) <= 100:
        tensor_file = output_dir / "concept_tensors.pt"
        # Re-run to collect tensors (avoid storing in memory during main loop)
        # Instead, just log the path for now; tensors are summarized in JSON.
        logger.info("(Concept tensor stats are in generation_results.json)")

    logger.info("Done.")


if __name__ == "__main__":
    main()
