"""ConceptPredictor end-to-end pipeline test with real GSM8K data.

Run with:
    python3 examples/nlcpV4/utest/test_concept_predictor.py \
        -c configs/nlcpV4/utest/test_concept_predictor.yml

DESIGN PHILOSOPHY:
    Single end-to-end pipeline driven by real GSM8K data:
    Builder (Stage 1) extracts GT concepts → Predictor (Stage 2) predicts them.
    Each pipeline step is verified and logged in detail.
    Diagnostic (log-based), not assertion-based.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

from nlcpV4.concept_builder import (
    ConceptPyramidBuilder,
    PyramidOutput,
)
from nlcpV4.concept_predictor import (
    ConceptPredictor,
    PredictorOutput,
    build_scale_causal_mask,
)
from nlcpV4.data_loader import NLCPV4DataLoader
from nlcpV4.losses import compute_predictor_concept_loss, compute_predictor_loss
from lmbase.utils.env_tools import get_device
from ram.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="ConceptPredictor test")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to YAML config file"
    )
    return parser.parse_args()


def log_check(name, cond, details=""):
    status = "OK" if cond else "WARN"
    msg = f"  [{status}] {name}"
    if details:
        msg += f" | {details}"
    logging.info(msg)


def log_value(name, value, unit=""):
    u = f" {unit}" if unit else ""
    logging.info(f"  [VAL] {name} = {value}{u}")


def log_section(title):
    logging.info(f"\n{'='*70}")
    logging.info(f"  {title}")
    logging.info(f"{'='*70}")


def run_pipeline(config, device):
    """Single end-to-end pipeline: real data → builder → predictor (train + infer) → gradient."""

    pyramid_cfg = config["model"]["pyramid"]
    num_levels = pyramid_cfg["num_levels"]
    level_lengths = pyramid_cfg["level_lengths"]
    hidden_dim = pyramid_cfg["hidden_dim"]
    batch_size = config["training"]["batch_size"]

    # ==================================================================
    # Step 1: Load Builder (Stage 1) + Data
    # ==================================================================
    log_section("Step 1: Load Builder + GSM8K Data")

    builder = ConceptPyramidBuilder(config)
    builder.to(device)
    builder.eval()
    log_value("reason_model_hidden_dim", builder.reason_model_hidden_dim)

    dataloader = NLCPV4DataLoader(
        data_cfg=config["data"],
        batch_size=batch_size,
        include_solution=False,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    batch = next(iter(dataloader))
    log_value("dataset_size", dataloader.dataset_size)
    log_value("batch_size", batch.batch_size)
    log_value("sample question[0][:80]", repr(batch.questions[0][:80]))
    log_value("sample cot_answer[0][:80]", repr(batch.cot_answers[0][:80]))

    # ==================================================================
    # Step 2: Builder extracts GT concept pyramid from real CoT
    # ==================================================================
    log_section("Step 2: Builder → GT Concept Pyramid from real CoT")

    pyramid = builder(batch)
    log_check("returns PyramidOutput", isinstance(pyramid, PyramidOutput))
    log_value("pyramid.num_levels", pyramid.num_levels)
    log_value("pyramid.total_concepts", pyramid.total_concepts)

    gt_concepts = [c.detach() for c in pyramid.concepts]
    for k, c in enumerate(gt_concepts):
        log_value(f"GT concepts[{k}].shape", list(c.shape))
        log_value(
            f"GT concepts[{k}] stats",
            f"mean={c.mean().item():.4f}, std={c.std().item():.4f}",
        )

    # ==================================================================
    # Step 3: Initialize Predictor (Stage 2) — shared model
    # ==================================================================
    log_section("Step 3: Initialize ConceptPredictor (shared backbone)")

    predictor = ConceptPredictor(config, builder=builder)
    predictor.to(device)

    log_check("reason_model is shared", predictor.reason_model is builder.reason_model)
    log_check("tokenizer is shared", predictor.tokenizer is builder.tokenizer)
    log_check("_owns_model is False", not predictor._owns_model)
    log_value(
        "q_proj dimensions",
        f"[{predictor.q_proj.in_features} → {predictor.q_proj.out_features}]",
    )
    log_value(
        "level_embeddings.num_embeddings", predictor.level_embeddings.num_embeddings
    )
    log_value(
        "position_embeddings.num_embeddings",
        predictor.position_embeddings.num_embeddings,
    )
    log_value("start_token.shape", list(predictor.start_token.shape))
    log_value("_num_levels", predictor._num_levels)
    log_value("_total_concepts", predictor._total_concepts)

    # Verify error handling: shared mode without builder
    try:
        ConceptPredictor(config, builder=None)
        log_check("ValueError when builder=None & shared=True", False, "No exception!")
    except ValueError:
        log_check("ValueError when builder=None & shared=True", True)

    # ==================================================================
    # Step 4: Scale-Level Causal Mask
    # ==================================================================
    log_section("Step 4: Verify scale-level causal mask")

    mask = build_scale_causal_mask(level_lengths, device=device)
    total_len = sum(level_lengths)
    log_value("mask.shape", list(mask.shape))
    log_check("shape [L_total, L_total]", mask.shape == (total_len, total_len))
    log_check("diagonal all 0 (self-attend)", (mask.diag() == 0).all().item())
    # Level 0 (pos 0) sees only itself
    log_check(
        "level 0 sees only pos 0",
        mask[0, 0] == 0 and (mask[0, 1:] == float("-inf")).all().item(),
    )
    # Last level sees all
    last_start = sum(level_lengths[:-1])
    log_check(
        "last level sees all positions",
        (mask[last_start, :] == 0).all().item(),
    )
    unique = mask.unique()
    log_check(
        "mask values are 0 and -inf only",
        all(v == 0 or v == float("-inf") for v in unique.tolist()),
    )

    # ==================================================================
    # Step 5: Tokenize real questions for predictor
    # ==================================================================
    log_section("Step 5: Tokenize real GSM8K questions")

    Q_tokens = predictor.tokenizer(
        batch.questions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=pyramid_cfg["max_seq_len"],
    )
    q_ids = Q_tokens["input_ids"].to(device)
    q_mask = Q_tokens["attention_mask"].to(device)
    log_value("q_ids.shape", list(q_ids.shape))
    log_value("q_mask valid tokens", q_mask.sum(dim=1).tolist())

    # encode_question
    q_hidden = predictor.encode_question(q_ids, q_mask)
    log_value("q_hidden.shape", list(q_hidden.shape))
    log_check("q_hidden is 3D", q_hidden.dim() == 3)
    log_check(
        "last dim == D_model", q_hidden.shape[-1] == predictor.reason_model_hidden_dim
    )

    # ==================================================================
    # Step 6: Training Forward (teacher-forcing with real GT)
    # ==================================================================
    log_section("Step 6: Training Forward (teacher-forcing)")

    predictor.train()
    train_out = predictor(
        q_ids, question_attention_mask=q_mask, gt_concepts=gt_concepts
    )

    log_check("returns PredictorOutput", isinstance(train_out, PredictorOutput))
    log_check("gt_concepts carried in output", train_out.gt_concepts is not None)
    log_value("predicted_concepts count", len(train_out.predicted_concepts))
    log_check("count == num_levels", len(train_out.predicted_concepts) == num_levels)

    # Compute per-level + total concept loss via losses.py (new API).
    concept_total, per_level = compute_predictor_concept_loss(
        train_out.predicted_concepts, train_out.gt_concepts, concept_loss_type="mse"
    )

    logging.info("  -- Per-level detail (training) --")
    logging.info(
        "  ┌──────────────────────────────────────────────────────────────────┐"
    )
    logging.info(
        "  │ Level │ L_k │ Shape              │ MSE Loss   │ mean    │ std     │"
    )
    logging.info(
        "  ├──────────────────────────────────────────────────────────────────┤"
    )
    for k in range(num_levels):
        pred = train_out.predicted_concepts[k]
        expected = (batch.batch_size, level_lengths[k], hidden_dim)
        shape_ok = pred.shape == expected
        loss_val = per_level[k].item() if k < len(per_level) else float("nan")
        logging.info(
            "  │  %d    │ %3d │ %-18s │ %10.4f │ %+7.4f │ %7.4f  │",
            k,
            level_lengths[k],
            str(list(pred.shape)),
            loss_val,
            pred.mean().item(),
            pred.std().item(),
        )
        log_check(f"  level {k} shape correct", shape_ok)
    logging.info(
        "  └──────────────────────────────────────────────────────────────────┘"
    )

    log_value("concept_total_loss", f"{concept_total.item():.4f}")
    log_check("concept_total is finite", torch.isfinite(concept_total).item())
    log_check("concept_total > 0", concept_total.item() > 0)

    # Verify total = mean of per-level
    manual_avg = sum(ll.item() for ll in per_level) / num_levels
    avg_diff = abs(concept_total.item() - manual_avg)
    log_check(
        "total_loss ≈ mean(per_level_losses)",
        avg_diff < 1e-4,
        f"total={concept_total.item():.6f}, manual={manual_avg:.6f}",
    )

    # Verify the combined compute_predictor_loss helper also works.
    combined_loss, combined_dict = compute_predictor_loss(
        train_out,
        loss_weights={"concept_loss_weight": 1.0, "reasoning_loss_weight": 1.0},
    )
    log_check(
        "compute_predictor_loss returns finite",
        torch.isfinite(combined_loss).item(),
    )
    log_value("compute_predictor_loss dict keys", list(combined_dict.keys()))

    # ==================================================================
    # Step 7: Inference Forward (autoregressive)
    # ==================================================================
    log_section("Step 7: Inference Forward (autoregressive)")

    predictor.eval()
    with torch.no_grad():
        infer_out = predictor(q_ids, question_attention_mask=q_mask)

    log_check("returns PredictorOutput", isinstance(infer_out, PredictorOutput))
    log_check("gt_concepts is None in inference", infer_out.gt_concepts is None)
    log_check(
        "reasoning_logits is None (no solution)", infer_out.reasoning_logits is None
    )
    log_value("predicted_concepts count", len(infer_out.predicted_concepts))

    logging.info("  -- Per-level detail (inference) --")
    for k, pred in enumerate(infer_out.predicted_concepts):
        expected = (batch.batch_size, level_lengths[k], hidden_dim)
        log_check(f"level {k} shape", pred.shape == expected, str(list(pred.shape)))
        log_value(
            f"level {k} stats",
            f"mean={pred.mean().item():.4f}, std={pred.std().item():.4f}",
        )

    # Training vs inference shape consistency
    logging.info("  -- Training-Inference consistency --")
    for k in range(num_levels):
        t_shape = train_out.predicted_concepts[k].shape
        i_shape = infer_out.predicted_concepts[k].shape
        log_check(f"level {k} shapes match", t_shape == i_shape)

    # ==================================================================
    # Step 8: Gradient Flow — backward through concept loss
    # ==================================================================
    log_section("Step 8: Gradient Flow (backward on concept loss)")

    predictor.train()
    predictor.zero_grad()
    train_out2 = predictor(
        q_ids, question_attention_mask=q_mask, gt_concepts=gt_concepts
    )
    concept_loss2, _ = compute_predictor_concept_loss(
        train_out2.predicted_concepts, train_out2.gt_concepts
    )
    concept_loss2.backward()

    # Check predictor-owned parameters
    param_checks = [
        ("q_proj.weight", predictor.q_proj.weight),
        ("q_proj_norm.weight", predictor.q_proj_norm.weight),
        ("start_token", predictor.start_token),
        ("level_embeddings.weight", predictor.level_embeddings.weight),
        ("position_embeddings.weight", predictor.position_embeddings.weight),
    ]
    for name, param in param_checks:
        has_grad = param.grad is not None
        grad_norm = f"{param.grad.norm().item():.6f}" if has_grad else "N/A"
        log_check(f"{name} has gradient", has_grad)
        log_value(f"{name} grad norm", grad_norm)

    # concept_head
    head_grads = sum(
        1 for p in predictor.concept_head.parameters() if p.grad is not None
    )
    head_total = sum(1 for _ in predictor.concept_head.parameters())
    log_check(
        f"concept_head: {head_grads}/{head_total} params have grad",
        head_grads == head_total,
    )

    # concept_transformer
    trans_grads = sum(
        1 for p in predictor.concept_transformer.parameters() if p.grad is not None
    )
    trans_total = sum(1 for _ in predictor.concept_transformer.parameters())
    log_check(
        f"concept_transformer: {trans_grads}/{trans_total} params have grad",
        trans_grads == trans_total,
    )

    # reason_model (should NOT have grads if frozen/shared)
    backbone_grad = any(
        p.grad is not None
        for p in predictor.reason_model.parameters()
        if p.requires_grad
    )
    log_check(
        "reason_model has gradients (only if un-frozen)",
        backbone_grad,
        "Expected WARN if freeze=True",
    )

    predictor.eval()
    logging.info("\n=== PREDICTOR PIPELINE TEST COMPLETE ===")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    yaml_config = load_config(str(config_path))

    device = str(get_device("auto"))
    logging.info("Device: %s", device)

    run_pipeline(yaml_config, device)


if __name__ == "__main__":
    main()
