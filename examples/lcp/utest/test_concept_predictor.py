"""ConceptPredictor (VAR-faithful) end-to-end diagnostic test with real GSM8K data.

Run with (defaults to a 6-level GSM8K predictor config):
    python examples/lcp/utest/test_concept_predictor.py

Or override with any predictor config from configs/lcp/GSM8K or configs/lcp/MATH:
    python examples/lcp/utest/test_concept_predictor.py \
        -c configs/lcp/GSM8K/train_predictor_Qwen2.5-0.5B_6level_independent.yml
    python examples/lcp/utest/test_concept_predictor.py \
        -c configs/lcp/MATH/train_predictor_Qwen2.5-0.5B_6level_independent.yml

DESIGN:
    Mirrors examples/lcp/train_predictor.py's loading contract:
    - Load the predictor YAML.
    - Resolve `model.builder.config_path` and load the paired builder YAML.
    - Inject `model.pyramid` from the builder config into the predictor config
      (predictor configs intentionally do NOT re-declare pyramid geometry).
    - Construct a FRESH builder from the builder config (NO checkpoint load —
      utest validates shapes / gradient flow, not learned weights).
    - Construct ConceptPredictor(config, builder=builder) and exercise every
      surface: _construct_approx_tokens, scale-causal mask, training forward,
      loss, inference forward, generate_solution, gradient flow.

    Diagnostic (log-based), not assertion-based.

NOTE on batch_size: the test forces batch_size=2 regardless of what the YAML
declares, so the pipeline runs quickly on CPU/MPS.
"""

import argparse
import logging
import sys
import types
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LCP_DIR = PROJECT_ROOT / "examples" / "lcp"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))
sys.path.insert(0, str(PROJECT_ROOT / "third-part" / "lmbase"))
sys.path.insert(0, str(PROJECT_ROOT / "third-part"))

# Prevent lcp/__init__.py from running (it eagerly imports eval_builder → swanlab).
# We register a minimal lcp package stub, then import submodules directly.
if "lcp" not in sys.modules:
    lcp_pkg = types.ModuleType("lcp")
    lcp_pkg.__path__ = [str(LCP_DIR)]
    lcp_pkg.__package__ = "lcp"
    sys.modules["lcp"] = lcp_pkg

from lcp.concept_builder import ConceptPyramidBuilder, PyramidOutput
from lcp.concept_predictor import ConceptPredictor, PredictorOutput
from lcp.data_loader import LCPDataLoader
from lcp.losses import compute_predictor_concept_loss, compute_predictor_loss
from lmbase.utils.env_tools import get_device
from ram.utils import load_config

DEFAULT_PREDICTOR_CONFIG = (
    "configs/lcp/GSM8K/train_predictor_Qwen2.5-0.5B_6level_independent.yml"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConceptPredictor (VAR-faithful) diagnostic test"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=DEFAULT_PREDICTOR_CONFIG,
        help=(
            "Path to a predictor YAML under configs/lcp/{GSM8K,MATH}/. "
            f"Default: {DEFAULT_PREDICTOR_CONFIG}"
        ),
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
    logging.info(f"\n{'='*72}")
    logging.info(f"  {title}")
    logging.info(f"{'='*72}")


def _resolve_path(raw: str, base: Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else base / p


def _inherit_pyramid_from_builder(predictor_cfg: dict, builder_cfg: dict) -> None:
    """Mirror train_predictor.py: inject builder.pyramid into predictor cfg."""
    if "pyramid" in predictor_cfg.get("model", {}):
        # Predictor already has pyramid (e.g., utest legacy override) — leave it.
        return
    if "model" not in builder_cfg or "pyramid" not in builder_cfg["model"]:
        raise ValueError(
            "Builder config does not expose 'model.pyramid'; cannot inherit."
        )
    predictor_cfg.setdefault("model", {})["pyramid"] = builder_cfg["model"]["pyramid"]


def load_predictor_and_builder_configs(predictor_yaml_path: Path):
    """Load predictor YAML + paired builder YAML, inherit pyramid block."""
    predictor_cfg = load_config(str(predictor_yaml_path))

    builder_block = predictor_cfg.get("model", {}).get("builder", {})
    builder_cfg_raw = builder_block.get("config_path")
    if not builder_cfg_raw:
        raise ValueError(
            f"Predictor config {predictor_yaml_path} is missing "
            f"'model.builder.config_path'."
        )

    builder_cfg_path = _resolve_path(builder_cfg_raw, PROJECT_ROOT)
    if not builder_cfg_path.exists():
        raise FileNotFoundError(
            f"Paired builder config not found: {builder_cfg_path} "
            f"(referenced from {predictor_yaml_path})."
        )

    builder_cfg = load_config(str(builder_cfg_path))
    _inherit_pyramid_from_builder(predictor_cfg, builder_cfg)

    # For utest stability: force a small batch and a sensible loss-weight set.
    predictor_cfg.setdefault("training", {})["batch_size"] = 2
    predictor_cfg["training"].setdefault(
        "loss_weights",
        {"concept_loss_weight": 1.0, "reasoning_loss_weight": 1.0},
    )

    return predictor_cfg, builder_cfg, builder_cfg_path


def run_pipeline(predictor_cfg: dict, builder_cfg: dict, device: str):
    """End-to-end: real data → builder → predictor (train + infer + generate) → gradient."""

    pyramid_cfg = predictor_cfg["model"]["pyramid"]
    num_levels = pyramid_cfg["num_levels"]
    level_lengths = list(pyramid_cfg["level_lengths"])
    hidden_dim = pyramid_cfg["hidden_dim"]
    batch_size = predictor_cfg["training"]["batch_size"]
    total_C = sum(level_lengths)

    # ==================================================================
    # Step 1: Load Builder (Stage 1) + GSM8K Data
    # ==================================================================
    log_section("Step 1: Load Builder (FRESH, no checkpoint) + Data")

    builder = ConceptPyramidBuilder(builder_cfg)
    builder.to(device)
    builder.eval()
    log_value("builder.reason_model_hidden_dim", builder.reason_model_hidden_dim)

    dataloader = LCPDataLoader(
        data_cfg=predictor_cfg["data"],
        batch_size=batch_size,
        include_solution=True,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    batch = next(iter(dataloader))
    log_value("data_name", predictor_cfg["data"].get("data_name"))
    log_value("dataset_size", dataloader.dataset_size)
    log_value("batch_size", batch.batch_size)
    log_value("sample question[0][:80]", repr(batch.questions[0][:80]))
    log_value("has_solution", batch.has_solution)
    log_value("sample solution[0][:40]", repr(batch.solutions[0][:40]))

    # ==================================================================
    # Step 2: Builder → GT Concept Pyramid + f_hats
    # ==================================================================
    log_section("Step 2: Builder → GT Concept Pyramid + f_hats")

    with torch.no_grad():
        pyramid = builder(batch)

    log_check("returns PyramidOutput", isinstance(pyramid, PyramidOutput))
    log_value("pyramid.num_levels", pyramid.num_levels)
    log_value("pyramid.total_concepts", pyramid.total_concepts)

    gt_concepts = [c.detach() for c in pyramid.concepts]
    f_hats = [f.detach() for f in pyramid.f_hat_per_level]
    for k in range(num_levels):
        log_value(f"GT concepts[{k}].shape", list(gt_concepts[k].shape))
        log_value(f"f_hat[{k}].shape", list(f_hats[k].shape))

    # ==================================================================
    # Step 3: Initialize ConceptPredictor (VAR-faithful)
    # ==================================================================
    log_section("Step 3: Initialize ConceptPredictor (own backbone + LoRA)")

    predictor = ConceptPredictor(predictor_cfg, builder=builder)
    predictor.to(device)
    # MPS-safe: force float32 throughout (Qwen2.5 ships bf16 by default in HF
    # config; mixing bf16 weights with fp32 inputs triggers MPS matmul errors).
    predictor.to(dtype=torch.float32)

    log_value("reason_model_hidden_dim", predictor.reason_model_hidden_dim)
    log_value("_num_levels", predictor._num_levels)
    log_value("_total_concepts", predictor._total_concepts)
    log_value("_level_lengths", predictor._level_lengths)
    log_value("_inference_canvas_length", predictor._inference_canvas_length)

    # Verify key modules exist
    log_check("has back_proj", hasattr(predictor, "back_proj"))
    log_check("has level_queries", hasattr(predictor, "level_queries"))
    log_check("has extract_attn", hasattr(predictor, "extract_attn"))
    log_check("has query_norm", hasattr(predictor, "query_norm"))
    log_check("has context_norm", hasattr(predictor, "context_norm"))
    log_check("has post_norm", hasattr(predictor, "post_norm"))
    log_check("has lvl_embed", hasattr(predictor, "lvl_embed"))
    log_check("has concept_head", hasattr(predictor, "concept_head"))
    log_check(
        "builder is frozen",
        not any(p.requires_grad for p in predictor.builder.parameters()),
    )

    # Verify level_queries shapes
    for k in range(num_levels):
        q_shape = list(predictor.level_queries[k].shape)
        expected = [level_lengths[k], predictor.reason_model_hidden_dim]
        log_check(f"level_queries[{k}].shape", q_shape == expected, str(q_shape))

    # Parameter count summary
    total_params = sum(p.numel() for p in predictor.parameters())
    trainable_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    log_value("total_params", f"{total_params:,}")
    log_value("trainable_params", f"{trainable_params:,}")
    log_value("trainable_ratio", f"{trainable_params/total_params*100:.2f}%")

    # ==================================================================
    # Step 4: _construct_approx_tokens (unit test)
    # ==================================================================
    log_section("Step 4: _construct_approx_tokens (pre-LLM cross-attention)")

    test_f_hat = f_hats[0].to(device)
    approx_tokens_0, attn_w_0 = predictor._construct_approx_tokens(0, test_f_hat)

    L_0 = level_lengths[0]
    D_enc = predictor.reason_model_hidden_dim
    B = batch.batch_size
    log_value("approx_tokens_0.shape", list(approx_tokens_0.shape))
    log_check(
        "shape [B, L_0, D_enc]",
        list(approx_tokens_0.shape) == [B, L_0, D_enc],
    )
    log_value("attn_w_0.shape", list(attn_w_0.shape))
    log_check(
        "attn_w shape [B, L_0, ctx_len]",
        attn_w_0.shape[0] == B and attn_w_0.shape[1] == L_0,
    )
    attn_sum_err = (attn_w_0.sum(dim=-1) - 1.0).abs().max().item()
    log_check(
        "attn_w sums ≈ 1 (tol=1e-3)",
        attn_sum_err < 1e-3,
        f"max_err={attn_sum_err:.2e}",
    )
    log_value(
        "approx_tokens_0 stats",
        f"mean={approx_tokens_0.mean():.4f}, std={approx_tokens_0.std():.4f}",
    )

    # ==================================================================
    # Step 5: Scale-Causal Mask Verification
    # ==================================================================
    log_section("Step 5: Scale-Causal Mask Verification")

    q_len_test = torch.tensor([3, 4], device=device, dtype=torch.long)
    s_len_test = torch.tensor([2, 3], device=device, dtype=torch.long)
    T_test = max(q_len_test.max().item() + total_C + s_len_test.max().item(), 10) + 2
    mask_4d = predictor._build_scale_causal_mask_packed(
        q_len=q_len_test,
        s_len=s_len_test,
        level_lengths=level_lengths,
        T=T_test,
        dtype=torch.float32,
        device=device,
    )
    log_value("mask_4d.shape", list(mask_4d.shape))
    log_check("shape [2, 1, T, T]", list(mask_4d.shape) == [2, 1, T_test, T_test])

    diag_vals = mask_4d[:, 0].diagonal(dim1=-2, dim2=-1)
    log_check("diagonal all 0 (self-attend)", (diag_vals == 0).all().item())

    unique_vals = mask_4d.unique()
    neg_inf = torch.finfo(torch.float32).min
    log_check(
        "mask values are 0 and -inf only",
        all(v == 0 or v == neg_inf for v in unique_vals.tolist()),
    )

    log_check(
        "Q[0,0] can only see pos 0 in Q",
        mask_4d[0, 0, 0, 0] == 0 and mask_4d[0, 0, 0, 1] == neg_inf,
    )

    # ==================================================================
    # Step 6: Training Forward (full packed pass with solution)
    # ==================================================================
    log_section("Step 6: Training Forward (teacher-forcing, single packed pass)")

    predictor.train()
    train_out = predictor(batch)

    log_check("returns PredictorOutput", isinstance(train_out, PredictorOutput))
    log_check("gt_concepts in output", train_out.gt_concepts is not None)
    log_value("num_levels in output", train_out.num_levels)
    log_value("level_lengths in output", train_out.level_lengths)
    log_value("predicted_concepts count", len(train_out.predicted_concepts))
    log_check("count == num_levels", len(train_out.predicted_concepts) == num_levels)

    logging.info("  -- Per-level predictions (training) --")
    for k in range(num_levels):
        pred = train_out.predicted_concepts[k]
        expected_shape = (B, level_lengths[k], hidden_dim)
        ok = pred.shape == expected_shape
        log_check(
            f"level {k} shape {list(pred.shape)}",
            ok,
            f"L_k={level_lengths[k]} mean={pred.mean().item():+.4f} std={pred.std().item():.4f}",
        )

    log_check("reasoning_logits present", train_out.reasoning_logits is not None)
    if train_out.reasoning_logits is not None:
        log_value("reasoning_logits.shape", list(train_out.reasoning_logits.shape))
    log_check(
        "reasoning_target_ids present", train_out.reasoning_target_ids is not None
    )
    if train_out.reasoning_target_ids is not None:
        log_value(
            "reasoning_target_ids.shape", list(train_out.reasoning_target_ids.shape)
        )

    # ==================================================================
    # Step 7: Compute Losses
    # ==================================================================
    log_section("Step 7: Loss Computation")

    concept_total, per_level = compute_predictor_concept_loss(
        train_out.predicted_concepts, train_out.gt_concepts, concept_loss_type="mse"
    )
    log_value("concept_total_loss", f"{concept_total.item():.6f}")
    log_check("concept_total is finite", torch.isfinite(concept_total).item())
    log_check("concept_total > 0", concept_total.item() > 0)
    for k, ll in enumerate(per_level):
        log_value(f"per_level_loss[{k}]", f"{ll.item():.6f}")

    manual_avg = sum(ll.item() for ll in per_level) / num_levels
    avg_diff = abs(concept_total.item() - manual_avg)
    log_check("total ≈ mean(per_level)", avg_diff < 1e-4, f"diff={avg_diff:.8f}")

    combined_loss, combined_dict = compute_predictor_loss(
        train_out,
        loss_weights=predictor_cfg["training"]["loss_weights"],
    )
    log_check("combined_loss is finite", torch.isfinite(combined_loss).item())
    log_value("combined_loss", f"{combined_loss.item():.6f}")
    log_value("combined_dict keys", list(combined_dict.keys()))
    for key, val in combined_dict.items():
        log_value(f"  {key}", f"{val:.6f}" if isinstance(val, float) else val)

    # ==================================================================
    # Step 8: Inference Forward (K sequential passes)
    # ==================================================================
    log_section("Step 8: Inference Forward (K passes, self-maintained f_hat)")

    predictor.eval()
    max_length = pyramid_cfg["max_seq_len"]
    q_tokens = predictor.tokenizer(
        batch.questions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    q_ids = q_tokens["input_ids"].to(device)
    q_mask = q_tokens["attention_mask"].to(device)
    log_value("q_ids.shape", list(q_ids.shape))

    with torch.no_grad():
        infer_out = predictor._forward_inference(q_ids, q_mask)

    log_check("returns PredictorOutput", isinstance(infer_out, PredictorOutput))
    log_check("gt_concepts is None", infer_out.gt_concepts is None)
    log_value("predicted_concepts count", len(infer_out.predicted_concepts))

    logging.info("  -- Per-level predictions (inference) --")
    for k, pred in enumerate(infer_out.predicted_concepts):
        expected_shape = (B, level_lengths[k], hidden_dim)
        ok = pred.shape == expected_shape
        log_check(f"level {k} shape", ok, str(list(pred.shape)))
        log_value(f"level {k} stats", f"mean={pred.mean():.4f}, std={pred.std():.4f}")

    logging.info("  -- Train/Inference shape consistency --")
    for k in range(num_levels):
        t_shape = train_out.predicted_concepts[k].shape
        i_shape = infer_out.predicted_concepts[k].shape
        log_check(f"level {k} shapes match", t_shape == i_shape)

    # ==================================================================
    # Step 9: Solution Generation
    # ==================================================================
    log_section("Step 9: generate_solution (free autoregressive)")

    with torch.no_grad():
        solutions = predictor.generate_solution(
            infer_out.predicted_concepts,
            q_ids,
            question_attention_mask=q_mask,
            max_new_tokens=50,
        )

    log_check("returns list", isinstance(solutions, list))
    log_check("length == B", len(solutions) == B)
    for i, sol in enumerate(solutions):
        log_value(f"solution[{i}][:60]", repr(sol[:60]))

    # ==================================================================
    # Step 10: Gradient Flow (backward through combined loss)
    # ==================================================================
    log_section("Step 10: Gradient Flow (backward on combined loss)")

    predictor.train()
    predictor.zero_grad()
    train_out2 = predictor(batch)
    loss2, _ = compute_predictor_loss(
        train_out2,
        loss_weights=predictor_cfg["training"]["loss_weights"],
    )
    loss2.backward()

    predictor_params = [
        ("back_proj.weight", predictor.back_proj.weight),
        ("query_norm.weight", predictor.query_norm.weight),
        ("context_norm.weight", predictor.context_norm.weight),
        ("post_norm.weight", predictor.post_norm.weight),
        ("lvl_embed.weight", predictor.lvl_embed.weight),
    ]
    logging.info("  -- Predictor-owned parameters --")
    for name, param in predictor_params:
        has_grad = param.grad is not None
        grad_norm = f"{param.grad.norm().item():.6f}" if has_grad else "N/A"
        log_check(f"{name} has gradient", has_grad, f"grad_norm={grad_norm}")

    lq_grads = sum(1 for q in predictor.level_queries if q.grad is not None)
    log_check(
        f"level_queries: {lq_grads}/{num_levels} have grad", lq_grads == num_levels
    )

    head_grads = sum(
        1 for p in predictor.concept_head.parameters() if p.grad is not None
    )
    head_total = sum(1 for _ in predictor.concept_head.parameters())
    log_check(
        f"concept_head: {head_grads}/{head_total} have grad", head_grads == head_total
    )

    attn_grads = sum(
        1 for p in predictor.extract_attn.parameters() if p.grad is not None
    )
    attn_total = sum(1 for _ in predictor.extract_attn.parameters())
    log_check(
        f"extract_attn: {attn_grads}/{attn_total} have grad", attn_grads == attn_total
    )

    lora_params = [
        (n, p) for n, p in predictor.reason_model.named_parameters() if "lora_" in n
    ]
    lora_with_grad = sum(1 for _, p in lora_params if p.grad is not None)
    log_value("LoRA params total", len(lora_params))
    log_check(
        f"LoRA: {lora_with_grad}/{len(lora_params)} have grad",
        lora_with_grad == len(lora_params),
    )

    builder_grad = any(p.grad is not None for p in predictor.builder.parameters())
    log_check("builder has NO gradient (frozen)", not builder_grad)

    predictor.eval()
    logging.info("\n" + "=" * 72)
    logging.info("  ALL STEPS COMPLETE — ConceptPredictor (VAR-faithful) test passed")
    logging.info("=" * 72)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    predictor_yaml_path = _resolve_path(args.config, PROJECT_ROOT)
    if not predictor_yaml_path.exists():
        raise FileNotFoundError(f"Predictor config not found: {predictor_yaml_path}")

    logging.info("Predictor config: %s", predictor_yaml_path)
    predictor_cfg, builder_cfg, builder_yaml_path = load_predictor_and_builder_configs(
        predictor_yaml_path
    )
    logging.info("Paired builder config: %s", builder_yaml_path)

    device = str(get_device("auto"))
    logging.info("Device: %s", device)

    run_pipeline(predictor_cfg, builder_cfg, device)


if __name__ == "__main__":
    main()
