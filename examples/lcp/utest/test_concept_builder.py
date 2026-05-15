"""ConceptPyramidBuilder end-to-end pipeline test with real GSM8K data.

Run with:
    python3 examples/lcp/utest/test_concept_builder.py \
        -c configs/lcp/GSM8K/train_builder_Qwen2.5-0.5B_6level.yml

DESIGN PHILOSOPHY:
    Single end-to-end pipeline driven by real GSM8K CoT data.
    Uses the same config files as real training (configs/lcp/GSM8K/).
    Each pipeline step is verified and logged in detail.
    Diagnostic (log-based), not assertion-based — randomly initialized
    weights produce stochastic outputs before training.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

from lcp.concept_builder import (
    ConceptPyramidBuilder,
    EncoderOutput,
    LevelOutput,
    PyramidOutput,
)
from lcp.data_loader import LCPDataLoader
from lcp.losses import compute_builder_loss
from lmbase.utils.env_tools import get_device
from ram.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="ConceptPyramidBuilder test")
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
    """Single end-to-end pipeline: real data → encode → pyramid → loss → gradient."""

    pyramid_cfg = config["model"]["pyramid"]
    num_levels = pyramid_cfg["num_levels"]
    level_lengths = pyramid_cfg["level_lengths"]
    hidden_dim = pyramid_cfg["hidden_dim"]
    batch_size = config["training"]["batch_size"]

    # ==================================================================
    # Step 1: Load Builder
    # ==================================================================
    log_section("Step 1: Load ConceptPyramidBuilder")

    builder = ConceptPyramidBuilder(config)
    builder.to(device)
    log_value("reason_model_hidden_dim", builder.reason_model_hidden_dim)
    log_value("pyramid.hidden_dim", hidden_dim)
    log_value("pyramid.num_levels", num_levels)
    log_value("pyramid.level_lengths", level_lengths)

    # Verify key components exist
    log_check("input_proj exists", hasattr(builder, "input_proj"))
    log_check("input_proj_norm exists", hasattr(builder, "input_proj_norm"))
    log_check("concept_queries count", len(builder.concept_queries) == num_levels)
    log_check("level_projs count", len(builder.level_projs) == num_levels)
    log_check("temperature is scalar", builder.temperature.numel() == 1)
    log_check("back_proj exists", builder.back_proj is not None)
    if builder.back_proj is not None:
        log_value(
            "back_proj.shape",
            f"[{builder.back_proj.in_features}, {builder.back_proj.out_features}]",
        )
        log_check(
            "back_proj: D → D_encoder",
            builder.back_proj.in_features == hidden_dim
            and builder.back_proj.out_features == builder.reason_model_hidden_dim,
        )
        # Pseudo-inverse init check
        weight_diff = (
            (builder.back_proj.weight - builder.input_proj.weight.T).abs().max().item()
        )
        log_check(
            "back_proj ≈ input_proj.T (pseudo-inverse init)",
            weight_diff < 1e-6,
            f"max_diff={weight_diff:.2e}",
        )

    # ==================================================================
    # Step 2: Load Real GSM8K Data
    # ==================================================================
    log_section("Step 2: Load GSM8K Data via LCPDataLoader")

    dataloader = LCPDataLoader(
        data_cfg=config["data"],
        batch_size=batch_size,
        include_solution=True,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    batch = next(iter(dataloader))
    log_value("dataset_size", dataloader.dataset_size)
    log_value("batch.batch_size", batch.batch_size)
    log_value("sample cot_answer[0][:80]", repr(batch.cot_answers[0][:80]))
    log_value("sample question[0][:80]", repr(batch.questions[0][:80]))
    log_check(
        "batch has questions", batch.questions is not None and len(batch.questions) > 0
    )
    log_check(
        "batch has cot_answers",
        batch.cot_answers is not None and len(batch.cot_answers) > 0,
    )
    log_check(
        "batch has solutions", batch.solutions is not None and len(batch.solutions) > 0
    )

    # ==================================================================
    # Step 3: encode_cot — CoT → Hidden States
    # ==================================================================
    log_section("Step 3: encode_cot(real CoT)")

    enc_out = builder.encode_cot(batch.cot_answers)
    H = enc_out.hidden_states
    log_check("returns EncoderOutput", isinstance(enc_out, EncoderOutput))
    log_value("H.shape (hidden_states)", list(H.shape))
    log_check("H is 3D [B, L, D_encoder]", H.dim() == 3)
    log_check("batch dim correct", H.shape[0] == batch.batch_size)
    log_check(
        "last dim == reason_model_hidden_dim",
        H.shape[-1] == builder.reason_model_hidden_dim,
    )
    log_check("attention_mask exists", enc_out.attention_mask is not None)
    if enc_out.attention_mask is not None:
        valid_tokens = enc_out.attention_mask.sum(dim=1).tolist()
        log_value("valid tokens per sample", [int(v) for v in valid_tokens])
    log_value("H stats: mean", f"{H.mean().item():.4f}")
    log_value("H stats: std", f"{H.std().item():.4f}")
    log_value("H stats: min/max", f"{H.min().item():.4f} / {H.max().item():.4f}")

    # ==================================================================
    # Step 4: forward() — Build Concept Pyramid
    # ==================================================================
    log_section("Step 4: forward() — Build Concept Pyramid")

    builder.eval()
    pyramid = builder(batch)
    log_check("returns PyramidOutput", isinstance(pyramid, PyramidOutput))
    log_value("num_levels", pyramid.num_levels)
    log_value("level_lengths", pyramid.level_lengths)
    log_value("total_concepts", pyramid.total_concepts)
    log_check("concepts count == num_levels", len(pyramid.concepts) == num_levels)
    log_check(
        "level_outputs count == num_levels", len(pyramid.level_outputs) == num_levels
    )

    seq_len = H.shape[1]
    logging.info("  -- Per-level detail --")
    for k, lo in enumerate(pyramid.level_outputs):
        Lk = level_lengths[k]
        logging.info("  Level %d (L_k=%d):", k, Lk)
        log_check(f"  isinstance LevelOutput", isinstance(lo, LevelOutput))
        log_check(
            f"  concepts.shape", lo.concepts.shape == (batch.batch_size, Lk, hidden_dim)
        )
        log_check(
            f"  attention_weights.shape",
            lo.attention_weights.shape == (batch.batch_size, Lk, seq_len),
        )
        log_check(
            f"  reconstruction.shape",
            lo.reconstruction.shape == (batch.batch_size, seq_len, hidden_dim),
        )
        # Attention softmax check
        attn_sum = lo.attention_weights.sum(dim=-1)
        max_dev = (attn_sum - 1.0).abs().max().item()
        log_check(f"  attention sums to 1", max_dev < 1e-5, f"max_dev={max_dev:.2e}")
        # Concept stats
        c_mean = lo.concepts.mean().item()
        c_std = lo.concepts.std().item()
        log_value(f"  concepts stats", f"mean={c_mean:.4f}, std={c_std:.4f}")

    # Residual decomposition identity: f_hat + f_rest == H_proj
    logging.info("  -- Residual decomposition --")
    recomposed = pyramid.reconstructed_hidden + pyramid.residual_hidden
    diff = torch.abs(recomposed - pyramid.projected_hidden).max().item()
    log_value("max |f_hat + f_rest - H_proj|", f"{diff:.2e}")
    log_check("residual identity holds", diff < 1e-3, f"diff={diff:.2e}")
    log_value("residual_hidden norm", f"{pyramid.residual_hidden.norm().item():.4f}")

    # cat_concepts
    cat = pyramid.cat_concepts()
    expected_cat = (batch.batch_size, sum(level_lengths), hidden_dim)
    log_check("cat_concepts shape", cat.shape == expected_cat)

    # ==================================================================
    # Step 5: compute_builder_loss — Loss Breakdown
    # ==================================================================
    log_section("Step 5: Loss Breakdown (recon + ordering + residual + reasoning)")

    builder.train()
    # Re-forward in train mode for gradient tracking
    pyramid = builder(batch)

    loss_weights = config["training"]["loss_weights"]
    ordering_loss_type = config["training"]["ordering_loss_type"]

    total_loss, loss_dict = compute_builder_loss(
        pyramid, loss_weights, ordering_loss_type=ordering_loss_type
    )

    recon_w = loss_weights["recon_loss_weight"]
    ordering_w = loss_weights["ordering_loss_weight"]
    residual_w = loss_weights["residual_loss_weight"]
    reasoning_w = loss_weights["reasoning_loss_weight"]

    logging.info(
        "  ┌─────────────────────────────────────────────────────────────────────┐"
    )
    logging.info(
        "  │ Component              │ Raw Loss     │ Weight   │ Weighted Value  │"
    )
    logging.info(
        "  ├─────────────────────────────────────────────────────────────────────┤"
    )
    logging.info(
        "  │ recon_loss             │ %11.4f  │ %7.3f  │ %13.4f   │",
        loss_dict["recon"],
        recon_w,
        loss_dict["recon"] * recon_w,
    )
    logging.info(
        "  │ ordering_loss          │ %11.4f  │ %7.3f  │ %13.4f   │",
        loss_dict["ordering"],
        ordering_w,
        loss_dict["ordering"] * ordering_w,
    )
    logging.info(
        "  │ residual_loss          │ %11.4f  │ %7.3f  │ %13.4f   │",
        loss_dict["residual"],
        residual_w,
        loss_dict["residual"] * residual_w,
    )
    reasoning_raw = loss_dict.get("reasoning", 0.0)
    logging.info(
        "  │ reasoning_loss (NTP)   │ %11.4f  │ %7.3f  │ %13.4f   │",
        reasoning_raw,
        reasoning_w,
        reasoning_raw * reasoning_w,
    )
    logging.info(
        "  ├─────────────────────────────────────────────────────────────────────┤"
    )
    logging.info(
        "  │ TOTAL                  │             │          │ %13.4f   │",
        loss_dict["total"],
    )
    logging.info(
        "  └─────────────────────────────────────────────────────────────────────┘"
    )

    log_check("recon loss is finite", loss_dict["recon"] == loss_dict["recon"])
    log_check("ordering loss is finite", loss_dict["ordering"] == loss_dict["ordering"])
    log_check("residual loss >= 0", loss_dict["residual"] >= 0)
    log_check("reasoning loss is finite", reasoning_raw == reasoning_raw)
    log_check("reasoning loss is positive", reasoning_raw > 0)
    log_check(
        "reasoning_texts populated",
        pyramid.reasoning_texts is not None
        and len(pyramid.reasoning_texts) == batch.batch_size,
    )
    logging.info("  Sample reasoning_texts[0]: %s", pyramid.reasoning_texts[0][:200])

    # ==================================================================
    # Step 6: Gradient Flow — backward through combined loss
    # ==================================================================
    log_section("Step 6: Gradient Flow (backward on total_loss)")

    builder.zero_grad()
    total_loss.backward()

    grad_report = {
        "input_proj.weight": builder.input_proj.weight,
        "temperature": builder.temperature,
        "back_proj.weight": (
            builder.back_proj.weight if builder.back_proj is not None else None
        ),
    }
    for name, param in grad_report.items():
        if param is None:
            continue
        has_grad = param.grad is not None
        grad_norm = f"{param.grad.norm().item():.6f}" if has_grad else "N/A"
        log_check(f"{name} has gradient", has_grad)
        log_value(f"{name} grad norm", grad_norm)

    for k in range(num_levels):
        q = builder.concept_queries[k]
        p = builder.level_projs[k].weight
        q_grad = q.grad is not None
        p_grad = p.grad is not None
        log_check(
            f"level {k}: queries grad={q_grad}, projs grad={p_grad}",
            q_grad and p_grad,
        )

    backbone_grad = any(p.grad is not None for p in builder.reason_model.parameters())
    log_check(
        "reason_model has gradients (only if un-frozen or LoRA)",
        backbone_grad,
        "Expected WARN if freeze=True and no LoRA",
    )

    builder.eval()
    logging.info("\n=== BUILDER PIPELINE TEST COMPLETE ===")


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
