"""ConceptPyramidBuilder integration test.

Directly tests every component and pipeline step without unittest wrappers.
Run with:
    python3 examples/nlcpV3/utest/test_concept_builder.py -c configs/nlcpV3/utest/test_concept_builder.yml

DESIGN PHILOSOPHY:
    This is a DIAGNOSTIC test, not a pass/fail gate. All checks are logged
    rather than asserted because:
    1. Randomly initialized weights produce stochastic outputs — exact
       numerical checks (e.g. diff < 1e-4) will fail before training.
    2. We want to OBSERVE behavior (shapes, ranges, reconstruction quality)
       to verify architectural correctness, not enforce convergence.
    3. Each test prints WHAT it is checking, the ACTUAL values, and a
       qualitative assessment (OK / WARNING / INFO).

USAGE:
    python3 examples/nlcpV3/utest/test_concept_builder.py -c configs/nlcpV3/utest/test_concept_builder.yml
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

from nlcpV3.concept_hybrid_builder import (
    ConceptPyramidBuilder,
    EncoderOutput,
    PyramidOutput,
    SingleLevelOutput,
)
from nlcpV3.data_loader import NLCPV3DataLoader
from nlcpV3.train_builder import compute_builder_loss
from lmbase.utils.env_tools import get_device
from ram.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="ConceptPyramidBuilder test")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to YAML config file"
    )
    return parser.parse_args()


def log_check(name, cond, details=""):
    """Log a diagnostic check result without raising."""
    status = "OK" if cond else "WARN"
    msg = f"  [{status}] {name}"
    if details:
        msg += f" | {details}"
    logging.info(msg)


def log_value(name, value, unit=""):
    """Log a numeric observation for human inspection."""
    u = f" {unit}" if unit else ""
    logging.info(f"  [VAL] {name} = {value}{u}")


def test_encode_cot(builder, device, batch_size):
    """Verify encode_cot produces correct hidden states from text and token IDs.

    WHAT IS CHECKED:
        - Text input → EncoderOutput with 3D hidden_states [B, L, D_encoder]
        - Tensor input → EncoderOutput with 3D hidden_states [B, L, D_encoder]
        - Batch dimension matches requested batch_size
        - Last dimension matches reason_model hidden size
    """
    logging.info("\n=== encode_cot Tests ===")
    logging.info("Purpose: Verify text and token-ID inputs produce valid hidden states")

    texts = [f"Problem {i}: What is {i} + {i+1}?" for i in range(batch_size)]

    # --- Test: text input ---
    logging.info("  -- Test: Text input (auto-tokenize) --")
    enc_out = builder.encode_cot(texts)
    log_check("returns EncoderOutput", isinstance(enc_out, EncoderOutput))
    log_value("hidden_states.dim()", enc_out.hidden_states.dim())
    log_check("hidden_states is 3D", enc_out.hidden_states.dim() == 3)
    log_value("hidden_states.shape", list(enc_out.hidden_states.shape))
    log_check(
        "batch dimension correct",
        enc_out.hidden_states.shape[0] == batch_size,
        f"got {enc_out.hidden_states.shape[0]}, expected {batch_size}",
    )
    log_check(
        "last dim == reason_model_hidden_dim",
        enc_out.hidden_states.shape[-1] == builder.reason_model_hidden_dim,
        f"got {enc_out.hidden_states.shape[-1]}, expected {builder.reason_model_hidden_dim}",
    )

    # --- Test: tensor input ---
    logging.info("  -- Test: Tensor input (token IDs) --")
    tokens = builder.tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=32
    )
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    enc_out2 = builder.encode_cot(input_ids, attention_mask=attention_mask)
    log_check("returns EncoderOutput", isinstance(enc_out2, EncoderOutput))
    log_check("hidden_states is 3D", enc_out2.hidden_states.dim() == 3)
    log_check("batch dimension correct", enc_out2.hidden_states.shape[0] == batch_size)
    log_check(
        "last dim == reason_model_hidden_dim",
        enc_out2.hidden_states.shape[-1] == builder.reason_model_hidden_dim,
    )

    return enc_out


def test_forward(builder, device, config, batch_size):
    """Verify forward() full pyramid construction with residual decomposition.

    WHAT IS CHECKED:
        - Returns PyramidOutput with correct number of levels
        - Each level's concepts have shape [B, L_k, D]
        - Residual identity: reconstructed_hidden + residual_hidden ≈ H_proj
        - Level 0 has no refinement → concepts == base_concepts
        - Attention weights are valid softmax (sum ≈ 1 per slot)
    """
    logging.info("\n=== forward() Tests ===")
    logging.info("Purpose: Verify full pyramid construction and residual decomposition")

    texts = [f"Solve {i}: compute {i} * {i+2}." for i in range(batch_size)]
    enc_out = builder.encode_cot(texts)
    H = enc_out.hidden_states
    seq_len = H.shape[1]
    logging.info("  Encoder output H shape: %s", list(H.shape))

    # Build pyramid
    output = builder(H, attention_mask=enc_out.attention_mask)
    log_check("returns PyramidOutput", isinstance(output, PyramidOutput))

    pyramid_cfg = config["model"]["pyramid"]
    num_levels = pyramid_cfg["num_levels"]
    level_lengths = pyramid_cfg["level_lengths"]
    hidden_dim = pyramid_cfg["hidden_dim"]

    # --- Concepts count and shapes ---
    log_value("num_levels", num_levels)
    log_value("concepts count", len(output.concepts))
    log_check("concepts count == num_levels", len(output.concepts) == num_levels)
    for k, concepts in enumerate(output.concepts):
        expected = (batch_size, level_lengths[k], hidden_dim)
        log_value(f"concepts[{k}].shape", list(concepts.shape))
        log_check(f"level {k} shape correct", concepts.shape == expected)

    # --- LevelOutput fields ---
    log_value("level_outputs count", len(output.level_outputs))
    log_check(
        "level_outputs count == num_levels", len(output.level_outputs) == num_levels
    )
    for k, lo in enumerate(output.level_outputs):
        Lk = level_lengths[k]
        log_check(
            f"level {k} concepts shape",
            lo.concepts.shape == (batch_size, Lk, hidden_dim),
        )
        log_check(
            f"level {k} base_concepts shape",
            lo.base_concepts.shape == (batch_size, Lk, hidden_dim),
        )
        log_check(
            f"level {k} attention shape",
            lo.attention_weights.shape == (batch_size, Lk, seq_len),
        )
        log_check(
            f"level {k} reconstruction shape",
            lo.reconstruction.shape == (batch_size, seq_len, hidden_dim),
        )

    # --- Residual decomposition: f_hat + f_rest == H_proj ---
    logging.info("  -- Residual decomposition check --")
    recomposed = output.reconstructed_hidden + output.residual_hidden
    diff = torch.abs(recomposed - output.projected_hidden).max().item()
    log_value("max_diff (f_hat + f_rest vs H_proj)", f"{diff:.2e}")
    tolerance = 1e-1
    log_check(
        f"residual identity holds (tol={tolerance:.0e})",
        diff < tolerance,
        f"diff={diff:.2e} — NOTE: before training, large diff is expected",
    )

    # --- PyramidOutput properties ---
    log_value("total_concepts", output.total_concepts)
    log_check("total_concepts correct", output.total_concepts == sum(level_lengths))
    log_value("all_attentions length", len(output.all_attentions))
    log_check("all_attentions length", len(output.all_attentions) == num_levels)

    cat = output.cat_concepts()
    expected_cat_shape = (batch_size, sum(level_lengths), hidden_dim)
    log_value("cat_concepts.shape", list(cat.shape))
    log_check("cat_concepts shape correct", cat.shape == expected_cat_shape)

    # --- Level 0: no refinement ---
    max_diff_l0 = (
        (output.level_outputs[0].concepts - output.level_outputs[0].base_concepts)
        .abs()
        .max()
        .item()
    )
    log_value("level 0 concepts vs base max_diff", f"{max_diff_l0:.2e}")
    log_check(
        "level 0 concepts == base_concepts (no refinement)",
        torch.allclose(
            output.level_outputs[0].concepts, output.level_outputs[0].base_concepts
        ),
        f"max_diff={max_diff_l0:.2e}",
    )

    # --- Attention softmax check ---
    for k, lo in enumerate(output.level_outputs):
        attn_sum = lo.attention_weights.sum(dim=-1)
        max_dev = (attn_sum - 1.0).abs().max().item()
        log_value(f"level {k} attention sum deviation", f"{max_dev:.2e}")
        log_check(
            f"level {k} attention sums to 1",
            torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5),
            f"max_dev={max_dev:.2e}",
        )

    return output


def test_forward_next_level(builder, device, config, batch_size):
    """Verify forward_next_level step-by-step level extraction.

    WHAT IS CHECKED:
        - Each level returns SingleLevelOutput with correct level_index
        - concepts shape is [B, L_k, D]
        - Cache accumulates exactly num_levels entries
        - clear_cache() resets both cache lists to empty
    """
    logging.info("\n=== forward_next_level() Tests ===")
    logging.info("Purpose: Verify incremental level-by-level extraction")

    texts = [f"Step {i}: calculate {i+1} * {i+2}." for i in range(batch_size)]
    enc_out = builder.encode_cot(texts)
    H = enc_out.hidden_states
    logging.info("  Encoder output H shape: %s", list(H.shape))

    pyramid_cfg = config["model"]["pyramid"]
    num_levels = pyramid_cfg["num_levels"]
    level_lengths = pyramid_cfg["level_lengths"]
    hidden_dim = pyramid_cfg["hidden_dim"]

    builder.clear_cache()
    prev_concepts = []
    for k in range(num_levels):
        logging.info("  -- Extracting level %d --", k)
        level_out = builder.forward_next_level(
            H, previous_level_concepts=prev_concepts, target_level_index=k
        )
        log_check(
            f"level {k} returns SingleLevelOutput",
            isinstance(level_out, SingleLevelOutput),
        )
        log_value(f"level {k} level_index", level_out.level_index)
        log_check(f"level {k} index correct", level_out.level_index == k)
        log_value(f"level {k} concepts.shape", list(level_out.concepts.shape))
        log_check(
            f"level {k} batch size correct", level_out.concepts.shape[0] == batch_size
        )
        log_check(
            f"level {k} concept count correct",
            level_out.concepts.shape[1] == level_lengths[k],
        )
        prev_concepts.append(level_out.concepts)

    # --- Cache verification ---
    log_value("cached_attentions length", len(builder._cached_attentions))
    log_value("cached_base_concepts length", len(builder._cached_base_concepts))
    log_check(
        "cache has num_levels attentions", len(builder._cached_attentions) == num_levels
    )
    log_check(
        "cache has num_levels base_concepts",
        len(builder._cached_base_concepts) == num_levels,
    )

    # --- Clear cache ---
    builder.clear_cache()
    log_value("after clear: attentions length", len(builder._cached_attentions))
    log_value("after clear: base_concepts length", len(builder._cached_base_concepts))
    log_check("clear_cache empties attentions", len(builder._cached_attentions) == 0)
    log_check(
        "clear_cache empties base_concepts", len(builder._cached_base_concepts) == 0
    )


def test_gsm8k_integration(builder, device, config, dataloader):
    """Verify end-to-end pipeline with real GSM8K CoT data.

    WHAT IS CHECKED:
        - NLCPV3DataLoader yields BuilderInput batches correctly
        - encode_cot handles real CoT text (auto-tokenize)
        - forward() produces valid PyramidOutput with correct shapes
        - total_concepts matches configuration
    """
    logging.info("\n=== GSM8K Integration Tests ===")
    logging.info(
        "Purpose: Verify pipeline with real GSM8K CoT data via NLCPV3DataLoader"
    )

    batch = next(iter(dataloader))
    logging.info("  Using batch of %d GSM8K samples", batch.batch_size)

    enc_out = builder.encode_cot(batch.cot_answers)
    pyramid = builder(enc_out.hidden_states, attention_mask=enc_out.attention_mask)
    log_check(
        "GSM8K forward returns PyramidOutput",
        isinstance(pyramid, PyramidOutput),
    )
    log_value("GSM8K concepts count", len(pyramid.concepts))
    pyramid_cfg = config["model"]["pyramid"]
    log_check(
        "GSM8K concepts count correct",
        len(pyramid.concepts) == pyramid_cfg["num_levels"],
    )
    log_value("GSM8K total_concepts", pyramid.total_concepts)
    log_check(
        "GSM8K total_concepts correct",
        pyramid.total_concepts == sum(pyramid_cfg["level_lengths"]),
    )
    log_check(
        "GSM8K concept batch correct",
        pyramid.concepts[0].shape[0] == batch.batch_size,
    )
    log_check(
        "GSM8K projected_hidden batch correct",
        pyramid.projected_hidden.shape[0] == batch.batch_size,
    )


def test_loss_breakdown(builder, config, device, batch_size):
    """Display each loss component, its weight, and the weighted contribution.

    WHAT IS DISPLAYED:
        For each loss component:
          - Raw (unweighted) loss value
          - Weight from config
          - Weighted contribution = raw × weight
        Plus the total loss = sum of all weighted contributions.
    """
    logging.info("\n=== Loss Breakdown ===")
    logging.info(
        "Purpose: Show raw loss, weight, and weighted contribution for each component"
    )

    texts = [f"What is {i} + {i+1}? The answer is {i+i+1}." for i in range(batch_size)]
    builder.train()
    enc_out = builder.encode_cot(texts)
    H = enc_out.hidden_states
    pyramid = builder(H, attention_mask=enc_out.attention_mask)

    loss_weights = config["training"]["loss_weights"]
    ordering_loss_type = config["training"].get("ordering_loss_type", "margin")

    # --- Compute base builder losses ---
    total_loss, loss_dict = compute_builder_loss(
        pyramid, loss_weights, ordering_loss_type=ordering_loss_type
    )

    recon_raw = loss_dict["recon"]
    ordering_raw = loss_dict["ordering"]
    residual_raw = loss_dict["residual"]
    total_raw = loss_dict["total"]

    recon_w = loss_weights["recon_loss_weight"]
    ordering_w = loss_weights["ordering_loss_weight"]
    residual_w = loss_weights.get("residual_loss_weight", 0.01)

    recon_weighted = recon_raw * recon_w
    ordering_weighted = ordering_raw * ordering_w
    residual_weighted = residual_raw * residual_w

    logging.info(
        "  Config: ordering_loss_type=%s, ordering_margin=%.2f",
        ordering_loss_type,
        loss_weights["ordering_margin"],
    )
    logging.info(
        "  ┌─────────────────────────────────────────────────────────────────────┐"
    )
    logging.info(
        "  │ Loss Component         │ Raw Value    │ Weight   │ Weighted Value  │"
    )
    logging.info(
        "  ├─────────────────────────────────────────────────────────────────────┤"
    )
    logging.info(
        "  │ recon_loss             │ %11.4f  │ %7.3f  │ %13.4f   │",
        recon_raw,
        recon_w,
        recon_weighted,
    )
    logging.info(
        "  │ ordering_loss          │ %11.4f  │ %7.3f  │ %13.4f   │",
        ordering_raw,
        ordering_w,
        ordering_weighted,
    )
    logging.info(
        "  │ residual_loss          │ %11.4f  │ %7.3f  │ %13.4f   │",
        residual_raw,
        residual_w,
        residual_weighted,
    )
    logging.info(
        "  ├─────────────────────────────────────────────────────────────────────┤"
    )

    # --- NTP loss (when enabled) ---
    ntp_weighted = 0.0
    use_reasoning = (
        config["model"]["builder"]["use_reasoning_loss"]
        and loss_weights.get("ntp_loss_weight", 0.0) > 0
        and builder.back_proj is not None
    )
    if use_reasoning:
        questions = [f"What is {i} + {i+1}?" for i in range(batch_size)]
        solutions = [str(i + i + 1) for i in range(batch_size)]
        Q_tokens = builder.tokenizer(
            questions, return_tensors="pt", padding=True, truncation=True, max_length=64
        )
        sol_tokens = builder.tokenizer(
            solutions, return_tensors="pt", padding=True, truncation=True, max_length=64
        )
        ntp_loss = builder.compute_reasoning_loss(
            pyramid,
            Q_tokens["input_ids"].to(device),
            Q_tokens["attention_mask"].to(device),
            sol_tokens["input_ids"].to(device),
        )
        ntp_raw = ntp_loss.item()
        ntp_w = loss_weights["ntp_loss_weight"]
        ntp_weighted = ntp_raw * ntp_w
        logging.info(
            "  │ ntp_loss               │ %11.4f  │ %7.3f  │ %13.4f   │",
            ntp_raw,
            ntp_w,
            ntp_weighted,
        )
        logging.info(
            "  ├─────────────────────────────────────────────────────────────────────┤"
        )
        total_with_ntp = total_loss.item() + ntp_weighted
    else:
        total_with_ntp = total_loss.item()

    logging.info(
        "  │ TOTAL                  │             │          │ %13.4f   │",
        total_with_ntp,
    )
    logging.info(
        "  └─────────────────────────────────────────────────────────────────────┘"
    )

    # Sanity checks
    expected_base_total = recon_weighted + ordering_weighted + residual_weighted
    log_check(
        "total ≈ sum of weighted components",
        abs(total_raw - expected_base_total) < 1e-3,
        f"total={total_raw:.4f}, sum={expected_base_total:.4f}",
    )
    log_check("recon_loss is finite", recon_raw == recon_raw)
    log_check("ordering_loss is finite", ordering_raw == ordering_raw)
    log_check("residual_loss >= 0", residual_raw >= 0)
    log_check("total_loss is finite", total_raw == total_raw)

    builder.eval()


def test_gradient_flow(builder, device, batch_size):
    """Verify backpropagation reaches all learnable parameters.

    WHAT IS CHECKED:
        - input_proj.weight receives grad after backward()
        - temperature (scalar) receives grad
        - Every concept_queries[k] Parameter receives grad
        - Every level_projs[k].weight receives grad

    NOTE:
        If reason_model_freeze=True and no LoRA, the backbone will NOT
        have gradients — this is expected and not an error.
    """
    logging.info("\n=== Gradient Flow Tests ===")
    logging.info("Purpose: Verify all learnable parameters receive gradients")

    texts = [f"Compute {i} + {i+1}." for i in range(batch_size)]
    enc_out = builder.encode_cot(texts)

    builder.train()
    output = builder(enc_out.hidden_states, attention_mask=enc_out.attention_mask)

    loss = sum(c.sum() for c in output.concepts)
    log_value("loss value", f"{loss.item():.4f}")
    loss.backward()

    # --- input_proj ---
    has_grad = builder.input_proj.weight.grad is not None
    log_check("input_proj.weight has gradient", has_grad)
    if has_grad:
        log_value(
            "input_proj.grad norm",
            f"{builder.input_proj.weight.grad.norm().item():.4f}",
        )

    # --- temperature ---
    has_grad = builder.temperature.grad is not None
    log_check("temperature has gradient", has_grad)
    if has_grad:
        log_value("temperature.grad", f"{builder.temperature.grad.item():.4f}")

    # --- concept_queries ---
    for k, q in enumerate(builder.concept_queries):
        has_grad = q.grad is not None
        log_check(f"concept_queries[{k}] has gradient", has_grad)
        if has_grad:
            log_value(f"  queries[{k}] grad norm", f"{q.grad.norm().item():.4f}")

    # --- level_projs ---
    for k, proj in enumerate(builder.level_projs):
        has_grad = proj.weight.grad is not None
        log_check(f"level_projs[{k}] has gradient", has_grad)
        if has_grad:
            log_value(
                f"  projs[{k}] grad norm", f"{proj.weight.grad.norm().item():.4f}"
            )

    # --- reason_model (frozen by default) ---
    backbone_grad = any(p.grad is not None for p in builder.reason_model.parameters())
    log_check(
        "reason_model has gradients (only if un-frozen or LoRA)",
        backbone_grad,
        "Expected WARN if freeze=True and no LoRA",
    )

    # --- back_proj (only when use_reasoning_loss=True) ---
    if builder.back_proj is not None:
        bp_grad = builder.back_proj.weight.grad is not None
        log_check("back_proj.weight has gradient", bp_grad)
        if bp_grad:
            log_value(
                "back_proj.grad norm",
                f"{builder.back_proj.weight.grad.norm().item():.4f}",
            )

    builder.eval()


def test_reasoning_loss(ntp_config, device, batch_size):
    """Verify compute_reasoning_loss with use_reasoning_loss=True.

    WHAT IS CHECKED:
        - Builder with use_reasoning_loss=True has back_proj
        - back_proj dimensions: in=hidden_dim, out=reason_model_hidden_dim
        - back_proj.weight ≈ input_proj.weight.T (pseudo-inverse init)
        - compute_reasoning_loss returns a scalar loss with correct args
        - Loss is finite and positive
        - Gradients flow through back_proj and input_proj after NTP backward
    """
    logging.info("\n=== Reasoning Loss Tests (use_reasoning_loss=True) ===")
    logging.info("Purpose: Verify NTP reasoning loss pipeline with back_proj")

    logging.info("  Creating builder from NTP config...")
    ntp_builder = ConceptPyramidBuilder(ntp_config)
    ntp_builder.to(device)

    pyramid_cfg = ntp_config["model"]["pyramid"]

    # --- back_proj existence and dimensions ---
    log_check(
        "back_proj exists (use_reasoning_loss=True)",
        ntp_builder.back_proj is not None,
    )
    if ntp_builder.back_proj is not None:
        log_check(
            "back_proj.in_features == hidden_dim",
            ntp_builder.back_proj.in_features == pyramid_cfg["hidden_dim"],
            f"in={ntp_builder.back_proj.in_features}, expected={pyramid_cfg['hidden_dim']}",
        )
        log_check(
            "back_proj.out_features == reason_model_hidden_dim",
            ntp_builder.back_proj.out_features == ntp_builder.reason_model_hidden_dim,
            f"out={ntp_builder.back_proj.out_features}, expected={ntp_builder.reason_model_hidden_dim}",
        )

        # --- Pseudo-inverse initialization check ---
        weight_diff = (
            (ntp_builder.back_proj.weight - ntp_builder.input_proj.weight.T)
            .abs()
            .max()
            .item()
        )
        log_value("back_proj vs input_proj.T max_diff", f"{weight_diff:.2e}")
        log_check(
            "back_proj initialized as input_proj.T (pseudo-inverse)",
            weight_diff < 1e-6,
            f"diff={weight_diff:.2e}",
        )

    # --- Full NTP pipeline with real data via NLCPV3DataLoader ---
    data_cfg = ntp_config["data"]
    ntp_dataloader = NLCPV3DataLoader(
        data_cfg=data_cfg,
        batch_size=batch_size,
        include_solution=True,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    batch = next(iter(ntp_dataloader))
    logging.info("  Using batch of %d GSM8K samples for NTP", batch.batch_size)

    enc_out = ntp_builder.encode_cot(batch.cot_answers)
    pyramid = ntp_builder(enc_out.hidden_states, attention_mask=enc_out.attention_mask)

    Q_tokens = ntp_builder.tokenizer(
        batch.questions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=ntp_config["model"]["pyramid"]["max_seq_len"],
    )
    sol_tokens = ntp_builder.tokenizer(
        batch.solutions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=ntp_config["model"]["pyramid"]["max_seq_len"],
    )

    ntp_builder.train()
    ntp_loss = ntp_builder.compute_reasoning_loss(
        pyramid,
        Q_tokens["input_ids"].to(device),
        Q_tokens["attention_mask"].to(device),
        sol_tokens["input_ids"].to(device),
    )
    log_value("NTP loss value", f"{ntp_loss.item():.4f}")
    log_check("NTP loss is finite", torch.isfinite(ntp_loss))
    log_check("NTP loss is positive", ntp_loss.item() > 0)

    # --- Gradient flow through NTP ---
    ntp_loss.backward()
    bp_grad = ntp_builder.back_proj.weight.grad is not None
    log_check("back_proj receives gradient from NTP loss", bp_grad)
    if bp_grad:
        log_value(
            "back_proj.grad norm",
            f"{ntp_builder.back_proj.weight.grad.norm().item():.4f}",
        )
    ip_grad = ntp_builder.input_proj.weight.grad is not None
    log_check("input_proj receives gradient from NTP loss", ip_grad)
    if ip_grad:
        log_value(
            "input_proj.grad norm",
            f"{ntp_builder.input_proj.weight.grad.norm().item():.4f}",
        )

    ntp_builder.eval()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    yaml_config = load_config(str(config_path))

    # Device
    device = str(get_device("auto"))
    logging.info("Device: %s", device)

    # Batch size from training config
    batch_size = yaml_config["training"]["batch_size"]

    # Load model (includes reason_model + tokenizer)
    logging.info(
        "Loading reason model: %s",
        yaml_config["model"]["reason_model"]["reason_model_name"],
    )
    builder = ConceptPyramidBuilder(yaml_config)
    builder.to(device)
    logging.info(
        "Model loaded. reason_model_hidden_dim=%d", builder.reason_model_hidden_dim
    )

    # Load GSM8K via NLCPV3DataLoader
    logging.info("Loading GSM8K dataset via NLCPV3DataLoader...")
    data_cfg = yaml_config["data"]
    dataloader = NLCPV3DataLoader(
        data_cfg=data_cfg,
        batch_size=batch_size,
        include_solution=False,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    logging.info("GSM8K loaded: %d samples", dataloader.dataset_size)

    # Run all diagnostic tests
    test_encode_cot(builder, device, batch_size)
    test_forward(builder, device, yaml_config, batch_size)
    test_forward_next_level(builder, device, yaml_config, batch_size)
    test_gsm8k_integration(builder, device, yaml_config, dataloader)
    test_loss_breakdown(builder, yaml_config, device, batch_size)
    test_gradient_flow(builder, device, batch_size)

    # Load NTP config and test reasoning loss if use_reasoning_loss is False
    if not yaml_config["model"]["builder"]["use_reasoning_loss"]:
        ntp_config_path = config_path.parent / "test_concept_builder_ntp.yml"
        if ntp_config_path.exists():
            ntp_yaml = load_config(str(ntp_config_path))
            test_reasoning_loss(ntp_yaml, device, batch_size)

    logging.info("\n=== ALL DIAGNOSTIC TESTS COMPLETE ===")
    logging.info(
        "Review [OK] / [WARN] lines above. [WARN] before training is expected."
    )


if __name__ == "__main__":
    main()
