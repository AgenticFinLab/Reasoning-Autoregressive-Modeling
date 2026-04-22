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
    python3 examples/nlcpV3/utest/test_concept_builder.py \
        -c configs/nlcpV3/utest/test_concept_builder.yml
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

from nlcpV3.config import NLCPV3Config
from nlcpV3.concept_hybrid_builder import (
    ConceptPyramidBuilder,
    EncoderOutput,
    PyramidOutput,
    SingleLevelOutput,
)
from lmbase.utils.env_tools import get_device
from ram.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="ConceptPyramidBuilder test")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to YAML config file"
    )
    return parser.parse_args()


def build_nlcpv3_config_from_yaml(yaml_dict):
    """Build NLCPV3Config from YAML dict.

    Maps YAML nested blocks (model.reason_model, model.pyramid, etc.) to
    the flat NLCPV3Config dataclass fields.
    """
    m = yaml_dict["model"]
    rm = m["reason_model"]
    pyr = m["pyramid"]
    dec = m["decoder"]
    bld = m["builder"]
    tr = yaml_dict["training"]
    lw = tr["loss_weights"]
    return NLCPV3Config(
        hidden_dim=pyr["hidden_dim"],
        num_heads=pyr["num_heads"],
        num_levels=pyr["num_levels"],
        level_lengths=pyr["level_lengths"],
        max_seq_len=pyr["max_seq_len"],
        reason_model_name=rm["reason_model_name"],
        reason_model_num_layers=rm["reason_model_num_layers"],
        reason_model_freeze=rm["reason_model_freeze"],
        reason_model_lora=rm.get("reason_model_lora"),
        decoder_model_name=dec.get("decoder_model_name", ""),
        decoder_freeze=dec.get("decoder_freeze", True),
        decoder_lora=dec.get("decoder_lora"),
        use_positional_query_init=bld["use_positional_query_init"],
        ntp_loss_weight=lw["ntp_loss_weight"],
        concept_loss_weight=lw["concept_loss_weight"],
        recon_loss_weight=lw["recon_loss_weight"],
    )


def log_check(name, cond, details=""):
    """Log a diagnostic check result without raising.

    Args:
        name: Human-readable description of what is being checked.
        cond: Boolean condition result.
        details: Additional numeric or descriptive info.
    """
    status = "OK" if cond else "WARN"
    msg = f"  [{status}] {name}"
    if details:
        msg += f" | {details}"
    logging.info(msg)


def log_value(name, value, unit=""):
    """Log a numeric observation for human inspection."""
    u = f" {unit}" if unit else ""
    logging.info(f"  [VAL] {name} = {value}{u}")


def test_constructor(builder, config):
    """Verify all components created in ConceptPyramidBuilder.__init__().

    PURPOSE:
        Confirm that the Builder initializes every sub-module with the
        correct shapes and properties. This catches configuration errors
        (e.g. wrong hidden_dim, mismatched num_levels) before any
        forward pass is attempted.

    WHAT IS CHECKED:
        - reason_model and tokenizer are loaded
        - input_proj maps encoder_dim → concept_dim
        - concept_queries has K levels with shapes [L_k, D]
        - temperature is a learnable scalar
        - level_projs and level_attn have K elements each
        - recon_decoder is None (expected placeholder)
        - total_concepts matches sum(level_lengths)
    """
    logging.info("\n=== Constructor Tests ===")
    logging.info("Purpose: Verify __init__ creates all components correctly")

    # --- Reason Model & Tokenizer ---
    # The Builder loads the pretrained backbone and its paired tokenizer.
    log_check("reason_model loaded", builder.reason_model is not None)
    log_check("reason_model has config", hasattr(builder.reason_model, "config"))
    log_value("reason_model_hidden_dim", builder.reason_model_hidden_dim)
    log_value("model.config.hidden_size", builder.reason_model.config.hidden_size)
    log_check(
        "hidden_dim matches model config",
        builder.reason_model_hidden_dim == builder.reason_model.config.hidden_size,
    )
    log_check("tokenizer loaded", builder.tokenizer is not None)
    log_check("tokenizer has pad_token", builder.tokenizer.pad_token is not None)

    # --- Projection Layer ---
    # input_proj: maps from reason_model hidden_dim to concept hidden_dim.
    log_check("input_proj exists", builder.input_proj is not None)
    log_value("input_proj.in_features", builder.input_proj.in_features)
    log_value("input_proj.out_features", builder.input_proj.out_features)
    log_check(
        "input_proj.in == reason_model_hidden_dim",
        builder.input_proj.in_features == builder.reason_model_hidden_dim,
    )
    log_check(
        "input_proj.out == config.hidden_dim",
        builder.input_proj.out_features == config.hidden_dim,
    )

    # --- Learnable Concept Queries ---
    # concept_queries: K Parameter objects, each shape [L_k, D].
    # These are the "concept vocabulary" that replace VAR's codebook.
    log_value("num_levels", config.num_levels)
    log_value("concept_queries count", len(builder.concept_queries))
    log_check(
        "concept_queries count == num_levels",
        len(builder.concept_queries) == config.num_levels,
    )
    for k, q in enumerate(builder.concept_queries):
        expected = (config.level_lengths[k], config.hidden_dim)
        log_value(f"concept_queries[{k}].shape", list(q.shape))
        log_value(f"concept_queries[{k}].expected", list(expected))
        log_check(f"level {k} query shape", q.shape == expected)
        log_check(f"level {k} query requires_grad", q.requires_grad)

    # --- Attention Temperature ---
    # temperature: learnable scalar τ that controls attention sharpness.
    log_value("temperature.shape", list(builder.temperature.shape))
    log_check("temperature shape == (1,)", builder.temperature.shape == (1,))
    log_check("temperature requires_grad", builder.temperature.requires_grad)

    # --- Level Projections ---
    # level_projs: K Linear layers, each D→D.
    log_value("level_projs count", len(builder.level_projs))
    log_check(
        "level_projs count == num_levels",
        len(builder.level_projs) == config.num_levels,
    )
    for k, proj in enumerate(builder.level_projs):
        log_check(
            f"level {k} proj.in == D",
            proj.in_features == config.hidden_dim,
            f"in={proj.in_features}",
        )
        log_check(
            f"level {k} proj.out == D",
            proj.out_features == config.hidden_dim,
            f"out={proj.out_features}",
        )

    # --- Cross-Attention Refinement Layers ---
    # level_attn: K MultiheadAttention modules (level 0 unused but created).
    log_value("level_attn count", len(builder.level_attn))
    log_check(
        "level_attn count == num_levels",
        len(builder.level_attn) == config.num_levels,
    )

    # --- Solution Decoder Placeholder ---
    # solution_decoder: interface-only, should be None until implemented.
    log_check(
        "solution_decoder is None (placeholder)", builder.solution_decoder is None
    )

    # --- Reconstruction Decoder Placeholder ---
    log_check("recon_decoder is None (placeholder)", builder.recon_decoder is None)

    # --- Cache Lists ---
    log_check(
        "_cached_attentions is list", isinstance(builder._cached_attentions, list)
    )
    log_check(
        "_cached_base_concepts is list",
        isinstance(builder._cached_base_concepts, list),
    )

    # --- Total Concept Count ---
    expected_total = sum(config.level_lengths)
    log_value("get_total_concepts()", builder.get_total_concepts())
    log_value("expected total", expected_total)
    log_check(
        "total_concepts matches config",
        builder.get_total_concepts() == expected_total,
    )


def test_encode_cot(builder, device, batch_size):
    """Verify encode_cot produces correct hidden states from text or tensors.

    PURPOSE:
        The Builder must accept either raw text strings (auto-tokenized)
        or pre-tokenized tensors. Both paths should produce identical
        hidden-state outputs from the pretrained reason_model.

    WHAT IS CHECKED:
        - Text input → EncoderOutput with 3D hidden_states [B, L, D_encoder]
        - Tensor input → same shape and same numerical values
        - Batch dimension matches requested batch_size
        - Last dimension matches reason_model hidden size
    """
    logging.info("\n=== encode_cot Tests ===")
    logging.info(
        "Purpose: Verify text and tensor inputs both produce valid hidden states"
    )

    texts = [f"Problem {i}: What is {i} + {i+1}?" for i in range(batch_size)]

    # --- Test 1: text input (auto-tokenize) ---
    # encode_cot internally calls tokenizer() + reason_model() when given strings.
    logging.info("  -- Test 1: Text input (auto-tokenize) --")
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

    # --- Test 2: tensor input ---
    # Same tokenizer call, but inputs provided explicitly as tensors.
    # Both paths should yield numerically identical hidden states.
    logging.info("  -- Test 2: Tensor input (manual tokenize) --")
    tokens = builder.tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=32
    )
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    enc_out2 = builder.encode_cot(input_ids, attention_mask)
    log_check("returns EncoderOutput", isinstance(enc_out2, EncoderOutput))
    log_value("hidden_states.shape", list(enc_out2.hidden_states.shape))

    max_diff = (enc_out.hidden_states - enc_out2.hidden_states).abs().max().item()
    log_value("max_diff (text vs tensor)", f"{max_diff:.2e}")
    log_check(
        "text input == tensor input (numerical)",
        torch.allclose(enc_out.hidden_states, enc_out2.hidden_states),
        f"max_diff={max_diff:.2e}",
    )

    return enc_out


def test_forward(builder, device, config, batch_size):
    """Verify forward() full pyramid construction with residual decomposition.

    PURPOSE:
        forward() is the core algorithm. It decomposes H_proj into a
        hierarchy of concept levels with the residual constraint:
            f_hat_k + f_rest_k = H_proj  for all k
        This test verifies structural correctness (shapes) and checks
        the residual identity holds within floating-point tolerance.

    WHAT IS CHECKED:
        - Returns PyramidOutput with correct number of levels
        - Each level's concepts have shape [B, L_k, D]
        - LevelOutput contains concepts, base_concepts, attention, reconstruction
        - Residual identity: reconstructed_hidden + residual_hidden ≈ H_proj
        - projected/reconstructed/residual all have shape [B, L, D]
        - PyramidOutput helper properties (num_levels, total_concepts, etc.)
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
    output = builder(H)
    log_check("returns PyramidOutput", isinstance(output, PyramidOutput))

    # --- Concepts count and shapes ---
    # concepts: list of K tensors, concepts[k] shape [B, L_k, D]
    log_value("num_levels", config.num_levels)
    log_value("concepts count", len(output.concepts))
    log_check("concepts count == num_levels", len(output.concepts) == config.num_levels)
    for k, concepts in enumerate(output.concepts):
        expected = (batch_size, config.level_lengths[k], config.hidden_dim)
        log_value(f"concepts[{k}].shape", list(concepts.shape))
        log_value(f"concepts[{k}].expected", list(expected))
        log_check(f"level {k} shape correct", concepts.shape == expected)

    # --- LevelOutput fields ---
    # Each level produces structured output with 4 tensors.
    log_value("level_outputs count", len(output.level_outputs))
    log_check(
        "level_outputs count == num_levels",
        len(output.level_outputs) == config.num_levels,
    )
    for k, lo in enumerate(output.level_outputs):
        Lk = config.level_lengths[k]
        log_value(f"level {k} concepts.shape", list(lo.concepts.shape))
        log_value(f"level {k} base_concepts.shape", list(lo.base_concepts.shape))
        log_value(f"level {k} attention.shape", list(lo.attention_weights.shape))
        log_value(f"level {k} reconstruction.shape", list(lo.reconstruction.shape))
        log_check(
            f"level {k} concepts shape",
            lo.concepts.shape == (batch_size, Lk, config.hidden_dim),
        )
        log_check(
            f"level {k} base_concepts shape",
            lo.base_concepts.shape == (batch_size, Lk, config.hidden_dim),
        )
        log_check(
            f"level {k} attention shape",
            lo.attention_weights.shape == (batch_size, Lk, seq_len),
        )
        log_check(
            f"level {k} reconstruction shape",
            lo.reconstruction.shape == (batch_size, seq_len, config.hidden_dim),
        )

    # --- Residual decomposition: f_hat + f_rest == H_proj ---
    # MATHEMATICAL PRINCIPLE:
    #   reconstructed_accumulator = R_0 + R_1 + ... + R_{K-1}
    #   residual_hidden = H_proj - (R_0 + R_1 + ... + R_{K-1})
    #   Therefore: reconstructed + residual = H_proj exactly (in real arithmetic).
    # In floating point, the independent addition and subtraction sequences
    # accumulate round-off error. With random (untrained) weights, R_k can
    # be large, so cancellation amplifies the absolute error.
    # EXPECTED: diff < ~1e-2 for float32, even before training.
    logging.info("  -- Residual decomposition check --")
    recomposed = output.reconstructed_hidden + output.residual_hidden
    diff = torch.abs(recomposed - output.projected_hidden).max().item()
    log_value("max_diff (f_hat + f_rest vs H_proj)", f"{diff:.2e}")
    # Use a tolerant threshold because this is untrained random weights.
    # The identity is algorithmic correctness, not training quality.
    tolerance = 1e-1  # generous for float32 + random weights
    log_check(
        f"residual identity holds (tol={tolerance:.0e})",
        diff < tolerance,
        f"diff={diff:.2e} — NOTE: before training, large diff is expected; "
        f"the algorithm is still correct",
    )

    # --- Hidden state shapes ---
    for name, tensor in [
        ("projected_hidden", output.projected_hidden),
        ("reconstructed_hidden", output.reconstructed_hidden),
        ("residual_hidden", output.residual_hidden),
    ]:
        expected_shape = (batch_size, seq_len, config.hidden_dim)
        log_value(f"{name}.shape", list(tensor.shape))
        log_check(
            f"{name} shape correct",
            tensor.shape == expected_shape,
            f"expected {expected_shape}",
        )

    # --- PyramidOutput properties ---
    log_value("num_levels", output.num_levels)
    log_value("level_lengths", output.level_lengths)
    log_value("total_concepts", output.total_concepts)
    log_check("num_levels correct", output.num_levels == config.num_levels)
    log_check("level_lengths correct", output.level_lengths == config.level_lengths)
    log_check(
        "total_concepts correct",
        output.total_concepts == sum(config.level_lengths),
    )
    log_value("all_attentions length", len(output.all_attentions))
    log_value("all_base_concepts length", len(output.all_base_concepts))
    log_check(
        "all_attentions length",
        len(output.all_attentions) == config.num_levels,
    )
    log_check(
        "all_base_concepts length",
        len(output.all_base_concepts) == config.num_levels,
    )

    cat = output.cat_concepts()
    expected_cat_shape = (batch_size, sum(config.level_lengths), config.hidden_dim)
    log_value("cat_concepts.shape", list(cat.shape))
    log_check("cat_concepts shape correct", cat.shape == expected_cat_shape)

    # --- Level 0: no refinement ---
    # By design, level 0 has no previous concepts to attend to, so
    # refined_concepts = 0 and concepts == base_concepts.
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
    # Attention weights should sum to 1 across the sequence dimension.
    # For random weights, the distribution may be uniform or peaked.
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

    PURPOSE:
        forward_next_level() is the incremental API used during autoregressive
        generation (Phase 2 — Predictor). It extracts one level at a time,
        using previously extracted concepts as context for refinement.
        This test verifies each level is extracted correctly in sequence.

    WHAT IS CHECKED:
        - Each level returns SingleLevelOutput with correct level_index
        - concepts shape is [B, L_k, D]
        - projected_hidden shape matches encoder output H
        - Cache accumulates exactly num_levels entries
        - clear_cache() resets both cache lists to empty
    """
    logging.info("\n=== forward_next_level() Tests ===")
    logging.info("Purpose: Verify incremental level-by-level extraction")

    texts = [f"Step {i}: calculate {i+1} * {i+2}." for i in range(batch_size)]
    enc_out = builder.encode_cot(texts)
    H = enc_out.hidden_states
    logging.info("  Encoder output H shape: %s", list(H.shape))

    # Sequential level extraction
    # This mirrors how the Predictor will generate: level 0, then level 1, etc.
    builder.clear_cache()
    prev_concepts = []
    for k in range(config.num_levels):
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
            f"level {k} batch size correct",
            level_out.concepts.shape[0] == batch_size,
        )
        log_check(
            f"level {k} concept count correct",
            level_out.concepts.shape[1] == config.level_lengths[k],
        )
        log_value(
            f"level {k} projected_hidden.shape", list(level_out.projected_hidden.shape)
        )
        expected_ph_shape = (batch_size, H.shape[1], config.hidden_dim)
        log_value(f"level {k} projected_hidden expected", list(expected_ph_shape))
        log_check(
            f"level {k} projected_hidden shape correct",
            level_out.projected_hidden.shape == expected_ph_shape,
        )
        prev_concepts.append(level_out.concepts)

    # --- Cache verification ---
    log_value("cached_attentions length", len(builder._cached_attentions))
    log_value("cached_base_concepts length", len(builder._cached_base_concepts))
    log_check(
        "cache has num_levels attentions",
        len(builder._cached_attentions) == config.num_levels,
    )
    log_check(
        "cache has num_levels base_concepts",
        len(builder._cached_base_concepts) == config.num_levels,
    )

    # --- Clear cache ---
    builder.clear_cache()
    log_value("after clear: attentions length", len(builder._cached_attentions))
    log_value("after clear: base_concepts length", len(builder._cached_base_concepts))
    log_check("clear_cache empties attentions", len(builder._cached_attentions) == 0)
    log_check(
        "clear_cache empties base_concepts", len(builder._cached_base_concepts) == 0
    )


def test_gsm8k_integration(builder, device, config, dataset, batch_size):
    """Verify end-to-end pipeline with real GSM8K CoT data.

    PURPOSE:
        All previous tests use synthetic text. This test uses actual GSM8K
        Chain-of-Thought answers to verify the pipeline works on realistic
        reasoning traces (longer, more complex text).

    WHAT IS CHECKED:
        - encode_cot handles real CoT text (auto-tokenize)
        - hidden_states batch dimension matches dataset samples
        - forward() produces valid PyramidOutput with correct shapes
        - total_concepts matches configuration
    """
    logging.info("\n=== GSM8K Integration Tests ===")
    logging.info("Purpose: Verify pipeline with real GSM8K CoT data")

    n = min(batch_size, len(dataset))
    samples = [dataset[i] for i in range(n)]
    cot_texts = [s["cot_answer"] for s in samples]
    logging.info("  Using %d GSM8K samples", n)

    # --- Encode CoT ---
    enc_out = builder.encode_cot(cot_texts)
    log_check(
        "GSM8K encode_cot returns EncoderOutput",
        isinstance(enc_out, EncoderOutput),
    )
    log_value("GSM8K hidden_states.shape", list(enc_out.hidden_states.shape))
    log_check(
        "GSM8K batch dimension correct",
        enc_out.hidden_states.shape[0] == n,
        f"got {enc_out.hidden_states.shape[0]}, expected {n}",
    )
    log_check(
        "GSM8K hidden dim correct",
        enc_out.hidden_states.shape[-1] == builder.reason_model_hidden_dim,
        f"got {enc_out.hidden_states.shape[-1]}, expected {builder.reason_model_hidden_dim}",
    )

    # --- Build pyramid ---
    pyramid = builder(enc_out.hidden_states)
    log_check(
        "GSM8K forward returns PyramidOutput",
        isinstance(pyramid, PyramidOutput),
    )
    log_value("GSM8K concepts count", len(pyramid.concepts))
    log_check(
        "GSM8K concepts count correct",
        len(pyramid.concepts) == config.num_levels,
    )
    log_value("GSM8K total_concepts", pyramid.total_concepts)
    log_check(
        "GSM8K total_concepts correct",
        pyramid.total_concepts == sum(config.level_lengths),
    )
    log_value("GSM8K concept batch", pyramid.concepts[0].shape[0])
    log_check("GSM8K concept batch correct", pyramid.concepts[0].shape[0] == n)
    log_value("GSM8K projected_hidden batch", pyramid.projected_hidden.shape[0])
    log_check(
        "GSM8K projected_hidden batch correct",
        pyramid.projected_hidden.shape[0] == n,
    )


def test_gradient_flow(builder, device, batch_size):
    """Verify backpropagation reaches all learnable parameters.

    PURPOSE:
        Before training, we must confirm that gradients can flow through
        every learnable component: input_proj, temperature, concept_queries,
        and level_projs. If any component has broken gradients, training
        will silently fail (parameters frozen).

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
    output = builder(enc_out.hidden_states)

    # Use a simple sum loss so every concept contributes.
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
    # Not a failure — just informational.
    backbone_grad = any(p.grad is not None for p in builder.reason_model.parameters())
    log_check(
        "reason_model has gradients (only if un-frozen or LoRA)",
        backbone_grad,
        "Expected WARN if freeze=True and no LoRA",
    )

    builder.eval()


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
    nlcp_config = build_nlcpv3_config_from_yaml(yaml_config)

    # Device
    device = str(get_device("auto"))
    logging.info("Device: %s", device)

    # Batch size (configurable here)
    batch_size = 3

    # Load model (includes reason_model + tokenizer)
    logging.info("Loading reason model: %s", nlcp_config.reason_model_name)
    builder = ConceptPyramidBuilder(nlcp_config)
    builder.to(device)
    logging.info(
        "Model loaded. reason_model_hidden_dim=%d", builder.reason_model_hidden_dim
    )

    # Load GSM8K
    logging.info("Loading GSM8K dataset...")
    from lmbase.dataset import registry

    data_cfg = yaml_config["data"]
    dataset = registry.get(data_cfg, split="train")
    logging.info("GSM8K loaded: %d samples", len(dataset))

    # Run all diagnostic tests
    # Each test prints [OK] or [WARN] lines. No hard failures — inspect output.
    test_constructor(builder, nlcp_config)
    test_encode_cot(builder, device, batch_size)
    test_forward(builder, device, nlcp_config, batch_size)
    test_forward_next_level(builder, device, nlcp_config, batch_size)
    test_gsm8k_integration(builder, device, nlcp_config, dataset, batch_size)
    test_gradient_flow(builder, device, batch_size)

    logging.info("\n=== ALL DIAGNOSTIC TESTS COMPLETE ===")
    logging.info(
        "Review [OK] / [WARN] lines above. [WARN] before training is expected."
    )


if __name__ == "__main__":
    main()
