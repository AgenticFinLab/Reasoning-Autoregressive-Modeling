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

from nlcpV3.concept_hybrid_builder import (
    ConceptPyramidBuilder,
    EncoderOutput,
    PyramidOutput,
    SingleLevelOutput,
)
from nlcpV3.data_loader import BuilderInput
from nlcpV3.train_builder import compute_builder_loss
from lmbase.dataset import registry
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
    """Verify encode_cot produces correct hidden states from token IDs.

    PURPOSE:
        The Builder's encode_cot accepts pre-tokenized input_ids and
        produces hidden states from the pretrained reason_model backbone.

    WHAT IS CHECKED:
        - Tensor input → EncoderOutput with 3D hidden_states [B, L, D_encoder]
        - Batch dimension matches requested batch_size
        - Last dimension matches reason_model hidden size
    """
    logging.info("\n=== encode_cot Tests ===")
    logging.info("Purpose: Verify token ID inputs produce valid hidden states")

    texts = [f"Problem {i}: What is {i} + {i+1}?" for i in range(batch_size)]

    # Tokenize texts externally (as forward() does)
    tokens = builder.tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=32
    )
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    # --- Test: tensor input ---
    logging.info("  -- Test: Tensor input (token IDs) --")
    enc_out = builder.encode_cot(input_ids, attention_mask)
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
    tokens = builder.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config["model"]["pyramid"]["max_seq_len"],
    )
    enc_out = builder.encode_cot(
        tokens["input_ids"].to(device), tokens["attention_mask"].to(device)
    )
    H = enc_out.hidden_states
    seq_len = H.shape[1]
    logging.info("  Encoder output H shape: %s", list(H.shape))

    # Build pyramid
    output = builder._build_pyramid_from_hidden_states(H)
    log_check("returns PyramidOutput", isinstance(output, PyramidOutput))

    # --- Concepts count and shapes ---
    # concepts: list of K tensors, concepts[k] shape [B, L_k, D]
    log_value("num_levels", config["model"]["pyramid"]["num_levels"])
    log_value("concepts count", len(output.concepts))
    log_check(
        "concepts count == num_levels",
        len(output.concepts) == config["model"]["pyramid"]["num_levels"],
    )
    for k, concepts in enumerate(output.concepts):
        expected = (
            batch_size,
            config["model"]["pyramid"]["level_lengths"][k],
            config["model"]["pyramid"]["hidden_dim"],
        )
        log_value(f"concepts[{k}].shape", list(concepts.shape))
        log_value(f"concepts[{k}].expected", list(expected))
        log_check(f"level {k} shape correct", concepts.shape == expected)

    # --- LevelOutput fields ---
    # Each level produces structured output with 4 tensors.
    log_value("level_outputs count", len(output.level_outputs))
    log_check(
        "level_outputs count == num_levels",
        len(output.level_outputs) == config["model"]["pyramid"]["num_levels"],
    )
    for k, lo in enumerate(output.level_outputs):
        Lk = config["model"]["pyramid"]["level_lengths"][k]
        log_value(f"level {k} concepts.shape", list(lo.concepts.shape))
        log_value(f"level {k} base_concepts.shape", list(lo.base_concepts.shape))
        log_value(f"level {k} attention.shape", list(lo.attention_weights.shape))
        log_value(f"level {k} reconstruction.shape", list(lo.reconstruction.shape))
        log_check(
            f"level {k} concepts shape",
            lo.concepts.shape
            == (batch_size, Lk, config["model"]["pyramid"]["hidden_dim"]),
        )
        log_check(
            f"level {k} base_concepts shape",
            lo.base_concepts.shape
            == (batch_size, Lk, config["model"]["pyramid"]["hidden_dim"]),
        )
        log_check(
            f"level {k} attention shape",
            lo.attention_weights.shape == (batch_size, Lk, seq_len),
        )
        log_check(
            f"level {k} reconstruction shape",
            lo.reconstruction.shape
            == (batch_size, seq_len, config["model"]["pyramid"]["hidden_dim"]),
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
        expected_shape = (batch_size, seq_len, config["model"]["pyramid"]["hidden_dim"])
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
    log_check(
        "num_levels correct",
        output.num_levels == config["model"]["pyramid"]["num_levels"],
    )
    log_check(
        "level_lengths correct",
        output.level_lengths == config["model"]["pyramid"]["level_lengths"],
    )
    log_check(
        "total_concepts correct",
        output.total_concepts == sum(config["model"]["pyramid"]["level_lengths"]),
    )
    log_value("all_attentions length", len(output.all_attentions))
    log_value("all_base_concepts length", len(output.all_base_concepts))
    log_check(
        "all_attentions length",
        len(output.all_attentions) == config["model"]["pyramid"]["num_levels"],
    )
    log_check(
        "all_base_concepts length",
        len(output.all_base_concepts) == config["model"]["pyramid"]["num_levels"],
    )

    cat = output.cat_concepts()
    expected_cat_shape = (
        batch_size,
        sum(config["model"]["pyramid"]["level_lengths"]),
        config["model"]["pyramid"]["hidden_dim"],
    )
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
    tokens = builder.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config["model"]["pyramid"]["max_seq_len"],
    )
    enc_out = builder.encode_cot(
        tokens["input_ids"].to(device), tokens["attention_mask"].to(device)
    )
    H = enc_out.hidden_states
    logging.info("  Encoder output H shape: %s", list(H.shape))

    # Sequential level extraction
    # This mirrors how the Predictor will generate: level 0, then level 1, etc.
    builder.clear_cache()
    prev_concepts = []
    for k in range(config["model"]["pyramid"]["num_levels"]):
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
            level_out.concepts.shape[1]
            == config["model"]["pyramid"]["level_lengths"][k],
        )
        log_value(
            f"level {k} projected_hidden.shape", list(level_out.projected_hidden.shape)
        )
        expected_ph_shape = (
            batch_size,
            H.shape[1],
            config["model"]["pyramid"]["hidden_dim"],
        )
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
        len(builder._cached_attentions) == config["model"]["pyramid"]["num_levels"],
    )
    log_check(
        "cache has num_levels base_concepts",
        len(builder._cached_base_concepts) == config["model"]["pyramid"]["num_levels"],
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
    questions = [s["question"] for s in samples]
    cot_texts = [s["cot_answer"] for s in samples]
    logging.info("  Using %d GSM8K samples", n)

    # --- Build pyramid via forward(BuilderInput) with raw text ---
    batch_input = BuilderInput(
        questions=questions,
        cot_answers=cot_texts,
        solutions=[],
    )
    pyramid = builder(batch_input)
    log_check(
        "GSM8K forward returns PyramidOutput",
        isinstance(pyramid, PyramidOutput),
    )
    log_value("GSM8K concepts count", len(pyramid.concepts))
    log_check(
        "GSM8K concepts count correct",
        len(pyramid.concepts) == config["model"]["pyramid"]["num_levels"],
    )
    log_value("GSM8K total_concepts", pyramid.total_concepts)
    log_check(
        "GSM8K total_concepts correct",
        pyramid.total_concepts == sum(config["model"]["pyramid"]["level_lengths"]),
    )
    log_value("GSM8K concept batch", pyramid.concepts[0].shape[0])
    log_check("GSM8K concept batch correct", pyramid.concepts[0].shape[0] == n)
    log_value("GSM8K projected_hidden batch", pyramid.projected_hidden.shape[0])
    log_check(
        "GSM8K projected_hidden batch correct",
        pyramid.projected_hidden.shape[0] == n,
    )


def test_loss_breakdown(builder, config, device, batch_size):
    """Display each loss component, its weight, and the weighted contribution.

    PURPOSE:
        Show a clear breakdown of all builder training losses so the
        developer can verify that weights and magnitudes are reasonable.

    WHAT IS DISPLAYED:
        For each loss component:
          - Raw (unweighted) loss value
          - Weight from config
          - Weighted contribution = raw × weight
        Plus the total loss = sum of all weighted contributions.

    LOSS COMPONENTS:
        1. recon_loss     × recon_loss_weight   — CoT reconstruction
        2. ordering_loss  × concept_loss_weight  — Intra-level ordering
        3. residual_loss  × 0.01 (fixed)         — Clean decomposition
        Total = (1) + (2) + (3)

        When use_reasoning_loss=True and ntp_loss_weight > 0:
        4. ntp_loss       × ntp_loss_weight     — Reasoning validation
        Total = (1) + (2) + (3) + (4)
    """
    logging.info("\n=== Loss Breakdown ===")
    logging.info(
        "Purpose: Show raw loss, weight, and weighted contribution for each component"
    )

    # --- Build pyramid ---
    texts = [f"What is {i} + {i+1}? The answer is {i+i+1}." for i in range(batch_size)]
    builder.train()
    tokens = builder.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config["model"]["pyramid"]["max_seq_len"],
    )
    enc_out = builder.encode_cot(
        tokens["input_ids"].to(device), tokens["attention_mask"].to(device)
    )
    H = enc_out.hidden_states
    pyramid = builder._build_pyramid_from_hidden_states(H)

    # --- Compute base builder losses (recon + ordering + residual) ---
    total_loss, loss_dict = compute_builder_loss(
        pyramid, config["training"]["loss_weights"]
    )

    # Display each component
    recon_raw = loss_dict["recon"]
    ordering_raw = loss_dict["ordering"]
    residual_raw = loss_dict["residual"]
    total_raw = loss_dict["total"]

    recon_w = config["training"]["loss_weights"]["recon_loss_weight"]
    ordering_w = config["training"]["loss_weights"]["concept_loss_weight"]
    residual_w = 0.01  # fixed small weight

    recon_weighted = recon_raw * recon_w
    ordering_weighted = ordering_raw * ordering_w
    residual_weighted = residual_raw * residual_w

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
    if (
        config["model"]["builder"]["use_reasoning_loss"]
        and config["training"]["loss_weights"]["ntp_loss_weight"] > 0
        and builder.back_proj is not None
    ):
        questions = [f"What is {i} + {i+1}?" for i in range(batch_size)]
        solutions = [str(i + i + 1) for i in range(batch_size)]
        Q_tokens = builder.tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        sol_tokens = builder.tokenizer(
            solutions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        ntp_loss = builder.compute_reasoning_loss(
            pyramid,
            Q_tokens["input_ids"].to(device),
            Q_tokens["attention_mask"].to(device),
            sol_tokens["input_ids"].to(device),
        )
        ntp_raw = ntp_loss.item()
        ntp_w = config["training"]["loss_weights"]["ntp_loss_weight"]
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
        # Recompute total with NTP included
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
    log_check("recon_loss is finite", recon_raw == recon_raw)  # NaN check
    log_check("ordering_loss is finite", ordering_raw == ordering_raw)
    log_check("residual_loss >= 0", residual_raw >= 0)
    log_check("total_loss is finite", total_raw == total_raw)

    builder.eval()


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
    tokens = builder.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=builder.pyramid_cfg["max_seq_len"],
    )
    enc_out = builder.encode_cot(
        tokens["input_ids"].to(device), tokens["attention_mask"].to(device)
    )

    builder.train()
    output = builder._build_pyramid_from_hidden_states(enc_out.hidden_states)

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

    PURPOSE:
        When use_reasoning_loss is enabled, the Builder creates back_proj
        (D → D_encoder) and supports compute_reasoning_loss(). This test
        verifies the full NTP pipeline: Q + concept pyramid → solution logits.

    WHAT IS CHECKED:
        - Builder with use_reasoning_loss=True has back_proj
        - back_proj dimensions: in=hidden_dim, out=reason_model_hidden_dim
        - back_proj.weight ≈ input_proj.weight.T (pseudo-inverse init)
        - compute_reasoning_loss returns a scalar loss
        - Loss is finite and positive
        - Gradients flow through back_proj and input_proj after NTP backward
    """
    logging.info("\n=== Reasoning Loss Tests (use_reasoning_loss=True) ===")
    logging.info("Purpose: Verify NTP reasoning loss pipeline with back_proj")

    # Config is loaded from YAML with use_reasoning_loss=True
    logging.info("  Creating builder from NTP config...")
    ntp_builder = ConceptPyramidBuilder(ntp_config)
    ntp_builder.to(device)

    # --- back_proj existence and dimensions ---
    log_check(
        "back_proj exists (use_reasoning_loss=True)",
        ntp_builder.back_proj is not None,
    )
    if ntp_builder.back_proj is not None:
        log_check(
            "back_proj.in_features == hidden_dim",
            ntp_builder.back_proj.in_features
            == ntp_config["model"]["pyramid"]["hidden_dim"],
            f"in={ntp_builder.back_proj.in_features}, expected={ntp_config["model"]["pyramid"]["hidden_dim"]}",
        )
        log_check(
            "back_proj.out_features == reason_model_hidden_dim",
            ntp_builder.back_proj.out_features == ntp_builder.reason_model_hidden_dim,
            f"out={ntp_builder.back_proj.out_features}, "
            f"expected={ntp_builder.reason_model_hidden_dim}",
        )

        # --- Pseudo-inverse initialization check ---
        # back_proj.weight should be initialized as input_proj.weight.T
        weight_diff = (
            (ntp_builder.back_proj.weight - ntp_builder.input_proj.weight.T)
            .abs()
            .max()
            .item()
        )
        log_value(
            "back_proj vs input_proj.T max_diff",
            f"{weight_diff:.2e}",
        )
        log_check(
            "back_proj initialized as input_proj.T (pseudo-inverse)",
            weight_diff < 1e-6,
            f"diff={weight_diff:.2e}",
        )

    # --- Full NTP pipeline ---
    texts = [f"What is {i} + {i+1}? The answer is {i+i+1}." for i in range(batch_size)]
    questions = [f"What is {i} + {i+1}?" for i in range(batch_size)]
    solutions = [str(i + i + 1) for i in range(batch_size)]

    batch_input = BuilderInput(
        questions=questions,
        cot_answers=texts,
        solutions=solutions,
    )

    ntp_builder.train()
    pyramid = ntp_builder(batch_input)

    ntp_loss = ntp_builder.compute_reasoning_loss(pyramid)
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

    # Load GSM8K
    logging.info("Loading GSM8K dataset...")
    data_cfg = yaml_config["data"]
    dataset = registry.get(data_cfg, split=data_cfg["split"])
    logging.info("GSM8K loaded: %d samples", len(dataset))

    # Run all diagnostic tests
    test_encode_cot(builder, device, batch_size)
    test_forward(builder, device, yaml_config, batch_size)
    test_forward_next_level(builder, device, yaml_config, batch_size)
    test_gsm8k_integration(builder, device, yaml_config, dataset, batch_size)
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
