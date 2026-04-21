"""ConceptPyramidBuilder integration test.

Directly tests every component and pipeline step without unittest wrappers.
Run with:
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


def check(cond, msg):
    if not cond:
        raise AssertionError(f"FAIL: {msg}")
    logging.info("PASS: %s", msg)


def test_constructor(builder, config):
    """Test every component created in __init__."""
    logging.info("\n=== Constructor Tests ===")

    # Reason model
    check(builder.reason_model is not None, "reason_model is not None")
    check(hasattr(builder.reason_model, "config"), "reason_model has config")
    check(
        builder.reason_model_hidden_dim == builder.reason_model.config.hidden_size,
        f"reason_model_hidden_dim={builder.reason_model_hidden_dim}",
    )

    # Tokenizer (built-in, paired with reason_model)
    check(builder.tokenizer is not None, "tokenizer is not None")
    check(builder.tokenizer.pad_token is not None, "tokenizer has pad_token")

    # input_proj
    check(builder.input_proj is not None, "input_proj exists")
    check(
        builder.input_proj.in_features == builder.reason_model_hidden_dim,
        "input_proj.in_features == reason_model_hidden_dim",
    )
    check(
        builder.input_proj.out_features == config.hidden_dim,
        "input_proj.out_features == hidden_dim",
    )

    # concept_queries
    check(
        len(builder.concept_queries) == config.num_levels,
        f"concept_queries count == {config.num_levels}",
    )
    for k, q in enumerate(builder.concept_queries):
        expected = (config.level_lengths[k], config.hidden_dim)
        check(q.shape == expected, f"concept_queries[{k}].shape == {expected}")
        check(q.requires_grad, f"concept_queries[{k}] requires_grad")

    # temperature
    check(builder.temperature.shape == (1,), "temperature shape == (1,)")
    check(builder.temperature.requires_grad, "temperature requires_grad")

    # level_projs
    check(
        len(builder.level_projs) == config.num_levels,
        f"level_projs count == {config.num_levels}",
    )
    for k, proj in enumerate(builder.level_projs):
        check(
            proj.in_features == config.hidden_dim,
            f"level_projs[{k}].in_features == hidden_dim",
        )
        check(
            proj.out_features == config.hidden_dim,
            f"level_projs[{k}].out_features == hidden_dim",
        )

    # level_attn
    check(
        len(builder.level_attn) == config.num_levels,
        f"level_attn count == {config.num_levels}",
    )

    # recon_decoder placeholder
    check(builder.recon_decoder is None, "recon_decoder is None (placeholder)")

    # Cache lists
    check(isinstance(builder._cached_attentions, list), "_cached_attentions is list")
    check(
        isinstance(builder._cached_base_concepts, list), "_cached_base_concepts is list"
    )

    # Helpers
    check(
        builder.get_total_concepts() == sum(config.level_lengths),
        f"get_total_concepts() == {sum(config.level_lengths)}",
    )


def test_encode_cot(builder, device, batch_size):
    """Test encode_cot with batch (text and tensor inputs)."""
    logging.info("\n=== encode_cot Tests ===")

    texts = [f"Problem {i}: What is {i} + {i+1}?" for i in range(batch_size)]

    # --- Test 1: text input (auto-tokenize) ---
    enc_out = builder.encode_cot(texts)
    check(isinstance(enc_out, EncoderOutput), "encode_cot(texts) returns EncoderOutput")
    check(enc_out.hidden_states.dim() == 3, "hidden_states dim == 3")
    check(
        enc_out.hidden_states.shape[0] == batch_size,
        f"hidden_states batch == {batch_size}",
    )
    check(
        enc_out.hidden_states.shape[-1] == builder.reason_model_hidden_dim,
        "hidden_states last_dim == reason_model_hidden_dim",
    )

    # --- Test 2: tensor input ---
    tokens = builder.tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=32
    )
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    enc_out2 = builder.encode_cot(input_ids, attention_mask)
    check(
        isinstance(enc_out2, EncoderOutput),
        "encode_cot(tensor) returns EncoderOutput",
    )
    check(
        enc_out2.hidden_states.shape[0] == batch_size,
        "encode_cot(tensor) batch correct",
    )
    # Same result as auto-tokenized
    check(
        torch.allclose(enc_out.hidden_states, enc_out2.hidden_states),
        "auto-tokenize == manual tokenize",
    )

    return enc_out


def test_forward(builder, device, config, batch_size):
    """Test forward() full pyramid construction."""
    logging.info("\n=== forward() Tests ===")

    texts = [f"Solve {i}: compute {i} * {i+2}." for i in range(batch_size)]
    enc_out = builder.encode_cot(texts)
    H = enc_out.hidden_states
    seq_len = H.shape[1]

    # Build pyramid
    output = builder(H)
    check(isinstance(output, PyramidOutput), "forward returns PyramidOutput")

    # Concepts count
    check(
        len(output.concepts) == config.num_levels,
        f"concepts count == {config.num_levels}",
    )

    # Per-level shapes
    for k, concepts in enumerate(output.concepts):
        expected = (batch_size, config.level_lengths[k], config.hidden_dim)
        check(concepts.shape == expected, f"concepts[{k}].shape == {expected}")

    # LevelOutput fields
    check(
        len(output.level_outputs) == config.num_levels,
        f"level_outputs count == {config.num_levels}",
    )
    for k, lo in enumerate(output.level_outputs):
        Lk = config.level_lengths[k]
        check(
            lo.concepts.shape == (batch_size, Lk, config.hidden_dim),
            f"level_outputs[{k}].concepts shape",
        )
        check(
            lo.base_concepts.shape == (batch_size, Lk, config.hidden_dim),
            f"level_outputs[{k}].base_concepts shape",
        )
        check(
            lo.attention_weights.shape == (batch_size, Lk, seq_len),
            f"level_outputs[{k}].attention_weights shape",
        )
        check(
            lo.reconstruction.shape == (batch_size, seq_len, config.hidden_dim),
            f"level_outputs[{k}].reconstruction shape",
        )

    # Residual decomposition: f_hat + f_rest == H_proj
    recomposed = output.reconstructed_hidden + output.residual_hidden
    diff = torch.abs(recomposed - output.projected_hidden).max().item()
    check(diff < 1e-4, f"f_hat + f_rest == H_proj (max_diff={diff:.2e})")

    # Hidden state shapes
    for name, tensor in [
        ("projected_hidden", output.projected_hidden),
        ("reconstructed_hidden", output.reconstructed_hidden),
        ("residual_hidden", output.residual_hidden),
    ]:
        check(
            tensor.shape == (batch_size, seq_len, config.hidden_dim),
            f"{name}.shape == ({batch_size}, {seq_len}, {config.hidden_dim})",
        )

    # PyramidOutput properties
    check(output.num_levels == config.num_levels, "num_levels correct")
    check(output.level_lengths == config.level_lengths, "level_lengths correct")
    check(output.total_concepts == sum(config.level_lengths), "total_concepts correct")
    check(
        len(output.all_attentions) == config.num_levels, "all_attentions length correct"
    )
    check(
        len(output.all_base_concepts) == config.num_levels,
        "all_base_concepts length correct",
    )
    cat = output.cat_concepts()
    check(
        cat.shape == (batch_size, sum(config.level_lengths), config.hidden_dim),
        "cat_concepts shape correct",
    )

    # Level 0: no refinement
    check(
        torch.allclose(
            output.level_outputs[0].concepts, output.level_outputs[0].base_concepts
        ),
        "level 0 concepts == base_concepts (no refinement)",
    )

    # Attention softmax: sum to 1 per concept slot
    for k, lo in enumerate(output.level_outputs):
        attn_sum = lo.attention_weights.sum(dim=-1)
        check(
            torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5),
            f"level {k} attention sums to 1",
        )

    return output


def test_forward_next_level(builder, device, config, batch_size):
    """Test forward_next_level step-by-step."""
    logging.info("\n=== forward_next_level() Tests ===")

    texts = [f"Step {i}: calculate {i+1} * {i+2}." for i in range(batch_size)]
    enc_out = builder.encode_cot(texts)
    H = enc_out.hidden_states

    # Sequential level extraction
    builder.clear_cache()
    prev_concepts = []
    for k in range(config.num_levels):
        level_out = builder.forward_next_level(
            H, previous_level_concepts=prev_concepts, target_level_index=k
        )
        check(
            isinstance(level_out, SingleLevelOutput),
            f"level {k} returns SingleLevelOutput",
        )
        check(level_out.level_index == k, f"level {k} index correct")
        check(
            level_out.concepts.shape[0] == batch_size, f"level {k} batch size correct"
        )
        check(
            level_out.concepts.shape[1] == config.level_lengths[k],
            f"level {k} concept count correct",
        )
        check(
            level_out.projected_hidden.shape == H.shape,
            f"level {k} projected_hidden shape matches H",
        )
        prev_concepts.append(level_out.concepts)

    check(
        len(builder._cached_attentions) == config.num_levels,
        "cache has num_levels attentions",
    )
    check(
        len(builder._cached_base_concepts) == config.num_levels,
        "cache has num_levels base_concepts",
    )

    # Clear cache
    builder.clear_cache()
    check(len(builder._cached_attentions) == 0, "clear_cache empties attentions")
    check(len(builder._cached_base_concepts) == 0, "clear_cache empties base_concepts")


def test_gsm8k_integration(builder, device, config, dataset, batch_size):
    """Test end-to-end with GSM8K data."""
    logging.info("\n=== GSM8K Integration Tests ===")

    n = min(batch_size, len(dataset))
    samples = [dataset[i] for i in range(n)]
    cot_texts = [s.cot_answer for s in samples]

    # Encode CoT via auto-tokenize
    enc_out = builder.encode_cot(cot_texts)
    check(isinstance(enc_out, EncoderOutput), "GSM8K encode_cot returns EncoderOutput")
    check(enc_out.hidden_states.shape[0] == n, f"GSM8K hidden_states batch == {n}")
    check(
        enc_out.hidden_states.shape[-1] == builder.reason_model_hidden_dim,
        "GSM8K hidden_states dim == reason_model_hidden_dim",
    )

    # Build pyramid
    pyramid = builder(enc_out.hidden_states)
    check(isinstance(pyramid, PyramidOutput), "GSM8K forward returns PyramidOutput")
    check(len(pyramid.concepts) == config.num_levels, "GSM8K concepts count correct")
    check(
        pyramid.total_concepts == sum(config.level_lengths),
        "GSM8K total_concepts correct",
    )
    check(pyramid.concepts[0].shape[0] == n, "GSM8K concept batch correct")
    check(
        pyramid.projected_hidden.shape[0] == n, "GSM8K projected_hidden batch correct"
    )


def test_gradient_flow(builder, device, batch_size):
    """Test that gradients flow through all learnable parameters."""
    logging.info("\n=== Gradient Flow Test ===")

    texts = [f"Compute {i} + {i+1}." for i in range(batch_size)]
    enc_out = builder.encode_cot(texts)

    builder.train()
    output = builder(enc_out.hidden_states)

    loss = sum(c.sum() for c in output.concepts)
    loss.backward()

    check(builder.input_proj.weight.grad is not None, "input_proj has gradient")
    check(builder.temperature.grad is not None, "temperature has gradient")
    for k, q in enumerate(builder.concept_queries):
        check(q.grad is not None, f"concept_queries[{k}] has gradient")
    for k, proj in enumerate(builder.level_projs):
        check(proj.weight.grad is not None, f"level_projs[{k}] has gradient")

    builder.eval()
    logging.info("PASS: all learnable parameters have gradients")


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

    # Run all tests
    test_constructor(builder, nlcp_config)
    test_encode_cot(builder, device, batch_size)
    test_forward(builder, device, nlcp_config, batch_size)
    test_forward_next_level(builder, device, nlcp_config, batch_size)
    test_gsm8k_integration(builder, device, nlcp_config, dataset, batch_size)
    test_gradient_flow(builder, device, batch_size)

    logging.info("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
