"""Comprehensive unit test for HybridConceptGenerator.

USAGE:
    # Run from project root:
    python examples/nlcpV3/utest/test_hybrid_generator.py -c configs/nlcpV2/utest/test_concept_pyramid.yml

    # Run specific test:
    python examples/nlcpV3/utest/test_hybrid_generator.py -c <config> --test training
    python examples/nlcpV3/utest/test_hybrid_generator.py -c <config> --test inference
    python examples/nlcpV3/utest/test_hybrid_generator.py -c <config> --test gsm8k

DESIGN SOURCE:
    Reference: examples/nlcpV3/concept_generator_hybrid.py
    Combines three best methods:
    1. ResidualAttentivePooling (coarse-to-fine backbone)
    2. MonotonicSoftAssignment (cross-attention bridge)
    3. AutoregressiveSoftBoundary (boundary ordering constraint)

TEST COVERAGE:
    1. Constructor: Parameter initialization, dimension checks
    2. Training Mode: forward(H) -> [C_0, ..., C_K] + aux with losses
    3. Inference Mode: forward(H, k, [C_0..C_{k-1}]) -> C_k
    4. forward_next_level: Step-by-step generation
    5. Dimension Flow: [B, L, D_encoder] -> [B, L_k, D]
    6. Loss Computation: recon_loss, order_loss, total_loss
    7. GSM8K Integration: Real data validation
    8. Edge Cases: Empty inputs, single level, gradient flow

DIMENSION SPECIFICATIONS:
    Input:  H ∈ [B, L, D_encoder]
    Output: [C_0, ..., C_K] where C_k ∈ [B, L_k, D]

    Level k processing:
        H_rest_k:      [B, L, D]          (residual hidden states)
        Q_k:           [L_k, D]           (learnable queries)
        A_k:           [B, L_k, L]        (attention weights)
        C_k:           [B, L_k, D]        (extracted concepts)
        H_recon_k:     [B, L, D]          (reconstruction from C_k)
"""

import argparse
from pathlib import Path
import sys
import traceback
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer

# Compute project paths relative to this file
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = PROJECT_ROOT / "examples"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLES_DIR))

from lmbase.dataset import registry
from nlcpV3.config import NLCPV3Config
from nlcpV3.encoder import NLCPV3Encoder
from nlcpV3.concept_generator_hybrid import HybridConceptGenerator
from ram.utils import load_config


# =============================================================================
# Test Result Tracking
# =============================================================================


class TestResults:
    """Track test results with detailed reporting."""

    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []

    def add_pass(self, test_name: str, details: str):
        self.passed.append((test_name, details))
        print(f"  ✓ PASS: {test_name}")
        if details:
            print(f"    {details}")

    def add_fail(self, test_name: str, error: str):
        self.failed.append((test_name, error))
        print(f"  ✗ FAIL: {test_name}")
        print(f"    Error: {error}")

    def add_warning(self, test_name: str, warning: str):
        self.warnings.append((test_name, warning))
        print(f"  ⚠ WARNING: {test_name}")
        print(f"    {warning}")

    def summary(self) -> str:
        total = len(self.passed) + len(self.failed)
        return (
            f"\n{'=' * 60}\n"
            f"Test Summary: {len(self.passed)}/{total} passed"
            f"{f', {len(self.failed)} failed' if self.failed else ''}"
            f"{f', {len(self.warnings)} warnings' if self.warnings else ''}"
            f"\n{'=' * 60}"
        )


# =============================================================================
# Configuration Builder
# =============================================================================


def build_nlcpV3_config(config: dict) -> NLCPV3Config:
    """Build NLCPV3Config from YAML configuration."""
    nlcp_config = config.get("nlcpV3", {})

    return NLCPV3Config(
        hidden_dim=nlcp_config.get("hidden_dim", 256),
        num_heads=nlcp_config.get("num_heads", 8),
        vocab_size=nlcp_config.get("vocab_size", 151936),
        num_levels=nlcp_config.get("num_levels", 6),
        level_lengths=nlcp_config.get("level_lengths", [1, 2, 4, 8, 16, 32]),
        max_seq_len=nlcp_config.get("max_seq_len", 512),
        dropout=nlcp_config.get("dropout", 0.1),
        rms_norm_eps=nlcp_config.get("rms_norm_eps", 1e-6),
        encoder_model_name=nlcp_config.get("encoder_model_name", "Qwen/Qwen2.5-0.5B"),
        encoder_num_layers=nlcp_config.get("encoder_num_layers", 4),
        encoder_freeze=nlcp_config.get("encoder_freeze", True),
        ntp_loss_weight=nlcp_config.get("ntp_loss_weight", 1.0),
        concept_loss_weight=nlcp_config.get("concept_loss_weight", 0.1),
        recon_loss_weight=nlcp_config.get("recon_loss_weight", 0.1),
        muP_scale=nlcp_config.get("muP_scale", 1.0),
    )


# =============================================================================
# Test Functions
# =============================================================================


def test_constructor(
    config: NLCPV3Config, results: TestResults
) -> HybridConceptGenerator:
    """Test HybridConceptGenerator constructor and parameter initialization."""
    print("\n--- Test: Constructor ---")

    try:
        encoder_hidden_dim = 896
        generator = HybridConceptGenerator(config, encoder_hidden_dim, 0.1, 1.0)

        # Check attributes
        assert generator.config == config, "Config mismatch"
        assert (
            generator.encoder_hidden_dim == encoder_hidden_dim
        ), "Encoder dim mismatch"
        results.add_pass("Attribute initialization", "")

        # Check components exist
        assert hasattr(generator, "input_proj"), "Missing input_proj"
        assert hasattr(generator, "concept_queries"), "Missing concept_queries"
        assert hasattr(generator, "temperature"), "Missing temperature"
        assert hasattr(generator, "level_projs"), "Missing level_projs"
        assert hasattr(generator, "level_attn"), "Missing level_attn"
        results.add_pass("Component existence", "")

        # Check dimensions
        assert generator.input_proj.in_features == encoder_hidden_dim
        assert generator.input_proj.out_features == config.hidden_dim
        results.add_pass(
            "Projection dimensions", f"{encoder_hidden_dim} -> {config.hidden_dim}"
        )

        # Check concept queries
        assert len(generator.concept_queries) == config.num_levels
        for i, queries in enumerate(generator.concept_queries):
            expected_shape = (config.level_lengths[i], config.hidden_dim)
            assert queries.shape == expected_shape, f"Level {i} query shape mismatch"
        results.add_pass(
            "Concept queries shapes",
            f"{[tuple(q.shape) for q in generator.concept_queries]}",
        )

        # Check level projections
        assert len(generator.level_projs) == config.num_levels
        for i, proj in enumerate(generator.level_projs):
            assert proj.in_features == config.hidden_dim
            assert proj.out_features == config.hidden_dim
        results.add_pass("Level projections dimensions", "")

        # Check cross-attention layers
        assert len(generator.level_attn) == config.num_levels
        results.add_pass("Cross-attention layers count", "")

        # Check total concepts
        total = generator.get_total_concepts()
        expected_total = sum(config.level_lengths)
        assert (
            total == expected_total
        ), f"Total concepts mismatch: {total} != {expected_total}"
        results.add_pass("Total concepts count", f"{total} concepts")

        return generator

    except Exception as e:
        results.add_fail("Constructor", str(e))
        traceback.print_exc()
        raise


def test_training_mode(
    generator: HybridConceptGenerator,
    config: NLCPV3Config,
    results: TestResults,
    batch_size: int = 2,
    seq_len: int = 128,
):
    """Test training mode: forward(H) -> all levels + losses."""
    print("\n--- Test: Training Mode ---")

    try:
        encoder_hidden_dim = 896
        H = torch.randn(batch_size, seq_len, encoder_hidden_dim)

        # Forward pass
        concepts, aux = generator(H, None, None)

        # Check output type
        assert isinstance(concepts, list), "Concepts should be a list"
        assert (
            len(concepts) == config.num_levels
        ), f"Expected {config.num_levels} levels"
        results.add_pass("Output structure", f"{len(concepts)} levels")

        # Check concept shapes
        for i, c in enumerate(concepts):
            expected_shape = (batch_size, config.level_lengths[i], config.hidden_dim)
            assert (
                c.shape == expected_shape
            ), f"Level {i} shape mismatch: {c.shape} != {expected_shape}"
        results.add_pass("Concept shapes", f"{[list(c.shape) for c in concepts]}")

        # Check aux dictionary
        required_keys = [
            "reconstructed_hidden",
            "residual_hidden",
            "recon_loss",
            "order_loss",
            "total_loss",
            "num_levels",
            "level_lengths",
            "method",
        ]
        for key in required_keys:
            assert key in aux, f"Missing aux key: {key}"
        results.add_pass("Aux dictionary keys", "")

        # Check loss values
        assert isinstance(
            aux["recon_loss"], torch.Tensor
        ), "recon_loss should be tensor"
        assert isinstance(
            aux["order_loss"], torch.Tensor
        ), "order_loss should be tensor"
        assert isinstance(
            aux["total_loss"], torch.Tensor
        ), "total_loss should be tensor"
        assert aux["recon_loss"].ndim == 0, "recon_loss should be scalar"
        assert aux["order_loss"].ndim == 0, "order_loss should be scalar"
        assert aux["total_loss"].ndim == 0, "total_loss should be scalar"
        results.add_pass(
            "Loss tensor shapes",
            f"recon={aux['recon_loss'].item():.4f}, order={aux['order_loss'].item():.4f}",
        )

        # Check loss relationship
        expected_total = (
            aux["recon_loss"] + generator.order_loss_weight * aux["order_loss"]
        )
        assert torch.allclose(
            aux["total_loss"], expected_total, rtol=1e-5
        ), "Total loss mismatch"
        results.add_pass("Loss computation", "")

        # Check reconstructed hidden
        assert aux["reconstructed_hidden"].shape == (
            batch_size,
            seq_len,
            config.hidden_dim,
        )
        results.add_pass("Reconstructed hidden shape", "")

        # Check residual hidden
        assert aux["residual_hidden"].shape == (batch_size, seq_len, config.hidden_dim)
        results.add_pass("Residual hidden shape", "")

        # Check method tag
        assert aux["method"] == "hybrid", f"Unexpected method: {aux['method']}"
        results.add_pass("Method tag", "")

        return concepts, aux

    except Exception as e:
        results.add_fail("Training mode", str(e))
        traceback.print_exc()
        raise


def test_inference_mode(
    generator: HybridConceptGenerator,
    config: NLCPV3Config,
    results: TestResults,
    batch_size: int = 2,
    seq_len: int = 128,
):
    """Test inference mode: forward(H, k, previous) -> single level."""
    print("\n--- Test: Inference Mode ---")

    try:
        encoder_hidden_dim = 896
        H = torch.randn(batch_size, seq_len, encoder_hidden_dim)

        # Reset cached attentions
        generator._cached_attentions = []

        # Level 0
        C_0, aux_0 = generator(H, 0, None)
        expected_shape = (batch_size, config.level_lengths[0], config.hidden_dim)
        assert (
            C_0.shape == expected_shape
        ), f"Level 0 shape mismatch: {C_0.shape} != {expected_shape}"
        assert aux_0["target_level_index"] == 0
        assert aux_0["method"] == "hybrid_next_level"
        results.add_pass("Level 0 generation", f"shape={list(C_0.shape)}")

        # Level 1
        C_1, aux_1 = generator(H, 1, [C_0])
        expected_shape = (batch_size, config.level_lengths[1], config.hidden_dim)
        assert (
            C_1.shape == expected_shape
        ), f"Level 1 shape mismatch: {C_1.shape} != {expected_shape}"
        assert aux_1["target_level_index"] == 1
        results.add_pass("Level 1 generation", f"shape={list(C_1.shape)}")

        # Level 2
        C_2, aux_2 = generator(H, 2, [C_0, C_1])
        expected_shape = (batch_size, config.level_lengths[2], config.hidden_dim)
        assert C_2.shape == expected_shape
        results.add_pass("Level 2 generation", f"shape={list(C_2.shape)}")

        # Generate all levels sequentially
        generator._cached_attentions = []
        all_concepts = []
        for k in range(config.num_levels):
            C_k, aux_k = generator(
                H,
                target_level_index=k,
                previous_level_concepts=all_concepts if all_concepts else None,
            )
            all_concepts.append(C_k)

            expected_shape = (batch_size, config.level_lengths[k], config.hidden_dim)
            assert C_k.shape == expected_shape

        results.add_pass(
            "Sequential generation all levels", f"Generated {len(all_concepts)} levels"
        )

        # Verify consistency with training mode
        generator._cached_attentions = []
        training_concepts, _ = generator(H, None, None)

        for k in range(config.num_levels):
            # Shapes should match
            assert all_concepts[k].shape == training_concepts[k].shape

        results.add_pass("Training-inference consistency", "")

        return all_concepts

    except Exception as e:
        results.add_fail("Inference mode", str(e))
        traceback.print_exc()
        raise


def test_forward_next_level(
    generator: HybridConceptGenerator,
    config: NLCPV3Config,
    results: TestResults,
    batch_size: int = 2,
    seq_len: int = 128,
):
    """Test forward_next_level method directly."""
    print("\n--- Test: forward_next_level ---")

    try:
        encoder_hidden_dim = 896
        H = torch.randn(batch_size, seq_len, encoder_hidden_dim)

        # Reset cache
        generator._cached_attentions = []

        # Level 0
        C_0 = generator.forward_next_level(H, None, 0)
        expected_shape = (batch_size, config.level_lengths[0], config.hidden_dim)
        assert C_0.shape == expected_shape
        results.add_pass("forward_next_level level 0", "")

        # Level 1
        C_1 = generator.forward_next_level(H, [C_0], 1)
        expected_shape = (batch_size, config.level_lengths[1], config.hidden_dim)
        assert C_1.shape == expected_shape
        results.add_pass("forward_next_level level 1", "")

        # Level 2 with cross-attention
        C_2 = generator.forward_next_level(H, [C_0, C_1], 2)
        expected_shape = (batch_size, config.level_lengths[2], config.hidden_dim)
        assert C_2.shape == expected_shape
        results.add_pass("forward_next_level level 2 (with cross-attn)", "")

        # Verify cache is populated
        assert (
            len(generator._cached_attentions) >= 3
        ), "Cache should have at least 3 entries"
        results.add_pass(
            "Attention cache population", f"{len(generator._cached_attentions)} entries"
        )

        return C_0, C_1, C_2

    except Exception as e:
        results.add_fail("forward_next_level", str(e))
        traceback.print_exc()
        raise


def test_gradient_flow(
    generator: HybridConceptGenerator,
    config: NLCPV3Config,
    results: TestResults,
    batch_size: int = 2,
    seq_len: int = 128,
):
    """Test gradient flow through the generator."""
    print("\n--- Test: Gradient Flow ---")

    try:
        encoder_hidden_dim = 896
        H = torch.randn(batch_size, seq_len, encoder_hidden_dim)

        # Forward pass
        concepts, aux = generator(H, None, None)
        loss = aux["total_loss"]

        # Backward pass
        loss.backward()

        # Check gradients exist
        assert (
            generator.input_proj.weight.grad is not None
        ), "input_proj has no gradient"
        results.add_pass("input_proj gradient", "")

        for i, queries in enumerate(generator.concept_queries):
            assert queries.grad is not None, f"Level {i} queries have no gradient"
        results.add_pass("Concept queries gradients", "")

        for i, proj in enumerate(generator.level_projs):
            assert proj.weight.grad is not None, f"Level {i} proj has no gradient"
        results.add_pass("Level projections gradients", "")

        # Check temperature gradient
        assert generator.temperature.grad is not None, "Temperature has no gradient"
        results.add_pass("Temperature gradient", "")

        # Check gradient magnitudes
        grad_norm = generator.input_proj.weight.grad.norm().item()
        results.add_pass("Gradient magnitude", f"input_proj grad norm: {grad_norm:.4f}")

    except Exception as e:
        results.add_fail("Gradient flow", str(e))
        traceback.print_exc()
        raise


def test_edge_cases(
    generator: HybridConceptGenerator,
    config: NLCPV3Config,
    results: TestResults,
):
    """Test edge cases and boundary conditions."""
    print("\n--- Test: Edge Cases ---")

    try:
        encoder_hidden_dim = 896

        # Test with batch_size=1
        H_1 = torch.randn(1, 64, encoder_hidden_dim)
        concepts, aux = generator(H_1, None, None)
        assert len(concepts) == config.num_levels
        for i, c in enumerate(concepts):
            assert c.shape[0] == 1, f"Batch size 1 failed at level {i}"
        results.add_pass("Batch size 1", "")

        # Test with longer sequence
        H_long = torch.randn(2, 256, encoder_hidden_dim)
        concepts, aux = generator(H_long, None, None)
        assert len(concepts) == config.num_levels
        results.add_pass("Long sequence (256)", "")

        # Test with short sequence
        H_short = torch.randn(2, 16, encoder_hidden_dim)
        concepts, aux = generator(H_short, None, None)
        assert len(concepts) == config.num_levels
        results.add_pass("Short sequence (16)", "")

        # Test inference with single level
        generator._cached_attentions = []
        H = torch.randn(2, 64, encoder_hidden_dim)
        C_0, aux = generator(H, 0, None)
        assert C_0.shape == (2, config.level_lengths[0], config.hidden_dim)
        results.add_pass("Single level inference", "")

    except Exception as e:
        results.add_fail("Edge cases", str(e))
        traceback.print_exc()
        raise


def test_gsm8k_integration(
    config: NLCPV3Config,
    config_dict: dict,
    results: TestResults,
    max_samples: int,
):
    """Test with real GSM8K data."""
    print("\n--- Test: GSM8K Integration ---")

    try:
        # Load GSM8K dataset
        data_cfg = config_dict["data"]
        dataset = registry.get(data_cfg, split=data_cfg["split"])
        results.add_pass("GSM8K dataset loaded", f"{len(dataset)} samples available")

        # Initialize encoder and generator
        encoder = NLCPV3Encoder(config)
        encoder_hidden_dim = encoder.model.config.hidden_size
        generator = HybridConceptGenerator(
            config,
            encoder_hidden_dim=encoder_hidden_dim,
            order_loss_weight=0.1,
            order_margin=1.0,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.encoder_model_name, trust_remote_code=True
        )

        for i in range(min(max_samples, len(dataset))):
            sample = dataset[i]
            if hasattr(sample, "question"):
                question = sample.question
                cot = (
                    sample.cot_answer
                    if hasattr(sample, "cot_answer")
                    else sample.answer
                )
            else:
                question = sample["question"]
                cot = sample.get("cot_answer", sample.get("answer", ""))

            # Tokenize
            text = f"Question: {question}\nCoT: {cot}"
            tokens = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=config.max_seq_len,
            )
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]

            # Encode
            with torch.no_grad():
                H = encoder.forward_training(input_ids, attention_mask)

                # Generate concepts
                concepts, aux = generator(H, None, None)

            # Verify outputs
            assert len(concepts) == config.num_levels
            for k, c in enumerate(concepts):
                assert c.shape == (1, config.level_lengths[k], config.hidden_dim)

            results.add_pass(
                f"GSM8K sample {i+1}",
                f"seq_len={input_ids.shape[1]}, losses=(recon={aux['recon_loss'].item():.3f}, order={aux['order_loss'].item():.3f})",
            )

    except Exception as e:
        results.add_fail("GSM8K integration", str(e))
        traceback.print_exc()


# =============================================================================
# Main Test Runner
# =============================================================================


def run_all_tests(config_path: str, specific_test: Optional[str]):
    """Run all tests or a specific test."""
    print("=" * 60)
    print("HybridConceptGenerator Comprehensive Test Suite")
    print("=" * 60)

    # Load configuration
    config_dict = load_config(config_path)
    config = build_nlcpV3_config(config_dict)

    print(f"\nConfiguration:")
    print(f"  hidden_dim: {config.hidden_dim}")
    print(f"  num_levels: {config.num_levels}")
    print(f"  level_lengths: {config.level_lengths}")
    print(f"  encoder_model: {config.encoder_model_name}")

    results = TestResults()

    try:
        # Test constructor
        if specific_test is None or specific_test == "constructor":
            generator = test_constructor(config, results)

        # Skip other tests if only testing constructor
        if specific_test == "constructor":
            print(results.summary())
            return len(results.failed) == 0

        # Re-create generator for other tests
        generator = HybridConceptGenerator(
            config, encoder_hidden_dim=896, order_loss_weight=0.1, order_margin=1.0
        )

        # Test training mode
        if specific_test is None or specific_test == "training":
            test_training_mode(generator, config, results)

        # Test inference mode
        if specific_test is None or specific_test == "inference":
            test_inference_mode(generator, config, results)

        # Test forward_next_level
        if specific_test is None or specific_test == "next_level":
            test_forward_next_level(generator, config, results)

        # Test gradient flow
        if specific_test is None or specific_test == "gradient":
            test_gradient_flow(generator, config, results)

        # Test edge cases
        if specific_test is None or specific_test == "edge":
            test_edge_cases(generator, config, results)

        # Test GSM8K integration
        if specific_test is None or specific_test == "gsm8k":
            test_gsm8k_integration(config, config_dict, results, 3)

    except Exception as e:
        print(f"\nFatal error: {e}")
        traceback.print_exc()
        return False

    # Print summary
    print(results.summary())

    return len(results.failed) == 0


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HybridConceptGenerator")
    parser.add_argument(
        "-c", "--config", required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--test",
        choices=[
            "constructor",
            "training",
            "inference",
            "next_level",
            "gradient",
            "edge",
            "gsm8k",
        ],
        help="Run specific test (default: all)",
    )

    args = parser.parse_args()

    success = run_all_tests(args.config, args.test)
    sys.exit(0 if success else 1)
