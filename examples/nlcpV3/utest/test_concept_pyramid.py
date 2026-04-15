"""Comprehensive test for all Concept Generator methods in NLCP V3.

USAGE:
    # Run from project root with config file:
    python examples/nlcpV3/utest/test_concept_pyramid.py -c configs/nlcpV2/utest/test_concept_pyramid.yml

DESIGN SOURCE:
    Reference: docs/concept-pyramid-V3.md
    - Section 2: Architecture
    - Section 3: Training (Concept Extraction)
    - Section 4: Inference (Concept Generation)

PURPOSE:
    Validate all concept generator implementations:

    BASIC TRAINING EXTRACTORS (6 methods):
        1. ResidualAttentivePoolingConceptGenerator - VAR-style residual decomposition
        2. PositionConstrainedConceptGenerator - Position-based attention bias
        3. HardOrderedMaskConceptGenerator - Hard position masks
        4. RecursiveOrderedConceptGenerator - Recursive concept extraction
        5. OrderConstrainedTrainingConceptGenerator - Order constraints
        6. RobustOrderedConceptGenerator - Robust ordered extraction

    ADVANCED CAUSAL TRAINING EXTRACTORS (5 methods):
        7. MonotonicSoftAssignmentConceptGenerator - Monotonic allocation matrix
        8. CausalSequentialRefinementConceptGenerator - Causal transformer refinement
        9. ContinuousCausalKernelConceptGenerator - Continuous position mapping
        10. AutoregressiveSoftBoundaryConceptGenerator - AR boundary prediction
        11. CausalSoftPoolingConceptGenerator - Complete pipeline

    INFERENCE GENERATOR (1 method):
        12. AutoregressiveConceptGenerator - Next-level AR generation

TEST COVERAGE:
    - All generators produce correct output shapes
    - forward(encoder_hidden_states) for all levels (training mode)
    - forward(encoder_hidden_states, target_level_index=k, previous_level_concepts=[...])
      for single level (inference mode, only for generators with forward_next_level implemented)
    - forward_next_level() for step-by-step generation (only for implemented generators)
    - Dimension flow: [B, L, D_encoder] -> [B, L_k, D] for each level
    - GSM8K real data integration

NOTE:
    Generators without forward_next_level() implementation are skipped for single-level tests.
    Only ResidualAttentivePoolingConceptGenerator currently implements the full next-level interface.
"""

import argparse
from pathlib import Path
import sys
import traceback
from typing import List, Dict, Any, Type

import torch
from torch import nn
from transformers import AutoTokenizer

# Compute project paths relative to this file
# This file: examples/nlcpV3/utest/test_concept_pyramid.py
# Project root: 3 levels up
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = PROJECT_ROOT / "examples"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLES_DIR))

from lmbase.dataset import registry
from nlcpV3.config import NLCPV3Config
from nlcpV3.encoder import NLCPV3Encoder
from nlcpV3.concept_generator import (
    BaseConceptGenerator,
    # Basic training extractors
    ResidualAttentivePoolingConceptGenerator,
    PositionConstrainedConceptGenerator,
    HardOrderedMaskConceptGenerator,
    RecursiveOrderedConceptGenerator,
    OrderConstrainedTrainingConceptGenerator,
    RobustOrderedConceptGenerator,
    # Advanced causal training extractors
    MonotonicSoftAssignmentConceptGenerator,
    CausalSequentialRefinementConceptGenerator,
    ContinuousCausalKernelConceptGenerator,
    AutoregressiveSoftBoundaryConceptGenerator,
    CausalSoftPoolingConceptGenerator,
    # Inference generator
    AutoregressiveConceptGenerator,
)
from ram.utils import load_config


# =============================================================================
# Configuration Builder
# =============================================================================


def build_nlcpV3_config(config: dict) -> NLCPV3Config:
    """Build NLCPV3Config from YAML configuration.

    Args:
        config: Configuration dictionary from YAML file

    Returns:
        NLCPV3Config instance
    """
    model_cfg = config["model"]
    encoder_cfg = model_cfg["encoder"]
    pyramid_cfg = model_cfg["pyramid"]
    decoder_cfg = model_cfg["decoder"]
    loss_cfg = model_cfg["loss_weights"]

    return NLCPV3Config(
        hidden_dim=pyramid_cfg["hidden_dim"],
        num_heads=pyramid_cfg["num_heads"],
        vocab_size=decoder_cfg["vocab_size"],
        num_levels=pyramid_cfg["num_levels"],
        level_lengths=pyramid_cfg["level_lengths"],
        max_seq_len=pyramid_cfg["max_seq_len"],
        dropout=decoder_cfg["dropout"],
        rms_norm_eps=decoder_cfg["rms_norm_eps"],
        encoder_model_name=encoder_cfg["encoder_model_name"],
        encoder_num_layers=encoder_cfg["encoder_num_layers"],
        encoder_freeze=encoder_cfg["encoder_freeze"],
        ntp_loss_weight=loss_cfg["ntp_loss_weight"],
        concept_loss_weight=loss_cfg["concept_loss_weight"],
        recon_loss_weight=loss_cfg["recon_loss_weight"],
        muP_scale=decoder_cfg["muP_scale"],
    )


# =============================================================================
# Data Loading
# =============================================================================


def load_gsm8k_batch(config: dict, nlcp_config: NLCPV3Config) -> Dict[str, Any]:
    """Load GSM8K batch and encode to hidden states.

    PURPOSE:
        Load real GSM8K data and encode to hidden states for testing.

    Returns:
        Dictionary with:
            - encoder_hidden_states: [B, L, D_encoder]
            - input_ids: [B, L]
            - attention_mask: [B, L]
            - questions: List of question strings
            - cot_texts: List of CoT strings
    """
    print("\n" + "=" * 70)
    print("LOADING GSM8K DATA")
    print("=" * 70)

    # Load GSM8K dataset via lmbase
    data_cfg = config["data"]
    dataset = registry.get(data_cfg, split=data_cfg["split"])
    print(f"  Dataset: {data_cfg['data_name']}")
    print(f"  Split: {data_cfg['split']}")
    print(f"  Total samples: {len(dataset)}")

    # Get batch_size samples
    batch_size = config["training"]["batch_size"]
    print(f"  Loading {batch_size} samples...")

    questions = []
    cot_texts = []
    for i in range(min(batch_size, len(dataset))):
        sample = dataset[i]
        if hasattr(sample, "question") and hasattr(sample, "cot_answer"):
            question = sample.question
            cot = sample.cot_answer
        elif isinstance(sample, dict):
            question = sample["question"]
            cot = sample["cot_answer"]
        else:
            question = str(sample)
            cot = ""
        questions.append(question)
        cot_texts.append(cot)

    # Pad if needed
    while len(questions) < batch_size:
        questions.append("")
        cot_texts.append("")

    # Tokenize Q+CoT
    full_texts = [
        f"Question: {q}\nReasoning: {c}" for q, c in zip(questions, cot_texts)
    ]

    tokenizer = AutoTokenizer.from_pretrained(nlcp_config.encoder_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(
        full_texts,
        max_length=nlcp_config.max_seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    print(f"  Tokenized shape: {list(input_ids.shape)}")

    # Encode to hidden states
    encoder = NLCPV3Encoder(nlcp_config)
    encoder.eval()

    with torch.no_grad():
        encoder_hidden_states = encoder.forward_training(input_ids, attention_mask)

    print(f"  Encoder hidden states: {list(encoder_hidden_states.shape)}")

    return {
        "encoder_hidden_states": encoder_hidden_states,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "questions": questions,
        "cot_texts": cot_texts,
    }


# =============================================================================
# Test: Individual Concept Generator
# =============================================================================


def test_concept_generator(
    generator_class: Type[nn.Module],
    generator_name: str,
    nlcp_config: NLCPV3Config,
    encoder_hidden_states: torch.Tensor,
    test_single_level: bool = True,
) -> Dict[str, Any]:
    """Test a single concept generator implementation.

    PURPOSE:
        Comprehensive test for one concept generator:
        1. All levels generation (training mode)
        2. Single level generation (inference mode) if supported
        3. Output shape validation

    Args:
        generator_class: The generator class to test
        generator_name: Human-readable name for logging
        nlcp_config: Configuration
        encoder_hidden_states: Input hidden states [B, L, D_encoder]
        test_single_level: Whether to test single-level inference mode

    Returns:
        Dictionary with test results
    """
    print(f"\n{'─' * 70}")
    print(f"TESTING: {generator_name}")
    print(f"{'─' * 70}")

    encoder_hidden_dim = encoder_hidden_states.shape[2]
    batch_size = encoder_hidden_states.shape[0]

    # Check if class has abstract method forward_next_level
    has_forward_next_level = False
    if hasattr(generator_class, "forward_next_level"):
        # Check if it's actually implemented (not abstract)
        import inspect

        method = getattr(generator_class, "forward_next_level")
        has_forward_next_level = not getattr(method, "__isabstractmethod__", False)

    # Instantiate generator
    try:
        generator = generator_class(nlcp_config, encoder_hidden_dim)
        generator.eval()
    except TypeError as e:
        if "abstract method" in str(e):
            print(f"  ⚠️  SKIPPED: Not yet implemented (abstract method)")
            return {
                "generator_name": generator_name,
                "skipped": True,
                "reason": "abstract_method_not_implemented",
            }
        raise

    # =========================================================================
    # Test 1: All levels generation (training mode)
    # =========================================================================
    print(f"\n  [1] Testing forward(encoder_hidden_states) - All levels")

    with torch.no_grad():
        output = generator(encoder_hidden_states)

    # Handle different return formats
    if isinstance(output, tuple):
        all_level_concepts, aux = output
    else:
        all_level_concepts = output
        aux = {}

    # Validate all levels output
    assert isinstance(
        all_level_concepts, list
    ), f"Expected list, got {type(all_level_concepts)}"
    assert (
        len(all_level_concepts) == nlcp_config.num_levels
    ), f"Expected {nlcp_config.num_levels} levels, got {len(all_level_concepts)}"

    print(f"      ✓ Generated {len(all_level_concepts)} levels")

    for level_idx, level_concepts in enumerate(all_level_concepts):
        expected_num_concepts = nlcp_config.level_lengths[level_idx]
        expected_shape = (batch_size, expected_num_concepts, nlcp_config.hidden_dim)
        actual_shape = tuple(level_concepts.shape)

        assert (
            actual_shape == expected_shape
        ), f"Level {level_idx}: expected {expected_shape}, got {actual_shape}"

        print(f"        Level {level_idx}: {list(actual_shape)} ✓")

    # Check aux info
    if aux:
        print(f"      Aux keys: {list(aux.keys())}")

    # =========================================================================
    # Test 2: Single level generation (inference mode) - if supported
    # =========================================================================
    if test_single_level and has_forward_next_level:
        print(
            f"\n  [2] Testing forward(encoder_hidden_states, target_level_index=k) - Single level"
        )

        # Reset any cached state
        if hasattr(generator, "_cached_attentions"):
            generator._cached_attentions = []

        # Generate level 0
        with torch.no_grad():
            level_0, aux_0 = generator(
                encoder_hidden_states,
                target_level_index=0,
                previous_level_concepts=None,
            )

        expected_shape = (
            batch_size,
            nlcp_config.level_lengths[0],
            nlcp_config.hidden_dim,
        )
        assert (
            tuple(level_0.shape) == expected_shape
        ), f"Level 0: expected {expected_shape}, got {tuple(level_0.shape)}"
        print(f"      Level 0: {list(level_0.shape)} ✓")

        # Generate level 1 (depends on level 0)
        with torch.no_grad():
            level_1, aux_1 = generator(
                encoder_hidden_states,
                target_level_index=1,
                previous_level_concepts=[level_0],
            )

        expected_shape = (
            batch_size,
            nlcp_config.level_lengths[1],
            nlcp_config.hidden_dim,
        )
        assert (
            tuple(level_1.shape) == expected_shape
        ), f"Level 1: expected {expected_shape}, got {tuple(level_1.shape)}"
        print(f"      Level 1: {list(level_1.shape)} ✓")

    # =========================================================================
    # Test 3: forward_next_level direct call - if supported
    # =========================================================================
    if has_forward_next_level:
        print(f"\n  [3] Testing forward_next_level() - Direct call")

        # Reset any cached state
        if hasattr(generator, "_cached_attentions"):
            generator._cached_attentions = []

        with torch.no_grad():
            level_0_direct = generator.forward_next_level(
                encoder_hidden_states,
                None,  # previous_level_concepts
                0,  # target_level_index
            )

        expected_shape = (
            batch_size,
            nlcp_config.level_lengths[0],
            nlcp_config.hidden_dim,
        )
        assert (
            tuple(level_0_direct.shape) == expected_shape
        ), f"Direct level 0: expected {expected_shape}, got {tuple(level_0_direct.shape)}"
        print(f"      Level 0 (direct): {list(level_0_direct.shape)} ✓")

    print(f"\n  ✅ {generator_name} PASSED")

    return {
        "generator_name": generator_name,
        "num_levels": len(all_level_concepts),
        "level_shapes": [list(c.shape) for c in all_level_concepts],
        "aux_keys": list(aux.keys()) if aux else [],
        "supports_next_level": has_forward_next_level,
    }


# =============================================================================
# Test: All Basic Training Extractors
# =============================================================================


def test_basic_training_extractors(
    nlcp_config: NLCPV3Config,
    encoder_hidden_states: torch.Tensor,
) -> List[Dict[str, Any]]:
    """Test all 6 basic training extractors.

    BASIC EXTRACTORS:
        1. ResidualAttentivePoolingConceptGenerator
        2. PositionConstrainedConceptGenerator
        3. HardOrderedMaskConceptGenerator
        4. RecursiveOrderedConceptGenerator
        5. OrderConstrainedTrainingConceptGenerator
        6. RobustOrderedConceptGenerator
    """
    print("\n" + "=" * 70)
    print("TESTING: BASIC TRAINING EXTRACTORS (6 methods)")
    print("=" * 70)

    basic_extractors = [
        (
            ResidualAttentivePoolingConceptGenerator,
            "ResidualAttentivePoolingConceptGenerator",
        ),
        (PositionConstrainedConceptGenerator, "PositionConstrainedConceptGenerator"),
        (HardOrderedMaskConceptGenerator, "HardOrderedMaskConceptGenerator"),
        (RecursiveOrderedConceptGenerator, "RecursiveOrderedConceptGenerator"),
        (
            OrderConstrainedTrainingConceptGenerator,
            "OrderConstrainedTrainingConceptGenerator",
        ),
        (RobustOrderedConceptGenerator, "RobustOrderedConceptGenerator"),
    ]

    results = []
    for extractor_class, name in basic_extractors:
        try:
            result = test_concept_generator(
                extractor_class, name, nlcp_config, encoder_hidden_states
            )
            results.append(result)
        except Exception as e:
            print(f"\n  ❌ {name} FAILED: {e}")
            traceback.print_exc()
            results.append({"generator_name": name, "error": str(e)})

    return results


# =============================================================================
# Test: All Advanced Causal Training Extractors
# =============================================================================


def test_advanced_causal_extractors(
    nlcp_config: NLCPV3Config,
    encoder_hidden_states: torch.Tensor,
) -> List[Dict[str, Any]]:
    """Test all 5 advanced causal training extractors.

    ADVANCED CAUSAL EXTRACTORS:
        1. MonotonicSoftAssignmentConceptGenerator
        2. CausalSequentialRefinementConceptGenerator
        3. ContinuousCausalKernelConceptGenerator
        4. AutoregressiveSoftBoundaryConceptGenerator
        5. CausalSoftPoolingConceptGenerator
    """
    print("\n" + "=" * 70)
    print("TESTING: ADVANCED CAUSAL TRAINING EXTRACTORS (5 methods)")
    print("=" * 70)

    advanced_extractors = [
        (
            MonotonicSoftAssignmentConceptGenerator,
            "MonotonicSoftAssignmentConceptGenerator",
        ),
        (
            CausalSequentialRefinementConceptGenerator,
            "CausalSequentialRefinementConceptGenerator",
        ),
        (
            ContinuousCausalKernelConceptGenerator,
            "ContinuousCausalKernelConceptGenerator",
        ),
        (
            AutoregressiveSoftBoundaryConceptGenerator,
            "AutoregressiveSoftBoundaryConceptGenerator",
        ),
        (CausalSoftPoolingConceptGenerator, "CausalSoftPoolingConceptGenerator"),
    ]

    results = []
    for extractor_class, name in advanced_extractors:
        try:
            result = test_concept_generator(
                extractor_class, name, nlcp_config, encoder_hidden_states
            )
            results.append(result)
        except Exception as e:
            print(f"\n  ❌ {name} FAILED: {e}")
            traceback.print_exc()
            results.append({"generator_name": name, "error": str(e)})

    return results


# =============================================================================
# Test: Inference Generator
# =============================================================================


def test_inference_generator(
    nlcp_config: NLCPV3Config,
    encoder_hidden_states: torch.Tensor,
) -> Dict[str, Any]:
    """Test the inference-only AutoregressiveConceptGenerator.

    PURPOSE:
        Test the next-level autoregressive concept generator used for
        inference (Q → Generated Concepts, without CoT).
    """
    print("\n" + "=" * 70)
    print("TESTING: INFERENCE GENERATOR")
    print("=" * 70)

    encoder_hidden_dim = encoder_hidden_states.shape[2]
    batch_size = encoder_hidden_states.shape[0]

    generator = AutoregressiveConceptGenerator(nlcp_config, encoder_hidden_dim)
    generator.eval()

    print(f"\n  Testing AutoregressiveConceptGenerator")
    print(f"  Note: This generator is for inference (no CoT)")

    # Test forward_inference
    with torch.no_grad():
        generated_concepts = generator(encoder_hidden_states)

    assert isinstance(
        generated_concepts, list
    ), f"Expected list, got {type(generated_concepts)}"
    assert (
        len(generated_concepts) == nlcp_config.num_levels
    ), f"Expected {nlcp_config.num_levels} levels, got {len(generated_concepts)}"

    print(f"\n  ✓ Generated {len(generated_concepts)} levels")

    for level_idx, level_concepts in enumerate(generated_concepts):
        expected_num_concepts = nlcp_config.level_lengths[level_idx]
        expected_shape = (batch_size, expected_num_concepts, nlcp_config.hidden_dim)
        actual_shape = tuple(level_concepts.shape)

        assert (
            actual_shape == expected_shape
        ), f"Level {level_idx}: expected {expected_shape}, got {actual_shape}"

        print(f"    Level {level_idx}: {list(actual_shape)} ✓")

    print(f"\n  ✅ AutoregressiveConceptGenerator PASSED")

    return {
        "generator_name": "AutoregressiveConceptGenerator",
        "num_levels": len(generated_concepts),
        "level_shapes": [list(c.shape) for c in generated_concepts],
    }


# =============================================================================
# Test Summary
# =============================================================================


def print_test_summary(all_results: Dict[str, Any]):
    """Print comprehensive test summary."""
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    total_passed = 0
    total_failed = 0
    total_skipped = 0

    # Basic extractors
    print("\n📊 BASIC TRAINING EXTRACTORS:")
    for result in all_results.get("basic_extractors", []):
        name = result.get("generator_name", "unknown")
        if result.get("skipped"):
            print(f"   ⏭️  {name}: SKIPPED - {result.get('reason', 'unknown')}")
            total_skipped += 1
        elif "error" in result:
            print(f"   ❌ {name}: FAILED - {result['error']}")
            total_failed += 1
        else:
            supports_next = "✓ next-level" if result.get("supports_next_level") else ""
            print(
                f"   ✅ {name}: PASSED ({result['num_levels']} levels) {supports_next}"
            )
            total_passed += 1

    # Advanced extractors
    print("\n📊 ADVANCED CAUSAL EXTRACTORS:")
    for result in all_results.get("advanced_extractors", []):
        name = result.get("generator_name", "unknown")
        if result.get("skipped"):
            print(f"   ⏭️  {name}: SKIPPED - {result.get('reason', 'unknown')}")
            total_skipped += 1
        elif "error" in result:
            print(f"   ❌ {name}: FAILED - {result['error']}")
            total_failed += 1
        else:
            supports_next = "✓ next-level" if result.get("supports_next_level") else ""
            print(
                f"   ✅ {name}: PASSED ({result['num_levels']} levels) {supports_next}"
            )
            total_passed += 1

    # Inference generator
    print("\n📊 INFERENCE GENERATOR:")
    inf_result = all_results.get("inference_generator", {})
    if inf_result.get("skipped"):
        print(f"   ⏭️  AutoregressiveConceptGenerator: SKIPPED")
        total_skipped += 1
    elif "error" in inf_result:
        print(f"   ❌ AutoregressiveConceptGenerator: FAILED - {inf_result['error']}")
        total_failed += 1
    else:
        print(f"   ✅ AutoregressiveConceptGenerator: PASSED")
        total_passed += 1

    # Total
    print("\n" + "─" * 70)
    print(
        f"TOTAL: {total_passed} passed, {total_failed} failed, {total_skipped} skipped"
    )
    print("─" * 70)

    if total_failed == 0 and total_passed > 0:
        print(
            f"\n🎉 ALL IMPLEMENTED TESTS PASSED! ({total_skipped} methods pending implementation)"
        )
    elif total_failed > 0:
        print(f"\n⚠️  {total_failed} TEST(S) FAILED")
    else:
        print("\n⚠️  NO TESTS RUN")


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="NLCP V3: Comprehensive Concept Generator Test"
    )
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to config file"
    )
    parser.add_argument(
        "--skip-basic", action="store_true", help="Skip basic extractors"
    )
    parser.add_argument(
        "--skip-advanced", action="store_true", help="Skip advanced extractors"
    )
    parser.add_argument(
        "--skip-inference", action="store_true", help="Skip inference generator"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    nlcp_config = build_nlcpV3_config(config)

    print("\n" + "=" * 70)
    print("NLCP V3: COMPREHENSIVE CONCEPT GENERATOR TEST")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Hidden dim: {nlcp_config.hidden_dim}")
    print(f"  Num levels: {nlcp_config.num_levels}")
    print(f"  Level lengths: {nlcp_config.level_lengths}")
    print(f"  Encoder model: {nlcp_config.encoder_model_name}")

    # Load data
    data = load_gsm8k_batch(config, nlcp_config)
    encoder_hidden_states = data["encoder_hidden_states"]

    # Run all tests
    all_results = {}

    try:
        # Test basic extractors
        if not args.skip_basic:
            all_results["basic_extractors"] = test_basic_training_extractors(
                nlcp_config, encoder_hidden_states
            )

        # Test advanced extractors
        if not args.skip_advanced:
            all_results["advanced_extractors"] = test_advanced_causal_extractors(
                nlcp_config, encoder_hidden_states
            )

        # Test inference generator
        if not args.skip_inference:
            all_results["inference_generator"] = test_inference_generator(
                nlcp_config, encoder_hidden_states
            )

        # Print summary
        print_test_summary(all_results)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
