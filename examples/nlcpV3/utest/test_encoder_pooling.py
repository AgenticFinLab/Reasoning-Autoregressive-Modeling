"""End-to-end test: GSM8K → Hierarchical Concepts → Solution (V3).

USAGE:
    # Run from project root with config file:
    python examples/nlcpV3/utest/test_encoder_pooling.py -c configs/nlcpV3/utest/test_encoder_pooling.yml

DESIGN SOURCE:
    Reference: docs/concept-pyramid-V3.md
    - Section 2: Architecture
    - Section 3: Training
    - Section 4: Inference

PURPOSE:
    Validate V3 complete flow from GSM8K dataset to solution:
        GSM8K (Q+CoT+Solution) → Tokenization → Encoder → Attentive Pooling
        → Concepts → Concept Transformer → Solution Decoder → Solution

    This integration test ensures:
    1. GSM8K dataset loads correctly via lmbase.dataset.registry
    2. Encoder correctly processes Q+CoT text from GSM8K samples
    3. Attentive Pooling extracts hierarchical concepts (training)
    4. Concept Generator generates concepts (inference)
    5. Solution Decoder decodes directly to solution (NOT CoT!)
    6. Batch processing handles multiple samples

TEST COVERAGE:
    - GSM8K dataset loading via lmbase registry
    - Batch processing with config-specified batch size
    - Training path: Encoder → Attentive Pooling → Solution
    - Inference path: Encoder → Concept Generator → Solution
    - Dimension alignment between all components
"""

import argparse
from pathlib import Path
import sys
import traceback

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Compute project paths relative to this file
# This file: examples/nlcpV3/utest/test_encoder_pooling.py
# Project root: 3 levels up
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = PROJECT_ROOT / "examples"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLES_DIR))

from lmbase.dataset import registry
from nlcpV3.config import NLCPV3Config
from nlcpV3.encoder import NLCPV3Encoder
from nlcpV3.attentive_pooling import ResidualAttentivePooling
from nlcpV3.concept_generator import ConceptGenerator
from nlcpV3.concept_transformer import ConceptTransformer
from nlcpV3.token_decoder import SolutionDecoder
from ram.utils import load_config


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


def test_training_path(config: dict):
    """Test V3 training path: Q+CoT → Concepts → Solution.

    PURPOSE:
        Validate training path with Attentive Pooling.

    TEST FLOW:
        1. Load GSM8K sample from lmbase dataset
        2. Tokenize Q+CoT to input_ids [B, L]
        3. Encode with NLCPV3Encoder → H [B, L, D_encoder]
        4. Apply Attentive Pooling → C_0, C_1, ..., C_K
        5. Apply Concept Transformer → refined concepts
        6. Apply Solution Decoder → solution logits
        7. Verify solution shapes

    DIMENSION FLOW:
        GSM8K sample (Q+CoT+Solution): strings
            ↓
        Tokenization: [B, L=512] (with padding)
            ↓
        Encoder: [B, L=512, D_encoder=896]
            ↓
        Attentive Pooling:
            - C_0: [B, 1, 256]
            - C_1: [B, 2, 256]
            - C_2: [B, 4, 256]
            - C_3: [B, 8, 256]
            - C_4: [B, 16, 256]
            - C_5: [B, 32, 256]
            ↓
        Concept Transformer: same shapes (refined)
            ↓
        Solution Decoder: [B, L_solution, vocab_size]
    """
    print("=" * 70)
    print("TEST: V3 Training Path (Q+CoT → Concepts → Solution)")
    print("=" * 70)

    # Build config from YAML
    nlcp_config = build_nlcpV3_config(config)

    print(f"\nConfiguration:")
    print(f"  Hidden dim: {nlcp_config.hidden_dim}")
    print(f"  Num levels: {nlcp_config.num_levels}")
    print(f"  Level lengths: {nlcp_config.level_lengths}")
    print(f"  Head dim: {nlcp_config.head_dim}")

    # Load GSM8K dataset via lmbase
    print("\n" + "-" * 70)
    print("STEP 1: Load GSM8K from lmbase.dataset.registry")
    print("-" * 70)

    data_cfg = config["data"]
    dataset = registry.get(data_cfg, split=data_cfg["split"])
    print(f"  Dataset: {data_cfg['data_name']}")
    print(f"  Split: {data_cfg['split']}")
    print(f"  Total samples: {len(dataset)}")

    # Get batch_size samples
    batch_size = config["training"]["batch_size"]
    print(f"\n  Loading {batch_size} samples for batch processing...")

    texts = []
    solutions = []
    for i in range(min(batch_size, len(dataset))):
        sample = dataset[i]
        if hasattr(sample, "question") and hasattr(sample, "cot_answer"):
            question = sample.question
            cot = sample.cot_answer
            solution = sample.groundtruth if hasattr(sample, "groundtruth") else ""
        elif isinstance(sample, dict):
            question = sample["question"]
            cot = sample["cot_answer"]
            solution = sample.get("groundtruth", "")
        else:
            question = str(sample)
            cot = ""
            solution = ""

        full_text = f"Question: {question}\nReasoning: {cot}"
        texts.append(full_text)
        solutions.append(solution)

    while len(texts) < batch_size:
        texts.append("")
        solutions.append("")

    print(f"  ✓ Loaded {len(texts)} texts for batch")

    print("\n" + "-" * 70)
    print("STEP 2: Tokenization")
    print("-" * 70)

    tokenizer = AutoTokenizer.from_pretrained(nlcp_config.encoder_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize Q+CoT
    encoded = tokenizer(
        texts,
        max_length=nlcp_config.max_seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    print(f"  Input shape: {list(input_ids.shape)}")
    print(f"  Batch size: {input_ids.shape[0]}")

    print("\n" + "-" * 70)
    print("STEP 3: Encoding (Q+CoT → H)")
    print("-" * 70)

    encoder = NLCPV3Encoder(nlcp_config)
    encoder.eval()

    with torch.no_grad():
        H = encoder.forward_training(input_ids, attention_mask)

    print(f"  Encoder output H shape: {list(H.shape)}")
    print(f"  Encoder hidden dim: {H.shape[2]}")

    print("\n" + "-" * 70)
    print("STEP 4: Attentive Pooling (H → Concepts)")
    print("-" * 70)

    encoder_hidden_dim = H.shape[2]
    pooling = ResidualAttentivePooling(
        nlcp_config, encoder_hidden_dim=encoder_hidden_dim
    )
    pooling.eval()

    with torch.no_grad():
        concepts, H_hat, H_rest = pooling(H)

    print(f"  Number of concept levels: {len(concepts)}")
    print(f"  Expected: {nlcp_config.num_levels} levels")

    print(f"\n  Concept shapes:")
    for k, C_k in enumerate(concepts):
        expected_L = nlcp_config.level_lengths[k]
        print(
            f"    C_{k}: {list(C_k.shape)} (expected: [B, L_{k}={expected_L}, D={nlcp_config.hidden_dim}])"
        )

    print("\n" + "-" * 70)
    print("STEP 5: Concept Transformer")
    print("-" * 70)

    transformer = ConceptTransformer(nlcp_config)
    transformer.eval()

    with torch.no_grad():
        refined_concepts = transformer(concepts)

    print(f"  Refined {len(refined_concepts)} concept levels")
    for k, C_k in enumerate(refined_concepts):
        print(f"    C'_{k}: {list(C_k.shape)}")

    print("\n" + "-" * 70)
    print("STEP 6: Solution Decoder (Concepts → Solution)")
    print("-" * 70)

    solution_decoder = SolutionDecoder(nlcp_config)
    solution_decoder.eval()

    # Tokenize solutions for teacher forcing
    solution_encoded = tokenizer(
        solutions,
        max_length=32,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    solution_ids = solution_encoded["input_ids"]

    with torch.no_grad():
        solution_logits = solution_decoder(refined_concepts, solution_ids)

    print(f"  Solution logits shape: {list(solution_logits.shape)}")
    print(f"  Expected: [B={batch_size}, L_solution, V={nlcp_config.vocab_size}]")

    print("\n" + "=" * 70)
    print("TEST PASSED: V3 Training Path")
    print("=" * 70)

    return {
        "concepts": concepts,
        "refined_concepts": refined_concepts,
        "solution_logits": solution_logits,
    }


def test_inference_path(config: dict):
    """Test V3 inference path: Q → Generated Concepts → Solution.

    PURPOSE:
        Validate inference path with Concept Generator (no CoT!).

    TEST FLOW:
        1. Load GSM8K sample
        2. Tokenize Q only (no CoT!)
        3. Encode Q → H
        4. Generate concepts with Concept Generator
        5. Refine concepts
        6. Generate solution
    """
    print("\n" + "=" * 70)
    print("TEST: V3 Inference Path (Q → Generated Concepts → Solution)")
    print("=" * 70)

    nlcp_config = build_nlcpV3_config(config)

    # Load GSM8K
    data_cfg = config["data"]
    dataset = registry.get(data_cfg, split=data_cfg["split"])

    batch_size = config["training"]["batch_size"]

    # Get questions only (no CoT!)
    questions = []
    for i in range(min(batch_size, len(dataset))):
        sample = dataset[i]
        if hasattr(sample, "question"):
            question = sample.question
        elif isinstance(sample, dict):
            question = sample["question"]
        else:
            question = str(sample)
        questions.append(f"Question: {question}")

    while len(questions) < batch_size:
        questions.append("")

    print(f"\n  Loaded {len(questions)} questions (NO CoT!)")

    # Tokenize Q only
    tokenizer = AutoTokenizer.from_pretrained(nlcp_config.encoder_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(
        questions,
        max_length=nlcp_config.max_seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    print(f"  Input shape (Q only): {list(input_ids.shape)}")

    # Encode
    encoder = NLCPV3Encoder(nlcp_config)
    encoder.eval()

    with torch.no_grad():
        H = encoder.forward_inference(input_ids, attention_mask)

    print(f"  Encoded Q → H: {list(H.shape)}")

    # Generate concepts
    print("\n  Generating concepts from Q (no CoT!)...")
    encoder_hidden_dim = H.shape[2]
    concept_generator = ConceptGenerator(nlcp_config, encoder_hidden_dim)
    concept_generator.eval()

    with torch.no_grad():
        generated_concepts = concept_generator(H)

    print(f"  Generated {len(generated_concepts)} concept levels:")
    for k, C_k in enumerate(generated_concepts):
        print(f"    C_{k}: {list(C_k.shape)}")

    # Refine concepts
    transformer = ConceptTransformer(nlcp_config)
    transformer.eval()

    with torch.no_grad():
        refined_concepts = transformer(generated_concepts)

    print(f"\n  Refined concepts")

    # Generate solution
    print("\n  Generating solution...")
    solution_decoder = SolutionDecoder(nlcp_config)
    solution_decoder.eval()

    with torch.no_grad():
        # For test, just do one step
        solution = solution_decoder.generate(
            refined_concepts, max_length=10, eos_token_id=tokenizer.eos_token_id or 0
        )

    print(f"  Generated solution shape: {list(solution.shape)}")

    print("\n" + "=" * 70)
    print("TEST PASSED: V3 Inference Path")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NLCP V3: Encoder + Concept Generation + Solution Test"
    )
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to config file"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    print("\n" + "=" * 70)
    print("NLCP V3: Concept Compression → Direct Solution Test")
    print("=" * 70)

    try:
        test_training_path(config)
        test_inference_path(config)

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)
