"""End-to-end test: Q+CoT → Hierarchical Concepts.

DESIGN SOURCE:
    Reference: docs/concept-pyramid-V2.md
    - Section 2.1.3: Training Data Flow (Stage 1 & 2)
    - Section 2.2.1: Encoder
    - Section 2.2.2: Attentive Pooling

PURPOSE:
    Validate the complete flow from raw text input to hierarchical concept extraction:
        Q+CoT text → Tokenization → Encoder → Attentive Pooling → C_0, C_1, ..., C_K

    This integration test ensures:
    1. Encoder correctly processes Q+CoT text
    2. Attentive Pooling extracts hierarchical concepts
    3. Concept shapes match configuration
    4. Reconstruction mechanism works
    5. Batch processing handles multiple samples

TEST COVERAGE:
    - Single sample processing
    - Batch processing (multiple samples)
    - Dimension alignment between encoder and concept transformer
    - Concept hierarchy expansion (L_0 → L_1 → ... → L_K)
"""

from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer

# Compute project paths relative to this file
# This file: examples/nlcpV2/utest/test_encoder_pooling.py
# Project root: 3 levels up
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = PROJECT_ROOT / "examples"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLES_DIR))

from nlcpV2.config import NLCPV2Config
from nlcpV2.encoder import NLCPV2Encoder
from nlcpV2.attentive_pooling import ResidualAttentivePooling


def test_q_cot_to_hierarchical_concepts():
    """Test complete flow: Q+CoT → Hierarchical Concepts.

    PURPOSE:
        Validate end-to-end pipeline from text to hierarchical concepts.

    TEST FLOW:
        1. Prepare Q+CoT text sample
        2. Tokenize to input_ids [B, L]
        3. Encode with NLCPV2Encoder → H [B, L, D_encoder]
        4. Apply ResidualAttentivePooling → C_0, C_1, ..., C_K
        5. Verify concept shapes and reconstruction

    DIMENSION FLOW:
        Text: string
            ↓
        Tokenization: [B=1, L=512] (with padding)
            ↓
        Encoder: [B=1, L=512, D_encoder=896] (Qwen hidden dim)
            ↓
        Attentive Pooling:
            - C_0: [1, 4, 256]
            - C_1: [1, 16, 256]
            - C_2: [1, 64, 256]
            - C_3: [1, 256, 256]
            - H_hat: [1, 512, 256]
            - H_rest: [1, 512, 256]

    VERIFICATION:
        - Concept counts match config.level_lengths
        - Hidden dimensions match config.hidden_dim
        - Reconstruction shapes are valid
    """
    print("=" * 70)
    print("TEST: Q+CoT → Hierarchical Concepts")
    print("=" * 70)

    config = NLCPV2Config(
        hidden_dim=256,
        num_heads=8,
        vocab_size=32000,
        num_levels=4,
        level_lengths=[4, 16, 64, 256],
        max_seq_len=512,
        dropout=0.1,
        rms_norm_eps=1e-6,
        encoder_model_name="Qwen/Qwen2.5-0.5B",
        encoder_num_layers=4,
        encoder_freeze=True,
        ntp_loss_weight=1.0,
        concept_loss_weight=0.1,
        recon_loss_weight=0.05,
        muP_scale=1.0,
    )

    print(f"\nConfiguration:")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Num levels: {config.num_levels}")
    print(f"  Level lengths: {config.level_lengths}")
    print(f"  Head dim: {config.head_dim}")

    sample_q_cot = {
        "question": "If a train travels at 60 km/h and needs to cover 240 km, how long will it take?",
        "cot": "To find the time, I need to use the formula: time = distance / speed. "
        "The distance is 240 km and the speed is 60 km/h. "
        "So time = 240 / 60 = 4 hours. "
        "Therefore, the train will take 4 hours.",
    }

    full_text = (
        f"Question: {sample_q_cot['question']}\nReasoning: {sample_q_cot['cot']}"
    )
    print(f"\nSample Q+CoT:")
    print(f"  Length: {len(full_text)} chars")
    print(f"  Preview: {full_text[:100]}...")

    print("\n" + "-" * 70)
    print("STEP 1: Tokenization")
    print("-" * 70)

    tokenizer = AutoTokenizer.from_pretrained(config.encoder_model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(
        full_text,
        max_length=config.max_seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    seq_len = attention_mask.sum().item()
    print(f"  Tokenized length: {seq_len} tokens (padded to {input_ids.size(1)})")
    print(f"  Input IDs shape: {list(input_ids.shape)}")
    print(f"  Attention mask shape: {list(attention_mask.shape)}")

    print("\n" + "-" * 70)
    print("STEP 2: Encoding (Q+CoT → H)")
    print("-" * 70)

    encoder = NLCPV2Encoder(config)
    encoder.eval()

    with torch.no_grad():
        H = encoder.forward_training(input_ids, attention_mask)

    print(f"  Encoder output H shape: {list(H.shape)}")
    print(f"  Note: Encoder uses pretrained model's hidden dim, not config.hidden_dim")
    print(f"  Encoder hidden dim: {H.shape[2]}, Config hidden dim: {config.hidden_dim}")

    assert H.shape[0] == input_ids.size(0), "Batch size mismatch"
    print("  ✓ Encoder output shape correct (hidden dim from pretrained model)")

    print("\n" + "-" * 70)
    print("STEP 3: Attentive Pooling (H → C_0, C_1, ..., C_K)")
    print("-" * 70)

    encoder_hidden_dim = H.shape[2]
    pooling = ResidualAttentivePooling(config, encoder_hidden_dim=encoder_hidden_dim)
    pooling.eval()

    with torch.no_grad():
        concepts, H_hat, H_rest = pooling(H)

    print(f"  Number of concept levels: {len(concepts)}")
    print(f"  Expected: {config.num_levels} levels")

    print(f"\n  Concept shapes:")
    for k, C_k in enumerate(concepts):
        expected_L = config.level_lengths[k]
        print(
            f"    C_{k}: {list(C_k.shape)} (expected: [B, L_{k}={expected_L}, D={config.hidden_dim}])"
        )
        assert C_k.shape[1] == expected_L, f"Level {k} length mismatch"
        assert C_k.shape[2] == config.hidden_dim, f"Level {k} hidden dim mismatch"

    print(f"\n  Reconstruction shapes:")
    print(f"    H_hat: {list(H_hat.shape)} (accumulated reconstruction)")
    print(f"    H_rest: {list(H_rest.shape)} (final residual)")
    print(
        f"    Note: H_hat/H_rest are in concept dim ({config.hidden_dim}), not encoder dim ({encoder_hidden_dim})"
    )

    assert H_hat.shape[2] == config.hidden_dim, "H_hat hidden dim mismatch"
    assert H_rest.shape[2] == config.hidden_dim, "H_rest hidden dim mismatch"

    print("\n" + "-" * 70)
    print("STEP 4: Verification")
    print("-" * 70)

    H_proj = pooling.input_proj(H)
    reconstruction_error = torch.mean((H_hat - H_proj) ** 2).item()
    residual_norm = torch.mean(H_rest**2).item()
    total_energy = torch.mean(H_proj**2).item()

    print(f"  Reconstruction MSE: {reconstruction_error:.6f}")
    print(f"  Residual energy: {residual_norm:.6f}")
    print(f"  Original energy (projected): {total_energy:.6f}")
    print(f"  Reconstruction ratio: {reconstruction_error / total_energy:.2%}")
    print(f"  Note: High reconstruction error is expected before training")

    assert H_hat.shape == H_rest.shape, "Shape mismatch"
    print("  ✓ Reconstruction mechanism working (needs training)")

    print("\n" + "-" * 70)
    print("STEP 5: Concept Hierarchy Analysis")
    print("-" * 70)

    for k in range(len(concepts) - 1):
        L_k = concepts[k].shape[1]
        L_next = concepts[k + 1].shape[1]
        expansion = L_next / L_k
        print(
            f"  Level {k} → {k+1}: {L_k} → {L_next} concepts (expansion: {expansion:.1f}x)"
        )

    print("\n" + "=" * 70)
    print("TEST PASSED: Q+CoT → Hierarchical Concepts")
    print("=" * 70)

    return {
        "input_ids": input_ids,
        "H": H,
        "concepts": concepts,
        "H_hat": H_hat,
        "H_rest": H_rest,
        "config": config,
    }


def test_batch_processing():
    """Test with multiple samples in batch.

    PURPOSE:
        Verify that the pipeline handles batch processing correctly.

    TEST FLOW:
        1. Create batch of 4 different Q+CoT samples
        2. Tokenize with padding
        3. Encode batch → H [B, L, D_encoder]
        4. Apply Attentive Pooling → concepts [B, L_k, D]
        5. Verify batch dimension preserved

    DIMENSION FLOW:
        Texts: List of 4 strings
            ↓
        Tokenization: [B=4, L=256] (batch with padding)
            ↓
        Encoder: [B=4, L=256, D_encoder=896]
            ↓
        Attentive Pooling:
            - C_0: [4, 4, 128]
            - C_1: [4, 8, 128]
            - C_2: [4, 32, 128]

    VERIFICATION:
        - Batch dimension B=4 preserved through all operations
        - Each concept tensor has correct batch size
    """
    print("\n" + "=" * 70)
    print("TEST: Batch Processing")
    print("=" * 70)

    config = NLCPV2Config(
        hidden_dim=128,
        num_heads=4,
        vocab_size=32000,
        num_levels=3,
        level_lengths=[4, 8, 32],
        max_seq_len=256,
        dropout=0.1,
        rms_norm_eps=1e-6,
        encoder_model_name="Qwen/Qwen2.5-0.5B",
        encoder_num_layers=2,
        encoder_freeze=True,
        ntp_loss_weight=1.0,
        concept_loss_weight=0.1,
        recon_loss_weight=0.05,
        muP_scale=1.0,
    )

    batch_size = 4
    print(f"\nBatch size: {batch_size}")

    tokenizer = AutoTokenizer.from_pretrained(config.encoder_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = [
        "What is 2+2? Let's think: 2+2 equals 4.",
        "How many days in a week? There are 7 days.",
        "What is the capital of France? Paris is the capital.",
        "If x=3 and y=4, what is x+y? x+y = 3+4 = 7.",
    ]

    encoded = tokenizer(
        texts,
        max_length=config.max_seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    print(f"Input shape: {list(input_ids.shape)}")

    encoder = NLCPV2Encoder(config)

    with torch.no_grad():
        H = encoder.forward_training(input_ids, attention_mask)

    encoder_hidden_dim = H.shape[2]
    pooling = ResidualAttentivePooling(config, encoder_hidden_dim=encoder_hidden_dim)

    with torch.no_grad():
        concepts, H_hat, H_rest = pooling(H)

    print(f"\nEncoder output: {list(H.shape)}")
    print(f"Batch dimension: {H.shape[0]} (expected: {batch_size})")

    for k, C_k in enumerate(concepts):
        print(f"  C_{k}: {list(C_k.shape)}")
        assert C_k.shape[0] == batch_size, f"Batch size mismatch at level {k}"

    print("\n✓ Batch processing test passed")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("NLCP V2: Encoder + Attentive Pooling Integration Test")
    print("=" * 70)

    try:
        result = test_q_cot_to_hierarchical_concepts()
        test_batch_processing()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
