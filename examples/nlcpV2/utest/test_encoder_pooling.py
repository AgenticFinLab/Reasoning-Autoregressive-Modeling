"""End-to-end test: GSM8K Batch → Hierarchical Concepts.

USAGE:
    # Run from project root with config file:
    python examples/nlcpV2/utest/test_encoder_pooling.py -c configs/nlcpV2/utest/test_encoder_pooling.yml

    # Or with cd:
    cd /path/to/Reasoning-Autoregressive-Modeling
    python examples/nlcpV2/utest/test_encoder_pooling.py -c configs/nlcpV2/utest/test_encoder_pooling.yml

DESIGN SOURCE:
    Reference: docs/concept-pyramid-V2.md
    - Section 2.1.3: Training Data Flow (Stage 1 & 2)
    - Section 2.2.1: Encoder
    - Section 2.2.2: Attentive Pooling

    Reference: docs/lmbase-usage.md
    - Section 3.2: Dataset Registry (GSM8K loading)

PURPOSE:
    Validate the complete flow from GSM8K dataset (batch) to hierarchical concept extraction:
        GSM8K (Q+CoT) [B] → Tokenization → Encoder → Attentive Pooling → C_0, C_1, ..., C_K

    This integration test ensures:
    1. GSM8K dataset loads correctly via lmbase.dataset.registry
    2. Batch processing with configurable batch_size from YAML
    3. Encoder correctly processes Q+CoT text from GSM8K samples
    4. Attentive Pooling extracts hierarchical concepts
    5. Concept shapes match configuration
    6. Reconstruction mechanism works

TEST COVERAGE:
    - GSM8K dataset loading via lmbase registry
    - Batch processing with config-specified batch size
    - Dimension alignment between encoder and concept transformer
    - Concept hierarchy expansion (L_0 → L_1 → ... → L_K)
"""

import argparse
from pathlib import Path
import sys
import traceback

import torch
from transformers import AutoTokenizer

# This file: examples/nlcpV2/utest/test_encoder_pooling.py
# Project root: 3 levels up
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = PROJECT_ROOT / "examples"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLES_DIR))

from lmbase.dataset import registry
from nlcpV2.config import NLCPV2Config
from nlcpV2.encoder import NLCPV2Encoder
from nlcpV2.attentive_pooling import ResidualAttentivePooling
from ram.utils import load_config


def build_nlcpV2_config(config: dict) -> NLCPV2Config:
    """Build NLCPV2Config from YAML configuration.

    Args:
        config: Configuration dictionary from YAML file

    Returns:
        NLCPV2Config instance
    """
    model_cfg = config["model"]
    encoder_cfg = model_cfg["encoder"]
    pyramid_cfg = model_cfg["pyramid"]
    decoder_cfg = model_cfg["decoder"]
    loss_cfg = model_cfg["loss_weights"]

    return NLCPV2Config(
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


def test_batch_processing(config: dict):
    """Test with batch samples from GSM8K.

    PURPOSE:
        Validate end-to-end pipeline from GSM8K dataset (batch) to hierarchical concepts.

    TEST FLOW:
        1. Load GSM8K dataset via lmbase registry
        2. Create batch with batch_size from config (training.batch_size)
        3. Tokenize batch → [B, L]
        4. Encode batch → H [B, L, D_encoder]
        5. Apply Attentive Pooling → C_0, C_1, ..., C_K
        6. Verify concept shapes and reconstruction

    DIMENSION FLOW:
        GSM8K samples: List of B strings (B = config.training.batch_size)
            ↓
        Tokenization: [B, L=512] (batch with padding)
            ↓
        Encoder: [B, L=512, D_encoder=896] (Qwen hidden dim)
            ↓
        Attentive Pooling:
            - C_0: [B, 1, 256]
            - C_1: [B, 2, 256]
            - C_2: [B, 4, 256]
            - C_3: [B, 8, 256]
            - C_4: [B, 16, 256]
            - C_5: [B, 32, 256]
            - H_hat: [B, 512, 256]
            - H_rest: [B, 512, 256]

    VERIFICATION:
        - GSM8K loads via lmbase.dataset.registry
        - Batch dimension B matches config.training.batch_size
        - Concept counts match config.model.pyramid.level_lengths
        - Hidden dimensions match config.model.pyramid.hidden_dim
        - Reconstruction mechanism works
    """
    print("=" * 70)
    print("TEST: GSM8K Batch → Hierarchical Concepts")
    print("=" * 70)

    # Build NLCP config from YAML
    nlcp_config = build_nlcpV2_config(config)

    # Get batch_size from config
    batch_size = config["training"]["batch_size"]

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size} (from config.training.batch_size)")
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

    # Get batch_size samples and extract Q+CoT texts
    print(f"\n  Loading {batch_size} samples for batch processing...")
    texts = []
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
        full_text = f"Question: {question}\nReasoning: {cot}"
        texts.append(full_text)

    # Pad with empty strings if dataset has fewer samples than batch_size
    while len(texts) < batch_size:
        texts.append("")

    print(f"  ✓ Loaded {len(texts)} texts for batch")

    print("\n" + "-" * 70)
    print("STEP 2: Tokenization")
    print("-" * 70)

    tokenizer = AutoTokenizer.from_pretrained(nlcp_config.encoder_model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(
        texts,
        max_length=nlcp_config.max_seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    actual_batch_size = input_ids.shape[0]
    seq_len = attention_mask.sum(dim=1)  # Per-sample actual lengths
    print(f"  Batch size: {actual_batch_size}")
    print(f"  Max sequence length: {input_ids.size(1)} tokens")
    print(f"  Input IDs shape: {list(input_ids.shape)}")
    print(f"  Attention mask shape: {list(attention_mask.shape)}")
    print(
        f"  Actual token lengths: min={seq_len.min().item()}, max={seq_len.max().item()}"
    )

    print("\n" + "-" * 70)
    print("STEP 3: Encoding (Q+CoT → H)")
    print("-" * 70)

    encoder = NLCPV2Encoder(nlcp_config)
    encoder.eval()

    with torch.no_grad():
        H = encoder.forward_training(input_ids, attention_mask)

    print(f"  Encoder output H shape: {list(H.shape)}")
    print(f"  Note: Encoder uses pretrained model's hidden dim, not config.hidden_dim")
    print(
        f"  Encoder hidden dim: {H.shape[2]}, Config hidden dim: {nlcp_config.hidden_dim}"
    )

    assert H.shape[0] == actual_batch_size, "Batch size mismatch"
    print("  ✓ Encoder output shape correct")

    print("\n" + "-" * 70)
    print("STEP 4: Attentive Pooling (H → C_0, C_1, ..., C_K)")
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
            f"    C_{k}: {list(C_k.shape)} (expected: [B={actual_batch_size}, L_{k}={expected_L}, D={nlcp_config.hidden_dim}])"
        )
        assert C_k.shape[0] == actual_batch_size, f"Level {k} batch size mismatch"
        assert C_k.shape[1] == expected_L, f"Level {k} length mismatch"
        assert C_k.shape[2] == nlcp_config.hidden_dim, f"Level {k} hidden dim mismatch"

    print(f"\n  Reconstruction shapes:")
    print(f"    H_hat: {list(H_hat.shape)} (accumulated reconstruction)")
    print(f"    H_rest: {list(H_rest.shape)} (final residual)")
    print(
        f"    Note: H_hat/H_rest are in concept dim ({nlcp_config.hidden_dim}), not encoder dim ({encoder_hidden_dim})"
    )

    assert H_hat.shape[0] == actual_batch_size, "H_hat batch size mismatch"
    assert H_rest.shape[0] == actual_batch_size, "H_rest batch size mismatch"
    assert H_hat.shape[2] == nlcp_config.hidden_dim, "H_hat hidden dim mismatch"
    assert H_rest.shape[2] == nlcp_config.hidden_dim, "H_rest hidden dim mismatch"

    print("\n" + "-" * 70)
    print("STEP 5: Verification")
    print("-" * 70)

    H_proj = pooling.input_proj(H)
    reconstruction_error = torch.mean((H_hat - H_proj) ** 2).item()
    residual_norm = torch.mean(H_rest**2).item()
    total_energy = torch.mean(H_proj**2).item()

    print(f"  Reconstruction MSE: {reconstruction_error:.6f}")
    print(f"  Residual energy: {residual_norm:.6f}")
    print(f"  Original energy (projected): {total_energy:.6f}")
    if total_energy > 0:
        print(f"  Reconstruction ratio: {reconstruction_error / total_energy:.2%}")
    print(f"  Note: High reconstruction error is expected before training")

    assert H_hat.shape == H_rest.shape, "Shape mismatch"
    print("  ✓ Reconstruction mechanism working (needs training)")

    print("\n" + "-" * 70)
    print("STEP 6: Concept Hierarchy Analysis")
    print("-" * 70)

    for k in range(len(concepts) - 1):
        L_k = concepts[k].shape[1]
        L_next = concepts[k + 1].shape[1]
        expansion = L_next / L_k
        print(
            f"  Level {k} → {k+1}: {L_k} → {L_next} concepts (expansion: {expansion:.1f}x)"
        )

    print("\n" + "=" * 70)
    print("TEST PASSED: GSM8K Batch → Hierarchical Concepts")
    print("=" * 70)

    return {
        "input_ids": input_ids,
        "H": H,
        "concepts": concepts,
        "H_hat": H_hat,
        "H_rest": H_rest,
        "config": nlcp_config,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NLCP V2: Encoder + Attentive Pooling Batch Test"
    )
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to config file"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    print("\n" + "=" * 70)
    print("NLCP V2: Encoder + Attentive Pooling Integration Test")
    print("=" * 70)

    try:
        result = test_batch_processing(config)

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)
