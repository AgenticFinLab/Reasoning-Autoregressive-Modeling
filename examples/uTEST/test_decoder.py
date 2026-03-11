"""Unit test for TextDecoder (Encoder -> Decoder pipeline).

Usage:
    python examples/uTEST/test_decoder.py -c configs/uTEST/decoder.yml

Tests:
    1. Encoder -> Decoder forward pass
    2. Output shape: [B, L, vocab_size]
    3. Reconstruction loss computation
"""

import argparse
import torch
import torch.nn.functional as F
from lmbase.dataset import registry

from ram.models.encoder import build_encoder
from ram.models.decoder import build_decoder
from ram.utils import load_config


def test_decoder(config: dict):
    """Comprehensive test for Encoder -> Decoder pipeline.

    Pipeline:
        texts -> Encoder -> [B, L, D] -> Decoder -> [B, L, vocab_size]
    """
    enc_cfg = config["model"]["encoder"]
    dec_cfg = config["model"]["decoder"]
    data_cfg = config["data"]
    train_cfg = config["train"]

    B = train_cfg["batch_size"]
    L = enc_cfg["max_length"]

    # Load data from lmbase
    dataset = registry.get(data_cfg, split=data_cfg["split"])
    texts = []
    for i in range(min(B, len(dataset))):
        sample = dataset[i]
        if "question" in sample:
            texts.append(sample["question"])
        elif "problem" in sample:
            texts.append(sample["problem"])
        else:
            texts.append(str(sample))

    print(f"Dataset: {data_cfg['data_name']}, {len(dataset)} samples")
    print(f"Batch: {B} texts, max_length={L}")
    print(f"Sample: {texts[0][:60]}...")
    print()

    # =================================================================
    # 1. Build encoder
    # =================================================================
    print("[1] Build Encoder")
    encoder = build_encoder(enc_cfg)
    print(f"    hidden_dim: {encoder.hidden_dim}")
    print(f"    output_dim: {encoder.output_dim}")
    print("    PASSED")

    # =================================================================
    # 2. Build decoder
    # =================================================================
    print("[2] Build Decoder")
    decoder = build_decoder(dec_cfg, input_dim=encoder.output_dim)
    print(f"    hidden_dim: {decoder.hidden_dim}")
    print(f"    vocab_size: {decoder.vocab_size}")
    print(f"    input_proj: {decoder.input_proj is not None}")
    print("    PASSED")

    # =================================================================
    # 3. Forward: Encoder -> Decoder
    # =================================================================
    print("[3] Forward: Encoder -> Decoder")

    # Encoder forward
    with torch.no_grad():
        hidden = encoder(inputs=texts)
    print(
        f"    Encoder output: [{hidden.shape[0]}, {hidden.shape[1]}, {hidden.shape[2]}]"
    )
    assert hidden.shape == (B, L, encoder.output_dim)

    # Decoder forward
    with torch.no_grad():
        logits = decoder(hidden)
    print(
        f"    Decoder output: [{logits.shape[0]}, {logits.shape[1]}, {logits.shape[2]}]"
    )
    assert logits.shape == (B, L, decoder.vocab_size)
    print("    PASSED")

    # =================================================================
    # 4. Reconstruction Loss
    # =================================================================
    print("[4] Reconstruction Loss")

    # Get target token ids from encoder's tokenizer
    tokens = encoder.tokenize(texts)
    target_ids = tokens["input_ids"]  # [B, L]

    # Compute cross-entropy loss
    loss = F.cross_entropy(
        logits.view(-1, decoder.vocab_size),
        target_ids.view(-1),
        ignore_index=encoder.tokenizer.pad_token_id or 0,
    )
    print(f"    Target: [{target_ids.shape[0]}, {target_ids.shape[1]}]")
    print(f"    Loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    print("    PASSED")

    # =================================================================
    # 5. Text Reconstruction
    # =================================================================
    print("[5] Text Reconstruction")

    # logits [B, L, vocab_size] -> argmax -> pred_ids [B, L]
    pred_ids = logits.argmax(dim=-1)  # [B, L]
    print(f"    Pred IDs: [{pred_ids.shape[0]}, {pred_ids.shape[1]}]")

    # Decode using decoder's tokenizer (GPT2)
    from transformers import AutoTokenizer

    dec_tokenizer = AutoTokenizer.from_pretrained(decoder.model_name)

    # Reconstruct first sample
    pred_text = dec_tokenizer.decode(pred_ids[0], skip_special_tokens=True)
    orig_text = texts[0]

    print(f"    Original: {orig_text[:60]}...")
    print(f"    Reconstructed: {pred_text[:60]}...")
    print("    (Note: reconstruction quality depends on training)")
    print("    PASSED")

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unit test for TextDecoder")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., configs/uTEST/decoder.yml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Config: {args.config}")
    print("=" * 60)
    test_decoder(config)
