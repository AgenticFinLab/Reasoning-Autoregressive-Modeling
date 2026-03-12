"""Unit test for TextEncoder.

Usage:
    python examples/uTEST/test_encoder.py -c configs/uTEST/encoder.yml
"""

import argparse
import torch
from lmbase.dataset import registry

from ram.models.encoder import build_encoder
from ram.utils import load_config


def test_encoder(config: dict):
    """Comprehensive test for TextEncoder.

    Tests:
        1. Initialization (tokenizer, encoder, dimensions)
        2. Forward with text inputs: List[str] -> [B, L, D]
        3. Forward with input_ids: [B, L] -> [B, L, D]
        4. Projection layer
        5. Freeze behavior
    """
    enc_cfg = config["model"]["encoder"]
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
    # 1. Build encoder and check initialization
    # =================================================================
    print("[1] Initialization")
    encoder = build_encoder(enc_cfg)

    print(f"    Model: {encoder.model_name}")
    print(f"    Hidden dim: {encoder.hidden_dim}")
    print(f"    Output dim: {encoder.output_dim}")
    print(f"    Max length: {encoder.max_length}")
    assert encoder.tokenizer is not None, "Tokenizer should be loaded"
    assert encoder.encoder is not None, "Encoder should be loaded"
    assert encoder.output_dim == encoder.hidden_dim, "No projection"
    assert encoder.proj is None, "No projection layer"
    print("    PASSED")

    # =================================================================
    # 2. Forward with text inputs
    # =================================================================
    print("[2] Forward with inputs=List[str]")
    with torch.no_grad():
        output = encoder(inputs=texts)
    print(f"    Input: {len(texts)} texts")
    print(f"    Output: [{output.shape[0]}, {output.shape[1]}, {output.shape[2]}]")
    assert output.shape == (B, L, encoder.output_dim)
    print("    PASSED")

    # =================================================================
    # 3. Forward with input_ids
    # =================================================================
    print("[3] Forward with input_ids")
    tokens = encoder.tokenize(texts)
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    with torch.no_grad():
        output2 = encoder(input_ids=input_ids, attention_mask=attention_mask)
    print(f"    Input: input_ids [{input_ids.shape[0]}, {input_ids.shape[1]}]")
    print(f"    Output: [{output2.shape[0]}, {output2.shape[1]}, {output2.shape[2]}]")
    assert output2.shape == (B, L, encoder.output_dim)
    assert torch.allclose(output, output2), "Same output for text vs input_ids"
    print("    PASSED")

    # =================================================================
    # 4. Projection layer
    # =================================================================
    print("[4] Projection layer")
    proj_dim = 256
    enc_cfg["output_dim"] = proj_dim
    encoder_proj = build_encoder(enc_cfg)
    # Restore original config
    enc_cfg["output_dim"] = None

    assert encoder_proj.proj is not None, "Projection layer should exist"
    assert encoder_proj.output_dim == proj_dim
    with torch.no_grad():
        output_proj = encoder_proj(inputs=texts)
    print(f"    Hidden: {encoder_proj.hidden_dim} -> Output: {encoder_proj.output_dim}")
    print(
        f"    Output: [{output_proj.shape[0]}, {output_proj.shape[1]}, {output_proj.shape[2]}]"
    )
    assert output_proj.shape == (B, L, proj_dim)
    print("    PASSED")

    # =================================================================
    # 5. Freeze behavior
    # =================================================================
    print("[5] Freeze behavior")
    enc_cfg["freeze"] = True
    encoder_frozen = build_encoder(enc_cfg)
    frozen_params = sum(p.requires_grad for p in encoder_frozen.encoder.parameters())

    enc_cfg["freeze"] = False
    encoder_trainable = build_encoder(enc_cfg)
    trainable_params = sum(
        p.requires_grad for p in encoder_trainable.encoder.parameters()
    )

    print(f"    Frozen: {frozen_params} trainable params")
    print(f"    Trainable: {trainable_params} trainable params")
    assert frozen_params == 0, "Frozen encoder should have 0 trainable params"
    assert trainable_params > 0, "Trainable encoder should have >0 trainable params"
    print("    PASSED")

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unit test for TextEncoder")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., configs/uTEST/encoder.yml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Config: {args.config}")
    print("=" * 60)
    test_encoder(config)
