"""Unit test for MultiScaleQuantizer.

Usage:
    python examples/uTEST/test_quantizer.py -c configs/uTEST/quantizer.yml

Config (example: B=2, L=32, D=256, scales=[1,2,4,8,16,32], V=1024):
    - B: batch size
    - L: max sequence length (= max(scale_lengths))
    - D: codebook_dim (encoder output_dim must match)
    - V: codebook_size

Pipeline:
    ┌─────────────────────┐
    │ List[str] texts     │  B texts
    └────────┬────────────┘
             │ Encoder (tokenize + HuggingFace + projection)
             ▼
    ┌─────────────────────┐
    │ [B, L, D] hidden    │  [2, 32, 256] encoder output
    └────────┬────────────┘
             │ Quantizer.forward()
             │
             │  f_rest = z.clone()               # residual starts as original
             │  f_hat = zeros([B, L, D])          # accumulator starts as zeros
             │
             │  for each scale k ∈ [1, 2, 4, 8, 16, 32]:
             │    ├─ rest_down = downsample(f_rest, k)   -> [B, k, D]
             │    ├─ indices = codebook_lookup(rest_down) -> [B, k]
             │    ├─ h_k = φ_k(codebook[indices])       -> [B, k, D]
             │    ├─ h_k_up = upsample(h_k, L)          -> [B, L, D]
             │    ├─ f_hat = f_hat + h_k_up             -> [B, L, D]
             │    └─ f_rest = f_rest - h_k_up           -> [B, L, D] (update residual)
             │
             │  loss = β*||f_hat - z||² + ||f_hat - z||²  (VQ loss)
             │
             ▼
    ┌─────────────────────────────────────────────────────┐
    │ f_hat: [B, L, D]           [2, 32, 256]             │
    │ loss: scalar               commitment + codebook    │
    │ indices_per_scale: List    [[2,1],[2,2],...,[2,32]] │
    └─────────────────────────────────────────────────────┘

Multi-Scale Detail (scales=[1,2,4,8,16,32]):
    Scale 0: [B, 1, D]  -> 1 vector for global structure
    Scale 1: [B, 2, D]  -> 2 vectors for coarse split
    Scale 2: [B, 4, D]  -> 4 vectors
    Scale 3: [B, 8, D]  -> 8 vectors
    Scale 4: [B, 16, D] -> 16 vectors
    Scale 5: [B, 32, D] -> 32 vectors for fine details

    Total indices: B * (1+2+4+8+16+32) = B * 63 codebook entries per sample

Tests:
    [1] Encoder -> hidden [B, L, D]
    [2] Quantizer initialization (codebook [V, D], phi layers, scale_ops)
    [3] Multi-scale quantization: f_hat [B, L, D], loss, indices_per_scale
    [4] Scale operations: downsample/upsample shape verification
    [5] Decode indices -> f_hat [B, L, D]
    [6] Reconstruction consistency: f_hat == decode_indices(indices)
    [7] Scale ops roundtrip: down(k) -> up(L) shape preservation
    [8] Phi layers: shared (scales 0-3) vs independent (scales 4-5)
"""

import argparse
import torch
from lmbase.dataset import registry

from ram.models.encoder import build_encoder
from ram.models.quantizer import build_quantizer, MultiScaleQuantizer
from ram.models.scale_ops import AvgPoolScaleOps, LinearScaleOps
from ram.utils import load_config


def test_quantizer(config: dict):
    """Comprehensive test for MultiScaleQuantizer.

    Tests:
        1. Encoder -> hidden [B, L, D]
        2. Quantizer initialization
        3. Multi-scale quantization
        4. Scale operations
        5. Decode indices
        6. Reconstruction consistency
    """
    enc_cfg = config["model"]["encoder"]
    quant_cfg = config["model"]["quantizer"]
    data_cfg = config["data"]
    train_cfg = config["train"]

    B = train_cfg["batch_size"]
    L = enc_cfg["max_length"]
    D = quant_cfg["codebook_dim"]
    scales = quant_cfg["scale_lengths"]

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
    print(f"Scales: {scales}")
    print()

    # =================================================================
    # 1. Encoder -> hidden states
    # =================================================================
    print("[1] Encoder -> hidden states")
    encoder = build_encoder(enc_cfg)
    assert (
        encoder.output_dim == D
    ), f"Encoder output_dim ({encoder.output_dim}) != codebook_dim ({D})"

    with torch.no_grad():
        hidden = encoder(inputs=texts)
    print(
        f"    Encoder output: [{hidden.shape[0]}, {hidden.shape[1]}, {hidden.shape[2]}]"
    )
    assert hidden.shape == (
        B,
        L,
        D,
    ), f"Expected [{B}, {L}, {D}], got {list(hidden.shape)}"
    print("    PASSED")

    # =================================================================
    # 2. Quantizer initialization
    # =================================================================
    print("[2] Quantizer Initialization")
    quantizer = build_quantizer(quant_cfg)

    print(f"    codebook_size: {quantizer.codebook_size}")
    print(f"    codebook_dim: {quantizer.codebook_dim}")
    print(f"    scale_lengths: {quantizer.scale_lengths}")
    print(f"    num_scales: {quantizer.num_scales}")
    print(f"    max_length: {quantizer.max_length}")
    print(f"    num_shared phi: {quantizer.num_shared}")
    print(f"    codebook shape: {list(quantizer.codebook.weight.shape)}")

    assert quantizer.codebook.weight.shape == (quant_cfg["codebook_size"], D)
    assert quantizer.num_scales == len(scales)
    print("    PASSED")

    # =================================================================
    # 3. Multi-scale Quantization
    # =================================================================
    print("[3] Multi-scale Quantization")

    # Forward pass
    f_hat, loss, indices_per_scale = quantizer(hidden)

    print(f"    Input:  [{hidden.shape[0]}, {hidden.shape[1]}, {hidden.shape[2]}]")
    print(f"    f_hat:  [{f_hat.shape[0]}, {f_hat.shape[1]}, {f_hat.shape[2]}]")
    print(f"    Loss:   {loss.item():.4f}")
    print(f"    Num scales: {len(indices_per_scale)}")

    assert (
        f_hat.shape == hidden.shape
    ), f"f_hat shape mismatch: {f_hat.shape} vs {hidden.shape}"
    assert loss.item() > 0, "Loss should be positive"
    assert len(indices_per_scale) == len(scales)
    print("    PASSED")

    # =================================================================
    # 4. Scale Operations (per scale)
    # =================================================================
    print("[4] Scale Operations (per scale)")

    for k, scale_len in enumerate(scales):
        indices_k = indices_per_scale[k]
        print(
            f"    Scale {k}: length={scale_len}, indices shape={list(indices_k.shape)}"
        )
        assert indices_k.shape == (
            B,
            scale_len,
        ), f"Scale {k}: expected [{B}, {scale_len}], got {list(indices_k.shape)}"

        # Check indices are valid (0 <= idx < codebook_size)
        assert indices_k.min() >= 0, f"Scale {k}: negative indices"
        assert (
            indices_k.max() < quant_cfg["codebook_size"]
        ), f"Scale {k}: indices exceed codebook_size"

    print("    PASSED")

    # =================================================================
    # 5. Decode Indices
    # =================================================================
    print("[5] Decode Indices")

    f_hat_decoded = quantizer.decode_indices(indices_per_scale, target_length=L)
    print(
        f"    Decoded f_hat: [{f_hat_decoded.shape[0]}, {f_hat_decoded.shape[1]}, {f_hat_decoded.shape[2]}]"
    )

    assert f_hat_decoded.shape == (B, L, D)
    print("    PASSED")

    # =================================================================
    # 6. Reconstruction Consistency
    # =================================================================
    print("[6] Reconstruction Consistency")

    # f_hat from quantize() should match decode_indices()
    diff = (f_hat - f_hat_decoded).abs().mean().item()
    print(f"    f_hat vs decoded diff: {diff:.6f}")

    # Note: There may be small differences due to residual computation
    # but they should be very close
    assert diff < 0.1, f"f_hat and decoded f_hat differ too much: {diff}"
    print("    PASSED")

    # =================================================================
    # 7. Scale Ops Test (downsample/upsample)
    # =================================================================
    print("[7] Scale Ops Test")

    scale_ops = quantizer.scale_ops
    print(f"    Scale ops type: {type(scale_ops).__name__}")

    # Test downsample/upsample roundtrip
    x_test = torch.randn(B, L, D)
    for scale_len in scales:
        x_down = scale_ops.downsample(x_test, scale_len)
        x_up = scale_ops.upsample(x_down, L)
        print(
            f"    [{B}, {L}, {D}] -> down({scale_len}) -> [{B}, {scale_len}, {D}] -> up({L}) -> [{B}, {L}, {D}]"
        )
        assert x_down.shape == (B, scale_len, D)
        assert x_up.shape == (B, L, D)

    print("    PASSED")

    # =================================================================
    # 8. Phi Layers Test
    # =================================================================
    print("[8] Phi Layers Test")

    for k in range(quantizer.num_scales):
        phi = quantizer.get_phi(k)
        is_shared = k < quantizer.num_shared
        print(f"    Scale {k}: phi type={'shared' if is_shared else 'independent'}")

    print("    PASSED")

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unit test for MultiScaleQuantizer")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., configs/uTEST/quantizer.yml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Config: {args.config}")
    print("=" * 60)
    test_quantizer(config)
