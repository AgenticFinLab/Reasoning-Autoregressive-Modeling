#!/usr/bin/env python3
"""
Text-VAR Demo Script

Demonstrates the refactored Text VQVAE architecture that mirrors VAR-main.
Tests the full pipeline: embed → encode → quantize → decode.

Structure mirrors:
    VAR-main/models/basic_vae.py  →  ram/models/basic_vae.py
    VAR-main/models/quant.py      →  ram/models/quant.py
    VAR-main/models/vqvae.py      →  ram/models/vqvae.py

For real data, use lmbase:
    from lmbase.dataset.registry import get
    config = {"data_name": "math", "data_path": "./data/math"}
    dataset = get(config, split="train")

Usage:
    python examples/text_var/demo.py
    python examples/text_var/demo.py --test model
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from ram.models import VQVAE, Encoder, Decoder, VectorQuantizer2
from ram.utils import load_config, set_seed, count_parameters
from ram.utils.helpers import get_device


def test_encoder():
    """Test Encoder (mirrors VAR's basic_vae.py Encoder)."""
    print("\n" + "="*60)
    print("Testing Encoder (mirrors basic_vae.py)")
    print("="*60)
    
    # Create encoder
    encoder = Encoder(
        ch=64,
        ch_mult=(1, 2, 4),        # Downsample ratio = 2^2 = 4
        num_res_blocks=1,
        in_channels=256,          # embed_dim
        z_channels=16,            # Cvae
    )
    
    # Test input: (B, embed_dim, L)
    B, L, D = 2, 128, 256
    x = torch.randn(B, D, L)
    
    print(f"Input shape: {x.shape}")
    print(f"Downsample ratio: {encoder.downsample_ratio}")
    
    # Forward pass
    f = encoder(x)
    expected_L = L // encoder.downsample_ratio
    
    print(f"Output shape: {f.shape} (expected: B={B}, Cvae=16, L={expected_L})")
    print(f"Encoder parameters: {count_parameters(encoder):,}")
    print("✓ Encoder test passed!")
    
    return encoder


def test_decoder():
    """Test Decoder (mirrors VAR's basic_vae.py Decoder)."""
    print("\n" + "="*60)
    print("Testing Decoder (mirrors basic_vae.py)")
    print("="*60)
    
    # Create decoder
    decoder = Decoder(
        ch=64,
        ch_mult=(1, 2, 4),
        num_res_blocks=1,
        z_channels=16,
        out_channels=256,
        vocab_size=10000,
    )
    
    # Test input: (B, Cvae, L_latent)
    B, L_latent = 2, 32
    z = torch.randn(B, 16, L_latent)
    
    print(f"Input shape: {z.shape}")
    
    # Forward pass
    logits = decoder(z)
    
    print(f"Output shape: {logits.shape}")
    print(f"Decoder parameters: {count_parameters(decoder):,}")
    print("✓ Decoder test passed!")
    
    return decoder


def test_quantizer():
    """Test VectorQuantizer2 (mirrors VAR's quant.py)."""
    print("\n" + "="*60)
    print("Testing VectorQuantizer2 (mirrors quant.py)")
    print("="*60)
    
    # Create quantizer
    quantizer = VectorQuantizer2(
        vocab_size=512,
        Cvae=16,
        v_patch_lens=(1, 2, 4, 8, 16, 32),  # Multi-scale lengths
        beta=0.25,
        quant_resi=0.5,
        share_quant_resi=4,
    )
    
    # Test input: (B, Cvae, L_latent)
    B, L_latent = 2, 32
    f = torch.randn(B, 16, L_latent)
    
    print(f"Input shape: {f.shape}")
    print(f"Multi-scale patch lengths: {quantizer.v_patch_lens}")
    
    # Forward pass (multi-scale residual quantization)
    f_hat, usages, vq_loss = quantizer(f, ret_usages=True)
    
    print(f"Output f_hat shape: {f_hat.shape}")
    print(f"VQ Loss: {vq_loss.item():.4f}")
    print(f"Codebook usage per scale: {usages}")
    
    # Test f_to_idxBl_or_fhat
    idx_list = quantizer.f_to_idxBl_or_fhat(f, to_fhat=False)
    print(f"Index shapes per scale: {[idx.shape for idx in idx_list]}")
    
    print(f"Quantizer parameters: {count_parameters(quantizer):,}")
    print("✓ Quantizer test passed!")
    
    return quantizer


def test_vqvae():
    """Test complete VQVAE (mirrors VAR's vqvae.py)."""
    print("\n" + "="*60)
    print("Testing VQVAE (mirrors vqvae.py)")
    print("="*60)
    
    # Create VQVAE with small config
    model = VQVAE(
        vocab_size=10000,
        embed_dim=256,
        z_channels=16,
        ch=64,
        ch_mult=(1, 2, 4),         # Downsample = 4
        num_res_blocks=1,
        beta=0.25,
        v_patch_lens=(1, 2, 4, 8, 16, 32),
    )
    
    # Test input: (B, L) token indices
    B, L = 2, 128
    inp = torch.randint(0, 10000, (B, L))
    
    print(f"Input shape: {inp.shape}")
    print(f"Downsample ratio: {model.downsample_ratio}")
    print(f"Total parameters: {count_parameters(model, trainable_only=False):,}")
    print(f"Trainable parameters: {count_parameters(model, trainable_only=True):,}")
    
    # Forward pass
    model.train()
    logits, usages, vq_loss = model(inp, ret_usages=True)
    
    print(f"\nOutput logits shape: {logits.shape}")
    print(f"VQ Loss: {vq_loss.item():.4f}")
    print(f"Codebook usage: {usages}")
    
    # Test compute_loss
    loss_dict = model.compute_loss(inp)
    print(f"\nTotal loss: {loss_dict['loss'].item():.4f}")
    print(f"Recon loss: {loss_dict['recon_loss'].item():.4f}")
    print(f"VQ loss: {loss_dict['vq_loss'].item():.4f}")
    
    # Backward pass
    loss_dict['loss'].backward()
    print("\n✓ Backward pass successful!")
    
    # Test encode/decode
    model.eval()
    with torch.no_grad():
        # Encode to indices
        idx_list = model.inp_to_idxBl(inp)
        print(f"\nEncoded indices shapes: {[idx.shape for idx in idx_list]}")
        
        # Decode from indices
        recon_logits = model.idxBl_to_logits(idx_list, same_shape=True, last_one=True)
        print(f"Reconstructed logits shape: {recon_logits.shape}")
    
    print("\n✓ VQVAE test passed!")
    
    return model


def test_training_loop():
    """Test a simple training loop with synthetic data."""
    print("\n" + "="*60)
    print("Testing Training Loop")
    print("="*60)
    
    # Create model
    device = get_device()
    print(f"Using device: {device}")
    
    model = VQVAE(
        vocab_size=10000,
        embed_dim=256,
        z_channels=16,
        ch=64,
        ch_mult=(1, 2, 4),
        num_res_blocks=1,
        v_patch_lens=(1, 2, 4, 8, 16, 32),
    ).to(device)
    
    # Synthetic data
    num_samples, L = 32, 128
    torch.manual_seed(42)
    input_ids = torch.randint(0, 10000, (num_samples, L))
    
    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    batch_size = 4
    
    print(f"\nRunning 3 training steps (batch_size={batch_size})...")
    for step in range(3):
        start_idx = (step * batch_size) % num_samples
        batch = input_ids[start_idx:start_idx + batch_size].to(device)
        
        loss_dict = model.compute_loss(batch)
        
        optimizer.zero_grad()
        loss_dict['loss'].backward()
        optimizer.step()
        
        print(f"  Step {step+1}: loss={loss_dict['loss'].item():.4f}, "
              f"vq_loss={loss_dict['vq_loss'].item():.4f}, "
              f"recon_loss={loss_dict['recon_loss'].item():.4f}")
    
    print("\n✓ Training loop test passed!")


def main():
    parser = argparse.ArgumentParser(description="Text-VAR Demo")
    parser.add_argument('--test', type=str, default='all', 
                       choices=['encoder', 'decoder', 'quantizer', 'vqvae', 'train', 'all'],
                       help="Which test to run")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    print("="*60)
    print("Text-VAR Demo: VQVAE mirroring VAR-main structure")
    print("="*60)
    
    if args.test == 'all':
        test_encoder()
        test_decoder()
        test_quantizer()
        test_vqvae()
        test_training_loop()
    elif args.test == 'encoder':
        test_encoder()
    elif args.test == 'decoder':
        test_decoder()
    elif args.test == 'quantizer':
        test_quantizer()
    elif args.test == 'vqvae':
        test_vqvae()
    elif args.test == 'train':
        test_training_loop()
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
