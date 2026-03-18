"""Encoder-Quantizer-Decoder with Multi-Scale Quantization (VAR-style).

================================================================================
Purpose: Test VAR-style multi-scale residual quantization for text
================================================================================

This is the unified file for multi-scale quantization experiments.
Different scale configurations are specified in the config file.

================================================================================
Core Concept: VAR Multi-Scale Residual Quantization
================================================================================

Unlike token-level VQ-VAE (direct 1:1 mapping), VAR uses multi-scale hierarchy:

    z [B, L, D]
        ↓
    Scale 0: z → pool(scale_0) → [B, s0, D] → VQ → upsample → [B, L, D] → add to f_hat
    Scale 1: residual → pool(scale_1) → [B, s1, D] → VQ → upsample → [B, L, D] → add
    ...
    Scale k: residual → pool(scale_k) → [B, sk, D] → VQ → upsample → [B, L, D] → add
        ↓
    f_hat [B, L, D] = sum of all upsampled quantized features

Key Insight:
    - Each scale captures different granularity
    - Coarse scales: global structure
    - Fine scales: local details
    - Residual: each scale quantizes what previous scales missed

================================================================================
Scale Configuration Options (set in config file)
================================================================================

Option A - VAR Original (designed for images):
    scale_lengths: [1, 2, 4, 8, 16, 32]
    total_codes: 63
    For L=512: 8:1 compression (TOO AGGRESSIVE for text)

Option B - More Codes (better for text):
    scale_lengths: [32, 64, 128, 256, 512]
    total_codes: 992
    For L=512: 1:2 expansion (preserves more information)

Option C - Custom:
    scale_lengths: [any list of integers]
    Configure in config file as needed

================================================================================
Dimensions
================================================================================
    B = batch_size (e.g., 4)
    L = max_length (e.g., 512)
    D = latent_dim (e.g., 256)
    V = vocab_size (GPT2=50257)
    K = codebook_size (e.g., 4096)

================================================================================
Why This Might Work for Text
================================================================================
    1. Multi-scale captures different semantic levels
    2. Residual quantization progressively refines representation
    3. More codes = less information loss

================================================================================
Why This Might NOT Work for Text
================================================================================
    1. Text has no spatial redundancy (unlike images)
    2. Pooling may lose position-sensitive information
    3. Coarse-to-fine may not match text structure

================================================================================
Usage
================================================================================
    python examples/PreExp/eqd_scales.py -c configs/PreExp/eqd_scales.yml
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from lmbase.dataset import registry
from transformers import AutoTokenizer

from ram.models.encoder import build_encoder
from ram.models.decoder import build_decoder
from ram.models.quantizer import build_quantizer
from ram.utils import (
    load_config,
    setup_environment,
    collate_fn_text,
    decode_logits_to_text,
    find_latest_checkpoint,
    resume_from_checkpoint,
)
from ram.losses import (
    build_loss_from_config,
    validate_loss_config,
)


def train_scales(config: dict):
    """
    Train Encoder-Quantizer-Decoder with Multi-Scale Quantization.

    Training Flow:
        Step 1: texts → tokenize → input_ids [B, L]
        Step 2: input_ids → Encoder → z [B, L, D]
        Step 3: z → Quantizer (multi-scale) → f_hat [B, L, D] + indices + vq_loss
        Step 4: f_hat → Decoder → logits [B, L, V]
        Step 5: loss = recon_loss + λ * vq_loss

    Scale Configuration:
        Set scale_lengths in config file:
        - [1, 2, 4, 8, 16, 32] = 63 codes (VAR original, for images)
        - [32, 64, 128, 256, 512] = 992 codes (better for text)
        - Custom: any list of integers
    """
    # =================================================================
    # Extract config
    # =================================================================
    enc_cfg = config["model"]["encoder"]
    dec_cfg = config["model"]["decoder"]
    quant_cfg = config["model"]["quantizer"]
    latent_dim = config["model"]["latent_dim"]
    data_cfg = config["data"]
    train_cfg = config["train"]
    env_cfg = config["environment"]
    log_cfg = config["logging"]

    # Training hyperparameters
    batch_size = train_cfg["batch_size"]
    learning_rate = train_cfg["learning_rate"]
    weight_decay = train_cfg["weight_decay"]
    num_epochs = train_cfg["num_epochs"]
    warmup_steps = train_cfg["warmup_steps"]
    gradient_clip = train_cfg["gradient_clip"]
    vq_loss_weight = train_cfg["vq_loss_weight"]
    resume = train_cfg["resume"]

    # Logging intervals
    log_interval = log_cfg["log_interval"]
    checkpoint_interval = log_cfg["checkpoint_interval"]

    output_dir = Path(log_cfg["output_dir"])
    checkpoint_dir = Path(log_cfg["checkpoint_dir"])
    log_dir = Path(log_cfg["log_dir"])

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Dimensions
    L = enc_cfg["max_length"]
    scale_lengths = quant_cfg["scale_lengths"]
    total_codes = sum(scale_lengths)

    # Setup environment
    device = setup_environment(env_cfg)

    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Max length: {L}")
    print(f"Scale lengths: {scale_lengths}")
    print(f"Total codes: {total_codes}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"Output dir: {output_dir}")
    print()

    # =================================================================
    # Build models
    # =================================================================
    print("[1] Building Encoder...")
    encoder = build_encoder(enc_cfg)
    encoder = encoder.to(device)
    D = encoder.output_dim
    print(f"    hidden_dim: {encoder.hidden_dim}, output_dim: {D}")

    print("[2] Building Quantizer (MORE SCALES)...")
    quant_config = quant_cfg.copy()
    quant_config["codebook_dim"] = latent_dim
    quantizer = build_quantizer(quant_config)
    quantizer = quantizer.to(device)
    print(f"    codebook_size: {quantizer.codebook_size}")
    print(f"    scale_lengths: {quantizer.scale_lengths}")
    print(f"    total_codes: {sum(quantizer.scale_lengths)}")

    print("[3] Building Decoder...")
    decoder = build_decoder(dec_cfg, input_dim=latent_dim)
    decoder = decoder.to(device)
    V = decoder.vocab_size
    print(f"    hidden_dim: {decoder.hidden_dim}, vocab_size: {V}")

    # Projection layer if encoder output != latent_dim
    # Encoder outputs [B, L, D], quantizer expects [B, L, latent_dim]
    if D != latent_dim:
        print(f"[4] Adding pre-quantization projection: {D} → {latent_dim}")
        pre_quant_proj = nn.Linear(D, latent_dim).to(device)
    else:
        pre_quant_proj = nn.Identity()
    print()

    # Decoder tokenizer
    dec_tokenizer = AutoTokenizer.from_pretrained(decoder.model_name)
    if dec_tokenizer.pad_token is None:
        dec_tokenizer.pad_token = dec_tokenizer.eos_token

    # =================================================================
    # Setup loss function
    # =================================================================
    print("[5] Setting up loss function...")
    loss_cfg = train_cfg["loss"]
    loss_type = loss_cfg["type"]
    print(f"    Loss type: {loss_type}")

    loss_warnings = validate_loss_config(
        config,
        enc_tokenizer=encoder.tokenizer,
        dec_tokenizer=dec_tokenizer,
    )
    if loss_warnings:
        print("    Warnings:")
        for w in loss_warnings:
            print(f"      - {w}")

    loss_fn, _ = build_loss_from_config(
        config,
        enc_tokenizer=encoder.tokenizer,
        dec_tokenizer=dec_tokenizer,
        dec_vocab_size=V,
        validate=False,
    )
    print(f"    Loss function: {type(loss_fn).__name__}")
    print()

    # =================================================================
    # Load data
    # =================================================================
    print("[6] Loading dataset...")
    dataset = registry.get(data_cfg, split=data_cfg["split"])
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_text,
        drop_last=True,
    )
    print(f"    Dataset: {data_cfg['data_name']}, {len(dataset)} samples")
    print(f"    Batches per epoch: {len(dataloader)}")
    print()

    # =================================================================
    # Setup optimizer and scheduler
    # =================================================================
    print("[7] Setting up optimizer...")
    params = (
        list(encoder.parameters())
        + list(quantizer.parameters())
        + list(decoder.parameters())
    )
    if isinstance(pre_quant_proj, nn.Linear):
        params += list(pre_quant_proj.parameters())

    optimizer = AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    total_steps = len(dataloader) * num_epochs
    scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    print(f"    Total steps: {total_steps}")
    print(f"    Warmup steps: {warmup_steps}")
    print()

    # =================================================================
    # Resume from checkpoint if requested
    # =================================================================
    start_epoch = 0
    global_step = 0
    history = {
        "config": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "max_length": L,
            "scale_lengths": scale_lengths,
            "total_codes": total_codes,
            "loss_type": loss_type,
            "approach": "more_scales",
        },
        "steps": [],
    }

    if resume:
        latest_ckpt = find_latest_checkpoint(checkpoint_dir)
        if latest_ckpt is not None:
            print(f"[7.5] Resuming from: {latest_ckpt.name}")
            models = {
                "encoder": encoder,
                "quantizer": quantizer,
                "decoder": decoder,
            }
            if isinstance(pre_quant_proj, nn.Linear):
                models["pre_quant_proj"] = pre_quant_proj

            start_epoch, global_step, history = resume_from_checkpoint(
                checkpoint_path=latest_ckpt,
                models=models,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                log_dir=log_dir,
            )
            print(f"    Resumed from epoch {start_epoch+1}, global_step {global_step}")
            print()
        else:
            print("[7.5] No checkpoint found, starting fresh")
            print()

    # =================================================================
    # Training loop
    # =================================================================
    print("[8] Starting training...")
    print("=" * 60)

    encoder.train()
    quantizer.train()
    decoder.train()

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_vq_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_texts in pbar:
            # Tokenize
            tokens = encoder.tokenize(batch_texts)
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]

            # Encode
            hidden = encoder(input_ids=input_ids, attention_mask=attention_mask)
            # hidden: [B, L, D]

            # Project encoder output to quantizer input dim
            # hidden [B, L, D] → pre_quant_proj → z [B, L, latent_dim]
            z = pre_quant_proj(hidden)

            # Quantize with MORE SCALES
            # z [B, L, latent_dim] → quantizer → f_hat [B, L, latent_dim]
            f_hat, vq_loss, indices_per_scale = quantizer(z)
            # vq_loss: scalar
            # indices_per_scale: List[[B, scale_k] for k in scale_lengths]

            # Decode (decoder expects latent_dim, f_hat is already latent_dim)
            # f_hat [B, L, latent_dim] → decoder → logits [B, L, V]
            logits = decoder(f_hat, attention_mask=attention_mask)
            # logits: [B, L, V]

            # Compute loss
            total_loss, loss_details = loss_fn(logits, batch_texts, vq_loss)
            recon_loss = loss_details["recon_loss"]

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(params, gradient_clip)
            optimizer.step()
            scheduler.step()

            # Logging
            epoch_loss += total_loss.item()
            epoch_recon_loss += recon_loss
            epoch_vq_loss += (
                vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss
            )
            num_batches += 1
            global_step += 1

            # Record history
            history["steps"].append(
                {
                    "epoch": epoch + 1,
                    "step_in_epoch": num_batches,
                    "global_step": global_step,
                    "loss": total_loss.item(),
                    "recon_loss": recon_loss,
                    "vq_loss": (
                        vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss
                    ),
                    "lr": scheduler.get_last_lr()[0],
                }
            )

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{total_loss.item():.4f}",
                    "recon": f"{recon_loss:.4f}",
                    "vq": f"{vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss:.4f}",
                }
            )

            # Log at intervals
            if global_step % log_interval == 0:
                avg_loss = epoch_loss / num_batches
                lr = scheduler.get_last_lr()[0]
                print(
                    f"    Step {global_step}: loss={total_loss.item():.4f}, "
                    f"recon={recon_loss:.4f}, vq={vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss:.4f}, lr={lr:.2e}"
                )

                # Decode and save samples
                with torch.no_grad():
                    decode_result = decode_logits_to_text(
                        logits, dec_tokenizer, batch_texts, attention_mask
                    )

                samples_path = (
                    log_dir
                    / f"samples-epoch{epoch+1}-step{num_batches}-global{global_step}.json"
                )
                with open(samples_path, "w", encoding="utf-8") as f:
                    json.dump(decode_result, f, indent=2, ensure_ascii=False)

                # Save history
                history_path = log_dir / "training_history.json"
                with open(history_path, "w") as f:
                    json.dump(history, f, indent=2)

            # Save checkpoint
            if global_step % checkpoint_interval == 0:
                checkpoint = {
                    "epoch": epoch + 1,
                    "step_in_epoch": num_batches,
                    "global_step": global_step,
                    "encoder_state_dict": encoder.state_dict(),
                    "quantizer_state_dict": quantizer.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": total_loss.item(),
                }
                if isinstance(pre_quant_proj, nn.Linear):
                    checkpoint["pre_quant_proj_state_dict"] = (
                        pre_quant_proj.state_dict()
                    )

                ckpt_name = f"checkpoint-epoch{epoch+1}-step{num_batches}-global{global_step}.pt"
                torch.save(checkpoint, checkpoint_dir / ckpt_name)
                print(f"    [Checkpoint saved: {ckpt_name}]")

        # Epoch summary
        avg_epoch_loss = epoch_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_vq_loss = epoch_vq_loss / num_batches
        print(
            f"Epoch {epoch+1} completed: loss={avg_epoch_loss:.4f}, recon={avg_recon_loss:.4f}, vq={avg_vq_loss:.4f}"
        )
        print()

    print("=" * 60)
    print("Training completed!")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encoder-Quantizer-Decoder with Multi-Scale Quantization"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Config: {args.config}")
    print("=" * 60)
    train_scales(config)
