"""Encoder-Quantizer-Decoder Training: VQ-AE Text Reconstruction.

Usage:
    python examples/uTEST/test_eqd_train.py -c configs/uTEST/eqd_train.yml

Task:
    Train VQ-AE (Vector Quantized Autoencoder) for text reconstruction.
    This adds multi-scale quantization bottleneck between encoder and decoder.

Config (example: B=4, L=64, D=256, V=50257, K=1024):
    - B: batch size
    - L: max sequence length
    - D: latent dimension (encoder output_dim = quantizer codebook_dim)
    - V: vocabulary size (GPT2=50257)
    - K: codebook size (quantizer)

Pipeline:
    ┌─────────────────────┐
    │ List[str] texts     │  B texts
    └────────┬────────────┘
             │ Encoder.tokenize()
             ▼
    ┌─────────────────────┐
    │ input_ids [B, L]    │  token IDs (target for loss)
    └────────┬────────────┘
             │ Encoder.forward()
             ▼
    ┌─────────────────────┐
    │ z [B, L, D]         │  [4, 64, 256] continuous features
    └────────┬────────────┘
             │ Quantizer.forward()
             ▼
    ┌─────────────────────┐
    │ f_hat [B, L, D]     │  [4, 64, 256] quantized features
    │ vq_loss (scalar)    │  VQ commitment + codebook loss
    │ indices [K scales]  │  discrete codes per scale
    └────────┬────────────┘
             │ Decoder.forward()
             ▼
    ┌─────────────────────┐
    │ logits [B, L, V]    │  [4, 64, 50257] token logits
    └────────┬────────────┘
             │ CrossEntropyLoss + vq_loss
             ▼
    ┌─────────────────────┐
    │ total_loss          │  recon_loss + vq_loss
    └────────┬────────────┘
             │ loss.backward() + optimizer.step()
             ▼
    ┌─────────────────────┐
    │ Updated weights     │  encoder + quantizer + decoder
    └─────────────────────┘

Multi-Scale Quantization (VAR's Innovation):
    For each scale k ∈ [1, 2, 4, 8, 16, 32, 64]:
        f_rest [B, L, D] -> downsample -> [B, k, D]
        [B, k, D] -> codebook lookup -> indices [B, k]
        [B, k, D] -> phi_k -> h_k -> upsample -> [B, L, D]
        f_hat += h_k_up, f_rest -= h_k_up

Dimensions:
    B = batch_size (e.g., 4)
    L = max_length (e.g., 64)
    D = latent_dim (e.g., 256)
    V = vocab_size (GPT2: 50257)
    K = codebook_size (e.g., 1024)

Restoration (inference):
    logits [B, L, V=50257] -> argmax(dim=-1) -> pred_ids [B, L]
    pred_ids [B, L] -> tokenizer.decode() -> List[str] texts

    V=50257 is GPT2's vocabulary size:
    - Each position has 50257 logits (one per token)
    - argmax selects the most likely token ID
    - tokenizer.decode converts IDs back to text
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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
)


def train_eqd(config: dict):
    """
    Train Encoder-Quantizer-Decoder (VQ-AE) for text reconstruction.

    Training Flow:
        Step 1: texts [B] -> Encoder.tokenize() -> input_ids [B, L]
        Step 2: input_ids [B, L] -> Encoder.forward() -> z [B, L, D]
        Step 3: z [B, L, D] -> Quantizer.forward() -> f_hat [B, L, D], vq_loss
        Step 4: f_hat [B, L, D] -> Decoder.forward() -> logits [B, L, V]
        Step 5: CrossEntropyLoss(logits, input_ids) + vq_loss -> total_loss
        Step 6: total_loss.backward() -> optimizer.step() -> update weights
    """
    # =================================================================
    # Extract config
    # =================================================================
    enc_cfg = config["model"]["encoder"]
    dec_cfg = config["model"]["decoder"]
    quant_cfg = config["model"]["quantizer"]
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
    log_interval = log_cfg["log_interval"]
    output_dir = Path(log_cfg["output_dir"])
    checkpoint_dir = Path(log_cfg["checkpoint_dir"])
    log_dir = Path(log_cfg["log_dir"])

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Dimensions
    L = enc_cfg["max_length"]
    D = enc_cfg["output_dim"]

    # Setup environment (seed + device)
    device = setup_environment(env_cfg)

    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Max length: {L}")
    print(f"Latent dim: {D}")
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
    print(f"    hidden_dim: {encoder.hidden_dim}, output_dim: {encoder.output_dim}")

    print("[2] Building Quantizer...")
    quantizer = build_quantizer(quant_cfg)
    quantizer = quantizer.to(device)
    print(f"    codebook_size: {quantizer.codebook_size}")
    print(f"    codebook_dim: {quantizer.codebook_dim}")
    print(f"    scale_lengths: {quantizer.scale_lengths}")
    print(f"    num_scales: {quantizer.num_scales}")

    print("[3] Building Decoder...")
    decoder = build_decoder(dec_cfg, input_dim=D)
    decoder = decoder.to(device)
    V = decoder.vocab_size
    print(f"    hidden_dim: {decoder.hidden_dim}, vocab_size: {V}")

    # Pad token ID for loss masking
    pad_token_id = encoder.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = 0
    print(f"    pad_token_id: {pad_token_id}")

    # Decoder tokenizer for text reconstruction
    dec_tokenizer = AutoTokenizer.from_pretrained(decoder.model_name)
    print()

    # =================================================================
    # Load data
    # =================================================================
    print("[4] Loading dataset...")
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
    print("[5] Setting up optimizer...")
    # Combine encoder, quantizer, and decoder parameters
    params = (
        list(encoder.parameters())
        + list(quantizer.parameters())
        + list(decoder.parameters())
    )
    optimizer = AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    # Linear warmup scheduler
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
    # Training loop
    # =================================================================
    print("[6] Starting training...")
    print("=" * 60)

    encoder.train()
    quantizer.train()
    decoder.train()

    # Training history for logging
    history = {
        "train_loss": [],
        "recon_loss": [],
        "vq_loss": [],
        "learning_rate": [],
        "config": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "max_length": L,
            "latent_dim": D,
            "codebook_size": quantizer.codebook_size,
            "scale_lengths": quantizer.scale_lengths,
        },
    }

    global_step = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_vq_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_texts in pbar:
            # Step 1: Tokenize
            # texts [B] -> input_ids [B, L], attention_mask [B, L]
            tokens = encoder.tokenize(batch_texts)
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)

            # Step 2: Encode
            # input_ids [B, L] -> z [B, L, D]
            z = encoder(input_ids=input_ids, attention_mask=attention_mask)

            # Step 3: Quantize
            # z [B, L, D] -> f_hat [B, L, D], vq_loss, indices_per_scale
            f_hat, vq_loss, indices_per_scale = quantizer(z)

            # Step 4: Decode
            # f_hat [B, L, D] -> logits [B, L, V]
            logits = decoder(f_hat, attention_mask=attention_mask)

            # Step 5: Compute loss
            # logits [B, L, V] vs input_ids [B, L] -> recon_loss (scalar)
            recon_loss = F.cross_entropy(
                logits.view(-1, V),
                input_ids.view(-1),
                ignore_index=pad_token_id,
            )
            # total_loss = recon_loss + vq_loss
            total_loss = recon_loss + vq_loss

            # Step 6: Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(params, gradient_clip)

            optimizer.step()
            scheduler.step()

            # Logging
            epoch_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_vq_loss += vq_loss.item()
            num_batches += 1
            global_step += 1

            # Record history
            history["train_loss"].append(total_loss.item())
            history["recon_loss"].append(recon_loss.item())
            history["vq_loss"].append(vq_loss.item())
            history["learning_rate"].append(scheduler.get_last_lr()[0])

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{total_loss.item():.4f}",
                    "recon": f"{recon_loss.item():.4f}",
                    "vq": f"{vq_loss.item():.4f}",
                }
            )

            # Detailed logging and checkpoint saving at log_interval
            if global_step % log_interval == 0:
                avg_loss = epoch_loss / num_batches
                avg_recon = epoch_recon_loss / num_batches
                avg_vq = epoch_vq_loss / num_batches
                lr = scheduler.get_last_lr()[0]
                print(
                    f"    Step {global_step}: loss={total_loss.item():.4f}, "
                    f"recon={recon_loss.item():.4f}, vq={vq_loss.item():.4f}, lr={lr:.2e}"
                )

                # Decode current batch to text for inspection
                with torch.no_grad():
                    decode_result = decode_logits_to_text(
                        logits, dec_tokenizer, batch_texts
                    )
                    # Add quantization info
                    decode_result["indices_per_scale"] = [
                        idx.cpu().tolist() for idx in indices_per_scale
                    ]

                # Save intermediate checkpoint
                step_in_epoch = num_batches
                checkpoint = {
                    "epoch": epoch + 1,
                    "step_in_epoch": step_in_epoch,
                    "global_step": global_step,
                    "encoder_state_dict": encoder.state_dict(),
                    "quantizer_state_dict": quantizer.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": total_loss.item(),
                    "recon_loss": recon_loss.item(),
                    "vq_loss": vq_loss.item(),
                }
                ckpt_name = f"checkpoint-epoch{epoch+1}-step{step_in_epoch}-global{global_step}.pt"
                ckpt_path = checkpoint_dir / ckpt_name
                torch.save(checkpoint, ckpt_path)

                # Save decoded text samples (full results)
                samples_path = (
                    log_dir
                    / f"samples-epoch{epoch+1}-step{step_in_epoch}-global{global_step}.json"
                )
                with open(samples_path, "w", encoding="utf-8") as f:
                    json.dump(decode_result, f, indent=2, ensure_ascii=False)

                # Save intermediate training history
                history_path = log_dir / "training_history.json"
                with open(history_path, "w") as f:
                    json.dump(history, f, indent=2)

        # Epoch summary
        avg_epoch_loss = epoch_loss / num_batches
        avg_epoch_recon = epoch_recon_loss / num_batches
        avg_epoch_vq = epoch_vq_loss / num_batches
        print(
            f"Epoch {epoch+1} completed: loss={avg_epoch_loss:.4f}, "
            f"recon={avg_epoch_recon:.4f}, vq={avg_epoch_vq:.4f}"
        )
        print()

        # Save checkpoint at end of each epoch
        step_in_epoch = num_batches
        checkpoint = {
            "epoch": epoch + 1,
            "step_in_epoch": step_in_epoch,
            "global_step": global_step,
            "encoder_state_dict": encoder.state_dict(),
            "quantizer_state_dict": quantizer.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "avg_loss": avg_epoch_loss,
            "avg_recon_loss": avg_epoch_recon,
            "avg_vq_loss": avg_epoch_vq,
        }
        ckpt_name = (
            f"checkpoint-epoch{epoch+1}-step{step_in_epoch}-global{global_step}.pt"
        )
        ckpt_path = checkpoint_dir / ckpt_name
        torch.save(checkpoint, ckpt_path)
        print(f"    Checkpoint saved: {ckpt_path}")

    print("=" * 60)
    print("Training completed!")
    print()

    # Save final checkpoint
    final_ckpt = checkpoint_dir / "checkpoint_final.pt"
    torch.save(checkpoint, final_ckpt)
    print(f"Final checkpoint saved: {final_ckpt}")

    # Save training history
    history_path = log_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved: {history_path}")
    print()

    # =================================================================
    # Evaluation: Sample reconstruction
    # =================================================================
    print("[7] Sample reconstruction...")

    encoder.eval()
    quantizer.eval()
    decoder.eval()

    # Get a sample batch
    sample_texts = collate_fn_text([dataset[i] for i in range(min(2, len(dataset)))])

    with torch.no_grad():
        # Encode
        tokens = encoder.tokenize(sample_texts)
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        z = encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Quantize
        f_hat, vq_loss, indices_per_scale = quantizer(z)

        # Decode
        logits = decoder(f_hat, attention_mask=attention_mask)

        # logits [B, L, V=50257] -> argmax(dim=-1) -> pred_ids [B, L]
        pred_ids = logits.argmax(dim=-1)

    # Use decode_logits_to_text for consistent decoding
    print("    Sample 1:")
    print(f"      Original:      {sample_texts[0][:80]}...")
    pred_text = dec_tokenizer.decode(pred_ids[0], skip_special_tokens=True)
    print(f"      Reconstructed: {pred_text[:80]}...")
    print(f"      VQ Loss:       {vq_loss.item():.4f}")
    print()

    if len(sample_texts) > 1:
        print("    Sample 2:")
        print(f"      Original:      {sample_texts[1][:80]}...")
        pred_text = dec_tokenizer.decode(pred_ids[1], skip_special_tokens=True)
        print(f"      Reconstructed: {pred_text[:80]}...")

    print()
    print("=" * 60)
    print("ALL DONE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encoder-Quantizer-Decoder Training for VQ-AE Text Reconstruction"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., configs/uTEST/eqd_train.yml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Config: {args.config}")
    print("=" * 60)
    train_eqd(config)
