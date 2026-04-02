"""Encoder-Decoder Training: Simple Text Reconstruction.

Usage:
    python examples/uTEST/test_ed_train.py -c configs/uTEST/ed_train.yml

Task:
    Train encoder-decoder to reconstruct input text.
    This is the simplest autoencoder baseline before adding quantization.

Config (example: B=4, L=64, D=768, V=50257):
    - B: batch size
    - L: max sequence length
    - D: hidden dimension (BERT=768, GPT2=768)
    - V: vocabulary size (GPT2=50257)

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
    │ hidden [B, L, D]    │  [4, 64, 768] continuous representations
    └────────┬────────────┘
             │ Decoder.forward()
             ▼
    ┌─────────────────────┐
    │ logits [B, L, V]    │  [4, 64, 50257] token logits
    └────────┬────────────┘
             │ loss_fn(logits, texts)
             ▼
    ┌─────────────────────┐
    │ loss (scalar)       │  reconstruction loss
    └────────┬────────────┘
             │ loss.backward() + optimizer.step()
             ▼
    ┌─────────────────────┐
    │ Updated weights     │  encoder + decoder parameters
    └─────────────────────┘

Dimensions:
    B = batch_size (e.g., 4)
    L = max_length (e.g., 64)
    D = hidden_dim (BERT/GPT2: 768)
    V = vocab_size (GPT2: 50257)

Restoration (inference):
    logits [B, L, V=50257] -> argmax(dim=-1) -> pred_ids [B, L]
    pred_ids [B, L] -> tokenizer.decode() -> List[str] texts

    V=50257 is GPT2's vocabulary size:
    - Each position has 50257 logits (one per token)
    - argmax selects the most likely token ID
    - tokenizer.decode converts IDs back to text

Loss Configuration:
    BERT (encoder) + GPT2 (decoder) = different tokenizers
    -> Use "dual_tokenizer_reconstruction" loss type
    -> Target IDs computed from decoder's tokenizer internally
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


def train_ed(config: dict):
    """
    Train Encoder-Decoder for text reconstruction.

    Training Flow:
        Step 1: texts [B] -> Encoder.tokenize() -> input_ids [B, L]
        Step 2: input_ids [B, L] -> Encoder.forward() -> hidden [B, L, D]
        Step 3: hidden [B, L, D] -> Decoder.forward() -> logits [B, L, V]
        Step 4: loss_fn(logits, texts) -> loss (using decoder's tokenizer!)
        Step 5: loss.backward() -> optimizer.step() -> update weights

    NOTE: BERT (encoder) + GPT2 (decoder) have different tokenizers.
          Target IDs are computed from DECODER's tokenizer internally.
    """
    # =================================================================
    # Extract config
    # =================================================================
    enc_cfg = config["encoder"]
    dec_cfg = config["decoder"]
    data_cfg = config["data"]
    env_cfg = config["environment"]
    log_cfg = config["log"]

    # Training hyperparameters
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    weight_decay = config.get("weight_decay", 0.0)
    num_epochs = config["num_epochs"]
    warmup_steps = config.get("warmup_steps", 100)
    gradient_clip = config["gradient"]["max_grad_norm"]
    resume = config.get("resume", True)

    # Logging intervals
    # Print & save samples/history
    log_interval = log_cfg["log_step_interval"]
    # Save model checkpoint
    checkpoint_interval = log_cfg["checkpoint_step_interval"]

    output_dir = Path(log_cfg["save_folder"])
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Dimensions
    L = enc_cfg["max_length"]

    # Setup environment (seed + device)
    device = setup_environment(env_cfg)

    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Max length: {L}")
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

    print("[2] Building Decoder...")
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
    if dec_tokenizer.pad_token is None:
        dec_tokenizer.pad_token = dec_tokenizer.eos_token
    print()

    # =================================================================
    # Setup loss function (from config with validation)
    # =================================================================
    print("[3.5] Setting up loss function...")
    loss_cfg = config["loss"]
    loss_type = loss_cfg["type"]
    print(f"    Loss type: {loss_type}")

    # Validate loss config against tokenizer setup
    loss_warnings = validate_loss_config(
        config,
        enc_tokenizer=encoder.tokenizer,
        dec_tokenizer=dec_tokenizer,
    )
    if loss_warnings:
        print("    Warnings:")
        for w in loss_warnings:
            print(f"      - {w}")

    # Build loss function
    # Already validated above, skip re-validation
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
    print("[3] Loading dataset...")
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
    print("[4] Setting up optimizer...")
    # Combine encoder and decoder parameters
    params = list(encoder.parameters()) + list(decoder.parameters())
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
            "loss_type": loss_type,
        },
        "steps": [],
    }

    if resume:
        latest_ckpt = find_latest_checkpoint(checkpoint_dir)
        if latest_ckpt is not None:
            print(f"[4.5] Resuming from: {latest_ckpt.name}")
            start_epoch, global_step, history = resume_from_checkpoint(
                checkpoint_path=latest_ckpt,
                models={"encoder": encoder, "decoder": decoder},
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                log_dir=log_dir,
            )
            print(f"    Resumed from epoch {start_epoch+1}, global_step {global_step}")
            print(f"    Loaded training history: {len(history['steps'])} steps")
            print()
        else:
            print("[4.5] No checkpoint found, starting fresh")
            print()

    # =================================================================
    # Training loop
    # =================================================================
    print("[5] Starting training...")
    print("=" * 60)

    encoder.train()
    decoder.train()

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_texts in pbar:
            # Step 1: Tokenize
            # texts [B] -> input_ids [B, L], attention_mask [B, L]
            tokens = encoder.tokenize(batch_texts)
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)

            # Step 2: Encode
            # input_ids [B, L] -> hidden [B, L, D]
            hidden = encoder(input_ids=input_ids, attention_mask=attention_mask)

            # Step 3: Decode
            # hidden [B, L, D] -> logits [B, L, V]
            logits = decoder(hidden, attention_mask=attention_mask)

            # Step 4: Compute loss using configured loss function
            # For dual_tokenizer types: loss_fn(logits, texts) -> loss, target_ids
            # For same_tokenizer types: loss_fn(logits, target_ids) -> loss
            if "dual_tokenizer" in loss_type:
                loss, _ = loss_fn(logits, batch_texts)
            else:
                loss = loss_fn(logits, input_ids, attention_mask=attention_mask)

            # Step 5: Backward and optimize
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(params, gradient_clip)

            optimizer.step()
            scheduler.step()

            # Logging
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Record history (every step)
            history["steps"].append(
                {
                    "epoch": epoch + 1,
                    "step_in_epoch": num_batches,
                    "global_step": global_step,
                    "loss": loss.item(),
                    "avg_loss": epoch_loss / num_batches,
                    "lr": scheduler.get_last_lr()[0],
                }
            )

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # --- Log & save samples at log_interval ---
            if global_step % log_interval == 0:
                avg_loss = epoch_loss / num_batches
                lr = scheduler.get_last_lr()[0]
                print(
                    f"    Step {global_step}: loss={loss.item():.4f}, "
                    f"avg_loss={avg_loss:.4f}, lr={lr:.2e}"
                )

                # Decode current batch to text for inspection
                with torch.no_grad():
                    decode_result = decode_logits_to_text(
                        logits, dec_tokenizer, batch_texts, attention_mask
                    )

                # Save decoded text samples (full results)
                step_in_epoch = num_batches
                samples_path = (
                    log_dir
                    / f"samples-epoch{epoch+1}-step{step_in_epoch}-global{global_step}.json"
                )
                with open(samples_path, "w", encoding="utf-8") as f:
                    json.dump(decode_result, f, indent=2, ensure_ascii=False)

                # Save training history
                history_path = log_dir / "training_history.json"
                with open(history_path, "w") as f:
                    json.dump(history, f, indent=2)

            # --- Save checkpoint at checkpoint_interval ---
            if global_step % checkpoint_interval == 0:
                step_in_epoch = num_batches
                avg_loss = epoch_loss / num_batches
                checkpoint = {
                    "epoch": epoch + 1,
                    "step_in_epoch": step_in_epoch,
                    "global_step": global_step,
                    "encoder_state_dict": encoder.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": loss.item(),
                    "avg_loss": avg_loss,
                }
                ckpt_name = f"checkpoint-epoch{epoch+1}-step{step_in_epoch}-global{global_step}.pt"
                ckpt_path = checkpoint_dir / ckpt_name
                torch.save(checkpoint, ckpt_path)
                print(f"    [Checkpoint saved: {ckpt_name}]")

        # Epoch summary
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} completed: avg_loss={avg_epoch_loss:.4f}")
        print()

        # Save checkpoint at end of each epoch
        step_in_epoch = num_batches
        checkpoint = {
            "epoch": epoch + 1,
            "step_in_epoch": step_in_epoch,
            "global_step": global_step,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "avg_loss": avg_epoch_loss,
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
    print("[6] Sample reconstruction...")

    encoder.eval()
    decoder.eval()

    # Get a sample batch
    sample_texts = collate_fn_text([dataset[i] for i in range(min(2, len(dataset)))])

    with torch.no_grad():
        # Encode
        tokens = encoder.tokenize(sample_texts)
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        hidden = encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Decode
        logits = decoder(hidden, attention_mask=attention_mask)

        # logits [B, L, V=50257] -> argmax(dim=-1) -> pred_ids [B, L]
        pred_ids = logits.argmax(dim=-1)

    # Use decode_logits_to_text for consistent decoding
    print("    Sample 1:")
    print(f"      Original:      {sample_texts[0][:80]}...")
    pred_text = dec_tokenizer.decode(pred_ids[0], skip_special_tokens=True)
    print(f"      Reconstructed: {pred_text[:80]}...")
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
        description="Encoder-Decoder Training for Text Reconstruction"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., configs/uTEST/ed_train.yml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Config: {args.config}")
    print("=" * 60)
    train_ed(config)
