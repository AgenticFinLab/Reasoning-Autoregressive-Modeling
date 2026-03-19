"""C3 Context Cascade Compression - Complete Training.

Usage:
    python examples/PreExp/c3_original.py -c configs/PreExp/c3_original.yml

Task:
    Train C3 (Context Cascade Compression) to compress and reconstruct text.
    This implements the full training pipeline from the official paper:
    "Context Cascade Compression: Exploring the Upper Limits of Text Compression"
    (arXiv:2511.15244)

Official Code Reference:
    third-part/C3-Context-Cascade-Compression-main/C3-master/C3/model/C3.py
    - Lines 21-23: Special tokens definition
    - Line 35: Q = nn.Embedding(N, D_encoder)
    - Line 36: mm_projector = nn.Linear(D_encoder, D_decoder)
    - Lines 66-119: Encoder forward
    - Lines 121-153: Decoder forward
    - Lines 372-376: chat() function showing token structure

Config (example: B=2, M=1280, N=32, D_enc=1536, D_dec=2048):
    - B: batch size
    - M: text sequence length (max_length)
    - N: number of latent tokens (num_latent_tokens)
    - D_enc: encoder hidden dimension (Qwen2.5-1.5B: 1536)
    - D_dec: decoder hidden dimension (Qwen2.5-3B: 2048)

Pipeline:
    ┌──────────────────────────────────────────────────────────────────────┐
    │ Input: text + <img> + <imgpad>*N + </img>                           │
    └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ C3Encoder (llm1: Qwen2.5-1.5B)
    ┌──────────────────────────────────────────────────────────────────────┐
    │ hidden_states [B, M+N+2, D_enc]                                      │
    └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ Extract Q positions: [img_pos+1 : img_pos+N+1]
    ┌──────────────────────────────────────────────────────────────────────┐
    │ latent_tokens [B, N, D_enc]    (40x compression: M=1280, N=32)      │
    └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ mm_projector: D_enc -> D_dec
    ┌──────────────────────────────────────────────────────────────────────┐
    │ projected_latent [B, N, D_dec]                                       │
    └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ Insert into decoder input
    ┌──────────────────────────────────────────────────────────────────────┐
    │ decoder_input: <img> + latent_1..N + </img> + "Repeat the text: "   │
    └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ C3Decoder (llm2: Qwen2.5-3B)
    ┌──────────────────────────────────────────────────────────────────────┐
    │ logits [B, L+N, vocab_size]                                          │
    └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ Cross-entropy loss
    ┌──────────────────────────────────────────────────────────────────────┐
    │ loss = CrossEntropy(logits, target_ids)                              │
    └──────────────────────────────────────────────────────────────────────┘

Compression Ratio:
    - Input: M text tokens (e.g., 1280)
    - Output: N latent tokens (e.g., 32)
    - Ratio: M/N = 40x compression
    - Paper reports 93% accuracy at 40x compression

Dimensions:
    B = batch_size
    M = max_length (text sequence length)
    N = num_latent_tokens (latent token count)
    D_enc = encoder hidden_dim (Qwen2.5-1.5B: 1536)
    D_dec = decoder hidden_dim (Qwen2.5-3B: 2048)
    V = vocab_size
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from lmbase.dataset import registry
from transformers import AutoTokenizer

from ram.models.encoder import build_c3_encoder
from ram.models.decoder import build_c3_decoder
from ram.models.encoder import (
    C3_IM_START_TOKEN,
    C3_IM_END_TOKEN,
    C3_IM_PATCH_TOKEN,
)
from ram.utils import (
    load_config,
    setup_environment,
    collate_fn_text,
    find_latest_checkpoint,
)


class C3ReconstructionLoss(nn.Module):
    """C3 Reconstruction Loss.

    Computes cross-entropy loss for text reconstruction.
    The decoder outputs logits for the entire sequence including latent tokens.

    Input:
        logits: [B, L+N, V] decoder output (includes latent token positions)
        target_texts: List[str] original texts to reconstruct

    Output:
        loss: scalar cross-entropy loss
        loss_dict: dict with loss breakdown
    """

    def __init__(
        self,
        tokenizer,
        max_length: int,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_index = ignore_index

    def forward(self, logits, target_texts, num_latent_tokens):
        """Compute reconstruction loss.

        Args:
            logits: [B, L_total, V] decoder output
            target_texts: List[str] original texts
            num_latent_tokens: int, number of latent tokens N

        Returns:
            loss: scalar
            loss_dict: dict with loss info
        """
        B, L_total, V = logits.shape
        N = num_latent_tokens
        L = L_total - N  # Actual text length

        # Tokenize target texts
        tokens = self.tokenizer(
            target_texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_ids = tokens["input_ids"].to(logits.device)  # [B, L]
        attention_mask = tokens["attention_mask"].to(logits.device)  # [B, L]

        # Shift for autoregressive prediction
        # logits contain [latent_positions, text_positions]
        # We want to predict text from latent tokens
        # Skip the first N positions (latent tokens) and predict text
        pred_logits = logits[:, N:-1, :]  # [B, L-1, V] - skip latent and last
        targets = target_ids[:, 1:]  # [B, L-1] - shifted targets
        mask = attention_mask[:, 1:]  # [B, L-1]

        # Mask padding positions
        targets = targets.masked_fill(mask == 0, self.ignore_index)

        # Compute cross-entropy loss
        loss = F.cross_entropy(
            pred_logits.reshape(-1, V),
            targets.reshape(-1),
            ignore_index=self.ignore_index,
        )

        return loss, {
            "recon_loss": loss.item(),
            "total_loss": loss.item(),
        }


def train_c3(config: dict):
    """
    Train C3 for text compression and reconstruction.

    Training Flow:
        Step 1: texts [B] -> C3Encoder -> latent_tokens [B, N, D_enc]
        Step 2: latent_tokens [B, N, D_enc] -> C3Decoder -> logits [B, L+N, V]
        Step 3: loss_fn(logits, texts) -> loss
        Step 4: loss.backward() -> optimizer.step() -> update weights
    """
    # =================================================================
    # Extract config
    # =================================================================
    enc_cfg = config["model"]["encoder"]
    dec_cfg = config["model"]["decoder"]
    data_cfg = config["data"]
    train_cfg = config["train"]
    env_cfg = config["environment"]
    log_cfg = config["logging"]

    # Training hyperparameters
    batch_size = train_cfg["batch_size"]
    learning_rate = train_cfg["learning_rate"]
    weight_decay = train_cfg["weight_decay"]
    num_epochs = train_cfg["num_epochs"]
    warmup_ratio = train_cfg["warmup_ratio"]
    gradient_clip = train_cfg["gradient_clip"]
    gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", 1)
    bf16 = train_cfg.get("bf16", True)
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
    N = enc_cfg["num_latent_tokens"]
    M = enc_cfg["max_length"]

    # Setup environment (seed + device)
    device = setup_environment(env_cfg)

    print("=" * 60)
    print("C3 Context Cascade Compression - Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"Max length (M): {M}")
    print(f"Num latent tokens (N): {N}")
    print(f"Compression ratio: {M/N:.1f}x")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"BFloat16: {bf16}")
    print(f"Output dir: {output_dir}")
    print()

    # =================================================================
    # Build models
    # =================================================================
    print("[1] Building C3Encoder (llm1)...")
    encoder = build_c3_encoder(enc_cfg)
    encoder = encoder.to(device)
    D_enc = encoder.hidden_dim
    print(f"    model: {encoder.model_name}")
    print(f"    hidden_dim: {D_enc}")
    print(f"    num_latent_tokens: {encoder.num_latent_tokens}")
    print()

    print("[2] Building C3Decoder (llm2)...")
    decoder = build_c3_decoder(
        dec_cfg,
        encoder_hidden_dim=D_enc,
        encoder_type="C3Encoder",
    )
    decoder = decoder.to(device)
    D_dec = decoder.hidden_dim
    V = decoder.vocab_size
    print(f"    model: {decoder.model_name}")
    print(f"    hidden_dim: {D_dec}")
    print(f"    vocab_size: {V}")
    print(f"    mm_projector: {D_enc} -> {D_dec}")
    print()

    # =================================================================
    # Tokenizer for loss computation
    # =================================================================
    print("[3] Setting up tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(dec_cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"    pad_token_id: {tokenizer.pad_token_id}")
    print()

    # =================================================================
    # Setup loss function
    # =================================================================
    print("[4] Setting up loss function...")
    loss_fn = C3ReconstructionLoss(
        tokenizer=tokenizer,
        max_length=M,
        ignore_index=-100,
    )
    print(f"    Loss: C3ReconstructionLoss")
    print()

    # =================================================================
    # Load data
    # =================================================================
    print("[5] Loading dataset...")
    dataset = registry.get(data_cfg, split=data_cfg["split"])
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_text,
        drop_last=True,
        num_workers=env_cfg.get("dataloader_num_workers", 8),
    )
    print(f"    Dataset: {data_cfg['data_name']}, {len(dataset)} samples")
    print(f"    Batches per epoch: {len(dataloader)}")
    print()

    # =================================================================
    # Setup optimizer and scheduler
    # =================================================================
    print("[6] Setting up optimizer...")
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    total_steps = len(dataloader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)

    print(f"    Total steps: {total_steps}")
    print(f"    Warmup steps: {warmup_steps}")
    print(f"    LR scheduler: cosine")
    print()

    # =================================================================
    # Resume from checkpoint
    # =================================================================
    start_epoch = 0
    global_step = 0
    history = {
        "config": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "max_length": M,
            "num_latent_tokens": N,
            "compression_ratio": M / N,
            "encoder_model": enc_cfg["model_name"],
            "decoder_model": dec_cfg["model_name"],
        },
        "steps": [],
    }

    if resume:
        latest_ckpt = find_latest_checkpoint(checkpoint_dir)
        if latest_ckpt is not None:
            print(f"[6.5] Resuming from: {latest_ckpt.name}")
            checkpoint = torch.load(latest_ckpt, map_location=device)
            encoder.load_state_dict(checkpoint["encoder_state_dict"])
            decoder.load_state_dict(checkpoint["decoder_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            global_step = checkpoint.get("global_step", 0)
            if "history" in checkpoint:
                history = checkpoint["history"]
            print(f"    Resumed from epoch {start_epoch+1}, global_step {global_step}")
            print()
        else:
            print("[6.5] No checkpoint found, starting fresh")
            print()

    # =================================================================
    # Training loop
    # =================================================================
    print("[7] Starting training...")
    print("=" * 60)

    encoder.train()
    decoder.train()

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=bf16)
    amp_dtype = torch.bfloat16 if bf16 else torch.float32

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        optimizer.zero_grad()

        for batch_idx, batch_texts in enumerate(pbar):
            # Step 1: Encode - texts -> latent_tokens
            # texts [B] -> latent_tokens [B, N, D_enc]
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=bf16):
                latent_tokens = encoder(inputs=batch_texts)

            # Step 2: Decode - latent_tokens -> logits
            # latent_tokens [B, N, D_enc] -> logits [B, L+N, V]
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=bf16):
                logits = decoder(latent_tokens)

            # Step 3: Compute loss
            # logits [B, L+N, V], texts -> loss
            with torch.amp.autocast("cuda", dtype=torch.float32, enabled=False):
                loss, loss_dict = loss_fn(logits, batch_texts, N)

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

            # Step 4: Backward
            scaler.scale(loss).backward()

            # Step 5: Optimizer step (with gradient accumulation)
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, gradient_clip)

                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Update learning rate
                if global_step >= warmup_steps:
                    scheduler.step()

            # Logging
            epoch_loss += loss_dict["total_loss"] * gradient_accumulation_steps
            num_batches += 1
            global_step += 1

            # Record history
            history["steps"].append(
                {
                    "epoch": epoch + 1,
                    "step_in_epoch": num_batches,
                    "global_step": global_step,
                    "loss": loss_dict["total_loss"],
                    "avg_loss": epoch_loss / num_batches,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss_dict['total_loss']:.4f}",
                    "avg": f"{epoch_loss/num_batches:.4f}",
                }
            )

            # --- Log at log_interval ---
            if global_step % log_interval == 0:
                avg_loss = epoch_loss / num_batches
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"    Step {global_step}: loss={loss_dict['total_loss']:.4f}, "
                    f"avg_loss={avg_loss:.4f}, lr={lr:.2e}"
                )

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
                    "loss": loss_dict["total_loss"],
                    "avg_loss": avg_loss,
                    "history": history,
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
            "history": history,
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
    print("[8] Sample reconstruction...")

    encoder.eval()
    decoder.eval()

    # Get a sample batch
    sample_texts = collate_fn_text([dataset[i] for i in range(min(2, len(dataset)))])

    with torch.no_grad():
        # Encode
        latent_tokens = encoder(inputs=sample_texts)
        print(f"    Input texts: {len(sample_texts[0])} chars")
        print(f"    Latent tokens shape: {latent_tokens.shape}")
        print(f"    Compression: {len(sample_texts[0])} chars -> {N} tokens")

        # Generate
        output_ids = decoder.generate(
            latent_tokens,
            prompt="Repeat the text: ",
            max_new_tokens=512,
            do_sample=False,
        )

    # Decode
    print()
    print("    Sample 1:")
    print(f"      Original:      {sample_texts[0][:100]}...")
    pred_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"      Reconstructed: {pred_text[:100]}...")

    if len(sample_texts) > 1:
        print()
        print("    Sample 2:")
        print(f"      Original:      {sample_texts[1][:100]}...")
        pred_text = tokenizer.decode(output_ids[1], skip_special_tokens=True)
        print(f"      Reconstructed: {pred_text[:100]}...")

    print()
    print("=" * 60)
    print("ALL DONE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="C3 Context Cascade Compression Training"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., configs/PreExp/c3_original.yml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Config: {args.config}")
    print("=" * 60)
    train_c3(config)
