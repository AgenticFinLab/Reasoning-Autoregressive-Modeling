"""C3 Context Cascade Compression - Training.

Usage:
    python examples/PreExp/c3_original.py -c configs/PreExp/c3_original.yml

GPU Assignment (Automatic):
    The system automatically assigns models to GPUs based on available memory:
    - Single GPU: All models on the same GPU
    - Multiple GPUs: Distribute by priority (decoder gets GPU with most free memory)

    Config example (configs/PreExp/c3_original.yml):
        model_devices:
          encoder: {device: auto, priority: 2}  # second best GPU
          decoder: {device: auto, priority: 1}  # best GPU (most free memory)

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Text Input                                                         │
    │      │                                                              │
    │      ▼                                                              │
    │  ┌─────────────────┐           ┌─────────────────┐                  │
    │  │ C3Encoder       │           │ C3Decoder       │                  │
    │  │ text → latent   │ ────────▶ │ latent → logits │                  │
    │  │ (Qwen2.5-0.5B)  │  transfer │ (Qwen2.5-1.5B)  │                  │
    │  └─────────────────┘           └─────────────────┘                  │
    │                        latent_tokens [B, N, D]                       │
    └─────────────────────────────────────────────────────────────────────┘

Memory Distribution (Example):
    Pipeline (0.5B + 1.5B): GPU 0 ~8 GB, GPU 1 ~21 GB -> 2× RTX 4090
    Pipeline (1.5B + 3B):   GPU 0 ~20 GB, GPU 1 ~40 GB -> 2× A100 80GB

Dimensions:
    B = batch_size
    M = max_length (text sequence length)
    N = num_latent_tokens (latent token count)
    D_enc = encoder hidden_dim
    D_dec = decoder hidden_dim
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
    set_seed,
    collate_fn_text,
    find_latest_checkpoint,
    assign_model_devices,
)


class C3ReconstructionLoss(nn.Module):
    """C3 Reconstruction Loss with Teacher Forcing.

    Training Flow (matching official C3):
        1. context_ids (text to compress) -> Encoder -> latent_tokens
        2. latent_tokens + input_ids (teacher forcing) -> Decoder -> logits
        3. logits vs labels -> cross-entropy loss

    Official Reference:
        third-part/C3-Context-Cascade-Compression-main/C3-master/C3/model/C3.py
        Lines 182-246: forward function with labels
        Lines 224-234: loss computation
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

    def forward(self, logits, labels, num_latent_tokens):
        """Compute reconstruction loss with teacher forcing.

        Args:
            logits: [B, L_total, V] decoder output
                L_total = N (latent) + L (text tokens)
            labels: [B, L] target token IDs (shifted for next-token prediction)
            num_latent_tokens: int, number of latent tokens N

        Returns:
            loss: scalar
            loss_dict: dict with loss info

        Dimensions:
            logits: [B, N+L, V] where N = num_latent_tokens, L = text length
            labels: [B, L] target token IDs

        Loss Computation (official Lines 224-234):
            # Shift logits and labels for autoregressive prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = CrossEntropyLoss(shift_logits, shift_labels)
        """
        B, L_total, V = logits.shape
        N = num_latent_tokens

        # Shift for autoregressive prediction
        # logits: [B, N+L, V] -> shift_logits: [B, N+L-1, V]
        # labels: [B, L] -> shift_labels: [B, L-1]
        shift_logits = logits[:, N:-1, :].contiguous()  # [B, L-1, V]
        shift_labels = labels[:, 1:].contiguous()  # [B, L-1]

        # Mask padding positions
        shift_labels = shift_labels.masked_fill(
            shift_labels == self.ignore_index, self.ignore_index
        )

        # Compute cross-entropy loss
        loss = F.cross_entropy(
            shift_logits.reshape(-1, V),
            shift_labels.reshape(-1),
            ignore_index=self.ignore_index,
        )

        return loss, {
            "recon_loss": loss.item(),
            "total_loss": loss.item(),
        }


def train_c3(config: dict):
    """
    Train C3 for text compression and reconstruction.

    Supports:
        - Single GPU mode: encoder and decoder on same GPU
        - Pipeline parallel mode: encoder on GPU 0, decoder on GPU 1

    Training Flow (matching official C3):
        Step 1: Tokenize texts -> input_ids [B, L], labels [B, L]
        Step 2: texts -> C3Encoder -> latent_tokens [B, N, D_enc]
        Step 3: latent_tokens + input_ids (teacher forcing) -> C3Decoder -> logits [B, N+L, V]
        Step 4: logits vs labels -> cross-entropy loss
        Step 5: loss.backward() -> optimizer.step() -> update weights

    Official Reference:
        third-part/C3-Context-Cascade-Compression-main/C3-master/C3/model/C3.py
        Lines 182-246: forward function with labels
        Lines 224-234: loss computation
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
    gradient_accumulation_steps = train_cfg["gradient_accumulation_steps"]
    bf16 = train_cfg["bf16"]
    resume = train_cfg["resume"]

    # Model device config
    model_devices_cfg = config["model_devices"]

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

    # Setup seed
    set_seed(env_cfg["seed"])

    print("=" * 60)
    print("C3 Context Cascade Compression - Training")
    print("=" * 60)
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
    # Assign GPU devices for each model
    # =================================================================
    model_devices = assign_model_devices(model_devices_cfg)
    encoder_device = model_devices["encoder"]
    decoder_device = model_devices["decoder"]
    use_pipeline = encoder_device != decoder_device
    print()

    # =================================================================
    # Build models - place on appropriate GPUs
    # =================================================================
    print(f"[1] Building C3Encoder on {encoder_device}...")
    encoder = build_c3_encoder(enc_cfg)
    encoder = encoder.to(encoder_device)
    D_enc = encoder.hidden_dim
    print(f"    model: {encoder.model_name}")
    print(f"    hidden_dim: {D_enc}")
    print(f"    num_latent_tokens: {encoder.num_latent_tokens}")

    print(f"\n[2] Building C3Decoder on {decoder_device}...")
    decoder = build_c3_decoder(
        dec_cfg,
        encoder_hidden_dim=D_enc,
        encoder_type="C3Encoder",
    )
    decoder = decoder.to(decoder_device)
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
        num_workers=env_cfg["dataloader_num_workers"],
    )
    print(f"    Dataset: {data_cfg['data_name']}, {len(dataset)} samples")
    print(f"    Batches per epoch: {len(dataloader)}")
    print()

    # =================================================================
    # Setup optimizers - separate for encoder and decoder
    # =================================================================
    print("[6] Setting up optimizers...")
    encoder_optimizer = AdamW(
        encoder.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    decoder_optimizer = AdamW(
        decoder.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    total_steps = len(dataloader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    encoder_scheduler = CosineAnnealingLR(
        encoder_optimizer,
        T_max=total_steps - warmup_steps,
    )
    decoder_scheduler = CosineAnnealingLR(
        decoder_optimizer,
        T_max=total_steps - warmup_steps,
    )

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
            "use_pipeline": use_pipeline,
            "encoder_device": encoder_device,
            "decoder_device": decoder_device,
        },
        "steps": [],
    }

    if resume:
        latest_ckpt = find_latest_checkpoint(checkpoint_dir)
        if latest_ckpt is not None:
            print(f"[6.5] Resuming from: {latest_ckpt.name}")
            checkpoint = torch.load(latest_ckpt, map_location="cpu")
            encoder.load_state_dict(checkpoint["encoder_state_dict"])
            decoder.load_state_dict(checkpoint["decoder_state_dict"])
            encoder_optimizer.load_state_dict(
                checkpoint["encoder_optimizer_state_dict"]
            )
            decoder_optimizer.load_state_dict(
                checkpoint["decoder_optimizer_state_dict"]
            )
            encoder_scheduler.load_state_dict(
                checkpoint["encoder_scheduler_state_dict"]
            )
            decoder_scheduler.load_state_dict(
                checkpoint["decoder_scheduler_state_dict"]
            )
            start_epoch = checkpoint["epoch"]
            global_step = checkpoint["global_step"]
            history = checkpoint["history"]
            # Move models to correct devices after loading
            encoder = encoder.to(encoder_device)
            decoder = decoder.to(decoder_device)
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

    # Mixed precision scalers
    encoder_scaler = torch.amp.GradScaler("cuda", enabled=bf16)
    decoder_scaler = torch.amp.GradScaler("cuda", enabled=bf16)
    amp_dtype = torch.bfloat16 if bf16 else torch.float32

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        for batch_idx, batch_texts in enumerate(pbar):
            # =========================================================
            # Step 1: Tokenize texts for teacher forcing
            # context_ids: for encoder (text to compress)
            # input_ids: for decoder (teacher forcing input)
            # labels: for loss computation (target)
            # =========================================================
            tokens = tokenizer(
                batch_texts,
                max_length=M,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"]  # [B, L]
            attention_mask = tokens["attention_mask"]  # [B, L]

            # Labels: shift input_ids for next-token prediction
            # labels[i] = input_ids[i, 1:] (predict next token)
            labels = input_ids.clone()
            # Mask padding tokens in labels
            labels[attention_mask == 0] = -100  # ignore_index

            # =========================================================
            # Step 2: Encode on encoder device
            # Encoder takes raw text and produces latent tokens
            # =========================================================
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=bf16):
                latent_tokens = encoder(inputs=batch_texts)
            # latent_tokens: [B, N, D_enc] on encoder_device

            # =========================================================
            # Step 3: Transfer latent tokens if pipeline mode
            # =========================================================
            if use_pipeline:
                latent_tokens = latent_tokens.to(decoder_device)

            # =========================================================
            # Step 4: Decode on decoder device with teacher forcing
            # Pass input_ids for teacher forcing
            # =========================================================
            input_ids_dev = input_ids.to(decoder_device)
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=bf16):
                logits = decoder(latent_tokens, prompt_ids=input_ids_dev)
            # logits: [B, N+L, V] on decoder_device

            # =========================================================
            # Step 5: Compute loss
            # =========================================================
            labels_dev = labels.to(decoder_device)
            with torch.amp.autocast(
                device_type="cuda", dtype=torch.float32, enabled=False
            ):
                loss, loss_dict = loss_fn(logits, labels_dev, N)

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

            # =========================================================
            # Step 6: Backward
            # =========================================================
            decoder_scaler.scale(loss).backward()

            # =========================================================
            # Step 7: Optimizer step (with gradient accumulation)
            # =========================================================
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                decoder_scaler.unscale_(decoder_optimizer)
                nn.utils.clip_grad_norm_(decoder.parameters(), gradient_clip)

                # Decoder optimizer step
                decoder_scaler.step(decoder_optimizer)
                decoder_scaler.update()
                decoder_optimizer.zero_grad()

                # Encoder optimizer step
                encoder_scaler.unscale_(encoder_optimizer)
                nn.utils.clip_grad_norm_(encoder.parameters(), gradient_clip)
                encoder_scaler.step(encoder_optimizer)
                encoder_scaler.update()
                encoder_optimizer.zero_grad()

                # Update learning rate
                if global_step >= warmup_steps:
                    encoder_scheduler.step()
                    decoder_scheduler.step()

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
                    "lr_encoder": encoder_optimizer.param_groups[0]["lr"],
                    "lr_decoder": decoder_optimizer.param_groups[0]["lr"],
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
                lr_enc = encoder_optimizer.param_groups[0]["lr"]
                lr_dec = decoder_optimizer.param_groups[0]["lr"]
                print(
                    f"    Step {global_step}: loss={loss_dict['total_loss']:.4f}, "
                    f"avg_loss={avg_loss:.4f}, lr_enc={lr_enc:.2e}, lr_dec={lr_dec:.2e}"
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
                    "encoder_optimizer_state_dict": encoder_optimizer.state_dict(),
                    "decoder_optimizer_state_dict": decoder_optimizer.state_dict(),
                    "encoder_scheduler_state_dict": encoder_scheduler.state_dict(),
                    "decoder_scheduler_state_dict": decoder_scheduler.state_dict(),
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
            "encoder_optimizer_state_dict": encoder_optimizer.state_dict(),
            "decoder_optimizer_state_dict": decoder_optimizer.state_dict(),
            "encoder_scheduler_state_dict": encoder_scheduler.state_dict(),
            "decoder_scheduler_state_dict": decoder_scheduler.state_dict(),
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

        # Transfer if pipeline mode
        if use_pipeline:
            latent_tokens = latent_tokens.to(decoder_device)

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
