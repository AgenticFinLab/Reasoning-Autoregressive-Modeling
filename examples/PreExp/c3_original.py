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
    N = latent_token_len (latent token count, official naming)
    D_enc = encoder hidden_dim
    D_dec = decoder hidden_dim
    V = vocab_size
"""

import argparse
from pathlib import Path

from lmbase.dataset import registry
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from ram.models.encoder import build_c3_encoder
from ram.models.decoder import build_c3_decoder
from ram.utils import (
    load_config,
    setup_environment,
    collate_fn_text,
    find_latest_checkpoint,
    assign_model_devices,
    decode_logits_to_text,
)
from ram.utils.tools import save_checkpoint, resume_from_checkpoint
from ram import (
    TrainingConfig,
    TrainingStep,
    TrainingHistory,
    TrainingLogger,
    CheckpointData,
    CheckpointMetadata,
    ReconstructionSampleStore,
    create_reconstruction_samples,
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

    def forward(self, logits, labels, latent_token_len):
        """Compute reconstruction loss with teacher forcing.

        Args:
            logits: [B, L_total, V] decoder output
                L_total = N (latent) + L (text tokens)
            labels: [B, L] target token IDs (shifted for next-token prediction)
            latent_token_len: int, number of latent tokens N (official naming)

        Returns:
            loss: scalar
            loss_dict: dict with loss info

        Dimensions:
            logits: [B, N+L, V] where N = latent_token_len, L = text length
            labels: [B, L] target token IDs

        Loss Computation (official Lines 224-234):
            # Shift logits and labels for autoregressive prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = CrossEntropyLoss(shift_logits, shift_labels)
        """
        # Shape: [B, N+L, V] where N = latent_token_len, L = text length
        _, _, V = logits.shape
        N = latent_token_len

        # Shift for autoregressive prediction
        # logits: [B, N+L, V] -> shift_logits: [B, N+L-1, V]
        # labels: [B, L] -> shift_labels: [B, L-1]
        # shift_logits shape: [B, L-1, V]
        # shift_labels shape: [B, L-1]
        shift_logits = logits[:, N:-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Mask padding positions
        shift_labels = shift_labels.masked_fill(
            shift_labels == self.ignore_index, self.ignore_index
        )

        # Compute cross-entropy loss (reconstruction loss)
        recon_loss = F.cross_entropy(
            shift_logits.reshape(-1, V),
            shift_labels.reshape(-1),
            ignore_index=self.ignore_index,
        )

        # Currently only one loss component, but structure supports multiple:
        # - recon_loss: reconstruction loss (cross-entropy)
        # - reg_loss: regularization (if added in future)
        # - total_loss: sum of all components
        # Sum of all loss components
        recon_loss_val = recon_loss.item()
        total_loss = recon_loss

        # Return total loss and loss dictionary for logging
        # recon_loss: Reconstruction loss (cross-entropy)
        # total_loss: Total = sum of all components
        return total_loss, {
            "recon_loss": recon_loss_val,
            "total_loss": recon_loss_val,
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
    warmup_ratio = config.get("warmup_ratio", 0.01)
    gradient_clip = config["gradient"]["max_grad_norm"]
    gradient_accumulation_steps = config["gradient"]["accumulation_steps"]
    bf16 = config.get("bf16", True)
    resume = config.get("resume", True)

    # Model device config (from environment.device_map)
    model_devices_cfg = config["environment"]["device_map"]

    # Logging intervals
    log_interval = log_cfg["log_step_interval"]
    checkpoint_interval = log_cfg["checkpoint_step_interval"]

    output_dir = Path(log_cfg["save_folder"])
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # =================================================================
    # Setup unified logging
    # =================================================================
    logger = TrainingLogger(
        name="c3_train",
        log_file=log_dir / "training.log",
    )

    logger.log_header("C3 Context Cascade Compression - Training")

    # =================================================================
    # Dimensions
    # =================================================================
    # Use official naming: latent_token_len (C3 config key)
    # Backward compatibility: also accept num_latent_tokens
    N = enc_cfg.get("latent_token_len", enc_cfg.get("num_latent_tokens", 32))
    M = enc_cfg["max_length"]

    # Setup environment (seed only, device assignment is done by assign_model_devices)
    # device=cpu means no GPU assignment here
    setup_environment({"seed": env_cfg["seed"], "device": "cpu"})

    # =================================================================
    # Assign GPU devices for each model
    # =================================================================
    model_devices = assign_model_devices(model_devices_cfg)
    encoder_device = model_devices["encoder"]
    decoder_device = model_devices["decoder"]
    use_pipeline = encoder_device != decoder_device

    # =================================================================
    # Create TrainingConfig
    # =================================================================
    training_config = TrainingConfig(
        experiment_name="c3_original",
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_clip=gradient_clip,
        bf16=bf16,
        latent_token_len=N,
        max_length=M,
        compression_ratio=M / N,
        use_pipeline=use_pipeline,
        encoder_device=encoder_device,
        decoder_device=decoder_device,
    )
    logger.log_config(training_config)
    logger.info("")

    # =================================================================
    # Build models - place on appropriate GPUs
    # =================================================================
    logger.info(f"[1] Building C3Encoder on {encoder_device}...")
    encoder = build_c3_encoder(enc_cfg)
    encoder = encoder.to(encoder_device)
    D_enc = encoder.hidden_dim
    logger.info(f"    model: {encoder.model_name}")
    logger.info(f"    hidden_dim: {D_enc}")
    logger.info(f"    latent_token_len: {encoder.latent_token_len}")

    logger.info("")
    logger.info(f"[2] Building C3Decoder on {decoder_device}...")
    decoder = build_c3_decoder(
        dec_cfg,
        encoder_hidden_dim=D_enc,
        encoder_type="C3Encoder",
    )
    decoder = decoder.to(decoder_device)
    D_dec = decoder.hidden_dim
    V = decoder.vocab_size
    logger.info(f"    model: {decoder.model_name}")
    logger.info(f"    hidden_dim: {D_dec}")
    logger.info(f"    vocab_size: {V}")
    logger.info(f"    mm_projector: {D_enc} -> {D_dec}")
    logger.info("")

    # =================================================================
    # Tokenizer for loss computation
    # =================================================================
    logger.info("[3] Setting up tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(dec_cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"    pad_token_id: {tokenizer.pad_token_id}")
    logger.info("")

    # =================================================================
    # Setup loss function
    # =================================================================
    logger.info("[4] Setting up loss function...")
    # Read loss config
    loss_cfg = config["loss"]
    ignore_index = loss_cfg["ignore_index"]
    loss_fn = C3ReconstructionLoss(
        tokenizer=tokenizer,
        max_length=M,
        ignore_index=ignore_index,
    )
    logger.info(f"    Loss: C3ReconstructionLoss, ignore_index={ignore_index}")
    logger.info("")

    # =================================================================
    # Load data
    # =================================================================
    logger.info("[5] Loading dataset...")
    dataset = registry.get(data_cfg, split=data_cfg["split"])
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_text,
        drop_last=True,
        num_workers=env_cfg["dataloader_num_workers"],
    )
    logger.info(f"    Dataset: {data_cfg['data_name']}, {len(dataset)} samples")
    logger.info(f"    Batches per epoch: {len(dataloader)}")
    logger.info("")

    # =================================================================
    # Setup optimizers - separate for encoder and decoder
    # =================================================================
    logger.info("[6] Setting up optimizers...")
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

    logger.info(f"    Total steps: {total_steps}")
    logger.info(f"    Warmup steps: {warmup_steps}")
    logger.info("    LR scheduler: cosine")
    logger.info("")

    # =================================================================
    # Resume from checkpoint
    # =================================================================
    start_epoch = 0
    global_step = 0

    # =================================================================
    # Setup training history and save config
    # =================================================================
    # Save training config (one-time snapshot)
    config_path = log_dir / "train_config.json"
    training_config.save(config_path)
    logger.info(f"    Training config saved: {config_path}")

    # Setup training history manager
    history_path = log_dir / "training_history.json"
    history = TrainingHistory(training_config, history_path)
    logger.info(f"    History file: {history_path}")

    # Setup reconstruction sample store (block-based storage)
    samples_store = ReconstructionSampleStore(
        folder=str(log_dir / "samples"),
        block_size=50,
    )
    logger.info(f"    Samples store: {log_dir / 'samples'}")
    logger.info("")

    if resume:
        latest_ckpt = find_latest_checkpoint(checkpoint_dir)
        if latest_ckpt is not None:
            logger.info(f"[6.5] Resuming from: {latest_ckpt.name}")
            # Use resume_from_checkpoint with multiple optimizers/schedulers
            start_epoch, global_step, _ = resume_from_checkpoint(
                checkpoint_path=latest_ckpt,
                models={"encoder": encoder, "decoder": decoder},
                optimizer={
                    "encoder": encoder_optimizer,
                    "decoder": decoder_optimizer,
                },
                scheduler={
                    "encoder": encoder_scheduler,
                    "decoder": decoder_scheduler,
                },
                # Load to CPU first, then move to correct devices
                device="cpu",
                log_dir=log_dir,
            )
            # Move models to correct devices after loading
            encoder = encoder.to(encoder_device)
            decoder = decoder.to(decoder_device)
            logger.info(
                f"    Resumed from epoch {start_epoch+1}, global_step {global_step}"
            )
            logger.info("")
        else:
            logger.info("[6.5] No checkpoint found, starting fresh")
            logger.info("")

    # =================================================================
    # Training loop
    # =================================================================
    logger.log_subheader("[7] Starting training...")

    encoder.train()
    decoder.train()

    # Mixed precision scalers
    # NOTE: GradScaler is NOT needed for BF16 (BF16 has same exponent range as FP32)
    # Only enable for FP16 training, disabled for BF16 and FP32
    # Disabled for BF16
    encoder_scaler = torch.amp.GradScaler("cuda", enabled=False)
    decoder_scaler = torch.amp.GradScaler("cuda", enabled=False)
    amp_dtype = torch.bfloat16 if bf16 else torch.float32

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        # =========================================================
        # Save epoch-start checkpoint (before any optimization)
        # =========================================================
        if epoch > 0 or start_epoch > 0:
            ckpt_name = f"checkpoint-epoch{epoch}-start.pt"
            save_checkpoint(
                checkpoint_path=checkpoint_dir / ckpt_name,
                models={"encoder": encoder, "decoder": decoder},
                optimizer={"encoder": encoder_optimizer, "decoder": decoder_optimizer},
                scheduler={"encoder": encoder_scheduler, "decoder": decoder_scheduler},
                epoch=epoch,
                global_step=global_step,
                extra_info={"step_in_epoch": 0, "total_steps": len(history)},
            )
            logger.info(f"    [Epoch-start checkpoint saved: {ckpt_name}]")

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
            # Token outputs: [B, L]
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]

            # Labels: shift input_ids for next-token prediction
            # labels[i] = input_ids[i, 1:] (predict next token)
            labels = input_ids.clone()
            # Mask padding tokens in labels
            labels[attention_mask == 0] = ignore_index

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
            # Record unscaled loss for accurate averaging
            # (loss was scaled for grad accumulation, but we want true loss for logging)
            unscaled_loss = loss_dict["total_loss"]
            epoch_loss += unscaled_loss
            num_batches += 1
            global_step += 1

            # Record history with TrainingStep
            step_record = TrainingStep(
                epoch=epoch + 1,
                step_in_epoch=num_batches,
                global_step=global_step,
                total_loss=loss_dict["total_loss"],
                recon_loss=loss_dict["recon_loss"],
                avg_loss=epoch_loss / num_batches,
                lr_encoder=encoder_optimizer.param_groups[0]["lr"],
                lr_decoder=decoder_optimizer.param_groups[0]["lr"],
            )
            history.append(step_record)

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss_dict['total_loss']:.4f}",
                    "avg": f"{epoch_loss/num_batches:.4f}",
                }
            )

            # --- Log at log_interval ---
            if global_step % log_interval == 0:
                logger.log_step(step_record, log_interval=1)

                # --- Save reconstruction samples ---
                # Use training logits for reconstruction (teacher forcing output)
                # logits: [B, N+L, V] -> skip N latent tokens -> [B, L, V]
                with torch.no_grad():
                    # Skip latent token positions for decoding
                    # text_logits: [B, L, V]
                    text_logits = logits[:, N:, :]
                    decode_result = decode_logits_to_text(
                        text_logits, tokenizer, batch_texts, attention_mask
                    )

                # Create reconstruction samples and add to step record
                recon_samples = create_reconstruction_samples(decode_result)
                step_record.reconstruction_samples = recon_samples

                # Save samples to block-based store for alignment with training history
                sample_key = samples_store.save_samples(step_record, recon_samples)
                logger.info(f"    [Samples saved: {sample_key}]")

            # --- Save checkpoint at checkpoint_interval ---
            if global_step % checkpoint_interval == 0:
                step_in_epoch = num_batches
                avg_loss = epoch_loss / num_batches
                ckpt_name = f"checkpoint-epoch{epoch+1}-step{step_in_epoch}-global{global_step}.pt"
                # Create checkpoint metadata
                ckpt_metadata = CheckpointMetadata(
                    epoch=epoch + 1,
                    global_step=global_step,
                    step_in_epoch=step_in_epoch,
                    avg_loss=avg_loss,
                    experiment_name="c3_original",
                )
                # Create checkpoint data
                ckpt_data = CheckpointData.from_models(
                    models={"encoder": encoder, "decoder": decoder},
                    optimizers={
                        "encoder": encoder_optimizer,
                        "decoder": decoder_optimizer,
                    },
                    schedulers={
                        "encoder": encoder_scheduler,
                        "decoder": decoder_scheduler,
                    },
                    metadata=ckpt_metadata,
                    extra={"total_steps": len(history)},
                )
                ckpt_data.save(checkpoint_dir / ckpt_name)
                logger.info(f"    [Checkpoint saved: {ckpt_name}]")

        # Epoch summary
        avg_epoch_loss = epoch_loss / num_batches
        logger.log_epoch(epoch + 1, avg_epoch_loss, num_epochs)

        # Save checkpoint at end of each epoch
        step_in_epoch = num_batches
        ckpt_name = (
            f"checkpoint-epoch{epoch+1}-step{step_in_epoch}-global{global_step}.pt"
        )
        # Create checkpoint metadata
        ckpt_metadata = CheckpointMetadata(
            epoch=epoch + 1,
            global_step=global_step,
            step_in_epoch=step_in_epoch,
            avg_loss=avg_epoch_loss,
            experiment_name="c3_original",
        )
        ckpt_data = CheckpointData.from_models(
            models={"encoder": encoder, "decoder": decoder},
            optimizers={"encoder": encoder_optimizer, "decoder": decoder_optimizer},
            schedulers={"encoder": encoder_scheduler, "decoder": decoder_scheduler},
            metadata=ckpt_metadata,
            extra={"total_steps": len(history)},
        )
        ckpt_data.save(checkpoint_dir / ckpt_name)
        logger.info(f"    Checkpoint saved: {checkpoint_dir / ckpt_name}")

    logger.log_header("Training completed!")

    # Save final checkpoint
    final_ckpt = checkpoint_dir / "checkpoint_final.pt"
    ckpt_metadata = CheckpointMetadata(
        epoch=epoch + 1,
        global_step=global_step,
        step_in_epoch=step_in_epoch,
        avg_loss=avg_epoch_loss,
        experiment_name="c3_original",
    )
    ckpt_data = CheckpointData.from_models(
        models={"encoder": encoder, "decoder": decoder},
        optimizers={"encoder": encoder_optimizer, "decoder": decoder_optimizer},
        schedulers={"encoder": encoder_scheduler, "decoder": decoder_scheduler},
        metadata=ckpt_metadata,
        extra={"total_steps": len(history)},
    )
    ckpt_data.save(final_ckpt)
    logger.info(f"Final checkpoint saved: {final_ckpt}")

    # Flush and persist training history to disk
    history.flush()
    logger.info(f"Training history saved: {history_path} ({len(history)} steps)")
    logger.info("")

    # =================================================================
    # Evaluation: Sample reconstruction
    # =================================================================
    logger.info("[8] Sample reconstruction...")

    encoder.eval()
    decoder.eval()

    # Get a sample batch
    sample_texts = collate_fn_text([dataset[i] for i in range(min(2, len(dataset)))])

    with torch.no_grad():
        # Use autocast for inference to match training dtype
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=bf16):
            # Encode
            latent_tokens = encoder(inputs=sample_texts)
            logger.info(f"    Input texts: {len(sample_texts[0])} chars")
            logger.info(f"    Latent tokens shape: {latent_tokens.shape}")
            logger.info(f"    Compression: {len(sample_texts[0])} chars -> {N} tokens")

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
    logger.info("")
    logger.info("    Sample 1:")
    logger.info(f"      Original:      {sample_texts[0][:100]}...")
    pred_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    logger.info(f"      Reconstructed: {pred_text[:100]}...")

    if len(sample_texts) > 1:
        logger.info("")
        logger.info("    Sample 2:")
        logger.info(f"      Original:      {sample_texts[1][:100]}...")
        pred_text = tokenizer.decode(output_ids[1], skip_special_tokens=True)
        logger.info(f"      Reconstructed: {pred_text[:100]}...")

    logger.info("")
    logger.log_header("ALL DONE")


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
    train_c3(config)
