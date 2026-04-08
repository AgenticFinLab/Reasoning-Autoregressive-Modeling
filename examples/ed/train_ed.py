"""ED (Encoder-Decoder) - Basic Training.

Usage:
    python examples/ed/train_ed.py -c configs/ed/config.yaml

Output Structure:
    EXPERIMENT/ed/
    ├── checkpoints/
    │   ├── checkpoint-epoch{N}-start.pt
    │   ├── checkpoint-epoch{N}-step{S}-global{G}.pt
    │   └── checkpoint_final.pt
    └── logs/
        ├── training.log
        ├── train_config.json
        ├── training_history.json
        └── samples/
            ├── block_0.json
            └── ...

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Text Input                                                         │
    │      │                                                              │
    │      ▼                                                              │
    │  ┌─────────────────┐           ┌─────────────────┐                  │
    │  │ Encoder         │           │ Decoder         │                  │
    │  │ text -> hidden  │ ────────▶ │ hidden -> logits│                  │
    │  │ (BERT-style)    │  transfer │ (GPT-style)     │                  │
    │  └─────────────────┘           └─────────────────┘                  │
    │                        hidden [B, L, D]                              │
    └─────────────────────────────────────────────────────────────────────┘
"""

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from model import EDModel
from ram import (
    CheckpointData,
    CheckpointMetadata,
    RamDataLoaderRegistry,
    RamSample,
    ReconstructionSampleStore,
    TrainingConfig,
    TrainingHistory,
    TrainingLogger,
    TrainingStep,
    create_reconstruction_samples,
)
from ram.evaluation import evaluate_reconstruction
from ram.utils import (
    decode_logits_to_text,
    find_latest_checkpoint,
    load_config,
    resume_from_checkpoint,
    save_checkpoint,
    setup_environment,
)


def train_ed(config: Dict) -> None:
    """Train ED (Encoder-Decoder) for text reconstruction.

    Training Flow:
        Step 1: Tokenize texts -> input_ids [B, L]
        Step 2: texts -> Encoder -> hidden [B, L, D_enc]
        Step 3: hidden -> Decoder -> logits [B, L, V]
        Step 4: logits + texts -> CrossEntropyLoss -> loss
        Step 5: loss.backward() -> optimizer.step() -> update weights

    Args:
        config: Training configuration dictionary
    """
    # =================================================================
    # Extract config
    # =================================================================
    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    env_cfg = config["environment"]
    log_cfg = config["log"]

    # Training hyperparameters
    batch_size = train_cfg["batch_size"]
    learning_rate = train_cfg["learning_rate"]
    weight_decay = train_cfg["weight_decay"]
    num_epochs = train_cfg["num_epochs"]
    warmup_ratio = train_cfg["warmup_ratio"]
    gradient_clip = train_cfg["gradient"]["max_grad_norm"]
    gradient_accumulation_steps = train_cfg["gradient"]["accumulation_steps"]
    bf16 = train_cfg["bf16"]
    use_checkpointing = train_cfg["gradient"]["checkpointing"]
    resume = train_cfg["resume"]

    # Logging intervals
    log_interval = log_cfg["log_step_interval"]
    checkpoint_interval = log_cfg["checkpoint_step_interval"]
    block_size = log_cfg["block_size"]

    output_dir = Path(log_cfg["save_folder"])
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # =================================================================
    # Setup environment
    # =================================================================
    device = setup_environment(env_cfg)

    # =================================================================
    # Setup unified logging
    # =================================================================
    logger = TrainingLogger(
        name="ed_train",
        log_file=log_dir / "training.log",
    )
    logger.log_header("ED (Encoder-Decoder) - Basic Training")

    # =================================================================
    # Build model
    # =================================================================
    logger.info("[1] Building ED model...")
    model = EDModel(model_cfg)
    model = model.to(device)

    if use_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("    Gradient checkpointing: ENABLED")

    logger.info(f"    Encoder: {model.encoder_model_name}")
    logger.info(f"    Decoder: {model.decoder_model_name}")
    logger.info(f"    Hidden dim: {model.hidden_dim}")
    logger.info(f"    Vocab size: {model.vocab_size}")
    logger.info("")

    # =================================================================
    # Setup tokenizer for reconstruction evaluation
    # =================================================================
    logger.info("[2] Setting up tokenizer...")
    tokenizer = model.dec_tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"    pad_token_id: {tokenizer.pad_token_id}")
    logger.info("")

    # =================================================================
    # Load data
    # =================================================================
    logger.info("[3] Loading dataset...")
    # Use RamDataLoaderRegistry for standardized data loading
    # Default: question + cot_answer combined as target
    dataloader = RamDataLoaderRegistry(
        {
            "data_name": data_cfg["data_name"],
            "data_dir": data_cfg.get("data_dir", ""),
            "split": data_cfg["split"],
            "batch_size": batch_size,
            "num_workers": env_cfg["dataloader_num_workers"],
            "shuffle": True,
            "drop_last": True,
        }
    )
    logger.info(f"    Dataset: {data_cfg['data_name']}")
    logger.info(f"    Target format: question + cot (default)")
    logger.info(f"    Batches per epoch: {len(dataloader)}")
    logger.info("")

    # =================================================================
    # Setup optimizer and scheduler
    # =================================================================
    logger.info("[4] Setting up optimizer...")
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    total_steps = len(dataloader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
    )

    logger.info(f"    Total steps: {total_steps}")
    logger.info(f"    Warmup steps: {warmup_steps}")
    logger.info("    LR scheduler: cosine")
    logger.info("")

    # =================================================================
    # Create TrainingConfig
    # =================================================================
    training_config = TrainingConfig(
        experiment_name="ed",
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_clip=gradient_clip,
        bf16=bf16,
    )
    logger.log_config(training_config)
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
        block_size=block_size,
    )
    logger.info(f"    Samples store: {log_dir / 'samples'}")
    logger.info("")

    if resume:
        latest_ckpt = find_latest_checkpoint(checkpoint_dir)
        if latest_ckpt is not None:
            logger.info(f"[4.5] Resuming from: {latest_ckpt.name}")
            start_epoch, global_step, _ = resume_from_checkpoint(
                checkpoint_path=latest_ckpt,
                models={"model": model},
                optimizer={"model": optimizer},
                scheduler={"model": scheduler},
                device=device,
                log_dir=log_dir,
            )
            logger.info(
                f"    Resumed from epoch {start_epoch+1}, global_step {global_step}"
            )
            logger.info("")
        else:
            logger.info("[4.5] No checkpoint found, starting fresh")
            logger.info("")

    # =================================================================
    # Training loop
    # =================================================================
    logger.log_subheader("[5] Starting training...")
    model.train()

    # Mixed precision scalers
    scaler = torch.amp.GradScaler("cuda", enabled=bf16)
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
                models={"model": model},
                optimizer={"model": optimizer},
                scheduler={"model": scheduler},
                epoch=epoch,
                global_step=global_step,
                extra_info={"step_in_epoch": 0, "total_steps": len(history)},
            )
            logger.info(f"    [Epoch-start checkpoint saved: {ckpt_name}]")

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        optimizer.zero_grad()

        for batch_idx, batch_samples in enumerate(pbar):
            # Extract target texts from RamSample batch
            batch_texts = [s.target_text for s in batch_samples]

            # =========================================================
            # Forward pass
            # =========================================================
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=bf16):
                logits, loss = model(
                    texts=batch_texts,
                    compute_loss=True,
                )

            # Tokenize for reconstruction evaluation
            tokens = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=logits.size(1),
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]

            # =========================================================
            # Scale loss for gradient accumulation
            # =========================================================
            loss = loss / gradient_accumulation_steps

            # =========================================================
            # Backward
            # =========================================================
            scaler.scale(loss).backward()

            # =========================================================
            # Optimizer step (with gradient accumulation)
            # =========================================================
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Update learning rate
                if global_step >= warmup_steps:
                    scheduler.step()

            # Logging
            unscaled_loss = loss.item() * gradient_accumulation_steps
            epoch_loss += unscaled_loss
            num_batches += 1
            global_step += 1

            # Record history with TrainingStep
            step_record = TrainingStep(
                epoch=epoch + 1,
                step_in_epoch=num_batches,
                global_step=global_step,
                total_loss=unscaled_loss,
                recon_loss=unscaled_loss,
                avg_loss=epoch_loss / num_batches,
                lr_encoder=optimizer.param_groups[0]["lr"],
                lr_decoder=optimizer.param_groups[0]["lr"],
            )
            history.append(step_record)

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{unscaled_loss:.4f}",
                    "avg": f"{epoch_loss/num_batches:.4f}",
                }
            )

            # --- Log at log_interval ---
            if global_step % log_interval == 0:
                logger.log_step(step_record, log_interval=1)

                # --- Save reconstruction samples ---
                with torch.no_grad():
                    decode_result = decode_logits_to_text(
                        logits, tokenizer, batch_texts, attention_mask
                    )

                # Create reconstruction samples and add to step record
                recon_samples = create_reconstruction_samples(decode_result)
                step_record.reconstruction_samples = recon_samples

                # Evaluate reconstruction quality for first sample in batch
                sample_metrics = None
                if recon_samples:
                    sample = recon_samples[0]
                    sample_metrics = evaluate_reconstruction(
                        original_text=sample.original,
                        reconstructed_text=sample.reconstructed,
                        tokenizer=tokenizer,
                    )
                    # Log key metrics
                    logger.info(
                        f"    [Metrics: token_precision={sample_metrics.get('token_precision', 0):.2%}, "
                        f"char_precision={sample_metrics['char_precision']:.2%}, "
                        f"bleu={sample_metrics['bleu_score']:.3f}]"
                    )

                # Save samples to block-based store
                sample_key = samples_store.save_samples(
                    step_record, recon_samples, metrics=sample_metrics
                )
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
                    experiment_name="ed",
                )
                # Create checkpoint data
                ckpt_data = CheckpointData.from_models(
                    models={"model": model},
                    optimizers={"model": optimizer},
                    schedulers={"model": scheduler},
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
        ckpt_metadata = CheckpointMetadata(
            epoch=epoch + 1,
            global_step=global_step,
            step_in_epoch=step_in_epoch,
            avg_loss=avg_epoch_loss,
            experiment_name="ed",
        )
        ckpt_data = CheckpointData.from_models(
            models={"model": model},
            optimizers={"model": optimizer},
            schedulers={"model": scheduler},
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
        experiment_name="ed",
    )
    ckpt_data = CheckpointData.from_models(
        models={"model": model},
        optimizers={"model": optimizer},
        schedulers={"model": scheduler},
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
    logger.info("[6] Sample reconstruction...")

    model.eval()

    # Get a sample batch from dataloader
    sample_batch = next(iter(dataloader))[:2]  # Get first 2 RamSample objects
    sample_texts = [s.target_text for s in sample_batch]

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=bf16):
            logits, _ = model(texts=sample_texts, compute_loss=False)

            # Decode predictions
            decode_result = decode_logits_to_text(
                logits,
                tokenizer,
                sample_texts,
                tokenizer(
                    sample_texts,
                    padding=True,
                    truncation=True,
                    max_length=logits.size(1),
                    return_tensors="pt",
                )["attention_mask"],
            )

    logger.info("")
    logger.info("    Sample 1:")
    logger.info(f"      Original:      {sample_texts[0][:100]}...")
    logger.info(
        f"      Reconstructed: {decode_result['reconstructed_texts'][0][:100]}..."
    )

    if len(sample_texts) > 1:
        logger.info("")
        logger.info("    Sample 2:")
        logger.info(f"      Original:      {sample_texts[1][:100]}...")
        logger.info(
            f"      Reconstructed: {decode_result['reconstructed_texts'][1][:100]}..."
        )

    logger.info("")
    logger.log_header("ALL DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ED - Basic Training")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train_ed(config)
