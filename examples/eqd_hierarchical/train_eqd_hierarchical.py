"""EQD Hierarchical - Basic Training.

Usage:
    python examples/ed/train_ed.py -c configs/ed/config.yaml

Output Structure:
    EXPERIMENT/ed/
    ├── checkpoints/
    │   ├── checkpoint-epoch{N}-step{S}.pt
    │   └── checkpoint_final.pt
    └── logs/
        ├── training.log
        └── train_config.json
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from lmbase.dataset import registry
from model import EQDHierarchicalModel
from ram import (
    TrainingConfig,
    TrainingHistory,
    TrainingLogger,
    TrainingStep,
)
from ram.utils import (
    collate_fn_text,
    load_config,
    setup_environment,
)


def train_eqd_hierarchical(config: dict):
    """Train ED (basic single-GPU training)."""
    # Extract config
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
    warmup_steps = train_cfg.get("warmup_steps", 100)
    gradient_clip = train_cfg["gradient"]["max_grad_norm"]
    use_checkpointing = train_cfg["gradient"]["checkpointing"]
    resume = train_cfg["resume"]

    log_interval = log_cfg["log_step_interval"]
    checkpoint_interval = log_cfg["checkpoint_step_interval"]

    output_dir = Path(log_cfg["save_folder"])
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup environment
    device = setup_environment(env_cfg)

    # Logger
    logger = TrainingLogger(
        name="eqd_hierarchical_train",
        log_file=log_dir / "training.log",
    )
    logger.log_header("EQD Hierarchical - Basic Training")

    # Build model
    logger.info("[1] Building ED model...")
    model = EQDHierarchicalModel(model_cfg)
    model = model.to(device)

    if use_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("    Gradient checkpointing: ENABLED")

    logger.info(f"    Encoder: {model.encoder_model_name}")
    logger.info(f"    Decoder: {model.decoder_model_name}")
    logger.info(f"    Hidden dim: {model.hidden_dim}")
    logger.info(f"    Vocab size: {model.vocab_size}")
    logger.info("")

    # Setup optimizer and scheduler
    logger.info("[2] Setting up optimizer...")
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    logger.info(f"    Optimizer: AdamW, lr={learning_rate}")
    logger.info("")

    # Load data
    logger.info("[3] Loading dataset...")
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
    logger.info(f"    Batch size: {batch_size}")
    logger.info("")

    # Training config
    training_config = TrainingConfig(
        experiment_name="ed",
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        gradient_clip=gradient_clip,
        bf16=train_cfg.get("bf16", True),
    )
    config_path = log_dir / "train_config.json"
    training_config.save(config_path)

    history_path = log_dir / "training_history.json"
    history = TrainingHistory(training_config, history_path)

    # Resume
    start_epoch = 0
    global_step = 0
    if resume:
        # TODO: implement checkpoint loading
        pass

    # Training loop
    logger.log_subheader("[4] Starting training...")
    model.train()

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_texts in pbar:
            # Forward
            vq_loss_weight = train_cfg.get("vq_loss_weight", 1.0)
            logits, loss, vq_loss = model(
                texts=batch_texts,
                compute_loss=True,
                vq_loss_weight=vq_loss_weight,
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            # Logging
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg": f"{epoch_loss/num_batches:.4f}",
                }
            )

            # Log at intervals
            if global_step % log_interval == 0:
                step_record = TrainingStep(
                    epoch=epoch + 1,
                    step_in_epoch=num_batches,
                    global_step=global_step,
                    total_loss=loss.item(),
                    vq_loss=vq_loss.item() if vq_loss is not None else 0.0,
                    avg_loss=epoch_loss / num_batches,
                    lr_encoder=optimizer.param_groups[0]["lr"],
                    lr_decoder=optimizer.param_groups[0]["lr"],
                )
                history.append(step_record)
                logger.log_step(step_record)

        scheduler.step()

        # Epoch summary
        avg_epoch_loss = epoch_loss / num_batches
        logger.log_epoch(epoch + 1, avg_epoch_loss, num_epochs)

    # Save final
    logger.log_header("Training completed!")
    final_ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": num_epochs,
        "global_step": global_step,
    }
    torch.save(final_ckpt, checkpoint_dir / "checkpoint_final.pt")
    logger.info(f"Final checkpoint saved: {checkpoint_dir / 'checkpoint_final.pt'}")
    history.flush()
    logger.log_header("ALL DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ED - Basic Training")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to config file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train_ed(config)
