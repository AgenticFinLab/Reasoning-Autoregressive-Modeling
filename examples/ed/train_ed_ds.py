"""ED (Encoder-Decoder) - DeepSpeed Training.

Usage:
    torchrun --nproc_per_node=4 examples/ed/train_ed_ds.py \
        --config configs/ed/config.yaml \
        --deepspeed configs/ed/zero2.json

DeepSpeed Features:
    - ZeRO-2: Shards optimizer states across GPUs
    - BF16 mixed precision (no GradScaler needed)
    - Gradient accumulation handled automatically
    - Gradient checkpointing via config

Output Structure:
    EXPERIMENT/ed/
    ├── checkpoints/
    │   ├── global_step_{N}/           # DeepSpeed checkpoint format
    │   └── latest -> global_step_{N}
    └── logs/
        ├── training.log
        ├── train_config.json
        ├── training_history.json
        └── terminal_output.json       # Auto-captured terminal output
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import deepspeed
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from lmbase.dataset import registry
from model import EDModel
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


class TeeLogger:
    """Captures stdout/stderr and writes to both terminal and JSON file."""

    def __init__(self, json_path: str | Path, rank: int = 0):
        self.json_path = Path(json_path)
        self.rank = rank
        self.original_stdout = None
        self.original_stderr = None
        self.buffer = []
        self._started = False

    def start(self):
        """Start capturing output."""
        if self._started:
            return
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
        self._started = True

    def stop(self):
        """Stop capturing and save to JSON."""
        if not self._started:
            return
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self._save_to_json()
        self._started = False

    def write(self, message: str):
        """Write to both terminal and buffer."""
        if self.original_stdout:
            self.original_stdout.write(message)
            self.original_stdout.flush()
        if self.rank == 0 and message.strip():
            self.buffer.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "message": message,
                }
            )

    def flush(self):
        """Flush the stream."""
        if self.original_stdout:
            self.original_stdout.flush()

    def _save_to_json(self):
        """Save buffered output to JSON file."""
        if self.rank != 0 or not self.buffer:
            return
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": {
                        "created_at": datetime.now().isoformat(),
                        "rank": self.rank,
                        "total_lines": len(self.buffer),
                    },
                    "logs": self.buffer,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )


def train_ed(config: dict, ds_config: dict, tee_logger: TeeLogger | None = None):
    """Train ED with DeepSpeed for multi-GPU training."""
    # Initialize DeepSpeed distributed
    deepspeed.init_distributed()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    is_main_process = rank == 0

    # Update tee_logger with correct rank
    if tee_logger is not None:
        tee_logger.rank = rank

    # Extract config
    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    env_cfg = config["environment"]
    log_cfg = config["log"]

    num_epochs = train_cfg["num_epochs"]
    learning_rate = train_cfg["learning_rate"]
    weight_decay = train_cfg["weight_decay"]
    warmup_steps = train_cfg.get("warmup_steps", 100)
    use_checkpointing = train_cfg["gradient"]["checkpointing"]
    resume = train_cfg["resume"]

    log_interval = log_cfg["log_step_interval"]
    checkpoint_interval = log_cfg["checkpoint_step_interval"]

    output_dir = Path(log_cfg["save_folder"])
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"

    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    if is_main_process:
        logger = TrainingLogger(
            name="ed_train",
            log_file=log_dir / "training.log",
        )
        logger.log_header("ED (Encoder-Decoder) - DeepSpeed Training")
    else:
        logger = None

    setup_environment({"seed": env_cfg["seed"], "device": "cpu"})

    # Build unified ED model
    if is_main_process:
        logger.info("[1] Building ED unified model...")

    ed_model = EDModel(model_cfg)
    ed_model = ed_model.to(device)

    if use_checkpointing:
        ed_model.gradient_checkpointing_enable()
        if is_main_process:
            logger.info("    Gradient checkpointing: ENABLED")

    D_enc = ed_model.hidden_dim
    V = ed_model.vocab_size

    if is_main_process:
        logger.info(f"    Encoder: {ed_model.encoder_model_name}")
        logger.info(f"    Encoder hidden_dim: {D_enc}")
        logger.info(f"    Decoder: {ed_model.decoder_model_name}")
        logger.info(f"    Decoder vocab_size: {V}")
        logger.info("")

    if is_main_process:
        logger.info("[2] Loading dataset...")

    per_device_batch_size = ds_config["train_micro_batch_size_per_gpu"]
    dataset = registry.get(data_cfg, split=data_cfg["split"])

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        sampler=sampler,
        collate_fn=collate_fn_text,
        drop_last=True,
        num_workers=env_cfg["dataloader_num_workers"],
    )

    if is_main_process:
        effective_batch_size = (
            per_device_batch_size
            * ds_config["gradient_accumulation_steps"]
            * world_size
        )
        logger.info(f"    Dataset: {data_cfg['data_name']}, {len(dataset)} samples")
        logger.info(f"    World size: {world_size}")
        logger.info(f"    Per-device batch size: {per_device_batch_size}")
        logger.info(f"    Gradient accumulation steps: {ds_config['gradient_accumulation_steps']}")
        logger.info(f"    Effective batch size: {effective_batch_size}")
        logger.info("")

    # Initialize DeepSpeed
    if is_main_process:
        logger.info("[3] Initializing DeepSpeed...")

    model_engine, _, _, _ = deepspeed.initialize(
        model=ed_model,
        model_parameters=ed_model.parameters(),
        config=ds_config,
    )

    if is_main_process:
        logger.info(f"    DeepSpeed ZeRO-{ds_config['zero_optimization']['stage']}")
        logger.info(f"    BF16 enabled: {ds_config['bf16']['enabled']}")
        logger.info("")

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0

    history = None

    if is_main_process:
        training_config = TrainingConfig(
            experiment_name="ed",
            batch_size=per_device_batch_size * ds_config["gradient_accumulation_steps"] * world_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_epochs=num_epochs,
            warmup_steps=warmup_steps,
            gradient_accumulation_steps=ds_config["gradient_accumulation_steps"],
            gradient_clip=ds_config["gradient_clipping"],
            bf16=ds_config["bf16"]["enabled"],
        )

        config_path = log_dir / "train_config.json"
        training_config.save(config_path)
        logger.info(f"    Training config saved: {config_path}")

        history_path = log_dir / "training_history.json"
        history = TrainingHistory(training_config, history_path)
        logger.info("")

    if resume:
        latest_ckpt_dir = checkpoint_dir / "latest"
        if latest_ckpt_dir.exists() and latest_ckpt_dir.is_symlink():
            ckpt_path = os.readlink(latest_ckpt_dir)
            if is_main_process:
                logger.info(f"[3.5] Resuming from: {ckpt_path}")

            _, _, _, client_state = model_engine.load_checkpoint(
                checkpoint_dir=str(checkpoint_dir),
                tag=ckpt_path,
            )

            if client_state is not None:
                start_epoch = client_state.get("epoch", 0)
                global_step = client_state.get("global_step", 0)

            if is_main_process:
                logger.info(f"    Resumed from epoch {start_epoch}, step {global_step}")
                logger.info("")

    # Training loop
    if is_main_process:
        logger.log_subheader("[4] Starting training...")

    model_engine.train()

    for epoch in range(start_epoch, num_epochs):
        sampler.set_epoch(epoch)

        epoch_loss = 0.0
        num_batches = 0

        if is_main_process:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        else:
            pbar = dataloader

        for _, batch_texts in enumerate(pbar):
            # Forward pass
            logits, loss = model_engine(
                texts=batch_texts,
                compute_loss=True,
            )

            loss_dict = {
                "total_loss": loss.item(),
                "recon_loss": loss.item(),
            }

            # Backward and optimizer step
            model_engine.backward(loss)
            model_engine.step()

            # Logging
            epoch_loss += loss_dict["total_loss"]
            num_batches += 1
            global_step += 1

            if is_main_process:
                pbar.set_postfix(
                    {
                        "loss": f"{loss_dict['total_loss']:.4f}",
                        "avg": f"{epoch_loss/num_batches:.4f}",
                    }
                )

            # Log at intervals
            if global_step % log_interval == 0 and is_main_process:
                assert history is not None

                step_record = TrainingStep(
                    epoch=epoch + 1,
                    step_in_epoch=num_batches,
                    global_step=global_step,
                    total_loss=loss_dict["total_loss"],
                    recon_loss=loss_dict["recon_loss"],
                    avg_loss=epoch_loss / num_batches,
                    lr_encoder=model_engine.get_lr()[0],
                    lr_decoder=model_engine.get_lr()[0],
                )
                history.append(step_record)
                logger.log_step(step_record, log_interval=1)
                logger.info(f"    [Step logged: global_step_{global_step}]")

            # Save checkpoint at intervals
            if global_step % checkpoint_interval == 0:
                client_state = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "step_in_epoch": num_batches,
                }
                model_engine.save_checkpoint(
                    save_dir=str(checkpoint_dir),
                    tag=f"global_step_{global_step}",
                    client_state=client_state,
                )
                if is_main_process:
                    logger.info(f"    [Checkpoint saved: global_step_{global_step}]")

        # Epoch summary
        avg_epoch_loss = epoch_loss / num_batches
        if is_main_process:
            assert history is not None
            logger.log_epoch(epoch + 1, avg_epoch_loss, num_epochs)

            client_state = {
                "epoch": epoch,
                "global_step": global_step,
                "step_in_epoch": num_batches,
            }
            model_engine.save_checkpoint(
                save_dir=str(checkpoint_dir),
                tag=f"epoch_{epoch+1}",
                client_state=client_state,
            )

    # Training complete
    if is_main_process:
        logger.log_header("Training completed!")

        client_state = {
            "epoch": num_epochs,
            "global_step": global_step,
        }
        model_engine.save_checkpoint(
            save_dir=str(checkpoint_dir),
            tag="final",
            client_state=client_state,
        )
        logger.info(f"Final checkpoint saved: {checkpoint_dir / 'final'}")

        history.flush()
        logger.info(f"Training history saved: {history_path}")
        logger.log_header("ALL DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ED (Encoder-Decoder) - DeepSpeed Training"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., configs/ed/config.yaml)",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        required=True,
        help="Path to DeepSpeed config file (e.g., configs/ed/zero2.json)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local rank for distributed training (set by torchrun)",
    )
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    with open(args.deepspeed, "r", encoding="utf-8") as f:
        ds_config = json.load(f)

    # Auto-capture terminal output
    log_folder = Path(config["log"]["save_folder"])
    capture_path = log_folder / "logs" / "terminal_output.json"
    tee_logger = TeeLogger(capture_path, rank=0)
    tee_logger.start()

    try:
        train_ed(config, ds_config, tee_logger)
    finally:
        tee_logger.stop()
