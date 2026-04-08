"""C3 Context Cascade Compression - DeepSpeed Training.

Usage:
    torchrun --nproc_per_node=4 examples/c3/train_c3_ds.py \
        --config configs/c3/config.yaml \
        --deepspeed configs/c3/zero2.json

DeepSpeed Features:
    - ZeRO-2: Shards optimizer states across GPUs
    - BF16 mixed precision (no GradScaler needed)
    - Gradient accumulation handled automatically
    - Gradient checkpointing via config

Output Structure:
    EXPERIMENT/c3/
    ├── checkpoints/
    │   ├── global_step_{N}/           # DeepSpeed ZeRO-2 checkpoint
    │   │   ├── mp_rank_00_model_states.pt          # Full model weights
    │   │   └── bf16_zero_pp_rank_*_optim_states.pt # Sharded optimizer states
    │   └── latest -> global_step_{N}
    └── logs/
        ├── training.log
        ├── train_config.json
        ├── training_history.json
        └── terminal_output.json       # Auto-captured terminal output

Checkpoint Format:
    DeepSpeed ZeRO-2 saves:
    - mp_rank_00_model_states.pt: Complete model weights (for inference)
    - bf16_zero_pp_rank_{0,1,2,3}_optim_states.pt: Optimizer states per rank

    For inference/visualization, only model_states.pt is needed.
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
from model import C3Model
from ram import (
    ReconstructionSample,
    ReconstructionSampleStore,
    TrainingConfig,
    TrainingHistory,
    TrainingLogger,
    TrainingStep,
    create_reconstruction_samples,
)
from ram.evaluation import evaluate_reconstruction
from ram.utils import (
    collate_fn_text,
    decode_logits_to_text,
    load_config,
    setup_environment,
)


class TeeLogger:
    """Captures stdout/stderr and writes to both terminal and JSON file.

    Features:
        - Real-time terminal output (no buffering delay)
        - Periodic JSON save (every N lines) to prevent data loss on crash
        - Final save on stop() for complete capture
    """

    def __init__(self, json_path: str | Path, rank: int = 0, save_interval: int = 100):
        self.json_path = Path(json_path)
        self.rank = rank
        self.save_interval = save_interval
        self.original_stdout = None
        self.original_stderr = None
        self.buffer = []
        self._started = False
        self._lines_since_save = 0

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
        self._save_to_json(final=True)
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
            self._lines_since_save += 1
            # Periodic save to prevent data loss on crash
            if self._lines_since_save >= self.save_interval:
                self._save_to_json(final=False)
                self._lines_since_save = 0

    def flush(self):
        """Flush the stream."""
        if self.original_stdout:
            self.original_stdout.flush()

    def _save_to_json(self, final: bool = False):
        """Save buffered output to JSON file.

        Args:
            final: If True, this is the final save (training complete)
        """
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
                        "final": final,
                    },
                    "logs": self.buffer,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )


def train_c3(config: dict, ds_config: dict, tee_logger: TeeLogger | None = None):
    """Train C3 with DeepSpeed for multi-GPU training."""
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
    warmup_ratio = train_cfg["warmup_ratio"]
    use_checkpointing = train_cfg["gradient"]["checkpointing"]
    resume = train_cfg["resume"]

    log_interval = log_cfg["log_step_interval"]
    checkpoint_interval = log_cfg["checkpoint_step_interval"]
    block_size = log_cfg["block_size"]

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
            name="c3_train",
            log_file=log_dir / "training.log",
        )
        logger.log_header("C3 Context Cascade Compression - DeepSpeed Training")
    else:
        logger = None

    N = model_cfg["encoder"]["latent_token_len"]
    M = model_cfg["encoder"]["max_length"]

    setup_environment({"seed": env_cfg["seed"], "device": "cpu"})

    # Build unified C3 model
    if is_main_process:
        logger.info("[1] Building C3 unified model...")

    c3_model = C3Model(model_cfg)
    c3_model = c3_model.to(device)

    if use_checkpointing:
        c3_model.gradient_checkpointing_enable()
        if is_main_process:
            logger.info("    Gradient checkpointing: ENABLED")

    D_enc = c3_model.encoder_hidden_dim
    D_dec = c3_model.decoder_hidden_dim
    V = c3_model.vocab_size

    if is_main_process:
        logger.info(f"    Encoder: {c3_model.encoder.model_name}")
        logger.info(f"    Encoder hidden_dim: {D_enc}")
        logger.info(f"    Decoder: {c3_model.decoder.model_name}")
        logger.info(f"    Decoder hidden_dim: {D_dec}")
        logger.info(f"    Vocab size: {V}")
        logger.info(f"    Latent tokens: {N}")
        logger.info(f"    mm_projector: {D_enc} -> {D_dec}")
        logger.info("")

    tokenizer = c3_model.tokenizer

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
        logger.info(
            f"    Gradient accumulation steps: {ds_config['gradient_accumulation_steps']}"
        )
        logger.info(f"    Effective batch size: {effective_batch_size}")
        logger.info("")

    # Initialize DeepSpeed
    if is_main_process:
        logger.info("[3] Initializing DeepSpeed...")

    model_engine, _, _, _ = deepspeed.initialize(
        model=c3_model,
        model_parameters=c3_model.parameters(),
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
    samples_store = None
    samples_dir = log_dir / "samples"

    if is_main_process:
        # Calculate effective batch size: per_gpu × grad_acc × world_size
        effective_batch_size = (
            per_device_batch_size
            * ds_config["gradient_accumulation_steps"]
            * world_size
        )
        training_config = TrainingConfig(
            experiment_name="c3",
            batch_size=effective_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_epochs=num_epochs,
            warmup_ratio=warmup_ratio,
            gradient_accumulation_steps=ds_config["gradient_accumulation_steps"],
            gradient_clip=ds_config["gradient_clipping"],
            bf16=ds_config["bf16"]["enabled"],
            latent_token_len=N,
            max_length=M,
            compression_ratio=M / N,
        )

        config_path = log_dir / "train_config.json"
        training_config.save(config_path)
        logger.info(f"    Training config saved: {config_path}")

        history_path = log_dir / "training_history.json"
        history = TrainingHistory(training_config, history_path)

        # Setup reconstruction sample store (block-based storage)
        samples_store = ReconstructionSampleStore(
            folder=str(samples_dir),
            block_size=block_size,
        )
        logger.info(f"    Samples store: {samples_dir}")
        logger.info("")

    if resume:
        latest_ckpt_dir = checkpoint_dir / "latest"
        ckpt_to_load = None

        # Find latest checkpoint
        if latest_ckpt_dir.exists() and latest_ckpt_dir.is_symlink():
            ckpt_to_load = os.readlink(latest_ckpt_dir)
        else:
            # Fallback: find latest global_step checkpoint
            ckpt_dirs = sorted(checkpoint_dir.glob("global_step_*"))
            if ckpt_dirs:
                ckpt_to_load = ckpt_dirs[-1].name

        if ckpt_to_load:
            if is_main_process:
                logger.info(f"[3.5] Resuming from: {ckpt_to_load}")

            _, _, _, client_state = model_engine.load_checkpoint(
                checkpoint_dir=str(checkpoint_dir),
                tag=ckpt_to_load,
            )

            if client_state is not None:
                start_epoch = client_state.get("epoch", 0)
                global_step = client_state.get("global_step", 0)

                # Restore training history if available
                if is_main_process and history is not None:
                    history_data = client_state["training_history"]
                    if history_data:
                        # Reconstruct TrainingStep objects from dicts
                        # Same pattern as TrainingHistory.load()
                        history.steps = []
                        for s in history_data:
                            samples = [
                                ReconstructionSample(**rs)
                                for rs in s.pop("reconstruction_samples", [])
                            ]
                            step = TrainingStep(**s)
                            step.reconstruction_samples = samples
                            history.steps.append(step)
                        logger.info(f"    Restored {len(history_data)} history entries")

            if is_main_process:
                logger.info(f"    Resumed from epoch {start_epoch}, step {global_step}")
                logger.info("")
        else:
            if is_main_process:
                logger.info("[3.5] No checkpoint found, starting from scratch")
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
                context_texts=batch_texts,
                target_texts=batch_texts,
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
                assert samples_store is not None

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

                # Generate reconstruction samples for inspection
                # Use training logits for reconstruction (teacher forcing output)
                # logits: [B, N+L, V] -> skip N latent tokens -> [B, L, V]
                with torch.no_grad():
                    # Get attention mask from tokenizer
                    encoded = tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=M,
                    )
                    attention_mask = encoded["attention_mask"].to(device)

                    # Skip latent token positions for decoding
                    # text_logits: [B, L, V]
                    text_logits = logits[:, N:, :]
                    decode_result = decode_logits_to_text(
                        text_logits, tokenizer, batch_texts, attention_mask
                    )

                # Create reconstruction samples and add to step record
                recon_samples = create_reconstruction_samples(decode_result)
                step_record.reconstruction_samples = recon_samples

                # Evaluate reconstruction quality for first sample in batch
                sample_metrics = None
                if recon_samples and is_main_process:
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

                # Save samples to block-based store for alignment with training history
                # Include metrics in the saved record
                sample_key = samples_store.save_samples(
                    step_record, recon_samples, metrics=sample_metrics
                )

                history.append(step_record)
                logger.log_step(step_record, log_interval=1)
                logger.info(f"    [Step logged: global_step_{global_step}]")
                logger.info(f"    [Samples saved: {sample_key}]")

            # Save checkpoint at intervals
            if global_step % checkpoint_interval == 0:
                client_state = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "step_in_epoch": num_batches,
                    "training_history": (
                        [s.to_dict() for s in history.steps]
                        if (is_main_process and history)
                        else []
                    ),
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

        # Save checkpoint at epoch end (all ranks must participate for DeepSpeed)
        client_state = {
            "epoch": epoch,
            "global_step": global_step,
            "step_in_epoch": num_batches,
            "training_history": (
                [s.to_dict() for s in history.steps]
                if (is_main_process and history)
                else []
            ),
        }
        model_engine.save_checkpoint(
            save_dir=str(checkpoint_dir),
            tag=f"epoch_{epoch+1}",
            client_state=client_state,
        )
        if is_main_process:
            logger.info(f"    [Checkpoint saved: epoch_{epoch+1}]")

    # Training complete
    if is_main_process:
        logger.log_header("Training completed!")

    # Save final checkpoint (all ranks must participate for DeepSpeed)
    client_state = {
        "epoch": num_epochs,
        "global_step": global_step,
        "training_history": (
            [s.to_dict() for s in history.steps]
            if (is_main_process and history)
            else []
        ),
    }
    model_engine.save_checkpoint(
        save_dir=str(checkpoint_dir),
        tag="final",
        client_state=client_state,
    )

    if is_main_process:
        logger.info(f"Final checkpoint saved: {checkpoint_dir / 'final'}")
        history.flush()
        logger.info(f"Training history saved: {history_path}")
        logger.log_header("ALL DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="C3 Context Cascade Compression - DeepSpeed Training"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., examples/c3/config.yaml)",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        required=True,
        help="Path to DeepSpeed config file (e.g., examples/c3/zero2.json)",
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
        train_c3(config, ds_config, tee_logger)
    finally:
        tee_logger.stop()
