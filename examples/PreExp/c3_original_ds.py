"""C3 Context Cascade Compression - DeepSpeed Training.

Usage:
    # Single node, 2 GPUs
    torchrun --nproc_per_node=2 examples/PreExp/c3_original_ds.py \
        -c configs/PreExp/c3_original_ds.yml \
        --deepspeed configs/PreExp/zero2.json

    # Multi-node (e.g., 2 nodes, 8 GPUs each)
    torchrun --nnodes=2 --nproc_per_node=8 \
        --node_rank=0 --master_addr="10.0.0.1" --master_port=29500 \
        examples/PreExp/c3_original_ds.py \
        -c configs/PreExp/c3_original_ds.yml \
        --deepspeed configs/PreExp/zero2.json

DeepSpeed Features:
    - ZeRO-2: Shards optimizer states across GPUs
    - BF16 mixed precision (no GradScaler needed)
    - Gradient accumulation handled automatically
    - Gradient checkpointing via config

Output Structure:
    EXPERIMENT/PreExp/c3_original_ds/
    ├── checkpoints/
    │   ├── global_step_{N}/           # DeepSpeed checkpoint format
    │   │   ├── mp_rank_00_model_states.pt
    │   │   └── zero_pp_rank_0_mp_rank_00_optim_states.pt
    │   └── latest -> global_step_{N}
    └── logs/
        ├── training.log
        ├── train_config.json
        └── training_history.json

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

Dimensions:
    B = batch_size
    M = max_length (text sequence length)
    N = latent_token_len (latent token count, official naming)
    D_enc = encoder hidden_dim
    D_dec = decoder hidden_dim
    V = vocab_size
"""

import argparse
import json
import os
from pathlib import Path

import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer

from lmbase.dataset import registry
from ram import (
    ReconstructionSampleStore,
    TrainingConfig,
    TrainingHistory,
    TrainingLogger,
    TrainingStep,
    create_reconstruction_samples,
)
from ram.models.decoder import build_c3_decoder
from ram.models.encoder import build_c3_encoder
from ram.utils import (
    collate_fn_text,
    decode_logits_to_text,
    load_config,
    setup_environment,
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
        """
        _, _, V = logits.shape
        N = latent_token_len

        # Shift for autoregressive prediction
        shift_logits = logits[:, N:-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Mask padding positions
        shift_labels = shift_labels.masked_fill(
            shift_labels == self.ignore_index, self.ignore_index
        )

        # Compute cross-entropy loss
        recon_loss = F.cross_entropy(
            shift_logits.reshape(-1, V),
            shift_labels.reshape(-1),
            ignore_index=self.ignore_index,
        )

        recon_loss_val = recon_loss.item()
        total_loss = recon_loss

        return total_loss, {
            "recon_loss": recon_loss_val,
            "total_loss": recon_loss_val,
        }


def print_rank0(msg, logger=None):
    """Print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            if logger:
                logger.info(msg)
            else:
                print(msg)
    else:
        if logger:
            logger.info(msg)
        else:
            print(msg)


def train_c3_ds(config: dict, ds_config: dict):
    """Train C3 with DeepSpeed for multi-GPU training.

    DeepSpeed handles:
        - Distributed training (DDP)
        - Mixed precision (BF16)
        - Gradient accumulation
        - Optimizer state sharding (ZeRO)
        - Checkpoint management
    """
    # =================================================================
    # Initialize DeepSpeed distributed
    # =================================================================
    deepspeed.init_distributed()

    # Get local rank from environment
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Rank and world size
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    is_main_process = rank == 0

    # =================================================================
    # Extract config
    # =================================================================
    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    env_cfg = config["environment"]
    log_cfg = config["log"]

    enc_cfg = model_cfg["encoder"]
    dec_cfg = model_cfg["decoder"]

    # Training hyperparameters
    num_epochs = train_cfg["num_epochs"]
    learning_rate = train_cfg["learning_rate"]
    weight_decay = train_cfg["weight_decay"]
    warmup_ratio = train_cfg["warmup_ratio"]
    use_checkpointing = train_cfg["gradient"]["checkpointing"]
    resume = train_cfg["resume"]

    # Logging intervals
    log_interval = log_cfg["log_step_interval"]
    checkpoint_interval = log_cfg["checkpoint_step_interval"]

    output_dir = Path(log_cfg["save_folder"])
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"

    # Create output directories (only on main process)
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

    # =================================================================
    # Setup unified logging (only on main process)
    # =================================================================
    if is_main_process:
        logger = TrainingLogger(
            name="c3_train_ds",
            log_file=log_dir / "training.log",
        )
        logger.log_header("C3 Context Cascade Compression - DeepSpeed Training")
    else:
        logger = None

    # =================================================================
    # Dimensions
    # =================================================================
    N = enc_cfg["latent_token_len"]
    M = enc_cfg["max_length"]

    # Setup environment (seed)
    setup_environment({"seed": env_cfg["seed"], "device": "cpu"})

    # =================================================================
    # Build models
    # =================================================================
    if is_main_process:
        logger.info("[1] Building C3Encoder...")

    encoder = build_c3_encoder(enc_cfg)
    encoder = encoder.to(device)

    # Enable gradient checkpointing if requested
    if use_checkpointing:
        encoder.llm.gradient_checkpointing_enable()
        if is_main_process:
            logger.info("    Gradient checkpointing: ENABLED")

    D_enc = encoder.hidden_dim

    if is_main_process:
        logger.info(f"    model: {encoder.model_name}")
        logger.info(f"    hidden_dim: {D_enc}")
        logger.info(f"    latent_token_len: {encoder.latent_token_len}")
        logger.info("")
        logger.info("[2] Building C3Decoder...")

    decoder = build_c3_decoder(
        dec_cfg,
        encoder_hidden_dim=D_enc,
        encoder_type="C3Encoder",
    )
    decoder = decoder.to(device)

    if use_checkpointing:
        decoder.llm.gradient_checkpointing_enable()

    D_dec = decoder.hidden_dim
    V = decoder.vocab_size

    if is_main_process:
        logger.info(f"    model: {decoder.model_name}")
        logger.info(f"    hidden_dim: {D_dec}")
        logger.info(f"    vocab_size: {V}")
        logger.info(f"    mm_projector: {D_enc} -> {D_dec}")
        logger.info("")

    # =================================================================
    # Tokenizer for loss computation
    # =================================================================
    if is_main_process:
        logger.info("[3] Setting up tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(dec_cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main_process:
        logger.info(f"    pad_token_id: {tokenizer.pad_token_id}")
        logger.info("")

    # =================================================================
    # Setup loss function
    # =================================================================
    loss_cfg = train_cfg["loss"]
    ignore_index = loss_cfg["ignore_index"]
    loss_fn = C3ReconstructionLoss(
        tokenizer=tokenizer,
        max_length=M,
        ignore_index=ignore_index,
    )
    loss_fn = loss_fn.to(device)

    if is_main_process:
        logger.info(f"[4] Loss: C3ReconstructionLoss, ignore_index={ignore_index}")
        logger.info("")

    # =================================================================
    # Load data
    # =================================================================
    if is_main_process:
        logger.info("[5] Loading dataset...")

    # Get batch size from DeepSpeed config (zero2.json)
    per_device_batch_size = ds_config["train_micro_batch_size_per_gpu"]

    dataset = registry.get(data_cfg, split=data_cfg["split"])

    # Use DistributedSampler for multi-GPU
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
        logger.info(f"    Dataset: {data_cfg['data_name']}, {len(dataset)} samples")
        logger.info(f"    World size: {world_size}")
        logger.info(f"    Per-device batch size: {per_device_batch_size}")
        logger.info(f"    Effective batch size: {per_device_batch_size * world_size}")
        logger.info("")

    # =================================================================
    # Initialize DeepSpeed
    # =================================================================
    if is_main_process:
        logger.info("[7] Initializing DeepSpeed...")

    # DeepSpeed config with model parameters
    ds_config_with_model = ds_config.copy()
    ds_config_with_model["train_micro_batch_size_per_gpu"] = per_device_batch_size

    # Initialize encoder with DeepSpeed
    model_engine_encoder, _, _, _ = deepspeed.initialize(
        model=encoder,
        model_parameters=encoder.parameters(),
        config=ds_config_with_model,
    )

    # Initialize decoder with DeepSpeed
    model_engine_decoder, _, _, _ = deepspeed.initialize(
        model=decoder,
        model_parameters=decoder.parameters(),
        config=ds_config_with_model,
    )

    if is_main_process:
        logger.info(
            f"    DeepSpeed initialized with ZeRO-{ds_config['zero_optimization']['stage']}"
        )
        logger.info(f"    BF16 enabled: {ds_config['bf16']['enabled']}")
        logger.info("")

    # =================================================================
    # Resume from checkpoint
    # =================================================================
    start_epoch = 0
    global_step = 0

    # Setup training history (only on main process)
    # =================================================================
    # Initialize to None for non-main processes
    history = None
    samples_store = None

    if is_main_process:
        training_config = TrainingConfig(
            experiment_name="c3_original_ds",
            batch_size=per_device_batch_size * world_size,
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

        samples_store = ReconstructionSampleStore(
            folder=str(log_dir / "samples"),
            block_size=50,
        )
        logger.info(f"    Samples store: {log_dir / 'samples'}")
        logger.info("")

    # Resume from DeepSpeed checkpoint
    if resume:
        latest_ckpt_dir = checkpoint_dir / "latest"
        if latest_ckpt_dir.exists() and latest_ckpt_dir.is_symlink():
            ckpt_path = os.readlink(latest_ckpt_dir)
            if is_main_process:
                logger.info(f"[7.5] Resuming from: {ckpt_path}")

            # Load DeepSpeed checkpoint
            _, _, _, enc_client_state = model_engine_encoder.load_checkpoint(
                checkpoint_dir=str(checkpoint_dir),
                tag=ckpt_path,
            )
            _, _, _, _ = model_engine_decoder.load_checkpoint(
                checkpoint_dir=str(checkpoint_dir),
                tag=ckpt_path,
            )

            if enc_client_state is not None:
                start_epoch = enc_client_state.get("epoch", 0)
                global_step = enc_client_state.get("global_step", 0)

            if is_main_process:
                logger.info(f"    Resumed from epoch {start_epoch}, step {global_step}")
                logger.info("")

    # =================================================================
    # Training loop
    # NOTE: DeepSpeed handles BF16 automatically, no manual autocast needed
    # =================================================================
    if is_main_process:
        logger.log_subheader("[8] Starting training...")

    model_engine_encoder.train()
    model_engine_decoder.train()

    for epoch in range(start_epoch, num_epochs):
        # Set sampler epoch for proper shuffling
        sampler.set_epoch(epoch)

        epoch_loss = 0.0
        num_batches = 0

        if is_main_process:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        else:
            pbar = dataloader

        for _, batch_texts in enumerate(pbar):
            # Tokenize
            tokens = tokenizer(
                batch_texts,
                max_length=M,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)

            labels = input_ids.clone()
            labels[attention_mask == 0] = ignore_index

            # Forward pass (DeepSpeed handles BF16 autocast internally)
            latent_tokens = model_engine_encoder(inputs=batch_texts)
            logits = model_engine_decoder(latent_tokens, prompt_ids=input_ids)

            # Compute loss (keep in float32 for stability)
            loss, loss_dict = loss_fn(logits, labels, N)

            # Backward
            model_engine_encoder.backward(loss)
            model_engine_decoder.backward(loss)

            # Optimizer step (DeepSpeed handles gradient accumulation)
            model_engine_encoder.step()
            model_engine_decoder.step()

            # Logging
            epoch_loss += loss_dict["total_loss"]
            num_batches += 1
            global_step += 1

            # Update progress bar (only on main process)
            if is_main_process:
                pbar.set_postfix(
                    {
                        "loss": f"{loss_dict['total_loss']:.4f}",
                        "avg": f"{epoch_loss/num_batches:.4f}",
                    }
                )

            # Log at intervals
            if global_step % log_interval == 0 and is_main_process:
                # These are guaranteed to be initialized on main process
                assert history is not None
                assert samples_store is not None

                step_record = TrainingStep(
                    epoch=epoch + 1,
                    step_in_epoch=num_batches,
                    global_step=global_step,
                    total_loss=loss_dict["total_loss"],
                    recon_loss=loss_dict["recon_loss"],
                    avg_loss=epoch_loss / num_batches,
                    lr_encoder=model_engine_encoder.get_lr()[0],
                    lr_decoder=model_engine_decoder.get_lr()[0],
                )
                history.append(step_record)
                logger.log_step(step_record, log_interval=1)

                # Save reconstruction samples
                with torch.no_grad():
                    text_logits = logits[:, N:, :]
                    decode_result = decode_logits_to_text(
                        text_logits, tokenizer, batch_texts, attention_mask
                    )
                recon_samples = create_reconstruction_samples(decode_result)
                sample_key = samples_store.save_samples(step_record, recon_samples)
                logger.info(f"    [Samples saved: {sample_key}]")

            # Save checkpoint at intervals
            if global_step % checkpoint_interval == 0:
                client_state = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "step_in_epoch": num_batches,
                }
                model_engine_encoder.save_checkpoint(
                    save_dir=str(checkpoint_dir),
                    tag=f"global_step_{global_step}",
                    client_state=client_state,
                )
                model_engine_decoder.save_checkpoint(
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

            # Save epoch checkpoint
            client_state = {
                "epoch": epoch,
                "global_step": global_step,
                "step_in_epoch": num_batches,
            }
            model_engine_encoder.save_checkpoint(
                save_dir=str(checkpoint_dir),
                tag=f"epoch_{epoch+1}",
                client_state=client_state,
            )
            model_engine_decoder.save_checkpoint(
                save_dir=str(checkpoint_dir),
                tag=f"epoch_{epoch+1}",
                client_state=client_state,
            )

    # Training complete
    if is_main_process:
        assert history is not None
        logger.log_header("Training completed!")

        # Save final checkpoint
        client_state = {
            "epoch": num_epochs,
            "global_step": global_step,
        }
        model_engine_encoder.save_checkpoint(
            save_dir=str(checkpoint_dir),
            tag="final",
            client_state=client_state,
        )
        model_engine_decoder.save_checkpoint(
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
        description="C3 Context Cascade Compression - DeepSpeed Training"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., configs/PreExp/c3_original_ds.yml)",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        required=True,
        help="Path to DeepSpeed config file (e.g., configs/PreExp/zero2.json)",
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

    train_c3_ds(config, ds_config)
