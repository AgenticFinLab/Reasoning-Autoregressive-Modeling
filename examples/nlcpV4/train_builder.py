"""Train NLCP V4 ConceptPyramidBuilder.

Usage:
    python3 examples/nlcpV4/train_builder.py -c configs/nlcpV4/GSM8K/train_builder_Qwen2.5-0.5B_6level.yml
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from torch.optim import AdamW
from tqdm import tqdm

import swanlab

# Ensure project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

from nlcpV4.concept_builder import ConceptPyramidBuilder
from nlcpV4.data_loader import NLCPV4DataLoader
from nlcpV4.eval_builder import (
    evaluate_builder,
    log_eval_results,
    log_terminal_entry,
)
from nlcpV4.losses import compute_builder_loss
from lmbase.utils.env_tools import get_device
from ram.utils import load_config, setup_environment


def _log_model_summary(builder: ConceptPyramidBuilder, config: dict, logger):
    """Log a detailed model architecture summary table."""
    reason_cfg = config["model"]["reason_model"]
    pyramid_cfg = config["model"]["pyramid"]
    train_rm_cfg = config["training"]["reason_model"]
    loss_weights = config["training"]["loss_weights"]

    total_params = sum(p.numel() for p in builder.parameters())
    trainable_params = sum(p.numel() for p in builder.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    # Per-module param counts
    def _count(module):
        return sum(p.numel() for p in module.parameters())

    def _count_trainable(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    reason_total = _count(builder.reason_model)
    reason_train = _count_trainable(builder.reason_model)
    proj_params = _count(builder.input_proj) + _count(builder.input_proj_norm)
    query_params = sum(q.numel() for q in builder.concept_queries)
    level_proj_params = _count(builder.level_projs)
    back_proj_params = _count(builder.back_proj)
    temp_params = builder.temperature.numel()

    level_lengths = pyramid_cfg["level_lengths"]
    D = pyramid_cfg["hidden_dim"]
    D_enc = builder.reason_model_hidden_dim

    lines = [
        "",
        "=" * 72,
        "  MODEL ARCHITECTURE SUMMARY",
        "=" * 72,
        "",
        "  Reason Model",
        "  ├─ name              : %s" % reason_cfg["reason_model_name"],
        "  ├─ encoder_dim       : %d" % D_enc,
        "  ├─ vocab_size        : %d" % builder.reason_model.config.vocab_size,
        "  ├─ num_layers        : %d" % builder.reason_model.config.num_hidden_layers,
        "  ├─ freeze            : %s" % train_rm_cfg["freeze"],
        "  ├─ lora              : %s" % (train_rm_cfg["lora"] or "None"),
        "  ├─ params            : %s (trainable: %s)"
        % (f"{reason_total:,}", f"{reason_train:,}"),
        "  ",
        "  Pyramid",
        "  ├─ hidden_dim (D)    : %d" % D,
        "  ├─ num_levels (K)    : %d" % pyramid_cfg["num_levels"],
        "  ├─ level_lengths     : %s  (total: %d)"
        % (level_lengths, sum(level_lengths)),
        "  ├─ max_seq_len       : %d" % pyramid_cfg["max_seq_len"],
        "  ",
        "  Modules                     Shape                 Params",
        "  " + "-" * 68,
        "  input_proj               : [%d, %d] + LN        %s"
        % (D_enc, D, f"{proj_params:,}"),
        "  concept_queries          : %d levels             %s"
        % (len(level_lengths), f"{query_params:,}"),
    ]
    for k, L_k in enumerate(level_lengths):
        lines.append("    level %d               : [%d, %d]" % (k, L_k, D))
    lines += [
        "  temperature              : [1]                   %d" % temp_params,
        "  level_projs              : %d × [%d, %d]        %s"
        % (len(level_lengths), D, D, f"{level_proj_params:,}"),
        "  back_proj                : [%d, %d]              %s"
        % (D, D_enc, f"{back_proj_params:,}"),
        "  ",
        "  Loss Weights",
        "  ├─ recon               : %s" % loss_weights["recon_loss_weight"],
        "  ├─ ordering            : %s" % loss_weights["ordering_loss_weight"],
        "  ├─ residual            : %s" % loss_weights["residual_loss_weight"],
        "  ├─ reasoning           : %s" % loss_weights["reasoning_loss_weight"],
        "  ├─ ordering_margin     : %s" % loss_weights["ordering_margin"],
        "  ",
        "  Parameter Summary",
        "  ├─ total               : %s" % f"{total_params:,}",
        "  ├─ trainable           : %s  (%.2f%%)"
        % (f"{trainable_params:,}", 100.0 * trainable_params / total_params),
        "  └─ frozen              : %s" % f"{frozen_params:,}",
        "=" * 72,
        "",
    ]
    for line in lines:
        logger.info(line)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ConceptPyramidBuilder")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--resume", type=str, default="", help="Path to checkpoint to resume from"
    )
    return parser.parse_args()


def save_checkpoint(
    builder: ConceptPyramidBuilder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    step: int,
    loss: float,
    checkpoint_dir: Path,
    is_best: bool,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "model_state_dict": builder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    path = checkpoint_dir / (
        "checkpoint_best.pt" if is_best else f"checkpoint-epoch{epoch}-step{step}.pt"
    )
    torch.save(checkpoint, path)
    return path


def load_checkpoint(
    checkpoint_path: Path,
    builder: ConceptPyramidBuilder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> tuple[int, int, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    builder.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"], checkpoint["step"], checkpoint["loss"]


def train_builder(config: dict, config_path: Path):
    """Main training loop."""
    # Extract sub-configs
    pyramid_cfg = config["model"]["pyramid"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    env_cfg = config["environment"]
    log_cfg = config["log"]
    loss_weights = train_cfg["loss_weights"]

    batch_size = train_cfg["batch_size"]
    learning_rate = train_cfg["learning_rate"]
    weight_decay = train_cfg["weight_decay"]
    num_epochs = train_cfg["num_epochs"]
    warmup_ratio = train_cfg["warmup_ratio"]
    gradient_clip = train_cfg["gradient_clip"]
    log_interval = log_cfg["log_step_interval"]
    checkpoint_interval = log_cfg["checkpoint_step_interval"]
    resume = train_cfg["resume"]
    ordering_loss_type = train_cfg["ordering_loss_type"]

    checkpoint_dir = Path(log_cfg["checkpoint_path"])
    log_dir = Path(log_cfg["log_path"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_cfg["log_level"].upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("train_builder")
    terminal_log_path = log_dir / "terminal_output.jsonl"

    seed = env_cfg["seed"]
    setup_environment({"seed": seed, "device": "auto"})
    device = str(get_device("auto"))
    logger.info("Device: %s", device)

    # ── Load .env and initialize SwanLab ─────────────────────────────
    dotenv_path = env_cfg["dotenv_path"]
    load_dotenv(dotenv_path)

    # Derive experiment name from config filename, e.g.
    #   configs/nlcpV4/GSM8K/train_builder_Qwen2.5-0.5B_6level.yml
    #   -> "GSM8K-train_builder_Qwen2.5-0.5B_6level"
    experiment_name = f"{config_path.parent.name}-{config_path.stem}"

    swanlab.init(
        project="ReasoningAR",
        experiment_name=experiment_name,
        config=config,
    )
    logger.info("SwanLab initialized")

    builder = ConceptPyramidBuilder(config)
    builder.to(device)
    _log_model_summary(builder, config, logger)

    trainable_params = [p for p in builder.parameters() if p.requires_grad]

    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

    dataloader = NLCPV4DataLoader(
        data_cfg=data_cfg,
        batch_size=batch_size,
        include_solution=True,
        shuffle=data_cfg["shuffle"],
        drop_last=data_cfg["drop_last"],
        num_workers=env_cfg["dataloader_num_workers"],
    )
    logger.info(
        "Dataset: %s | Batches/epoch: %d | Batch size: %d",
        data_cfg["data_name"],
        len(dataloader),
        batch_size,
    )

    # ── Evaluation setup ─────────────────────────────────────────
    eval_cfg = config["evaluation"]
    eval_interval = eval_cfg["eval_step_interval"]
    eval_enabled = eval_interval > 0
    eval_dataloader = None
    eval_history = []
    quick_eval_batches = 0
    full_eval_batches = 0

    if eval_enabled:
        eval_data_cfg = eval_cfg["data"]
        eval_dataloader = NLCPV4DataLoader(
            data_cfg=eval_data_cfg,
            batch_size=batch_size,
            include_solution=True,
            shuffle=True,
            drop_last=False,
            num_workers=env_cfg["dataloader_num_workers"],
        )
        eval_dataset_size = eval_dataloader.dataset_size

        # Resolve log_num_samples: (0,1] = proportion, >1 = exact count
        raw_log_ns = eval_data_cfg["log_num_samples"]
        if 0 < raw_log_ns <= 1.0:
            log_eval_samples = int(eval_dataset_size * raw_log_ns)
        else:
            log_eval_samples = int(raw_log_ns)
        quick_eval_batches = max(1, (log_eval_samples + batch_size - 1) // batch_size)

        # Resolve eval_num_samples: (0,1] = proportion, >1 = exact count
        raw_eval_ns = eval_data_cfg["eval_num_samples"]
        if 0 < raw_eval_ns <= 1.0:
            full_eval_samples = int(eval_dataset_size * raw_eval_ns)
        else:
            full_eval_samples = int(raw_eval_ns)
        full_eval_batches = max(1, (full_eval_samples + batch_size - 1) // batch_size)

        logger.info(
            "Eval: %s (split=%s) | dataset_size=%d | full_eval=%d batches | quick_eval=%d batches",
            eval_data_cfg["data_name"],
            eval_data_cfg["split"],
            eval_dataset_size,
            full_eval_batches,
            quick_eval_batches,
        )

    total_steps = len(dataloader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    logger.info(
        "Total steps: %d | Warmup: %d | LR: %s",
        total_steps,
        warmup_steps,
        learning_rate,
    )

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    best_eval_loss = float("inf")
    history = []

    if resume:
        resume_path = Path(resume)
        if resume_path.exists():
            start_epoch, global_step, best_loss = load_checkpoint(
                resume_path, builder, optimizer, scheduler
            )
            logger.info("Resumed from epoch %d, step %d", start_epoch, global_step)
        else:
            logger.warning("Resume path not found: %s", resume_path)

    config_save_path = log_dir / "config.json"
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)

    builder.train()

    for epoch in range(start_epoch, num_epochs):
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in pbar:
            # Forward pass: batch → pyramid (encode + build + reasoning)
            pyramid = builder(batch)

            total_loss, loss_dict = compute_builder_loss(
                pyramid,
                loss_weights,
                ordering_loss_type=ordering_loss_type,
            )

            total_loss.backward()

            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, gradient_clip)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            epoch_losses.append(loss_dict["total"])
            global_step += 1

            # Compute weighted individual losses
            w = {
                "recon": loss_dict["recon"] * loss_weights["recon_loss_weight"],
                "ordering": loss_dict["ordering"]
                * loss_weights["ordering_loss_weight"],
                "residual": loss_dict["residual"]
                * loss_weights["residual_loss_weight"],
            }
            if "reasoning" in loss_dict:
                w["reasoning"] = (
                    loss_dict["reasoning"] * loss_weights["reasoning_loss_weight"]
                )

            if global_step % log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(
                    {
                        "loss": f"{loss_dict['total']:.4f}",
                        "recon": f"{loss_dict['recon']:.4f}",
                        "order": f"{loss_dict['ordering']:.4f}",
                        "lr": f"{lr:.2e}",
                    }
                )
                # Console: raw/weighted for each component
                reasoning_part = ""
                if "reasoning" in loss_dict:
                    reasoning_part = " reasoning=%.4f/%.4f" % (
                        loss_dict["reasoning"],
                        w["reasoning"],
                    )
                logger.info(
                    "Step %5d | total=%.4f recon=%.4f/%.4f ordering=%.4f/%.4f"
                    " residual=%.4f/%.4f%s lr=%.2e",
                    global_step,
                    loss_dict["total"],
                    loss_dict["recon"],
                    w["recon"],
                    loss_dict["ordering"],
                    w["ordering"],
                    loss_dict["residual"],
                    w["residual"],
                    reasoning_part,
                    lr,
                )
                # terminal_output.jsonl: raw + weighted
                terminal_entry = {
                    "step": global_step,
                    "epoch": epoch,
                    "total": round(loss_dict["total"], 6),
                    "recon": round(loss_dict["recon"], 6),
                    "recon_w": round(w["recon"], 6),
                    "ordering": round(loss_dict["ordering"], 6),
                    "ordering_w": round(w["ordering"], 6),
                    "residual": round(loss_dict["residual"], 6),
                    "residual_w": round(w["residual"], 6),
                    "lr": lr,
                }
                if "reasoning" in loss_dict:
                    terminal_entry["reasoning"] = round(loss_dict["reasoning"], 6)
                    terminal_entry["reasoning_w"] = round(w["reasoning"], 6)
                log_terminal_entry(terminal_log_path, terminal_entry)

                # SwanLab: raw + weighted as separate metrics
                swanlab_metrics = {
                    "train/total_loss": loss_dict["total"],
                    "train/recon_raw": loss_dict["recon"],
                    "train/recon_weighted": w["recon"],
                    "train/ordering_raw": loss_dict["ordering"],
                    "train/ordering_weighted": w["ordering"],
                    "train/residual_raw": loss_dict["residual"],
                    "train/residual_weighted": w["residual"],
                    "train/lr": lr,
                }
                if "reasoning" in loss_dict:
                    swanlab_metrics["train/reasoning_raw"] = loss_dict["reasoning"]
                    swanlab_metrics["train/reasoning_weighted"] = w["reasoning"]
                swanlab.log(swanlab_metrics, step=global_step)

                # ── Quick eval (skip when full eval fires at same step) ──
                if eval_enabled and not (global_step % eval_interval == 0):
                    eval_losses = evaluate_builder(
                        builder,
                        eval_dataloader,
                        loss_weights,
                        ordering_loss_type,
                        max_batches=quick_eval_batches,
                    )
                    log_eval_results(
                        eval_losses,
                        loss_weights,
                        "quick",
                        global_step,
                        terminal_log_path,
                        eval_history,
                        log_dir,
                        "eval_quick",
                    )

            if global_step % checkpoint_interval == 0:
                avg_loss = (
                    sum(epoch_losses[-100:]) / len(epoch_losses[-100:])
                    if epoch_losses
                    else float("inf")
                )
                is_best = avg_loss < best_loss
                if is_best:
                    best_loss = avg_loss
                path = save_checkpoint(
                    builder,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    avg_loss,
                    checkpoint_dir,
                    is_best=is_best,
                )
                logger.info("Checkpoint: %s", path.name)

            # ── Full eval at eval_interval ──────────────────────
            if eval_enabled and global_step % eval_interval == 0:
                eval_losses = evaluate_builder(
                    builder,
                    eval_dataloader,
                    loss_weights,
                    ordering_loss_type,
                    max_batches=full_eval_batches,
                )
                log_eval_results(
                    eval_losses,
                    loss_weights,
                    "full",
                    global_step,
                    terminal_log_path,
                    eval_history,
                    log_dir,
                    "eval",
                )
                # Best eval checkpoint
                if eval_losses["total"] < best_eval_loss:
                    best_eval_loss = eval_losses["total"]
                    ckpt = {
                        "epoch": epoch,
                        "step": global_step,
                        "eval_loss": eval_losses["total"],
                        "model_state_dict": builder.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    }
                    torch.save(ckpt, checkpoint_dir / "checkpoint_best_eval.pt")
                    logger.info(
                        "Best eval checkpoint: step %d, eval_loss=%.4f",
                        global_step,
                        eval_losses["total"],
                    )

            # Record step history with raw + weighted losses
            step_record = {"step": global_step, "epoch": epoch, **loss_dict}
            step_record.update({f"{k}_w": v for k, v in w.items()})
            history.append(step_record)

        avg_epoch_loss = (
            sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("inf")
        )
        logger.info("Epoch %d avg loss: %.4f", epoch + 1, avg_epoch_loss)
        log_terminal_entry(
            terminal_log_path,
            {"epoch": epoch, "avg_epoch_loss": round(avg_epoch_loss, 6)},
        )

        # ── SwanLab epoch-level logging ──────────────────────────
        swanlab.log(
            {"epoch/avg_loss": avg_epoch_loss, "epoch/epoch": epoch + 1},
            step=global_step,
        )

        path = save_checkpoint(
            builder,
            optimizer,
            scheduler,
            epoch + 1,
            global_step,
            avg_epoch_loss,
            checkpoint_dir,
            is_best=False,
        )
        logger.info("Epoch checkpoint: %s", path.name)

        with open(log_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, default=str)

        if eval_history:
            with open(log_dir / "eval_history.json", "w", encoding="utf-8") as f:
                json.dump(eval_history, f, indent=2, default=str)

    logger.info("Training complete!")
    logger.info("Best checkpoint: %s", checkpoint_dir / "checkpoint_best.pt")

    # ── Finish SwanLab run ────────────────────────────────────────
    swanlab.finish()
    logger.info("SwanLab run finished")


def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    yaml_config = load_config(str(config_path))

    # Merge resume flag from CLI if not in config
    if args.resume and not yaml_config["training"]["resume"]:
        yaml_config["training"]["resume"] = args.resume

    train_builder(yaml_config, config_path=config_path)


if __name__ == "__main__":
    main()
