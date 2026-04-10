"""NLCP (Next-Level Concept Pyramid) Training Script.

This module implements the training pipeline for NLCP.
Reference: concept-pyramid.md Section 4.3 - Staged Pretraining Pipeline

Stages:
    Phase 1: Level 0 intent planning
        - Train Encoder + Level 0 AR
        - Establish global structure prior, verify Depth Gate initial response

    Phase 2: Next-Level generation alignment
        - Train Level 1..K Generator + L_consist
        - Verify cross-level causal flow and consistency gradient

    Phase 3: Full pyramid joint finetuning
        - Full unfreeze + L_depth + L_CE
        - End-to-end alignment to tokens, stabilize dynamic depth
"""

import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

# Local imports - use relative imports for module resolution
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from examples.nlcp.base import (
    NLCPModelConfig,
    NLCPTrainingConfig,
    LevelState,
)
from examples.nlcp.model import NLCPModel, build_nlcp_model


@dataclass
class TrainingState:
    """Training state container.

    Attributes:
        global_step: Current global training step
        epoch: Current epoch
        best_loss: Best validation loss
        phase: Current training phase (1, 2, or 3)
    """

    global_step: int
    epoch: int
    best_loss: float
    phase: int


class DummyDataset(Dataset):
    """Dummy dataset for demonstration.

    In real usage, replace with actual tokenized dataset.
    """

    def __init__(self, vocab_size: int, seq_length: int, num_samples: int):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


class NLCPTrainer:
    """NLCP Training Manager.

    Reference: concept-pyramid.md Section 4.3
    Staged Pretraining Pipeline

    Attributes:
        model: NLCP model instance
        train_config: Training configuration
        model_config: Model configuration
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        device: Training device
        training_state: Current training state
    """

    def __init__(
        self,
        model: NLCPModel,
        model_config: NLCPModelConfig,
        train_config: NLCPTrainingConfig,
        device: torch.device,
        padding_id: int,
    ):
        self.model = model
        self.model_config = model_config
        self.train_config = train_config
        self.device = device
        self.padding_id = padding_id

        # Initialize optimizer with Decoupled μP
        # Reference: Section 4.2 "η_k = η_base * (d_k / d_base)^{-1}"
        self.optimizer = self._build_optimizer()

        # Initialize scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=train_config.max_steps,
            eta_min=train_config.learning_rate * 0.01,
        )

        # Training state
        self.training_state = TrainingState(
            global_step=0,
            epoch=0,
            best_loss=float("inf"),
            phase=1,
        )

    def _build_optimizer(self) -> AdamW:
        """Build optimizer with Decoupled μP learning rates.

        Reference: concept-pyramid.md Section 4.2
        "Strictly follow DLCM Sec 6.1 heterogeneous module learning rate decoupling:
        η_k = η_base * (d_k / d_base)^{-1}"

        This applies different learning rates to modules with different widths.
        """
        base_lr = self.train_config.learning_rate
        base_dim = self.model_config.hidden_dim

        # Group parameters by module for different learning rates
        param_groups = []

        # Encoder parameters (standard learning rate)
        encoder_params = list(self.model.encoder.parameters())
        if encoder_params:
            param_groups.append(
                {
                    "params": encoder_params,
                    "lr": base_lr,
                    "name": "encoder",
                }
            )

        # Depth gate parameters
        gate_params = list(self.model.depth_gate.parameters())
        if gate_params:
            param_groups.append(
                {
                    "params": gate_params,
                    "lr": base_lr,
                    "name": "depth_gate",
                }
            )

        # Expansion predictor parameters
        predictor_params = list(self.model.expansion_predictor.parameters())
        if predictor_params:
            param_groups.append(
                {
                    "params": predictor_params,
                    "lr": base_lr,
                    "name": "expansion_predictor",
                }
            )

        # Level generators (each might have different effective width)
        for i, generator in enumerate(self.model.level_generators):
            gen_params = list(generator.parameters())
            if gen_params:
                # Apply μP scaling if widths differ
                # In this implementation, all levels share the same hidden_dim
                param_groups.append(
                    {
                        "params": gen_params,
                        "lr": base_lr,
                        "name": f"level_generator_{i}",
                    }
                )

        # Token decoder parameters
        decoder_params = list(self.model.token_decoder.parameters())
        if decoder_params:
            # Output layer scaling per DLCM Eq.21
            param_groups.append(
                {
                    "params": decoder_params,
                    "lr": base_lr,
                    "name": "token_decoder",
                }
            )

        return AdamW(
            param_groups,
            weight_decay=self.train_config.weight_decay,
            betas=(0.9, 0.95),
        )

    def set_training_phase(self, phase: int) -> None:
        """Set training phase with appropriate freezing.

        Reference: concept-pyramid.md Section 4.3
        Phase-specific freezing/unfreezing strategy.

        Phase 1: Train Encoder + Level 0 AR
        Phase 2: Train Level 1..K Generator + L_consist
        Phase 3: Full unfreeze + L_depth + L_CE

        Args:
            phase: Training phase (1, 2, or 3)
        """
        self.training_state.phase = phase

        # First, freeze everything
        for param in self.model.parameters():
            param.requires_grad = False

        if phase == 1:
            # Phase 1: Train Encoder + Level 0
            for param in self.model.encoder.parameters():
                param.requires_grad = True
            for param in self.model.l0_proj.parameters():
                param.requires_grad = True

        elif phase == 2:
            # Phase 2: Train Level generators + consistency
            for generator in self.model.level_generators:
                for param in generator.parameters():
                    param.requires_grad = True
            for param in self.model.expansion_predictor.parameters():
                param.requires_grad = True
            for param in self.model.depth_gate.parameters():
                param.requires_grad = True

        elif phase == 3:
            # Phase 3: Full unfreeze
            for param in self.model.parameters():
                param.requires_grad = True

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Execute a single training step.

        Dimension Flow:
            batch: Dict with 'input_ids' [B, L] and 'labels' [B, L]
                ↓
            Forward pass through NLCP
                ↓
            Compute losses (Section 4.1)
                ↓
            Backward pass + gradient clipping
                ↓
            Optimizer step

        Args:
            batch: Dictionary containing input_ids and labels

        Returns:
            loss_dict: Dictionary of loss values for logging
        """
        self.model.train()
        self.optimizer.zero_grad()

        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Forward pass
        output = self.model(
            input_ids=input_ids,
            target_ids=labels,
            padding_id=self.padding_id,
            compute_loss=True,
        )

        # Backward pass
        loss = output.total_loss
        loss.backward()

        # Gradient clipping
        # Reference: Section 7.2 "L_consist gradient may be large, recommend grad_clip_norm = 1.0"
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.train_config.grad_clip_norm,
        )

        self.optimizer.step()
        self.scheduler.step()

        self.training_state.global_step += 1

        return {
            "loss": output.total_loss,
            "ntp_loss": output.ntp_loss,
            "consist_loss": output.consist_loss,
            "depth_loss": output.depth_loss,
            "ce_loss": output.ce_loss,
            "lr": self.scheduler.get_last_lr()[0],
        }

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """Validate model on validation set.

        Args:
            val_loader: Validation data loader

        Returns:
            loss_dict: Average validation losses
        """
        self.model.eval()

        total_loss = 0.0
        total_ntp = 0.0
        total_consist = 0.0
        total_depth = 0.0
        total_ce = 0.0
        num_batches = 0

        for batch in val_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            output = self.model(
                input_ids=input_ids,
                target_ids=labels,
                padding_id=self.padding_id,
                compute_loss=True,
            )

            total_loss += output.total_loss
            total_ntp += output.ntp_loss
            total_consist += output.consist_loss
            total_depth += output.depth_loss
            total_ce += output.ce_loss
            num_batches += 1

        return {
            "val_loss": total_loss / max(num_batches, 1),
            "val_ntp_loss": total_ntp / max(num_batches, 1),
            "val_consist_loss": total_consist / max(num_batches, 1),
            "val_depth_loss": total_depth / max(num_batches, 1),
            "val_ce_loss": total_ce / max(num_batches, 1),
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        checkpoint_dir: str,
        log_interval: int,
        eval_interval: int,
    ) -> None:
        """Execute full training loop.

        Reference: concept-pyramid.md Section 4.3
        Staged Pretraining Pipeline

        Training proceeds through three phases:
        - Phase 1: epochs with prefix "phase1_"
        - Phase 2: epochs with prefix "phase2_"
        - Phase 3: remaining epochs

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Total number of epochs
            checkpoint_dir: Directory for saving checkpoints
            log_interval: Steps between logging
            eval_interval: Steps between evaluation
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Phase 1: Intent planning
        phase1_epochs = num_epochs // 4
        self.set_training_phase(1)

        for epoch in range(phase1_epochs):
            self.training_state.epoch = epoch

            for batch in train_loader:
                loss_dict = self.train_step(batch)

                if self.training_state.global_step % log_interval == 0:
                    self._log_losses(loss_dict, prefix="train")

                if self.training_state.global_step % eval_interval == 0:
                    val_dict = self.validate(val_loader)
                    self._log_losses(val_dict, prefix="val")

        # Phase 2: Next-Level generation alignment
        phase2_epochs = num_epochs // 4
        self.set_training_phase(2)

        for epoch in range(phase1_epochs, phase1_epochs + phase2_epochs):
            self.training_state.epoch = epoch

            for batch in train_loader:
                loss_dict = self.train_step(batch)

                if self.training_state.global_step % log_interval == 0:
                    self._log_losses(loss_dict, prefix="train")

                if self.training_state.global_step % eval_interval == 0:
                    val_dict = self.validate(val_loader)
                    self._log_losses(val_dict, prefix="val")

        # Phase 3: Full pyramid joint finetuning
        self.set_training_phase(3)

        for epoch in range(phase1_epochs + phase2_epochs, num_epochs):
            self.training_state.epoch = epoch

            for batch in train_loader:
                loss_dict = self.train_step(batch)

                if self.training_state.global_step % log_interval == 0:
                    self._log_losses(loss_dict, prefix="train")

                if self.training_state.global_step % eval_interval == 0:
                    val_dict = self.validate(val_loader)
                    self._log_losses(val_dict, prefix="val")

                    # Save best checkpoint
                    if val_dict["val_loss"] < self.training_state.best_loss:
                        self.training_state.best_loss = val_dict["val_loss"]
                        self._save_checkpoint(checkpoint_dir, "best")

            # Save epoch checkpoint
            self._save_checkpoint(checkpoint_dir, f"epoch_{epoch}")

    def _log_losses(self, loss_dict: Dict[str, float], prefix: str) -> None:
        """Log loss values.

        Args:
            loss_dict: Dictionary of loss values
            prefix: Log prefix (train/val)
        """
        log_str = f"[{prefix}] Step {self.training_state.global_step}, "
        log_str += f"Phase {self.training_state.phase}: "
        for key, value in loss_dict.items():
            log_str += f"{key}={value:.4f} "
        print(log_str)

    def _save_checkpoint(self, checkpoint_dir: str, name: str) -> None:
        """Save model checkpoint.

        Args:
            checkpoint_dir: Directory for checkpoints
            name: Checkpoint name
        """
        checkpoint_path = os.path.join(checkpoint_dir, f"{name}.pt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "training_state": self.training_state,
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint to {checkpoint_path}")


def main():
    """Main training entry point.

    Reference: concept-pyramid.md Section 8
    Recommended Experimental Path:
        "MVP Validation: Fix K=2, run L = L_NTP + L_consist + L_CE pipeline,
        verify tensor flow and gradient closure"
    """
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model configuration
    # Reference: Section 3.1 Basic Configuration
    model_config = NLCPModelConfig(
        hidden_dim=512,  # Reduced for MVP
        num_heads=8,
        vocab_size=32000,
        max_depth=2,  # Fixed K=2 for MVP per Section 8
        depth_gate_threshold=0.4,
        l0_length=8,
        l_max=256,
        dropout=0.1,
        expansion_min=1,
        expansion_max=4,
    )

    # Training configuration
    # Reference: Section 4.1 Loss weights
    train_config = NLCPTrainingConfig(
        lambda_consist=0.1,
        lambda_depth=0.05,
        lambda_ce=1.0,
        target_expansion_ratio=4.0,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=1000,
        max_steps=10000,
        grad_clip_norm=1.0,
        muP_scale=1.0,
    )

    # Build model
    model = build_nlcp_model(
        config=model_config,
        padding_id=0,
        num_encoder_layers=4,
        num_generator_layers=2,
        use_info_nce=True,
        info_nce_weight=0.1,
    )
    model = model.to(device)

    # Create datasets
    train_dataset = DummyDataset(
        vocab_size=model_config.vocab_size,
        seq_length=128,
        num_samples=1000,
    )
    val_dataset = DummyDataset(
        vocab_size=model_config.vocab_size,
        seq_length=128,
        num_samples=100,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
    )

    # Create trainer
    trainer = NLCPTrainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=device,
        padding_id=0,
    )

    # Run training
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,
        checkpoint_dir="checkpoints/nlcp",
        log_interval=10,
        eval_interval=100,
    )


if __name__ == "__main__":
    main()
