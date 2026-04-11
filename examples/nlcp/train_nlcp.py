"""NLCP (Next-Level Concept Pyramid) Training Script.

Usage:
    python examples/nlcp/train_nlcp.py -c configs/nlcp/main.yml

DESIGN SOURCE:
    Reference: docs/concept-pyramid-V1.md
    - Section 4.1: Complete Loss Function and Training Data Format
    - Section 4.1.2: Why Intermediate Layers Have No Text Supervision
    - Section 4.2: Decoupled muP Adaptation
    - Section 4.3: Staged Pretraining Pipeline

    Additional reference: docs/concept-pyramid-critic.md (solutions for V1 issues)

================================================================================
TRAINABLE MODULES OVERVIEW
================================================================================

Based on concept-pyramid-V1.md Section 2.2 (Module Tasks and Connection Logic)
and the actual model architecture in model.py, NLCP has 6 trainable modules:

+----+----------------------+------------------------------------------+------------------+
| #  | Module               | Function                                 | Parameters       |
+----+----------------------+------------------------------------------+------------------+
| 1  | encoder              | Q → H_0 (problem abstraction)            | HF pretrained    |
|    | (HFCausalEncoder)    | - HuggingFace pretrained model           | + Pool & Project |
|    |                      | - Pool & project to L_0 concepts         | (trainable)      |
+----+----------------------+------------------------------------------+------------------+
| 2  | l0_proj              | Pool encoder output to L_0 concepts      | Linear(d, d)     |
|    | (nn.Linear)          | Shape: [B, L_q, d] → [B, L_0, d]         |                  |
+----+----------------------+------------------------------------------+------------------+
| 3  | depth_gate           | H_k → p_cont ∈ [0,1]                     | Pool + MLP       |
|    | (DepthGate)          | Decide whether to continue expansion     |                  |
|    |                      | - AttentionPool/ MeanPool                |                  |
|    |                      | - 2-layer MLP with sigmoid               |                  |
+----+----------------------+------------------------------------------+------------------+
| 4  | expansion_predictor  | H_k → expand_mask ∈ [1, λ_max]^{L_k}     | MLP + Softplus   |
|    | (ExpansionPredictor) | Predict per-position expansion rate       |                  |
|    |                      | - Determines L_{k+1} = sum(expand_mask)  |                  |
+----+----------------------+------------------------------------------+------------------+
| 5  | level_generators     | H_k → H_{k+1} (next level generation)    | M Transformer    |
|    | (NextLevelGenerator) | - Cross-Level Causal Attention            | layers per level |
|    |                      | - Self-Attention with causal mask        |                  |
|    |                      | - FFN + LayerNorm                        |                  |
|    |                      | - One generator per level transition     |                  |
+----+----------------------+------------------------------------------+------------------+
| 6  | token_decoder        | H_K → Logits ∈ R^{L_K × V}               | Linear(d, V)     |
|    | (TokenDecoder)       | Final vocabulary projection              | + μP scaling     |
+----+----------------------+------------------------------------------+------------------+

Total Parameters:
    - encoder: N_enc × (d² × 4 + 2d) ≈ N_enc × 4d²
    - l0_proj: d²
    - depth_gate: d × 256 + 256 × 1 ≈ 256d
    - expansion_predictor: d × 512 + 512 × 1 ≈ 512d
    - level_generators: (K-1) × M × 4d² ≈ K × M × 4d²
    - token_decoder: d × V

    Where:
        d = hidden_dim (e.g., 768)
        N_enc = num_encoder_layers (e.g., 4)
        M = num_generator_layers (e.g., 2)
        K = max_depth (e.g., 4)
        V = vocab_size (e.g., 128000)

================================================================================
AVAILABLE LOSSES (from losses.py)
================================================================================

Based on concept-pyramid-V1.md Section 4.1.2:

L_total = L_NTP(H_K → C)                (ONLY final layer has text supervision)
        + λ_1 × L_consist                (cross-level consistency)
        + λ_2 × L_depth                  (expansion rate regularization)

+----+----------------------+------------------------------------------+------------------+
| #  | Loss                 | Formula                                  | Module(s) Affected|
+----+----------------------+------------------------------------------+------------------+
| 1  | L_NTP                | -Σ log P(c_t | H_K, c_{<t})              | token_decoder,   |
|    | (NextTokenPrediction)| Cross-entropy at final level only        | level_generators,|
|    |                      |                                          | encoder          |
+----+----------------------+------------------------------------------+------------------+
| 2  | L_consist            | Σ_k ||MeanPool(H_{k+1}) - H_k||²        | level_generators |
|    | (CrossLevelConsist.)  | + optional InfoNCE                       |                  |
|    |                      | Enforces: fine level aggregated ≈ coarse |                  |
|    |                      | Types: standard, directional, residual,mi|                  |
+----+----------------------+------------------------------------------+------------------+
| 3  | L_depth              | Σ_k (L_{k+1}/L_k - R_target)²           | expansion_predictor|
|    | (ExpansionReg.)       | Prevents expansion collapse/explosion    |                  |
+----+----------------------+------------------------------------------+------------------+
| 4  | L_CE                 | CE(H_K @ W_unemb, target_tokens)         | token_decoder    |
|    | (FinalAlignment)      | Final alignment to vocabulary            |                  |
+----+----------------------+------------------------------------------+------------------+
| 5  | InfoNCE              | Contrastive loss on H_k vs H_{k+1}       | level_generators |
|    | (Optional)           | Enhances cross-level dependency          |                  |
+----+----------------------+------------------------------------------+------------------+

Loss Weights (Section 4.1 defaults):
    λ_1 (consist) = 0.1
    λ_2 (depth)   = 0.05
    λ_3 (ce)      = 1.0
    (cosine decay during training)

KEY INSIGHT - WHY ONLY FINAL LAYER HAS TEXT SUPERVISION (Section 4.1.2):

    Unlike VAR (f_rest provides per-layer supervision) or
    DLCM (Concept = Token Pool, naturally contains information),
    NLCP intermediate layers H_0, H_1, ..., H_{K-1} have NO direct text supervision.

    They are shaped through:
    1. Gradient backpropagation: L_NTP → ∂L/∂H_K → ... → ∂L/∂H_0
    2. Consistency constraints: Provide "pseudo-residual" signal
    3. Conditional generation: H_{k+1} depends on H_k via Cross-Attn

================================================================================
THREE-PHASE TRAINING STRATEGY (Section 4.3 Table)
================================================================================

+---------+----------------------------------+----------------------------+------------------+
| Phase   | Goal                             | Freeze/Train               | Active Losses    |
+---------+----------------------------------+----------------------------+------------------+
| Phase 1 | Level 0 intent planning          | Train: encoder, l0_proj    | L_NTP (Level 0)  |
|         |                                  | Freeze: All other modules  |                  |
|         | Duration: 25% epochs             |                            |                  |
+---------+----------------------------------+----------------------------+------------------+
| Phase 2 | Next-Level generation alignment  | Train: level_generators,   | L_NTP, L_consist |
|         |                                  |        expansion_predictor,|                  |
|         |                                  |        depth_gate           |                  |
|         |                                  | Freeze: encoder, l0_proj   |                  |
|         | Duration: 25% epochs             |                            |                  |
+---------+----------------------------------+----------------------------+------------------+
| Phase 3 | Full pyramid joint finetuning    | Train: All modules         | L_NTP, L_consist,|
|         |                                  |        (full unfreeze)     | L_depth, L_CE    |
|         | Duration: 50% epochs             |                            |                  |
+---------+----------------------------------+----------------------------+------------------+

PHASE DETAILS (Section 4.3):
    Phase 1: Level 0 Intent Planning
        - Establish stable global structure prior
        - Verify Depth Gate initial response
        - Duration: ~25% of total epochs
        - Loss: L_NTP at Level 0 only

    Phase 2: Next-Level Generation Alignment
        - Learn expansion and generation
        - Verify cross-level causal flow
        - Verify consistency gradient
        - Duration: ~25% of total epochs
        - Loss: L_NTP + L_consist
        - NOTE: Expansion predictor has limited learning due to floor()

    Phase 3: Full Pyramid Joint Finetuning
        - End-to-end alignment to tokens
        - Stabilize dynamic depth
        - Duration: ~50% of total epochs
        - Loss: L_NTP + L_consist + L_depth + L_CE

HYPERPARAMETERS (Section 4.1):
    Learning rate: eta = 1e-4 (base)
    muP scaling: eta_k = eta_base * (d_k / d_base)^{-1} for heterogeneous widths
    Weight decay: 0.01
    Warmup: 2000 steps
    Loss weights: lambda_1=0.1, lambda_2=0.05, lambda_3=1.0 (cosine decay)

CRITICAL CONSIDERATIONS (from concept-pyramid-critic.md):

    ISSUE 1 - Expansion Predictor Learning in Phase 2:
        Problem: floor() is non-differentiable
        Impact: Expansion predictor cannot learn from NTP loss directly
        Current workaround: L_depth regularization provides indirect signal
        Recommendation: Use Gumbel-Softmax (Solution 1A) before Phase 2

    ISSUE 3 - Depth Gate Causality:
        Problem: Non-causal pooling during training
        Impact: Phase 1-3 may overfit to full-sequence context
        Recommendation: Add causal masking to DepthGate before Phase 1

    ISSUE 2 - Consistency Loss Bottleneck:
        Problem: Strict L2 forces MeanPool(H_{k+1}) ≈ H_k
        Impact: Limits fine level expressiveness in Phase 2-3
        Recommendation: Use DirectionalConsistency (Solution 2A) in Phase 2
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from lmbase.dataset import registry
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Local imports - use relative imports for module resolution
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from examples.nlcp.base import NLCPModelConfig, NLCPTrainingConfig
from examples.nlcp.model import NLCPModel, build_nlcp_model
from ram.utils import (
    assign_model_devices,
    collate_fn_text,
    load_config,
    setup_environment,
)


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

        # Initialize scheduler with cosine annealing
        # T_max: Total steps for cosine decay
        # eta_min: Minimum learning rate (1% of base)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=train_config.max_steps,
            eta_min=train_config.learning_rate * 0.01,
        )

        # Initialize training state with default values
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

        Returns:
            AdamW optimizer with parameter groups for each module
        """
        base_lr = self.train_config.learning_rate

        # Group parameters by module for different learning rates
        param_groups = []

        # Encoder parameters (standard learning rate)
        # Shape: All encoder weights and biases
        encoder_params = list(self.model.encoder.parameters())
        if encoder_params:
            param_groups.append(
                {
                    "params": encoder_params,
                    "lr": base_lr,
                    "name": "encoder",
                }
            )

        # Depth gate parameters for early stopping prediction
        gate_params = list(self.model.depth_gate.parameters())
        if gate_params:
            param_groups.append(
                {
                    "params": gate_params,
                    "lr": base_lr,
                    "name": "depth_gate",
                }
            )

        # Expansion predictor parameters for level expansion
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
        # Flow: Iterate through all level generators and add their parameters
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

        # Token decoder parameters for final vocabulary prediction
        # Output layer scaling per DLCM Eq.21
        decoder_params = list(self.model.token_decoder.parameters())
        if decoder_params:
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

        # Forward pass through NLCP model
        # Input shape: [B, L] -> Output: NLCPOutput with logits and losses
        output = self.model(
            input_ids=input_ids,
            target_ids=labels,
            padding_id=self.padding_id,
            compute_loss=True,
        )

        # Backward pass with gradient accumulation
        loss = output.total_loss
        loss.backward()

        # Gradient clipping to prevent exploding gradients
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

        # Compute average losses across all batches
        # Use max(num_batches, 1) to avoid division by zero
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

        # Phase 1: Intent planning (25% of epochs)
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

        # Phase 2: Next-Level generation alignment (25% of epochs)
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

        # Phase 3: Full pyramid joint finetuning (50% of epochs)
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

                    # Save best checkpoint based on validation loss
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


def train_nlcp(config: dict):
    """Train NLCP model with config-driven setup.

    Reference: concept-pyramid-V1.md Section 8
    Recommended Experimental Path:
        "MVP Validation: Fix K=2, run L = L_NTP + L_consist + L_CE pipeline,
        verify tensor flow and gradient closure"

    Args:
        config: Configuration dictionary loaded from YAML
    """
    # =================================================================
    # Extract config sections
    # All parameters must be defined in config files (no defaults)
    # =================================================================
    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    env_cfg = config["environment"]
    log_cfg = config["log"]
    eval_cfg = config["evaluation"]

    # Extract nlcp-specific config
    nlcp_cfg = model_cfg["nlcp_config"]
    nlcp_train_cfg = train_cfg["nlcp_training"]

    # Training hyperparameters from config
    batch_size = train_cfg["batch_size"]
    learning_rate = train_cfg["learning_rate"]
    num_epochs = train_cfg["num_epochs"]
    gradient_clip = train_cfg["gradient"]["max_grad_norm"]
    weight_decay = train_cfg["weight_decay"]
    max_steps = train_cfg["max_steps"]

    # Model device config
    model_devices_cfg = env_cfg["device_map"]

    # Logging settings
    checkpoint_dir = Path(log_cfg["checkpoint_path"])
    log_dir = Path(log_cfg["log_path"])
    log_interval = log_cfg["log_step_interval"]

    # Evaluation settings
    eval_interval = eval_cfg["eval_step_interval"]

    # Dataloader settings
    dataloader_num_workers = env_cfg["dataloader_num_workers"]

    # Create output directories
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # =================================================================
    # Setup environment
    # =================================================================
    setup_environment({"seed": env_cfg["seed"], "device": "cpu"})

    # Assign device using priority-based allocation
    model_devices = assign_model_devices(model_devices_cfg)
    device = model_devices["model"]
    if isinstance(device, str):
        device = torch.device(device)

    print(f"[1] Using device: {device}")

    # =================================================================
    # Load tokenizer and get vocab_size
    # Vocab size is derived from tokenizer, not hardcoded
    # =================================================================
    print("[2] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["lm_name"])
    vocab_size = tokenizer.vocab_size
    print(f"    Tokenizer: {model_cfg['lm_name']}, vocab_size={vocab_size}")

    # =================================================================
    # Build model config from YAML
    # All parameters from config, no defaults in code
    # =================================================================
    print("[3] Building model config...")

    # Extract number of concepts per level for l0_length and l_max derivation
    level_num_concepts = nlcp_cfg["level_num_concepts"]

    model_config = NLCPModelConfig(
        hidden_dim=nlcp_cfg["hidden_dim"],
        num_heads=nlcp_cfg["num_heads"],
        vocab_size=vocab_size,
        max_depth=nlcp_cfg["max_depth"],
        depth_gate_threshold=nlcp_cfg["depth_threshold"],
        l0_length=level_num_concepts[0],
        l_max=max(level_num_concepts),
        dropout=nlcp_cfg["dropout"],
        expansion_min=1,
        expansion_max=nlcp_cfg["max_expansion_factor"],
        depth_gate_type="causal" if nlcp_cfg["depth_gate_enabled"] else "standard",
        expansion_predictor_type=(
            "gumbel" if nlcp_cfg["expansion_mode"] == "predict" else "floor"
        ),
        cross_attention_type="relaxed",
        consistency_loss_type="directional",
    )

    # Training config from YAML
    train_config = NLCPTrainingConfig(
        lambda_consist=nlcp_train_cfg["consistency_weight"],
        lambda_depth=nlcp_train_cfg["depth_loss_weight"],
        lambda_ce=nlcp_train_cfg["ntp_weight"],
        target_expansion_ratio=nlcp_cfg["max_expansion_factor"],
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=nlcp_train_cfg["warmup_steps"],
        max_steps=max_steps,
        grad_clip_norm=gradient_clip,
        muP_scale=1.0,
    )

    # =================================================================
    # Build model
    # =================================================================
    print("[4] Building NLCP model...")
    model = build_nlcp_model(
        config=model_config,
        padding_id=0,
        num_encoder_layers=nlcp_cfg["num_encoder_layers"],
        num_generator_layers=nlcp_cfg["num_generator_layers"],
        use_info_nce=True,
        info_nce_weight=0.1,
    )
    model = model.to(device)
    print(f"    Model built with hidden_dim={model_config.hidden_dim}")

    # =================================================================
    # Load dataset using lmbase registry
    # =================================================================
    print("[5] Loading dataset...")
    train_dataset = registry.get(data_cfg, split=data_cfg["split"])
    print(f"    Dataset: {data_cfg['data_name']}, {len(train_dataset)} samples")

    # Create dataloader with collate_fn_text from ram.utils
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_text,
        drop_last=True,
        num_workers=dataloader_num_workers,
    )
    print(f"    Batches per epoch: {len(train_loader)}")

    # Validation loader using test split
    val_dataset = registry.get(data_cfg, split="test")
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_text,
        drop_last=False,
        num_workers=dataloader_num_workers,
    )

    # =================================================================
    # Create trainer and run training
    # =================================================================
    print("[6] Creating trainer...")
    trainer = NLCPTrainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=device,
        padding_id=0,
    )

    print("[7] Starting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        checkpoint_dir=str(checkpoint_dir),
        log_interval=log_interval,
        eval_interval=eval_interval,
    )

    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLCP Training")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., configs/nlcp/main.yml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train_nlcp(config)
