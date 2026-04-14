"""NLCP V2 Type Definitions.

DESIGN SOURCE:
    Reference: docs/concept-pyramid-V2.md
    - Section 2.1: Framework Overview

PURPOSE:
    Define type aliases and data structures used throughout the NLCP V2
    implementation. These types ensure consistent interfaces across modules
    and provide clear documentation of tensor shapes.

TYPE CONVENTIONS:
    B: Batch size
    L: Sequence length (token count)
    L_k: Concept count at level k
    D: Hidden dimension (config.hidden_dim)
    D_encoder: Encoder hidden dimension (from pretrained model)
    V: Vocabulary size
    K: Number of levels
"""

from typing import List, NamedTuple, Optional

import torch


class NLCPV2Output(NamedTuple):
    """Output structure for NLCP V2 model forward pass.

    DESIGN SOURCE: concept-pyramid-V2.md Section 2.1.3 (Training Data Flow)

    PURPOSE:
        Encapsulates all outputs from NLCP V2 forward pass for training
        and inference. Provides access to logits, hidden states, concept
        targets, and loss components.

    DIMENSION SUMMARY:
        logits: [B, L_K, V] - Vocabulary logits from final concept level
        hidden_states: List of [B, L_k, D] for k = 0, 1, ..., K-1
        concept_targets: List of [B, L_k, D] from Attentive Pooling (training only)
        total_loss: Scalar - Combined loss L_total = L_NTP + α*L_concept + β*L_recon
        ntp_loss: Scalar - Next Token Prediction loss
        concept_loss: Scalar - Concept prediction loss
        recon_loss: Scalar - Reconstruction loss

    Attributes:
        logits: [B, L_K, V] Vocabulary logits from final level H_K
            Used for: Next token prediction during training and inference
            Source: Token Decoder projection of H_K

        hidden_states: List of [B, L_k, D] Hidden states for each level
            Length: K (number of levels)
            hidden_states[k]: Concepts at level k, shape [B, L_k, D]
            Used for: Concept loss computation, intermediate representations

        concept_targets: Optional[List[torch.Tensor]] Concept targets from Attentive Pooling
            Length: K (number of levels)
            concept_targets[k]: Target concepts at level k, shape [B, L_k, D]
            Source: Attentive Pooling extraction from encoder output H
            Used for: Supervising Concept Transformer during training
            Note: Only available during training (requires Q+CoT input)

        total_loss: [1] Combined loss for backpropagation
            Formula: L_total = λ*L_NTP + α*L_concept + β*L_recon
            Used for: Gradient computation and optimization

        ntp_loss: [1] Next Token Prediction loss
            Formula: L_NTP = -sum(log P(x_t | x_{<t})) / T
            Used for: Ensuring H_K can generate correct tokens

        concept_loss: [1] Concept prediction loss
            Formula: L_concept = (1/K) * sum_k ||H_k - C_k||²
            Used for: Supervising intermediate concept layers

        recon_loss: [1] Reconstruction loss
            Formula: L_recon = ||H_hat - H_proj||²
            Used for: Ensuring Attentive Pooling preserves information
    """

    logits: torch.Tensor
    hidden_states: List[torch.Tensor]
    concept_targets: Optional[List[torch.Tensor]]
    total_loss: torch.Tensor
    ntp_loss: torch.Tensor
    concept_loss: torch.Tensor
    recon_loss: torch.Tensor
