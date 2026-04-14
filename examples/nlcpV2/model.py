"""NLCP V2 Complete Model Implementation.

DESIGN SOURCE:
    Reference: docs/concept-pyramid-V2.md
    - Section 2: Architecture Overview
    - Section 3: Training Methodology
    - Section 4: Inference Pipeline

PURPOSE:
    Implements the complete NLCP V2 model with training-inference separation.
    This is the top-level module that orchestrates all four components:
    Encoder, Attentive Pooling, Concept Transformer, and Token Decoder.

ARCHITECTURE OVERVIEW:
    ┌─────────────────────────────────────────────────────────────┐
    │                    NLCP V2 Model                            │
    ├─────────────────────────────────────────────────────────────┤
    │  ┌─────────────┐                                            │
    │  │   Encoder   │  Q+CoT → H (training)                      │
    │  │             │  Q → H_0 (inference)                       │
    │  └──────┬──────┘                                            │
    │         │                                                   │
    │  ┌──────▼──────┐  (training only)                           │
    │  │ Attentive   │  H → C_0, C_1, ..., C_K                   │
    │  │ Pooling     │  Extracts concept targets                 │
    │  └─────────────┘                                            │
    │         │                                                   │
    │  ┌──────▼──────────────┐                                    │
    │  │ Concept Transformer │  H_0 → H_1 → ... → H_K            │
    │  │                     │  Next-level generation            │
    │  └──────────┬──────────┘                                    │
    │             │                                               │
    │  ┌──────────▼──────────┐                                    │
    │  │    Token Decoder    │  H_K → tokens                     │
    │  │                     │  Cross-attention generation       │
    │  └─────────────────────┘                                    │
    └─────────────────────────────────────────────────────────────┘

TRAINING-INFERENCE SEPARATION:
    ┌─────────────────────────────────────────────────────────────┐
    │  TRAINING (requires Q+CoT)        INFERENCE (Q only)        │
    ├─────────────────────────────────────────────────────────────┤
    │  1. Encoder: Q+CoT → H            1. Encoder: Q → H_0       │
    │  2. Attentive Pooling: H → C_k    2. (skip Attentive        │
    │  3. Concept Transformer: H_0 → H_k   Pooling - no CoT)      │
    │  4. Token Decoder: H_K → logits   3. Concept Transformer    │
    │  5. Compute L_NTP, L_concept,        H_0 → H_K              │
    │     L_recon                     4. Token Decoder: H_K →     │
    │                                     generated tokens        │
    └─────────────────────────────────────────────────────────────┘

LOSS FUNCTION:
    L_total = λ * L_NTP + α * L_concept + β * L_recon

    Where:
        L_NTP: Next Token Prediction loss (standard cross-entropy)
        L_concept: Concept prediction loss (MSE between H_k and C_k)
        L_recon: Reconstruction loss (MSE between H_hat and H)
        λ, α, β: Loss weights from config
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from nlcpV2.config import NLCPV2Config
from nlcpV2.types import NLCPV2Output
from nlcpV2.utils import compute_ntp_loss
from nlcpV2.encoder import NLCPV2Encoder
from nlcpV2.attentive_pooling import ResidualAttentivePooling
from nlcpV2.concept_transformer import ConceptTransformer
from nlcpV2.token_decoder import TokenDecoder


class NLCPV2Model(nn.Module):
    """Next-Level Concept Pyramid V2 Model.

    DESIGN SOURCE: concept-pyramid-V2.md Section 2.1

    ARCHITECTURE:
        Four core modules working together:
        1. Encoder: Encodes input text to hidden representations
           - Training: Encodes Q+CoT to H
           - Inference: Encodes Q and pools/projects to H_0

        2. Attentive Pooling: Extracts hierarchical concept targets
           - Only used during training (requires ground-truth CoT)
           - Implements residual attentive pooling
           - Produces C_0, C_1, ..., C_K as supervision targets

        3. Concept Transformer: Generates concepts level by level
           - Implements VAR-style "Next-Scale" generation
           - Inter-level causal, intra-level parallel
           - H_0 → H_1 → ... → H_K

        4. Token Decoder: Projects concepts to vocabulary
           - Uses causal cross-attention (tokens Q, concepts KV)
           - Generates tokens autoregressively
           - H_K → vocabulary logits

    TRAINING-INFERENCE SEPARATION:
        This is a key design principle of NLCP V2:

        Training:
            - Input: Q+CoT (full reasoning chain)
            - Uses Attentive Pooling to extract concept targets
            - Supervises Concept Transformer with these targets
            - End-to-end gradient flow

        Inference:
            - Input: Q only (question without answer)
            - No Attentive Pooling (no CoT to extract from)
            - Concept Transformer predicts concepts directly
            - Token Decoder generates answer tokens

    LOSS FUNCTION:
        L_total = λ * L_NTP + α * L_concept + β * L_recon

        L_NTP ensures H_K can generate correct tokens.
        L_concept ensures intermediate concepts are meaningful.
        L_recon ensures Attentive Pooling preserves information.

    Attributes:
        config: NLCPV2Config instance with all hyperparameters
        encoder: NLCPV2Encoder for input encoding
        attentive_pooling: ResidualAttentivePooling (training only)
        concept_transformer: ConceptTransformer for concept generation
        token_decoder: TokenDecoder for vocabulary projection
    """

    def __init__(self, config: NLCPV2Config):
        """Initialize NLCP V2 model.

        INITIALIZATION:
            Creates all four core modules:
            1. Encoder with pretrained model loading
            2. Attentive Pooling with learnable queries
            3. Concept Transformer with K layers
            4. Token Decoder with cross-attention

        Args:
            config: NLCPV2Config with all hyperparameters
        """
        super().__init__()
        self.config = config

        self.encoder = NLCPV2Encoder(config)
        self.attentive_pooling = ResidualAttentivePooling(config)
        self.concept_transformer = ConceptTransformer(config)
        self.token_decoder = TokenDecoder(config)

    def forward_training(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        padding_id: int,
    ) -> NLCPV2Output:
        """Forward pass for training.

        DESIGN SOURCE: concept-pyramid-V2.md Section 2.1.3 (Training Data Flow)

        TRAINING DATA FLOW:
            Input: Q+CoT token IDs [B, L]
                ↓
            1. Encoder: Q+CoT → H
               H: [B, L, D_encoder] (full token representations)
                ↓
            2. Attentive Pooling: H → C_0, C_1, ..., C_K
               concepts: List of [B, L_k, D] (concept targets)
               H_hat: [B, L, D] (reconstructed H)
               H_rest: [B, L, D] (residual, should be ~0)
                ↓
            3. Pool & Project: H → H_0
               H_0: [B, L_0, D] (initial concept for transformer)
                ↓
            4. Concept Transformer: H_0 → H_0, H_1, ..., H_K
               hidden_states: List of [B, L_k, D] (predicted concepts)
                ↓
            5. Token Decoder: H_K, target_ids → logits
               logits: [B, T, V] (vocabulary logits)
                ↓
            6. Compute losses:
               L_NTP: Cross-entropy on logits vs target_ids
               L_concept: MSE between hidden_states and concept_targets
               L_recon: MSE of H_rest (should be small)
                ↓
            Output: NLCPV2Output with all losses and states

        DIMENSION FLOW:
            input_ids: [B, L]
            target_ids: [B, T]
                ↓
            H: [B, L, D_encoder]
                ↓
            concepts[k]: [B, L_k, D] for k = 0, ..., K-1
            H_hat: [B, L, D]
            H_rest: [B, L, D]
                ↓
            H_0: [B, L_0, D]
                ↓
            hidden_states[k]: [B, L_k, D] for k = 0, ..., K-1
                ↓
            logits: [B, T, V]

        Args:
            input_ids: [B, L] Q+CoT token IDs
                Full question + chain-of-thought tokens
            target_ids: [B, T] Target token IDs for NTP loss
                Typically the CoT portion shifted by one position
            padding_id: Padding token ID for loss computation

        Returns:
            NLCPV2Output with:
                - logits: [B, T, V] for next token prediction
                - hidden_states: List of [B, L_k, D] predicted concepts
                - concept_targets: List of [B, L_k, D] target concepts
                - total_loss: Combined loss for backpropagation
                - ntp_loss: Next token prediction loss
                - concept_loss: Concept prediction loss
                - recon_loss: Reconstruction loss
        """
        H = self.encoder.forward_training(input_ids)

        concept_targets, H_hat, H_rest = self.attentive_pooling(H)

        H_0 = self._pool_and_project(H)

        hidden_states = self.concept_transformer(H_0)

        logits = self.token_decoder(hidden_states[-1], target_ids)

        ntp_loss = compute_ntp_loss(logits, target_ids, padding_id)

        concept_loss = self._compute_concept_loss(hidden_states, concept_targets)

        recon_loss = torch.mean(H_rest**2)

        total_loss = (
            self.config.ntp_loss_weight * ntp_loss
            + self.config.concept_loss_weight * concept_loss
            + self.config.recon_loss_weight * recon_loss
        )

        return NLCPV2Output(
            logits=logits,
            hidden_states=hidden_states,
            concept_targets=concept_targets,
            total_loss=total_loss,
            ntp_loss=ntp_loss,
            concept_loss=concept_loss,
            recon_loss=recon_loss,
        )

    def forward_inference(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for inference.

        DESIGN SOURCE: concept-pyramid-V2.md Section 2.1.4 (Inference Data Flow)

        INFERENCE DATA FLOW:
            Input: Q token IDs [B, L_q] (question only, no CoT)
                ↓
            1. Encoder: Q → H_0 (Pool & Project)
               H_0: [B, L_0, D] (initial concept)
               Note: No Attentive Pooling (no CoT to extract from)
                ↓
            2. Concept Transformer: H_0 → H_0, H_1, ..., H_K
               hidden_states: List of [B, L_k, D]
                ↓
            Output: H_K [B, L_K, D] (final level concepts)

        DIMENSION FLOW:
            input_ids: [B, L_q]
                ↓
            H_0: [B, L_0, D]
                ↓
            hidden_states[k]: [B, L_k, D] for k = 0, ..., K-1
                ↓
            H_K: [B, L_K, D] (where K = num_levels)

        Args:
            input_ids: [B, L_q] Q token IDs (question only)

        Returns:
            [B, L_K, D] Final level concepts H_K
                L_K = config.level_lengths[-1]
                These concepts will be used by Token Decoder for generation
        """
        H_0 = self.encoder.forward_inference(input_ids)

        hidden_states = self.concept_transformer(H_0)

        return hidden_states[-1]

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressive generation.

        DESIGN SOURCE: concept-pyramid-V2.md Section 4: Inference Pipeline

        GENERATION PIPELINE:
            Input: Q token IDs [B, L_q]
                ↓
            1. Encode Q → H_0
                ↓
            2. Concept Transformer: H_0 → H_K
                ↓
            3. Token Decoder generates tokens autoregressively:
               For t = 1 to max_new_tokens:
                   a. H_K attends to generated tokens so far
                   b. Predict next token distribution
                   c. Sample next token
                   d. Append to sequence
                ↓
            Output: Q + generated tokens [B, L_q + max_new_tokens]

        DIMENSION FLOW:
            input_ids: [B, L_q]
                ↓
            H_0: [B, L_0, D]
                ↓
            H_K: [B, L_K, D]
                ↓
            generated: [B, max_new_tokens]
                ↓
            output: [B, L_q + max_new_tokens]

        Args:
            input_ids: [B, L_q] Q token IDs
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (default: 1.0)
                Lower = more deterministic, Higher = more random
            top_k: Top-k sampling parameter (default: 0, disabled)
                Only sample from top k most likely tokens
            top_p: Top-p (nucleus) sampling parameter (default: 1.0, disabled)
                Sample from smallest set with cumulative prob >= p

        Returns:
            [B, L_q + max_new_tokens] Generated token IDs
                Concatenation of input Q and generated answer
        """
        H_0 = self.encoder.forward_inference(input_ids)
        hidden_states = self.concept_transformer(H_0)
        H_K = hidden_states[-1]

        generated_tokens = self.token_decoder.generate(
            H_K,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        return torch.cat([input_ids, generated_tokens], dim=-1)

    def _pool_and_project(self, H: torch.Tensor) -> torch.Tensor:
        """Pool and project encoder output to initial concept H_0.

        PURPOSE:
            Converts encoder output H (from Q+CoT) to initial concept H_0
            that matches the dimension expected by Concept Transformer.

        OPERATIONS:
            1. Transpose: [B, L, D] → [B, D, L]
            2. Adaptive pooling: L → L_0 (compress sequence)
            3. Transpose: [B, D, L_0] → [B, L_0, D]
            4. Linear projection: D_encoder → D (align dimensions)

        DIMENSION FLOW:
            Input H: [B, L, D_encoder]
                ↓
            Transpose: [B, D_encoder, L]
                ↓
            AdaptiveAvgPool1d(L → L_0): [B, D_encoder, L_0]
                ↓
            Transpose: [B, L_0, D_encoder]
                ↓
            Linear projection: [B, L_0, D]
                ↓
            Output H_0: [B, L_0, D]

        WHY ADAPTIVE POOLING?
            Encoder output has variable length L (depends on input).
            Concept Transformer expects fixed L_0 concepts.
            Adaptive pooling compresses variable L to fixed L_0.

        Args:
            H: [B, L, D_encoder] Encoder output
                L = sequence length (variable)
                D_encoder = encoder hidden dimension

        Returns:
            [B, L_0, D] Initial concept H_0
                L_0 = config.level_lengths[0]
                D = config.hidden_dim
        """
        H_transposed = H.transpose(1, 2)
        pool = nn.AdaptiveAvgPool1d(self.config.level_lengths[0])
        H_pooled = pool(H_transposed)
        H_pooled = H_pooled.transpose(1, 2)

        projection = nn.Linear(H.size(-1), self.config.hidden_dim, device=H.device)
        H_0 = projection(H_pooled)

        return H_0

    def _compute_concept_loss(
        self,
        hidden_states: List[torch.Tensor],
        concept_targets: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute concept prediction loss.

        PURPOSE:
            Measures how well Concept Transformer predicts the concept targets
            extracted by Attentive Pooling.

        MATHEMATICAL FORMULATION:
            L_concept = (1/K) * sum_{k=0}^{K-1} MSE(H_k, C_k)

            Where:
                K = number of levels
                H_k = predicted concepts from Concept Transformer
                C_k = target concepts from Attentive Pooling
                MSE = mean squared error

        DIMENSION:
            hidden_states[k]: [B, L_k, D]
            concept_targets[k]: [B, L_k, D]
            loss: scalar

        Args:
            hidden_states: List of K tensors [B, L_k, D]
                Predicted concepts from Concept Transformer
            concept_targets: List of K tensors [B, L_k, D]
                Target concepts from Attentive Pooling

        Returns:
            Scalar concept loss (average MSE across all levels)
        """
        total_loss = 0.0
        num_levels = len(hidden_states)

        for k in range(num_levels):
            pred = hidden_states[k]
            target = concept_targets[k]

            loss = F.mse_loss(pred, target)
            total_loss = total_loss + loss

        return total_loss / num_levels
