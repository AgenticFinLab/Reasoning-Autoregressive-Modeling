"""NLCP V2 Encoder Module.

DESIGN SOURCE:
    Reference: docs/concept-pyramid-V2.md
    - Section 2.2.1: Encoder
    - Section 2.1.3: Training Data Flow
    - Section 2.1.4: Inference Data Flow

PURPOSE:
    Encode input text (Q+CoT for training, Q for inference) into hidden
    representations. Uses pretrained HuggingFace models for efficient
    feature extraction with optional freezing.

TRAINING-INFERENCE SEPARATION:
    Training mode: Encodes full Q+CoT sequence for Attentive Pooling
    Inference mode: Encodes only Q, then pools/projects to initial concept H_0

DIMENSION FLOW:
    Training:
        Input:  [B, L] token IDs (Q+CoT)
        Output: [B, L, D_encoder] hidden states (D_encoder from pretrained model)

    Inference:
        Input:  [B, L_q] token IDs (Q only)
        Output: [B, L_0, D] initial concept H_0
            where L_0 = config.level_lengths[0]
            and D = config.hidden_dim
"""

from typing import Optional

import torch
import torch.nn as nn

from nlcpV2.config import NLCPV2Config


class NLCPV2Encoder(nn.Module):
    """Encoder for NLCP V2 with training-inference separation.

    DESIGN SOURCE: concept-pyramid-V2.md Section 2.2.1

    ARCHITECTURE:
        Standard Causal Transformer that reuses pretrained weights.
        Uses HuggingFace Model classes (e.g., Qwen2Model) for feature extraction.

        Why HuggingFace Model (not ForCausalLM)?
            - Model class outputs hidden states only (no lm_head)
            - Suitable for feature extraction as Encoder
            - ForCausalLM includes token prediction head (unnecessary for Encoder)

    TRAINING MODE (forward_training):
        Input:  Q+CoT token IDs [B, L]
        Process: Full transformer encoding with causal attention
        Output: H ∈ R^{L×D_encoder} (full token-level representations)

        Purpose: Produce representations for Attentive Pooling to extract concepts

    INFERENCE MODE (forward_inference):
        Input:  Q token IDs [B, L_q]
        Process:
            1. Encode Q with transformer
            2. Pool: L_q → L_0 (adaptive average pooling)
            3. Project: D_encoder → D
        Output: H_0 ∈ R^{L_0×D} (initial concept)

        Purpose: Generate initial concept for Concept Transformer to refine

    DIMENSION TRANSFORMATION:
        Pretrained model hidden dim (D_encoder) may differ from config.hidden_dim (D).
        Example: Qwen2.5-0.5B has D_encoder=896, but config may use D=256.

        Training output: [B, L, D_encoder] (no projection needed)
        Inference output: [B, L_0, D] (projection applied after pooling)

    Attributes:
        config: NLCPV2Config instance with encoder parameters
        model: HuggingFace transformer backbone (e.g., Qwen2Model)
        pool_to_l0: Adaptive pooling to compress L_q → L_0 concepts
        l0_proj: Linear projection from D_encoder to D
    """

    def __init__(self, config: NLCPV2Config):
        """Initialize encoder with configuration.

        INITIALIZATION PROCESS:
            1. Load pretrained HuggingFace model
            2. Optionally freeze model parameters
            3. Create pooling layer for L_0 concept compression
            4. Create projection layer for dimension alignment

        Args:
            config: NLCPV2Config with encoder_model_name, encoder_freeze, etc.
        """
        super().__init__()
        self.config = config

        self.model = self._load_hf_model(config)

        if config.encoder_freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.pool_to_l0 = nn.AdaptiveAvgPool1d(config.level_lengths[0])

        encoder_hidden = self.model.config.hidden_size
        self.l0_proj = nn.Linear(encoder_hidden, config.hidden_dim)

    def _load_hf_model(self, config: NLCPV2Config):
        """Load HuggingFace model for encoder backbone.

        PURPOSE:
            Load pretrained transformer model from HuggingFace Hub.
            Optionally truncate to fewer layers for efficiency.

        SUPPORTED MODELS:
            - GPT-2 family: GPT2Model
            - Llama family: LlamaModel
            - Qwen family: Qwen2Model

        LAYER TRUNCATION:
            If encoder_num_layers is specified, only keep first N layers.
            This reduces computation while maintaining representation quality.

        DIMENSION NOTE:
            Model's hidden_size (D_encoder) is determined by pretrained weights.
            This may differ from config.hidden_dim (D).

        Args:
            config: Configuration with encoder_model_name and encoder_num_layers

        Returns:
            HuggingFace model instance (e.g., Qwen2Model)
        """
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            config.encoder_model_name,
            torch_dtype=torch.float32,
        )

        if config.encoder_num_layers is not None:
            if hasattr(model, "layers"):
                model.layers = model.layers[: config.encoder_num_layers]
            elif hasattr(model, "model") and hasattr(model.model, "layers"):
                model.model.layers = model.model.layers[: config.encoder_num_layers]

        return model

    def forward_training(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for training mode.

        DESIGN SOURCE: concept-pyramid-V2.md Section 2.1.3

        PURPOSE:
            Encodes Q+CoT to produce full token-level representations.
            These representations are used by Attentive Pooling to extract
            hierarchical concept targets C_0, C_1, ..., C_K.

        DIMENSION FLOW:
            Input:  [B, L] Q+CoT token IDs
                ↓
            Transformer Encoding (causal attention)
                ↓
            Output: [B, L, D_encoder] Token-level hidden states H

        WHY CAUSAL ATTENTION?
            Ensures each position only attends to previous positions,
            maintaining autoregressive property for language modeling.

        Args:
            input_ids: [B, L] Q+CoT token IDs
            attention_mask: [B, L] Attention mask (optional)
                1 for real tokens, 0 for padding

        Returns:
            [B, L, D_encoder] Token-level hidden states H
                D_encoder = self.model.config.hidden_size
                (May differ from config.hidden_dim)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = outputs.last_hidden_state

        return hidden_states

    def forward_inference(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for inference mode.

        DESIGN SOURCE: concept-pyramid-V2.md Section 2.1.4

        PURPOSE:
            Encodes Q and produces initial concept H_0 via Pool & Project.
            H_0 serves as the starting point for Concept Transformer to
            generate hierarchical concepts H_1, H_2, ..., H_K.

        DIMENSION FLOW:
            Input:  [B, L_q] Q token IDs
                ↓
            Transformer Encoding
                ↓
            [B, L_q, D_encoder] hidden states
                ↓
            Transpose: [B, D_encoder, L_q]
                ↓
            Adaptive Pooling: L_q → L_0
                ↓
            Transpose: [B, L_0, D_encoder]
                ↓
            Linear Projection: D_encoder → D
                ↓
            Output: [B, L_0, D] Initial concept H_0

        POOLING OPERATION:
            AdaptiveAvgPool1d compresses variable-length question (L_q)
            to fixed number of concepts (L_0).

            Example: L_q=50 tokens → L_0=4 concepts via average pooling

        PROJECTION OPERATION:
            Aligns encoder dimension (D_encoder) with concept dimension (D).

            Example: D_encoder=896 → D=256 via linear projection

        Args:
            input_ids: [B, L_q] Q token IDs (question only, no CoT)
            attention_mask: [B, L_q] Attention mask (optional)

        Returns:
            [B, L_0, D] Initial concept H_0
                L_0 = config.level_lengths[0]
                D = config.hidden_dim
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = outputs.last_hidden_state

        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.pool_to_l0(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)

        H_0 = self.l0_proj(hidden_states)

        return H_0
