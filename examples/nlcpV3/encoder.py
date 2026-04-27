"""NLCP V3 Encoder: Causal Transformer for Q+CoT (Training) / Q (Inference).

USAGE:
    from nlcpV3.encoder import NLCPV3Encoder
    encoder = NLCPV3Encoder(config_dict)  # pass raw YAML dict

    # Training: Encode Q+CoT
    H = encoder.forward_training(input_ids, attention_mask)

    # Inference: Encode Q only
    H = encoder.forward_inference(input_ids, attention_mask)

DESIGN SOURCE:
    Reference: docs/concept-pyramid-V3.md
    - Section 2.2.1: Encoder
    - Section 3: Training vs Inference

PURPOSE:
    Encode input text (Q+CoT for training, Q for inference) into continuous
    hidden states H. Uses pretrained causal LM (Qwen2.5) as backbone.

    Key difference between training and inference:
    - Training: Input is Q+CoT (full reasoning context)
    - Inference: Input is Q only (no CoT available!)

ARCHITECTURE:
    Base Model: Qwen2.5-0.5B (or similar causal LM)

    Why causal LM (not encoder-only)?
    - CoT is sequential by nature (step-by-step reasoning)
    - Causal attention captures left-to-right reasoning flow
    - Pretrained on text generation, good for reasoning patterns

DIMENSION FLOW:
    Input: input_ids [B, L] (token indices)
           attention_mask [B, L] (padding mask)

    Output: H [B, L, D_encoder] (hidden states)

    Where:
        B = batch size
        L = sequence length
        D_encoder = hidden dimension of pretrained model (e.g., 896 for Qwen2.5-0.5B)

CRITICAL NOTE:
    Use AutoModel (not AutoModelForCausalLM) for feature extraction.
    We need hidden states, not token predictions.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class NLCPV3Encoder(nn.Module):
    """Encoder for NLCP V3.

    PURPOSE:
        Encode Q+CoT (training) or Q (inference) into hidden states H.
        Uses pretrained causal LM as backbone for reasoning pattern capture.

    ATTRIBUTES:
        model: Pretrained transformer model (e.g., Qwen2.5-0.5B)
        tokenizer: Tokenizer for the pretrained model

    DIMENSION FLOW:
        Constructor:
            config → initializes model with D_encoder from pretrained

        Forward Training:
            input_ids [B, L] + attention_mask [B, L] → H [B, L, D_encoder]

        Forward Inference:
            input_ids [B, L'] + attention_mask [B, L'] → H [B, L', D_encoder]
            (L' < L because no CoT)
    """

    def __init__(self, config: dict):
        """Initialize encoder with pretrained model.

        Args:
            config: Raw config dict with model.encoder settings.

        PURPOSE:
            Load pretrained model and optionally freeze parameters.
        """
        super().__init__()
        self.config = config
        encoder_cfg = config["model"]["encoder"]

        # Load pretrained model for feature extraction
        # Use AutoModel (not AutoModelForCausalLM) to get hidden states
        self.model = AutoModel.from_pretrained(encoder_cfg["encoder_model_name"])

        # Load tokenizer for potential preprocessing needs
        self.tokenizer = AutoTokenizer.from_pretrained(
            encoder_cfg["encoder_model_name"]
        )

        # Freeze encoder if specified
        if encoder_cfg["encoder_freeze"]:
            for param in self.model.parameters():
                param.requires_grad = False

        # Get encoder hidden dimension from pretrained model
        self.encoder_hidden_dim = self.model.config.hidden_size

    def forward_training(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for training: Encode Q+CoT.

        PURPOSE:
            Encode full Q+CoT text during training. This provides complete
            reasoning context for concept extraction.

        DIMENSION FLOW:
            Input:
                input_ids: [B, L] - Token indices for Q+CoT
                attention_mask: [B, L] - Padding mask (1 for real tokens, 0 for pad)

            Process:
                1. Pass through pretrained transformer
                2. Extract last hidden states

            Output:
                H: [B, L, D_encoder] - Hidden states for each token

        Args:
            input_ids: Token indices [B, L]
            attention_mask: Attention mask [B, L]

        Returns:
            H: Hidden states [B, L, D_encoder]
        """
        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )

        # Extract last hidden states
        H = outputs.last_hidden_state

        return H

    def forward_inference(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for inference: Encode Q only.

        PURPOSE:
            Encode question only during inference. No CoT is available!
            The model must generate concepts from Q alone.

        DIMENSION FLOW:
            Input:
                input_ids: [B, L'] - Token indices for Q only (L' < L)
                attention_mask: [B, L'] - Padding mask

            Output:
                H: [B, L', D_encoder] - Hidden states for question tokens

        Args:
            input_ids: Token indices [B, L']
            attention_mask: Attention mask [B, L']

        Returns:
            H: Hidden states [B, L', D_encoder]
        """
        # Same implementation as forward_training
        # But input is Q only, not Q+CoT
        return self.forward_training(input_ids, attention_mask)

    def get_encoder_hidden_dim(self) -> int:
        """Get encoder hidden dimension.

        PURPOSE:
            Return hidden dimension for downstream modules to configure
            input projection layers.

        Returns:
            Encoder hidden dimension (e.g., 896 for Qwen2.5-0.5B)
        """
        return self.encoder_hidden_dim
