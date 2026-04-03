"""C3 Unified Model for Context Cascade Compression.

This module provides a unified C3 model that combines encoder and decoder
into a single nn.Module, following the official C3 architecture pattern.

Architecture:
    C3Model (unified)
    ├── encoder (C3Encoder from ram.models.encoder)
    │   └── Small LLM (e.g., Qwen2.5-0.5B) with Context Query tokens
    └── decoder (C3Decoder from ram.models.decoder)
        └── Large LLM (e.g., Qwen2.5-1.5B) for reconstruction

Benefits:
    1. Single model for DeepSpeed training (one optimizer)
    2. Clean interface matching official C3
    3. Easy to extend with quantizers or other components
    4. Simplified checkpoint management

Official Reference:
    third-part/C3-Context-Cascade-Compression-main/C3-master/C3/model/C3.py
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple

from ram.models.encoder import build_c3_encoder
from ram.models.decoder import build_c3_decoder


class C3Model(nn.Module):
    """Unified C3 Model combining encoder and decoder.

    This class wraps the modular C3Encoder and C3Decoder from ram/models
    into a single unified model, similar to official C3QwenForCausalLM.

    Paper: "Context Cascade Compression: Exploring the Upper Limits of Text Compression"
           (arXiv:2511.15244)

    Forward Flow:
        1. Input texts -> Encoder -> latent_tokens [B, N, D_enc]
        2. latent_tokens -> Decoder -> logits [B, L, V]
        3. (Optional) Compute loss with labels

    Example:
        >>> config = {
        ...     "encoder": {...},  # C3Encoder config
        ...     "decoder": {...},  # C3Decoder config
        ... }
        >>> model = C3Model(config)
        >>>
        >>> # Training forward
        >>> logits, loss = model(
        ...     context_texts=["long context..."],
        ...     target_texts=["target text..."],
        ...     compute_loss=True
        ... )
        >>>
        >>> # Inference forward
        >>> logits = model(
        ...     context_texts=["long context..."],
        ...     prompt_texts=["prompt..."]
        ... )

    Args:
        config: Dict with 'encoder' and 'decoder' configurations

    Attributes:
        encoder: C3Encoder instance (small LLM for compression)
        decoder: C3Decoder instance (large LLM for reconstruction)
        latent_token_len: Number of latent tokens (N)
        max_length: Max sequence length
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        enc_cfg = config["encoder"]
        dec_cfg = config["decoder"]

        # Build encoder first to get hidden_dim for decoder
        self.encoder = build_c3_encoder(enc_cfg)
        encoder_hidden_dim = self.encoder.hidden_dim

        # Build decoder with encoder's hidden_dim
        self.decoder = build_c3_decoder(
            dec_cfg,
            encoder_hidden_dim=encoder_hidden_dim,
            encoder_type="C3Encoder",
        )

        # Store config attributes
        self.latent_token_len = self.encoder.latent_token_len
        self.max_length = self.encoder.max_length
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = self.decoder.hidden_dim
        self.vocab_size = self.decoder.vocab_size

        # Get tokenizer from decoder (for loss computation)
        self.tokenizer = self.decoder.tokenizer

    def forward(
        self,
        context_texts: List[str],
        target_texts: Optional[List[str]] = None,
        prompt_texts: Optional[List[str]] = None,
        compute_loss: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through C3 model.

        Args:
            context_texts: List of context texts to compress
            target_texts: List of target texts for reconstruction (training)
            prompt_texts: List of prompts for generation (inference)
            compute_loss: Whether to compute and return loss

        Returns:
            logits: [B, L, V] decoder output logits
            loss: Scalar loss if compute_loss=True and target_texts provided

        Raises:
            ValueError: If neither target_texts nor prompt_texts provided
            ValueError: If compute_loss=True but target_texts is None
        """
        # Encode context texts to latent tokens
        # latent_tokens: [B, N, D_enc]
        latent_tokens = self.encoder(inputs=context_texts)

        # Determine decoder input
        if target_texts is not None:
            # Training mode: use target texts as prompt
            decoder_input = target_texts
        elif prompt_texts is not None:
            # Inference mode: use provided prompts
            decoder_input = prompt_texts
        else:
            raise ValueError(
                "Either target_texts (for training) or prompt_texts (for inference) "
                "must be provided."
            )

        # Decode latent tokens to logits
        # logits: [B, L, V]
        logits = self.decoder(
            latent_tokens=latent_tokens,
            prompt_texts=decoder_input,
        )

        # Compute loss if requested
        loss = None
        if compute_loss and target_texts is not None:
            loss = self._compute_loss(logits, target_texts)

        return logits, loss

    def _compute_loss(
        self,
        logits: torch.Tensor,
        target_texts: List[str],
    ) -> torch.Tensor:
        """Compute cross-entropy loss for reconstruction.

        Args:
            logits: [B, L, V] decoder output
            target_texts: List of target texts

        Returns:
            loss: Scalar cross-entropy loss
        """
        # Tokenize target texts
        tokens = self.tokenizer(
            target_texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].to(logits.device)
        attention_mask = tokens["attention_mask"].to(logits.device)

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # Mask padding positions
        shift_attention_mask = attention_mask[..., 1:].contiguous()
        shift_labels = shift_labels.masked_fill(shift_attention_mask == 0, -100)

        # Compute cross-entropy loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
        )

        return loss

    def generate(
        self,
        context_texts: List[str],
        prompt: str = "Repeat the text: ",
        max_new_tokens: int = 512,
        **generate_kwargs,
    ) -> torch.Tensor:
        """Generate text from context and prompt.

        Args:
            context_texts: List of context texts to compress
            prompt: Prompt for generation (single string, applied to all)
            max_new_tokens: Maximum number of new tokens to generate
            **generate_kwargs: Additional arguments for decoder.generate()

        Returns:
            generated_ids: [B, L] generated token IDs
        """
        # Encode context
        latent_tokens = self.encoder(inputs=context_texts)

        # Generate using decoder
        generated_ids = self.decoder.generate(
            latent_tokens=latent_tokens,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )

        return generated_ids

    def get_encoder(self) -> nn.Module:
        """Get the encoder module."""
        return self.encoder

    def get_decoder(self) -> nn.Module:
        """Get the decoder module."""
        return self.decoder

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for both encoder and decoder."""
        if hasattr(self.encoder, "llm"):
            self.encoder.llm.gradient_checkpointing_enable()
        if hasattr(self.decoder, "llm"):
            self.decoder.llm.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for both encoder and decoder."""
        if hasattr(self.encoder, "llm"):
            self.encoder.llm.gradient_checkpointing_disable()
        if hasattr(self.decoder, "llm"):
            self.decoder.llm.gradient_checkpointing_disable()


def build_c3_model(config: Dict[str, Any]) -> C3Model:
    """Build C3 unified model from config.

    Args:
        config: Dict with 'encoder' and 'decoder' configurations

    Returns:
        C3Model instance

    Example:
        >>> config = {
        ...     "encoder": {
        ...         "model_name": "Qwen/Qwen2.5-0.5B",
        ...         "latent_token_len": 32,
        ...         "max_length": 2048,
        ...         "pretrained": True,
        ...         "freeze": False,
        ...     },
        ...     "decoder": {
        ...         "model_name": "Qwen/Qwen2.5-1.5B",
        ...         "latent_token_len": 32,
        ...         "pretrained": True,
        ...         "freeze": False,
        ...     },
        ... }
        >>> model = build_c3_model(config)
    """
    return C3Model(config)
