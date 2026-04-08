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
from ram.models.decoder import (
    build_c3_decoder,
    C3_IM_START_TOKEN,
    C3_IM_END_TOKEN,
    C3_IM_PATCH_TOKEN,
)
from ram.losses import DualTokenizerReconstructionLoss


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
        >>> # Reconstruction mode (default prompt: "Repeat the text: ")
        >>> logits, loss = model(
        ...     texts=["input text..."],
        ...     compute_loss=True
        ... )
        >>>
        >>> # Custom generation prompts
        >>> logits = model(
        ...     texts=["input text..."],
        ...     decode_prompts=["Summarize: "]
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

        # Create loss function
        self.loss_fn = DualTokenizerReconstructionLoss(
            dec_tokenizer=self.tokenizer,
            dec_vocab_size=self.vocab_size,
            ignore_index=-100,
            max_length=self.max_length,
            label_smoothing=0.0,
            latent_token_len=self.latent_token_len,
        )

    def forward(
        self,
        texts: List[str],
        decode_prompts: Optional[List[str]] = None,
        compute_loss: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through C3 model.

        Args:
            texts: List of input texts to compress
            decode_prompts: Optional list of prompts for decoder.
                           If None, uses default "Repeat the text: " prompt.
                           If provided, uses custom prompts for generation.
            compute_loss: Whether to compute and return loss

        Returns:
            logits: [B, L, V] decoder output logits
            loss: Scalar loss if compute_loss=True
        """
        # Encode texts to latent tokens
        # latent_tokens: [B, N, D_enc]
        latent_tokens = self.encoder(inputs=texts)

        # Determine decoder prompts
        if decode_prompts is not None:
            # Use provided prompts
            decoder_prompts = decode_prompts
        else:
            # Default reconstruction prompt
            decoder_prompts = ["Repeat the text: "] * len(texts)

        # Wrap prompts with special tokens for C3
        # Format: <img> <imgpad>*N </img> prompt
        wrapped_prompts = []
        for prompt in decoder_prompts:
            wrapped = (
                C3_IM_START_TOKEN
                + C3_IM_PATCH_TOKEN * self.latent_token_len
                + C3_IM_END_TOKEN
                + "\n"
                + prompt
            )
            wrapped_prompts.append(wrapped)

        # Tokenize wrapped prompts
        prompt_ids = self.tokenizer(
            wrapped_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
            + self.latent_token_len
            + 3,  # Account for special tokens
        )["input_ids"].to(latent_tokens.device)

        # Decode latent tokens to logits
        # logits: [B, L, V]
        logits = self.decoder(
            latent_tokens=latent_tokens,
            prompt_ids=prompt_ids,
        )

        # Compute loss if requested
        loss = None
        if compute_loss:
            loss = self._compute_loss(logits, texts)

        return logits, loss

    def _compute_loss(
        self,
        logits: torch.Tensor,
        target_texts: List[str],
    ) -> torch.Tensor:
        """Compute cross-entropy loss for reconstruction.

        Uses DualTokenizerReconstructionLoss which handles latent token skipping.

        Args:
            logits: [B, N+L, V] decoder output (includes latent tokens)
            target_texts: List of target texts

        Returns:
            loss: Scalar cross-entropy loss
        """
        loss, _ = self.loss_fn(logits, target_texts)
        return loss

    def generate(
        self,
        texts: List[str],
        prompt: str = "Repeat the text: ",
        max_new_tokens: int = 512,
        **generate_kwargs,
    ) -> torch.Tensor:
        """Generate text from compressed texts and prompt.

        Args:
            texts: List of input texts to compress
            prompt: Prompt for generation (single string, applied to all)
            max_new_tokens: Maximum number of new tokens to generate
            **generate_kwargs: Additional arguments for decoder.generate()

        Returns:
            generated_ids: [B, L] generated token IDs
        """
        # Encode texts
        latent_tokens = self.encoder(inputs=texts)

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
