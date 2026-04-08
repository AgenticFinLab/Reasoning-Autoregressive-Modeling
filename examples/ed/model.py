"""ED (Encoder-Decoder) Unified Model.

This module provides a unified Encoder-Decoder model for text reconstruction.
Combines encoder and decoder into a single nn.Module for flexible training.

Architecture:
    EDModel (unified)
    ├── encoder (from ram.models.encoder)
    │   └── BERT or similar encoder
    └── decoder (from ram.models.decoder)
        └── GPT2 or similar decoder

Benefits:
    1. Single model for training (one optimizer)
    2. Clean interface
    3. Simplified checkpoint management
    4. Compatible with both basic and DeepSpeed training
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ram.models.decoder import build_decoder
from ram.models.encoder import build_encoder


class EDModel(nn.Module):
    """Unified Encoder-Decoder Model.

    This class wraps the modular encoder and decoder into a single unified model.

    Forward Flow:
        1. Input texts -> Encoder -> hidden [B, L, D_enc]
        2. hidden -> Decoder -> logits [B, L, V]
        3. (Optional) Compute loss with labels

    Example:
        >>> config = {
        ...     "encoder": {"model_name": "bert-base-uncased", ...},
        ...     "decoder": {"model_name": "gpt2", ...},
        ... }
        >>> model = EDModel(config)
        >>>
        >>> # Training forward
        >>> logits, loss = model(
        ...     texts=["input text..."],
        ...     compute_loss=True
        ... )

    Args:
        config: Dict with 'encoder' and 'decoder' configurations

    Attributes:
        encoder: Encoder instance (e.g., BERT)
        decoder: Decoder instance (e.g., GPT2)
        hidden_dim: Encoder hidden dimension
        vocab_size: Decoder vocabulary size
        encoder_model_name: Name of the encoder model
        decoder_model_name: Name of the decoder model
        enc_tokenizer: Encoder tokenizer
        dec_tokenizer: Decoder tokenizer
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the ED model.

        Args:
            config: Configuration dictionary containing 'encoder' and 'decoder' settings
        """
        super().__init__()

        enc_cfg = config["encoder"]
        dec_cfg = config["decoder"]

        # Build encoder
        self.encoder = build_encoder(enc_cfg)
        encoder_hidden_dim = self.encoder.output_dim

        # Build decoder with encoder's output dim as input
        self.decoder = build_decoder(
            dec_cfg,
            input_dim=encoder_hidden_dim,
        )

        # Store attributes
        self.hidden_dim = encoder_hidden_dim
        self.vocab_size = self.decoder.vocab_size
        self.encoder_model_name = enc_cfg.get("model_name", "unknown")
        self.decoder_model_name = dec_cfg.get("model_name", "unknown")

        # Store tokenizer references
        self.enc_tokenizer = self.encoder.tokenizer
        self.dec_tokenizer = self.decoder.tokenizer

    def forward(
        self, texts: List[str], compute_loss: bool = False, **loss_kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through encoder-decoder.

        Dimension Flow:
            texts: List[str] with B texts
                ↓
            hidden: [B, L, D] encoder output (continuous features)
                ↓
            logits: [B, L, V] decoder output (vocabulary logits)

        Args:
            texts: List of input texts (batch of B texts)
            compute_loss: Whether to compute and return loss
            **loss_kwargs: Additional arguments for loss computation

        Returns:
            Tuple of (logits, loss) where:
                logits: [B, L, V] vocabulary logits for each position
                loss: Scalar cross-entropy loss (if compute_loss=True, else None)
        """
        # Encode: texts -> hidden [B, L, D]
        # Encoder processes raw texts and outputs continuous features
        hidden = self.encoder(texts)

        # Decode: hidden -> logits [B, L, V]
        logits = self.decoder(hidden)

        # Compute loss if requested
        loss = None
        if compute_loss:
            loss = self.compute_loss(logits, texts, **loss_kwargs)

        return logits, loss

    def compute_loss(
        self, logits: torch.Tensor, target_texts: List[str], **kwargs
    ) -> torch.Tensor:
        """Compute reconstruction loss.

        Uses cross-entropy loss for next-token prediction.

        Args:
            logits: [B, L, V] token logits from decoder
            target_texts: List of target texts
            **kwargs: Additional loss arguments (unused)

        Returns:
            loss: Scalar loss value
        """
        # Tokenize targets with decoder's tokenizer
        target_ids = self.dec_tokenizer(
            target_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=logits.size(1),
        )["input_ids"].to(logits.device)

        # Shift for next-token prediction
        # logits[..., :-1, :]: predictions for tokens 1..L
        # target_ids[..., 1:]: target tokens 1..L
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()

        # Compute cross-entropy loss
        loss_fct = nn.CrossEntropyLoss(
            ignore_index=self.dec_tokenizer.pad_token_id or -100
        )
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        return loss

    def generate(
        self, texts: List[str], max_length: int = 128, **generate_kwargs
    ) -> List[str]:
        """Generate text from input.

        Performs greedy generation by taking argmax at each step.

        Args:
            texts: List of input texts
            max_length: Maximum generation length
            **generate_kwargs: Additional generation arguments (unused)

        Returns:
            List of generated texts
        """
        self.eval()
        with torch.no_grad():
            # Encode
            hidden = self.encoder(texts)

            # Decode (greedy generation)
            logits = self.decoder(hidden)

            # Get predictions
            pred_ids = torch.argmax(logits, dim=-1)

            # Decode to texts
            generated_texts = self.dec_tokenizer.batch_decode(
                pred_ids, skip_special_tokens=True
            )

        return generated_texts

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        if hasattr(self.decoder, "gradient_checkpointing_enable"):
            self.decoder.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        if hasattr(self.encoder, "gradient_checkpointing_disable"):
            self.encoder.gradient_checkpointing_disable()
        if hasattr(self.decoder, "gradient_checkpointing_disable"):
            self.decoder.gradient_checkpointing_disable()
