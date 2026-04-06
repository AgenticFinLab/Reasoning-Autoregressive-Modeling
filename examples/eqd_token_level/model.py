"""EQD Token Level (Encoder-Quantizer-Decoder with Token-Level Quantization).

Unified model combining encoder, token-level VQ-VAE quantizer, and decoder.
Uses 1:1 mapping: L tokens → L discrete codes (no compression).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple

from ram.models.encoder import build_encoder
from ram.models.decoder import build_decoder


class TokenLevelQuantizer(nn.Module):
    """Token-Level Quantizer (Standard VQ-VAE).

    1:1 mapping: each token position gets one code.
    """

    def __init__(
        self,
        codebook_size: int = 4096,
        codebook_dim: int = 256,
        beta: float = 0.25,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.beta = beta

        # Codebook
        self.embedding = nn.Embedding(codebook_size, codebook_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

    def forward(self, z: torch.Tensor):
        """Token-level quantization.

        Dimension Flow:
            Input: z [B, L, D] encoder output (continuous features)
                ↓
            Flatten: z [B, L, D] -> z_flat [B*L, D]
            VQ: z_flat -> distances [B*L, K] -> indices [B*L] (nearest codebook entry)
            Lookup: indices -> q_flat [B*L, D] (codebook vectors)
            Reshape: q_flat [B*L, D] -> q [B, L, D], indices [B*L] -> indices [B, L]
                ↓
            Output: q [B, L, D], indices [B, L], vq_loss

        Args:
            z: [B, L, D] encoder output (continuous features)

        Returns:
            q: [B, L, D] quantized features (codebook vectors for each position)
            indices: [B, L] codebook indices (discrete codes for each position)
            vq_loss: scalar VQ commitment loss
        """
        B, L, D = z.shape

        # Flatten for VQ: [B, L, D] -> [B*L, D]
        # This allows vectorized distance computation to all codebook entries
        z_flat = z.reshape(-1, D)  # [B*L, D] - use reshape for safety

        # Compute distances to codebook
        distances = torch.cdist(z_flat, self.embedding.weight)  # [B*L, K]
        indices = torch.argmin(distances, dim=-1)  # [B*L]

        # Get quantized vectors
        q_flat = self.embedding(indices)  # [B*L, D]
        # Use reshape for safety (embedding output is contiguous but reshape is safer)
        q = q_flat.reshape(B, L, D)  # [B, L, D]
        indices = indices.reshape(B, L)  # [B, L]

        # VQ loss
        vq_loss = F.mse_loss(q.detach(), z) + self.beta * F.mse_loss(q, z.detach())

        return q, indices, vq_loss


class EQDTokenLevelModel(nn.Module):
    """Unified EQD Token Level Model."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        enc_cfg = config["encoder"]
        dec_cfg = config["decoder"]
        quant_cfg = config.get("quantizer", {})

        # Build encoder
        self.encoder = build_encoder(enc_cfg)
        encoder_hidden_dim = self.encoder.output_dim

        # Build quantizer
        self.quantizer = TokenLevelQuantizer(
            codebook_size=quant_cfg.get("codebook_size", 4096),
            codebook_dim=quant_cfg.get("codebook_dim", encoder_hidden_dim),
            beta=quant_cfg.get("beta", 0.25),
        )

        # Build decoder
        self.decoder = build_decoder(
            dec_cfg,
            input_dim=self.quantizer.codebook_dim,
        )

        # Store attributes
        self.hidden_dim = encoder_hidden_dim
        self.vocab_size = self.decoder.vocab_size
        self.encoder_model_name = enc_cfg.get("model_name", "unknown")
        self.decoder_model_name = dec_cfg.get("model_name", "unknown")
        self.enc_tokenizer = self.encoder.tokenizer
        self.dec_tokenizer = self.decoder.tokenizer

    def forward(
        self,
        texts: List[str],
        compute_loss: bool = False,
        vq_loss_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass."""
        # Encode
        z = self.encoder(texts)

        # Quantize
        q, indices, vq_loss = self.quantizer(z)

        # Decode
        logits = self.decoder(q)

        # Compute loss
        loss = None
        if compute_loss:
            target_ids = self.dec_tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=logits.size(1),
            )["input_ids"].to(logits.device)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(
                ignore_index=self.dec_tokenizer.pad_token_id or -100
            )
            recon_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            loss = recon_loss + vq_loss_weight * vq_loss

        return logits, loss, vq_loss if compute_loss else None

    def gradient_checkpointing_enable(self):
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        if hasattr(self.decoder, "gradient_checkpointing_enable"):
            self.decoder.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        if hasattr(self.encoder, "gradient_checkpointing_disable"):
            self.encoder.gradient_checkpointing_disable()
        if hasattr(self.decoder, "gradient_checkpointing_disable"):
            self.decoder.gradient_checkpointing_disable()
