"""Combined Losses for VQ-AE Text Autoencoder.

This module provides combined loss functions that integrate:
- Reconstruction loss (token-level)
- VQ loss (quantization regularization)
- Optional KL divergence (for VAE variants)

Combined Loss for VQ-AE:
========================

Standard VQ-AE Loss:
    L_total = L_recon + λ_vq * L_vq

    Where:
        L_recon = CrossEntropy(logits, target_ids)
            - Reconstruction at token level
            - logits [B, L, V] vs target_ids [B, L]

        L_vq = ||sg[z] - q||² + β * ||z - sg[q]||²
            - Vector quantization regularization
            - Keeps encoder close to codebook

        λ_vq = VQ loss weight (typically 1.0)
        β = Commitment cost (typically 0.25)

Flow Diagram:
=============
    texts
      ↓
    Encoder → z [B, L, D]
      ↓
    Quantizer → f_hat [B, L, D], vq_loss
      ↓
    Decoder → logits [B, L, V]
      ↓
    ┌─────────────────────────────────────┐
    │ L_recon = CE(logits, target_ids)    │
    │ L_vq = vq_loss (from quantizer)     │
    │ L_total = L_recon + λ * L_vq        │
    └─────────────────────────────────────┘

Tokenizer Scenarios:
====================

Case 1: Same Tokenizer (T5, BART)
    target_ids = input_ids (from shared tokenizer)
    → Standard CrossEntropy

Case 2: Different Tokenizers (BERT + GPT2)
    target_ids = dec_tokenizer(texts)  # Must re-tokenize!
    → Use DualTokenizerReconstructionLoss

See reconstruction.py for detailed tokenizer handling.
"""

from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn

from .reconstruction import (
    ReconstructionLoss,
    DualTokenizerReconstructionLoss,
    compute_reconstruction_loss,
)
from .vq_loss import VQLoss, compute_vq_loss


class VQAELoss(nn.Module):
    """Combined loss for VQ-AE (Vector Quantized AutoEncoder).

    Combines reconstruction loss and VQ loss with configurable weights.

    Total Loss:
        L = L_recon + λ_vq * L_vq

    Where:
        L_recon = CrossEntropy(logits, target_ids)
        L_vq = codebook_loss + β * commitment_loss

    Args:
        vq_weight: Weight for VQ loss (λ_vq), default 1.0
        beta: Commitment cost for VQ loss, default 0.25
        same_tokenizer: Whether encoder/decoder share tokenizer
        ignore_index: Pad token ID to ignore in loss
        label_smoothing: Label smoothing for reconstruction loss

    Input:
        logits: [B, L, V] decoder output
        target_ids: [B, L] target token IDs
        vq_loss: Scalar, VQ loss from quantizer (optional)
        z: [B, L, D] encoder output (if computing VQ loss internally)
        q: [B, L, D] quantized output (if computing VQ loss internally)

    Output:
        total_loss: Combined loss
        loss_dict: Breakdown of all loss components

    Example:
        >>> loss_fn = VQAELoss(vq_weight=1.0, beta=0.25)
        >>> # Option 1: Pass pre-computed vq_loss from quantizer
        >>> total, details = loss_fn(logits, target_ids, vq_loss=vq_loss)
        >>>
        >>> # Option 2: Compute VQ loss internally
        >>> total, details = loss_fn(logits, target_ids, z=z, q=q)
    """

    def __init__(
        self,
        vq_weight: float = 1.0,
        beta: float = 0.25,
        same_tokenizer: bool = True,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.vq_weight = vq_weight
        self.beta = beta

        self.recon_loss = ReconstructionLoss(
            same_tokenizer=same_tokenizer,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
        self.vq_loss_fn = VQLoss(beta=beta)

    def forward(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        vq_loss: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        q: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute combined VQ-AE loss.

        Args:
            logits: Decoder output [B, L, V]
            target_ids: Target token IDs [B, L]
            vq_loss: Pre-computed VQ loss (from quantizer)
            z: Encoder output [B, L, D] (for internal VQ loss computation)
            q: Quantized output [B, L, D] (for internal VQ loss computation)
            attention_mask: [B, L] mask for reconstruction loss

        Returns:
            total_loss: L_recon + λ * L_vq
            loss_dict: All loss components

        Validation:
            - Either vq_loss OR (z and q) must be provided
            - If neither, VQ loss is set to 0
        """
        # Compute reconstruction loss
        recon_loss = self.recon_loss(logits, target_ids, attention_mask)

        # Compute or use provided VQ loss
        if vq_loss is not None:
            vq_loss_value = vq_loss
            vq_details = {"source": "provided", "value": vq_loss.item()}
        elif z is not None and q is not None:
            vq_loss_value, vq_details = self.vq_loss_fn(z, q)
            vq_details["source"] = "computed"
        else:
            vq_loss_value = torch.tensor(0.0, device=logits.device)
            vq_details = {"source": "none", "value": 0.0}

        # Combined loss
        total_loss = recon_loss + self.vq_weight * vq_loss_value

        loss_dict = {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "vq_loss": (
                vq_loss_value.item()
                if isinstance(vq_loss_value, torch.Tensor)
                else vq_loss_value
            ),
            "vq_weight": self.vq_weight,
            "beta": self.beta,
            "vq_details": vq_details,
        }

        return total_loss, loss_dict


class DualTokenizerVQAELoss(nn.Module):
    """VQ-AE loss for encoder-decoder with DIFFERENT tokenizers.

    This is the correct loss to use when encoder and decoder have
    different tokenizers (e.g., BERT encoder + GPT2 decoder).

    CRITICAL: Target IDs are computed using DECODER's tokenizer!

    Flow:
        texts → Encoder (BERT tokenizer) → z [B, L_enc, D]
                                            ↓
        z → Quantizer → f_hat [B, L, D], vq_loss
                                            ↓
        f_hat → Decoder → logits [B, L_dec, V_dec]
                                            ↓
        texts → Decoder tokenizer → target_ids [B, L_dec]
                                            ↓
        L = CE(logits, target_ids) + λ * vq_loss

    Args:
        dec_tokenizer: DECODER's tokenizer (REQUIRED)
        dec_vocab_size: Decoder vocabulary size
        vq_weight: Weight for VQ loss
        beta: Commitment cost
        ignore_index: Pad token ID
        max_length: Max sequence length for decoder tokenization

    Example:
        >>> from transformers import AutoTokenizer
        >>> dec_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> loss_fn = DualTokenizerVQAELoss(
        ...     dec_tokenizer=dec_tokenizer,
        ...     dec_vocab_size=50257,
        ... )
        >>> total, details = loss_fn(logits, texts, vq_loss=vq_loss)
    """

    def __init__(
        self,
        dec_tokenizer,
        dec_vocab_size: int,
        vq_weight: float = 1.0,
        beta: float = 0.25,
        ignore_index: int = -100,
        max_length: int = 512,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.vq_weight = vq_weight
        self.beta = beta

        self.recon_loss = DualTokenizerReconstructionLoss(
            dec_tokenizer=dec_tokenizer,
            dec_vocab_size=dec_vocab_size,
            ignore_index=ignore_index,
            max_length=max_length,
            label_smoothing=label_smoothing,
        )
        self.vq_loss_fn = VQLoss(beta=beta)

    def forward(
        self,
        logits: torch.Tensor,
        texts: list,
        vq_loss: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        q: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute combined loss with decoder-tokenized targets.

        Args:
            logits: Decoder output [B, L, V_dec]
            texts: List of B input texts (will be re-tokenized by decoder tokenizer)
            vq_loss: Pre-computed VQ loss
            z: Encoder output (for internal VQ computation)
            q: Quantized output (for internal VQ computation)

        Returns:
            total_loss: L_recon + λ * L_vq
            loss_dict: All components including dec_target_ids
        """
        # Compute reconstruction loss with decoder tokenizer
        recon_loss, dec_target_ids = self.recon_loss(logits, texts)

        # Compute or use provided VQ loss
        if vq_loss is not None:
            vq_loss_value = vq_loss
        elif z is not None and q is not None:
            vq_loss_value, _ = self.vq_loss_fn(z, q)
        else:
            vq_loss_value = torch.tensor(0.0, device=logits.device)

        # Combined loss
        total_loss = recon_loss + self.vq_weight * vq_loss_value

        loss_dict = {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "vq_loss": (
                vq_loss_value.item()
                if isinstance(vq_loss_value, torch.Tensor)
                else vq_loss_value
            ),
            "vq_weight": self.vq_weight,
            "dec_target_ids_shape": list(dec_target_ids.shape),
        }

        return total_loss, loss_dict


def compute_vqae_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    vq_loss: torch.Tensor,
    vq_weight: float = 1.0,
    ignore_index: int = -100,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Functional API for VQ-AE loss.

    Simple function to compute combined VQ-AE loss without class instantiation.

    Formula:
        L_total = L_recon + λ * L_vq

    Args:
        logits: [B, L, V] decoder output
        target_ids: [B, L] target token IDs
        vq_loss: Scalar VQ loss from quantizer
        vq_weight: Weight for VQ loss (λ)
        ignore_index: Pad token ID to ignore
        attention_mask: [B, L] optional mask

    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction loss component

    Dimension semantics:
        B = batch size
        L = sequence length
        V = vocab size (e.g., 50257 for GPT2)
        → logits [B, L, V=50257]
        → argmax(dim=-1) → pred_ids [B, L]
        → tokenizer.decode() → reconstructed text
    """
    recon_loss = compute_reconstruction_loss(
        logits=logits,
        target_ids=target_ids,
        attention_mask=attention_mask,
        ignore_index=ignore_index,
    )

    total_loss = recon_loss + vq_weight * vq_loss

    return total_loss, recon_loss
