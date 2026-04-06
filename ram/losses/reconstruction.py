"""Reconstruction Losses for Text Autoencoder.

This module implements reconstruction losses for different encoder-decoder
tokenizer configurations. The choice of loss depends on whether encoder
and decoder share the same tokenizer.

Tokenizer Compatibility Scenarios:
==================================

Scenario 1: Same Tokenizer (Recommended)
----------------------------------------
    Encoder tokenizer == Decoder tokenizer
    Example: T5-encoder + T5-decoder, BART-encoder + BART-decoder

    Flow:
        texts → Tokenizer → input_ids [B, L]
                               ↓
        input_ids → Encoder → hidden [B, L, D]
                               ↓
        hidden → Decoder → logits [B, L, V]
                               ↓
        CrossEntropyLoss(logits, input_ids)

    This is the standard case where token-level reconstruction is valid.

Scenario 2: Different Tokenizers (Requires Special Handling)
------------------------------------------------------------
    Encoder tokenizer != Decoder tokenizer
    Example: BERT-encoder (WordPiece) + GPT2-decoder (BPE)

    Problem:
        - BERT vocab_size = 30522, GPT2 vocab_size = 50257
        - Same text produces different token IDs
        - input_ids from BERT cannot be used as target for GPT2 logits

    Solution:
        texts → BERT Tokenizer → enc_input_ids [B, L_enc]
                                      ↓
        enc_input_ids → Encoder → hidden [B, L_enc, D]
                                      ↓
        hidden → Decoder → logits [B, L_dec, V_dec=50257]
                                      ↓
        texts → GPT2 Tokenizer → dec_target_ids [B, L_dec]
                                      ↓
        CrossEntropyLoss(logits, dec_target_ids)

    Key: Target IDs must come from DECODER's tokenizer!

Scenario 3: Latent Space / VAE (No Token-Level Reconstruction)
--------------------------------------------------------------
    When using VAE-style training, we don't require token-level match.

    Flow:
        texts → Encoder → z [B, D] (latent vector)
                            ↓
        z → Decoder (autoregressive) → generated_text
                            ↓
        Loss = -log P(texts | z) + β * KL(q(z|x) || p(z))

    Here, decoder generates freely conditioned on latent, not reconstructing
    specific token IDs.

Usage Guidelines:
=================
1. ALWAYS validate tokenizer compatibility before training
2. Use `validate_tokenizer_compatibility()` to check configuration
3. Choose appropriate loss function based on scenario
4. When in doubt, use same tokenizer for encoder and decoder (T5/BART)
"""

from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    """Reconstruction loss for text autoencoder.

    Handles different tokenizer scenarios with built-in validation.

    Scenario 1 (same_tokenizer=True):
        Standard cross-entropy between decoder logits and encoder input_ids.
        Flow: logits [B, L, V] vs input_ids [B, L] → loss

    Scenario 2 (same_tokenizer=False):
        Cross-entropy between decoder logits and decoder-tokenized target_ids.
        Flow: logits [B, L, V_dec] vs dec_target_ids [B, L] → loss
        REQUIRES: dec_target_ids must be provided separately!

    Args:
        same_tokenizer: Whether encoder and decoder share the same tokenizer.
            - True: Can use encoder input_ids as target (standard case)
            - False: Must provide decoder-tokenized target_ids separately
        ignore_index: Token ID to ignore in loss computation (usually pad_token_id).
        label_smoothing: Label smoothing factor (0.0 = no smoothing).
        reduction: Loss reduction method ('mean', 'sum', 'none').

    Input shapes:
        logits: [B, L, V] where V = decoder vocab_size
        target_ids: [B, L] token IDs from appropriate tokenizer
        attention_mask: [B, L] optional mask (1 = valid, 0 = ignore)

    Output:
        loss: Scalar tensor (if reduction='mean' or 'sum')
              or [B, L] tensor (if reduction='none')

    Example (same tokenizer - T5/BART):
        >>> loss_fn = ReconstructionLoss(same_tokenizer=True, ignore_index=0)
        >>> logits = decoder(encoder_output)  # [B, L, V]
        >>> loss = loss_fn(logits, input_ids)  # input_ids from shared tokenizer

    Example (different tokenizers - BERT + GPT2):
        >>> loss_fn = ReconstructionLoss(same_tokenizer=False, ignore_index=50256)
        >>> logits = decoder(encoder_output)  # [B, L, 50257]
        >>> # MUST use GPT2 tokenizer for target!
        >>> dec_target_ids = gpt2_tokenizer(texts)["input_ids"]
        >>> loss = loss_fn(logits, dec_target_ids)
    """

    def __init__(
        self,
        same_tokenizer: bool = True,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        latent_token_len: int = 0,
    ):
        super().__init__()
        self.same_tokenizer = same_tokenizer
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        # Official C3 naming
        self.latent_token_len = latent_token_len

        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )

    def forward(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        enc_vocab_size: Optional[int] = None,
        dec_vocab_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute reconstruction loss with validation.

        Args:
            logits: Decoder output [B, L, V] or [B, N+L, V] if latent_token_len > 0
            target_ids: Target token IDs [B, L]
            attention_mask: Optional mask [B, L], 1=valid, 0=pad
            enc_vocab_size: Encoder vocab size (for validation when same_tokenizer=False)
            dec_vocab_size: Decoder vocab size (for validation)

        Returns:
            loss: Scalar or tensor depending on reduction

        Raises:
            ValueError: If tokenizer mismatch detected or invalid configuration
        """
        B, L_total, V = logits.shape
        N = self.latent_token_len

        # Handle latent tokens: skip first N positions in logits
        # logits: [B, N+L, V] -> text_logits: [B, L, V]
        if N > 0:
            # Skip first N latent tokens
            logits = logits[:, N:, :]
            L_total = logits.shape[1]

        # Validation: Check vocab size consistency
        if dec_vocab_size is not None and V != dec_vocab_size:
            raise ValueError(
                f"Logits vocab_size ({V}) != expected dec_vocab_size ({dec_vocab_size}). "
                f"Decoder output dimension mismatch!"
            )

        # Validation: Check target_ids range
        max_target_id = target_ids.max().item()
        if max_target_id >= V:
            raise ValueError(
                f"target_ids contains ID {max_target_id} >= vocab_size {V}. "
                f"This indicates tokenizer mismatch! "
                f"When same_tokenizer=False, target_ids MUST come from decoder's tokenizer."
            )

        # Validation: Check sequence length match
        if target_ids.shape[1] != L_total:
            raise ValueError(
                f"target_ids length ({target_ids.shape[1]}) != logits length ({L_total}). "
                f"Check latent_token_len setting!"
            )

        # Validation: Warn if using different tokenizers
        if not self.same_tokenizer:
            if enc_vocab_size is not None and enc_vocab_size != V:
                # This is expected - different vocab sizes are expected in this mode
                pass

        # Apply attention mask if provided
        if attention_mask is not None:
            # Set ignored positions to ignore_index
            target_ids = target_ids.clone()
            target_ids[attention_mask == 0] = self.ignore_index

        # Reshape for cross entropy: [B*L, V] vs [B*L]
        # Use reshape instead of view for compatibility with non-contiguous tensors
        logits_flat = logits.reshape(-1, V)
        target_flat = target_ids.reshape(-1)

        loss = self.ce_loss(logits_flat, target_flat)

        return loss


class DualTokenizerReconstructionLoss(nn.Module):
    """Reconstruction loss for encoder-decoder with DIFFERENT tokenizers.

    This class explicitly handles the case where encoder and decoder use
    different tokenizers (e.g., BERT encoder + GPT2 decoder).

    CRITICAL: This loss REQUIRES the decoder's tokenizer to be provided
    so that target_ids can be generated correctly.

    Architecture Diagram:
        texts ──┬── Encoder Tokenizer ──→ enc_input_ids [B, L_enc]
                │                              ↓
                │                         Encoder
                │                              ↓
                │                         hidden [B, L_enc, D]
                │                              ↓
                │                         Decoder
                │                              ↓
                │                         logits [B, L_dec, V_dec]
                │                              ↓
                └── Decoder Tokenizer ──→ dec_target_ids [B, L_dec]
                                              ↓
                                    CrossEntropyLoss(logits, dec_target_ids)

    Why This Matters:
        - BERT tokenizer: "Hello" → [101, 7592, 102] (vocab_size=30522)
        - GPT2 tokenizer: "Hello" → [15496] (vocab_size=50257)
        - Using BERT's IDs as target for GPT2 logits is WRONG!
        - Loss would be computed against non-existent token positions

    Args:
        dec_tokenizer: Decoder's tokenizer (REQUIRED)
        dec_vocab_size: Decoder's vocabulary size (for validation)
        ignore_index: Pad token ID to ignore
        max_length: Maximum sequence length for decoder tokenization
        label_smoothing: Label smoothing factor
        latent_token_len: Number of latent tokens to skip in logits (default: 0)

    Input:
        logits: [B, L, V_dec] decoder output or [B, N+L, V_dec] if latent_token_len > 0
        texts: List[str] original input texts (will be re-tokenized)

    Output:
        loss: Scalar reconstruction loss
        dec_target_ids: [B, L] decoder target IDs (for logging/debugging)
    """

    def __init__(
        self,
        dec_tokenizer,
        dec_vocab_size: int,
        ignore_index: int = -100,
        max_length: int = 512,
        label_smoothing: float = 0.0,
        latent_token_len: int = 0,
    ):
        super().__init__()
        self.dec_tokenizer = dec_tokenizer
        self.dec_vocab_size = dec_vocab_size
        self.ignore_index = ignore_index
        self.max_length = max_length
        # Official C3 naming
        self.latent_token_len = latent_token_len

        # Set pad token if not set
        if self.dec_tokenizer.pad_token is None:
            self.dec_tokenizer.pad_token = self.dec_tokenizer.eos_token

        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction="mean",
        )

    def forward(
        self,
        logits: torch.Tensor,
        texts: list,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute loss using decoder-tokenized targets.

        Args:
            logits: Decoder output [B, L, V_dec] or [B, N+L, V_dec] if latent_token_len > 0
            texts: List of B input texts

        Returns:
            loss: Scalar reconstruction loss
            dec_target_ids: [B, L] target IDs from decoder tokenizer

        Flow:
            texts → dec_tokenizer → dec_target_ids [B, L]
            logits [B, L, V] + dec_target_ids [B, L] → CrossEntropyLoss → loss
        """
        B, L_total, V = logits.shape
        device = logits.device
        N = self.latent_token_len

        # Validation
        if V != self.dec_vocab_size:
            raise ValueError(
                f"Logits vocab_size ({V}) != dec_vocab_size ({self.dec_vocab_size}). "
                f"Ensure decoder output matches expected vocabulary!"
            )

        # Handle latent tokens: skip first N positions in logits
        # logits: [B, N+L, V] -> text_logits: [B, L, V]
        if N > 0:
            # Skip first N latent tokens
            logits = logits[:, N:, :]
            L_total = logits.shape[1]

        # Tokenize with DECODER's tokenizer
        # Match logits sequence length after skipping latent tokens
        dec_tokens = self.dec_tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=L_total,
            return_tensors="pt",
        )
        dec_target_ids = dec_tokens["input_ids"].to(device)
        dec_attention_mask = dec_tokens["attention_mask"].to(device)

        # Apply mask: set padded positions to ignore_index
        dec_target_ids = dec_target_ids.clone()
        dec_target_ids[dec_attention_mask == 0] = self.ignore_index

        # Compute loss
        # Use reshape instead of view to handle non-contiguous tensors
        logits_flat = logits.reshape(-1, V)
        target_flat = dec_target_ids.reshape(-1)
        loss = self.ce_loss(logits_flat, target_flat)

        return loss, dec_target_ids


def validate_tokenizer_compatibility(
    enc_tokenizer,
    dec_tokenizer,
    sample_texts: list = None,
) -> Dict[str, Any]:
    """Validate encoder-decoder tokenizer compatibility.

    This function checks whether encoder and decoder tokenizers are compatible
    and provides recommendations for loss function selection.

    Args:
        enc_tokenizer: Encoder's tokenizer
        dec_tokenizer: Decoder's tokenizer
        sample_texts: Optional sample texts for detailed comparison

    Returns:
        dict with keys:
            - same_tokenizer: bool, whether tokenizers are the same
            - enc_vocab_size: int
            - dec_vocab_size: int
            - recommendation: str, which loss to use
            - warnings: List[str], potential issues
            - sample_comparison: dict, token comparison on sample texts (if provided)

    Example:
        >>> from transformers import AutoTokenizer
        >>> enc_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> dec_tok = AutoTokenizer.from_pretrained("gpt2")
        >>> result = validate_tokenizer_compatibility(enc_tok, dec_tok)
        >>> print(result["recommendation"])
        "Use DualTokenizerReconstructionLoss - tokenizers are different!"
    """
    result = {
        "same_tokenizer": False,
        "enc_vocab_size": enc_tokenizer.vocab_size,
        "dec_vocab_size": dec_tokenizer.vocab_size,
        "recommendation": "",
        "warnings": [],
        "sample_comparison": None,
    }

    # Check if same tokenizer
    same_vocab_size = enc_tokenizer.vocab_size == dec_tokenizer.vocab_size

    # Check tokenizer class
    enc_class = type(enc_tokenizer).__name__
    dec_class = type(dec_tokenizer).__name__
    same_class = enc_class == dec_class

    # Check vocabulary overlap (sample)
    if same_vocab_size and same_class:
        # Likely same tokenizer
        result["same_tokenizer"] = True
        result["recommendation"] = (
            "Use ReconstructionLoss(same_tokenizer=True) - "
            "tokenizers appear identical."
        )
    else:
        result["same_tokenizer"] = False
        result["warnings"].append(
            f"Vocab size mismatch: encoder={enc_tokenizer.vocab_size}, "
            f"decoder={dec_tokenizer.vocab_size}"
        )
        result["warnings"].append(
            f"Tokenizer class: encoder={enc_class}, decoder={dec_class}"
        )
        result["recommendation"] = (
            "Use DualTokenizerReconstructionLoss - "
            "tokenizers are different! Target IDs MUST come from decoder tokenizer."
        )

    # Detailed comparison on sample texts
    # Limit to 3 samples
    if sample_texts:
        comparison = []
        for text in sample_texts[:3]:
            enc_ids = enc_tokenizer.encode(text, add_special_tokens=True)
            dec_ids = dec_tokenizer.encode(text, add_special_tokens=True)
            # First 10 tokens for comparison
            comparison.append(
                {
                    "text": text[:50] + "..." if len(text) > 50 else text,
                    "enc_ids": enc_ids[:10],
                    "dec_ids": dec_ids[:10],
                    "enc_len": len(enc_ids),
                    "dec_len": len(dec_ids),
                    "ids_match": enc_ids == dec_ids,
                }
            )
        result["sample_comparison"] = comparison

        # Check if any samples have matching IDs
        if all(c["ids_match"] for c in comparison):
            result["same_tokenizer"] = True
            result["recommendation"] = (
                "Use ReconstructionLoss(same_tokenizer=True) - "
                "token IDs match on sample texts."
            )

    return result


def compute_reconstruction_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Functional API for reconstruction loss.

    Simple cross-entropy loss for text reconstruction.
    Use this when you're sure tokenizers match.

    Args:
        logits: [B, L, V] decoder output logits
        target_ids: [B, L] target token IDs
        attention_mask: [B, L] optional, 1=valid, 0=pad
        ignore_index: Token ID to ignore (pad)
        label_smoothing: Label smoothing factor

    Returns:
        loss: Scalar tensor

    Dimension semantics:
        V = vocab_size (e.g., 50257 for GPT2)
        → argmax(logits, dim=-1) → predicted_ids [B, L]
        → tokenizer.decode(predicted_ids) → reconstructed text
    """
    B, L, V = logits.shape

    # Validate target range
    max_id = target_ids.max().item()
    if max_id >= V:
        raise ValueError(
            f"target_ids max ({max_id}) >= vocab_size ({V}). "
            f"Tokenizer mismatch detected!"
        )

    # Apply attention mask
    if attention_mask is not None:
        target_ids = target_ids.clone()
        target_ids[attention_mask == 0] = ignore_index

    # Compute loss
    # Use reshape instead of view for compatibility with non-contiguous tensors
    loss = F.cross_entropy(
        logits.reshape(-1, V),
        target_ids.reshape(-1),
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )

    return loss
