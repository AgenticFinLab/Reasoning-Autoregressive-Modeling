"""Encoder-Quantizer-Decoder with Hierarchical Text Structure.

================================================================================
Purpose: Test text-native hierarchical quantization (True TAR)
================================================================================

Problem with VAR-style multi-scale:
    - Scales [1, 2, 4, 8, 16, 32] designed for image patches
    - No semantic meaning for text (what is "scale 4" for text?)
    - Text has linguistic hierarchy: sentence → clause → phrase → word

Solution: Hierarchical Text Structure
    - Design scales based on linguistic structure
    - Scale 0: sentence-level (topic)
    - Scale 1: clause-level
    - Scale 2: phrase-level
    - Scale 3: word-level
    - Scale 4: subword-level

================================================================================
Approach: Hierarchical Text Structure (True TAR)
================================================================================

Key Idea:
    - Map text's linguistic hierarchy to quantization scales
    - Coarse scales capture high-level semantics
    - Fine scales capture low-level details

Flow:
    ┌─────────────────────────────────────────────────────────────┐
    │  Text → Tokenize → input_ids [B, L]                         │
    │         │                                                    │
    │         ▼                                                    │
    │  Encoder → z [B, L, D]                                       │
    │         │                                                    │
    │         │  Hierarchical Multi-Scale Quantization:           │
    │         │                                                    │
    │         │  Scale 0: z → pool(1) → [B, 1, D] → sentence emb  │
    │         │  Scale 1: z → pool(4) → [B, 4, D] → clause embs  │
    │         │  Scale 2: z → pool(16) → [B, 16, D] → phrase embs│
    │         │  Scale 3: z → pool(64) → [B, 64, D] → word embs  │
    │         │  Scale 4: z → pool(256) → [B, 256, D] → subword  │
    │         │                                                    │
    │         │  Each scale: VQ → upsample → accumulate to f_hat  │
    │         │                                                    │
    │         ▼                                                    │
    │  f_hat [B, L, D] + indices_per_scale + vq_loss              │
    │         │                                                    │
    │         ▼                                                    │
    │  Decoder → logits [B, L, V]                                  │
    └─────────────────────────────────────────────────────────────┘

================================================================================
Dimensions (for L=256):
================================================================================
    B = batch_size (e.g., 4)
    L = max_length (e.g., 256)
    D = latent_dim (e.g., 256)
    V = vocab_size (GPT2=50257)
    K = codebook_size (e.g., 4096)

    Scale 0: [B, 1, D]   → 1 sentence embedding   (topic)
    Scale 1: [B, 4, D]   → 4 clause embeddings    (structure)
    Scale 2: [B, 16, D]  → 16 phrase embeddings   (phrasing)
    Scale 3: [B, 64, D]  → 64 word embeddings     (words)
    Scale 4: [B, 256, D] → 256 subword embeddings (tokens)

    Total codes: 1 + 4 + 16 + 64 + 256 = 341

================================================================================
Why This Might Work:
================================================================================
    1. Linguistically motivated hierarchy
    2. Coarse scales capture semantics, fine scales capture syntax
    3. Enables "coarse-to-fine" text generation
    4. More interpretable than arbitrary scales

================================================================================
Why This Might NOT Work:
================================================================================
    1. Requires careful scale design
    2. May not align with actual text structure
    3. Pooling may lose position information

================================================================================
Comparison:
================================================================================
    VAR Image Scales:
        [1, 2, 4, 8, 16, 32] = 63 codes
        Spatial hierarchy: pixel → patch → region → image

    TAR Text Scales (this approach):
        [1, 4, 16, 64, 256] = 341 codes
        Linguistic hierarchy: subword → word → phrase → clause → sentence

Usage:
    python examples/PreExp/eqd_hierarchical.py -c configs/PreExp/eqd_hierarchical.yml
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from lmbase.dataset import registry
from transformers import AutoTokenizer

from ram.models.encoder import build_encoder
from ram.models.decoder import build_decoder
from ram.utils import (
    load_config,
    setup_environment,
    collate_fn_text,
    decode_logits_to_text,
    find_latest_checkpoint,
    resume_from_checkpoint,
)


class HierarchicalPhi(nn.Module):
    """Phi transformation for hierarchical quantization.

    Applies learned transformation after codebook lookup.
    Similar to VAR's phi layers but for text hierarchy.

    Dimension Flow:
        Input:  x [B, scale_len, D] codebook vectors
        Output: h [B, scale_len, D] transformed vectors

        h = residual_ratio * conv(x) + (1 - residual_ratio) * x
          = residual_ratio * Linear(x) + (1 - residual_ratio) * x
    """

    def __init__(self, dim: int, residual_ratio: float = 0.5):
        """Initialize phi transformation.

        Args:
            dim: Feature dimension D (e.g., 256)
            residual_ratio: Weight for learned path vs residual (0.5 = equal mix)
        """
        super().__init__()
        # Linear transformation: [B, scale_len, D] → [B, scale_len, D]
        self.conv = nn.Linear(dim, dim)
        self.residual_ratio = residual_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply phi transformation.

        Args:
            x: [B, scale_len, D] codebook output

        Returns:
            h: [B, scale_len, D] transformed output

        Dimension Flow:
            x [B, scale_len, D]
                ↓
            conv(x) = Linear(x) → [B, scale_len, D]
                ↓
            h = residual_ratio * conv(x) + (1 - residual_ratio) * x
              → [B, scale_len, D]
        """
        # Apply linear transformation: [B, scale_len, D] → [B, scale_len, D]
        transformed = self.conv(x)

        # Residual connection with weighted combination
        # h [B, scale_len, D] = ratio * transformed + (1-ratio) * x
        return self.residual_ratio * transformed + (1 - self.residual_ratio) * x


class HierarchicalTextQuantizer(nn.Module):
    """Hierarchical Text Quantizer.

    Implements multi-scale quantization with text-native hierarchy:
    - Scale 0: sentence-level (global topic)
    - Scale 1: clause-level (sentence structure)
    - Scale 2: phrase-level (phrasing)
    - Scale 3: word-level (words)
    - Scale 4: subword-level (tokens)

    Input:
        z: [B, L, D] encoder output

    Output:
        f_hat: [B, L, D] quantized output (accumulated)
        indices_per_scale: List[[B, scale_k] for k in scales]
        vq_loss: scalar VQ loss
    """

    def __init__(
        self,
        codebook_size: int = 4096,
        codebook_dim: int = 256,
        scale_lengths: tuple = (1, 4, 16, 64, 256),  # Hierarchical text scales
        beta: float = 0.25,
        quant_resi: float = 0.5,
    ):
        """Initialize hierarchical text quantizer.

        Args:
            codebook_size: Number of codebook vectors K (e.g., 4096)
            codebook_dim: Dimension of each codebook vector D (e.g., 256)
            scale_lengths: Tuple of scale lengths (e.g., (1, 4, 16, 64, 256))
                          Total codes = sum(scale_lengths) = 341
            beta: Commitment loss weight for VQ (e.g., 0.25)
            quant_resi: Residual ratio for phi transformation (e.g., 0.5)

        Dimension Info:
            codebook_size K: 4096 (number of discrete codes)
            codebook_dim D: 256 (dimension of each code)
            scale_lengths: [1, 4, 16, 64, 256] (hierarchical scales)
            total_codes: 1 + 4 + 16 + 64 + 256 = 341
        """
        super().__init__()
        self.codebook_size = codebook_size  # K
        self.codebook_dim = codebook_dim  # D
        self.scale_lengths = scale_lengths
        self.beta = beta

        # Shared codebook for all scales
        # embedding: [K, D] codebook matrix
        # Input: indices [B, scale_len] → Output: vectors [B, scale_len, D]
        self.embedding = nn.Embedding(codebook_size, codebook_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

        # Phi transformations for each scale (partially shared)
        # num_phi: number of unique phi layers (share across scales)
        num_phi = min(4, len(scale_lengths))  # Share some phi layers
        self.phi_layers = nn.ModuleList(
            [HierarchicalPhi(codebook_dim, quant_resi) for _ in range(num_phi)]
        )

        # Mapping from scale index to phi index
        # scale_to_phi[i] = which phi layer to use for scale i
        # Example with 5 scales and 4 phi layers: [0, 1, 2, 3, 3]
        self.scale_to_phi = [min(i, num_phi - 1) for i in range(len(scale_lengths))]

    def forward(self, z: torch.Tensor):
        """Hierarchical multi-scale quantization.

        Args:
            z: [B, L, D] encoder output

        Returns:
            f_hat: [B, L, D] accumulated quantized features
            indices_per_scale: List of [B, scale_k] indices for each scale
            vq_loss: scalar VQ loss

        Dimension Flow (for L=256, scale_lengths=[1,4,16,64,256]):
            Input: z [B, L, D] (e.g., [4, 256, 256])
                ↓
            For each scale (scale_len in [1, 4, 16, 64, 256]):

                Scale 0 (scale_len=1, sentence-level):
                    f_rest [B, L, D] → pool → rest_down [B, 1, D]
                    rest_down [B, 1, D] → VQ → q [B, 1, D], indices [B, 1]
                    q [B, 1, D] → phi → h [B, 1, D] → upsample → h_up [B, L, D]
                    f_hat += h_up, f_rest -= h_up

                Scale 1 (scale_len=4, clause-level):
                    f_rest [B, L, D] → pool → rest_down [B, 4, D]
                    rest_down [B, 4, D] → VQ → q [B, 4, D], indices [B, 4]
                    q [B, 4, D] → phi → h [B, 4, D] → upsample → h_up [B, L, D]
                    f_hat += h_up, f_rest -= h_up

                Scale 2 (scale_len=16, phrase-level):
                    f_rest [B, L, D] → pool → rest_down [B, 16, D]
                    rest_down [B, 16, D] → VQ → q [B, 16, D], indices [B, 16]
                    q [B, 16, D] → phi → h [B, 16, D] → upsample → h_up [B, L, D]
                    f_hat += h_up, f_rest -= h_up

                Scale 3 (scale_len=64, word-level):
                    f_rest [B, L, D] → pool → rest_down [B, 64, D]
                    rest_down [B, 64, D] → VQ → q [B, 64, D], indices [B, 64]
                    q [B, 64, D] → phi → h [B, 64, D] → upsample → h_up [B, L, D]
                    f_hat += h_up, f_rest -= h_up

                Scale 4 (scale_len=256, subword-level):
                    f_rest [B, L, D] → no pool → rest_down [B, 256, D]
                    rest_down [B, 256, D] → VQ → q [B, 256, D], indices [B, 256]
                    q [B, 256, D] → phi → h [B, 256, D] → no upsample → h_up [B, L, D]
                    f_hat += h_up, f_rest -= h_up

            Output:
                f_hat [B, L, D]: accumulated quantized features
                indices_per_scale: [[B,1], [B,4], [B,16], [B,64], [B,256]]
                vq_loss: scalar
        """
        B, L, D = z.shape  # Batch, Length, Dimension (e.g., [4, 256, 256])
        device = z.device

        # Initialize
        # f_rest: residual features to be quantized, starts as z [B, L, D]
        f_rest = z.clone()  # [B, L, D]
        # f_hat: accumulator for quantized features [B, L, D]
        f_hat = torch.zeros_like(z)  # [B, L, D]
        indices_per_scale = []  # Store indices for each scale
        total_vq_loss = 0.0  # Accumulate VQ loss across scales

        # Process each scale in hierarchy
        for scale_idx, scale_len in enumerate(self.scale_lengths):
            # Downsample residual to current scale length
            # f_rest [B, L, D] → rest_down [B, scale_len, D]
            if scale_len == L:
                # No downsampling needed for full-resolution scale
                rest_down = f_rest  # [B, L, D]
            else:
                # Adaptive average pooling to downsample
                # f_rest [B, L, D] → transpose → [B, D, L]
                # → adaptive_avg_pool1d → [B, D, scale_len]
                # → transpose → rest_down [B, scale_len, D]
                rest_down = F.adaptive_avg_pool1d(
                    f_rest.transpose(1, 2), scale_len
                ).transpose(1, 2)

            # Find nearest codebook entries (Vector Quantization)
            # Compute distances between rest_down and all codebook vectors
            # rest_down [B, scale_len, D] → reshape → [B*scale_len, D]
            # distances [B*scale_len, K] where K = codebook_size
            distances = torch.cdist(
                rest_down.reshape(-1, D), self.embedding.weight
            )  # [B*scale_len, K]

            # Get indices of nearest codebook entries
            # distances [B*scale_len, K] → argmin → [B*scale_len] → reshape → [B, scale_len]
            indices = distances.argmin(dim=1).reshape(B, scale_len)
            indices_per_scale.append(indices)

            # Lookup codebook vectors using indices
            # indices [B, scale_len] → embedding → q [B, scale_len, D]
            q = self.embedding(indices)  # [B, scale_len, D]

            # Apply phi transformation (learned residual transformation)
            # Select phi layer for this scale (with sharing)
            phi_idx = self.scale_to_phi[scale_idx]
            # q [B, scale_len, D] → phi_layers → h [B, scale_len, D]
            h = self.phi_layers[phi_idx](q)  # [B, scale_len, D]

            # Upsample to full sequence length
            if scale_len == L:
                # No upsampling needed for full-resolution scale
                h_up = h  # [B, L, D]
            else:
                # Linear interpolation upsampling
                # h [B, scale_len, D] → transpose → [B, D, scale_len]
                # → interpolate → [B, D, L]
                # → transpose → h_up [B, L, D]
                h_up = F.interpolate(
                    h.transpose(1, 2), size=L, mode="linear", align_corners=False
                ).transpose(1, 2)

            # Accumulate quantized features
            # f_hat [B, L, D] += h_up [B, L, D]
            f_hat = f_hat + h_up

            # Update residual (subtract quantized contribution)
            # f_rest [B, L, D] -= h_up [B, L, D]
            f_rest = f_rest - h_up

            # Compute VQ loss for this scale
            # Commitment loss: encoder output should be close to codebook vector
            # Codebook loss: codebook vector should be close to encoder output
            commitment_loss = F.mse_loss(rest_down, q.detach())  # scalar
            codebook_loss = F.mse_loss(rest_down.detach(), q)  # scalar
            scale_vq_loss = codebook_loss + self.beta * commitment_loss  # scalar
            total_vq_loss = total_vq_loss + scale_vq_loss  # Accumulate

        # Straight-through estimator for gradient flow
        # Allows gradients to pass through quantization
        # f_hat [B, L, D] = z + (f_hat - z).detach()
        # In forward: returns f_hat (quantized)
        # In backward: gradient flows to z (original)
        f_hat = z + (f_hat - z).detach()

        return f_hat, indices_per_scale, total_vq_loss

    def decode_indices(
        self,
        indices_per_scale: list,
        target_length: int,
    ) -> torch.Tensor:
        """Decode indices to f_hat.

        Args:
            indices_per_scale: List of [B, scale_k] indices for each scale
            target_length: Target sequence length L (e.g., 256)

        Returns:
            f_hat: [B, L, D] reconstructed features

        Dimension Flow:
            Input: indices_per_scale = [
                indices_0 [B, 1],      # Scale 0: sentence-level
                indices_1 [B, 4],      # Scale 1: clause-level
                indices_2 [B, 16],     # Scale 2: phrase-level
                indices_3 [B, 64],     # Scale 3: word-level
                indices_4 [B, 256],    # Scale 4: subword-level
            ]

            For each scale:
                indices [B, scale_len] → embedding → q [B, scale_len, D]
                q [B, scale_len, D] → phi_layers → h [B, scale_len, D]
                h [B, scale_len, D] → upsample → h_up [B, L, D]

            Output: f_hat [B, L, D] = sum of all h_up across scales
        """
        B = indices_per_scale[0].shape[0]  # Batch size
        D = self.codebook_dim  # Codebook dimension (e.g., 256)
        device = indices_per_scale[0].device

        # Initialize accumulator: [B, L, D]
        f_hat = torch.zeros(B, target_length, D, device=device)

        for scale_idx, indices in enumerate(indices_per_scale):
            scale_len = indices.shape[
                1
            ]  # Current scale length (e.g., 1, 4, 16, 64, 256)

            # Lookup codebook vectors
            # indices [B, scale_len] → embedding → q [B, scale_len, D]
            q = self.embedding(indices)

            # Apply phi transformation
            # q [B, scale_len, D] → phi_layers → h [B, scale_len, D]
            phi_idx = self.scale_to_phi[scale_idx]
            h = self.phi_layers[phi_idx](q)

            # Upsample to target length
            if scale_len == target_length:
                # No upsampling needed: h [B, L, D]
                h_up = h
            else:
                # Linear interpolation upsampling
                # h [B, scale_len, D] → transpose → [B, D, scale_len]
                # → interpolate → [B, D, L] → transpose → h_up [B, L, D]
                h_up = F.interpolate(
                    h.transpose(1, 2),
                    size=target_length,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)

            # Accumulate: f_hat [B, L, D] += h_up [B, L, D]
            f_hat = f_hat + h_up

        # Return accumulated features: [B, L, D]
        return f_hat


class DualTokenizerVQAELoss(nn.Module):
    """VQAE Loss for dual tokenizer setup.

    Dimension Flow:
        Input:
            logits: [B, L, V] decoder output logits
            texts: List[str] original text strings (for target tokenization)
            vq_loss: scalar VQ quantization loss

        Output:
            total_loss: scalar = recon_loss + vq_weight * vq_loss
            loss_dict: dict with recon_loss, vq_loss, total_loss
    """

    def __init__(
        self,
        dec_tokenizer,
        dec_vocab_size: int,
        ignore_index: int = -100,
        vq_weight: float = 1.0,
    ):
        super().__init__()
        self.dec_tokenizer = dec_tokenizer
        self.dec_vocab_size = dec_vocab_size
        self.ignore_index = ignore_index
        self.vq_weight = vq_weight

    def forward(self, logits, texts, vq_loss):
        """Compute VQAE loss.

        Args:
            logits: [B, L, V] decoder output logits
            texts: List[str] original text strings
            vq_loss: scalar VQ loss

        Returns:
            total_loss: scalar combined loss
            loss_dict: dict with component losses

        Dimension Flow:
            logits [B, L, V]
                ↓
            Tokenize texts with decoder tokenizer
                ↓
            target_ids [B, L], attention_mask [B, L]
                ↓
            Shift for autoregressive prediction:
                pred_logits = logits[:, :-1, :]  → [B, L-1, V]
                targets = target_ids[:, 1:]      → [B, L-1]
                mask = attention_mask[:, 1:]     → [B, L-1]
                ↓
            Apply mask: targets [B, L-1] with ignore_index where mask==0
                ↓
            recon_loss = cross_entropy(
                pred_logits.reshape(-1, V)  → [(B*(L-1)), V]
                targets.reshape(-1)         → [(B*(L-1))]
            ) → scalar
                ↓
            total_loss = recon_loss + vq_weight * vq_loss → scalar
        """
        B, L, V = logits.shape  # Batch, Length, Vocab_size

        # Tokenize target texts with decoder tokenizer
        # texts List[str] → tokens dict
        tokens = self.dec_tokenizer(
            texts,
            max_length=L,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # target_ids [B, L], attention_mask [B, L]
        target_ids = tokens["input_ids"].to(logits.device)
        attention_mask = tokens["attention_mask"].to(logits.device)

        # Shift for autoregressive prediction
        # pred_logits: predict next token, so use logits up to second-to-last position
        # [B, L, V] → [B, L-1, V]
        pred_logits = logits[:, :-1, :]

        # targets: target tokens start from position 1 (second token)
        # [B, L] → [B, L-1]
        targets = target_ids[:, 1:]

        # mask: corresponding attention mask
        # [B, L] → [B, L-1]
        mask = attention_mask[:, 1:]

        # Apply mask: set targets to ignore_index where mask is 0 (padding)
        # targets [B, L-1] with ignore_index for padding positions
        targets = targets.masked_fill(mask == 0, self.ignore_index)

        # Compute reconstruction loss
        # pred_logits [(B*(L-1)), V] vs targets [(B*(L-1))]
        recon_loss = F.cross_entropy(
            pred_logits.reshape(-1, V),
            targets.reshape(-1),
            ignore_index=self.ignore_index,
        )

        # Total loss: reconstruction + weighted VQ loss
        total_loss = recon_loss + self.vq_weight * vq_loss

        return total_loss, {
            "recon_loss": recon_loss.item(),
            "vq_loss": vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss,
            "total_loss": total_loss.item(),
        }


def train_hierarchical(config: dict):
    """
    Train Encoder-Quantizer-Decoder with Hierarchical Text Structure.

    Training Flow:
        Step 1: texts → tokenize → input_ids [B, L]
        Step 2: input_ids → Encoder → z [B, L, D]
        Step 3: z → HierarchicalTextQuantizer → f_hat + indices + vq_loss
        Step 4: f_hat → Decoder → logits [B, L, V]
        Step 5: loss = recon_loss + λ * vq_loss

    Key Difference from eqd_original.py:
        - Uses text-native hierarchical scales
        - [1, 4, 16, 64, 256] instead of [1, 2, 4, 8, 16, 32]
        - Linguistically motivated hierarchy
    """
    # =================================================================
    # Extract config
    # =================================================================
    enc_cfg = config["model"]["encoder"]
    dec_cfg = config["model"]["decoder"]
    quant_cfg = config["model"]["quantizer"]
    latent_dim = config["model"]["latent_dim"]
    data_cfg = config["data"]
    train_cfg = config["train"]
    env_cfg = config["environment"]
    log_cfg = config["logging"]

    batch_size = train_cfg["batch_size"]
    learning_rate = train_cfg["learning_rate"]
    weight_decay = train_cfg["weight_decay"]
    num_epochs = train_cfg["num_epochs"]
    warmup_steps = train_cfg["warmup_steps"]
    gradient_clip = train_cfg["gradient_clip"]
    vq_loss_weight = train_cfg["vq_loss_weight"]
    resume = train_cfg["resume"]

    log_interval = log_cfg["log_interval"]
    checkpoint_interval = log_cfg["checkpoint_interval"]

    output_dir = Path(log_cfg["output_dir"])
    checkpoint_dir = Path(log_cfg["checkpoint_dir"])
    log_dir = Path(log_cfg["log_dir"])

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    L = enc_cfg["max_length"]
    scale_lengths = quant_cfg["scale_lengths"]
    total_codes = sum(scale_lengths)
    device = setup_environment(env_cfg)

    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Max length: {L}")
    print(f"Scale lengths: {scale_lengths}")
    print(f"Total codes: {total_codes}")
    print(f"Hierarchy: sentence → clause → phrase → word → subword")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"Output dir: {output_dir}")
    print()

    # =================================================================
    # Build models
    # =================================================================
    print("[1] Building Encoder...")
    encoder = build_encoder(enc_cfg)
    encoder = encoder.to(device)
    D = encoder.output_dim
    print(f"    hidden_dim: {encoder.hidden_dim}, output_dim: {D}")

    print("[2] Building Hierarchical Text Quantizer...")
    quantizer = HierarchicalTextQuantizer(
        codebook_size=quant_cfg["codebook_size"],
        codebook_dim=latent_dim,
        scale_lengths=tuple(scale_lengths),
        beta=quant_cfg["beta"],
        quant_resi=quant_cfg["quant_resi"],
    )
    quantizer = quantizer.to(device)
    print(f"    codebook_size: {quantizer.codebook_size}")
    print(f"    scale_lengths: {quantizer.scale_lengths}")
    print(f"    total_codes: {sum(quantizer.scale_lengths)}")

    print("[3] Building Decoder...")
    decoder = build_decoder(dec_cfg, input_dim=latent_dim)
    decoder = decoder.to(device)
    V = decoder.vocab_size
    print(f"    hidden_dim: {decoder.hidden_dim}, vocab_size: {V}")

    # Projection layer if encoder output != latent_dim
    # Encoder outputs [B, L, D], quantizer expects [B, L, latent_dim]
    if D != latent_dim:
        print(f"[4] Adding pre-quantization projection: {D} → {latent_dim}")
        pre_quant_proj = nn.Linear(D, latent_dim).to(device)
    else:
        pre_quant_proj = nn.Identity()
    print()

    # Tokenizer
    dec_tokenizer = AutoTokenizer.from_pretrained(decoder.model_name)
    if dec_tokenizer.pad_token is None:
        dec_tokenizer.pad_token = dec_tokenizer.eos_token

    # =================================================================
    # Setup loss function
    # =================================================================
    print("[5] Setting up loss function...")
    loss_fn = DualTokenizerVQAELoss(
        dec_tokenizer=dec_tokenizer,
        dec_vocab_size=V,
        vq_weight=vq_loss_weight,
    )
    print(f"    Loss: DualTokenizerVQAELoss")
    print(f"    VQ weight: {vq_loss_weight}")
    print()

    # =================================================================
    # Load data
    # =================================================================
    print("[6] Loading dataset...")
    dataset = registry.get(data_cfg, split=data_cfg["split"])
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_text,
        drop_last=True,
    )
    print(f"    Dataset: {data_cfg['data_name']}, {len(dataset)} samples")
    print(f"    Batches per epoch: {len(dataloader)}")
    print()

    # =================================================================
    # Setup optimizer and scheduler
    # =================================================================
    print("[7] Setting up optimizer...")
    params = (
        list(encoder.parameters())
        + list(quantizer.parameters())
        + list(decoder.parameters())
    )
    if isinstance(pre_quant_proj, nn.Linear):
        params += list(pre_quant_proj.parameters())

    optimizer = AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    total_steps = len(dataloader) * num_epochs
    scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    print(f"    Total steps: {total_steps}")
    print(f"    Warmup steps: {warmup_steps}")
    print()

    # =================================================================
    # Resume from checkpoint
    # =================================================================
    start_epoch = 0
    global_step = 0
    history = {
        "config": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "max_length": L,
            "scale_lengths": scale_lengths,
            "total_codes": total_codes,
            "approach": "hierarchical",
        },
        "steps": [],
    }

    if resume:
        latest_ckpt = find_latest_checkpoint(checkpoint_dir)
        if latest_ckpt is not None:
            print(f"[7.5] Resuming from: {latest_ckpt.name}")
            models = {
                "encoder": encoder,
                "quantizer": quantizer,
                "decoder": decoder,
            }
            if isinstance(pre_quant_proj, nn.Linear):
                models["pre_quant_proj"] = pre_quant_proj

            start_epoch, global_step, history = resume_from_checkpoint(
                checkpoint_path=latest_ckpt,
                models=models,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                log_dir=log_dir,
            )
            print(f"    Resumed from epoch {start_epoch+1}, global_step {global_step}")
            print()
        else:
            print("[7.5] No checkpoint found, starting fresh")
            print()

    # =================================================================
    # Training loop
    # =================================================================
    print("[8] Starting training...")
    print("=" * 60)

    encoder.train()
    quantizer.train()
    decoder.train()

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_vq_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_texts in pbar:
            # Tokenize
            tokens = encoder.tokenize(batch_texts)
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]

            # Encode
            # input_ids [B, L] → encoder → hidden [B, L, D]
            hidden = encoder(input_ids=input_ids, attention_mask=attention_mask)
            # hidden [B, L, D] → pre_quant_proj → z [B, L, latent_dim]
            z = pre_quant_proj(hidden)

            # Hierarchical Quantize
            # z [B, L, latent_dim] → quantizer → f_hat [B, L, latent_dim]
            f_hat, indices_per_scale, vq_loss = quantizer(z)

            # Decode (decoder expects latent_dim, f_hat is already latent_dim)
            # f_hat [B, L, latent_dim] → decoder → logits [B, L, V]
            logits = decoder(f_hat, attention_mask=attention_mask)

            # Compute loss
            total_loss, loss_details = loss_fn(logits, batch_texts, vq_loss)
            recon_loss = loss_details["recon_loss"]

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(params, gradient_clip)
            optimizer.step()
            scheduler.step()

            # Logging
            epoch_loss += total_loss.item()
            epoch_recon_loss += recon_loss
            epoch_vq_loss += (
                vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss
            )
            num_batches += 1
            global_step += 1

            history["steps"].append(
                {
                    "epoch": epoch + 1,
                    "step_in_epoch": num_batches,
                    "global_step": global_step,
                    "loss": total_loss.item(),
                    "recon_loss": recon_loss,
                    "vq_loss": (
                        vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss
                    ),
                    "lr": scheduler.get_last_lr()[0],
                }
            )

            pbar.set_postfix(
                {
                    "loss": f"{total_loss.item():.4f}",
                    "recon": f"{recon_loss:.4f}",
                    "vq": f"{vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss:.4f}",
                }
            )

            if global_step % log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"    Step {global_step}: loss={total_loss.item():.4f}, "
                    f"recon={recon_loss:.4f}, vq={vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss:.4f}, lr={lr:.2e}"
                )

                with torch.no_grad():
                    decode_result = decode_logits_to_text(
                        logits, dec_tokenizer, batch_texts, attention_mask
                    )

                samples_path = (
                    log_dir
                    / f"samples-epoch{epoch+1}-step{num_batches}-global{global_step}.json"
                )
                with open(samples_path, "w", encoding="utf-8") as f:
                    json.dump(decode_result, f, indent=2, ensure_ascii=False)

                history_path = log_dir / "training_history.json"
                with open(history_path, "w") as f:
                    json.dump(history, f, indent=2)

            if global_step % checkpoint_interval == 0:
                checkpoint = {
                    "epoch": epoch + 1,
                    "step_in_epoch": num_batches,
                    "global_step": global_step,
                    "encoder_state_dict": encoder.state_dict(),
                    "quantizer_state_dict": quantizer.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": total_loss.item(),
                }
                if isinstance(pre_quant_proj, nn.Linear):
                    checkpoint["pre_quant_proj_state_dict"] = (
                        pre_quant_proj.state_dict()
                    )

                ckpt_name = f"checkpoint-epoch{epoch+1}-step{num_batches}-global{global_step}.pt"
                torch.save(checkpoint, checkpoint_dir / ckpt_name)
                print(f"    [Checkpoint saved: {ckpt_name}]")

        avg_epoch_loss = epoch_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_vq_loss = epoch_vq_loss / num_batches
        print(
            f"Epoch {epoch+1} completed: loss={avg_epoch_loss:.4f}, recon={avg_recon_loss:.4f}, vq={avg_vq_loss:.4f}"
        )
        print()

    print("=" * 60)
    print("Training completed!")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encoder-Quantizer-Decoder with Hierarchical Text Structure"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Config: {args.config}")
    print("=" * 60)
    train_hierarchical(config)
