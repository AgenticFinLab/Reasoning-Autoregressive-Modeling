"""Text VQVAE: Vector Quantized Variational Autoencoder for Text.

Design pattern inspired by VAR-main/models/vqvae.py.
Implementation is original for text next-scaling task.

Text-VAR VQVAE Architecture Flow:
==================================
1. Token Embedding: (B, L) token_ids -> (B, L, embed_dim) embeddings
2. Encoder: (B, embed_dim, L) -> (B, Cvae, L_latent) compressed features
3. quant_conv: (B, Cvae, L_latent) -> (B, Cvae, L_latent) projection
4. VectorQuantizer2: Multi-scale residual quantization
   - Input: (B, Cvae, L_latent) continuous features
   - Process: Residual coding at scales [1, 2, 4, ..., L_latent]
   - Output: (B, Cvae, L_latent) accumulated f_hat = sum of all scale features
5. post_quant_conv: (B, Cvae, L_latent) -> (B, Cvae, L_latent) projection
6. Decoder: (B, Cvae, L_latent) -> (B, L, vocab_size) token logits

Key Text-VAR Insight:
- The decoder receives the SUMMED multi-scale features (f_hat)
- This single pass through decoder reconstructs the full sequence
- Unlike sequential decoding, this is efficient single-shot reconstruction

Tensor dimension conventions:
- B: batch size
- L: original sequence length (e.g., 512 tokens)
- L_latent: compressed sequence length = L // downsample_ratio (e.g., 64)
- Cvae: latent channel dimension (e.g., 32)
- embed_dim: token embedding dimension (e.g., 768)
- vocab_size: vocabulary size (e.g., 32000)
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .basic_vae import Encoder, Decoder
from .quant import VectorQuantizer2

__all__ = ["VQVAE"]


class VQVAE(nn.Module):
    """
    Text VQVAE model for multi-scale latent text representation.

    This is the foundation of Text-VAR, learning discrete multi-scale
    representations that capture both global structure and local details.

    Complete Shape Flow:
    ====================
    Input:    (B, L) token indices, e.g., (32, 512)
              ↓ tok_embed
    Embedded: (B, L, embed_dim) = (32, 512, 768)
              ↓ transpose for Conv1d
              (B, embed_dim, L) = (32, 768, 512)
              ↓ encoder (with downsampling)
    Encoded:  (B, Cvae, L_latent) = (32, 32, 64)  [8x compression]
              ↓ quant_conv
    Pre-VQ:   (B, Cvae, L_latent) = (32, 32, 64)
              ↓ VectorQuantizer2 (multi-scale)
              │  Scale 1: (32, 32, 1)  → upsample → (32, 32, 64) → f_hat += h_1
              │  Scale 2: (32, 32, 2)  → upsample → (32, 32, 64) → f_hat += h_2
              │  Scale 3: (32, 32, 4)  → upsample → (32, 32, 64) → f_hat += h_3
              │  ...
              │  Scale 6: (32, 32, 64) → (no upsample)          → f_hat += h_6
    f_hat:    (B, Cvae, L_latent) = (32, 32, 64)  [accumulated sum]
              ↓ post_quant_conv
    Post-VQ:  (B, Cvae, L_latent) = (32, 32, 64)
              ↓ decoder (with upsampling)
    Decoded:  (B, L, embed_dim) = (32, 512, 768)
              ↓ vocab_proj
    Output:   (B, L, vocab_size) = (32, 512, 32000) logits

    Args:
        vocab_size: Token vocabulary size (default: 32000)
        embed_dim: Token embedding dimension (default: 768)
        z_channels: Latent channel dimension Cvae (default: 32)
        ch: Base channel count for encoder/decoder (default: 128)
        ch_mult: Channel multipliers per resolution (default: (1,2,4,8))
        num_res_blocks: ResNet blocks per resolution (default: 2)
        dropout: Dropout rate (default: 0.0)
        beta: VQ commitment loss weight (default: 0.25)
        using_znorm: Use z-normalization in VQ (default: False)
        quant_conv_ks: Kernel size for quant/post_quant conv (default: 3)
        quant_resi: Residual ratio for phi layers (default: 0.5)
        share_quant_resi: Phi sharing mode (default: 4)
        v_patch_lens: Scale lengths for VQ (default: (1,2,4,8,16,32))
        test_mode: Freeze weights and set eval mode (default: False)
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        embed_dim: int = 768,
        z_channels: int = 32,
        ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        beta: float = 0.25,
        using_znorm: bool = False,
        quant_conv_ks: int = 3,
        quant_resi: float = 0.5,
        share_quant_resi: int = 4,
        default_qresi_counts: int = 0,
        v_patch_lens: Tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        test_mode: bool = False,
    ):
        super().__init__()
        self.test_mode = test_mode
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.V = vocab_size  # Codebook vocab (alias for compatibility)
        self.Cvae = z_channels

        # Token embedding
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)

        # Encoder: (B, embed_dim, L) → (B, z_channels, L // downsample_ratio)
        self.encoder = Encoder(
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            in_channels=embed_dim,
            z_channels=z_channels,
            using_sa=True,
            using_mid_sa=True,
        )
        self.downsample_ratio = self.encoder.downsample_ratio

        # Decoder: (B, z_channels, L // downsample_ratio) → (B, L, vocab_size)
        self.decoder = Decoder(
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            z_channels=z_channels,
            out_channels=embed_dim,
            vocab_size=vocab_size,
            using_sa=True,
            using_mid_sa=True,
        )

        # Vector Quantizer
        self.quantize = VectorQuantizer2(
            vocab_size=vocab_size,  # Codebook size (can be different from token vocab)
            Cvae=z_channels,
            using_znorm=using_znorm,
            beta=beta,
            default_qresi_counts=default_qresi_counts,
            v_patch_lens=v_patch_lens,
            quant_resi=quant_resi,
            share_quant_resi=share_quant_resi,
        )

        # Quant convolutions (mirror vqvae.py lines 48-49)
        self.quant_conv = nn.Conv1d(
            z_channels, z_channels, quant_conv_ks, stride=1, padding=quant_conv_ks // 2
        )
        self.post_quant_conv = nn.Conv1d(
            z_channels, z_channels, quant_conv_ks, stride=1, padding=quant_conv_ks // 2
        )

        if self.test_mode:
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)

    # ===================== forward: Full VQVAE pass for training =====================
    def forward(
        self, inp: torch.Tensor, ret_usages: bool = False
    ) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        """
        Full forward pass for VQVAE training.

        Args:
            inp: (B, L) input token indices
                 B = batch size, L = sequence length
            ret_usages: Whether to return codebook usage stats

        Returns:
            logits: (B, L, vocab_size) reconstructed token logits
            usages: Codebook usage per scale (if ret_usages=True, else None)
            vq_loss: Vector quantization loss (commitment + reconstruction)

        Shape Flow:
            inp: (B, L) = (32, 512)
            → embed: (B, L, embed_dim) = (32, 512, 768)
            → transpose: (B, embed_dim, L) = (32, 768, 512)
            → encode: (B, Cvae, L_latent) = (32, 32, 64)
            → VQ (multi-scale sum): (B, Cvae, L_latent) = (32, 32, 64)
            → decode: (B, L, vocab_size) = (32, 512, 32000)
        """
        # Embed tokens
        x = self.tok_embed(inp)  # (B, L, embed_dim)
        x = x.transpose(1, 2)  # (B, embed_dim, L)

        # Encode → quantize → decode
        f = self.encoder(x)  # (B, Cvae, L // downsample)
        f = self.quant_conv(f)
        f_hat, usages, vq_loss = self.quantize(f, ret_usages=ret_usages)
        logits = self.decoder(self.post_quant_conv(f_hat))  # (B, L, vocab_size)

        return logits, usages, vq_loss

    # ===================== fhat_to_logits: Decode accumulated features =====================
    def fhat_to_logits(self, f_hat: torch.Tensor) -> torch.Tensor:
        """
        Decode accumulated multi-scale features to token logits.

        This is the final reconstruction step where the SUMMED multi-scale
        features (f_hat) are decoded back to token predictions.

        Args:
            f_hat: (B, Cvae, L_latent) accumulated features from VQ
                   This is the sum of all scale contributions:
                   f_hat = h_1 + h_2 + h_3 + ... + h_K

        Returns:
            (B, L, vocab_size) token logits for each position

        Note:
            The decoder is applied ONCE to the summed features,
            not separately for each scale. This is the key efficiency
            gain of the summation-based architecture.
        """
        return self.decoder(self.post_quant_conv(f_hat))

    # ===================== inp_to_idxBl: Encode to multi-scale indices =====================
    def inp_to_idxBl(
        self, inp: torch.Tensor, v_patch_lens: Optional[Sequence[int]] = None
    ) -> List[torch.LongTensor]:
        """
        Encode input tokens to multi-scale codebook indices.

        Used to prepare training data for the TAR next-scale predictor.
        Each scale gets its own set of discrete codes.

        Args:
            inp: (B, L) input token indices
            v_patch_lens: Custom scale lengths (default: use model's scales)

        Returns:
            List of index tensors, one per scale:
            [(B, 1), (B, 2), (B, 4), (B, 8), (B, 16), (B, 32)]

        Example for L=512, scales=(1,2,4,8,16,32), downsample=8:
            L_latent = 64, so actual scales might be (1,2,4,8,16,32,64)
        """
        x = self.tok_embed(inp).transpose(1, 2)  # (B, embed_dim, L)
        f = self.quant_conv(self.encoder(x))
        return self.quantize.f_to_idxBl_or_fhat(
            f, to_fhat=False, v_patch_lens=v_patch_lens
        )

    # ===================== idxBl_to_logits: Decode indices to logits =====================
    def idxBl_to_logits(
        self,
        ms_idx_Bl: List[torch.Tensor],
        same_shape: bool = True,
        last_one: bool = False,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Decode multi-scale indices to token logits.

        Used during inference after TAR generates codebook indices.

        Args:
            ms_idx_Bl: List of index tensors for each scale
                       [(B, 1), (B, 2), (B, 4), ..., (B, L_latent)]
            same_shape: If True, upsample all to same shape before summing
            last_one: If True, return only final logits (after all scales)

        Returns:
            If last_one: (B, L, vocab_size) final token logits
            Else: List of logits showing progressive reconstruction
        """
        B = ms_idx_Bl[0].shape[0]
        ms_h_BCl = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            ms_h_BCl.append(
                self.quantize.embedding(idx_Bl).transpose(1, 2)
            )  # (B, Cvae, l)
        return self.embed_to_logits(
            ms_h_BCl=ms_h_BCl, all_to_max_scale=same_shape, last_one=last_one
        )

    # ===================== embed_to_logits: Embeddings to logits =====================
    def embed_to_logits(
        self,
        ms_h_BCl: List[torch.Tensor],
        all_to_max_scale: bool = True,
        last_one: bool = False,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Convert multi-scale embeddings to logits.

        Args:
            ms_h_BCl: List of embeddings at each scale
                      [(B, Cvae, 1), (B, Cvae, 2), ..., (B, Cvae, L_latent)]
            all_to_max_scale: If True, upsample all to max scale for summing
            last_one: If True, return only final result

        Returns:
            If last_one: (B, L, vocab_size) final token logits
            Else: List of progressively accumulated logits
        """
        if last_one:
            f_hat = self.quantize.embed_to_fhat(
                ms_h_BCl, all_to_max_scale=all_to_max_scale, last_one=True
            )
            return self.decoder(self.post_quant_conv(f_hat))
        else:
            ls_f_hat = self.quantize.embed_to_fhat(
                ms_h_BCl, all_to_max_scale=all_to_max_scale, last_one=False
            )
            return [self.decoder(self.post_quant_conv(f_hat)) for f_hat in ls_f_hat]

    # ===================== inp_to_reconstructed: Full encode-decode cycle =====================
    def inp_to_reconstructed(
        self,
        inp: torch.Tensor,
        v_patch_lens: Optional[Sequence[int]] = None,
        last_one: bool = False,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Full reconstruction: encode -> quantize -> decode.

        Useful for testing reconstruction quality at each scale.

        Args:
            inp: (B, L) input token indices
            v_patch_lens: Custom scale lengths
            last_one: If True, return only final reconstruction

        Returns:
            If last_one: (B, L, vocab_size) final reconstructed logits
            Else: List of logits showing progressive reconstruction
                  as more scales are added to the sum
        """
        x = self.tok_embed(inp).transpose(1, 2)
        f = self.quant_conv(self.encoder(x))
        ls_f_hat_BCL = self.quantize.f_to_idxBl_or_fhat(
            f, to_fhat=True, v_patch_lens=v_patch_lens
        )

        if last_one:
            return self.decoder(self.post_quant_conv(ls_f_hat_BCL[-1]))
        else:
            return [self.decoder(self.post_quant_conv(f_hat)) for f_hat in ls_f_hat_BCL]

    # ===================== compute_loss: Training loss computation =====================
    def compute_loss(
        self,
        inp: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss for VQVAE.

        Loss = reconstruction_loss + vq_loss
        - reconstruction_loss: Cross-entropy between predicted and target tokens
        - vq_loss: Codebook commitment loss from VectorQuantizer2

        Args:
            inp: (B, L) input token indices
            labels: (B, L) target token indices (default: same as inp)
            attention_mask: (B, L) mask for valid positions (1=valid, 0=pad)

        Returns:
            Dict containing:
            - 'loss': Total loss (reconstruction + VQ)
            - 'recon_loss': Reconstruction cross-entropy loss
            - 'vq_loss': Vector quantization loss
            - 'logits': (B, L, vocab_size) output logits
        """
        if labels is None:
            labels = inp

        logits, usages, vq_loss = self.forward(inp, ret_usages=False)

        # Reconstruction loss
        logits_flat = logits.view(-1, self.vocab_size)
        labels_flat = labels.view(-1)

        if attention_mask is not None:
            mask_flat = attention_mask.view(-1).bool()
            recon_loss = nn.functional.cross_entropy(
                logits_flat[mask_flat], labels_flat[mask_flat], reduction="mean"
            )
        else:
            recon_loss = nn.functional.cross_entropy(
                logits_flat, labels_flat, reduction="mean"
            )

        loss = recon_loss + vq_loss

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "logits": logits,
        }

    def load_state_dict(
        self, state_dict: Dict[str, Any], strict: bool = True, assign: bool = False
    ):
        """Handle state dict loading with EMA buffer size mismatch."""
        if "quantize.ema_vocab_hit_SV" in state_dict:
            if (
                state_dict["quantize.ema_vocab_hit_SV"].shape[0]
                != self.quantize.ema_vocab_hit_SV.shape[0]
            ):
                state_dict["quantize.ema_vocab_hit_SV"] = self.quantize.ema_vocab_hit_SV
        return super().load_state_dict(
            state_dict=state_dict, strict=strict, assign=assign
        )
