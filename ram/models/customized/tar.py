"""TAR: Text Auto-Regressive Next-Scale Prediction Model.

Design pattern inspired by VAR-main/models/var.py.
Implementation is original for text next-scaling task.

Text-TAR Architecture Overview:
================================
TAR predicts text at multiple scales autoregressively:
Scale 1 (coarsest, len=1) -> Scale 2 (len=2) -> ... -> Scale K (finest, len=L_latent)

Key difference from VAR (images):
- VAR: Predicts 2D patch grids (1x1, 2x2, ..., 16x16)
- TAR: Predicts 1D sequence positions (1, 2, 4, 8, 16, 32)

Next-Scale Prediction:
======================
Given accumulated features f_hat from scales 1..k-1:
  1. Downsample f_hat to scale k length
  2. Project to transformer input space
  3. Predict codebook indices for all positions at scale k
  4. Look up embeddings, upsample, add to f_hat
  5. Repeat for next scale

Training (Teacher Forcing):
- Input: Ground-truth indices from all previous scales
- Target: Predict indices for all scales
- Loss: Cross-entropy over codebook indices

Inference (Autoregressive):
- Start with SOS embedding
- Generate each scale's indices sequentially
- Accumulate features and decode final text

Tensor dimension conventions:
- B: batch size
- L: sequence length (varies per scale)
- C: transformer embedding dimension
- D: conditioning dimension (same as C)
- Cvae: latent channel dimension from VQVAE
"""

import math
from functools import partial
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn

from .basic_tar import AdaLNBeforeHead, AdaLNSelfAttn
from ..utils.sampling import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from .vqvae import VQVAE
from .quant import VectorQuantizer2


__all__ = ["TAR", "ScaleAdaptiveLinear"]


class ScaleAdaptiveLinear(nn.Linear):
    """
    Scale-adaptive linear layer for generating AdaLN conditioning parameters.

    Generates 6 parameters per transformer block for adaptive modulation:
    (gamma1, gamma2, scale1, scale2, shift1, shift2)

    These parameters modulate the attention and FFN outputs based on
    which scale is being predicted.

    Shape:
        Input:  (B, D) scale/SOS conditioning embedding
        Output: (B, 1, 6, C) reshaped for AdaLN blocks
    """

    def forward(self, scale_cond: torch.Tensor) -> torch.Tensor:
        C = self.weight.shape[0] // 6
        # (B, D) -> (B, 6*C) -> (B, 1, 6, C) for broadcasting to all positions
        return super().forward(scale_cond).view(-1, 1, 6, C)


class TAR(nn.Module):
    """
    TAR: Text Auto-Regressive Next-Scale Prediction Model.

    Predicts text codebook indices at multiple scales autoregressively.
    Each scale is predicted conditioned on accumulated features from
    all previous scales.

    Complete Shape Flow (Example: scales=(1,2,4,8,16,32), embed_dim=1024):
    =====================================================================

    Training (forward pass with teacher forcing):
    ----------------------------------------------
    teacher_input: (B, sum(scales[1:]), Cvae) = (B, 62, 32)
      |-- Concatenated GT embeddings for scales 2..K
      |-- Each scale k provides scale_lens[k] positions

    SOS + position embedding: (B, scale_lens[0], C) = (B, 1, 1024)
    After token_proj: (B, 62, 1024) teacher forcing input
    Full input: (B, 63, 1024) = SOS + teacher
      |-- Add scale_embed: which scale each position belongs to
      |-- Add pos_embed: absolute position within full sequence

    Transformer blocks: (B, 63, 1024) -> (B, 63, 1024)
      |-- Causal attention: each scale sees all previous scales
      |-- AdaLN modulation based on SOS conditioning

    Output logits: (B, 63, codebook_size) = (B, 63, 4096)
      |-- Predict codebook index for each position

    Inference (autoregressive generation):
    --------------------------------------
    1. Start: SOS embed -> predict scale 1 indices (B, 1)
    2. Scale 2: f_hat from scale 1 -> predict (B, 2) indices
    3. Scale 3: f_hat from scales 1+2 -> predict (B, 4) indices
    ...continue until final scale...
    Final: f_hat = sum of all scales -> decode to text

    Args:
        tvqvae: Pre-trained Text VQ-VAE (provides codebook)
        depth: Number of transformer blocks
        embed_dim: Transformer hidden dimension
        num_heads: Number of attention heads
        scale_lens: Sequence lengths per scale, e.g., (1,2,4,8,16,32)
    """

    def __init__(
        self,
        tvqvae,  # Pre-trained Text VQ-VAE
        depth: int = 16,
        embed_dim: int = 1024,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_eps: float = 1e-6,
        shared_aln: bool = False,
        cond_drop_rate: float = 0.1,
        attn_l2_norm: bool = False,
        scale_lens: Tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        flash_if_available: bool = True,
        fused_if_available: bool = True,
    ):
        super().__init__()

        # 0. Hyperparameters
        assert embed_dim % num_heads == 0
        self.latent_dim = tvqvae.Cvae  # Latent channel dimension
        self.codebook_size = tvqvae.quantize.vocab_size  # Codebook vocabulary size
        self.depth = depth
        self.C = embed_dim
        self.D = embed_dim  # Conditioning dimension
        self.num_heads = num_heads

        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1  # Progressive training (not yet supported)

        self.scale_lens: Tuple[int, ...] = scale_lens
        self.total_len = sum(self.scale_lens)  # Total sequence length across all scales
        self.first_scale_len = self.scale_lens[0]  # First scale length
        self.scale_ranges = []  # (start, end) indices for each scale
        cur = 0
        for i, sl in enumerate(self.scale_lens):
            self.scale_ranges.append((cur, cur + sl))
            cur += sl

        self.num_scales_minus_1 = len(self.scale_lens) - 1
        self.rng = torch.Generator(device="cpu")

        # 1. Token embedding: project from latent_dim to transformer embed_dim
        # When predicting scale k, input is f_hat (accumulated from scales 1..k-1)
        # downsampled to scale k length, then projected to transformer space
        quantizer = tvqvae.quantize
        self.tvqvae_ref = (tvqvae,)  # Store reference to TVQVAE
        self.quantizer_ref = (quantizer,)  # Store reference to quantizer
        self.token_proj = nn.Linear(self.latent_dim, self.C)  # (Cvae, C)

        # 2. Start-of-sequence (SOS) embedding
        # For text generation, SOS provides initial conditioning
        # Unlike VAR's class embedding (1000 ImageNet classes),
        # we use a single learned SOS token
        init_std = math.sqrt(1 / self.C / 3)
        self.sos_embed = nn.Parameter(torch.randn(1, self.C) * init_std)  # (1, C)

        # Position embedding for first scale (separate from pos_embed)
        # Shape: (1, first_scale_len, C) = (1, 1, 1024) for scales starting with 1
        self.pos_start = nn.Parameter(torch.empty(1, self.first_scale_len, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)

        # 3. Absolute position embedding for all scales concatenated
        # For scales (1,2,4,8,16,32), total_len = 63 positions
        # pos_embed: (1, 63, C) - unique position for each slot across all scales
        pos_1LC = []
        for i, sl in enumerate(self.scale_lens):
            pe = torch.empty(1, sl, self.C)  # (1, scale_len, C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)  # (1, total_len, C)
        assert tuple(pos_1LC.shape) == (1, self.total_len, self.C)
        self.pos_embed = nn.Parameter(pos_1LC)  # (1, total_len, C)

        # Scale level embedding - tells transformer which scale is being predicted
        # Shape: (num_scales, C) = (6, 1024) for 6 scales
        self.scale_embed = nn.Embedding(len(self.scale_lens), self.C)
        nn.init.trunc_normal_(self.scale_embed.weight.data, mean=0, std=init_std)

        # 4. Backbone transformer blocks with AdaLN conditioning
        # AdaLN modulates transformer based on scale/SOS embedding
        self.shared_ada_lin = (
            nn.Sequential(
                nn.SiLU(inplace=False),
                ScaleAdaptiveLinear(self.D, 6 * self.C),  # (D, 6*C) for 6 AdaLN params
            )
            if shared_aln
            else nn.Identity()
        )

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # Stochastic depth

        self.blocks = nn.ModuleList(
            [
                AdaLNSelfAttn(
                    cond_dim=self.D,
                    shared_aln=shared_aln,
                    block_idx=block_idx,
                    embed_dim=self.C,
                    norm_layer=norm_layer,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[block_idx],
                    last_drop_p=0 if block_idx == 0 else dpr[block_idx - 1],
                    attn_l2_norm=attn_l2_norm,
                    flash_if_available=flash_if_available,
                    fused_if_available=fused_if_available,
                )
                for block_idx in range(depth)
            ]
        )

        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)

        # 5. Attention mask for training (causal within scales)
        # Each position can attend to same or earlier scales
        # scale_idx[i] = which scale position i belongs to
        # For scales (1,2,4,8,16,32): positions 0 are scale 0, positions 1-2 are scale 1, etc.
        scale_idx: torch.Tensor = torch.cat(
            [torch.full((sl,), i) for i, sl in enumerate(self.scale_lens)]
        ).view(
            1, self.total_len, 1
        )  # (1, total_len, 1)
        scale_idx_T = scale_idx.transpose(1, 2)  # (1, 1, total_len)
        scale_1L = scale_idx_T[:, 0].contiguous()  # (1, total_len)
        self.register_buffer("scale_1L", scale_1L)
        # Mask: can attend if query_scale >= key_scale (causal across scales)
        # Shape: (1, 1, total_len, total_len)
        attn_mask = torch.where(scale_idx >= scale_idx_T, 0.0, -torch.inf).reshape(
            1, 1, self.total_len, self.total_len
        )
        self.register_buffer("attn_mask", attn_mask.contiguous())

        # 6. Prediction head: transforms to codebook logits
        # Shape: (B, L, C) -> (B, L, codebook_size)
        self.head_norm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.codebook_size)  # (C, codebook_size)

    def get_logits(
        self,
        h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        scale_cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Get codebook logits from transformer output.

        Args:
            h_or_h_and_residual: Either (B, L, C) tensor or tuple of tensors
            scale_cond: (B, D) conditioning for AdaLN

        Returns:
            (B, L, codebook_size) logits for codebook prediction
        """
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual
            h = resi + self.blocks[-1].drop_path(h)
        else:
            h = h_or_h_and_residual
        # AdaLN normalization then project to codebook logits
        return self.head(self.head_norm(h.float(), scale_cond).float()).float()

    @torch.no_grad()
    def autoregressive_infer(
        self,
        B: int,
        g_seed: Optional[int] = None,
        top_k: int = 0,
        top_p: float = 0.0,
        more_smooth: bool = False,
    ) -> torch.Tensor:
        """
        Autoregressive inference for text generation.

        Generates codebook indices scale-by-scale, accumulating features
        in latent_accum. After all scales, latent_accum can be decoded to text.

        Args:
            B: Batch size
            g_seed: Random seed for reproducibility
            top_k: Top-k sampling (0 = disabled)
            top_p: Nucleus sampling threshold (0 = disabled)
            more_smooth: Use Gumbel softmax for smoother outputs

        Returns:
            latent_accum: (B, Cvae, max_scale_len) accumulated latent features
                          This is the sum of all scale contributions:
                          latent_accum = h_1 + h_2 + ... + h_K
                          Pass to tvqvae.fhat_to_logits() for final text

        Generation Flow:
            1. SOS embedding -> predict scale 1 (B, 1) indices
            2. Lookup scale 1 embeddings, upsample, add to latent_accum
            3. Downsample latent_accum to scale 2 length -> predict (B, 2) indices
            4. Continue accumulating until final scale
            5. Return latent_accum for decoder
        """
        if g_seed is None:
            rng = None
        else:
            self.rng.manual_seed(g_seed)
            rng = self.rng

        # Start-of-sequence embedding
        sos = scale_cond = self.sos_embed.expand(B, -1)  # (B, C)

        scale_pos = self.scale_embed(self.scale_1L) + self.pos_embed
        next_token_map = (
            sos.unsqueeze(1).expand(B, self.first_scale_len, -1)
            + self.pos_start.expand(B, self.first_scale_len, -1)
            + scale_pos[:, : self.first_scale_len]
        )

        # Initialize latent accumulator at max scale length
        # This will store the sum of all scale features
        cur_pos = 0
        max_len = self.scale_lens[-1]
        latent_accum = sos.new_zeros(B, self.latent_dim, max_len)  # (B, Cvae, L_max)

        # Enable KV caching
        for b in self.blocks:
            b.attn.kv_caching(True)

        for si, sl in enumerate(self.scale_lens):
            ratio = si / self.num_scales_minus_1 if self.num_scales_minus_1 > 0 else 0
            cur_pos += sl

            scale_cond_proc = self.shared_ada_lin(scale_cond)
            x = next_token_map

            for b in self.blocks:
                x = b(x=x, cond_BD=scale_cond_proc, attn_bias=None)

            logits = self.get_logits(x, scale_cond)

            # Sample codebook indices from logits
            # idx: (B, scale_len) discrete codes for this scale
            idx = sample_with_top_k_top_p_(
                logits, rng=rng, top_k=top_k, top_p=top_p, num_samples=1
            )[
                :, :, 0
            ]  # (B, scale_len)

            # Look up embeddings from codebook
            # h: (B, Cvae, scale_len) embeddings for predicted indices
            if not more_smooth:
                h = self.quantizer_ref[0].embedding(idx)  # (B, scale_len, Cvae)
            else:
                # Gumbel softmax for differentiable soft selection
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                h = gumbel_softmax_with_rng(
                    logits.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng
                ) @ self.quantizer_ref[0].embedding.weight.unsqueeze(
                    0
                )  # (B, scale_len, Cvae)

            h = h.transpose(1, 2)  # (B, Cvae, scale_len)

            # Update latent_accum with this scale's contribution
            # get_next_autoregressive_input does:
            #   1. Upsample h from scale_len to max_len
            #   2. Apply phi transformation
            #   3. Add to latent_accum (accumulation!)
            #   4. Return downsampled latent_accum for next scale conditioning
            latent_accum, next_token_map = self.quantizer_ref[
                0
            ].get_next_autoregressive_input(
                si, len(self.scale_lens), latent_accum, h
            )  # latent_accum: (B, Cvae, max_len), next_token_map: (B, Cvae, next_scale_len)

            # Prepare input for next scale prediction
            if si != self.num_scales_minus_1:
                # Transpose: (B, Cvae, next_len) -> (B, next_len, Cvae)
                # Project to transformer space and add position embeddings
                next_token_map = next_token_map.transpose(1, 2)  # (B, next_len, Cvae)
                next_token_map = (
                    self.token_proj(next_token_map)
                    + scale_pos[:, cur_pos : cur_pos + self.scale_lens[si + 1]]
                )

        # Disable KV caching after inference
        for b in self.blocks:
            b.attn.kv_caching(False)

        # Return accumulated features: (B, Cvae, max_len)
        # Use tvqvae.fhat_to_logits(latent_accum) to get final text logits
        return latent_accum

    def forward(
        self,
        teacher_input: torch.Tensor,
        scale_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training with teacher forcing.

        Args:
            teacher_input: (B, total_len - first_scale_len, Cvae)
                           Ground-truth embeddings for all scales except first.
                           For scales (1,2,4,8,16,32), this is (B, 62, 32)
            scale_cond: (B, D) optional conditioning (default: use SOS)

        Returns:
            logits: (B, total_len, codebook_size) predictions for all positions
                    For scales (1,2,4,8,16,32), this is (B, 63, 4096)

        Training Process:
            1. Concatenate SOS with projected teacher_input
            2. Add scale embeddings (which scale each position is)
            3. Add position embeddings (absolute position)
            4. Apply transformer blocks with causal attention
            5. Predict codebook logits for all positions
        """
        start, end = (
            self.scale_ranges[self.prog_si]
            if self.prog_si >= 0
            else (0, self.total_len)
        )
        B = teacher_input.shape[0]

        with torch.cuda.amp.autocast(enabled=False):
            # Use SOS embedding if no conditioning provided
            if scale_cond is None:
                scale_cond = self.sos_embed.expand(B, -1)

            # Apply conditioning dropout during training
            if self.training and self.cond_drop_rate > 0:
                mask = torch.rand(B, device=scale_cond.device) < self.cond_drop_rate
                scale_cond = torch.where(
                    mask.unsqueeze(1), torch.zeros_like(scale_cond), scale_cond
                )

            sos = scale_cond
            sos = sos.unsqueeze(1).expand(
                B, self.first_scale_len, -1
            ) + self.pos_start.expand(B, self.first_scale_len, -1)

            if self.prog_si == 0:
                x = sos
            else:
                x = torch.cat((sos, self.token_proj(teacher_input.float())), dim=1)

            x += (
                self.scale_embed(self.scale_1L[:, :end].expand(B, -1))
                + self.pos_embed[:, :end]
            )

        attn_bias = self.attn_mask[:, :, :end, :end]
        scale_cond_proc = self.shared_ada_lin(scale_cond)

        # Get dtype for mixed precision
        temp = x.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype

        x = x.to(dtype=main_type)
        scale_cond_proc = scale_cond_proc.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        # Apply transformer blocks
        for i, b in enumerate(self.blocks):
            x = b(x=x, cond_BD=scale_cond_proc, attn_bias=attn_bias)

        x = self.get_logits(x.float(), scale_cond)

        # Ensure gradients flow to token_proj even when prog_si=0
        if self.prog_si == 0:
            if isinstance(self.token_proj, nn.Linear):
                x[0, 0, 0] += (
                    self.token_proj.weight[0, 0] * 0 + self.token_proj.bias[0] * 0
                )
            else:
                s = 0
                for p in self.token_proj.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x[0, 0, 0] += s

        return x  # logits: (B, total_len, codebook_size)

    def init_weights(
        self,
        init_adaln: float = 0.5,
        init_adaln_gamma: float = 1e-5,
        init_head: float = 0.02,
        init_std: float = 0.02,
        conv_std_or_gain: float = 0.02,
    ):
        """Initialize weights with truncated normal distribution."""
        if init_std < 0:
            init_std = (1 / self.C / 3) ** 0.5

        print(f"[init_weights] {type(self).__name__} with {init_std=:g}")
        for m in self.modules():
            with_weight = hasattr(m, "weight") and m.weight is not None
            with_bias = hasattr(m, "bias") and m.bias is not None

            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(
                m,
                (
                    nn.LayerNorm,
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                    nn.SyncBatchNorm,
                    nn.GroupNorm,
                    nn.InstanceNorm1d,
                    nn.InstanceNorm2d,
                    nn.InstanceNorm3d,
                ),
            ):
                if with_weight:
                    m.weight.data.fill_(1.0)
                if with_bias:
                    m.bias.data.zero_()
            elif isinstance(
                m,
                (
                    nn.Conv1d,
                    nn.Conv2d,
                    nn.Conv3d,
                    nn.ConvTranspose1d,
                    nn.ConvTranspose2d,
                    nn.ConvTranspose3d,
                ),
            ):
                if conv_std_or_gain > 0:
                    nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else:
                    nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias:
                    m.bias.data.zero_()

        # Initialize head
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()

        # Initialize AdaLN before head
        if isinstance(self.head_norm, AdaLNBeforeHead):
            self.head_norm.ada_lin[-1].weight.data.mul_(init_adaln)
            if (
                hasattr(self.head_norm.ada_lin[-1], "bias")
                and self.head_norm.ada_lin[-1].bias is not None
            ):
                self.head_norm.ada_lin[-1].bias.data.zero_()

        # Initialize transformer blocks
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, "fcg") and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, "ada_lin"):
                sab.ada_lin[-1].weight.data[2 * self.C :].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[: 2 * self.C].mul_(init_adaln_gamma)
                if (
                    hasattr(sab.ada_lin[-1], "bias")
                    and sab.ada_lin[-1].bias is not None
                ):
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, "ada_gss"):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)

    def extra_repr(self) -> str:
        return f"drop_path_rate={self.drop_path_rate:g}"


def build_tar(
    device,
    scale_lens: Tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    # TVQVAE args
    vocab_size: int = 32000,
    embed_dim: int = 768,
    codebook_size: int = 4096,
    latent_dim: int = 32,
    ch: int = 128,
    share_quant_resi: int = 4,
    # TAR args
    depth: int = 16,
    shared_aln: bool = False,
    attn_l2_norm: bool = True,
    flash_if_available: bool = True,
    fused_if_available: bool = True,
    init_adaln: float = 0.5,
    init_adaln_gamma: float = 1e-5,
    init_head: float = 0.02,
    init_std: float = -1,
) -> Tuple[VQVAE, TAR]:
    """
    Build TVQVAE and TAR models for text next-scaling.

    Args:
        device: Target device
        scale_lens: Sequence lengths for each scale
        vocab_size: Token vocabulary size
        embed_dim: Token embedding dimension
        codebook_size: Codebook vocabulary size
        latent_dim: VAE latent channel dimension
        ch: Base channel count for VAE
        share_quant_resi: Phi layer sharing mode
        depth: Number of TAR transformer blocks
        shared_aln: Use shared adaptive layer norm
        attn_l2_norm: Use L2 normalized attention
        flash_if_available: Use flash attention
        fused_if_available: Use fused operations
        init_adaln: AdaLN initialization scale
        init_adaln_gamma: AdaLN gamma initialization scale
        init_head: Head initialization scale
        init_std: Standard deviation for initialization (-1 for auto)

    Returns:
        tvqvae: Text VQ-VAE model
        tar: TAR model
    """
    heads = depth
    width = depth * 64
    dpr = 0.1 * depth / 24

    # Disable built-in initialization for speed
    for clz in (
        nn.Linear,
        nn.LayerNorm,
        nn.BatchNorm2d,
        nn.SyncBatchNorm,
        nn.Conv1d,
        nn.Conv2d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
    ):
        setattr(clz, "reset_parameters", lambda self: None)

    # Build TVQVAE
    tvqvae = VQVAE(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        z_channels=latent_dim,
        ch=ch,
        v_patch_lens=scale_lens,
        share_quant_resi=share_quant_resi,
        test_mode=True,
    ).to(device)

    # Build TAR
    tar = TAR(
        tvqvae=tvqvae,
        depth=depth,
        embed_dim=width,
        num_heads=heads,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=dpr,
        norm_eps=1e-6,
        shared_aln=shared_aln,
        cond_drop_rate=0.1,
        attn_l2_norm=attn_l2_norm,
        scale_lens=scale_lens,
        flash_if_available=flash_if_available,
        fused_if_available=fused_if_available,
    ).to(device)

    tar.init_weights(
        init_adaln=init_adaln,
        init_adaln_gamma=init_adaln_gamma,
        init_head=init_head,
        init_std=init_std,
    )

    return tvqvae, tar
