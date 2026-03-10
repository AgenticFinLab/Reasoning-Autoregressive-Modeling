"""Text VAE basic building blocks.

Design pattern inspired by VAR-main/models/basic_vae.py.
Implementation is original for text sequence processing.

Key differences from VAR (image-based):
- VAR uses Conv2D for 2D images: (B, C, H, W)
- We use Conv1D for 1D text sequences: (B, C, L)

Text-specific considerations:
- L = sequence length (e.g., 512 tokens)
- C = embedding/channel dimension (e.g., 768 for BERT-base)
- Downsampling/upsampling along sequence dimension only
- No spatial 2D operations - purely sequential

Tensor dimension conventions:
- B: batch size
- C: channel/embedding dimension
- L: sequence length
- Shape annotations use format: (B, C, L) = (batch, channels, length)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Encoder", "Decoder"]


# ============================================================
# Basic Building Blocks for 1D Text Sequences
# Design pattern inspired by VAR's basic_vae.py
# ============================================================


def Normalize1D(in_channels, num_groups=32):
    """
    GroupNorm for 1D text sequences.

    GroupNorm normalizes across channel groups, independent of batch size.
    More stable than BatchNorm for small batches common in text processing.

    Args:
        in_channels: Number of input channels (e.g., 768)
        num_groups: Number of groups for GroupNorm (default: 32)

    Input shape: (B, C, L) where C = in_channels
    Output shape: (B, C, L) - same as input
    """
    return nn.GroupNorm(
        num_groups=min(num_groups, in_channels),
        num_channels=in_channels,
        eps=1e-6,
        affine=True,
    )


class Downsample1D(nn.Module):
    """
    1D downsampling for text sequences using strided convolution.

    Reduces sequence length by factor of 2 while preserving channel dimension.
    This creates coarser-grained representations for multi-scale processing.

    Text-specific: Unlike VAR's 2D spatial downsampling for images,
    we only downsample along the sequence (temporal) dimension.

    Args:
        in_channels: Number of input/output channels

    Shape:
        Input:  (B, C, L)     - e.g., (32, 768, 512)
        Output: (B, C, L//2)  - e.g., (32, 768, 256)
    """

    def __init__(self, in_channels):
        super().__init__()
        # stride=2 halves sequence length: L -> L//2
        self.conv = nn.Conv1d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        # x: (B, C, L) -> (B, C, L//2)
        return self.conv(x)


class Upsample1D(nn.Module):
    """
    1D upsampling for text sequences using nearest-neighbor + convolution.

    Doubles sequence length to restore fine-grained temporal resolution.
    Uses nearest-neighbor interpolation (repeat) followed by learned conv.

    Text-specific: For discrete text, we use 'nearest' mode (repeat tokens)
    rather than 'linear' interpolation which might create invalid intermediate values.
    The convolution then learns to refine these repeated representations.

    Args:
        in_channels: Number of input/output channels

    Shape:
        Input:  (B, C, L)    - e.g., (32, 768, 256)
        Output: (B, C, L*2)  - e.g., (32, 768, 512)
    """

    def __init__(self, in_channels):
        super().__init__()
        # Conv refines the upsampled representation
        self.conv = nn.Conv1d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        # x: (B, C, L)
        # Step 1: nearest-neighbor upsample L -> L*2 (each position repeated)
        # Step 2: conv refines the repeated values
        # Output: (B, C, L*2)
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class ResnetBlock1D(nn.Module):
    """
    1D ResNet block for text sequence processing.

    Structure: norm1 -> SiLU -> conv1 -> norm2 -> dropout -> SiLU -> conv2 + shortcut

    Uses residual connection for stable training of deep networks.
    SiLU activation (Swish) provides smooth gradients.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (default: same as in_channels)
        dropout: Dropout probability (default: 0.0)

    Shape:
        Input:  (B, in_channels, L)
        Output: (B, out_channels, L)
    """

    def __init__(self, *, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize1D(in_channels)  # (B, in_ch, L) -> (B, in_ch, L)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = Normalize1D(out_channels)  # (B, out_ch, L) -> (B, out_ch, L)
        self.dropout = nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity()
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        # Shortcut projection if channel dimensions differ
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x):
        # x: (B, in_channels, L)
        h = self.conv1(F.silu(self.norm1(x), inplace=True))  # (B, out_channels, L)
        h = self.conv2(
            self.dropout(F.silu(self.norm2(h), inplace=True))
        )  # (B, out_channels, L)
        return self.nin_shortcut(x) + h  # (B, out_channels, L) + residual


class AttnBlock1D(nn.Module):
    """
    1D self-attention block for text sequences.

    Enables long-range dependencies in the sequence by allowing each position
    to attend to all other positions. Essential for capturing global context.

    Args:
        in_channels: Number of input/output channels

    Shape:
        Input:  (B, C, L)
        Output: (B, C, L)

    Attention computation:
        1. Project to Q, K, V: (B, C, L) -> (B, 3C, L) -> 3x(B, C, L)
        2. Attention scores: Q^T @ K -> (B, L, L)
        3. Weighted values: V @ softmax(scores) -> (B, C, L)
    """

    def __init__(self, in_channels):
        super().__init__()
        self.C = in_channels

        self.norm = Normalize1D(in_channels)  # (B, C, L) -> (B, C, L)
        self.qkv = nn.Conv1d(
            in_channels, 3 * in_channels, kernel_size=1, stride=1, padding=0
        )
        self.w_ratio = int(in_channels) ** (-0.5)  # Scale factor: 1/sqrt(C)
        self.proj_out = nn.Conv1d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        # x: (B, C, L)
        qkv = self.qkv(self.norm(x))  # (B, 3C, L)
        B, _, L = qkv.shape
        C = self.C

        # Split into Q, K, V: each (B, C, L)
        q, k, v = qkv.reshape(B, 3, C, L).unbind(1)

        # Compute attention scores: (B, L, L)
        q = q.permute(0, 2, 1).contiguous()  # (B, L, C) - queries
        k = k.contiguous()  # (B, C, L) - keys
        w = torch.bmm(q, k).mul_(self.w_ratio)  # (B, L, L) = Q @ K^T / sqrt(C)
        w = F.softmax(w, dim=2)  # Normalize attention over key positions

        # Apply attention to values
        v = v.contiguous()  # (B, C, L) - values
        w = w.permute(0, 2, 1).contiguous()  # (B, L, L) - transpose for bmm
        h = torch.bmm(v, w)  # (B, C, L) = V @ attention_weights

        return x + self.proj_out(h)  # (B, C, L) + residual


def make_attn(in_channels, using_sa=True):
    """
    Factory function for attention blocks.

    Args:
        in_channels: Number of input channels
        using_sa: Whether to use self-attention (if False, returns Identity)
    """
    return AttnBlock1D(in_channels) if using_sa else nn.Identity()


# ============================================================
# Encoder for Text Sequences
# Design pattern inspired by VAR's Encoder
# ============================================================


class Encoder(nn.Module):
    """
    1D Encoder for text sequences - extracts multi-scale latent representations.

    Progressively downsamples text embeddings to create compressed latent codes.
    These codes capture hierarchical semantics at different granularities.

    Architecture:
        Input Projection -> [ResBlock + Downsample] x N -> Middle Block -> Output Projection

    Shape Flow (example with ch_mult=(1,2,4,8), ch=128, L=512):
        Input:  (B, embed_dim, L)     = (B, 768, 512)  - token embeddings
        conv_in: (B, ch, L)           = (B, 128, 512)  - project to base channels
        down[0]: (B, ch*1, L)         = (B, 128, 512)  - resolution 0
        down[0].downsample: (B, 128, 256)              - halve length
        down[1]: (B, ch*2, L/2)       = (B, 256, 256)  - resolution 1
        down[1].downsample: (B, 256, 128)              - halve length
        down[2]: (B, ch*4, L/4)       = (B, 512, 128)  - resolution 2
        down[2].downsample: (B, 512, 64)               - halve length
        down[3]: (B, ch*8, L/8)       = (B, 1024, 64)  - resolution 3 (lowest)
        middle:  (B, ch*8, L/8)       = (B, 1024, 64)  - middle processing
        Output:  (B, z_channels, L/8) = (B, 32, 64)    - latent codes

    Args:
        ch: Base channel count (default: 128)
        ch_mult: Channel multipliers per resolution (default: (1,2,4,8))
        num_res_blocks: ResNet blocks per resolution (default: 2)
        dropout: Dropout rate (default: 0.0)
        in_channels: Input embedding dimension (default: 768)
        z_channels: Output latent dimension (default: 32)
        using_sa: Use self-attention at lowest resolution
        using_mid_sa: Use self-attention in middle block
    """

    def __init__(
        self,
        *,
        ch=128,  # Base channel count
        ch_mult=(1, 2, 4, 8),  # Channel multipliers per resolution
        num_res_blocks=2,  # ResNet blocks per resolution
        dropout=0.0,
        in_channels=768,  # Input: embedding dimension
        z_channels=32,  # Output: latent dimension
        using_sa=True,  # Use self-attention at lowest resolution
        using_mid_sa=True,  # Use self-attention in middle block
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.downsample_ratio = 2 ** (self.num_resolutions - 1)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # Input projection: embed_dim → ch
        self.conv_in = nn.Conv1d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        # Downsampling blocks
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = ch

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock1D(
                        in_channels=block_in, out_channels=block_out, dropout=dropout
                    )
                )
                block_in = block_out
                # Self-attention only at lowest resolution
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample1D(block_in)
            self.down.append(down)

        # Middle block
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(
            in_channels=block_in, out_channels=block_in, dropout=dropout
        )
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock1D(
            in_channels=block_in, out_channels=block_in, dropout=dropout
        )

        # Output projection: block_in → z_channels
        self.norm_out = Normalize1D(block_in)
        self.conv_out = nn.Conv1d(
            block_in, z_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        """
        Encode text embeddings to latent codes.

        Args:
            x: (B, in_channels, L) or (B, L, in_channels) - token embeddings

        Returns:
            (B, z_channels, L // downsample_ratio) - compressed latent codes

        Example (ch_mult=(1,2,4,8), downsample_ratio=8):
            Input:  (32, 768, 512)  -> 32 samples, 768-dim embeddings, 512 tokens
            Output: (32, 32, 64)    -> 32 samples, 32-dim latents, 64 positions
        """
        # Handle (B, L, D) input format - transpose to (B, D, L) for Conv1d
        if x.dim() == 3 and x.size(2) == self.in_channels:
            x = x.transpose(1, 2)  # (B, L, D) -> (B, D, L)

        # Input projection: (B, embed_dim, L) -> (B, ch, L)
        h = self.conv_in(x)

        # Progressive downsampling through resolutions
        for i_level in range(self.num_resolutions):
            # Apply ResBlocks at current resolution
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)  # (B, ch*mult[i], L_i)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)  # Self-attention
            # Downsample (except at final resolution)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)  # (B, C, L_i) -> (B, C, L_i//2)

        # Middle block at lowest resolution: (B, ch*max_mult, L_min)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))

        # Output projection: (B, ch*max_mult, L_min) -> (B, z_channels, L_min)
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        return h  # (B, z_channels, L // downsample_ratio)


# ============================================================
# Decoder for Text Reconstruction
# Design pattern inspired by VAR's Decoder
# ============================================================


class Decoder(nn.Module):
    """
    1D Decoder for text reconstruction - reconstructs tokens from latent codes.

    Takes accumulated multi-scale latent features (f_hat from VQ) and reconstructs
    original token sequence. This is the final step in the Text-VAR pipeline.

    Key Text-VAR concept:
        The decoder receives f_hat which is the SUMMATION of all scale features:
        f_hat = sum(upsample(scale_k_features) for k in 1..K)
        This accumulated feature contains both global structure (coarse scales)
        and local details (fine scales) combined additively.

    Architecture:
        Input Projection -> Middle Block -> [ResBlock + Upsample] x N -> Output Projection -> Vocab

    Shape Flow (example with ch_mult=(1,2,4,8), ch=128):
        Input:   (B, z_channels, L_min)  = (B, 32, 64)    - accumulated f_hat
        conv_in: (B, ch*8, L_min)        = (B, 1024, 64)  - project to max channels
        middle:  (B, ch*8, L_min)        = (B, 1024, 64)  - middle processing
        up[3]:   (B, ch*8, L_min)        = (B, 1024, 64)  - resolution 3
        up[2]:   (B, ch*4, L/4)          = (B, 512, 128)  - upsample + process
        up[1]:   (B, ch*2, L/2)          = (B, 256, 256)  - upsample + process
        up[0]:   (B, ch*1, L)            = (B, 128, 512)  - upsample + process
        conv_out:(B, out_channels, L)    = (B, 768, 512)  - project to embed dim
        Output:  (B, L, vocab_size)      = (B, 512, 32000)- token logits

    Args:
        ch: Base channel count
        ch_mult: Channel multipliers per resolution
        num_res_blocks: ResNet blocks per resolution
        dropout: Dropout rate
        z_channels: Input latent dimension
        out_channels: Output dimension before vocab projection
        vocab_size: Vocabulary size for final projection
        using_sa: Use self-attention at lowest resolution
        using_mid_sa: Use self-attention in middle block
    """

    def __init__(
        self,
        *,
        ch=128,  # Base channel count
        ch_mult=(1, 2, 4, 8),  # Channel multipliers per resolution
        num_res_blocks=2,  # ResNet blocks per resolution
        dropout=0.0,
        z_channels=32,  # Input: latent dimension
        out_channels=768,  # Output dimension before vocab projection
        vocab_size=32000,  # Vocabulary size for final projection
        using_sa=True,  # Use self-attention at lowest resolution
        using_mid_sa=True,  # Use self-attention in middle block
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.vocab_size = vocab_size

        # Compute block_in at lowest resolution
        block_in = ch * ch_mult[self.num_resolutions - 1]

        # Input projection: z_channels → block_in
        self.conv_in = nn.Conv1d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # Middle block
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(
            in_channels=block_in, out_channels=block_in, dropout=dropout
        )
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock1D(
            in_channels=block_in, out_channels=block_in, dropout=dropout
        )

        # Upsampling blocks
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]

            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock1D(
                        in_channels=block_in, out_channels=block_out, dropout=dropout
                    )
                )
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))

            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample1D(block_in)
            self.up.insert(0, up)

        # Output projection
        self.norm_out = Normalize1D(block_in)
        self.conv_out = nn.Conv1d(
            block_in, out_channels, kernel_size=3, stride=1, padding=1
        )

        # Vocabulary projection
        self.vocab_proj = nn.Linear(out_channels, vocab_size)

    def forward(self, z):
        """
        Decode accumulated latent features to token logits.

        Args:
            z: (B, z_channels, L_latent) - accumulated f_hat from multi-scale VQ
               This is the sum of all upsampled scale features:
               z = f_hat = sum_k(upsample_to_L(scale_k_embedding))

        Returns:
            (B, L, vocab_size) - logits for each token position

        Example:
            Input:  (32, 32, 64)    -> accumulated multi-scale features
            Output: (32, 512, 32000) -> logits for 512 tokens over 32k vocab
        """
        # Input projection + middle block at lowest resolution
        # z: (B, z_channels, L_min) -> (B, ch*max_mult, L_min)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(self.conv_in(z))))

        # Progressive upsampling through resolutions (reverse of encoder)
        for i_level in reversed(range(self.num_resolutions)):
            # Apply ResBlocks at current resolution
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)  # (B, ch*mult[i], L_i)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)  # Self-attention
            # Upsample (except at final resolution)
            if i_level != 0:
                h = self.up[i_level].upsample(h)  # (B, C, L_i) -> (B, C, L_i*2)

        # Output projection: (B, ch, L) -> (B, out_channels, L)
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))

        # Transpose and project to vocabulary: (B, out_channels, L) -> (B, L, vocab_size)
        h = h.transpose(1, 2)  # (B, L, out_channels)
        logits = self.vocab_proj(h)  # (B, L, vocab_size)

        return logits
