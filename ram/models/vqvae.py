"""
Text VQVAE: Vector Quantized Variational Autoencoder for Text.

Mirrors: third-part/VAR-main/models/vqvae.py
- VQVAE: Combines Encoder + VectorQuantizer2 + Decoder

Architecture flow (mirror VAR's vqvae.py lines 56-59):
1. Encoder: text embeddings → latent features f
2. quant_conv: f → f (optional projection)
3. VectorQuantizer2: f → f_hat (multi-scale residual quantization)
4. post_quant_conv: f_hat → f_hat (optional projection)
5. Decoder: f_hat → token logits
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .basic_vae import Encoder, Decoder
from .quant import VectorQuantizer2

__all__ = ['VQVAE']


class VQVAE(nn.Module):
    """
    Text VQVAE model.
    Mirror: vqvae.py VQVAE (lines 16-95)
    
    Architecture:
        Input (B, L) token ids
        → Embedding (B, L, embed_dim)
        → Encoder → (B, Cvae, L//downsample)
        → quant_conv → (B, Cvae, L//downsample)
        → VectorQuantizer2 → f_hat (B, Cvae, L//downsample)
        → post_quant_conv → (B, Cvae, L//downsample)
        → Decoder → (B, L, vocab_size) logits
    
    Args:
        vocab_size: Token vocabulary size
        embed_dim: Input embedding dimension
        z_channels: Latent channel dimension (Cvae)
        ch: Base channel count for encoder/decoder
        ch_mult: Channel multipliers per resolution
        num_res_blocks: ResNet blocks per resolution
        dropout: Dropout rate
        beta: VQ commitment loss weight
        using_znorm: Whether to use z-normalization in VQ
        quant_conv_ks: Kernel size for quant/post_quant conv
        quant_resi: Residual ratio for φ layers
        share_quant_resi: How to share φ layers (0=none, 1=full, N=partial)
        v_patch_lens: Sequence lengths for each scale
        test_mode: If True, freeze weights and set eval mode
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
        self.quant_conv = nn.Conv1d(z_channels, z_channels, quant_conv_ks, stride=1, padding=quant_conv_ks // 2)
        self.post_quant_conv = nn.Conv1d(z_channels, z_channels, quant_conv_ks, stride=1, padding=quant_conv_ks // 2)
        
        if self.test_mode:
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
    
    # ===================== forward: used in VQVAE training =====================
    def forward(
        self, 
        inp: torch.Tensor, 
        ret_usages: bool = False
    ) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        """
        Full forward pass for training.
        Mirror: vqvae.py forward (lines 56-59)
        
        Args:
            inp: (B, L) input token indices
            ret_usages: Whether to return codebook usage stats
            
        Returns:
            logits: (B, L, vocab_size) reconstructed token logits
            usages: Codebook usage per scale (if ret_usages=True)
            vq_loss: Vector quantization loss
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
    
    # ===================== fhat_to_logits: decode f_hat to logits =====================
    def fhat_to_logits(self, f_hat: torch.Tensor) -> torch.Tensor:
        """
        Decode f_hat to token logits.
        Mirror: vqvae.py fhat_to_img (lines 62-63)
        
        Args:
            f_hat: (B, Cvae, L_latent) accumulated features
            
        Returns:
            (B, L, vocab_size) token logits
        """
        return self.decoder(self.post_quant_conv(f_hat))
    
    # ===================== inp_to_idxBl: encode input to multi-scale indices =====================
    def inp_to_idxBl(
        self, 
        inp: torch.Tensor, 
        v_patch_lens: Optional[Sequence[int]] = None
    ) -> List[torch.LongTensor]:
        """
        Encode input tokens to multi-scale codebook indices.
        Mirror: vqvae.py img_to_idxBl (lines 65-67)
        
        Args:
            inp: (B, L) input token indices
            v_patch_lens: Custom patch lengths
            
        Returns:
            List of (B, pl) index tensors for each scale
        """
        x = self.tok_embed(inp).transpose(1, 2)  # (B, embed_dim, L)
        f = self.quant_conv(self.encoder(x))
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_lens=v_patch_lens)
    
    # ===================== idxBl_to_logits: decode indices to logits =====================
    def idxBl_to_logits(
        self, 
        ms_idx_Bl: List[torch.Tensor], 
        same_shape: bool = True, 
        last_one: bool = False
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Decode multi-scale indices to token logits.
        Mirror: vqvae.py idxBl_to_img (lines 69-76)
        
        Args:
            ms_idx_Bl: List of index tensors for each scale
            same_shape: Whether to upsample all to same shape
            last_one: Whether to return only final result
            
        Returns:
            List of logits at each scale, or final logits if last_one=True
        """
        B = ms_idx_Bl[0].shape[0]
        ms_h_BCl = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            ms_h_BCl.append(self.quantize.embedding(idx_Bl).transpose(1, 2))  # (B, Cvae, l)
        return self.embed_to_logits(ms_h_BCl=ms_h_BCl, all_to_max_scale=same_shape, last_one=last_one)
    
    # ===================== embed_to_logits: embeddings to logits =====================
    def embed_to_logits(
        self, 
        ms_h_BCl: List[torch.Tensor], 
        all_to_max_scale: bool = True, 
        last_one: bool = False
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Convert multi-scale embeddings to logits.
        Mirror: vqvae.py embed_to_img (lines 78-82)
        
        Args:
            ms_h_BCl: List of embeddings at each scale
            all_to_max_scale: Whether to upsample all to max scale
            last_one: Whether to return only final result
            
        Returns:
            List of logits, or final logits if last_one=True
        """
        if last_one:
            f_hat = self.quantize.embed_to_fhat(ms_h_BCl, all_to_max_scale=all_to_max_scale, last_one=True)
            return self.decoder(self.post_quant_conv(f_hat))
        else:
            ls_f_hat = self.quantize.embed_to_fhat(ms_h_BCl, all_to_max_scale=all_to_max_scale, last_one=False)
            return [self.decoder(self.post_quant_conv(f_hat)) for f_hat in ls_f_hat]
    
    # ===================== inp_to_reconstructed: full reconstruction =====================
    def inp_to_reconstructed(
        self, 
        inp: torch.Tensor, 
        v_patch_lens: Optional[Sequence[int]] = None, 
        last_one: bool = False
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Reconstruct input through full encode-quantize-decode.
        Mirror: vqvae.py img_to_reconstructed_img (lines 84-90)
        
        Args:
            inp: (B, L) input token indices
            v_patch_lens: Custom patch lengths
            last_one: Whether to return only final result
            
        Returns:
            List of reconstructed logits at each scale, or final logits
        """
        x = self.tok_embed(inp).transpose(1, 2)
        f = self.quant_conv(self.encoder(x))
        ls_f_hat_BCL = self.quantize.f_to_idxBl_or_fhat(f, to_fhat=True, v_patch_lens=v_patch_lens)
        
        if last_one:
            return self.decoder(self.post_quant_conv(ls_f_hat_BCL[-1]))
        else:
            return [self.decoder(self.post_quant_conv(f_hat)) for f_hat in ls_f_hat_BCL]
    
    # ===================== compute_loss: for training =====================
    def compute_loss(
        self, 
        inp: torch.Tensor, 
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            inp: (B, L) input token indices
            labels: (B, L) target token indices (default: same as inp for autoencoding)
            attention_mask: (B, L) mask for valid positions
            
        Returns:
            Dict with 'loss', 'recon_loss', 'vq_loss', 'logits'
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
                logits_flat[mask_flat], 
                labels_flat[mask_flat],
                reduction='mean'
            )
        else:
            recon_loss = nn.functional.cross_entropy(logits_flat, labels_flat, reduction='mean')
        
        loss = recon_loss + vq_loss
        
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'logits': logits,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True, assign: bool = False):
        """Handle state dict loading with EMA buffer size mismatch."""
        if 'quantize.ema_vocab_hit_SV' in state_dict:
            if state_dict['quantize.ema_vocab_hit_SV'].shape[0] != self.quantize.ema_vocab_hit_SV.shape[0]:
                state_dict['quantize.ema_vocab_hit_SV'] = self.quantize.ema_vocab_hit_SV
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)
