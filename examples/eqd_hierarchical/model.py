"""EQD Hierarchical (Encoder-Quantizer-Decoder with Hierarchical Quantization).

Unified model combining encoder, hierarchical quantizer, and decoder.
Uses text-native hierarchical scales: sentence → clause → phrase → word → subword.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple

from ram.models.encoder import build_encoder
from ram.models.decoder import build_decoder


class HierarchicalPhi(nn.Module):
    """Phi transformation for hierarchical quantization."""

    def __init__(self, dim: int, residual_ratio: float = 0.5):
        super().__init__()
        self.conv = nn.Linear(dim, dim)
        self.residual_ratio = residual_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply phi transformation.

        Args:
            x: [B, scale_len, D] codebook output

        Returns:
            h: [B, scale_len, D] transformed output
        """
        transformed = self.conv(x)
        return self.residual_ratio * transformed + (1 - self.residual_ratio) * x


class HierarchicalTextQuantizer(nn.Module):
    """Hierarchical Text Quantizer with text-native scales.

    Scales:
        - Scale 0: sentence-level (1 token)
        - Scale 1: clause-level (4 tokens)
        - Scale 2: phrase-level (16 tokens)
        - Scale 3: word-level (64 tokens)
        - Scale 4: subword-level (256 tokens)
    """

    def __init__(
        self,
        codebook_size: int = 4096,
        codebook_dim: int = 256,
        scale_lengths: tuple = (1, 4, 16, 64, 256),
        beta: float = 0.25,
        quant_resi: float = 0.5,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.scale_lengths = scale_lengths
        self.beta = beta

        # Shared codebook
        self.embedding = nn.Embedding(codebook_size, codebook_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

        # Phi transformations (partially shared)
        num_phi = min(4, len(scale_lengths))
        self.phi_layers = nn.ModuleList(
            [HierarchicalPhi(codebook_dim, quant_resi) for _ in range(num_phi)]
        )
        self.scale_to_phi = [min(i, num_phi - 1) for i in range(len(scale_lengths))]

    def forward(self, z: torch.Tensor):
        """Hierarchical multi-scale quantization.

        Dimension Flow (for L=256, scale_lengths=[1,4,16,64,256]):
            Input: z [B, L, D] (e.g., [4, 256, 256])
                ↓
            For each scale (scale_len in [1, 4, 16, 64, 256]):
                - Pool: f_rest [B, L, D] -> rest_down [B, scale_len, D]
                - VQ: rest_down -> indices [B, scale_len], q [B, scale_len, D]
                - Phi: q -> h [B, scale_len, D]
                - Upsample: h -> h_up [B, L, D]
                - Accumulate: f_hat += h_up, f_rest -= h_up
                ↓
            Output: f_hat [B, L, D], indices_per_scale, vq_loss

        Args:
            z: [B, L, D] encoder output (continuous features)

        Returns:
            f_hat: [B, L, D] accumulated quantized features (sum of all scales)
            indices_per_scale: List of [B, scale_k] codebook indices for each scale
            vq_loss: scalar VQ commitment loss (encourages encoder to commit to codes)
        """
        B, L, D = z.shape
        f_rest = z  # Residual starts as full encoder output
        f_hat = torch.zeros_like(z)
        indices_per_scale = []
        vq_loss = 0.0

        for scale_idx, scale_len in enumerate(self.scale_lengths):
            if scale_len > L:
                continue

            # Pool to scale length
            pool_size = L // scale_len
            rest_down = F.avg_pool1d(f_rest.transpose(1, 2), pool_size).transpose(1, 2)

            # VQ
            distances = torch.cdist(rest_down, self.embedding.weight)
            indices = torch.argmin(distances, dim=-1)
            q = self.embedding(indices)

            # Phi transformation
            phi_idx = self.scale_to_phi[scale_idx]
            h = self.phi_layers[phi_idx](q)

            # Upsample and accumulate
            h_up = h.repeat_interleave(pool_size, dim=1)[:, :L, :]
            f_hat = f_hat + h_up
            f_rest = f_rest - h_up

            # VQ loss
            vq_loss = (
                vq_loss
                + F.mse_loss(q.detach(), rest_down)
                + self.beta * F.mse_loss(q, rest_down.detach())
            )

            indices_per_scale.append(indices)

        return f_hat, indices_per_scale, vq_loss


class EQDHierarchicalModel(nn.Module):
    """Unified EQD Hierarchical Model."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        enc_cfg = config["encoder"]
        dec_cfg = config["decoder"]
        quant_cfg = config.get("quantizer", {})

        # Build encoder
        self.encoder = build_encoder(enc_cfg)
        encoder_hidden_dim = self.encoder.output_dim

        # Build quantizer
        self.quantizer = HierarchicalTextQuantizer(
            codebook_size=quant_cfg.get("codebook_size", 4096),
            codebook_dim=quant_cfg.get("codebook_dim", encoder_hidden_dim),
            scale_lengths=tuple(quant_cfg.get("scale_lengths", [1, 4, 16, 64, 256])),
            beta=quant_cfg.get("beta", 0.25),
            quant_resi=quant_cfg.get("quant_resi", 0.5),
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
        """Forward pass.

        Dimension Flow:
            texts: List[str] with B texts
                ↓
            z: [B, L, D] encoder output (continuous features)
                ↓
            f_hat: [B, L, D] quantized features (discrete codes via codebook)
            indices_per_scale: List of [B, scale_k] codebook indices per scale
            vq_loss: scalar VQ commitment loss
                ↓
            logits: [B, L, V] decoder output (vocabulary logits)

        Returns:
            logits: [B, L, V] vocabulary logits for each position
            loss: total loss = recon_loss + vq_loss_weight * vq_loss (if compute_loss=True)
            vq_loss: scalar VQ commitment loss (if compute_loss=True)
        """
        # Encode: texts -> z [B, L, D]
        z = self.encoder(texts)

        # Quantize
        f_hat, indices_per_scale, vq_loss = self.quantizer(z)

        # Decode
        logits = self.decoder(f_hat)

        # Compute loss
        loss = None
        if compute_loss:
            # Reconstruction loss
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
