"""Text VQ-VAE combining encoder, quantizer, decoder.

Architecture:
    [B, L] token_ids
    -> TextEncoder -> [B, L, D]
    -> Project -> [B, L, C]
    -> MultiScaleQuantizer -> [B, L, C] f_hat
    -> Project -> [B, L, D']
    -> TextDecoder -> [B, L, vocab] logits
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn

from .encoder import TextEncoder, build_encoder
from .decoder import TextDecoder, build_decoder
from .quantizer import MultiScaleQuantizer, build_quantizer

__all__ = ["TextVQVAE", "build_text_vqvae"]


class TextVQVAE(nn.Module):
    """Text VQ-VAE model.

    Args:
        encoder: TextEncoder (HuggingFace)
        decoder: TextDecoder (HuggingFace)
        quantizer: MultiScaleQuantizer
        latent_dim: Latent space dimension

    Input:  [B, L] token_ids
    Output: dict with logits, loss, f_hat
    """

    def __init__(
        self,
        encoder: TextEncoder,
        decoder: TextDecoder,
        quantizer: MultiScaleQuantizer,
        latent_dim: int = 256,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.latent_dim = latent_dim

        # Projections
        self.enc_to_latent = nn.Linear(encoder.output_dim, latent_dim)
        self.latent_to_dec = nn.Linear(latent_dim, decoder.hidden_dim)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode tokens to latent.

        Args:
            input_ids: [B, L]
            attention_mask: [B, L]

        Returns:
            [B, L, latent_dim]
        """
        hidden = self.encoder(input_ids, attention_mask)
        return self.enc_to_latent(hidden)

    def decode(
        self,
        f_hat: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode f_hat to logits.

        Args:
            f_hat: [B, L, latent_dim]
            attention_mask: [B, L]

        Returns:
            [B, L, vocab_size]
        """
        dec_input = self.latent_to_dec(f_hat)
        return self.decoder(dec_input, attention_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_indices: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward: encode -> quantize -> decode.

        Args:
            input_ids: [B, L]
            attention_mask: [B, L]
            return_indices: Whether to return codebook indices

        Returns:
            dict with:
                logits: [B, L, vocab_size]
                loss: scalar
                f_hat: [B, L, latent_dim]
                indices: List[Tensor] (if return_indices)
        """
        z = self.encode(input_ids, attention_mask)
        f_hat, quant_loss, indices = self.quantizer.quantize(z)
        logits = self.decode(f_hat, attention_mask)

        result = {"logits": logits, "loss": quant_loss, "f_hat": f_hat}
        if return_indices:
            result["indices"] = indices
        return result


def build_text_vqvae(config: Dict[str, Any]) -> TextVQVAE:
    """Build TextVQVAE from config.

    Config (all required):
        encoder:
            model_name: str
            pretrained: bool
            freeze: bool
        decoder:
            model_name: str
            pretrained: bool
            freeze: bool
        quantizer:
            codebook_size: int
            scale_lengths: list[int]
            beta: float
            quant_resi: float
            share_quant_resi: int
        latent_dim: int
    """
    latent_dim = config["latent_dim"]

    encoder = build_encoder(config["encoder"])

    quant_config = config["quantizer"].copy()
    quant_config["codebook_dim"] = latent_dim
    quantizer = build_quantizer(quant_config)

    decoder = build_decoder(config["decoder"], input_dim=latent_dim)

    return TextVQVAE(
        encoder=encoder,
        decoder=decoder,
        quantizer=quantizer,
        latent_dim=latent_dim,
    )
