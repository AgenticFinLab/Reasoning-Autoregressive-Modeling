"""Text Decoder using HuggingFace models.

Directly uses pretrained HuggingFace decoders (GPT2, etc.)
for decoding quantized features to token logits.

Input:  [B, L, D] hidden_states (from quantizer f_hat)
Output: [B, L, vocab_size] logits
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

__all__ = ["TextDecoder", "build_decoder"]


class TextDecoder(nn.Module):
    """Text Decoder using HuggingFace pretrained model.

    Args:
        model_name: HuggingFace model name (e.g., 'gpt2')
        pretrained: Whether to load pretrained weights
        freeze: Whether to freeze decoder weights
        input_dim: Input dimension (for projection to model dim)

    Supported models:
        - GPT2: gpt2, gpt2-medium, gpt2-large
        - OPT: facebook/opt-125m, facebook/opt-350m
        - LLaMA: meta-llama/Llama-2-7b (if available)
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        pretrained: bool = True,
        freeze: bool = False,
        input_dim: Optional[int] = None,
    ):
        super().__init__()
        self.model_name = model_name

        # Load HuggingFace model directly
        if pretrained:
            self.decoder = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.decoder = AutoModelForCausalLM.from_config(config)

        # Get dimensions from model config
        self.hidden_dim = self.decoder.config.hidden_size
        self.vocab_size = self.decoder.config.vocab_size

        # Input projection if dimensions don't match
        if input_dim is not None and input_dim != self.hidden_dim:
            self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        else:
            self.input_proj = None

        # Freeze if requested
        if freeze:
            for param in self.decoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode hidden states to logits.

        Args:
            hidden_states: [B, L, input_dim] from quantizer (f_hat)
            attention_mask: [B, L] attention mask (optional)

        Returns:
            [B, L, vocab_size] logits
        """
        # Project if needed: [B, L, hidden_dim]
        if self.input_proj is not None:
            hidden_states = self.input_proj(hidden_states)

        # Use inputs_embeds instead of input_ids
        outputs = self.decoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
        )

        return outputs.logits


def build_decoder(
    config: Dict[str, Any], input_dim: Optional[int] = None
) -> TextDecoder:
    """Build decoder from config dict.

    Config keys (all required):
        - model_name: HuggingFace model name (e.g., 'gpt2')
        - pretrained: bool
        - freeze: bool

    Args:
        config: Config dict
        input_dim: Input dimension from quantizer
    """
    return TextDecoder(
        model_name=config["model_name"],
        pretrained=config["pretrained"],
        freeze=config["freeze"],
        input_dim=input_dim,
    )
