"""Text Decoder using HuggingFace models.

Directly uses pretrained HuggingFace decoders for decoding continuous
representations back to token logits.

Available Models:
    Decoder-only (causal LM, recommended for generation):
        - GPT2: gpt2 (768, 50257), gpt2-medium (1024), gpt2-large (1280)
        - GPT-Neo: EleutherAI/gpt-neo-125m (768), gpt-neo-1.3B (2048)
        - OPT: facebook/opt-125m (768), opt-350m (512), opt-1.3b (2048)
        - Pythia: EleutherAI/pythia-70m (512), pythia-160m (768)
        - LLaMA: meta-llama/Llama-2-7b (4096) - requires access
        - Mistral: mistralai/Mistral-7B-v0.1 (4096) - requires access

    Encoder-Decoder (use decoder part with cross-attention):
        - T5: t5-small (512), t5-base (768), t5-large (1024)
        - BART: facebook/bart-base (768), facebook/bart-large (1024)

Model Selection Principles:
    1. Dimension alignment: Match input_dim with model's hidden_dim,
       or use input_proj to adapt (e.g., BERT 768 -> GPT2 768: no proj needed)
    2. Vocabulary: GPT2 (50257) vs OPT (50272) vs LLaMA (32000) - affects output
    3. Efficiency: GPT2/OPT-125m for fast iteration, larger for quality
    4. Generation quality: Larger models (GPT-Neo, LLaMA) for better fluency
    5. Memory: Small models (<500M params) for local development

Recommended Combinations:
    - BERT (768) + GPT2 (768): No projection needed, balanced
    - RoBERTa (768) + GPT2 (768): Strong encoder + standard decoder
    - BERT (768) + OPT-125m (768): Fast, memory-efficient

Pipeline:
    1. Input: [B, L, input_dim] hidden states (from encoder or quantizer f_hat)
    2. Projection (optional): hidden [B, L, input_dim] -> [B, L, hidden_dim]
    3. HuggingFace Decoder: hidden [B, L, hidden_dim] -> logits [B, L, vocab_size]
    4. Output: [B, L, vocab_size] token logits

    Flow Diagram:
    ┌─────────────────────┐
    │ [B, L, input_dim]   │  (from encoder/quantizer)
    └────────┬────────────┘
             │ projection (optional)
             ▼
    ┌─────────────────────┐
    │ [B, L, hidden_dim]  │
    └────────┬────────────┘
             │ HuggingFace Decoder
             ▼
    ┌─────────────────────┐
    │ [B, L, vocab_size]  │
    └────────┬────────────┘
             │ argmax(dim=-1)
             ▼
    ┌─────────────────────┐
    │ [B, L] token_ids    │
    └────────┬────────────┘
             │ tokenizer.decode()
             ▼
    ┌─────────────────────┐
    │ List[str] texts     │
    └─────────────────────┘

Dimensions:
    B: batch size
    L: sequence length
    input_dim: input dimension from encoder/quantizer
    hidden_dim: HuggingFace decoder's hidden size (e.g., GPT2: 768)
    vocab_size: vocabulary size (e.g., GPT2: 50257)

Example:
    decoder = build_decoder(config['model']['decoder'], input_dim=768)
    # Input: [2, 128, 768] from encoder
    logits = decoder(hidden_states)
    # Output: [2, 128, 50257]
"""

from typing import Optional, Dict, Any
import logging
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

logger = logging.getLogger(__name__)

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
            hidden_states: [B, L, input_dim] from encoder or quantizer (f_hat)
            attention_mask: [B, L] attention mask (optional)

        Returns:
            logits: [B, L, vocab_size] token logits

        Dimensions:
            B = batch size
            L = sequence length
            input_dim = input dimension from encoder/quantizer
            hidden_dim = decoder hidden size (e.g., GPT2: 768)
            vocab_size = vocabulary size (e.g., GPT2: 50257)

        Flow:
            Step 1: hidden_states [B, L, input_dim] -> projection (optional) -> [B, L, hidden_dim]
            Step 2: [B, L, hidden_dim] -> HuggingFace Decoder -> logits [B, L, vocab_size]

        Restoration (after this forward):
            logits [B, L, vocab_size] -> argmax(dim=-1) -> token_ids [B, L]
            token_ids [B, L] -> tokenizer.decode() -> List[str] texts
        """
        # Step 1: Input Projection (optional)
        if self.input_proj is not None:
            hidden_states = self.input_proj(hidden_states)

        # Step 2: HuggingFace Decoder
        outputs = self.decoder(
            inputs_embeds=hidden_states, attention_mask=attention_mask
        )

        return outputs.logits


def build_decoder(config: Dict[str, Any], input_dim: int) -> TextDecoder:
    """Build decoder from config dict.

    Config keys (all required):
        - model_name: str - HuggingFace model name (e.g., 'gpt2')
        - pretrained: bool
        - freeze: bool

    Args:
        config: Config dict
        input_dim: Input dimension from encoder/quantizer (required)
    """
    decoder = TextDecoder(
        model_name=config["model_name"],
        pretrained=config["pretrained"],
        freeze=config["freeze"],
        input_dim=input_dim,
    )

    # Logging
    freeze_str = "frozen" if config["freeze"] else "trainable"
    proj_str = " -> proj" if decoder.input_proj else ""
    logger.info(
        "[Decoder] %s (%s) - i(%d)%s -> h(%d) -> v(%d)",
        decoder.model_name,
        freeze_str,
        input_dim,
        proj_str,
        decoder.hidden_dim,
        decoder.vocab_size,
    )

    return decoder
