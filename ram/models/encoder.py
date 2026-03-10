"""Text Encoder using HuggingFace models.

Directly uses pretrained HuggingFace encoders (BERT, RoBERTa, etc.)
for encoding text sequences.

Input:  List[str] texts OR [B, L] input_ids
Output: [B, L, D] hidden_states
"""

from typing import Optional, Dict, Any, List, Union
import logging
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer

logger = logging.getLogger(__name__)

__all__ = ["TextEncoder", "build_encoder"]


class TextEncoder(nn.Module):
    """Text Encoder using HuggingFace pretrained model.

    Args:
        model_name: HuggingFace model name (e.g., 'bert-base-uncased')
        pretrained: Whether to load pretrained weights
        freeze: Whether to freeze encoder weights
        output_dim: If set, project to this dimension
        max_length: Max sequence length for tokenization

    Supported models:
        - BERT: bert-base-uncased, bert-large-uncased
        - RoBERTa: roberta-base, roberta-large
        - GPT2: gpt2, gpt2-medium (uses hidden states)
        - T5: t5-small, t5-base (encoder only)
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        pretrained: bool = True,
        freeze: bool = False,
        output_dim: Optional[int] = None,
        max_length: int = 512,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load HuggingFace model
        if pretrained:
            self.encoder = AutoModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_config(config)

        # Get hidden dimension from model config
        self.hidden_dim = self.encoder.config.hidden_size

        # Optional projection layer
        if output_dim is not None and output_dim != self.hidden_dim:
            self.proj = nn.Linear(self.hidden_dim, output_dim)
            self.output_dim = output_dim
        else:
            self.proj = None
            self.output_dim = self.hidden_dim

        # Freeze if requested
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def tokenize(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize text strings.

        Args:
            texts: List of text strings
            max_length: Override max_length (optional)

        Returns:
            Dict with input_ids: [B, L], attention_mask: [B, L]
        """
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length or self.max_length,
        )

    def forward(
        self,
        inputs: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Encode text to hidden states.

        Args:
            inputs: List[str] raw text strings (primary input)
            input_ids: [B, L] pre-tokenized token IDs (optional, if inputs not provided)
            attention_mask: [B, L] attention mask (optional, auto-generated if inputs provided)
            max_length: Override max_length for tokenization (optional)

        Returns:
            hidden: [B, L, output_dim] hidden states
        """
        # Tokenize if raw text provided
        if inputs is not None:
            tokens = self.tokenize(inputs, max_length=max_length)
            input_ids = tokens["input_ids"].to(self.encoder.device)
            attention_mask = tokens["attention_mask"].to(self.encoder.device)
        elif input_ids is None:
            raise ValueError(
                "Either 'inputs' (List[str]) or 'input_ids' (Tensor) must be provided"
            )

        # Encode: [B, L] -> [B, L, hidden_dim]
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state

        # Project if needed: [B, L, hidden_dim] -> [B, L, output_dim]
        if self.proj is not None:
            hidden = self.proj(hidden)

        return hidden


def build_encoder(config: Dict[str, Any]) -> TextEncoder:
    """Build encoder from config dict.

    Config keys (all required):
        - model_name: str - HuggingFace model name
        - pretrained: bool
        - freeze: bool
        - output_dim: int or null
        - max_length: int
    """
    encoder = TextEncoder(
        model_name=config["model_name"],
        pretrained=config["pretrained"],
        freeze=config["freeze"],
        output_dim=config["output_dim"],
        max_length=config["max_length"],
    )

    # Logging
    freeze_str = "frozen" if config["freeze"] else "trainable"
    if encoder.proj is not None:
        logger.info(f"[Encoder] {encoder.model_name} ({freeze_str}) - h({encoder.hidden_dim}) -> proj -> o({encoder.output_dim})")
    else:
        logger.info(f"[Encoder] {encoder.model_name} ({freeze_str}) - h({encoder.hidden_dim}) -> o({encoder.output_dim})")

    return encoder
