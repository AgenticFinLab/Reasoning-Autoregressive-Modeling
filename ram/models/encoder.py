"""Text Encoder using HuggingFace models.

Directly uses pretrained HuggingFace encoders for encoding text sequences
into continuous representations.

Available Models:
    Encoder-only (bidirectional, recommended for encoding):
        - BERT: bert-base-uncased (768), bert-large-uncased (1024)
        - RoBERTa: roberta-base (768), roberta-large (1024)
        - ALBERT: albert-base-v2 (768), albert-large-v2 (1024)
        - DistilBERT: distilbert-base-uncased (768) - faster, smaller
        - DeBERTa: microsoft/deberta-v3-base (768) - better performance

    Encoder-Decoder (use encoder part):
        - T5: t5-small (512), t5-base (768), t5-large (1024)
        - BART: facebook/bart-base (768), facebook/bart-large (1024)

    Decoder-only (causal, use hidden states):
        - GPT2: gpt2 (768), gpt2-medium (1024), gpt2-large (1280)

Model Selection Principles:
    1. Task alignment: Use bidirectional encoders (BERT/RoBERTa) for understanding,
       causal decoders (GPT2) only if generation context is needed
    2. Dimension matching: Ensure hidden_dim aligns with decoder's input_dim,
       or use projection layer to adapt
    3. Efficiency: DistilBERT for speed, BERT-base for balance, large models
       for best quality
    4. Domain: RoBERTa for general NLP, DeBERTa for SOTA performance

Pipeline:
    1. Input: List[str] texts OR [B, L] input_ids
    2. Tokenize (if texts): texts -> input_ids [B, L], attention_mask [B, L]
    3. HuggingFace Encoder: input_ids [B, L] -> hidden [B, L, hidden_dim]
    4. Projection (optional): hidden [B, L, hidden_dim] -> output [B, L, output_dim]
    5. Output: [B, L, output_dim] continuous representations

    Flow Diagram:
    ┌─────────────────┐
    │ List[str] texts │
    └────────┬────────┘
             │ tokenize
             ▼
    ┌─────────────────┐
    │ [B, L] input_ids│
    └────────┬────────┘
             │ HuggingFace Encoder
             ▼
    ┌─────────────────────┐
    │ [B, L, hidden_dim]  │
    └────────┬────────────┘
             │ projection (optional)
             ▼
    ┌─────────────────────┐
    │ [B, L, output_dim]  │
    └─────────────────────┘

Dimensions:
    B: batch size
    L: sequence length (max_length after padding/truncation)
    hidden_dim: HuggingFace model's hidden size (e.g., BERT-base: 768)
    output_dim: final output dimension (= hidden_dim if no projection)

Example:
    encoder = build_encoder(config['model']['encoder'])
    # Input: 2 texts, max_length=128, BERT hidden=768
    output = encoder(inputs=["Hello world", "Test"])
    # Output: [2, 128, 768]
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
        config: Dict with required keys:
            - model_name: HuggingFace model name (e.g., 'bert-base-uncased')
            - pretrained: Whether to load pretrained weights
            - freeze: Whether to freeze encoder weights
            - output_dim: If set, project to this dimension (null for no projection)
            - max_length: Max sequence length for tokenization

    Supported models:
        - BERT: bert-base-uncased, bert-large-uncased
        - RoBERTa: roberta-base, roberta-large
        - GPT2: gpt2, gpt2-medium (uses hidden states)
        - T5: t5-small, t5-base (encoder only)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # HuggingFace model identifier (e.g., 'bert-base-uncased', 'roberta-base')
        model_name = config["model_name"]
        # Whether to load pretrained weights (True) or random init (False)
        pretrained = config["pretrained"]
        # Whether to freeze encoder weights during training
        freeze = config["freeze"]
        # Output dimension after projection (null = use model's hidden_dim)
        output_dim = config["output_dim"]
        # Max sequence length for tokenization (padding/truncation)
        max_length = config["max_length"]

        self.model_name = model_name
        self.max_length = max_length

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load HuggingFace model
        if pretrained:
            self.encoder = AutoModel.from_pretrained(model_name)
        else:
            hf_config = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_config(hf_config)

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
            inputs: List[str] raw text strings (primary input), len = B
            input_ids: [B, L] pre-tokenized token IDs (optional)
            attention_mask: [B, L] attention mask (optional)
            max_length: Override max_length for tokenization (optional)

        Returns:
            hidden: [B, L, output_dim] hidden states

        Dimensions:
            B = batch size (len(inputs))
            L = max_length (sequence length after padding/truncation)
            hidden_dim = model hidden size (e.g., BERT: 768)
            output_dim = final output (= hidden_dim if no proj, else config value)

        Flow:
            Step 1: inputs [B texts] -> tokenize -> input_ids [B, L], attention_mask [B, L]
            Step 2: input_ids [B, L] -> HuggingFace Encoder -> hidden [B, L, hidden_dim]
            Step 3: hidden [B, L, hidden_dim] -> projection (optional) -> output [B, L, output_dim]
        """
        # Step 1: Tokenize
        if inputs is not None:
            tokens = self.tokenize(inputs, max_length=max_length)
            input_ids = tokens["input_ids"].to(self.encoder.device)
            attention_mask = tokens["attention_mask"].to(self.encoder.device)
        elif input_ids is None:
            raise ValueError(
                "Either 'inputs' (List[str]) or 'input_ids' (Tensor) must be provided"
            )

        # Step 2: HuggingFace Encoder
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        # Step 3: Projection (optional)
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
    encoder = TextEncoder(config)

    # Logging
    freeze_str = "frozen" if config["freeze"] else "trainable"
    proj_str = " -> proj" if encoder.proj else ""
    logger.info(
        "[Encoder] %s (%s) - h(%d)%s -> o(%d)",
        encoder.model_name,
        freeze_str,
        encoder.hidden_dim,
        proj_str,
        encoder.output_dim,
    )

    return encoder
