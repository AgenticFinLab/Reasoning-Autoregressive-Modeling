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

================================================================================
C3 Cascade Encoder (Context Cascade Compression)
================================================================================

Paper: "Context Cascade Compression: Exploring the Upper Limits of Text Compression"
       (arXiv:2511.15244)

Core Idea:
    Cascade two LLMs of different sizes for text compression:
    - Small LLM (encoder): compresses long text into fixed-length latent tokens
    - Large LLM (decoder): reconstructs text from latent tokens

Key Innovation - Context Query Tokens:
    Learnable embeddings that extract compressed representation from input text.

    Input Sequence: [Context_Query_1, ..., Context_Query_N, Text_Token_1, ..., Text_Token_M]
                                                                          ↓
                                                                Encoder LLM (small)
                                                                          ↓
    Output: Extract hidden states of Context_Query tokens as latent representation
            latent_tokens: [B, N, D] where N = fixed latent length, D = hidden dim

Compression Ratio:
    - 20x compression (M/N = 20): 98% reconstruction accuracy
    - 40x compression: 93% accuracy

C3Encoder Pipeline:
    ┌──────────────────────────────────────────────────────────────────────┐
    │ Input: List[str] texts (long context, e.g., 1280 tokens)            │
    └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ tokenize
    ┌──────────────────────────────────────────────────────────────────────┐
    │ input_ids [B, M] where M = text_length                               │
    └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ prepend Context Query tokens
    ┌──────────────────────────────────────────────────────────────────────┐
    │ combined_ids [B, N+M] = [Context_Query, Text_Tokens]                 │
    │ combined_embeds [B, N+M, D] = [learnable_Q, text_embeddings]         │
    └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ Encoder LLM
    ┌──────────────────────────────────────────────────────────────────────┐
    │ hidden_states [B, N+M, D]                                            │
    └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ extract first N positions
    ┌──────────────────────────────────────────────────────────────────────┐
    │ latent_tokens [B, N, D] - compressed representation                  │
    └──────────────────────────────────────────────────────────────────────┘

Dimensions:
    B: batch size
    M: text sequence length (variable, can be very long)
    N: number of latent tokens (fixed, e.g., 32, 64)
    D: hidden dimension of encoder LLM (e.g., Qwen2.5-1.5B: 1536)

Example:
    encoder = C3Encoder(config)
    # Input: 1 text with 1280 tokens, N=32 latent tokens
    latent = encoder(texts=["Long text here..."])
    # Output: [1, 32, 1536] - compressed to 32 latent tokens
    # Compression ratio: 1280/32 = 40x
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

__all__ = ["TextEncoder", "build_encoder", "C3Encoder", "build_c3_encoder"]


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


# ============================================================================
# C3 Cascade Encoder (Context Cascade Compression)
# ============================================================================

# Special tokens for C3 (same as official implementation)
# Source: third-part/C3-Context-Cascade-Compression-main/C3-master/C3/model/C3.py Lines 21-23
C3_IM_START_TOKEN = "<img>"
C3_IM_END_TOKEN = "</img>"
C3_IM_PATCH_TOKEN = "<imgpad>"


class C3Encoder(nn.Module):
    """C3 Cascade Encoder for Context Compression.

    Paper: "Context Cascade Compression: Exploring the Upper Limits of Text Compression"
           (arXiv:2511.15244)

    Official Implementation: third-part/C3-Context-Cascade-Compression-main/C3-master/C3/model/C3.py

    Architecture (from official code):
        Uses a small LLM (e.g., Qwen2.5-1.5B) with learnable Context Query tokens
        to compress long text into fixed-length latent tokens.

    Key Components (official code references):
        1. Context Query Q: nn.Embedding(N, D) - Line 35
        2. Encoder LLM (llm1): Small pretrained LLM - Lines 109-114
        3. Special tokens: <img>, <imgpad>, </img> - Lines 21-23

    CRITICAL: Context Query Position
        The Context Query tokens are APPENDED AFTER the text, wrapped by special tokens:

        Input Sequence: [Text, <img>, Q_1, Q_2, ..., Q_N, </img>]
                                    ↑                ↑
                              im_start_token    im_end_token

        After encoder LLM, extract hidden states at Q positions:
        hidden_states[im_start_pos+1 : im_start_pos+N+1] -> latent_tokens [B, N, D]

    Forward Flow (matching official Lines 66-119):
        Step 1: texts -> tokenize -> context_ids [B, M]
        Step 2: context_ids -> context_embeds [B, M, D] via llm1.model.embed_tokens
        Step 3: Insert Q.weight between <img> and </img> tokens
        Step 4: new_context_embeds [B, M+N, D] -> llm1 -> hidden_states [B, M+N, D]
        Step 5: Extract Q positions -> latent_tokens [B, N, D]

    Args:
        config: Dict with keys:
            - model_name: HuggingFace model name (e.g., 'Qwen/Qwen2.5-1.5B')
            - pretrained: Whether to load pretrained weights
            - freeze: Whether to freeze LLM weights
            - latent_token_len: Number of latent tokens N (e.g., 32, 64)
            - max_length: Max text length M for tokenization

    Dimensions:
        B: batch size
        M: text sequence length (max_length)
        N: number of latent tokens (latent_token_len)
        D: hidden dimension of encoder LLM
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # ====================================================================
        # Configuration
        # ====================================================================
        model_name = config["model_name"]
        pretrained = config["pretrained"]
        freeze = config["freeze"]
        # Use official naming: latent_token_len (C3 config key)
        latent_token_len = config["latent_token_len"]
        max_length = config["max_length"]

        self.model_name = model_name
        # Official C3 naming
        self.latent_token_len = latent_token_len
        self.max_length = max_length

        # ====================================================================
        # Load Tokenizer
        # ====================================================================
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ====================================================================
        # Add Special Tokens for C3
        # Source: third-part/C3-Context-Cascade-Compression-main/C3-master/C3/model/C3.py
        #         Lines 21-23, 361-366
        # ====================================================================
        # Add special tokens to tokenizer
        special_tokens_dict = {
            "additional_special_tokens": [
                C3_IM_START_TOKEN,
                C3_IM_END_TOKEN,
                C3_IM_PATCH_TOKEN,
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)

        # Get special token IDs
        self.im_start_token_id = self.tokenizer.convert_tokens_to_ids(C3_IM_START_TOKEN)
        self.im_end_token_id = self.tokenizer.convert_tokens_to_ids(C3_IM_END_TOKEN)
        self.im_patch_token_id = self.tokenizer.convert_tokens_to_ids(C3_IM_PATCH_TOKEN)

        # ====================================================================
        # Load Encoder LLM (llm1 in official code)
        # Source: third-part/C3-Context-Cascade-Compression-main/C3-master/C3/model/C3.py
        #         Lines 328-333 (from_pretrained), 109-114 (forward)
        # ====================================================================
        if pretrained:
            self.llm = AutoModel.from_pretrained(model_name)
        else:
            hf_config = AutoConfig.from_pretrained(model_name)
            self.llm = AutoModel.from_config(hf_config)

        # Resize embeddings to accommodate new special tokens
        # Source: Lines 360-366
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # Get hidden dimension from model config
        self.hidden_dim = self.llm.config.hidden_size

        # ====================================================================
        # Context Query Tokens (Learnable Embedding)
        # Source: third-part/C3-Context-Cascade-Compression-main/C3-master/C3/model/C3.py
        #         Line 35: self.Q = nn.Embedding(config.latent_token_len, config.contexts_compression_llm_hidden_size)
        # ====================================================================
        # Official uses nn.Embedding, not nn.Parameter
        # Shape: [N, D] where N = latent_token_len, D = hidden_dim
        self.Q = nn.Embedding(latent_token_len, self.hidden_dim)
        # Initialize with small values for stable training
        nn.init.normal_(self.Q.weight, mean=0.0, std=0.02)

        # ====================================================================
        # Freeze LLM if requested
        # ====================================================================
        if freeze:
            for param in self.llm.parameters():
                param.requires_grad = False

    def tokenize(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text strings.

        Args:
            texts: List of text strings, len = B
            max_length: Override max_length (optional)

        Returns:
            Dict with:
                - input_ids: [B, M] token IDs
                - attention_mask: [B, M] attention mask
        """
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length or self.max_length,
        )

    def _prepare_context_with_query_tokens(
        self,
        context_ids: torch.Tensor,
        context_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare context embeddings with Context Query tokens inserted.

        Source: third-part/C3-Context-Cascade-Compression-main/C3-master/C3/model/C3.py
                Lines 73-108

        This function:
        1. Finds <img> (im_start_token) positions in context_ids
        2. Inserts Q.weight embeddings between <img> and </img>
        3. Returns new context_embeds with Q tokens inserted

        Args:
            context_ids: [B, M] token IDs
            context_embeds: [B, M, D] embeddings

        Returns:
            new_context_embeds: [B, M+N, D] embeddings with Q tokens inserted
            image_start_positions: [B] positions of <img> tokens
        """
        batch_size = context_ids.shape[0]
        N = self.latent_token_len
        device = context_embeds.device
        dtype = context_embeds.dtype

        # Source: Lines 73-76
        # context_features = []
        # for i in range(context_embeds.shape[0]):
        #     context_features.append([self.Q.weight])
        # Each batch item gets the same Q.weight
        # Shape: [N, D]
        query_embeds = self.Q.weight.to(device=device, dtype=dtype)

        # Source: Lines 80-102
        new_context_embeds = []
        image_start_positions = []

        for b in range(batch_size):
            # Shape: [M]
            cur_context_ids = context_ids[b]
            # Shape: [M, D]
            cur_context_embeds = context_embeds[b]

            # Source: Lines 85-86
            # Find position of <img> token
            image_start_tokens = torch.where(cur_context_ids == self.im_start_token_id)[
                0
            ]

            if len(image_start_tokens) == 0:
                # No <img> token found, append Q tokens at the end
                # This is a fallback for cases without special tokens
                new_embeds = torch.cat([cur_context_embeds, query_embeds], dim=0)
                image_start_pos = cur_context_ids.shape[0] - 1
                image_start_positions.append(image_start_pos)
            else:
                # Source: Lines 88-101
                image_start_pos = image_start_tokens[0].item()
                image_start_positions.append(image_start_pos)

                # Source: Lines 94-101
                # cur_context_embeds = torch.cat((
                #     cur_context_embeds[:image_start_token_pos+1],
                #     per_cur_image_features,  # Q.weight
                #     cur_context_embeds[image_start_token_pos + num_patches + 1:]
                # ), dim=0)
                #
                # Structure: [Text, <img>, Q_1, ..., Q_N, </img>]
                #            [:pos+1]   [Q]         [pos+N+1:]
                # Elements:
                #   cur_context_embeds[: image_start_pos + 1] - Text + <img>
                #   query_embeds - Q_1, ..., Q_N
                #   cur_context_embeds[image_start_pos + N + 1:] - </img> + remaining
                new_embeds = torch.cat(
                    [
                        cur_context_embeds[: image_start_pos + 1],
                        query_embeds,
                        cur_context_embeds[image_start_pos + N + 1 :],
                    ],
                    dim=0,
                )

            new_context_embeds.append(new_embeds)

        # Source: Line 106
        image_start_positions = torch.tensor(image_start_positions, device=device)

        # Source: Line 108
        new_context_embeds = torch.stack(new_context_embeds, dim=0)

        return new_context_embeds, image_start_positions

    def forward(
        self,
        inputs: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Compress text into latent tokens using Context Query mechanism.

        Source: third-part/C3-Context-Cascade-Compression-main/C3-master/C3/model/C3.py
                Lines 66-119 (encoder section)

        Args:
            inputs: List[str] raw text strings (primary input), len = B
            input_ids: [B, M] pre-tokenized token IDs (optional)
            attention_mask: [B, M] attention mask (optional)
            max_length: Override max_length for tokenization (optional)

        Returns:
            latent_tokens: [B, N, D] compressed latent representation
                - N = num_latent_tokens (fixed)
                - D = hidden_dim

        Dimensions Flow (matching official Lines 66-119):
            Step 1 (Line 64): context_ids -> context_embeds via llm1.model.embed_tokens
            Step 2 (Lines 73-108): Insert Q.weight between <img> and </img>
            Step 3 (Lines 109-114): new_context_embeds -> llm1 -> hidden_states
            Step 4 (Lines 115-119): Extract Q positions -> latent_tokens
        """
        # Get device from LLM
        device = self.llm.device

        # ====================================================================
        # Step 1: Tokenize (if raw text input)
        # ====================================================================
        if inputs is not None:
            # Append special tokens to input text
            # Source: Line 376
            # context = context + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*N + DEFAULT_IM_END_TOKEN
            texts_with_tokens = [
                text
                + C3_IM_START_TOKEN
                + C3_IM_PATCH_TOKEN * self.latent_token_len
                + C3_IM_END_TOKEN
                for text in inputs
            ]
            tokens = self.tokenize(texts_with_tokens, max_length=max_length)
            context_ids = tokens["input_ids"].to(device)
            context_attention_mask = tokens["attention_mask"].to(device)
        elif input_ids is not None:
            context_ids = input_ids.to(device)
            context_attention_mask = (
                attention_mask.to(device) if attention_mask is not None else None
            )
        else:
            raise ValueError(
                "Either 'inputs' (List[str]) or 'input_ids' (Tensor) must be provided"
            )

        batch_size = context_ids.shape[0]
        N = self.latent_token_len
        D = self.hidden_dim

        # ====================================================================
        # Step 2: Get context embeddings from LLM's embedding layer
        # Source: Line 64
        # context_embeds = self.llm1.model.embed_tokens(context_ids)
        # ====================================================================
        context_embeds = self.llm.get_input_embeddings()(context_ids)

        # ====================================================================
        # Step 3: Insert Context Query tokens between <img> and </img>
        # Source: Lines 73-108
        # ====================================================================
        new_context_embeds, image_start_positions = (
            self._prepare_context_with_query_tokens(context_ids, context_embeds)
        )

        # Update attention mask to include Q tokens
        # Q tokens always attend (mask=1)
        if context_attention_mask is not None:
            query_mask = torch.ones(
                batch_size, N, device=device, dtype=context_attention_mask.dtype
            )
            new_attention_mask = torch.cat([context_attention_mask, query_mask], dim=1)
        else:
            new_attention_mask = None

        # ====================================================================
        # Step 4: Forward through Encoder LLM (llm1)
        # Source: Lines 109-114
        # llm1_hidden_states = self.llm1.forward(
        #     input_ids=None, attention_mask=context_attention_mask,
        #     inputs_embeds=context_embeds, ...
        # )['hidden_states'][-1]
        # ====================================================================
        outputs = self.llm(
            inputs_embeds=new_context_embeds,
            attention_mask=new_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # Source: Line 114 - use last hidden state
        # Shape: [B, M+N, D]
        hidden_states = outputs.last_hidden_state

        # ====================================================================
        # Step 5: Extract latent tokens at Q positions
        # Source: Lines 115-119
        # latent_contexts = []
        # for i, llm1_hidden_state in enumerate(llm1_hidden_states):
        #     image_start_token_pos = image_start_tokens_list[i]
        #     llm1_hidden_state = llm1_hidden_state[image_start_token_pos+1:image_start_token_pos + num_patches+1]
        #     latent_contexts.append(llm1_hidden_state)
        # ====================================================================
        latent_tokens = []
        for b in range(batch_size):
            if b < len(image_start_positions):
                image_start_pos = image_start_positions[b].item()
            else:
                # Fallback: extract last N tokens
                image_start_pos = hidden_states.shape[1] - N - 1

            # Source: Line 118
            # Extract [image_start_pos+1 : image_start_pos+N+1]
            # This extracts the Q token positions
            latent = hidden_states[b, image_start_pos + 1 : image_start_pos + N + 1, :]
            latent_tokens.append(latent)

        # Stack to [B, N, D]
        latent_tokens = torch.stack(latent_tokens, dim=0)

        return latent_tokens

    def get_compression_ratio(self, text_length: int) -> float:
        """Calculate compression ratio for given text length.

        Args:
            text_length: Number of text tokens M

        Returns:
            Compression ratio M/N
        """
        return text_length / self.latent_token_len


def build_c3_encoder(config: Dict[str, Any]) -> C3Encoder:
    """Build C3 Cascade Encoder from config dict.

    Config keys (all required):
        - model_name: str - HuggingFace model name (e.g., 'Qwen/Qwen2.5-1.5B')
        - pretrained: bool - Whether to load pretrained weights
        - freeze: bool - Whether to freeze LLM weights
        - latent_token_len: int - Number of latent tokens N (e.g., 32, 64)
        - max_length: int - Max text length M for tokenization

    Example Config:
        config = {
            "model_name": "Qwen/Qwen2.5-1.5B",
            "pretrained": True,
            "freeze": False,
            "latent_token_len": 32,
            "max_length": 1280,  # 40x compression ratio
        }
        encoder = build_c3_encoder(config)
    """
    encoder = C3Encoder(config)

    # Logging
    freeze_str = "frozen" if config["freeze"] else "trainable"
    latent_len = config["latent_token_len"]
    compression_ratio = config["max_length"] / latent_len
    logger.info(
        "[C3Encoder] %s (%s) - text(%d) -> latent(%d) = %.1fx compression, hidden(%d)",
        encoder.model_name,
        freeze_str,
        config["max_length"],
        latent_len,
        compression_ratio,
        encoder.hidden_dim,
    )

    return encoder
