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

================================================================================
C3 Cascade Decoder (Context Cascade Compression)
================================================================================

Paper: "Context Cascade Compression: Exploring the Upper Limits of Text Compression"
       (arXiv:2511.15244)

Core Idea:
    The decoder LLM (larger than encoder) takes compressed latent tokens
    and reconstructs the original text.

    Key Constraint:
    - MUST be paired with C3Encoder (cascade architecture)
    - Encoder and Decoder must share compatible hidden_dim
    - Decoder typically uses larger LLM than encoder (e.g., 3B vs 1.5B)

C3Decoder Pipeline:
    ┌──────────────────────────────────────────────────────────────────────┐
    │ Input: latent_tokens [B, N, D] from C3Encoder                        │
    └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ prepend BOS token (optional)
    ┌──────────────────────────────────────────────────────────────────────┐
    │ decoder_input [B, N+1, D] = [latent_tokens, bos_embed]               │
    └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ Decoder LLM (large)
    ┌──────────────────────────────────────────────────────────────────────┐
    │ hidden_states [B, N+1, D]                                            │
    └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ lm_head
    ┌──────────────────────────────────────────────────────────────────────┐
    │ logits [B, N+1, vocab_size]                                          │
    └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ generate() autoregressive decoding
    ┌──────────────────────────────────────────────────────────────────────┐
    │ output_ids [B, output_length]                                        │
    └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ tokenizer.decode()
    ┌──────────────────────────────────────────────────────────────────────┐
    │ List[str] reconstructed_texts                                        │
    └──────────────────────────────────────────────────────────────────────┘

Cascade Constraint:
    C3Decoder MUST be used with C3Encoder:
    - C3Encoder: Small LLM (e.g., Qwen2.5-1.5B, hidden_dim=1536)
    - C3Decoder: Large LLM (e.g., Qwen2.5-3B, hidden_dim=2048)
    - If hidden_dim differs, projection layer is added automatically

Dimensions:
    B: batch size
    N: number of latent tokens (from encoder, e.g., 32)
    D: hidden dimension (must match encoder's hidden_dim or use projection)
    vocab_size: vocabulary size of decoder LLM
"""

from typing import Optional, Dict, Any, List, Tuple
import logging
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

logger = logging.getLogger(__name__)

__all__ = ["TextDecoder", "build_decoder", "C3Decoder", "build_c3_decoder"]


class TextDecoder(nn.Module):
    """Text Decoder using HuggingFace pretrained model.

    Args:
        config: Dict with required keys:
            - model_name: HuggingFace model name (e.g., 'gpt2')
            - pretrained: Whether to load pretrained weights
            - freeze: Whether to freeze decoder weights
        input_dim: Input dimension from encoder/quantizer (required)

    Supported models:
        - GPT2: gpt2, gpt2-medium, gpt2-large
        - OPT: facebook/opt-125m, facebook/opt-350m
        - LLaMA: meta-llama/Llama-2-7b (if available)
    """

    def __init__(self, config: Dict[str, Any], input_dim: int):
        super().__init__()

        # HuggingFace model identifier (e.g., 'gpt2', 'facebook/opt-125m')
        model_name = config["model_name"]
        # Whether to load pretrained weights (True) or random init (False)
        pretrained = config["pretrained"]
        # Whether to freeze decoder weights during training
        freeze = config["freeze"]

        self.model_name = model_name

        # Load HuggingFace model directly
        if pretrained:
            self.decoder = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            hf_config = AutoConfig.from_pretrained(model_name)
            self.decoder = AutoModelForCausalLM.from_config(hf_config)

        # Get dimensions from model config
        self.hidden_dim = self.decoder.config.hidden_size
        self.vocab_size = self.decoder.config.vocab_size

        # Input projection if dimensions don't match
        if input_dim != self.hidden_dim:
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
    decoder = TextDecoder(config, input_dim)

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


# ============================================================================
# C3 Cascade Decoder (Context Cascade Compression)
# ============================================================================

# Special tokens for C3 (same as official implementation)
# Source: third-part/C3-Context-Cascade-Compression-main/C3-master/C3/model/C3.py Lines 21-23
C3_IM_START_TOKEN = "<img>"
C3_IM_END_TOKEN = "</img>"
C3_IM_PATCH_TOKEN = "<imgpad>"


class C3Decoder(nn.Module):
    """C3 Cascade Decoder for Text Reconstruction.

    Paper: "Context Cascade Compression: Exploring the Upper Limits of Text Compression"
           (arXiv:2511.15244)

    Official Implementation: third-part/C3-Context-Cascade-Compression-main/C3-master/C3/model/C3.py

    Architecture (from official code):
        Uses a large LLM (e.g., Qwen2.5-3B) to decode compressed latent tokens
        back to original text.

    Key Components (official code references):
        1. mm_projector: nn.Linear(encoder_hidden, decoder_hidden) - Line 36
        2. Decoder LLM: Large pretrained LLM - Lines 155-160
        3. Special tokens: <img>, <imgpad>, </img> - Lines 21-23

    CASCADE CONSTRAINT:
        This decoder MUST be paired with C3Encoder:
        - C3Encoder: Small LLM compresses text -> latent tokens [B, N, D_encoder]
        - C3Decoder: Large LLM reconstructs text <- latent tokens

    CRITICAL: Latent Token Position in Decoder
        The projected latent tokens are inserted at the BEGINNING, wrapped by special tokens:

        Decoder Input: [<img>, latent_1, ..., latent_N, </img>, prompt]
                              ↑                    ↑
                        im_start_token      im_end_token

        Source: Lines 372-373
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*N + DEFAULT_IM_END_TOKEN + '\n' + qs

    Forward Flow (matching official Lines 121-153):
        Step 1: latent_tokens [B, N, D_encoder] -> mm_projector -> [B, N, D_decoder]
        Step 2: Create decoder input with special tokens
        Step 3: Insert projected latent tokens between <img> and </img>
        Step 4: new_input_embeds [B, N+prompt_len, D_decoder] -> Decoder LLM -> logits

    Args:
        config: Dict with keys:
            - model_name: HuggingFace model name (e.g., 'Qwen/Qwen2.5-3B')
            - pretrained: Whether to load pretrained weights
            - freeze: Whether to freeze LLM weights
            - num_latent_tokens: Number of latent tokens N (must match encoder)
        encoder_hidden_dim: Hidden dimension from C3Encoder (required)

    Dimensions:
        B: batch size
        N: number of latent tokens (from encoder)
        D_encoder: encoder hidden dimension (encoder_hidden_dim)
        D_decoder: decoder hidden dimension (hidden_dim)
        vocab_size: vocabulary size of decoder LLM
    """

    # Valid encoder types for cascade constraint
    VALID_ENCODER_TYPES = ("C3Encoder",)

    def __init__(self, config: Dict[str, Any], encoder_hidden_dim: int):
        super().__init__()

        # ====================================================================
        # Validate cascade constraint
        # ====================================================================
        if encoder_hidden_dim is None:
            raise ValueError(
                "C3Decoder requires encoder_hidden_dim from C3Encoder. "
                "This is a cascade constraint - C3Decoder must be paired with C3Encoder."
            )

        # ====================================================================
        # Configuration
        # ====================================================================
        model_name = config["model_name"]
        pretrained = config["pretrained"]
        freeze = config["freeze"]
        num_latent_tokens = config["num_latent_tokens"]

        self.model_name = model_name
        self.encoder_hidden_dim = encoder_hidden_dim
        self.num_latent_tokens = num_latent_tokens

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
        # Load Decoder LLM (Causal LM for generation)
        # Source: third-part/C3-Context-Cascade-Compression-main/C3-master/C3/model/C3.py
        #         Lines 164-176
        # ====================================================================
        if pretrained:
            self.llm = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            hf_config = AutoConfig.from_pretrained(model_name)
            self.llm = AutoModelForCausalLM.from_config(hf_config)

        # Resize embeddings to accommodate new special tokens
        # Source: Lines 360-366
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # Get dimensions from model config
        self.hidden_dim = self.llm.config.hidden_size
        self.vocab_size = self.llm.config.vocab_size

        # ====================================================================
        # mm_projector: Projects encoder hidden_dim to decoder hidden_dim
        # Source: third-part/C3-Context-Cascade-Compression-main/C3-master/C3/model/C3.py
        #         Line 36: self.mm_projector = nn.Linear(config.contexts_compression_llm_hidden_size, config.hidden_size)
        # ====================================================================
        # This is the key component that bridges encoder and decoder
        # C3Encoder (e.g., Qwen2.5-1.5B, hidden=1536) -> C3Decoder (e.g., Qwen2.5-3B, hidden=2048)
        self.mm_projector = nn.Linear(encoder_hidden_dim, self.hidden_dim)

        # ====================================================================
        # Freeze LLM if requested
        # ====================================================================
        if freeze:
            for param in self.llm.parameters():
                param.requires_grad = False

    @classmethod
    def validate_encoder(cls, encoder_type: str) -> bool:
        """Validate that encoder is compatible with C3Decoder.

        CASCADE CONSTRAINT CHECK:
            C3Decoder must be paired with C3Encoder.

        Args:
            encoder_type: Class name of the encoder (e.g., 'C3Encoder')

        Returns:
            True if encoder is valid for cascade

        Raises:
            ValueError: If encoder type is not valid for C3 cascade
        """
        if encoder_type not in cls.VALID_ENCODER_TYPES:
            raise ValueError(
                f"C3Decoder cascade constraint violated: "
                f"Expected encoder type in {cls.VALID_ENCODER_TYPES}, "
                f"got '{encoder_type}'. "
                f"C3Decoder must be paired with C3Encoder."
            )
        return True

    def _prepare_decoder_input_with_latent_tokens(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        latent_features: torch.Tensor,
    ) -> torch.Tensor:
        """Prepare decoder input embeddings with latent tokens inserted.

        Source: third-part/C3-Context-Cascade-Compression-main/C3-master/C3/model/C3.py
                Lines 129-153

        This function:
        1. Finds <img> (im_start_token) positions in input_ids
        2. Inserts projected latent tokens between <img> and </img>
        3. Returns new input_embeds with latent tokens inserted

        Args:
            input_ids: [B, L] token IDs (prompt with special tokens)
            input_embeds: [B, L, D_decoder] embeddings
            latent_features: [B, N, D_decoder] projected latent tokens

        Returns:
            new_input_embeds: [B, L+N, D_decoder] embeddings with latent tokens inserted
        """
        batch_size = input_ids.shape[0]
        N = self.num_latent_tokens
        device = input_embeds.device
        dtype = input_embeds.dtype

        # Source: Lines 129-153
        new_input_embeds = []

        for b in range(batch_size):
            cur_input_ids = input_ids[b]  # [L]
            cur_input_embeds = input_embeds[b]  # [L, D_decoder]
            cur_latent_features = latent_features[b]  # [N, D_decoder]

            # Source: Lines 135
            # Find position of <img> token
            image_start_tokens = torch.where(cur_input_ids == self.im_start_token_id)[0]

            if len(image_start_tokens) == 0:
                # No <img> token found, prepend latent tokens
                new_embeds = torch.cat([cur_latent_features, cur_input_embeds], dim=0)
            else:
                # Source: Lines 136-148
                image_start_pos = image_start_tokens[0].item()

                # Source: Lines 141-148
                # cur_input_embeds = torch.cat((
                #     cur_input_embeds[:image_start_token_pos+1],
                #     per_cur_latent_features,
                #     cur_input_embeds[image_start_token_pos + num_patches + 1:]
                # ), dim=0)
                #
                # Structure: [<img>, latent_1, ..., latent_N, </img>, prompt]
                #            [:pos+1]  [latent]          [pos+N+1:]
                new_embeds = torch.cat(
                    [
                        cur_input_embeds[: image_start_pos + 1],  # <img>
                        cur_latent_features,  # latent_1, ..., latent_N
                        cur_input_embeds[image_start_pos + N + 1 :],  # </img> + prompt
                    ],
                    dim=0,
                )

            new_input_embeds.append(new_embeds)

        # Source: Line 153
        new_input_embeds = torch.stack(new_input_embeds, dim=0)

        return new_input_embeds

    def forward(
        self,
        latent_tokens: torch.Tensor,
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode latent tokens to logits for reconstruction.

        Source: third-part/C3-Context-Cascade-Compression-main/C3-master/C3/model/C3.py
                Lines 121-160 (decoder section), 182-246 (forward)

        Args:
            latent_tokens: [B, N, D_encoder] from C3Encoder
            prompt_ids: [B, L] prompt token IDs (with special tokens)
            prompt_embeds: [B, L, D_decoder] pre-computed prompt embeddings (optional)
            attention_mask: [B, L] attention mask (optional)
            labels: [B, L+N] labels for loss computation (optional)

        Returns:
            logits: [B, L+N, vocab_size] token logits

        Dimensions Flow (matching official Lines 121-153):
            Step 1 (Lines 124-126): latent_tokens -> mm_projector -> projected_latent [B, N, D_decoder]
            Step 2 (Lines 129-153): Insert projected_latent between <img> and </img>
            Step 3 (Lines 155-160): new_input_embeds -> Decoder LLM -> hidden_states
            Step 4 (Line 218): hidden_states -> lm_head -> logits
        """
        # Get device from LLM
        device = self.llm.device

        # Move latent_tokens to device
        latent_tokens = latent_tokens.to(device)

        batch_size = latent_tokens.shape[0]
        N = self.num_latent_tokens

        # ====================================================================
        # Step 1: Project latent tokens via mm_projector
        # Source: Lines 124-126
        # for latent_context in latent_contexts:
        #     latent_context = self.mm_projector(latent_context)
        # ====================================================================
        projected_latent = self.mm_projector(latent_tokens)  # [B, N, D_decoder]

        # ====================================================================
        # Step 2: Prepare decoder input
        # ====================================================================
        if prompt_embeds is not None:
            input_embeds = prompt_embeds.to(device)
            input_ids = prompt_ids.to(device) if prompt_ids is not None else None
        elif prompt_ids is not None:
            input_ids = prompt_ids.to(device)
            # Get embeddings from LLM's embedding layer
            # Source: Line 62
            input_embeds = self.llm.get_input_embeddings()(input_ids)
        else:
            # Create default prompt with special tokens
            # Source: Lines 372-373
            # qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*N + DEFAULT_IM_END_TOKEN + '\n' + qs
            prompt_text = (
                C3_IM_START_TOKEN
                + C3_IM_PATCH_TOKEN * N
                + C3_IM_END_TOKEN
                + "\nRepeat the text: "
            )
            prompt_ids_list = [
                self.tokenizer.encode(prompt_text, return_tensors="pt")[0]
            ] * batch_size

            # Pad to same length
            max_len = max(len(p) for p in prompt_ids_list)
            input_ids = torch.zeros(
                batch_size, max_len, dtype=torch.long, device=device
            )
            for i, p in enumerate(prompt_ids_list):
                input_ids[i, : len(p)] = p

            input_embeds = self.llm.get_input_embeddings()(input_ids)

        # ====================================================================
        # Step 3: Insert projected latent tokens between <img> and </img>
        # Source: Lines 129-153
        # ====================================================================
        new_input_embeds = self._prepare_decoder_input_with_latent_tokens(
            input_ids, input_embeds, projected_latent
        )

        # ====================================================================
        # Step 4: Forward through Decoder LLM
        # Source: Lines 155-160
        # return super(C3QwenModel, self).forward(
        #     input_ids=None, attention_mask=attention_mask,
        #     inputs_embeds=inputs_embeds, ...
        # )
        # ====================================================================
        outputs = self.llm(
            inputs_embeds=new_input_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

        # Source: Lines 217-219
        # hidden_states = outputs[0]
        # logits = self.lm_head(hidden_states)
        return outputs.logits

    def generate(
        self,
        latent_tokens: torch.Tensor,
        prompt: str = "Repeat the text: ",
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """Generate text from latent tokens autoregressively.

        Source: third-part/C3-Context-Cascade-Compression-main/C3-master/C3/model/C3.py
                Lines 368-412 (chat function)

        Args:
            latent_tokens: [B, N, D_encoder] from C3Encoder
            prompt: Text prompt for generation (default: "Repeat the text: ")
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling probability
            do_sample: Whether to sample (True) or greedy decode (False)

        Returns:
            output_ids: [B, generated_length] generated token IDs

        Dimensions Flow:
            Step 1: latent_tokens -> mm_projector -> projected_latent [B, N, D_decoder]
            Step 2: Create prompt with special tokens
            Step 3: Insert projected_latent between <img> and </img>
            Step 4: Autoregressive generation
        """
        # Get device and dtype from LLM
        device = self.llm.device
        dtype = self.llm.dtype  # Get model dtype (BF16 if loaded with BF16)

        # Move latent_tokens to device and match dtype
        latent_tokens = latent_tokens.to(device=device, dtype=dtype)

        batch_size = latent_tokens.shape[0]
        N = self.num_latent_tokens

        # ====================================================================
        # Step 1: Project latent tokens via mm_projector
        # NOTE: Cast mm_projector to same dtype as model to avoid dtype mismatch
        # ====================================================================
        mm_projector = self.mm_projector.to(dtype=dtype)
        projected_latent = mm_projector(latent_tokens)  # [B, N, D_decoder]

        # ====================================================================
        # Step 2: Create prompt with special tokens
        # Source: Lines 372-373
        # qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*N + DEFAULT_IM_END_TOKEN + '\n' + qs
        # ====================================================================
        prompt_with_tokens = (
            C3_IM_START_TOKEN + C3_IM_PATCH_TOKEN * N + C3_IM_END_TOKEN + "\n" + prompt
        )

        # Tokenize prompt
        prompt_ids = self.tokenizer(
            [prompt_with_tokens] * batch_size,
            return_tensors="pt",
            padding=True,
        )
        input_ids = prompt_ids["input_ids"].to(device)
        attention_mask = prompt_ids["attention_mask"].to(device)

        # Get embeddings
        input_embeds = self.llm.get_input_embeddings()(input_ids)

        # ====================================================================
        # Step 3: Insert projected latent tokens
        # ====================================================================
        new_input_embeds = self._prepare_decoder_input_with_latent_tokens(
            input_ids, input_embeds, projected_latent
        )

        # Update attention mask for new sequence length
        new_seq_len = new_input_embeds.shape[1]
        new_attention_mask = torch.ones(
            batch_size, new_seq_len, device=device, dtype=attention_mask.dtype
        )

        # ====================================================================
        # Step 4: Autoregressive Generation
        # Source: Lines 396-405
        # output_ids = self.generate(
        #     input_ids,
        #     context_ids=inputs_context_ids,
        #     do_sample=False,
        #     num_beams=1,
        #     ...
        # )
        # ====================================================================
        output_ids = self.llm.generate(
            inputs_embeds=new_input_embeds,
            attention_mask=new_attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        return output_ids

    def decode_tokens(self, token_ids: torch.Tensor) -> List[str]:
        """Decode token IDs to text strings.

        Args:
            token_ids: [B, L] generated token IDs

        Returns:
            List[str]: Decoded text strings, len = B
        """
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)


def build_c3_decoder(
    config: Dict[str, Any],
    encoder_hidden_dim: int,
    encoder_type: str = "C3Encoder",
) -> C3Decoder:
    """Build C3 Cascade Decoder from config dict.

    CASCADE CONSTRAINT:
        This function validates that the encoder is C3Encoder.
        C3Decoder must be paired with C3Encoder for the cascade architecture.

    Config keys (all required):
        - model_name: str - HuggingFace model name (e.g., 'Qwen/Qwen2.5-3B')
        - pretrained: bool - Whether to load pretrained weights
        - freeze: bool - Whether to freeze LLM weights
        - num_latent_tokens: int - Number of latent tokens N (must match encoder)

    Args:
        config: Config dict
        encoder_hidden_dim: Hidden dimension from C3Encoder (required)
        encoder_type: Type name of encoder for validation (default: 'C3Encoder')

    Example Config:
        # Encoder: Qwen2.5-1.5B (hidden_dim=1536)
        encoder_config = {
            "model_name": "Qwen/Qwen2.5-1.5B",
            "pretrained": True,
            "freeze": False,
            "num_latent_tokens": 32,
            "max_length": 1280,
        }

        # Decoder: Qwen2.5-3B (hidden_dim=2048)
        decoder_config = {
            "model_name": "Qwen/Qwen2.5-3B",
            "pretrained": True,
            "freeze": False,
            "num_latent_tokens": 32,  # Must match encoder
        }

        encoder = build_c3_encoder(encoder_config)
        decoder = build_c3_decoder(
            decoder_config,
            encoder_hidden_dim=encoder.hidden_dim,  # 1536
            encoder_type="C3Encoder",
        )
        # mm_projector: 1536 -> 2048
    """
    # Validate cascade constraint
    C3Decoder.validate_encoder(encoder_type)

    decoder = C3Decoder(config, encoder_hidden_dim)

    # Logging
    freeze_str = "frozen" if config["freeze"] else "trainable"
    logger.info(
        "[C3Decoder] %s (%s) - encoder_h(%d) -> mm_projector -> decoder_h(%d) -> vocab(%d)",
        decoder.model_name,
        freeze_str,
        encoder_hidden_dim,
        decoder.hidden_dim,
        decoder.vocab_size,
    )

    return decoder
