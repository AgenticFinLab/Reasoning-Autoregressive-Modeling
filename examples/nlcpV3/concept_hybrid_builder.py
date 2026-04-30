"""NLCP V3 Concept Pyramid Builder: Groundtruth Concept Extraction.

DESIGN SOURCE:
    Based on hybrid-analysis.md: Concept Pyramid Architecture

TWO-PHASE ARCHITECTURE (hybrid-analysis.md Section 1.4):
    Phase 1: ConceptPyramidBuilder (this file) — Extract groundtruth from CoT
    Phase 2: ConceptPredictor (separate)    — Generate autoregressively from Q

BUILDER ROLE (hybrid-analysis.md Section 4.1):
    Input: (Q, CoT, Solution)
    - CoT:      Core source for building the concept pyramid
    - Q:        Context/prior (conditions extraction, doesn't enter pyramid)
    - Solution: Used for validation (outside this module)

    Mechanism (purely residual, following VAR's VQ-VAE Stage 1):
        H_CoT = Encoder(CoT)                                   # Encode CoT
        H_proj = Linear(H_CoT)                                 # Project to D
        H_rest_0 = H_proj
        for k in range(K):                                     # K=6 levels
            A_k = softmax(Q_k @ H_rest_k^T / sqrt(D))         # Soft attention
            C_k = level_proj(A_k @ H_rest_k)                   # Base concept
            R_k = A_k^T @ C_k                                  # Reconstruct
            H_rest_{k+1} = H_rest_k - R_k                      # Residual update

    Output: Groundtruth concept pyramid [C_0, C_1, ..., C_{K-1}]

NOTE: This module does NOT compute losses. Loss computation is handled
    externally (e.g., in the training loop) using the returned concepts
    and auxiliary data. See hybrid-analysis.md Section 5 for loss design.

KEY DESIGN PRINCIPLES (hybrid-analysis.md):
    1. Query expansion:         Section 1.1, 6.2  — 1→2→4→8→16→32 learnable queries
    2. Soft attention:          Section 3.2       — Competition-based segment-concept correspondence
    3. Residual reconstruction: Section 2.1-2.3   — Coarse-to-fine information decomposition
    4. Purely residual:         VAR.md            — No cross-scale conditioning (Stage 1)
    5. Intra-level ordering:    Section 3.2       — Concepts ordered by CoT position
    6. Builder-Predictor separation: Section 4   — Builder for groundtruth, Predictor for generation

ENCODER INTEGRATION (hybrid-analysis.md Section 1.2):
    self.reason_model is loaded as AutoModelForCausalLM (e.g., Qwen2.5
    with lm_head). A SINGLE model serves both roles:
      (1) Encoding: reason_model.model (backbone) → CoT hidden states
      (2) Decoding: reason_model (full) → solution token logits via lm_head
    No separate solution_decoder is needed. The lm_head enables
    NTP / reasoning loss to validate that the concept pyramid
    supports effective reasoning.

    back_proj (D → D_encoder) maps concept embeddings back to encoder
    map concept embeddings back to the model's input space. The NTP loss
    is computed as: Q + back_proj(concepts) → reason_model → solution logits.

    Usage:
        config = load_config("path/to/config.yml")  # Raw dict

        builder = ConceptPyramidBuilder(config)
        # Reason model + tokenizer are created internally
        # builder.reason_model_hidden_dim is derived from the loaded model

        # Stage 1: Encode CoT → EncoderOutput (accepts text or tokens)
        enc_out = builder.encode_cot(cot_texts)  # auto-tokenize
        enc_out = builder.encode_cot(cot_input_ids, attention_mask=cot_mask)
        # enc_out.hidden_states: [B, L, D_encoder]

        # Stage 2a: Build full pyramid → PyramidOutput
        pyramid = builder(enc_out.hidden_states)
        # pyramid.concepts: List[Tensor] — [C_0, ..., C_{K-1}]
        # pyramid.level_outputs: List[LevelOutput] — per-level detail
        # pyramid.reconstructed_hidden: [B, L, D] — for recon loss

DIMENSION FLOW:
    Input:  CoT tokens → encoder → H_CoT [B, L, D_encoder]
            → input_proj → H_proj [B, L, D]
    Output: PyramidOutput (forward)

    Level k processing (captured in LevelOutput):
        H_rest_k:      [B, L, D]          (residual hidden states)
        Q_k:           [L_k, D]           (learnable queries)
        A_k:           [B, L_k, L]        (attention weights)
        C_k:           [B, L_k, D]        (concept — purely from residual)
        R_k:           [B, L, D]          (reconstruction from level k)

REFERENCES:
    - hybrid-analysis.md: Full architectural analysis
    - VAR.md Section 5.2.2: Residual decomposition (f_hat + f_rest)
"""

import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import LoraConfig, get_peft_model

    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False


# =========================================================================
# Output Dataclasses — structured outputs for each Builder stage
# =========================================================================
# PRINCIPLE: Each dataclass corresponds to one processing stage of the
#   Builder, replacing loose Dict[str, Any] with typed, self-documenting
#   containers. This ensures uniform handling and IDE discoverability.
#
# DESIGN SOURCE (hybrid-analysis.md):
#   - Section 1.2: Encoder → H_CoT
#   - Section 2.1-2.3: Residual flow (f_hat, f_rest, commit-refinement)
#   - Section 3.2: Soft attention A_k
#   - Section 4.1: Builder mechanism overview
#
# DATA FLOW:
#   EncoderOutput  →  PyramidOutput (contains List[LevelOutput])


@dataclass
class EncoderOutput:
    """Output of the CoT encoding stage.

    PRINCIPLE (hybrid-analysis.md Section 1.2):
        H_CoT = Encoder(CoT). The encoder produces token-level hidden
        states from CoT, analogous to DLCM's encoder.

    PURPOSE:
        Encapsulate the raw encoder output so that downstream stages
        (projection, residual decomposition) receive a typed object
        instead of a bare tensor.

    DIMENSION FLOW:
        hidden_states: [B, L, D_encoder] — last layer hidden states
        attention_mask: [B, L] — 1=valid, 0=pad (optional)

    Attributes:
        hidden_states: Encoder hidden states [B, L, D_encoder]
        attention_mask: Token validity mask [B, L] (optional)
    """

    hidden_states: torch.Tensor  # [B, L, D_encoder] — H_CoT
    attention_mask: Optional[torch.Tensor] = None  # [B, L]


@dataclass
class LevelOutput:
    """Per-level intermediate/output data from one pyramid level.

    PRINCIPLE (VAR.md — purely residual Stage 1):
        Each level produces concepts purely from residual decomposition.
        C_k = level_proj(A_k @ H_rest_k) — no cross-scale conditioning.
        R_k = A_k^T @ C_k enters the residual flow.

    PURPOSE:
        Capture all per-level data needed for:
        - External loss computation (Section 5):
          L_reconstruction uses R_k (reconstruction)
          L_ordering uses A_k (attention_weights)
        - Stage 2 predictor targets: concepts
        - Visualization / debugging

    DIMENSION FLOW (level k):
        concepts:          [B, L_k, D]  — C_k (purely from residual)
        attention_weights: [B, L_k, L]  — A_k (soft attention)
        reconstruction:    [B, L, D]    — R_k = A_k^T @ C_k

    Attributes:
        concepts: Concepts from residual decomposition [B, L_k, D]
            C_k = level_proj(A_k @ H_rest_k).
        attention_weights: Soft attention weights [B, L_k, L]
            A_k = softmax(Q_k @ H_rest_k^T / (sqrt(D) * tau)).
            For ordering loss (Section 5.1.2).
        reconstruction: Reconstruction from this level [B, L, D]
            R_k = A_k^T @ C_k.
    """

    concepts: torch.Tensor  # [B, L_k, D]  — C_k (from residual)
    attention_weights: torch.Tensor  # [B, L_k, L]  — A_k (soft attention)
    reconstruction: torch.Tensor  # [B, L, D]    — R_k = A_k^T @ C_k


@dataclass
class PyramidOutput:
    """Full output of forward() — all K levels of the concept pyramid.

    PRINCIPLE (hybrid-analysis.md Section 4.1, Section 2.1):
        The Builder extracts groundtruth concepts level by level using
        soft attention over residual hidden states. After K levels:
        - f_hat_K  = H_proj (total reconstruction, if decomposition is exact)
        - f_rest_K = H_proj - f_hat_K (residual, should approach zero)

    PURPOSE:
        Encapsulate the complete concept pyramid plus all intermediate
        data needed for external loss computation (Section 5):
        - L_reconstruction: uses projected_hidden vs reconstructed_hidden
        - L_ordering:       uses level_outputs[].attention_weights
        - L_solution:       uses concepts (cat of all levels)

    DIMENSION FLOW:
        concepts:            List of [B, L_k, D] for k=0..K-1
        level_outputs:       List[LevelOutput] for k=0..K-1
        encoder_hidden_states: [B, L, D_encoder] — original H_CoT (frozen)
        projected_hidden:    [B, L, D] — H_proj = input_proj(H_CoT)
        reconstructed_hidden:[B, L, D] — f_hat_K = sum of R_k
        reconstructed_encoder_hidden: [B, L, D_encoder] — back_proj(f_hat_K)
        residual_hidden:     [B, L, D] — f_rest_K = H_proj - f_hat_K

    Attributes:
        concepts: Concepts per level [C_0, ..., C_{K-1}]
            Each C_k: [B, L_k, D]. Purely from residual decomposition.
        level_outputs: Per-level detailed outputs [LevelOutput_0, ..., LevelOutput_{K-1}]
            Contains concepts, attention_weights, reconstruction
            for each level — needed for external loss computation.
        encoder_hidden_states: Original CoT encoder output [B, L, D_encoder]
            H_CoT from frozen reason_model. This is the stable
            reconstruction target, analogous to VAR's frozen encoder output.
        projected_hidden: Projected encoder output [B, L, D]
            H_proj = Linear(H_CoT). Internal concept space representation.
        reconstructed_hidden: Accumulated reconstruction [B, L, D]
            f_hat_K = sum_{k=0}^{K-1} R_k in concept space.
        reconstructed_encoder_hidden: Back-projected reconstruction [B, L, D_encoder]
            back_proj(f_hat_K). Reconstruction target comparison:
            L_recon = ||back_proj(f_hat_K) - H_CoT||^2.
        residual_hidden: Final residual [B, L, D]
            f_rest_K = H_proj - f_hat_K. Should approach zero for
            exact decomposition (Section 2.1).
        num_levels: Number of levels K
        level_lengths: Concepts per level [L_0, L_1, ..., L_{K-1}]
        attention_mask: Optional mask [B, L] for loss computation.
            1=valid token, 0=pad. Passed through from forward() input.
    """

    concepts: List[torch.Tensor]  # [C_0, ..., C_{K-1}], each [B, L_k, D]
    level_outputs: List[LevelOutput]  # Per-level detailed data
    encoder_hidden_states: torch.Tensor  # [B, L, D_encoder] — H_CoT (frozen)
    projected_hidden: torch.Tensor  # [B, L, D] — H_proj
    reconstructed_hidden: torch.Tensor  # [B, L, D] — f_hat_K
    reconstructed_encoder_hidden: torch.Tensor  # [B, L, D_encoder] — back_proj(f_hat_K)
    residual_hidden: torch.Tensor  # [B, L, D] — f_rest_K
    num_levels: int  # K
    level_lengths: List[int]  # [L_0, L_1, ..., L_{K-1}]
    attention_mask: Optional[torch.Tensor] = None  # [B, L]

    @property
    def total_concepts(self) -> int:
        """Total concepts across all levels: sum(L_k) for k=0..K-1."""
        return sum(self.level_lengths)

    @property
    def all_attentions(self) -> List[torch.Tensor]:
        """Convenience: extract attention weights from all levels."""
        return [lo.attention_weights for lo in self.level_outputs]

    @property
    def all_reconstructions(self) -> List[torch.Tensor]:
        """Convenience: extract reconstructions from all levels."""
        return [lo.reconstruction for lo in self.level_outputs]

    def cat_concepts(self) -> torch.Tensor:
        """Concatenate all concepts: [B, sum(L_k), D].

        PURPOSE: Useful for solution loss (Section 5.1.3) where
            all concepts are pooled to predict the solution.
        """
        return torch.cat(self.concepts, dim=1)  # [B, sum(L_k), D]


class ConceptPyramidBuilder(nn.Module):
    """Build groundtruth concept pyramids from CoT.

    PURPOSE (hybrid-analysis.md Section 4.1):
        Phase 1 of the two-phase architecture. Extracts hierarchical
        groundtruth concepts from Chain-of-Thought using soft attention
        with learnable query expansion and residual reconstruction.
        The output serves as groundtruth for training the
        ConceptPredictor (Phase 2).

    PRINCIPLE (hybrid-analysis.md Section 1.3):
        The concept pyramid has two structural dimensions:
        - Inter-level: coarse-to-fine granularity (k=0..K-1)
        - Intra-level: positional ordering within each level (j=0..L_k-1)

    METHOD:
        forward():  All levels in one pass (training)

    ATTRIBUTES:
        reason_model: The decoder-only Transformer (e.g., Qwen), loaded as
            AutoModelForCausalLM. Used for BOTH:
            (1) CoT hidden state extraction via its backbone (model.model)
            (2) Solution generation via its lm_head (future NTP loss)
            This is the SINGLE model around which the architecture is built:
            extract concepts from CoT, then generate solutions from Q + concepts.
            Can be frozen, pruned, or LoRA-adapted via config.
            Initialized by _init_reason_model().
        tokenizer: Tokenizer paired with reason_model for text encoding.
        input_proj: Projection from reason_model hidden_dim to concept_dim
        input_proj_norm: LayerNorm after input_proj for numerical stability
        concept_queries: Learnable queries per level [K levels]
        temperature: Learnable attention temperature
        level_projs: Level-specific output projections
        back_proj: Projection from concept_dim back to encoder_dim.
            Maps concept embeddings into the model's input space for
            reasoning loss computation. Initialized as transpose of
            input_proj (pseudo-inverse).
    """

    def __init__(
        self,
        config: dict,
    ):
        """Initialize Concept Pyramid Builder.

        PRINCIPLE (hybrid-analysis.md Section 4.1, Section 1.2):
            The Builder extracts groundtruth concepts from CoT using the
            SAME decoder-only model that will later generate the Solution.
            The reason_model is loaded as AutoModelForCausalLM so it has
            both the backbone (for CoT feature extraction) and the lm_head
            (for future NTP / reasoning loss computation).

        PURPOSE:
            Initialize all components for concept pyramid extraction,
            including the reason_model and tokenizer loaded internally
            so they participate in end-to-end training.

        METHOD:
            - Load pretrained reason_model via AutoModelForCausalLM
            - Load paired tokenizer via AutoTokenizer.from_pretrained()
            - Apply training strategy: freeze backbone (configurable), apply LoRA
            - Derive reason_model_hidden_dim from model config
            - Construct projection, queries, attention layers

        Args:
            config: Raw config dict with hyperparameters.
                Caches sub-configs: reason_cfg, pyramid_cfg, builder_cfg, train_rm_cfg.
                Uses reason_cfg["reason_model_name"] to load the model.
                Uses reason_cfg["reason_model_num_layers"] for layer pruning.
                Uses train_rm_cfg["freeze"] for backbone freezing.
                Uses train_rm_cfg["lora"] for optional LoRA adaptation.
                Uses builder_cfg["use_positional_query_init"] for query init mode.
        """
        super().__init__()
        self.config = config
        # Cache sub-configs to eliminate repeated deep dict lookups
        self.reason_cfg = config["model"]["reason_model"]
        self.pyramid_cfg = config["model"]["pyramid"]
        self.builder_cfg = config["model"]["builder"]
        self.use_positional_query_init = self.builder_cfg["use_positional_query_init"]
        # Training strategy for reason_model (freeze, lora)
        self.train_rm_cfg = config["training"]["reason_model"]

        # =================================================================
        # Component 0: Reason Model (decoder-only Transformer + lm_head)
        # =================================================================
        # PRINCIPLE: One model, two roles:
        #   (1) Encoding: reason_model.model(CoT) → H_CoT [B, L, D_reason]
        #       The backbone produces hidden states for concept extraction.
        #   (2) Decoding: reason_model(Q + concept_embeds) → logits [B, L, V]
        #       The lm_head enables NTP / reasoning loss on solution tokens.
        # This is why we load AutoModelForCausalLM instead of AutoModel.
        self.reason_model, self.tokenizer, self.reason_model_hidden_dim = (
            self._init_reason_model(self.reason_cfg, self.train_rm_cfg)
        )

        # =================================================================
        # Dimension consistency check (VAR-faithful principle)
        # =================================================================
        # PRINCIPLE: In VAR, quant_conv preserves dimension (in_ch == out_ch).
        #   When hidden_dim != encoder hidden_size, input_proj becomes a lossy
        #   compression, and back_proj cannot perfectly invert it. This creates
        #   a theoretical floor on reconstruction error unrelated to the
        #   pyramid's capacity. Set hidden_dim = encoder hidden_size to avoid.
        concept_dim = self.pyramid_cfg["hidden_dim"]
        if concept_dim != self.reason_model_hidden_dim:
            warnings.warn(
                f"\u26a0\ufe0f  pyramid.hidden_dim ({concept_dim}) != "
                f"encoder hidden_size ({self.reason_model_hidden_dim}). "
                f"This creates a lossy projection bottleneck — "
                f"reconstruction error has a non-zero theoretical floor. "
                f"Set hidden_dim = {self.reason_model_hidden_dim} for "
                f"VAR-faithful lossless projection.",
                stacklevel=2,
            )

        # =================================================================
        # Component 1: Projection (encoder_dim → concept_dim) + LayerNorm
        # =================================================================
        # PRINCIPLE (hybrid-analysis.md Section 1.2):
        #   H_proj = LayerNorm(Linear(H_CoT)) ∈ ℝ^{B×L×D}
        #   This is the "CoT information to decompose" via residual flow.
        # PURPOSE: Project reason_model output to the concept dimension D,
        #   then normalize to unit scale. Without LayerNorm, the encoder
        #   hidden states have large magnitudes (std ~10, max ~200 for
        #   Qwen2.5), causing the random pyramid to explode (reconstructed
        #   std ~200 vs projected std ~12, making recon_loss ~44000).
        #   LayerNorm stabilizes the input to the residual decomposition,
        #   ensuring recon_loss starts at a reasonable magnitude.
        # METHOD: Linear layer [D_reason → D] followed by LayerNorm(D).
        #   Input:  [B, L, D_reason]
        #   Output: [B, L, D] (normalized to mean=0, std≈1 per token)
        self.input_proj = nn.Linear(
            self.reason_model_hidden_dim, self.pyramid_cfg["hidden_dim"]
        )
        self.input_proj_norm = nn.LayerNorm(self.pyramid_cfg["hidden_dim"])

        # =================================================================
        # Component 2: Learnable Concept Queries (Query Expansion)
        # =================================================================
        # PRINCIPLE (hybrid-analysis.md Section 1.1):
        #   L_k = 2^k for k < K. Each level has L_k learnable query vectors.
        #   Expansion: 1→2→4→8→16→32 (for K=6).
        #   These queries replace VAR's codebook (Section 7.1).
        # PURPOSE: Define "what to attend to" at each level.
        #   Q_{k,j} learns to attend to the j-th segment structure at level k.
        # METHOD: nn.ParameterList with one [L_k, D] parameter per level.
        #   Level 0: [1, D], Level 1: [2, D], ..., Level 5: [32, D]
        self.concept_queries = nn.ParameterList(
            [
                nn.Parameter(torch.randn(length, self.pyramid_cfg["hidden_dim"]))
                for length in self.pyramid_cfg["level_lengths"]
            ]
        )

        # =================================================================
        # Component 3: Attention Temperature
        # =================================================================
        # PRINCIPLE (hybrid-analysis.md Section 3.4):
        #   A_k = softmax(Q_k @ H_rest_k^T / (√D × τ))
        #   Too high τ → diffuse attention; too low → sharp but inflexible.
        # PURPOSE: Control attention sharpness across all levels.
        # METHOD: Learnable scalar τ, initialized to 1.
        self.temperature = nn.Parameter(torch.ones(1))

        # =================================================================
        # Component 4: Level-Specific Projections
        # =================================================================
        # PRINCIPLE (hybrid-analysis.md Section 1.2, Section 3.5):
        #   C_{k,j}_base = level_proj(A_{k,j} @ H_rest_k)
        #   level_proj transforms raw pooled representations into
        #   task-relevant concept features.
        # PURPOSE: Project attended residual to concept space, per level.
        # METHOD: Linear layer [D → D] for each level.
        #   Input:  A_k @ H_rest_k → [B, L_k, D] (raw pooled)
        #   Output: C_k_base → [B, L_k, D] (base concept)
        self.level_projs = nn.ModuleList(
            [
                nn.Linear(
                    self.pyramid_cfg["hidden_dim"],
                    self.pyramid_cfg["hidden_dim"],
                )
                for _ in range(self.pyramid_cfg["num_levels"])
            ]
        )

        # =================================================================
        # Component 6: Back-Projection (concept_dim → encoder_dim)
        # =================================================================
        # PRINCIPLE: back_proj maps concept embeddings (D) back to encoder
        #   dimension (D_encoder), enabling NTP reasoning loss computation.
        #   The model operates in D_encoder space, but concepts are in D space.
        #   back_proj bridges this dimension gap.
        #
        # INITIALIZATION: back_proj.weight is initialized as the transpose
        #   of input_proj.weight (pseudo-inverse). This gives a natural
        #   starting point: if input_proj maps H_CoT → H_proj, then
        #   back_proj approximately maps H_proj → H_CoT.
        #   back_proj is then free to learn during training.
        #
        # DIMENSION FLOW:
        #   Input:  concepts [B, total_C, D]
        #   Output: concept_embeds [B, total_C, D_encoder]
        self.back_proj = nn.Linear(
            self.pyramid_cfg["hidden_dim"],
            self.reason_model_hidden_dim,
            bias=False,
        )

        self._init_weights()

    # =====================================================================
    # Model Initialization Methods
    # =====================================================================

    def _init_reason_model(self, reason_cfg: dict, train_rm_cfg: dict) -> tuple:
        """Initialize reason_model (backbone + lm_head), tokenizer, and hidden_dim.

        PRINCIPLE (hybrid-analysis.md Section 1.2):
            The reason_model serves DUAL roles in the architecture:
              (1) Encoding: backbone produces CoT hidden states for concept extraction
              (2) Decoding: lm_head enables NTP / reasoning loss on solution tokens
            We load AutoModelForCausalLM (includes lm_head) so a single model
            handles both roles. No separate solution_decoder is needed.

            For encoding, we access the backbone via reason_model.model
            (which is the Qwen2Model inside AutoModelForCausalLM).
            For decoding, we use the full reason_model which includes lm_head.

        PURPOSE:
            Encapsulate reason_model initialization with support for:
            (1) Loading pretrained model (AutoModelForCausalLM)
            (2) Loading paired tokenizer
            (3) Optional layer pruning (reason_model_num_layers)
            (4) Configurable freeze strategy (train_rm_cfg["freeze"])
            (5) Optional LoRA fine-tuning (train_rm_cfg["lora"])

        CRITICAL:
            Use AutoModelForCausalLM (not AutoModel) because:
            - We need the lm_head for NTP / reasoning loss computation
            - A single model serves both encoding and decoding roles
            - This avoids maintaining a separate solution_decoder copy

        Args:
            reason_cfg: Sub-config dict under config["model"]["reason_model"].
                Contains model name, num_layers, etc.
            train_rm_cfg: Sub-config dict under config["training"]["reason_model"].
                Contains freeze (bool) and lora (dict or null).

        Returns:
            Tuple of (reason_model, tokenizer, hidden_dim)
        """
        # Step 1: Load pretrained model with lm_head
        # AutoModelForCausalLM = backbone (Qwen2Model) + lm_head
        reason_model = AutoModelForCausalLM.from_pretrained(
            reason_cfg["reason_model_name"]
        )
        # hidden_dim: D_reason (e.g., 896 for Qwen2.5-0.5B)
        hidden_dim = reason_model.config.hidden_size

        # Step 2: Load paired tokenizer
        tokenizer = AutoTokenizer.from_pretrained(reason_cfg["reason_model_name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Step 3: Apply LoRA if configured
        # PURPOSE: Enable parameter-efficient fine-tuning of the backbone.
        #   LoRA adapters are small trainable matrices injected into
        #   target linear layers (e.g., q_proj, v_proj), allowing the
        #   base model weights to remain frozen while still adapting.
        lora_cfg = train_rm_cfg["lora"]
        if lora_cfg is not None:
            if not _PEFT_AVAILABLE:
                raise ImportError(
                    "PEFT library is required for LoRA fine-tuning. "
                    "Install with: pip install peft"
                )
            lora_config = LoraConfig(
                r=lora_cfg["r"],
                lora_alpha=lora_cfg["lora_alpha"],
                target_modules=lora_cfg["target_modules"],
                lora_dropout=lora_cfg["lora_dropout"],
                bias=lora_cfg["bias"],
            )
            reason_model = get_peft_model(reason_model, lora_config)
            # NOTE: LoRA adapters are trainable regardless of freeze setting.
            #   After get_peft_model, only LoRA params have requires_grad=True.

        # Step 4: Freeze backbone if configured
        # PRINCIPLE: Like VAR's frozen VQVAE encoder, freezing the reason_model
        #   produces stable CoT encodings that serve as a fixed reconstruction
        #   target. When freeze=false, the backbone is also trained (end-to-end).
        if train_rm_cfg["freeze"]:
            for param in reason_model.parameters():
                param.requires_grad = False
            # If LoRA is applied, re-enable LoRA adapter gradients
            if lora_cfg is not None:
                reason_model.enable_adapter_layers()
                for name, param in reason_model.named_parameters():
                    if "lora_" in name:
                        param.requires_grad = True

        # Step 5: Prune layers if specified
        # PURPOSE: Reduce computation by using fewer Transformer layers.
        #   reason_model_num_layers=-1 means use ALL layers (no pruning).
        #
        # Layer access paths for AutoModelForCausalLM:
        #   Plain:         reason_model.model.layers
        #   PEFT-wrapped:  reason_model.base_model.model.layers
        if reason_cfg["reason_model_num_layers"] > 0:
            layers_pruned = False
            # Try all known access paths for the transformer layers
            for obj in [
                reason_model,
                getattr(reason_model, "model", None),
                getattr(getattr(reason_model, "base_model", None), "model", None),
            ]:
                if obj is not None and hasattr(obj, "layers"):
                    if reason_cfg["reason_model_num_layers"] < len(obj.layers):
                        obj.layers = obj.layers[: reason_cfg["reason_model_num_layers"]]
                        layers_pruned = True
                        break
            if not layers_pruned:
                warnings.warn(
                    f"Could not find layers to prune in {type(reason_model).__name__}. "
                    f"Requested {reason_cfg['reason_model_num_layers']} layers."
                )

        return reason_model, tokenizer, hidden_dim

    def _get_backbone(self) -> nn.Module:
        """Get the Transformer backbone from reason_model for encoding.

        PRINCIPLE:
            reason_model is loaded as AutoModelForCausalLM, which wraps
            the backbone (Qwen2Model) inside `reason_model.model`.
            For CoT feature extraction we only need the backbone — the
            lm_head is reserved for NTP / reasoning loss computation.

        PURPOSE:
            Provide consistent access to the backbone regardless of
            whether the model is PEFT-wrapped or not.

        Access paths:
            Plain model:         reason_model.model  (Qwen2Model)
            PEFT-wrapped model:  reason_model.base_model.model  (Qwen2Model)

        Returns:
            The Transformer backbone module (e.g., Qwen2Model)
        """
        if hasattr(self.reason_model, "base_model"):
            # PEFT-wrapped: reason_model.base_model.model
            inner = self.reason_model.base_model
            if hasattr(inner, "model"):
                return inner.model  # Qwen2Model under PEFT
            return inner
        elif hasattr(self.reason_model, "model"):
            # Plain AutoModelForCausalLM: reason_model.model
            return self.reason_model.model  # Qwen2Model
        else:
            # Fallback (shouldn't happen for standard HF models)
            return self.reason_model

    def _init_weights(self):
        """Initialize weights.

        PRINCIPLE (hybrid-analysis.md Section 6.2):
            Positional query initialization provides a starting point where
            query j at level k is biased toward position j/L_k.
            This accelerates convergence by providing DLCM-style
            segment-concept correspondence as a prior.

        PURPOSE:
            Initialize projection layers and concept queries.

        METHOD:
            - input_proj: Xavier uniform
            - concept_queries (positional): xavier + α × PE(j/L_k), α=0.5
            - concept_queries (random): xavier uniform
            - level_projs: Xavier uniform
        """
        # Projection: Xavier uniform
        nn.init.xavier_uniform_(self.input_proj.weight)  # [D, D_encoder]
        nn.init.zeros_(self.input_proj.bias)  # [D]

        # Concept queries: positional or random initialization
        if self.use_positional_query_init:
            # Section 6.2: Q_{k,j} = xavier_uniform(j, D) + α × PE(j / L_k)
            positional_init_alpha = self.builder_cfg["positional_init_alpha"]

            for level_idx, queries in enumerate(self.concept_queries):
                L_k = queries.shape[0]  # Number of queries at this level
                D = queries.shape[1]  # Concept dimension

                # Step 1: Xavier uniform base
                nn.init.xavier_uniform_(queries)  # [L_k, D]

                # Step 2: Sinusoidal positional encoding at normalized positions
                # positions_norm: [0, 1/L_k, 2/L_k, ..., (L_k-1)/L_k]
                positions_norm = torch.arange(L_k, dtype=torch.float32) / L_k  # [L_k]

                # Standard sinusoidal PE (Vaswani et al., 2017)
                dim_half = D // 2
                pe = torch.zeros(L_k, D)  # [L_k, D]
                div_term = torch.exp(
                    torch.arange(0, dim_half, dtype=torch.float32)
                    * -(math.log(10000.0) / dim_half)
                )  # [dim_half]

                # PE[:, 0::2] = sin(pos × div), PE[:, 1::2] = cos(pos × div)
                pe[:, 0::2] = torch.sin(
                    positions_norm.unsqueeze(1) * div_term.unsqueeze(0)
                )  # [L_k, dim_half]
                pe[:, 1::2] = torch.cos(
                    positions_norm.unsqueeze(1) * div_term.unsqueeze(0)
                )  # [L_k, dim_half]

                # Add positional signal: Q_k[j] += α * PE(j/L_k)
                with torch.no_grad():
                    queries.add_(positional_init_alpha * pe)  # [L_k, D]
        else:
            # Random initialization: pure Xavier uniform
            for queries in self.concept_queries:
                nn.init.xavier_uniform_(queries)  # [L_k, D]

        # Level projections: Xavier uniform
        for proj in self.level_projs:
            nn.init.xavier_uniform_(proj.weight)  # [D, D]
            nn.init.zeros_(proj.bias)  # [D]

        # Back-projection: initialize as transpose of input_proj (pseudo-inverse)
        # This gives a natural starting point where back_proj ≈ input_proj^{-1}
        if self.back_proj is not None:
            with torch.no_grad():
                self.back_proj.weight.copy_(self.input_proj.weight.T.clone())

    def encode_cot(
        self,
        inputs: Union[List[str], torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
    ) -> EncoderOutput:
        """Encode CoT using the reason_model's Transformer backbone.

        PRINCIPLE (hybrid-analysis.md Section 1.2):
            H_CoT = ReasonModel(CoT). We use reason_model.model (the
            backbone, NOT the full AutoModelForCausalLM) to produce
            hidden states. The lm_head is NOT used here — it is used
            later for NTP / reasoning loss on solution tokens.

        PURPOSE:
            Extract token-level features from CoT. Accepts either raw text
            (auto-tokenized internally) or pre-tokenized tensors.

        METHOD:
            - If inputs is List[str]: auto-tokenize via self.tokenizer
            - If inputs is torch.Tensor: use directly as token IDs
            - Forward through reason_model.model (backbone only)
            - Extract last hidden state as H_CoT

        DIMENSION FLOW:
            Input:  texts [B] (strings)  OR  input_ids [B, L] (token IDs)
                    attention_mask [B, L] (optional, 0=pad, 1=valid)
            Output: EncoderOutput with hidden_states [B, L, D_reason]

        Args:
            inputs: Either a list of text strings or token ID tensor [B, L]
            attention_mask: Attention mask [B, L] (optional, used when
                inputs is a tensor). Ignored when inputs is text.
            max_length: Max sequence length for auto-tokenization (used
                when inputs is text). Defaults to self.pyramid_cfg["max_seq_len"].

        Returns:
            EncoderOutput with hidden_states: [B, L, D_reason]
        """
        # Auto-tokenize if text strings are provided
        if isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], str):
            if max_length is None:
                max_length = self.pyramid_cfg["max_seq_len"]
            tokens = self.tokenizer(
                inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = tokens["input_ids"].to(
                next(self.reason_model.parameters()).device
            )
            attention_mask = tokens["attention_mask"].to(input_ids.device)
        else:
            input_ids = inputs  # [B, L]
            if attention_mask is not None:
                attention_mask = attention_mask.to(input_ids.device)

        # Forward through backbone only (NOT the full AutoModelForCausalLM)
        # reason_model is AutoModelForCausalLM = model (backbone) + lm_head
        # We only need hidden states from the backbone for concept extraction.
        # The lm_head is reserved for NTP / reasoning loss computation.
        backbone = self._get_backbone()
        outputs = backbone(
            input_ids=input_ids,  # [B, L]
            attention_mask=attention_mask,  # [B, L]
            output_hidden_states=True,
        )
        # Extract last hidden state: [B, L, D_reason]
        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state  # [B, L, D_reason]
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hidden = outputs.hidden_states[-1]  # [B, L, D_reason]
        else:
            hidden = outputs[0]  # [B, L, D_reason]

        return EncoderOutput(
            hidden_states=hidden,  # [B, L, D_reason]
            attention_mask=attention_mask,  # [B, L]
        )

    def compute_reasoning_loss(
        self,
        pyramid: PyramidOutput,
        question_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        solution_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NTP / reasoning loss: Q + concept pyramid → solution.

        PRINCIPLE:
            The concept pyramid extracted from CoT must support effective
            reasoning. This loss validates that by feeding Q + concept
            embeddings into the reason_model's lm_head and computing
            cross-entropy on solution tokens.

            Data flow:
                1. Concatenate concepts (all levels) → [B, total_C, D]
                2. back_proj: concepts [B, total_C, D] → [B, total_C, D_encoder]
                3. embed Q tokens: [B, L_Q, D_encoder]
                4. Concatenate: [concept_embeds; Q_embeds] → [B, total_C+L_Q, D_encoder]
                5. Forward through reason_model (full, includes lm_head)
                6. Cross-entropy on solution portion of the output logits

        PURPOSE:
            Validate that the concept pyramid supports reasoning. A pyramid
            that perfectly reconstructs CoT hidden states but cannot produce
            the correct solution is useless.

        Args:
            pyramid: PyramidOutput from forward()
            question_ids: Token IDs for the question [B, L_Q]
            question_attention_mask: Attention mask for question [B, L_Q]
            solution_ids: Token IDs for the solution (target) [B, L_S]

        Returns:
            Scalar NTP loss (cross-entropy on solution tokens)
        """
        assert self.back_proj is not None, "back_proj is None"

        device = question_ids.device
        batch_size = question_ids.shape[0]

        # Step 1: Concatenate all concept levels
        concepts = pyramid.cat_concepts()  # [B, total_C, D]

        # Step 2: Back-project concepts to encoder dimension
        concept_embeds = self.back_proj(concepts)  # [B, total_C, D_encoder]

        # Step 3: Get Q token embeddings from the model's embed_tokens
        backbone = self._get_backbone()
        embed_layer = backbone.get_input_embeddings()  # embed_tokens
        Q_embeds = embed_layer(question_ids)  # [B, L_Q, D_encoder]

        # Step 4: Concatenate concept embeddings + Q embeddings
        #   [concept_embeds; Q_embeds] as input to the decoder
        decoder_input_embeds = torch.cat(
            [concept_embeds, Q_embeds], dim=1
        )  # [B, total_C + L_Q, D_encoder]

        # Attention mask: concepts are all valid (no padding),
        # then append the question attention mask
        concept_mask = torch.ones(
            batch_size,
            concept_embeds.shape[1],
            device=device,
            dtype=question_attention_mask.dtype,
        )  # [B, total_C]
        decoder_attention_mask = torch.cat(
            [concept_mask, question_attention_mask], dim=1
        )  # [B, total_C + L_Q]

        # Step 5: Forward through reason_model (full, includes lm_head)
        #   Use inputs_embeds instead of input_ids since we provide embeddings directly
        outputs = self.reason_model(
            inputs_embeds=decoder_input_embeds,
            attention_mask=decoder_attention_mask,
        )
        logits = outputs.logits  # [B, total_C + L_Q, V]

        # Step 6: Compute cross-entropy on the solution tokens
        #   We only care about the logits at positions after Q
        #   The solution should be predicted autoregressively at the
        #   Q token positions (teacher-forcing: shift left)
        #   So we take logits at Q positions → predict solution tokens
        L_Q = question_ids.shape[1]
        total_C = concept_embeds.shape[1]

        # Logits at Q positions predict the solution tokens (shifted)
        solution_logits = logits[:, total_C:, :]  # [B, L_Q, V]

        # Use solution_ids as target (same length as Q for teacher-forcing)
        # The model should predict solution tokens at each Q position
        L_min = min(solution_logits.shape[1], solution_ids.shape[1])
        reasoning_loss = F.cross_entropy(
            solution_logits[:, :L_min, :].reshape(-1, solution_logits.shape[-1]),
            solution_ids[:, :L_min].reshape(-1),
            ignore_index=-100,  # ignore padding tokens
        )

        return reasoning_loss

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> PyramidOutput:
        """Build concept pyramid from CoT hidden states (all levels).

        PRINCIPLE (hybrid-analysis.md Section 4.1):
            The Builder extracts groundtruth concepts level by level using
            soft attention over residual hidden states. Each level k:
            (1) Attends to H_rest_k with learnable queries Q_k
            (2) Extracts concepts C_k = level_proj(A_k @ H_rest_k)
            (3) Reconstructs R_k = A_k^T @ C_k
            (4) Updates residual: H_rest_{k+1} = H_rest_k - R_k

        PURPOSE:
            Extract all K levels of concepts in one forward pass.
            Used during training to build groundtruth concept pyramids.

        METHOD:
            Iterate k=0..K-1, applying soft attention + residual flow
            at each level. Purely residual — no cross-scale conditioning.
            Collect per-level data into LevelOutput objects, wrap into
            PyramidOutput.

        DIMENSION FLOW:
            Input:  encoder_hidden_states [B, L, D_encoder]
                    attention_mask [B, L] (optional, 1=valid, 0=pad)
            Output: PyramidOutput with concepts, level_outputs, etc.

        Args:
            encoder_hidden_states: CoT hidden states [B, L, D_encoder]
                from self.reason_model or pre-computed via encode_cot()
            attention_mask: Optional mask [B, L] where 1=valid token, 0=pad.
                When provided, padded positions are excluded from attention
                and reconstruction loss computation.

        Returns:
            PyramidOutput containing:
                concepts: [C_0, ..., C_{K-1}], each [B, L_k, D] (purely residual)
                level_outputs: [LevelOutput_0, ..., LevelOutput_{K-1}]
                encoder_hidden_states: [B, L, D_encoder] — original H_CoT
                projected_hidden: [B, L, D]
                reconstructed_hidden: [B, L, D]
                reconstructed_encoder_hidden: [B, L, D_encoder]
                residual_hidden: [B, L, D]
                attention_mask: [B, L] (passed through for loss masking)
        """
        batch_size, seq_len, _ = encoder_hidden_states.shape
        # batch_size: B, seq_len: L, _: D_encoder

        # =================================================================
        # Step 1: Project encoder hidden states to concept dimension
        # =================================================================
        # PRINCIPLE (hybrid-analysis.md Section 1.2):
        #   H_proj = LayerNorm(Linear(H_CoT)) — "CoT information to decompose"
        # PURPOSE: Map encoder output to concept space D, then normalize.
        # METHOD: Linear projection + LayerNorm.
        projected_hidden = self.input_proj_norm(self.input_proj(encoder_hidden_states))
        # projected_hidden: [B, L, D]

        # =================================================================
        # Step 2: Initialize residual decomposition
        # =================================================================
        # PRINCIPLE (hybrid-analysis.md Section 2.1, VAR.md Section 5.2.2):
        #   f_rest = "what still needs encoding" — starts at H_proj, decreases
        #   f_hat  = "what has been encoded"     — starts at 0, accumulates
        #   Constraint: f_hat + f_rest = H_proj (exact decomposition)
        residual_hidden = projected_hidden.clone()
        # residual_hidden: [B, L, D] — H_rest_0 = H_proj

        reconstructed_accumulator = torch.zeros_like(projected_hidden)
        # reconstructed_accumulator: [B, L, D] — H_hat_0 = 0

        all_level_concepts: List[torch.Tensor] = []
        all_level_outputs: List[LevelOutput] = []

        # =================================================================
        # Step 3: Extract all levels with residual decomposition
        # =================================================================
        # PRINCIPLE (hybrid-analysis.md Section 2.1-2.3):
        #   Rank bottleneck guarantees coarse-to-fine:
        #     Level 0 (L_0=1): rank 1 → one global direction
        #     Level 5 (L_5=32): rank 32 → 32 independent directions
        #
        #   Purely residual (VAR.md — no cross-scale conditioning):
        #     C_k = level_proj(A_k @ H_rest_k)
        #     R_k = A_k^T @ C_k
        #     Each level only sees current residual f_rest, nothing else.
        for level_idx in range(self.pyramid_cfg["num_levels"]):
            # level_idx: k ∈ {0, 1, ..., K-1}

            # ── 3a: Get learnable queries for this level ──────────────
            # PRINCIPLE (Section 1.1): L_k = 2^k learnable queries per level
            # PURPOSE: Define "what to attend to" at this granularity.
            level_queries = self.concept_queries[level_idx]
            # level_queries: [L_k, D] — learnable queries

            expanded_queries = level_queries.unsqueeze(0).expand(batch_size, -1, -1)
            # expanded_queries: [B, L_k, D] — queries expanded for batch

            # ── 3b: Compute attention over residual ───────────────────
            # PRINCIPLE (Section 3.2, Mechanism 1 — Softmax Competition):
            #   A_k = softmax(Q_k @ H_rest_k^T / (√D × τ))
            #   Softmax forces Σ_j A_{k,j}(t) = 1 per position t,
            #   creating competition between concept slots.
            attention_scores = torch.bmm(
                expanded_queries, residual_hidden.transpose(1, 2)
            )
            # expanded_queries: [B, L_k, D]
            # residual_hidden.T: [B, D, L]
            # attention_scores: [B, L_k, L]

            attention_scores = attention_scores / (
                math.sqrt(self.pyramid_cfg["hidden_dim"]) * self.temperature
            )
            # attention_scores: [B, L_k, L] — scaled by √D × τ

            # Mask padded positions before softmax so concepts don't attend to them
            if attention_mask is not None:
                # attention_mask: [B, L] → [B, 1, L] for broadcasting
                mask = attention_mask.unsqueeze(1)
                attention_scores = attention_scores.masked_fill(
                    mask == 0, float("-inf")
                )

            soft_boundaries = F.softmax(attention_scores, dim=-1)
            # soft_boundaries: [B, L_k, L] — A_k, soft boundary weights
            # NaN check: if a concept has no valid positions to attend to,
            # softmax of all -inf produces NaN. Replace with zeros.
            if attention_mask is not None:
                soft_boundaries = torch.nan_to_num(soft_boundaries, nan=0.0)

            # ── 3c: Extract concepts (purely residual) ─────────────
            # PRINCIPLE (VAR.md — purely residual Stage 1):
            #   C_k = level_proj(A_k @ H_rest_k)
            #   Each level only looks at the current residual.
            #   No conditioning on previous levels (that's Stage 2).
            level_concepts = torch.bmm(soft_boundaries, residual_hidden)
            # soft_boundaries: [B, L_k, L]
            # residual_hidden: [B, L, D]
            # level_concepts: [B, L_k, D] — raw pooled concepts

            level_concepts = self.level_projs[level_idx](level_concepts)
            # level_concepts: [B, L_k, D] — projected concepts

            # ── 3d: Reconstruct ────────────────────────────────────
            # PRINCIPLE (VAR.md Section 5.2.2):
            #   R_k = A_k^T @ C_k
            #   This is the VAR f_hat update: f_hat += R_k
            reconstruction = torch.bmm(soft_boundaries.transpose(1, 2), level_concepts)
            # soft_boundaries.T: [B, L, L_k]
            # level_concepts: [B, L_k, D]
            # reconstruction: [B, L, D] — R_k

            # ── 3e: Update residual flow ─────────────────────────────
            # PRINCIPLE (Section 2.1, Section 3.2 Mechanism 2):
            #   H_hat_{k+1} = H_hat_k + R_k  (f_hat accumulation)
            #   H_rest_{k+1} = H_rest_k - R_k (f_rest update)
            #   This removes already-captured information, forcing
            #   finer levels to focus on residual details.
            reconstructed_accumulator = reconstructed_accumulator + reconstruction
            # reconstructed_accumulator: [B, L, D] — H_hat_{k+1}

            residual_hidden = residual_hidden - reconstruction
            # residual_hidden: [B, L, D] — H_rest_{k+1}

            all_level_concepts.append(level_concepts)

            # ── 3f: Collect per-level output ──────────────────────────
            # PURPOSE: Wrap per-level data into LevelOutput for
            #   structured access by external loss computation.
            all_level_outputs.append(
                LevelOutput(
                    concepts=level_concepts,  # [B, L_k, D] — C_k
                    attention_weights=soft_boundaries,  # [B, L_k, L] — A_k
                    reconstruction=reconstruction,  # [B, L, D]   — R_k
                )
            )

        # =================================================================
        # Step 4: Back-project reconstruction to encoder space
        # =================================================================
        # PRINCIPLE (VAR-faithful reconstruction):
        #   Reconstruction loss must compare against the ORIGINAL stable
        #   encoder output (H_CoT), not the projected version (H_proj).
        #   back_proj maps f_hat_K from concept space D back to D_encoder.
        #   L_recon = ||back_proj(f_hat_K) - H_CoT||^2
        reconstructed_encoder_hidden = self.back_proj(reconstructed_accumulator)
        # reconstructed_encoder_hidden: [B, L, D_encoder]

        # =================================================================
        # Step 5: Build PyramidOutput
        # =================================================================
        # PURPOSE: Return structured PyramidOutput for external
        #   loss computation (hybrid-analysis.md Section 5).
        return PyramidOutput(
            concepts=all_level_concepts,  # [C_0, ..., C_{K-1}] (purely residual)
            level_outputs=all_level_outputs,  # [LevelOutput_0, ...]
            encoder_hidden_states=encoder_hidden_states,  # [B, L, D_encoder] — H_CoT
            projected_hidden=projected_hidden,  # [B, L, D] — H_proj
            reconstructed_hidden=reconstructed_accumulator,  # [B, L, D] — f_hat_K
            reconstructed_encoder_hidden=reconstructed_encoder_hidden,  # [B, L, D_encoder]
            residual_hidden=residual_hidden,  # [B, L, D] — f_rest_K
            num_levels=self.pyramid_cfg["num_levels"],  # K
            level_lengths=list(
                self.pyramid_cfg["level_lengths"]
            ),  # [L_0, ..., L_{K-1}]
            attention_mask=attention_mask,  # [B, L] (optional)
        )
