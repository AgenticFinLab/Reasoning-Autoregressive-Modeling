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

    Mechanism:
        H_CoT = Encoder(CoT)                                   # Encode CoT
        H_proj = Linear(H_CoT)                                 # Project to D
        H_rest_0 = H_proj
        for k in range(K):                                     # K=6 levels
            A_k = softmax(Q_k @ H_rest_k^T / sqrt(D))         # Soft attention
            C_k_base = level_proj(A_k @ H_rest_k)             # Commit path
            C_k_refined = CrossAttn(Q_k, context, context)     # Refinement path
            C_k = C_k_base + C_k_refined                       # Output concept
            R_k = A_k^T @ C_k_base                             # Reconstruct (base only)
            H_rest_{k+1} = H_rest_k - R_k                      # Residual update

    Output: Groundtruth concept pyramid [C_0, C_1, ..., C_{K-1}]

NOTE: This module does NOT compute losses. Loss computation is handled
    externally (e.g., in the training loop) using the returned concepts
    and auxiliary data. See hybrid-analysis.md Section 5 for loss design.

KEY DESIGN PRINCIPLES (hybrid-analysis.md):
    1. Query expansion:         Section 1.1, 6.2  — 1→2→4→8→16→32 learnable queries
    2. Soft attention:          Section 3.2       — Competition-based segment-concept correspondence
    3. Residual reconstruction: Section 2.1-2.3   — Coarse-to-fine information decomposition
    4. Commit-refinement:       Section 2.3       — Only base concepts enter residual flow
    5. Intra-level ordering:    Section 3.2       — Concepts ordered by CoT position
    6. Builder-Predictor separation: Section 4   — Builder for groundtruth, Predictor for generation

ENCODER INTEGRATION (hybrid-analysis.md Section 1.2):
    self.encoder is a decoder-only Transformer that extracts CoT features.
    This is analogous to DLCM's encoder: both produce per-token hidden
    states from text, which are then processed by attentive pooling.

    Usage:
        from nlcpV3.config import NLCPV3Config

        config = NLCPV3Config(
            encoder_model_name="Qwen/Qwen2.5-0.5B",
            encoder_freeze=False,  # Set True to freeze encoder
            ...
        )
        builder = ConceptPyramidBuilder(config)
        # Encoder is created internally from config.encoder_model_name
        # builder.encoder_hidden_dim is derived from the loaded model

        # Stage 1: Encode CoT → EncoderOutput
        enc_out = builder.encode_cot(cot_input_ids, attention_mask=cot_mask)
        # enc_out.hidden_states: [B, L, D_encoder]

        # Stage 2a: Build full pyramid → PyramidOutput
        pyramid = builder(enc_out.hidden_states)
        # pyramid.concepts: List[Tensor] — [C_0, ..., C_{K-1}]
        # pyramid.level_outputs: List[LevelOutput] — per-level detail
        # pyramid.reconstructed_hidden: [B, L, D] — for recon loss

        # Stage 2b: Or build one level at a time → SingleLevelOutput
        builder.clear_cache()
        level0 = builder.forward_next_level(enc_out.hidden_states, target_level_index=0)
        # level0.concepts: [B, L_0, D], level0.attention_weights: [B, L_0, L]

DIMENSION FLOW:
    Input:  CoT tokens → encoder → H_CoT [B, L, D_encoder]
            → input_proj → H_proj [B, L, D]
    Output: PyramidOutput (forward) or SingleLevelOutput (forward_next_level)

    Level k processing (captured in LevelOutput):
        H_rest_k:      [B, L, D]          (residual hidden states)
        Q_k:           [L_k, D]           (learnable queries)
        A_k:           [B, L_k, L]        (attention weights)
        C_k_base:      [B, L_k, D]        (base concept — enters residual)
        C_k:           [B, L_k, D]        (refined concept — goes to decoder)
        R_k:           [B, L, D]          (reconstruction from level k)

REFERENCES:
    - hybrid-analysis.md: Full architectural analysis
    - VAR.md Section 5.2.2: Residual decomposition (f_hat + f_rest)
"""

import math
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM

from nlcpV3.config import NLCPV3Config


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
#                 →  SingleLevelOutput (one level, sequential mode)


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

    Attributes:
        hidden_states: Encoder hidden states [B, L, D_encoder]
    """

    hidden_states: torch.Tensor  # [B, L, D_encoder] — H_CoT


@dataclass
class LevelOutput:
    """Per-level intermediate/output data from one pyramid level.

    PRINCIPLE (hybrid-analysis.md Section 2.3, Commit-Refinement Separation):
        Each level produces two concept streams:
        - C_k_base (commit path): enters residual flow f_rest
        - refined_k (refinement path): improves output quality only
        - C_k = C_k_base + refined_k → goes to decoder
        Only base concepts reconstruct H_proj via R_k = A_k^T @ C_k_base.

    PURPOSE:
        Capture all per-level data needed for:
        - External loss computation (Section 5):
          L_reconstruction uses R_k (reconstruction)
          L_ordering uses A_k (attention_weights)
        - Decoder input: concepts (refined)
        - Visualization / debugging

    DIMENSION FLOW (level k):
        concepts:          [B, L_k, D]  — C_k = C_k_base + refined_k
        base_concepts:     [B, L_k, D]  — C_k_base (commit path)
        attention_weights: [B, L_k, L]  — A_k (soft attention)
        reconstruction:    [B, L, D]    — R_k = A_k^T @ C_k_base

    Attributes:
        concepts: Final refined concepts [B, L_k, D]
            C_k = C_k_base + refined_k. This goes to the decoder.
        base_concepts: Base concepts [B, L_k, D]
            C_k_base = level_proj(A_k @ H_rest_k).
            COMMIT path — enters residual flow (Section 2.3).
        attention_weights: Soft attention weights [B, L_k, L]
            A_k = softmax(Q_k @ H_rest_k^T / (sqrt(D) * tau)).
            For ordering loss (Section 5.1.2).
        reconstruction: Reconstruction from base only [B, L, D]
            R_k = A_k^T @ C_k_base.
            Only BASE reconstructs, not refined (Section 2.3).
    """

    concepts: torch.Tensor  # [B, L_k, D]  — C_k (refined, goes to decoder)
    base_concepts: torch.Tensor  # [B, L_k, D]  — C_k_base (commit path)
    attention_weights: torch.Tensor  # [B, L_k, L]  — A_k (soft attention)
    reconstruction: torch.Tensor  # [B, L, D]    — R_k = A_k^T @ C_k_base


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
        projected_hidden:    [B, L, D] — H_proj
        reconstructed_hidden:[B, L, D] — f_hat_K = sum of R_k
        residual_hidden:     [B, L, D] — f_rest_K = H_proj - f_hat_K

    Attributes:
        concepts: Refined concepts per level [C_0, ..., C_{K-1}]
            Each C_k: [B, L_k, D]. Goes to decoder.
        level_outputs: Per-level detailed outputs [LevelOutput_0, ..., LevelOutput_{K-1}]
            Contains base_concepts, attention_weights, reconstruction
            for each level — needed for external loss computation.
        projected_hidden: Projected encoder output [B, L, D]
            H_proj = Linear(H_CoT). The "CoT information to decompose".
        reconstructed_hidden: Accumulated reconstruction [B, L, D]
            f_hat_K = sum_{k=0}^{K-1} R_k. Target for reconstruction loss:
            L_recon = ||f_hat_K - H_proj||^2 (Section 5.1.1).
        residual_hidden: Final residual [B, L, D]
            f_rest_K = H_proj - f_hat_K. Should approach zero for
            exact decomposition (Section 2.1).
        num_levels: Number of levels K
        level_lengths: Concepts per level [L_0, L_1, ..., L_{K-1}]
    """

    concepts: List[torch.Tensor]  # [C_0, ..., C_{K-1}], each [B, L_k, D]
    level_outputs: List[LevelOutput]  # Per-level detailed data
    projected_hidden: torch.Tensor  # [B, L, D] — H_proj
    reconstructed_hidden: torch.Tensor  # [B, L, D] — f_hat_K
    residual_hidden: torch.Tensor  # [B, L, D] — f_rest_K
    num_levels: int  # K
    level_lengths: List[int]  # [L_0, L_1, ..., L_{K-1}]

    @property
    def total_concepts(self) -> int:
        """Total concepts across all levels: sum(L_k) for k=0..K-1."""
        return sum(self.level_lengths)

    @property
    def all_attentions(self) -> List[torch.Tensor]:
        """Convenience: extract attention weights from all levels."""
        return [lo.attention_weights for lo in self.level_outputs]

    @property
    def all_base_concepts(self) -> List[torch.Tensor]:
        """Convenience: extract base concepts from all levels."""
        return [lo.base_concepts for lo in self.level_outputs]

    @property
    def all_reconstructions(self) -> List[torch.Tensor]:
        """Convenience: extract reconstructions from all levels."""
        return [lo.reconstruction for lo in self.level_outputs]

    def cat_concepts(self) -> torch.Tensor:
        """Concatenate all refined concepts: [B, sum(L_k), D].

        PURPOSE: Useful for solution loss (Section 5.1.3) where
            all concepts are pooled to predict the solution.
        """
        return torch.cat(self.concepts, dim=1)  # [B, sum(L_k), D]


@dataclass
class SingleLevelOutput:
    """Output of forward_next_level() — one level at a time.

    PRINCIPLE (hybrid-analysis.md Section 4.1):
        The Builder can extract concepts level by level. Each level k
        depends on previous levels through the residual flow:
        H_rest_k = H_proj - sum_{i<k} R_i.

    PURPOSE:
        Encapsulate the output of a single level extraction for:
        - Sequential level-by-level processing
        - Debugging / visualization of individual levels
        - Curriculum training strategies

    DIMENSION FLOW (level k):
        concepts:          [B, L_k, D]  — C_k (refined)
        base_concepts:     [B, L_k, D]  — C_k_base (commit path)
        attention_weights: [B, L_k, L]  — A_k
        projected_hidden:  [B, L, D]    — H_proj
        level_index:       int          — k

    Attributes:
        concepts: Final refined concepts [B, L_k, D]
            C_k = C_k_base + refined_k.
        base_concepts: Base concepts [B, L_k, D]
            C_k_base (commit path). Cached internally for
            subsequent forward_next_level calls.
        attention_weights: Soft attention weights [B, L_k, L]
            A_k for this level. For ordering loss (Section 5.1.2).
        projected_hidden: Projected encoder output [B, L, D]
            H_proj. Stored for external use (e.g., reconstruction loss).
        level_index: Level index k (0-indexed)
    """

    concepts: torch.Tensor  # [B, L_k, D]  — C_k (refined)
    base_concepts: torch.Tensor  # [B, L_k, D]  — C_k_base (commit path)
    attention_weights: torch.Tensor  # [B, L_k, L]  — A_k
    projected_hidden: torch.Tensor  # [B, L, D]    — H_proj
    level_index: int  # k


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
        forward():             All levels in one pass (training)
        forward_next_level():  One level at a time (sequential / debugging)

    ATTRIBUTES:
        encoder: Decoder-only Transformer for CoT feature extraction
            Internally created from config.encoder_model_name.
            Uses AutoModel (not ForCausalLM) for feature extraction.
            Can be frozen via config.encoder_freeze.
        input_proj: Projection from encoder_dim to hidden_dim
        concept_queries: Learnable queries per level [K levels]
        temperature: Learnable attention temperature
        level_projs: Level-specific output projections
        level_attn: Cross-attention layers for refinement
    """

    def __init__(
        self,
        config: NLCPV3Config,
        use_positional_query_init: bool = True,
    ):
        """Initialize Concept Pyramid Builder.

        PRINCIPLE (hybrid-analysis.md Section 4.1, Section 1.2):
            The Builder extracts groundtruth concepts from CoT. It needs:
            (1) An encoder to produce token-level features H_CoT
            (2) Learnable queries Q_k for each level to attend to H_rest
            (3) Residual flow to decompose H_proj coarse-to-fine
            (4) Cross-attention refinement for context-aware concepts

        PURPOSE:
            Initialize all components needed for concept pyramid extraction,
            including the encoder which is created internally so it can be
            trained end-to-end with the rest of the Builder.

        METHOD:
            - Load pretrained encoder via AutoModel.from_pretrained()
            - Optionally freeze encoder via config.encoder_freeze
            - Derive encoder_hidden_dim from model config
            - Construct projection, queries, attention layers

        Args:
            config: NLCPV3Config with hyperparameters.
                Uses config.encoder_model_name to load the pretrained model.
                Uses config.encoder_freeze to control parameter freezing.
                Uses config.encoder_num_layers for potential layer pruning.
            use_positional_query_init: If True, initialize concept queries
                with positional priors (hybrid-analysis.md Section 6)
        """
        super().__init__()
        self.config = config
        self.use_positional_query_init = use_positional_query_init

        # =================================================================
        # Component 0: Encoder (decoder-only Transformer for CoT)
        # =================================================================
        # PRINCIPLE (hybrid-analysis.md Section 1.2):
        #   H_CoT = Encoder(CoT). The encoder produces token-level
        #   features from CoT, analogous to DLCM's encoder.
        # PURPOSE: Extract per-token hidden states from CoT text.
        # METHOD: Load pretrained model via AutoModel (not ForCausalLM)
        #   so we get hidden states directly. This is the same pattern
        #   as NLCPV3Encoder: AutoModel gives the pure Transformer
        #   backbone without the lm_head, outputting last_hidden_state.
        #   The encoder is created INTERNALLY so it participates in
        #   end-to-end training (unless config.encoder_freeze=True).
        #   H_CoT: [B, L, D_encoder] — last layer hidden states.
        #
        # CRITICAL (from common pitfall):
        #   Use AutoModel (Qwen2Model), NOT AutoModelForCausalLM
        #   (Qwen2ForCausalLM). ForCausalLM adds lm_head for token
        #   prediction which we don't need — we only need features.
        self.encoder = AutoModel.from_pretrained(config.encoder_model_name)
        self.encoder_hidden_dim = self.encoder.config.hidden_size
        # self.encoder_hidden_dim: D_encoder (e.g., 896 for Qwen2.5-0.5B)

        # Freeze encoder if specified in config
        # PURPOSE: When encoder_freeze=True, encoder weights are frozen
        #   and only the Builder's own parameters are trained.
        #   When encoder_freeze=False, the encoder is trained end-to-end
        #   with the concept pyramid extraction, allowing the encoder to
        #   adapt its representations for optimal concept extraction.
        if config.encoder_freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Optional: prune encoder layers if encoder_num_layers is specified
        # PURPOSE: Reduce computation by using fewer Transformer layers.
        # METHOD: Slice the model's layer list to keep only the first
        #   config.encoder_num_layers layers.
        #   encoder_num_layers=-1 means use ALL layers (no pruning).
        if config.encoder_num_layers > 0:
            if hasattr(self.encoder, "layers") and config.encoder_num_layers < len(
                self.encoder.layers
            ):
                self.encoder.layers = self.encoder.layers[: config.encoder_num_layers]
            elif (
                hasattr(self.encoder, "model")
                and hasattr(self.encoder.model, "layers")
                and config.encoder_num_layers < len(self.encoder.model.layers)
            ):
                self.encoder.model.layers = self.encoder.model.layers[
                    : config.encoder_num_layers
                ]

        # =================================================================
        # Component 1: Projection (encoder_dim → concept_dim)
        # =================================================================
        # PRINCIPLE (hybrid-analysis.md Section 1.2):
        #   H_proj = Linear(H_CoT) ∈ ℝ^{B×L×D}
        #   This is the "CoT information to decompose" via residual flow.
        # PURPOSE: Project encoder output to the concept dimension D.
        # METHOD: Linear layer [D_encoder → D].
        #   Input:  [B, L, D_encoder]
        #   Output: [B, L, D]
        self.input_proj = nn.Linear(self.encoder_hidden_dim, config.hidden_dim)

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
                nn.Parameter(torch.randn(length, config.hidden_dim))
                for length in config.level_lengths
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
                nn.Linear(config.hidden_dim, config.hidden_dim)
                for _ in range(config.num_levels)
            ]
        )

        # =================================================================
        # Component 5: Cross-Attention Refinement
        # =================================================================
        # PRINCIPLE (hybrid-analysis.md Section 2.3):
        #   Commit-refinement separation:
        #     C_k_base → enters f_rest (commit path, residual flow)
        #     refined_k → improves output quality only (refinement path)
        #     C_k = C_k_base + refined_k → goes to decoder
        #   Refined concepts do NOT enter residual flow to prevent
        #   double-counting (Section 2.3).
        # PURPOSE: Add context-aware refinement that doesn't pollute f_rest.
        # METHOD: MultiheadAttention for each level k > 0.
        #   Query: expanded_queries [B, L_k, D]
        #   Key/Value: context [B, L + ΣL_i, D] = [H_proj, C_0, ..., C_{k-1}]
        #   Output: refined_k [B, L_k, D]
        self.level_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    batch_first=True,
                )
                for _ in range(config.num_levels)
            ]
        )

        # =================================================================
        # Cache for forward_next_level (level-by-level inference)
        # =================================================================
        # PURPOSE: Store intermediate results needed for sequential
        #   level-by-level concept extraction.
        # METHOD: Lists populated during forward_next_level calls.
        #   _cached_attentions: A_k for each level (for f_rest computation)
        #   _cached_base_concepts: C_k_base for each level (for f_hat)
        self._cached_attentions: List[torch.Tensor] = []
        self._cached_base_concepts: List[torch.Tensor] = []

        # =================================================================
        # Component 6: CoT Reconstruction Decoder (Placeholder)
        # =================================================================
        # PRINCIPLE (hybrid-analysis.md Section 5.1.1, extended):
        #   f_hat_K = Σ_k R_k reconstructs H_proj [B, L, D] in concept space.
        #   To reconstruct CoT tokens, we need to map from concept space
        #   back to token vocabulary: [B, L, D] → [B, L, V].
        #
        #   This is a non-trivial mapping because:
        #   (1) Length: 63 concepts → L tokens (L may be 500+)
        #   (2) Dimension: D (concept) vs D_encoder (token encoder space)
        #   (3) Structure: compressed semantic units → sequential language
        #
        #   The residual flow's attention matrices A_k already handle
        #   length expansion: f_hat_K = Σ_k A_k^T @ C_k_base [B, L, D].
        #   The remaining challenge is dimension conversion (D → D_encoder)
        #   and token prediction (D_encoder → V).
        #
        # TODO: Implement CoT reconstruction from concept pyramid.
        #   Options under consideration:
        #   - f_hat_K + back_proj + pretrained lm_head
        #   - Cross-attention expansion from concepts to positions
        #   - Tied projection (pseudo-inverse of input_proj)
        #
        # PURPOSE: Validate that concept pyramid preserves CoT information
        #   by enabling token-level reconstruction (external loss).
        # METHOD: Currently None (placeholder). Will be set to a module
        #   that maps reconstructed hidden states to CoT token logits.
        #
        # DIMENSION FLOW (future implementation):
        #   Input:  f_hat_K [B, L, D] or concepts [B, ΣL_k, D]
        #   Output: CoT logits [B, L, vocab_size]
        self.recon_decoder: Optional[nn.Module] = None

        self._init_weights()

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
            positional_init_alpha = 0.5  # α: signal strength

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

    def encode_cot(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> EncoderOutput:
        """Encode CoT using the decoder-only Transformer encoder.

        PRINCIPLE (hybrid-analysis.md Section 1.2):
            H_CoT = Encoder(CoT). The encoder produces token-level hidden
            states, analogous to DLCM's encoder. Both extract per-token
            features from text for subsequent attentive pooling.

        PURPOSE:
            Extract token-level features from CoT.

        METHOD:
            Forward pass through self.encoder (HuggingFace or custom).
            Extract last hidden state as H_CoT. Wrap in EncoderOutput.

        DIMENSION FLOW:
            Input:  input_ids [B, L] (token IDs)
                    attention_mask [B, L] (optional, 0=pad, 1=valid)
            Output: EncoderOutput with hidden_states [B, L, D_encoder]

        Args:
            input_ids: Token IDs [B, L]
            attention_mask: Attention mask [B, L] (optional)

        Returns:
            EncoderOutput with hidden_states: [B, L, D_encoder]
        """
        # HuggingFace-style model (e.g., Qwen2.5): has .config or .model
        if hasattr(self.encoder, "model") or hasattr(self.encoder, "config"):
            outputs = self.encoder(
                input_ids=input_ids,  # [B, L]
                attention_mask=attention_mask,  # [B, L]
                output_hidden_states=True,
            )
            # Extract last hidden state: [B, L, D_encoder]
            if hasattr(outputs, "last_hidden_state"):
                hidden = outputs.last_hidden_state  # [B, L, D_encoder]
            elif hasattr(outputs, "hidden_states"):
                hidden = outputs.hidden_states[-1]  # [B, L, D_encoder]
            else:
                hidden = outputs[0]  # [B, L, D_encoder]
        else:
            # Custom encoder — direct forward call
            hidden = self.encoder(input_ids)  # [B, L, D_encoder]

        return EncoderOutput(hidden_states=hidden)  # .hidden_states: [B, L, D_encoder]

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
    ) -> PyramidOutput:
        """Build concept pyramid from CoT hidden states (all levels).

        PRINCIPLE (hybrid-analysis.md Section 4.1):
            The Builder extracts groundtruth concepts level by level using
            soft attention over residual hidden states. Each level k:
            (1) Attends to H_rest_k with learnable queries Q_k
            (2) Extracts base concepts C_k_base (commit path)
            (3) Refines with cross-attention (refinement path)
            (4) Updates residual: H_rest_{k+1} = H_rest_k - R_k

        PURPOSE:
            Extract all K levels of concepts in one forward pass.
            Used during training to build groundtruth concept pyramids.

        METHOD:
            Iterate k=0..K-1, applying soft attention + residual flow
            + cross-attention refinement at each level. Collect all
            per-level data into LevelOutput objects, wrap into PyramidOutput.

        DIMENSION FLOW:
            Input:  encoder_hidden_states [B, L, D_encoder]
            Output: PyramidOutput with concepts, level_outputs, etc.

        Args:
            encoder_hidden_states: CoT hidden states [B, L, D_encoder]
                from self.encoder or pre-computed via encode_cot()

        Returns:
            PyramidOutput containing:
                concepts: [C_0, ..., C_{K-1}], each [B, L_k, D]
                level_outputs: [LevelOutput_0, ..., LevelOutput_{K-1}]
                projected_hidden: [B, L, D]
                reconstructed_hidden: [B, L, D]
                residual_hidden: [B, L, D]
        """
        batch_size, seq_len, _ = encoder_hidden_states.shape
        # batch_size: B, seq_len: L, _: D_encoder

        # =================================================================
        # Step 1: Project encoder hidden states to concept dimension
        # =================================================================
        # PRINCIPLE (hybrid-analysis.md Section 1.2):
        #   H_proj = Linear(H_CoT) — "CoT information to decompose"
        # PURPOSE: Map encoder output to concept space D.
        # METHOD: Linear projection.
        projected_hidden = self.input_proj(encoder_hidden_states)
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
        #   Commit-refinement separation (Section 2.3):
        #     C_k_base → enters f_rest (commit path)
        #     refined_k → does NOT enter f_rest (refinement path)
        #     R_k = A_k^T @ C_k_base — only base reconstructs
        for level_idx in range(self.config.num_levels):
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
                math.sqrt(self.config.hidden_dim) * self.temperature
            )
            # attention_scores: [B, L_k, L] — scaled by √D × τ

            level_attention = F.softmax(attention_scores, dim=-1)
            # level_attention: [B, L_k, L] — A_k, attention weights

            # ── 3c: Extract BASE concepts (commit path) ──────────────
            # PRINCIPLE (Section 2.3, Commit-Refinement Separation):
            #   C_k_base = level_proj(A_k @ H_rest_k)
            #   This is the COMMIT path — enters residual flow.
            #   Only base concepts reconstruct H, ensuring clean f_rest.
            level_concepts_base = torch.bmm(level_attention, residual_hidden)
            # level_attention: [B, L_k, L]
            # residual_hidden: [B, L, D]
            # level_concepts_base: [B, L_k, D] — raw pooled concepts

            level_concepts_base = self.level_projs[level_idx](level_concepts_base)
            # level_concepts_base: [B, L_k, D] — projected base concepts

            # ── 3d: Reconstruct from BASE only ───────────────────────
            # PRINCIPLE (Section 2.3):
            #   R_k = A_k^T @ C_k_base (only BASE, not refined)
            #   This is the VAR f_hat update: f_hat += R_k
            #   Using refined concepts here would double-count context.
            reconstruction = torch.bmm(
                level_attention.transpose(1, 2), level_concepts_base
            )
            # level_attention.T: [B, L, L_k]
            # level_concepts_base: [B, L_k, D]
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

            # ── 3f: Cross-attention refinement ───────────────────────
            # PRINCIPLE (Section 2.3, Refinement Path):
            #   refined_k = CrossAttn(Q_k, context, context)
            #   Context = [H_proj, C_0, ..., C_{k-1}]
            #   refined_k does NOT enter f_rest — only improves output.
            #   C_k = C_k_base + refined_k → goes to decoder
            if level_idx > 0:
                # Build accumulated context
                prev_concepts_cat = torch.cat(all_level_concepts, dim=1)
                # prev_concepts_cat: [B, Σ_{i<k} L_i, D]

                context = torch.cat([projected_hidden, prev_concepts_cat], dim=1)
                # context: [B, L + Σ_{i<k} L_i, D]

                refined_concepts, _ = self.level_attn[level_idx](
                    expanded_queries, context, context
                )
                # expanded_queries: [B, L_k, D] — query
                # context: [B, L + ΣL_i, D] — key/value
                # refined_concepts: [B, L_k, D] — refinement output

                # Refined output: goes to decoder, NOT to residual flow
                level_concepts = level_concepts_base + refined_concepts
                # level_concepts: [B, L_k, D] — final concept for this level
            else:
                # Level 0: no previous concepts, no refinement
                level_concepts = level_concepts_base
                # level_concepts: [B, L_k, D] = C_0_base

            all_level_concepts.append(level_concepts)

            # ── 3g: Collect per-level output ─────────────────────────
            # PURPOSE: Wrap per-level data into LevelOutput for
            #   structured access by external loss computation.
            all_level_outputs.append(
                LevelOutput(
                    concepts=level_concepts,  # [B, L_k, D] — C_k (refined)
                    base_concepts=level_concepts_base,  # [B, L_k, D] — C_k_base (commit)
                    attention_weights=level_attention,  # [B, L_k, L] — A_k
                    reconstruction=reconstruction,  # [B, L, D]   — R_k
                )
            )

        # =================================================================
        # Step 4: Build PyramidOutput
        # =================================================================
        # PURPOSE: Return structured PyramidOutput for external
        #   loss computation (hybrid-analysis.md Section 5).
        return PyramidOutput(
            concepts=all_level_concepts,  # [C_0, ..., C_{K-1}]
            level_outputs=all_level_outputs,  # [LevelOutput_0, ...]
            projected_hidden=projected_hidden,  # [B, L, D] — H_proj
            reconstructed_hidden=reconstructed_accumulator,  # [B, L, D] — f_hat_K
            residual_hidden=residual_hidden,  # [B, L, D] — f_rest_K
            num_levels=self.config.num_levels,  # K
            level_lengths=list(self.config.level_lengths),  # [L_0, ..., L_{K-1}]
        )

    def forward_next_level(
        self,
        encoder_hidden_states: torch.Tensor,
        previous_level_concepts: Optional[List[torch.Tensor]] = None,
        target_level_index: int = 0,
    ) -> SingleLevelOutput:
        """Build concepts for a single level (sequential mode).

        PRINCIPLE (hybrid-analysis.md Section 4.1):
            The Builder extracts concepts level by level. Each level k
            depends on previous levels through the residual flow:
            H_rest_k = H_proj - Σ_{i<k} R_i.
            This method computes one level at a time, using cached
            attention and base concepts from previous calls.

        PURPOSE:
            Extract concepts for a single level. Useful for:
            - Sequential level-by-level processing
            - Debugging / visualization of individual levels
            - Curriculum training strategies

        METHOD:
            1. Project encoder hidden states
            2. Compute residual from cached previous levels
            3. Apply soft attention with queries
            4. Refine with cross-attention
            5. Cache results for subsequent calls
            6. Wrap into SingleLevelOutput

        DIMENSION FLOW:
            Input:
                encoder_hidden_states: [B, L, D_encoder]
                previous_level_concepts: [C_0, ..., C_{k-1}] or None
                target_level_index: int (0-indexed level k)
            Output:
                SingleLevelOutput with concepts [B, L_k, D], etc.

        Args:
            encoder_hidden_states: Hidden states [B, L, D_encoder]
            previous_level_concepts: Previous concepts or None for level 0
            target_level_index: Level to extract (0-indexed)

        Returns:
            SingleLevelOutput with:
                concepts: [B, L_k, D] — refined concepts
                base_concepts: [B, L_k, D] — base concepts (commit path)
                attention_weights: [B, L_k, L] — A_k
                projected_hidden: [B, L, D] — H_proj
                level_index: int — k
        """
        batch_size, seq_len, _ = encoder_hidden_states.shape
        # batch_size: B, seq_len: L, _: D_encoder

        # =================================================================
        # Step 1: Project encoder hidden states
        # =================================================================
        # PRINCIPLE (hybrid-analysis.md Section 1.2):
        #   H_proj = Linear(H_CoT)
        projected_hidden = self.input_proj(encoder_hidden_states)
        # projected_hidden: [B, L, D]

        # =================================================================
        # Step 2: Compute residual from previous levels
        # =================================================================
        # PRINCIPLE (hybrid-analysis.md Section 2.1):
        #   H_rest_k = H_proj - Σ_{i<k} R_i = H_proj - f_hat
        #   Uses cached attentions and base concepts from previous calls.
        #   CRITICAL: only C_k_BASE enters f_rest (Section 2.3).
        if previous_level_concepts is None or len(previous_level_concepts) == 0:
            # Level 0: no previous levels, use full H_proj
            residual_hidden = projected_hidden
            # residual_hidden: [B, L, D] = H_proj
        else:
            # Level k > 0: subtract reconstruction from previous BASE concepts
            reconstructed_hidden = torch.zeros_like(projected_hidden)
            # reconstructed_hidden: [B, L, D] — f_hat accumulator

            for prev_base_concept, prev_attention in zip(
                self._cached_base_concepts, self._cached_attentions
            ):
                # prev_attention: [B, L_prev, L] — A_i
                # prev_base_concept: [B, L_prev, D] — C_i_base (BASE only)
                # Reconstruction: A_i^T @ C_i_base → [B, L, D]
                reconstructed_hidden = reconstructed_hidden + torch.bmm(
                    prev_attention.transpose(1, 2), prev_base_concept
                )
                # reconstructed_hidden: [B, L, D] — accumulating f_hat

            # Residual = H_proj - f_hat (f_rest = "still needs encoding")
            residual_hidden = projected_hidden - reconstructed_hidden
            # residual_hidden: [B, L, D] = H_rest_k

        # =================================================================
        # Step 3: Extract BASE concepts via soft attention
        # =================================================================
        # PRINCIPLE (Section 3.2): A_k = softmax(Q_k @ H_rest_k^T / (√D × τ))
        level_queries = self.concept_queries[target_level_index]
        # level_queries: [L_k, D]

        expanded_queries = level_queries.unsqueeze(0).expand(batch_size, -1, -1)
        # expanded_queries: [B, L_k, D]

        attention_scores = torch.bmm(expanded_queries, residual_hidden.transpose(1, 2))
        # attention_scores: [B, L_k, L]

        attention_scores = attention_scores / (
            math.sqrt(self.config.hidden_dim) * self.temperature
        )
        # attention_scores: [B, L_k, L] — scaled

        level_attention = F.softmax(attention_scores, dim=-1)
        # level_attention: [B, L_k, L] — A_k

        # Cache attention for future residual computation
        if target_level_index >= len(self._cached_attentions):
            self._cached_attentions.append(level_attention.detach())
        else:
            self._cached_attentions[target_level_index] = level_attention.detach()

        # Extract and project base concepts
        level_concepts_base = torch.bmm(level_attention, residual_hidden)
        # level_concepts_base: [B, L_k, D] — raw pooled

        level_concepts_base = self.level_projs[target_level_index](level_concepts_base)
        # level_concepts_base: [B, L_k, D] — projected

        # Cache BASE concepts for future residual computation
        # (only BASE enters f_rest, not refined — Section 2.3)
        if target_level_index >= len(self._cached_base_concepts):
            self._cached_base_concepts.append(level_concepts_base.detach())
        else:
            self._cached_base_concepts[target_level_index] = (
                level_concepts_base.detach()
            )

        # =================================================================
        # Step 4: Cross-attention refinement
        # =================================================================
        # PRINCIPLE (Section 2.3): refined_k does NOT enter f_rest.
        if target_level_index > 0 and previous_level_concepts is not None:
            # Context: [H_proj, C_0, ..., C_{k-1}]
            prev_concepts_cat = torch.cat(previous_level_concepts, dim=1)
            # prev_concepts_cat: [B, Σ_{i<k} L_i, D]

            context = torch.cat([projected_hidden, prev_concepts_cat], dim=1)
            # context: [B, L + Σ_{i<k} L_i, D]

            refined_concepts, _ = self.level_attn[target_level_index](
                expanded_queries, context, context
            )
            # refined_concepts: [B, L_k, D]

            level_concepts = level_concepts_base + refined_concepts
            # level_concepts: [B, L_k, D]
        else:
            level_concepts = level_concepts_base
            # level_concepts: [B, L_k, D] = C_0_base

        # =================================================================
        # Step 5: Build SingleLevelOutput
        # =================================================================
        # PURPOSE: Wrap per-level data into SingleLevelOutput for
        #   structured access by downstream consumers.
        return SingleLevelOutput(
            concepts=level_concepts,  # [B, L_k, D] — C_k (refined)
            base_concepts=level_concepts_base,  # [B, L_k, D] — C_k_base (commit)
            attention_weights=level_attention,  # [B, L_k, L] — A_k
            projected_hidden=projected_hidden,  # [B, L, D]   — H_proj
            level_index=target_level_index,  # k
        )

    def clear_cache(self):
        """Clear cached attentions and base concepts.

        PURPOSE:
            Reset the cache used by forward_next_level.
            Must be called before starting a new sequence of
            forward_next_level calls.
        """
        self._cached_attentions = []
        self._cached_base_concepts = []

    def get_level_config(self, level_idx: int) -> Dict[str, Any]:
        """Get configuration for a specific level.

        PURPOSE: Provide level metadata for external use.

        Args:
            level_idx: Level index k ∈ {0, ..., K-1}

        Returns:
            Dictionary with level configuration
        """
        return {
            "level_idx": level_idx,
            "num_concepts": self.config.level_lengths[level_idx],  # L_k
            "hidden_dim": self.config.hidden_dim,  # D
            "query_shape": tuple(self.concept_queries[level_idx].shape),  # (L_k, D)
        }

    def get_total_concepts(self) -> int:
        """Get total number of concepts across all levels.

        PRINCIPLE (hybrid-analysis.md Section 1.1):
            Total = Σ_{k=0}^{K-1} L_k = 1+2+4+8+16+32 = 63 (for K=6)
        """
        return sum(self.config.level_lengths)
