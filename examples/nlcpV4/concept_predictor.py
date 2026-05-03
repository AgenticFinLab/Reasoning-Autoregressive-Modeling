"""NLCP V4 Concept Predictor — Stage 2: Next-Level Concept Prediction.

PURPOSE (VAR.md Section 5.3, docs/VAR.md):
    Stage 2 of the two-stage architecture. Given a trained (frozen)
    ConceptPyramidBuilder that produces ground truth concept pyramids
    from CoT, the ConceptPredictor learns to predict next-level concepts
    autoregressively — analogous to VAR's Transformer that predicts
    next-scale codebook indices.

TWO-STAGE ARCHITECTURE:
    Stage 1 (Builder):  CoT → ground truth concept pyramid [C_0, ..., C_{K-1}]
                        Trained with recon_loss + reasoning_loss.
                        Frozen during Stage 2 training.

    Stage 2 (Predictor): Q + [C_0, ..., C_{k-1}] → predict C_k
                         Trained with prediction_loss (MSE or cosine).
                         Uses scale-level causal attention.

VAR ANALOGY:
    VAR Stage 1 (VQ-VAE):      Image → multi-scale discrete indices
    VAR Stage 2 (Transformer):  class_emb + prev_scales → predict next scale indices

    NLCP Stage 1 (Builder):     CoT → multi-level concept vectors
    NLCP Stage 2 (Predictor):   Q + prev_levels → predict next level concepts

SCALE-LEVEL CAUSALITY (VAR.md Section 5.3.1):
    Position i can attend to position j iff level[i] >= level[j].
    Within the same level, all concept slots are mutually visible
    (parallel prediction). Across levels, strict causality.

MODEL SELECTION:
    The predictor backbone can be configured via config:
    - use_shared_model: true  → reuse builder's reason_model (shared weights)
    - use_shared_model: false → load a separate model from predictor_model_name

USAGE:
    from nlcpV4.concept_predictor import ConceptPredictor

    # Option A: shared model with builder
    predictor = ConceptPredictor(config, builder=builder)

    # Option B: standalone (own model)
    predictor = ConceptPredictor(config)

    # Training (teacher-forcing with GT concepts from frozen builder)
    predicted_concepts = predictor(
        question_ids=Q_ids,
        question_attention_mask=Q_mask,
        gt_concepts=[C_0, C_1, ..., C_{K-1}],  # from builder
    )

    # Inference (autoregressive, no GT)
    predicted_concepts = predictor.predict(
        question_ids=Q_ids,
        question_attention_mask=Q_mask,
    )

DIMENSION FLOW:
    Input:
        question_ids:     [B, L_Q]           — tokenized question
        gt_concepts:      List of [B, L_k, D] for k=0..K-1 (training only)

    Internal:
        Q_hidden:         [B, L_Q, D_model]  — question hidden states
        concept_sequence: [B, L_total, D]    — flattened concept tokens
        scale_causal_mask:[L_total, L_total] — scale-level causality

    Output:
        predicted:        List of [B, L_k, D] for k=0..K-1
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================================================================
# Output Dataclass
# =========================================================================


@dataclass
class PredictorOutput:
    """Output of ConceptPredictor forward pass.

    DIMENSION FLOW:
        predicted_concepts: List of [B, L_k, D] for k=0..K-1
        prediction_loss:    scalar (if gt_concepts provided)
        per_level_losses:   List of K scalars (per-level breakdown)

    Attributes:
        predicted_concepts: Predicted concept vectors per level.
            Each [B, L_k, D]. During training, these are one-step-ahead
            predictions from teacher-forcing. During inference, these are
            autoregressively generated.
        prediction_loss: Overall prediction loss (MSE between predicted
            and GT concepts). None if gt_concepts not provided.
        per_level_losses: Per-level loss breakdown for diagnostics.
            Empty list if gt_concepts not provided.
    """

    predicted_concepts: List[torch.Tensor]
    prediction_loss: Optional[torch.Tensor] = None
    per_level_losses: List[torch.Tensor] = field(default_factory=list)


# =========================================================================
# Scale-Level Causal Mask
# =========================================================================


def build_scale_causal_mask(
    level_lengths: List[int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build scale-level causal attention mask.

    PRINCIPLE (VAR.md Section 2.4, Section 5.3.1):
        Position i can attend to position j iff level[i] >= level[j].
        Within the same level: full visibility (parallel prediction).
        Across levels: strict causality (only attend to earlier levels).

    Args:
        level_lengths: [L_0, L_1, ..., L_{K-1}] — concepts per level.
        device: Target device.
        dtype: Target dtype (float for additive mask).

    Returns:
        mask: [L_total, L_total] — additive mask where 0 = attend,
            -inf = block. L_total = sum(level_lengths).

    Example (K=3, level_lengths=[1, 2, 4]):
        Level IDs:  [0, 1, 1, 2, 2, 2, 2]
        Mask (1=attend, 0=block):
            pos: 0  1  2  3  4  5  6
          0:  [1  0  0  0  0  0  0]  ← level 0 sees only level 0
          1:  [1  1  1  0  0  0  0]  ← level 1 sees levels 0,1
          2:  [1  1  1  0  0  0  0]
          3:  [1  1  1  1  1  1  1]  ← level 2 sees levels 0,1,2
          4:  [1  1  1  1  1  1  1]
          5:  [1  1  1  1  1  1  1]
          6:  [1  1  1  1  1  1  1]
    """
    total_len = sum(level_lengths)

    # Build level ID for each position
    level_ids = []
    for level_idx, length in enumerate(level_lengths):
        level_ids.extend([level_idx] * length)
    # Level ID tensor: [L_total]
    level_ids = torch.tensor(level_ids, device=device)

    # mask[i, j] = 1 if level_ids[i] >= level_ids[j], else 0
    # level_ids[i] as rows, level_ids[j] as columns
    # row_levels: [L_total, 1], col_levels: [1, L_total]
    row_levels = level_ids.unsqueeze(1)
    col_levels = level_ids.unsqueeze(0)
    # bool_mask: [L_total, L_total]
    bool_mask = row_levels >= col_levels

    # Convert to additive mask: 0 where attend, -inf where block
    mask = torch.zeros(total_len, total_len, device=device, dtype=dtype)
    mask.masked_fill_(~bool_mask, float("-inf"))

    # Result: [L_total, L_total]
    return mask


# =========================================================================
# ConceptPredictor
# =========================================================================


class ConceptPredictor(nn.Module):
    """Stage 2: Predict next-level concepts autoregressively.

    PURPOSE (VAR.md Section 5.3):
        Given a frozen builder that extracts ground truth concept pyramids,
        the predictor learns to generate concept pyramids from questions
        alone — without access to CoT at inference time.

    ARCHITECTURE:
        1. Encode question Q via backbone → Q_hidden [B, L_Q, D_model]
        2. Project Q_hidden to concept space → Q_proj [B, L_Q, D]
        3. During training (teacher-forcing):
           - Concatenate [Q_proj, C_0_gt, C_1_gt, ..., C_{K-2}_gt]
           - Apply Transformer with scale-level causal mask
           - Predict [C_0_pred, C_1_pred, ..., C_{K-1}_pred]
        4. During inference (autoregressive):
           - Start with Q_proj
           - Predict C_0, feed back, predict C_1, ...

    MODEL SELECTION:
        config["model"]["predictor"]["use_shared_model"]:
            true  → reuse builder.reason_model (pass builder= in __init__)
            false → load own model from predictor_model_name

    ATTRIBUTES:
        reason_model: Backbone Transformer (shared or own).
        concept_proj: Projects backbone hidden states → concept space D.
        level_embeddings: Learnable per-level embeddings [K, D] to
            distinguish concept levels (analogous to VAR's level_embed).
        concept_head: Prediction head, Transformer hidden → concept D.
    """

    def __init__(
        self,
        config: dict,
        builder=None,
    ):
        """Initialize Concept Predictor.

        Args:
            config: Full config dict. Expected keys:
                config["model"]["predictor"]:
                    use_shared_model: bool
                    predictor_model_name: str (if use_shared_model=false)
                    predictor_num_layers: int (-1 = all)
                config["model"]["pyramid"]:
                    hidden_dim, num_levels, level_lengths, max_seq_len
                config["training"]["predictor"]:
                    freeze: bool, lora: dict or null
            builder: Optional ConceptPyramidBuilder instance.
                Required when use_shared_model=true.
        """
        super().__init__()
        self.config = config
        self.pyramid_cfg = config["model"]["pyramid"]
        self.predictor_cfg = config["model"]["predictor"]

        num_levels = self.pyramid_cfg["num_levels"]
        concept_dim = self.pyramid_cfg["hidden_dim"]
        level_lengths = self.pyramid_cfg["level_lengths"]

        # =================================================================
        # Component 0: Backbone Model (shared or own)
        # =================================================================
        use_shared = self.predictor_cfg["use_shared_model"]

        if use_shared:
            if builder is None:
                raise ValueError(
                    "builder must be provided when use_shared_model=true. "
                    "Pass the trained ConceptPyramidBuilder instance."
                )
            self.reason_model = builder.reason_model
            self.tokenizer = builder.tokenizer
            self.reason_model_hidden_dim = builder.reason_model_hidden_dim
            self._owns_model = False
        else:
            pred_model_cfg = self.predictor_cfg
            train_pred_cfg = config["training"]["predictor"]
            self.reason_model, self.tokenizer, self.reason_model_hidden_dim = (
                self._init_reason_model(pred_model_cfg, train_pred_cfg)
            )
            self._owns_model = True

        # =================================================================
        # Component 1: Question Projection (D_model → D)
        # =================================================================
        # PRINCIPLE: Map question hidden states to concept space.
        #   Analogous to VAR's Linear(Cvae→D) that projects codebook
        #   embeddings to Transformer hidden dimension.
        self.q_proj = nn.Linear(self.reason_model_hidden_dim, concept_dim)
        self.q_proj_norm = nn.LayerNorm(concept_dim)

        # =================================================================
        # Component 2: Level Embeddings
        # =================================================================
        # PRINCIPLE (VAR.md Section 5.3.2, Step 5):
        #   lvl_emb marks which scale each position belongs to.
        #   This tells the Transformer "I am a level-k concept".
        self.level_embeddings = nn.Embedding(num_levels, concept_dim)

        # =================================================================
        # Component 3: Position Embeddings (within each level)
        # =================================================================
        # PRINCIPLE: Within a level, concepts have positional structure
        #   (coarse-to-fine ordering). Learnable per-position embedding
        #   within each level distinguishes concept slot positions.
        max_concepts_per_level = max(level_lengths)
        self.position_embeddings = nn.Embedding(max_concepts_per_level, concept_dim)

        # =================================================================
        # Component 4: Concept Transformer Blocks
        # =================================================================
        # PRINCIPLE (VAR.md Section 5.3.2, Step 6):
        #   Transformer blocks with scale-level causal attention.
        #   Processes the concept sequence to predict next-level concepts.
        num_heads = self.pyramid_cfg["num_heads"]
        num_predictor_layers = self.predictor_cfg["num_transformer_layers"]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=concept_dim,
            nhead=num_heads,
            dim_feedforward=concept_dim * 4,
            dropout=self.predictor_cfg["dropout"],
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.concept_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_predictor_layers,
        )

        # =================================================================
        # Component 5: Concept Prediction Head
        # =================================================================
        # PRINCIPLE: Project Transformer output to concept vectors.
        #   Each position predicts the concept vector at that slot.
        #   Input dimension = concept_dim (from transformer),
        #   Output dimension = concept_dim (target concept space).
        self.concept_head = nn.Sequential(
            nn.Linear(concept_dim, concept_dim),
            nn.GELU(),
            nn.Linear(concept_dim, concept_dim),
        )

        # =================================================================
        # Component 6: Start-of-Pyramid Token
        # =================================================================
        # PRINCIPLE (VAR.md Section 5.3.2, Step 2):
        #   VAR uses class_emb as the "start token" for scale 0.
        #   We use a learnable start token that aggregates question info.
        self.start_token = nn.Parameter(torch.randn(1, 1, concept_dim))

        # =================================================================
        # Precompute level lengths and offsets
        # =================================================================
        self._level_lengths = list(level_lengths)
        self._total_concepts = sum(level_lengths)
        self._num_levels = num_levels

        # Cache for scale-level causal mask
        self._cached_mask: Optional[torch.Tensor] = None

        self._init_weights()

    def _init_reason_model(self, pred_cfg: dict, train_cfg: dict) -> tuple:
        """Initialize own reason_model (when not sharing with builder).

        Mirrors ConceptPyramidBuilder._init_reason_model() logic.

        Args:
            pred_cfg: config["model"]["predictor"] sub-dict.
            train_cfg: config["training"]["predictor"] sub-dict.

        Returns:
            Tuple of (reason_model, tokenizer, hidden_dim).
        """
        reason_model = AutoModelForCausalLM.from_pretrained(
            pred_cfg["predictor_model_name"]
        )
        hidden_dim = reason_model.config.hidden_size

        tokenizer = AutoTokenizer.from_pretrained(pred_cfg["predictor_model_name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Optional LoRA
        lora_cfg = train_cfg["lora"]
        if lora_cfg is not None:
            lora_config = LoraConfig(
                r=lora_cfg["r"],
                lora_alpha=lora_cfg["lora_alpha"],
                target_modules=lora_cfg["target_modules"],
                lora_dropout=lora_cfg["lora_dropout"],
                bias=lora_cfg["bias"],
            )
            reason_model = get_peft_model(reason_model, lora_config)

        # Freeze if configured
        if train_cfg["freeze"]:
            for param in reason_model.parameters():
                param.requires_grad = False
            if lora_cfg is not None:
                reason_model.enable_adapter_layers()
                for name, param in reason_model.named_parameters():
                    if "lora_" in name:
                        param.requires_grad = True

        # Layer pruning
        num_layers = pred_cfg["predictor_num_layers"]
        if num_layers > 0:
            for obj in [
                reason_model,
                getattr(reason_model, "model", None),
                getattr(getattr(reason_model, "base_model", None), "model", None),
            ]:
                if obj is not None and hasattr(obj, "layers"):
                    if num_layers < len(obj.layers):
                        obj.layers = obj.layers[:num_layers]
                        break

        return reason_model, tokenizer, hidden_dim

    def _get_backbone(self) -> nn.Module:
        """Get Transformer backbone (handles PEFT wrapping).

        Returns:
            The backbone module (e.g., Qwen2Model).
        """
        if hasattr(self.reason_model, "base_model"):
            inner = self.reason_model.base_model
            if hasattr(inner, "model"):
                return inner.model
            return inner
        elif hasattr(self.reason_model, "model"):
            return self.reason_model.model
        else:
            return self.reason_model

    def _init_weights(self):
        """Initialize predictor-specific weights."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.normal_(self.level_embeddings.weight, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, std=0.02)
        nn.init.normal_(self.start_token, std=0.02)

        for module in self.concept_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _get_scale_causal_mask(
        self, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Get or build cached scale-level causal mask.

        Returns:
            mask: [L_total, L_total] additive mask.
        """
        if self._cached_mask is None or self._cached_mask.device != device:
            self._cached_mask = build_scale_causal_mask(
                self._level_lengths, device, dtype
            )
        return self._cached_mask

    def encode_question(
        self,
        question_ids: torch.Tensor,
        question_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode question tokens via backbone.

        DIMENSION FLOW:
            Input:  question_ids [B, L_Q]
            Output: q_hidden [B, L_Q, D_model]

        Args:
            question_ids: Token IDs [B, L_Q].
            question_attention_mask: Mask [B, L_Q] (optional).

        Returns:
            Hidden states [B, L_Q, D_model].
        """
        backbone = self._get_backbone()
        outputs = backbone(
            input_ids=question_ids,
            attention_mask=question_attention_mask,
            output_hidden_states=True,
        )
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            return outputs.hidden_states[-1]
        else:
            return outputs[0]

    def _build_concept_input(
        self,
        q_context: torch.Tensor,
        gt_concepts: Optional[List[torch.Tensor]] = None,
        predicted_so_far: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Build the concept token sequence for the Transformer.

        PRINCIPLE (VAR.md Section 5.3.2, Step 2-4):
            Training (teacher-forcing):
                input = [start_token, C_0_gt, C_1_gt, ..., C_{K-2}_gt]
                target = [C_0_gt, C_1_gt, ..., C_{K-1}_gt]
                (shifted by one level, like VAR's teacher-forcing)

            Inference (autoregressive, partial):
                input = [start_token, C_0_pred, ..., C_{k-1}_pred]
                predict C_k from the output positions for level k.

        Args:
            q_context: Aggregated question context [B, 1, D] or [B, L_Q, D].
            gt_concepts: Ground truth concepts (training). List of K tensors.
            predicted_so_far: Previously predicted concepts (inference).

        Returns:
            Tuple of:
                concept_input: [B, L_input, D] — input to concept transformer.
                q_len: Length of question context prefix.
        """
        batch_size = q_context.shape[0]
        device = q_context.device
        concept_dim = self.pyramid_cfg["hidden_dim"]

        # Start token: [B, 1, D]
        start = self.start_token.expand(batch_size, -1, -1)

        # Build input sequence with level + position embeddings
        # start_token has no level/pos embedding
        parts = [start]

        if gt_concepts is not None:
            # Teacher-forcing: input is [start, C_0, C_1, ..., C_{K-2}]
            # (we don't include C_{K-1} in input since it's the last target)
            concepts_to_feed = gt_concepts[:-1]
        elif predicted_so_far is not None:
            concepts_to_feed = predicted_so_far
        else:
            concepts_to_feed = []

        for level_idx, concepts in enumerate(concepts_to_feed):
            # concepts: [B, L_k, D]
            L_k = concepts.shape[1]

            # Level embedding: [L_k, D]
            lvl_emb = self.level_embeddings(
                torch.full((L_k,), level_idx, device=device, dtype=torch.long)
            )
            # Expand to batch: [B, L_k, D]
            lvl_emb = lvl_emb.unsqueeze(0).expand(batch_size, -1, -1)

            # Position embedding (within level)
            # pos_ids: [L_k], pos_emb: [L_k, D]
            pos_ids = torch.arange(L_k, device=device)
            pos_emb = self.position_embeddings(pos_ids)
            # Expand to batch: [B, L_k, D]
            pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)

            parts.append(concepts + lvl_emb + pos_emb)

        # Concatenated input: [B, L_input, D]
        concept_input = torch.cat(parts, dim=1)
        # q_len = 1 (start token)
        return concept_input, 1

    def forward(
        self,
        question_ids: torch.Tensor,
        question_attention_mask: Optional[torch.Tensor] = None,
        gt_concepts: Optional[List[torch.Tensor]] = None,
    ) -> PredictorOutput:
        """Forward pass: predict concept pyramid.

        PRINCIPLE (VAR.md Section 5.3.2):
            Training (teacher-forcing):
                Input:  [start_token, C_0_gt, ..., C_{K-2}_gt]
                Target: [C_0_gt, C_1_gt, ..., C_{K-1}_gt]
                Loss:   MSE(predicted, target) averaged over all levels.

            Inference (no gt_concepts):
                Autoregressive generation level by level.

        DIMENSION FLOW:
            Input:
                question_ids: [B, L_Q]
                gt_concepts: List of [B, L_k, D] for k=0..K-1 (optional)
            Output:
                PredictorOutput with predicted_concepts, prediction_loss

        Args:
            question_ids: Question token IDs [B, L_Q].
            question_attention_mask: Question mask [B, L_Q] (optional).
            gt_concepts: Ground truth concepts from frozen builder (training).
                If None, runs in inference mode (autoregressive).

        Returns:
            PredictorOutput with predicted concepts and optional loss.
        """
        if gt_concepts is not None:
            return self._forward_training(
                question_ids, question_attention_mask, gt_concepts
            )
        else:
            return self._forward_inference(question_ids, question_attention_mask)

    def _forward_training(
        self,
        question_ids: torch.Tensor,
        question_attention_mask: Optional[torch.Tensor],
        gt_concepts: List[torch.Tensor],
    ) -> PredictorOutput:
        """Training forward with teacher-forcing.

        PRINCIPLE (VAR.md Section 5.3.2):
            Feed ground truth concepts as input (shifted by one level).
            The Transformer predicts the next level at each position.

        DIMENSION FLOW:
            1. Encode Q → q_hidden [B, L_Q, D_model]
            2. Project → q_proj [B, L_Q, D]
            3. Aggregate → q_context [B, 1, D] (mean pool)
            4. Build input: [start, C_0_gt, ..., C_{K-2}_gt]
               Total input length = 1 + sum(L_0..L_{K-2})
            5. Apply concept_transformer with scale-level causal mask
            6. Extract predictions at target positions
            7. Compute MSE loss against GT
        """
        batch_size = question_ids.shape[0]
        device = question_ids.device

        # Step 1-3: Encode question and aggregate
        q_hidden = self.encode_question(question_ids, question_attention_mask)
        # q_hidden: [B, L_Q, D_model]
        # Project and normalize: [B, L_Q, D]
        q_proj = self.q_proj_norm(self.q_proj(q_hidden))

        # Mean-pool question to single context vector
        if question_attention_mask is not None:
            mask_expanded = question_attention_mask.unsqueeze(-1).float()
            q_context = (q_proj * mask_expanded).sum(dim=1, keepdim=True) / (
                mask_expanded.sum(dim=1, keepdim=True).clamp(min=1.0)
            )
        else:
            q_context = q_proj.mean(dim=1, keepdim=True)
        # q_context: [B, 1, D]

        # Inject question context into start token
        start = self.start_token.expand(batch_size, -1, -1) + q_context
        # start: [B, 1, D]

        # Step 4: Build teacher-forcing input
        # Input:  [start, C_0_gt, C_1_gt, ..., C_{K-2}_gt]
        # Target: [C_0_gt, C_1_gt, ..., C_{K-1}_gt]
        parts = [start]
        for level_idx in range(self._num_levels - 1):
            # GT concepts at this level: [B, L_k, D]
            concepts = gt_concepts[level_idx]
            L_k = concepts.shape[1]

            lvl_emb = (
                self.level_embeddings(
                    torch.full((L_k,), level_idx, device=device, dtype=torch.long)
                )
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
            )

            pos_emb = (
                self.position_embeddings(torch.arange(L_k, device=device))
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
            )

            parts.append(concepts + lvl_emb + pos_emb)

        transformer_input = torch.cat(parts, dim=1)
        # transformer_input: [B, 1 + sum(L_0..L_{K-2}), D]

        # Step 5: Build attention mask for the input sequence
        # The input has: [start(1), level_0(L_0), level_1(L_1), ..., level_{K-2}(L_{K-2})]
        # We need scale-level causality over the concept part.
        # Start token is visible to all (like VAR's class token at scale -1).
        input_len = transformer_input.shape[1]
        attn_mask = torch.zeros(
            input_len, input_len, device=device, dtype=transformer_input.dtype
        )

        # Build level assignment for each position in the input
        # Start token: level -1 (visible to all)
        # C_0 tokens: level 0, C_1 tokens: level 1, ...
        # Start token gets level -1 (visible to all)
        input_level_ids = [-1]
        for level_idx in range(self._num_levels - 1):
            L_k = self._level_lengths[level_idx]
            input_level_ids.extend([level_idx] * L_k)
        input_level_ids = torch.tensor(input_level_ids, device=device)

        # Position i can attend to j iff level[i] >= level[j]
        # row_levels: [L_input, 1], col_levels: [1, L_input]
        row_levels = input_level_ids.unsqueeze(1)
        col_levels = input_level_ids.unsqueeze(0)
        # can_attend: [L_input, L_input]
        can_attend = row_levels >= col_levels
        attn_mask.masked_fill_(~can_attend, float("-inf"))

        # Step 6: Forward through concept transformer
        transformer_output = self.concept_transformer(transformer_input, mask=attn_mask)
        # transformer_output: [B, L_input, D]

        # Step 7: Extract predictions at target positions and compute loss
        # The output at position range for level k predicts level k concepts.
        # Mapping: start(1) predicts C_0, C_0 positions predict C_1, etc.
        predicted_concepts = []
        per_level_losses = []
        total_loss = torch.tensor(0.0, device=device)

        # Position 0 (start token output) → predicts C_0
        # Positions [1, 1+L_0) (C_0 output) → predicts C_1
        # Positions [1+L_0, 1+L_0+L_1) (C_1 output) → predicts C_2
        # etc.
        offset = 0
        for level_idx in range(self._num_levels):
            # Target for this level: [B, L_k, D]
            target = gt_concepts[level_idx]
            L_k = target.shape[1]

            # Extract transformer output for the prediction positions
            if level_idx == 0:
                # Start token output → expand to L_0 predictions
                # [B, 1, D] → [B, L_k, D]
                pred_hidden = transformer_output[:, 0:1, :]
                pred_hidden = pred_hidden.expand(-1, L_k, -1)
            else:
                # Previous level's output positions → predict this level
                prev_start = 1 + sum(self._level_lengths[: level_idx - 1])
                prev_end = prev_start + self._level_lengths[level_idx - 1]
                pred_hidden = transformer_output[:, prev_start:prev_end, :]
                # pred_hidden: [B, L_{k-1}, D]

                # Pool or project to match target length L_k
                # Use linear interpolation to go from L_{k-1} → L_k
                # Interpolate: [B, L_{k-1}, D] → [B, D, L_{k-1}] → [B, D, L_k] → [B, L_k, D]
                if pred_hidden.shape[1] != L_k:
                    pred_hidden = pred_hidden.transpose(1, 2)
                    pred_hidden = F.interpolate(
                        pred_hidden, size=L_k, mode="linear", align_corners=False
                    )
                    pred_hidden = pred_hidden.transpose(1, 2)

            # Apply prediction head
            # Apply prediction head: [B, L_k, D]
            predicted = self.concept_head(pred_hidden)

            # Add target-level embedding to prediction
            target_lvl_emb = (
                self.level_embeddings(
                    torch.full((L_k,), level_idx, device=device, dtype=torch.long)
                )
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
            )
            target_pos_emb = (
                self.position_embeddings(torch.arange(L_k, device=device))
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
            )

            predicted = predicted + target_lvl_emb + target_pos_emb

            predicted_concepts.append(predicted)

            # Per-level MSE loss
            level_loss = F.mse_loss(predicted, target.detach())
            per_level_losses.append(level_loss)
            total_loss = total_loss + level_loss

        # Average loss across levels
        total_loss = total_loss / self._num_levels

        return PredictorOutput(
            predicted_concepts=predicted_concepts,
            prediction_loss=total_loss,
            per_level_losses=per_level_losses,
        )

    def _forward_inference(
        self,
        question_ids: torch.Tensor,
        question_attention_mask: Optional[torch.Tensor],
    ) -> PredictorOutput:
        """Inference forward: autoregressive level-by-level generation.

        PRINCIPLE (VAR.md Section 6 — Inference):
            Without ground truth, generate concepts level by level.
            Each level uses all previously predicted levels as context.

        DIMENSION FLOW:
            For each level k:
                Input:  [start, C_0_pred, ..., C_{k-1}_pred]
                Output: C_k_pred [B, L_k, D]
        """
        batch_size = question_ids.shape[0]
        device = question_ids.device

        # Encode question
        q_hidden = self.encode_question(question_ids, question_attention_mask)
        q_proj = self.q_proj_norm(self.q_proj(q_hidden))

        if question_attention_mask is not None:
            mask_expanded = question_attention_mask.unsqueeze(-1).float()
            q_context = (q_proj * mask_expanded).sum(dim=1, keepdim=True) / (
                mask_expanded.sum(dim=1, keepdim=True).clamp(min=1.0)
            )
        else:
            q_context = q_proj.mean(dim=1, keepdim=True)

        start = self.start_token.expand(batch_size, -1, -1) + q_context

        predicted_concepts = []

        for level_idx in range(self._num_levels):
            L_k = self._level_lengths[level_idx]

            # Build input: [start, C_0_pred, ..., C_{k-1}_pred]
            parts = [start]
            for prev_idx, prev_concepts in enumerate(predicted_concepts):
                L_prev = prev_concepts.shape[1]
                lvl_emb = (
                    self.level_embeddings(
                        torch.full((L_prev,), prev_idx, device=device, dtype=torch.long)
                    )
                    .unsqueeze(0)
                    .expand(batch_size, -1, -1)
                )
                pos_emb = (
                    self.position_embeddings(torch.arange(L_prev, device=device))
                    .unsqueeze(0)
                    .expand(batch_size, -1, -1)
                )
                parts.append(prev_concepts + lvl_emb + pos_emb)

            transformer_input = torch.cat(parts, dim=1)

            # Build causal mask for current input
            input_len = transformer_input.shape[1]
            input_level_ids = [-1]
            for prev_idx in range(level_idx):
                input_level_ids.extend([prev_idx] * self._level_lengths[prev_idx])
            input_level_ids = torch.tensor(input_level_ids, device=device)
            row_levels = input_level_ids.unsqueeze(1)
            col_levels = input_level_ids.unsqueeze(0)
            can_attend = row_levels >= col_levels
            attn_mask = torch.zeros(
                input_len,
                input_len,
                device=device,
                dtype=transformer_input.dtype,
            )
            attn_mask.masked_fill_(~can_attend, float("-inf"))

            # Forward
            transformer_output = self.concept_transformer(
                transformer_input, mask=attn_mask
            )

            # Extract prediction from the last positions
            if level_idx == 0:
                pred_hidden = transformer_output[:, 0:1, :].expand(-1, L_k, -1)
            else:
                prev_L = self._level_lengths[level_idx - 1]
                pred_hidden = transformer_output[:, -prev_L:, :]
                if pred_hidden.shape[1] != L_k:
                    pred_hidden = pred_hidden.transpose(1, 2)
                    pred_hidden = F.interpolate(
                        pred_hidden, size=L_k, mode="linear", align_corners=False
                    )
                    pred_hidden = pred_hidden.transpose(1, 2)

            predicted = self.concept_head(pred_hidden)

            # Add level/position embeddings
            target_lvl_emb = (
                self.level_embeddings(
                    torch.full((L_k,), level_idx, device=device, dtype=torch.long)
                )
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
            )
            target_pos_emb = (
                self.position_embeddings(torch.arange(L_k, device=device))
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
            )
            predicted = predicted + target_lvl_emb + target_pos_emb

            predicted_concepts.append(predicted)

        return PredictorOutput(predicted_concepts=predicted_concepts)
