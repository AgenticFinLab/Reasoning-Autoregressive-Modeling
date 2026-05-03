"""NLCP V4 Concept Predictor — Stage 2: Next-Level Concept Prediction.

PURPOSE (VAR.md Section 5.3, docs/VAR.md):
    Stage 2 of the two-stage architecture. Given a trained (frozen)
    ConceptPyramidBuilder that produces ground-truth concept pyramids
    from CoT, the ConceptPredictor learns to predict concepts
    autoregressively — analogous to VAR's Transformer that predicts
    next-scale codebook indices.

TWO-STAGE ARCHITECTURE:
    Stage 1 (Builder):   CoT -> GT pyramid [C_0, ..., C_{K-1}]
                         Trained with recon + reasoning + ordering losses.
                         Frozen (or LoRA) during Stage 2 training.

    Stage 2 (Predictor): Q + [C_0, ..., C_{k-1}] -> predict C_k
                         Trained with two losses (computed in losses.py):
                           (1) concept reconstruction loss  (MSE vs GT)
                           (2) solution cross-entropy loss  (NTP on S)

VAR-FAITHFUL TEACHER-FORCING LAYOUT:
    The sequence fed to the Transformer contains ONE position group per
    level, with L_k positions for level k. Total length == sum(L_k).

        group 0 (L_0 positions):  start_token + q_context         (level 0 input)
        group 1 (L_1 positions):  upsample(C_0_gt,  L_1)          (level 1 input)
        group 2 (L_2 positions):  upsample(C_1_gt,  L_2)          (level 2 input)
        ...
        group K-1 (L_{K-1} pos):  upsample(C_{K-2}_gt, L_{K-1})   (level K-1 input)

    Each position also receives a level embedding (lvl_emb[k]) and a
    within-level position embedding (pos_emb[j]).

    Scale-level causal mask: position i (level a) attends to position j
    (level b) iff  a >= b  (strict inter-level causality; full intra-level
    visibility). This matches VAR Section 2.4 / Section 5.3.1.

    Output at group k's L_k positions --concept_head--> predicted C_k.
    Shapes naturally align with GT targets without post-hoc reshape.

MODEL SELECTION:
    config["model"]["predictor"]["use_shared_model"]:
        true  -> reuse builder.reason_model (+ builder.back_proj)
        false -> load own model from predictor_model_name (+ own back_proj)

USAGE:
    from nlcpV4.concept_predictor import ConceptPredictor

    # Shared-backbone mode (recommended for Stage 2 training)
    predictor = ConceptPredictor(config, builder=builder)

    # Training (teacher-forcing + reasoning loss)
    out = predictor(
        question_ids=Q_ids,
        question_attention_mask=Q_mask,
        gt_concepts=[C_0, ..., C_{K-1}],      # from frozen builder
        solution_ids=S_ids,                    # optional, for reasoning CE
        solution_attention_mask=S_mask,
    )

    from nlcpV4.losses import compute_predictor_loss
    total_loss, loss_dict = compute_predictor_loss(out, loss_weights)

    # Inference (autoregressive, no GT)
    out = predictor(question_ids=Q_ids, question_attention_mask=Q_mask)

DIMENSION FLOW:
    Input:
        question_ids:     [B, L_Q]
        gt_concepts:      list of [B, L_k, D] for k=0..K-1 (training)
        solution_ids:     [B, L_S]                           (optional)

    Internal:
        q_hidden:         [B, L_Q, D_model]
        q_context:        [B, 1, D]           (mean-pooled, masked)
        transformer_input:[B, sum(L_k), D]
        scale_causal_mask:[sum(L_k), sum(L_k)]

    Output (PredictorOutput):
        predicted_concepts:   list of [B, L_k, D]   (one per level)
        gt_concepts:          list of [B, L_k, D]   (pass-through)
        reasoning_logits:     [B, L_S, V]           (if solution given)
        reasoning_target_ids: [B, L_S]              (if solution given)
        reasoning_texts:      list[str]             (argmax decode)
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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
    """Full output of ConceptPredictor.forward().

    PURPOSE:
        Carry all tensors required by losses.py to compute the two
        predictor losses:
          (1) concept reconstruction MSE (predicted_concepts vs gt_concepts)
          (2) solution cross-entropy    (reasoning_logits vs reasoning_target_ids)

    DIMENSION FLOW:
        predicted_concepts:   List of K tensors, each [B, L_k, D]
        gt_concepts:          List of K tensors, each [B, L_k, D]
        reasoning_logits:     [B, L_S, V]
        reasoning_target_ids: [B, L_S]  (padding positions set to -100)
        reasoning_texts:      list of B strings

    Attributes:
        predicted_concepts: Predicted concept vectors per level.
            Training (teacher-forcing): one-step predictions aligned with
            each level's input slot. Inference: autoregressively generated.
        gt_concepts: Ground-truth concepts from the (frozen) builder,
            passed through unchanged. None during inference.
        num_levels: Number of pyramid levels K.
        level_lengths: [L_0, ..., L_{K-1}].
        reasoning_logits: NTP logits for solution tokens, computed from
            [Q_embeds, back_proj(predicted_concepts), S_embeds]. None if
            solution_ids was not provided.
        reasoning_target_ids: Ground-truth solution token IDs with padding
            positions set to -100 so cross_entropy can ignore them.
        reasoning_texts: Teacher-forced argmax decoding of reasoning_logits
            (B strings), useful for qualitative diagnostics.
    """

    predicted_concepts: List[torch.Tensor]
    gt_concepts: Optional[List[torch.Tensor]] = None
    num_levels: int = 0
    level_lengths: List[int] = field(default_factory=list)
    reasoning_logits: Optional[torch.Tensor] = None
    reasoning_target_ids: Optional[torch.Tensor] = None
    reasoning_texts: Optional[List[str]] = None


# =========================================================================
# Scale-Level Causal Mask
# =========================================================================


def build_scale_causal_mask(
    level_lengths: List[int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a scale-level causal attention mask.

    PRINCIPLE (VAR.md Section 2.4 / Section 5.3.1):
        Position i can attend to position j iff level[i] >= level[j].
        - Within the same level: full visibility (parallel prediction).
        - Across levels: strict causality (only earlier levels).

    Args:
        level_lengths: [L_0, L_1, ..., L_{K-1}] concepts per level.
        device: target device.
        dtype: float dtype for the additive mask.

    Returns:
        mask: [L_total, L_total] additive mask (0 = attend, -inf = block).

    Example (K=3, level_lengths=[1, 2, 4]):
        Level IDs:  [0, 1, 1, 2, 2, 2, 2]
        Mask (1=attend, 0=block):
            pos: 0  1  2  3  4  5  6
          0:  [1  0  0  0  0  0  0]  <- level 0 sees only level 0
          1:  [1  1  1  0  0  0  0]  <- level 1 sees levels 0,1
          2:  [1  1  1  0  0  0  0]
          3:  [1  1  1  1  1  1  1]  <- level 2 sees levels 0,1,2
          4:  [1  1  1  1  1  1  1]
          5:  [1  1  1  1  1  1  1]
          6:  [1  1  1  1  1  1  1]
    """
    total_len = sum(level_lengths)

    # Build level ID for each position.
    level_ids: List[int] = []
    for level_idx, length in enumerate(level_lengths):
        level_ids.extend([level_idx] * length)
    # level_ids_t: [L_total]
    level_ids_t = torch.tensor(level_ids, device=device)

    # mask[i, j] = 1 iff level_ids[i] >= level_ids[j].
    # row_levels: [L_total, 1], col_levels: [1, L_total]
    row_levels = level_ids_t.unsqueeze(1)
    col_levels = level_ids_t.unsqueeze(0)
    # bool_mask: [L_total, L_total]
    bool_mask = row_levels >= col_levels

    # Additive mask: 0 where attend, -inf where block.
    mask = torch.zeros(total_len, total_len, device=device, dtype=dtype)
    mask.masked_fill_(~bool_mask, float("-inf"))
    return mask


# =========================================================================
# ConceptPredictor
# =========================================================================


class ConceptPredictor(nn.Module):
    """Stage 2: predict concept pyramids autoregressively from Q.

    PURPOSE (VAR.md Section 5.3):
        Given a frozen builder that extracts ground-truth concept
        pyramids from CoT, the predictor learns to generate concept
        pyramids from questions alone (no CoT at inference time).

    ARCHITECTURE:
        1. Encode Q via backbone                 -> q_hidden [B, L_Q, D_model]
        2. Project to concept space              -> q_proj   [B, L_Q, D]
        3. Mean-pool (masked) to q_context       -> q_context[B, 1, D]
        4. Teacher-forcing input (see module docstring):
             group 0: start + q_context                   -> L_0 positions
             group k: upsample(C_{k-1}_gt, L_k)           -> L_k positions
           Add level + intra-level position embeddings.
        5. Transformer with scale-level causal mask.
        6. concept_head on each group's L_k outputs -> predicted C_k.
        7. (Optional) reasoning: back_proj(cat(predicted)) + Q_embeds + S_embeds
           -> reason_model(inputs_embeds) -> logits aligned to S_ids.

    ATTRIBUTES:
        reason_model:      backbone + lm_head (shared or own).
        tokenizer:         tokenizer paired with reason_model.
        q_proj:            D_model -> D projection for question hidden.
        q_proj_norm:       LayerNorm on q_proj output.
        level_embeddings:  [K, D] per-level marker.
        position_embeddings: [max(L_k), D] intra-level position marker.
        start_token:       learnable [1, 1, D] seed for level 0 input.
        concept_transformer: nn.TransformerEncoder with scale-causal mask.
        concept_head:      per-position MLP D -> D for concept prediction.
        back_proj:         D -> D_encoder map for reasoning loss
                           (shared with builder if use_shared_model=True).
    """

    # --------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------

    def __init__(self, config: dict, builder=None):
        """Initialize ConceptPredictor.

        Args:
            config: full config dict. See module docstring.
            builder: optional ConceptPyramidBuilder. REQUIRED when
                config["model"]["predictor"]["use_shared_model"] is True.
                When present, its reason_model, tokenizer, and back_proj
                are shared (weight-tied) with this predictor.
        """
        super().__init__()
        self.config = config
        self.pyramid_cfg = config["model"]["pyramid"]
        self.predictor_cfg = config["model"]["predictor"]

        num_levels = self.pyramid_cfg["num_levels"]
        concept_dim = self.pyramid_cfg["hidden_dim"]
        level_lengths = self.pyramid_cfg["level_lengths"]

        # ----------------------------------------------------------------
        # Component 0: Backbone model (shared or own)
        # ----------------------------------------------------------------
        use_shared = self.predictor_cfg["use_shared_model"]
        if use_shared:
            if builder is None:
                raise ValueError(
                    "builder must be provided when use_shared_model=True. "
                    "Pass the trained ConceptPyramidBuilder instance."
                )
            self.reason_model = builder.reason_model
            self.tokenizer = builder.tokenizer
            self.reason_model_hidden_dim = builder.reason_model_hidden_dim
            self._owns_model = False
        else:
            self.reason_model, self.tokenizer, self.reason_model_hidden_dim = (
                self._init_reason_model(
                    self.predictor_cfg, config["training"]["predictor"]
                )
            )
            self._owns_model = True

        # ----------------------------------------------------------------
        # Component 1: Question projection (D_model -> D)
        # ----------------------------------------------------------------
        self.q_proj = nn.Linear(self.reason_model_hidden_dim, concept_dim)
        self.q_proj_norm = nn.LayerNorm(concept_dim)

        # ----------------------------------------------------------------
        # Component 2: Level embeddings [K, D]
        # ----------------------------------------------------------------
        # PRINCIPLE (VAR.md Section 5.3.2): marks "this is a level-k slot".
        self.level_embeddings = nn.Embedding(num_levels, concept_dim)

        # ----------------------------------------------------------------
        # Component 3: Intra-level position embeddings
        # ----------------------------------------------------------------
        # PRINCIPLE: within a level, concept slots have a coarse-to-fine
        #   ordering. Max size equals the largest L_k (e.g. 32).
        max_concepts_per_level = max(level_lengths)
        self.position_embeddings = nn.Embedding(max_concepts_per_level, concept_dim)

        # ----------------------------------------------------------------
        # Component 4: Concept Transformer
        # ----------------------------------------------------------------
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
            encoder_layer, num_layers=num_predictor_layers
        )

        # ----------------------------------------------------------------
        # Component 5: Concept prediction head (D -> D)
        # ----------------------------------------------------------------
        self.concept_head = nn.Sequential(
            nn.Linear(concept_dim, concept_dim),
            nn.GELU(),
            nn.Linear(concept_dim, concept_dim),
        )

        # ----------------------------------------------------------------
        # Component 6: Start-of-pyramid token
        # ----------------------------------------------------------------
        # PRINCIPLE (VAR.md 5.3.2 Step 2): analogue to VAR's class_emb.
        self.start_token = nn.Parameter(torch.randn(1, 1, concept_dim))

        # ----------------------------------------------------------------
        # Component 7: back_proj (D -> D_encoder) for reasoning loss
        # ----------------------------------------------------------------
        # PRINCIPLE: the reason_model runs in D_encoder space. To feed
        #   predicted concepts (in D space) into it for NTP / reasoning
        #   CE loss, map them back with back_proj.
        # SHARING: when builder is provided, reuse its back_proj so the
        #   predictor's concepts live in the same encoder-space basis the
        #   builder has already learned.
        if (
            use_shared
            and builder is not None
            and getattr(builder, "back_proj", None) is not None
        ):
            self.back_proj = builder.back_proj
            self._owns_back_proj = False
        else:
            self.back_proj = nn.Linear(
                concept_dim, self.reason_model_hidden_dim, bias=False
            )
            self._owns_back_proj = True

        # ----------------------------------------------------------------
        # Cached precomputations
        # ----------------------------------------------------------------
        self._level_lengths = list(level_lengths)
        self._total_concepts = sum(level_lengths)
        self._num_levels = num_levels
        # Cache for scale-causal mask keyed by (device, dtype).
        self._cached_mask: Optional[torch.Tensor] = None

        self._init_weights()

    # --------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------

    def _init_reason_model(self, pred_cfg: dict, train_cfg: dict) -> tuple:
        """Initialize own reason_model when not sharing with a builder."""
        reason_model = AutoModelForCausalLM.from_pretrained(
            pred_cfg["predictor_model_name"]
        )
        hidden_dim = reason_model.config.hidden_size

        tokenizer = AutoTokenizer.from_pretrained(pred_cfg["predictor_model_name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

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

        if train_cfg["freeze"]:
            for param in reason_model.parameters():
                param.requires_grad = False
            if lora_cfg is not None:
                reason_model.enable_adapter_layers()
                for name, param in reason_model.named_parameters():
                    if "lora_" in name:
                        param.requires_grad = True

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
        """Return the Transformer backbone, handling PEFT wrapping."""
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

        # Only re-init back_proj when we own it (otherwise it is shared
        # with the builder and already has its own initialization).
        if self._owns_back_proj:
            nn.init.xavier_uniform_(self.back_proj.weight)

    def _get_scale_causal_mask(
        self, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Return cached scale-level causal mask, rebuilding on device/dtype change."""
        if (
            self._cached_mask is None
            or self._cached_mask.device != device
            or self._cached_mask.dtype != dtype
        ):
            self._cached_mask = build_scale_causal_mask(
                self._level_lengths, device, dtype
            )
        return self._cached_mask

    # --------------------------------------------------------------------
    # Question encoding
    # --------------------------------------------------------------------

    def encode_question(
        self,
        question_ids: torch.Tensor,
        question_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode question tokens via the backbone.

        DIMENSION FLOW:
            Input:  question_ids [B, L_Q]
            Output: q_hidden     [B, L_Q, D_model]
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

    def _compute_q_context(
        self,
        question_ids: torch.Tensor,
        question_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode Q -> project -> masked mean-pool to a single context vector.

        Returns:
            q_context: [B, 1, D] in concept space.
        """
        q_hidden = self.encode_question(question_ids, question_attention_mask)
        q_proj = self.q_proj_norm(self.q_proj(q_hidden))
        # q_proj: [B, L_Q, D]
        if question_attention_mask is not None:
            mask_exp = question_attention_mask.unsqueeze(-1).to(q_proj.dtype)
            q_context = (q_proj * mask_exp).sum(dim=1, keepdim=True) / (
                mask_exp.sum(dim=1, keepdim=True).clamp(min=1.0)
            )
        else:
            q_context = q_proj.mean(dim=1, keepdim=True)
        # q_context: [B, 1, D]
        return q_context

    # --------------------------------------------------------------------
    # Per-level input-group builders
    # --------------------------------------------------------------------

    def _level0_input(
        self, q_context: torch.Tensor, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Build the level-0 input group: start_token + q_context + lvl0/pos embeddings.

        Returns:
            [B, L_0, D]
        """
        L_0 = self._level_lengths[0]
        # start + q_context, broadcast to L_0 slots.
        start = self.start_token.expand(batch_size, L_0, -1) + q_context
        lvl = self.level_embeddings(
            torch.zeros(L_0, device=device, dtype=torch.long)
        ).unsqueeze(0)
        pos = self.position_embeddings(torch.arange(L_0, device=device)).unsqueeze(0)
        return start + lvl + pos

    def _upsample_prev_to_level(
        self, prev_concepts: torch.Tensor, level_idx: int, device: torch.device
    ) -> torch.Tensor:
        """Upsample previous level's concepts to level `level_idx`'s length.

        Args:
            prev_concepts: [B, L_{k-1}, D] — C_{k-1} (GT or predicted).
            level_idx:     k >= 1.

        Returns:
            [B, L_k, D] — upsampled + lvl_k + pos_k embeddings added.
        """
        L_k = self._level_lengths[level_idx]
        L_prev = prev_concepts.shape[1]
        if L_prev != L_k:
            # F.interpolate expects [N, C, L] — transpose to [B, D, L_prev].
            up = prev_concepts.transpose(1, 2)
            up = F.interpolate(up, size=L_k, mode="linear", align_corners=False)
            up = up.transpose(1, 2)
        else:
            up = prev_concepts
        # up: [B, L_k, D]

        lvl = self.level_embeddings(
            torch.full((L_k,), level_idx, device=device, dtype=torch.long)
        ).unsqueeze(0)
        pos = self.position_embeddings(torch.arange(L_k, device=device)).unsqueeze(0)
        return up + lvl + pos

    # --------------------------------------------------------------------
    # Forward
    # --------------------------------------------------------------------

    def forward(
        self,
        question_ids: torch.Tensor,
        question_attention_mask: Optional[torch.Tensor] = None,
        gt_concepts: Optional[List[torch.Tensor]] = None,
        solution_ids: Optional[torch.Tensor] = None,
        solution_attention_mask: Optional[torch.Tensor] = None,
    ) -> PredictorOutput:
        """Predict concept pyramid from Q (training or inference).

        TRAINING (gt_concepts provided):
            VAR-faithful teacher-forcing in a single Transformer pass.
            Outputs at each level's L_k positions directly match targets.

        INFERENCE (gt_concepts is None):
            Autoregressive level-by-level generation (K passes).

        When solution_ids is provided, reasoning logits are also computed
        (teacher-forced NTP on [Q, back_proj(predicted), S]).

        Args:
            question_ids:             [B, L_Q] question token IDs.
            question_attention_mask:  [B, L_Q] (optional, 1=valid).
            gt_concepts:              training-only list of [B, L_k, D].
            solution_ids:             [B, L_S] solution token IDs
                                       (optional; enables reasoning loss).
            solution_attention_mask:  [B, L_S] (required iff solution_ids).

        Returns:
            PredictorOutput — see class docstring.
        """
        if gt_concepts is not None:
            out = self._forward_training(
                question_ids, question_attention_mask, gt_concepts
            )
        else:
            out = self._forward_inference(question_ids, question_attention_mask)

        if solution_ids is not None:
            if solution_attention_mask is None:
                raise ValueError(
                    "solution_attention_mask is required when solution_ids is provided."
                )
            self._prepare_reasoning(
                out,
                question_ids,
                question_attention_mask,
                solution_ids,
                solution_attention_mask,
            )
        return out

    # --------------------------------------------------------------------
    # Training forward: one-pass teacher forcing
    # --------------------------------------------------------------------

    def _forward_training(
        self,
        question_ids: torch.Tensor,
        question_attention_mask: Optional[torch.Tensor],
        gt_concepts: List[torch.Tensor],
    ) -> PredictorOutput:
        """Single-pass teacher-forcing forward.

        PIPELINE:
            1. q_context = masked mean-pool of projected Q.
            2. Build one input group per level:
                 level 0: start + q_context      (L_0 positions)
                 level k: upsample(C_{k-1}_gt, L_k) (L_k positions, k>=1)
               Add level + position embeddings to every slot.
            3. Concatenate all groups -> transformer_input [B, sum(L_k), D].
            4. Apply concept_transformer with scale-level causal mask.
            5. For each level k, apply concept_head to the level-k
               output slice -> predicted C_k [B, L_k, D].

        Returns:
            PredictorOutput with predicted_concepts and gt_concepts set.
            reasoning_* fields are None here (added by _prepare_reasoning
            when caller supplies solution tokens).
        """
        if len(gt_concepts) != self._num_levels:
            raise ValueError(
                f"gt_concepts has {len(gt_concepts)} levels, "
                f"expected {self._num_levels}."
            )

        batch_size = question_ids.shape[0]
        device = question_ids.device

        # 1. Question context (single vector per example).
        q_context = self._compute_q_context(question_ids, question_attention_mask)
        # q_context: [B, 1, D]

        # 2. Build per-level input groups.
        parts: List[torch.Tensor] = [self._level0_input(q_context, batch_size, device)]
        for k in range(1, self._num_levels):
            # Teacher forcing: feed GT from previous level.
            parts.append(self._upsample_prev_to_level(gt_concepts[k - 1], k, device))

        # 3. Concatenate: [B, sum(L_k), D].
        transformer_input = torch.cat(parts, dim=1)

        # 4. Scale-level causal mask (additive float, matching input dtype).
        attn_mask = self._get_scale_causal_mask(device, transformer_input.dtype)

        # 5. Transformer pass.
        transformer_output = self.concept_transformer(transformer_input, mask=attn_mask)
        # transformer_output: [B, sum(L_k), D]

        # 6. Extract per-level predictions via concept_head.
        predicted_concepts: List[torch.Tensor] = []
        offset = 0
        for k in range(self._num_levels):
            L_k = self._level_lengths[k]
            # Slice for level k: [B, L_k, D]
            hidden_k = transformer_output[:, offset : offset + L_k, :]
            predicted_concepts.append(self.concept_head(hidden_k))
            offset += L_k

        return PredictorOutput(
            predicted_concepts=predicted_concepts,
            gt_concepts=gt_concepts,
            num_levels=self._num_levels,
            level_lengths=list(self._level_lengths),
        )

    # --------------------------------------------------------------------
    # Inference forward: autoregressive (K sequential passes)
    # --------------------------------------------------------------------

    def _forward_inference(
        self,
        question_ids: torch.Tensor,
        question_attention_mask: Optional[torch.Tensor],
    ) -> PredictorOutput:
        """Autoregressive level-by-level generation without GT.

        For level k:
            input = concat(
                level 0 group (start + q_context),
                upsample(predicted C_0, L_1),
                ...,
                upsample(predicted C_{k-1}, L_k),
            )
            mask  = scale-level causal mask over the truncated lengths.
            C_k   = concept_head(transformer(input)[-L_k:])

        Returns:
            PredictorOutput with predicted_concepts only (gt_concepts=None).
        """
        batch_size = question_ids.shape[0]
        device = question_ids.device

        q_context = self._compute_q_context(question_ids, question_attention_mask)
        # q_context: [B, 1, D]

        predicted_concepts: List[torch.Tensor] = []

        for k in range(self._num_levels):
            L_k = self._level_lengths[k]

            # Build input groups [0..k].
            parts: List[torch.Tensor] = [
                self._level0_input(q_context, batch_size, device)
            ]
            for j in range(1, k + 1):
                parts.append(
                    self._upsample_prev_to_level(predicted_concepts[j - 1], j, device)
                )
            transformer_input = torch.cat(parts, dim=1)
            # transformer_input: [B, sum(L_0..L_k), D]

            # Scale-level causal mask over the truncated prefix of levels.
            partial_lengths = self._level_lengths[: k + 1]
            attn_mask = build_scale_causal_mask(
                partial_lengths, device, transformer_input.dtype
            )

            transformer_output = self.concept_transformer(
                transformer_input, mask=attn_mask
            )
            # Level-k slice is the LAST L_k positions of transformer_output.
            hidden_k = transformer_output[:, -L_k:, :]
            predicted = self.concept_head(hidden_k)
            predicted_concepts.append(predicted)

        return PredictorOutput(
            predicted_concepts=predicted_concepts,
            gt_concepts=None,
            num_levels=self._num_levels,
            level_lengths=list(self._level_lengths),
        )

    # --------------------------------------------------------------------
    # Reasoning (NTP) preparation
    # --------------------------------------------------------------------

    def _prepare_reasoning(
        self,
        output: PredictorOutput,
        question_ids: torch.Tensor,
        question_attention_mask: Optional[torch.Tensor],
        solution_ids: torch.Tensor,
        solution_attention_mask: torch.Tensor,
    ) -> None:
        """Populate `output.reasoning_*` by teacher-forcing through reason_model.

        PRINCIPLE (mirrors builder._prepare_reasoning):
            The predicted concept pyramid replaces CoT in the
            autoregressive flow Q -> (pyramid) -> S. The reasoning CE
            loss tests whether the predicted pyramid retains enough
            information to regenerate the solution.

        DATA FLOW (teacher-forcing):
            1. Cat predicted concepts               -> [B, total_C, D]
            2. back_proj to encoder dim             -> [B, total_C, D_enc]
            3. Embed Q tokens via embed_tokens      -> [B, L_Q, D_enc]
            4. Embed S tokens via embed_tokens      -> [B, L_S, D_enc]
            5. Concat [Q, concepts, S]              -> [B, L_Q+total_C+L_S, D_enc]
            6. Concat masks (concepts mask = ones)  -> [B, L_Q+total_C+L_S]
            7. reason_model(inputs_embeds, mask)    -> logits[..., V]
            8. Slice logits at positions predicting S:
                  start = L_Q + total_C - 1
                  end   = L_Q + total_C + L_S - 1
               -> reasoning_logits [B, L_S, V]
            9. Build reasoning_target_ids with -100 on padded positions.
           10. Argmax decode reasoning_texts for diagnostics.

        The gradient flows through predicted concepts -> back_proj ->
        reason_model, so the reasoning loss (computed in losses.py)
        trains the predictor to generate concepts that actually
        support solution decoding.
        """
        assert self.back_proj is not None, "back_proj is required for reasoning."
        if question_attention_mask is None:
            raise ValueError("question_attention_mask is required for reasoning loss.")

        device = question_ids.device
        batch_size = question_ids.shape[0]

        # 1. Cat predicted concepts: [B, total_C, D]
        concepts = torch.cat(output.predicted_concepts, dim=1)
        total_C = concepts.shape[1]

        # 2. Back-project to encoder space: [B, total_C, D_enc]
        concept_embeds = self.back_proj(concepts)

        # 3-4. Embed Q and S tokens using the backbone's input embeddings.
        backbone = self._get_backbone()
        embed_layer = backbone.get_input_embeddings()
        Q_embeds = embed_layer(question_ids)  # [B, L_Q, D_enc]
        L_Q = Q_embeds.shape[1]
        S_embeds = embed_layer(solution_ids)  # [B, L_S, D_enc]
        L_S = S_embeds.shape[1]

        # Align dtype (concepts may be bf16/fp16 from autocast while
        # embed_layer outputs model dtype; torch.cat requires matching dtype).
        target_dtype = Q_embeds.dtype
        if concept_embeds.dtype != target_dtype:
            concept_embeds = concept_embeds.to(target_dtype)

        # 5. Concat embeddings: [B, L_Q + total_C + L_S, D_enc]
        decoder_input_embeds = torch.cat([Q_embeds, concept_embeds, S_embeds], dim=1)

        # 6. Concat masks; concept mask is all ones (no padding in concepts).
        concept_mask = torch.ones(
            batch_size,
            total_C,
            device=device,
            dtype=question_attention_mask.dtype,
        )
        decoder_attention_mask = torch.cat(
            [question_attention_mask, concept_mask, solution_attention_mask], dim=1
        )

        # 7. Forward through reason_model (includes lm_head).
        model_out = self.reason_model(
            inputs_embeds=decoder_input_embeds,
            attention_mask=decoder_attention_mask,
        )
        # logits: [B, L_Q + total_C + L_S, V]
        logits = model_out.logits

        # 8. Slice logits that predict S tokens.
        # In a causal LM, logits at position t predict token at t+1.
        # Position (L_Q + total_C - 1) predicts S_0.
        # Position (L_Q + total_C + L_S - 2) predicts S_{L_S-1}.
        sol_start = L_Q + total_C - 1
        sol_end = L_Q + total_C + L_S - 1
        solution_logits = logits[:, sol_start:sol_end, :]
        # solution_logits: [B, L_S, V]

        # 9. Build targets with -100 on padded positions.
        targets = solution_ids.clone()
        targets[solution_attention_mask == 0] = -100

        output.reasoning_logits = solution_logits
        output.reasoning_target_ids = targets

        # 10. Teacher-forced argmax decode for diagnostics.
        with torch.no_grad():
            predicted_ids = solution_logits.argmax(dim=-1)
            output.reasoning_texts = self.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )
