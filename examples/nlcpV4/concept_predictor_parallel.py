"""NLCP V4 Concept Predictor — Option Y: per-level queries + cross-attention.

================================================================================
PURPOSE
================================================================================
Two-stage companion to `concept_predictor.py` (Option X).  Same task —
generate the concept pyramid [C_0, ..., C_{K-1}] from the question alone —
but with a different architecture that mirrors the Builder's query-based
design instead of flat causal AR.

================================================================================
CENTRAL IDEA — Option Y
================================================================================
Use the LLM backbone to CONTEXTUALISE content only:

        inputs_embeds = [ Q_embeds ,  back_decode(C_0..C_{K-1}) + slot markers ]

and use a SEPARATE per-level cross-attention head

        level_queries[k]  ∈  ℝ^{L_k × D_enc}

to EXTRACT the prediction for level k.  Learnable parameters and real
content embeddings NEVER share the same LLM input sequence.

Architectural symmetry with the Builder:

    Builder (Stage 1)                 Predictor Option Y (Stage 2)
    ─────────────────                 ──────────────────────────────
    encoder(CoT)                      reason_model([Q, back_decode(C_<k)])
           │                                     │
    concept_queries[k] @ H_CoT       level_queries[k]  @ H_prefix_k
           │                                     │
      C_k (concept space D)            hat_C_k (concept space D)

Both sides use learnable queries to extract per-level outputs; both avoid
interpolation entirely.

================================================================================
GLOSSARY (same as Option X)
================================================================================
B = 4, L_Q = 40, K = 6, level_lengths = [1,2,4,8,16,32], total_C = 63,
D = D_enc = 896, num_heads = 8.

================================================================================
TWO-STAGE PIPELINE (training, single LLM pass + K parallel cross-attentions)
================================================================================
    ┌──────────────────────────────────────────────────────────────────────┐
    │  Stage 1 — CONTENT BACKBONE (D_enc)                                  │
    │                                                                      │
    │   question_ids                 gt_concepts (K tensors)               │
    │         │                            │                               │
    │         ▼ embed_tokens               ▼ cat + back_decode + markers   │
    │   Q_embeds [B, L_Q, 896]       concept_embeds [B, 63, 896]           │
    │         └─────────┬────────────────┘                                 │
    │                   ▼  torch.cat(dim=1)                                │
    │        inputs_embeds [B, L_Q + 63, 896]                              │
    │                   │                                                  │
    │                   ▼  reason_model backbone (causal mask)             │
    │        hidden H   [B, L_Q + 63, 896]                                 │
    └─────────────────────┬────────────────────────────────────────────────┘
                          │
    ┌─────────────────────▼────────────────────────────────────────────────┐
    │  Stage 2 — PER-LEVEL CROSS-ATTENTION HEAD (D_enc → D)                │
    │                                                                      │
    │   for each level k ∈ [0, K):                                         │
    │       prefix_len_k = L_Q + Σ_{j<k} L_j                               │
    │       context_k    = H[:, : prefix_len_k, :]                         │
    │       queries_k    = level_queries[k]  [L_k, 896]                    │
    │                       expanded to [B, L_k, 896]                      │
    │                                                                      │
    │       attn_out_k   = cross_attn(queries_k, context_k, context_k)     │
    │                      [B, L_k, 896]                                   │
    │       hat_C_k      = concept_head(attn_out_k + queries_k)            │
    │                      [B, L_k, D]                                     │
    │                                                                      │
    │   All K levels run in PARALLEL — no sequential dependency.           │
    └──────────────────────────────────────────────────────────────────────┘

================================================================================
WHAT context_k LOOKS LIKE (per-level prefix window)
================================================================================
For K=6, L_Q=40, level_lengths=[1,2,4,8,16,32]:

    prefix_len_k = L_Q + cum_lengths[k]

        k   cum_lengths[k]   prefix_len_k   covers
        ─   ──────────────   ────────────   ──────────────────────────────
        0         0                40        Q only
        1         1                41        Q + C_0
        2         3                43        Q + C_0..C_1
        3         7                47        Q + C_0..C_2
        4        15                55        Q + C_0..C_3
        5        31                71        Q + C_0..C_4

    So level 5's query attends over 71 positions; level 0's over 40.
    Level 5 never sees its own gt (C_5) during training — only the
    residual history that precedes it.

================================================================================
CROSS-ATTENTION (visual)
================================================================================
For level k=3 (L_k=8, prefix_len_k=47):

        queries_k [B, 8, 896]
              │ │ │ ... │        ← 8 learnable queries
              ▼ ▼ ▼     ▼
        ┌─────────────────────────────────────────────────────┐
        │  attention(query_i, context) for i ∈ [0..7]         │
        │     scores ∈ [B, 8, 47]                             │
        │     probs  = softmax(scores / √d)                   │
        │     out[i] = Σ_j probs[i,j] * context[j]            │
        └─────────────────────────────────────────────────────┘
              │ │ │ ... │
              ▼ ▼ ▼     ▼
        attn_out [B, 8, 896]
              + residual(queries_k)  [B, 8, 896]
              → post_norm
              → concept_head (D_enc → D)
        hat_C_3 [B, 8, D]

No causal mask on the cross-attention — every query is allowed to look at
every context position (but the context itself already stops at prefix_len_k).

================================================================================
INFERENCE (K = 6 sequential passes, NOT 63)
================================================================================
Because predicting level k requires back_decode(C_<k) inside the LLM
context, the inference loop runs K times, growing the KV cache by L_{k-1}
positions per pass (instead of 1 position per step as in Option X):

    pass 0:                                      LLM tokens processed: 40 (Q)
        hidden = LLM(Q_embeds)                   # KV cache covers 40 positions
        context_0 = hidden                       [B, 40, 896]
        C_0 = head(CrossAttn(level_queries[0], context_0))
                                                 [B,  1, D]

    pass 1:                                      LLM tokens new: 1   → cache 41
        x = back_decode(C_0) + lvl + pos         [B,  1, 896]
        hidden_new = LLM(x, pkv)                 [B,  1, 896]
        context_1 = cat(context_0, hidden_new)   [B, 41, 896]
        C_1 = head(CrossAttn(level_queries[1], context_1))
                                                 [B,  2, D]

    pass 2..5: same pattern, adding L_{k-1} new positions.

    Cumulative LLM cache size per pass:
        pass k    new tokens    total cache
        0            40            40
        1             1            41
        2             2            43
        3             4            47
        4             8            55
        5            16            71

================================================================================
LOSS PATH (identical to Option X)
================================================================================
Returns the same PredictorOutput dataclass.  `losses.py` is unchanged.

================================================================================
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


# =========================================================================
# Output Dataclass (identical to Option X)
# =========================================================================


@dataclass
class PredictorOutput:
    """Full output of ConceptPredictorParallel.forward().

    Identical in shape / semantics to ConceptPredictor.PredictorOutput
    so both implementations share the same losses.py code path.

    Attributes:
        predicted_concepts: List of K tensors in concept space.
            Shape: each [B, L_k, D].
            Example (K=6, B=4, D=896):
                [0] [4,  1, 896]
                [1] [4,  2, 896]
                [2] [4,  4, 896]
                [3] [4,  8, 896]
                [4] [4, 16, 896]
                [5] [4, 32, 896]
        gt_concepts: Optional list of K tensors from the frozen
            Builder, pass-through.  None at inference.
        num_levels: K.
        level_lengths: [L_0, ..., L_{K-1}].
        reasoning_logits: Optional next-token-prediction logits.
            Shape: [B, L_S, V].
        reasoning_target_ids: Optional target ids with -100 on pads.
            Shape: [B, L_S].
        reasoning_texts: Optional argmax decode, list of B strings.
    """

    predicted_concepts: List[torch.Tensor]
    gt_concepts: Optional[List[torch.Tensor]] = None
    num_levels: int = 0
    level_lengths: List[int] = field(default_factory=list)
    reasoning_logits: Optional[torch.Tensor] = None
    reasoning_target_ids: Optional[torch.Tensor] = None
    reasoning_texts: Optional[List[str]] = None


# =========================================================================
# ConceptPredictorParallel — Option Y
# =========================================================================


class ConceptPredictorParallel(nn.Module):
    """Two-stage predictor with per-level queries + cross-attention.

    ARCHITECTURE SUMMARY
    --------------------
        Stage 1 (reason_model backbone, D_enc):
            [Q_embeds || back_decode(C_0..C_{K-1}) + slot markers]
            → causal LLM → hidden H  [B, L_Q + total_C, D_enc]

        Stage 2 (cross-attention head):
            for each level k:
                context_k = H[:, : L_Q + Σ_{j<k} L_j, :]
                query_k   = level_queries[k]      [L_k, D_enc]
                out_k     = CrossAttn(query_k, context_k, context_k)
                hat_C_k   = concept_head(out_k + query_k)

    COMPONENTS
    ----------
        reason_model           Backbone (shared with Builder when
                                use_shared_model=True).  Used as a
                                content-only context encoder.
        back_proj              Linear(D → D_enc), shared or owned.
        level_embeddings       Embedding(K, D_enc)          per-slot markers
        position_embeddings    Embedding(max(L_k), D_enc)   per-slot markers
        level_queries          ParameterList of K tensors, each
                                [L_k, D_enc] — the per-level learnable
                                queries.  Analogous to
                                Builder.concept_queries but living in
                                encoder space.
        query_norm             LayerNorm before cross-attention (query side).
        context_norm           LayerNorm before cross-attention (context side).
        cross_attn             nn.MultiheadAttention, shared across levels.
        post_norm              LayerNorm after attention + residual.
        concept_head           MLP D_enc → D.

    INVARIANTS
    ----------
        * learnable queries NEVER in the LLM input sequence
        * only real question tokens and lifted content concepts in the LLM
        * NO interpolation
        * NO start_token
    """

    # ------------------------------------------------------------------ #
    #  construction                                                      #
    # ------------------------------------------------------------------ #

    def __init__(self, config: dict, builder: Optional[nn.Module] = None):
        """Instantiate the parallel (two-stage) predictor.

        Args:
            config: Full config dict; required keys identical to
                ConceptPredictor (see concept_predictor.py).  Notable
                additional usage:
                    config["model"]["pyramid"]["num_heads"] — cross-attn heads.
                    config["model"]["predictor"]["dropout"] — cross-attn dropout.
            builder: ConceptPyramidBuilder — required when
                use_shared_model=True.  Its reason_model, tokenizer
                and back_proj are weight-tied into this predictor.
        """
        super().__init__()
        self.config = config
        self.pyramid_cfg = config["model"]["pyramid"]
        self.predictor_cfg = config["model"]["predictor"]

        num_levels = self.pyramid_cfg["num_levels"]
        concept_dim = self.pyramid_cfg["hidden_dim"]
        num_heads = self.pyramid_cfg["num_heads"]
        level_lengths = list(self.pyramid_cfg["level_lengths"])

        # Cache pyramid geometry.
        self._level_lengths = level_lengths
        self._num_levels = num_levels
        self._concept_dim = concept_dim
        self._total_concepts = sum(level_lengths)

        # ================================================================
        # Precomputed flat-slot → (level_id, intra_pos) tables.
        # Identical in purpose to Option X; reused here for the slot
        # markers that go on top of back_decoded concepts fed to the LLM.
        # ================================================================
        # Example (K=6, level_lengths=[1,2,4,8,16,32]):
        #     level_ids_flat = [0, 1,1, 2,2,2,2, 3×8, 4×16, 5×32]
        #     pos_ids_flat   = [0, 0,1, 0,1,2,3, 0..7, 0..15, 0..31]
        #
        # Shape: each buffer is [total_C] = [63] for K=6.
        level_ids: List[int] = []
        pos_ids: List[int] = []
        for k, Lk in enumerate(level_lengths):
            level_ids.extend([k] * Lk)
            pos_ids.extend(list(range(Lk)))
        self.register_buffer(
            "_level_ids_flat",
            torch.tensor(level_ids, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_pos_ids_flat",
            torch.tensor(pos_ids, dtype=torch.long),
            persistent=False,
        )

        # Cumulative concept counts per level.
        # Purpose: gives the "prefix length" boundary for cross-attention
        # context at each level.
        #
        # Logic: cum_lengths[k] = sum(level_lengths[:k]).
        # For K=6 and level_lengths=[1,2,4,8,16,32]:
        #     cum_lengths = [0, 1, 3, 7, 15, 31, 63]
        # Context prefix size at level k is L_Q + cum_lengths[k].
        cum = [0]
        for Lk in level_lengths:
            cum.append(cum[-1] + Lk)
        self._cum_lengths = cum

        # ================================================================
        # Component 0: backbone (reason_model + back_proj)
        # ================================================================
        # Same shared-vs-owned logic as Option X.
        use_shared = self.predictor_cfg["use_shared_model"]
        if use_shared:
            if builder is None:
                raise ValueError(
                    "ConceptPredictorParallel requires `builder` when "
                    "config.model.predictor.use_shared_model=True."
                )
            self.reason_model = builder.reason_model
            self.tokenizer = builder.tokenizer
            self.reason_model_hidden_dim = builder.reason_model_hidden_dim
            self._owns_model = False

            self.back_proj = builder.back_proj
            self._owns_back_proj = False
        else:
            self.reason_model, self.tokenizer, self.reason_model_hidden_dim = (
                self._init_reason_model(
                    self.predictor_cfg, config["training"]["predictor"]
                )
            )
            self._owns_model = True
            self.back_proj = nn.Linear(
                concept_dim, self.reason_model_hidden_dim, bias=False
            )
            self._owns_back_proj = True

        D_enc = self.reason_model_hidden_dim

        # ================================================================
        # Component 1: per-slot level / intra-pos embeddings in D_enc.
        # These go ON TOP of the back-decoded concept vectors BEFORE
        # feeding them to the LLM backbone.  Identical to Option X.
        #
        # Shape table (K=6, max(L_k)=32, D_enc=896):
        #     level_embeddings.weight    : [6,  896]
        #     position_embeddings.weight : [32, 896]
        # ================================================================
        max_len_per_level = max(level_lengths)
        self.level_embeddings = nn.Embedding(num_levels, D_enc)
        self.position_embeddings = nn.Embedding(max_len_per_level, D_enc)

        # ================================================================
        # Component 2: per-level learnable queries in D_enc space.
        # ================================================================
        # This is the CORE of Option Y.  Each element is one
        # [L_k, D_enc] tensor that acts as the "question" side of the
        # cross-attention head for level k.
        #
        # Parameter table (K=6, D_enc=896):
        #     level_queries[0] : [ 1, 896]
        #     level_queries[1] : [ 2, 896]
        #     level_queries[2] : [ 4, 896]
        #     level_queries[3] : [ 8, 896]
        #     level_queries[4] : [16, 896]
        #     level_queries[5] : [32, 896]
        # Total: 63 * 896 ≈ 56k extra params.
        self.level_queries = nn.ParameterList(
            [
                nn.Parameter(torch.randn(Lk, D_enc))
                for Lk in level_lengths
            ]
        )

        # ================================================================
        # Component 3: cross-attention head (query = level_queries[k],
        # key = value = LLM hidden prefix H[:, :prefix_len_k, :]).
        # ================================================================
        # Pre-LayerNorm on both the query and the context side
        # stabilises training (standard Transformer practice).
        # Post-LayerNorm on the (attn_out + queries_k) residual gives
        # a controlled magnitude for the concept_head input.
        #
        # Shape flow for one level (B=4, L_k=8, prefix_len=47, D_enc=896):
        #     Q:     [4,  8, 896]
        #     K, V:  [4, 47, 896]
        #     out:   [4,  8, 896]
        self.query_norm = nn.LayerNorm(D_enc)
        self.context_norm = nn.LayerNorm(D_enc)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=D_enc,
            num_heads=num_heads,
            dropout=self.predictor_cfg["dropout"],
            batch_first=True,
        )
        self.post_norm = nn.LayerNorm(D_enc)

        # ================================================================
        # Component 4: concept_head — MLP D_enc → D.
        # Shape: [B, L_k, D_enc] → Linear → GELU → Linear → [B, L_k, D].
        # ================================================================
        self.concept_head = nn.Sequential(
            nn.Linear(D_enc, D_enc),
            nn.GELU(),
            nn.Linear(D_enc, concept_dim),
        )

        self._init_weights()

    # ------------------------------------------------------------------ #
    #  helpers                                                           #
    # ------------------------------------------------------------------ #

    def _init_reason_model(self, pred_cfg: dict, train_cfg: dict) -> tuple:
        """Load an independent reason_model (use_shared_model=False only).

        Principle and logic are identical to Option X; see that file for
        the detailed block comment.

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

        lora_cfg = train_cfg["lora"]
        if lora_cfg is not None:
            reason_model = get_peft_model(
                reason_model,
                LoraConfig(
                    r=lora_cfg["r"],
                    lora_alpha=lora_cfg["lora_alpha"],
                    target_modules=lora_cfg["target_modules"],
                    lora_dropout=lora_cfg["lora_dropout"],
                    bias=lora_cfg["bias"],
                ),
            )
        if train_cfg["freeze"]:
            for p in reason_model.parameters():
                p.requires_grad = False
            if lora_cfg is not None:
                for n, p in reason_model.named_parameters():
                    if "lora_" in n:
                        p.requires_grad = True

        num_layers = pred_cfg["predictor_num_layers"]
        if num_layers is not None and num_layers > 0:
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
        """Return the underlying Transformer backbone (handles PEFT wrap)."""
        if hasattr(self.reason_model, "base_model"):
            inner = self.reason_model.base_model
            if hasattr(inner, "model"):
                return inner.model
            return inner
        if hasattr(self.reason_model, "model"):
            return self.reason_model.model
        return self.reason_model

    def _init_weights(self) -> None:
        """Initialise predictor-specific parameters.

        Choice notes:
            - level_queries are seeded from N(0, 0.02): small enough
              that the initial cross-attention output stays well within
              LayerNorm's happy range, large enough to break symmetry
              across the L_k query slots of a given level.
        """
        nn.init.normal_(self.level_embeddings.weight, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, std=0.02)
        for q in self.level_queries:
            nn.init.normal_(q, std=0.02)
        for m in self.concept_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if self._owns_back_proj:
            nn.init.xavier_uniform_(self.back_proj.weight)

    def back_decode(self, concept_space_tensor: torch.Tensor) -> torch.Tensor:
        """Lift a concept-space tensor into encoder space (D → D_enc).

        Args:
            concept_space_tensor: Shape [..., D].

        Returns:
            Shape [..., D_enc].
        """
        return self.back_proj(concept_space_tensor)

    def _embed_questions(self, question_ids: torch.Tensor) -> torch.Tensor:
        """Embed question token ids via the backbone's embed_tokens.

        Args:
            question_ids: [B, L_Q].

        Returns:
            Q_embeds: [B, L_Q, D_enc].
        """
        embed_layer = self._get_backbone().get_input_embeddings()
        return embed_layer(question_ids)

    def _build_concept_input_embeds(
        self,
        concepts_flat: torch.Tensor,
        start_slot: int,
    ) -> torch.Tensor:
        """Lift N concept vectors into D_enc and add per-slot markers.

        Identical in purpose and behaviour to Option X's helper.  See
        concept_predictor.py for the fully annotated version.

        Args:
            concepts_flat: [B, N, D].
            start_slot: Flat-slot index of the first of the N vectors.

        Returns:
            [B, N, D_enc].

        Example (start_slot=1, N=2):
            slot_ids  = [1, 2]
            level_ids = [1, 1]     (both at level 1)
            pos_ids   = [0, 1]
        """
        B, N, _ = concepts_flat.shape
        emb = self.back_decode(concepts_flat)

        slot_ids = torch.arange(
            start_slot, start_slot + N, device=emb.device
        )
        lvl = self.level_embeddings(
            self._level_ids_flat.to(emb.device)[slot_ids]
        )
        pos = self.position_embeddings(
            self._pos_ids_flat.to(emb.device)[slot_ids]
        )
        markers = (lvl + pos).unsqueeze(0).expand(B, -1, -1)
        if markers.dtype != emb.dtype:
            markers = markers.to(emb.dtype)
        return emb + markers

    # ------------------------------------------------------------------ #
    #  per-level cross-attention readout                                  #
    # ------------------------------------------------------------------ #

    def _extract_level(
        self,
        level_idx: int,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attention readout for one pyramid level.

        Principle:
            Each level_queries[k] acts like a DETR-style object query —
            it learns "what to ask for" from the LLM's hidden prefix.
            All L_k queries attend over the whole prefix in parallel;
            the softmax over the prefix acts as the soft selection
            mechanism, analogous to Builder.attention_weights A_k.

        Logic:
            1. LayerNorm both query and context (pre-norm stabilises
               cross-attention at init).
            2. Multi-head cross-attention.
            3. Residual add of the raw queries_k back onto the attn
               output: if attention were zero-initialised, the residual
               ensures the head still produces a meaningful
               (query-seeded) signal.
            4. Post-LayerNorm.
            5. concept_head (D_enc → D).

        Flow (B=4, level_idx=3, L_k=8, prefix=47, D_enc=896, D=896):
            queries [8, 896] → expand → [4, 8, 896]
            context                    → [4, 47, 896]
            MultiheadAttention:        → [4, 8, 896]
            residual + post_norm:      → [4, 8, 896]
            concept_head:              → [4, 8, D=896]

        Args:
            level_idx: Pyramid level k ∈ [0, K).
            context: LLM hidden prefix covering [Q, levels < k].
                Shape: [B, prefix_len_k, D_enc].

        Returns:
            hat_C_k: Predicted concepts for level k in concept space.
                Shape: [B, L_k, D].
        """
        B = context.shape[0]

        queries = self.level_queries[level_idx].unsqueeze(0).expand(B, -1, -1)

        # Dtype alignment with the LLM hidden stream.
        if queries.dtype != context.dtype:
            queries = queries.to(context.dtype)

        q_n = self.query_norm(queries)
        c_n = self.context_norm(context)

        attn_out, _ = self.cross_attn(
            query=q_n, key=c_n, value=c_n, need_weights=False
        )

        # Residual ensures a non-degenerate signal even if the attention
        # head is nearly zero at initialisation.
        out = self.post_norm(attn_out + queries)

        return self.concept_head(out)

    # ------------------------------------------------------------------ #
    #  forward dispatch                                                  #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        question_ids: torch.Tensor,
        question_attention_mask: Optional[torch.Tensor] = None,
        gt_concepts: Optional[List[torch.Tensor]] = None,
        solution_ids: Optional[torch.Tensor] = None,
        solution_attention_mask: Optional[torch.Tensor] = None,
    ) -> PredictorOutput:
        """Predict the concept pyramid from Q.

        Branches on `gt_concepts`:
            * Supplied  — teacher-forced single LLM pass + K parallel
                          cross-attentions (training).
            * Missing   — K sequential LLM passes with KV cache, one
                          cross-attention per pass (inference).

        If `solution_ids` is provided, the reasoning fields of the
        returned PredictorOutput are populated.

        Args:
            question_ids: [B, L_Q].
            question_attention_mask: [B, L_Q] or None.
            gt_concepts: List of K tensors, each [B, L_k, D] (training).
            solution_ids: [B, L_S] (optional — enables reasoning CE).
            solution_attention_mask: [B, L_S] (required iff solution_ids).

        Returns:
            PredictorOutput.
        """
        if gt_concepts is not None:
            out = self._forward_training(
                question_ids, question_attention_mask, gt_concepts
            )
        else:
            out = self._forward_inference(
                question_ids, question_attention_mask
            )

        if solution_ids is not None:
            if solution_attention_mask is None:
                raise ValueError(
                    "solution_attention_mask is required when solution_ids is given."
                )
            self._prepare_reasoning(
                out,
                question_ids,
                question_attention_mask,
                solution_ids,
                solution_attention_mask,
            )
        return out

    # ------------------------------------------------------------------ #
    #  training — single LLM pass + K parallel cross-attentions           #
    # ------------------------------------------------------------------ #

    def _forward_training(
        self,
        question_ids: torch.Tensor,
        question_attention_mask: Optional[torch.Tensor],
        gt_concepts: List[torch.Tensor],
    ) -> PredictorOutput:
        """Teacher-forced training forward (two-stage).

        Pipeline (B=4, L_Q=40, K=6, total_C=63, D=D_enc=896):

            # Stage 1 — content backbone
            1. torch.cat(gt_concepts, dim=1)      → [4, 63, 896]
            2. back_decode + slot markers          → [4, 63, 896]
            3. embed_tokens(Q_ids)                 → [4, 40, 896]
            4. concat [Q, concepts]                → [4, 103, 896]
            5. attention mask                      → [4, 103]
            6. backbone(inputs_embeds, mask)       → hidden H [4, 103, 896]

            # Stage 2 — per-level cross-attention, K=6 times in parallel
            for k in 0..5:
                prefix_len_k = 40 + cum_lengths[k]
                context_k    = H[:, :prefix_len_k, :]
                hat_C_k      = _extract_level(k, context_k)
                    k=0: context [4, 40, 896] → hat_C_0 [4,  1, 896]
                    k=1: context [4, 41, 896] → hat_C_1 [4,  2, 896]
                    k=2: context [4, 43, 896] → hat_C_2 [4,  4, 896]
                    k=3: context [4, 47, 896] → hat_C_3 [4,  8, 896]
                    k=4: context [4, 55, 896] → hat_C_4 [4, 16, 896]
                    k=5: context [4, 71, 896] → hat_C_5 [4, 32, 896]

        Args:
            question_ids: [B, L_Q].
            question_attention_mask: [B, L_Q] or None.
            gt_concepts: List of K tensors, each [B, L_k, D].

        Returns:
            PredictorOutput with predicted_concepts and gt_concepts set.
        """
        if len(gt_concepts) != self._num_levels:
            raise ValueError(
                f"gt_concepts has {len(gt_concepts)} levels, "
                f"expected {self._num_levels}."
            )

        B = question_ids.shape[0]
        device = question_ids.device

        # -------- Stage 1 — build LLM input and get hidden H --------
        concepts_flat = torch.cat(gt_concepts, dim=1)
        concept_embeds = self._build_concept_input_embeds(
            concepts_flat, start_slot=0
        )

        Q_embeds = self._embed_questions(question_ids)
        L_Q = Q_embeds.shape[1]
        total_C = concept_embeds.shape[1]

        if concept_embeds.dtype != Q_embeds.dtype:
            concept_embeds = concept_embeds.to(Q_embeds.dtype)

        inputs_embeds = torch.cat([Q_embeds, concept_embeds], dim=1)

        if question_attention_mask is not None:
            concept_mask = torch.ones(
                B, total_C, device=device, dtype=question_attention_mask.dtype
            )
            attention_mask = torch.cat(
                [question_attention_mask, concept_mask], dim=1
            )
        else:
            attention_mask = None

        backbone = self._get_backbone()
        backbone_out = backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        if hasattr(backbone_out, "last_hidden_state"):
            hidden = backbone_out.last_hidden_state
        else:
            hidden = backbone_out[0]

        # -------- Stage 2 — per-level cross-attention (parallel) --------
        # Principle: to predict level k we give the query access to Q and
        # to ALL earlier levels, which is information-consistent with the
        # teacher-forced canvas.  We intentionally exclude level k itself
        # from the context so the prediction is not trivial.
        predicted_concepts: List[torch.Tensor] = []
        for k in range(self._num_levels):
            t_end = L_Q + self._cum_lengths[k]
            context = hidden[:, :t_end, :]
            predicted_concepts.append(self._extract_level(k, context))

        return PredictorOutput(
            predicted_concepts=predicted_concepts,
            gt_concepts=gt_concepts,
            num_levels=self._num_levels,
            level_lengths=list(self._level_lengths),
        )

    # ------------------------------------------------------------------ #
    #  inference — K sequential LLM passes with KV-cache                  #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _forward_inference(
        self,
        question_ids: torch.Tensor,
        question_attention_mask: Optional[torch.Tensor],
    ) -> PredictorOutput:
        """Autoregressive over LEVELS — K passes total (not 63).

        State maintained across passes:
            pkv           — HuggingFace past_key_values (LLM KV cache).
            running_mask  — full attention mask covering all positions
                            currently in the cache.
            context       — running concatenation of LLM hidden states
                            over all processed positions.  Used only by
                            the cross-attention head (not by the LLM,
                            which maintains its own KV cache).

        Pass diagram (B=4, L_Q=40, K=6, level_lengths=[1,2,4,8,16,32]):

            pass 0:
                x                  = Q_embeds           [4, 40, 896]
                out                = LLM(x)
                context            = out.last_hidden    [4, 40, 896]
                running_mask       =                    [4, 40]
                C_0 = extract_level(0, context)         [4,  1, D]

            pass 1:
                x = back_decode(C_0)+lvl+pos            [4,  1, 896]
                out = LLM(x, pkv)                       [4,  1, 896]
                context ← cat(context, out.last_hidden) [4, 41, 896]
                running_mask ← concat +1                [4, 41]
                C_1 = extract_level(1, context)         [4,  2, D]

            pass 2: feed C_1 (2 positions),   context → [4, 43, 896]
            pass 3: feed C_2 (4 positions),   context → [4, 47, 896]
            pass 4: feed C_3 (8 positions),   context → [4, 55, 896]
            pass 5: feed C_4 (16 positions),  context → [4, 71, 896]

        Args:
            question_ids: [B, L_Q].
            question_attention_mask: [B, L_Q] or None.

        Returns:
            PredictorOutput with predicted_concepts; gt_concepts=None.
        """
        B = question_ids.shape[0]
        device = question_ids.device
        backbone = self._get_backbone()

        # =============== Pass 0: feed Q, predict level 0 ===============
        Q_embeds = self._embed_questions(question_ids)
        out = backbone(
            inputs_embeds=Q_embeds,
            attention_mask=question_attention_mask,
            use_cache=True,
        )
        pkv = out.past_key_values
        context = out.last_hidden_state

        if question_attention_mask is not None:
            running_mask = question_attention_mask
        else:
            running_mask = torch.ones(
                B, Q_embeds.shape[1], device=device, dtype=torch.long
            )

        predicted_concepts: List[torch.Tensor] = []
        C_0 = self._extract_level(0, context)
        predicted_concepts.append(C_0)

        # =============== Passes 1..K-1: feed prev level, predict next ==
        for k in range(1, self._num_levels):
            prev_level = predicted_concepts[-1]

            # start_slot is the flat-slot index of the FIRST of the
            # L_{k-1} concepts being fed into the LLM.  It equals
            # cum_lengths[k-1] (concepts of earlier levels come first).
            #
            # Example (K=6):
            #     k=1 → start_slot=0  feeding L_0=1 slot
            #     k=2 → start_slot=1  feeding L_1=2 slots
            #     k=3 → start_slot=3  feeding L_2=4 slots
            #     k=4 → start_slot=7  feeding L_3=8 slots
            #     k=5 → start_slot=15 feeding L_4=16 slots
            start_slot = self._cum_lengths[k - 1]
            x = self._build_concept_input_embeds(prev_level, start_slot=start_slot)
            if x.dtype != context.dtype:
                x = x.to(context.dtype)

            L_prev = x.shape[1]
            running_mask = torch.cat(
                [
                    running_mask,
                    torch.ones(B, L_prev, device=device, dtype=running_mask.dtype),
                ],
                dim=1,
            )
            out = backbone(
                inputs_embeds=x,
                attention_mask=running_mask,
                past_key_values=pkv,
                use_cache=True,
            )
            pkv = out.past_key_values

            # Grow the running context by the L_prev new hidden positions.
            # The LLM itself relies on its own KV cache; `context` is
            # purely for the cross-attention head downstream.
            context = torch.cat([context, out.last_hidden_state], dim=1)

            C_k = self._extract_level(k, context)
            predicted_concepts.append(C_k)

        return PredictorOutput(
            predicted_concepts=predicted_concepts,
            gt_concepts=None,
            num_levels=self._num_levels,
            level_lengths=list(self._level_lengths),
        )

    # ------------------------------------------------------------------ #
    #  optional reasoning CE loss path                                   #
    # ------------------------------------------------------------------ #

    def _prepare_reasoning(
        self,
        output: PredictorOutput,
        question_ids: torch.Tensor,
        question_attention_mask: Optional[torch.Tensor],
        solution_ids: torch.Tensor,
        solution_attention_mask: torch.Tensor,
    ) -> None:
        """Populate output.reasoning_* via teacher-forced NTP.

        Structure is identical to Option X and to
        ConceptPyramidBuilder._prepare_reasoning.  Reproduced here
        without slot markers so the two reasoning losses (Stage-1 and
        Stage-2) are numerically comparable.

        Sequence layout (B=4, L_Q=40, total_C=63, L_S=30, T=133):

            ◄──── L_Q = 40 ────►◄──── total_C = 63 ────►◄── L_S = 30 ──►
            [  Q_embeds         ][ back_decode(predicted) ][  S_embeds  ]

            logits[:, 102 : 132, :]  → solution_logits   [B, 30, V]

        Gradient path: predicted_concepts → back_decode → reason_model → CE.

        Args:
            output: PredictorOutput to mutate in-place.
            question_ids: [B, L_Q].
            question_attention_mask: [B, L_Q] (required).
            solution_ids: [B, L_S].
            solution_attention_mask: [B, L_S].
        """
        if question_attention_mask is None:
            raise ValueError(
                "question_attention_mask is required for reasoning loss."
            )

        device = question_ids.device
        B = question_ids.shape[0]

        concepts = torch.cat(output.predicted_concepts, dim=1)
        total_C = concepts.shape[1]
        concept_embeds = self.back_decode(concepts)

        embed_layer = self._get_backbone().get_input_embeddings()
        Q_embeds = embed_layer(question_ids)
        L_Q = Q_embeds.shape[1]
        S_embeds = embed_layer(solution_ids)
        L_S = S_embeds.shape[1]

        if concept_embeds.dtype != Q_embeds.dtype:
            concept_embeds = concept_embeds.to(Q_embeds.dtype)

        decoder_input_embeds = torch.cat(
            [Q_embeds, concept_embeds, S_embeds], dim=1
        )
        concept_mask = torch.ones(
            B, total_C, device=device, dtype=question_attention_mask.dtype
        )
        decoder_attention_mask = torch.cat(
            [question_attention_mask, concept_mask, solution_attention_mask], dim=1
        )

        model_out = self.reason_model(
            inputs_embeds=decoder_input_embeds,
            attention_mask=decoder_attention_mask,
        )
        logits = model_out.logits

        sol_start = L_Q + total_C - 1
        sol_end = L_Q + total_C + L_S - 1
        solution_logits = logits[:, sol_start:sol_end, :]

        targets = solution_ids.clone()
        targets[solution_attention_mask == 0] = -100

        output.reasoning_logits = solution_logits
        output.reasoning_target_ids = targets

        with torch.no_grad():
            predicted_ids = solution_logits.argmax(dim=-1)
            output.reasoning_texts = self.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )
