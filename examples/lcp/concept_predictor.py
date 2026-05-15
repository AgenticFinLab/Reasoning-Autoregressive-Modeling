"""lcp Concept Predictor.

VAR-faithful single-sequence architecture with pre-LLM approximation-token
construction. This is the SOLE predictor implementation in lcp — there are
no alternative modes (no flat-AR variant, no shared/independent toggle).
The predictor trains concept-pyramid prediction and reasoning NTP jointly
in ONE packed forward pass over
[real_Q | approx_tokens_0..K-1 | real_S | tail_pad].

===============================================================================
1. POSITION IN THE TWO-PHASE PIPELINE
===============================================================================

  ┌────────────────────────────┐          ┌────────────────────────────────┐
  │ Stage 1: Builder           │          │ Stage 2: Predictor (THIS FILE) │
  │ (concept_builder.py)       │          │ (concept_predictor.py)         │
  ├────────────────────────────┤          ├────────────────────────────────┤
  │ Input: (Q, CoT, Solution)  │          │ Train: (Q, CoT, Solution)      │
  │                            │          │ Infer: Q only                  │
  │ Output:                    │  ─────►  │                                │
  │   concepts [C_0..C_{K-1}] │ (frozen) │ Output:                        │
  │   f_hat_per_level          │          │   C_hat_0..C_hat_{K-1}         │
  │   (GT targets + TF input)  │          │   reasoning_logits/targets     │
  └────────────────────────────┘          └────────────────────────────────┘

  The Builder is FROZEN inside the Predictor. It provides:
    - gt_concepts:      supervision targets for concept prediction loss
    - f_hat_per_level:  teacher-forcing inputs (cumulative reconstructions)

===============================================================================
2. CENTRAL PRINCIPLE — Cumulative Reconstruction f_hat
===============================================================================

  Each concept C_k is a RESIDUAL contribution (VAR §5.2.2). The correct
  conditioning for predicting level k is NOT raw concepts, but the
  cumulative reconstruction:

      f_hat_k = Σ_{j<k} R_j   ("what has been explained so far")

  This mirrors VAR's residual quantization: each scale encodes only what
  previous scales left unexplained. The predictor observes f_hat (not C)
  to decide what to predict next.

===============================================================================
3. MODULE ARCHITECTURE
===============================================================================

  ┌═══════════════════════════════════════════════════════════════════════════┐
  ║                    ConceptPredictor(nn.Module)                           ║
  ╠═══════════════════════════════════════════════════════════════════════════╣
  ║                                                                         ║
  ║  ┌─────────────────────── Frozen ────────────────────────┐              ║
  ║  │  builder: ConceptPyramidBuilder                       │              ║
  ║  │    → gt_concepts: List[Tensor [B, L_k, D]]            │              ║
  ║  │    → f_hat_per_level: List[Tensor [B, L_canvas, D]]   │              ║
  ║  └───────────────────────────────────────────────────────┘              ║
  ║                                                                         ║
  ║  ┌─────────────────────── Trainable ─────────────────────┐              ║
  ║  │                                                       │              ║
  ║  │  reason_model: AutoModelForCausalLM (+ LoRA)          │              ║
  ║  │    Hidden dim = D_enc (e.g., 896 for Qwen2.5-0.5B)    │              ║
  ║  │    Vocab size = V                                     │              ║
  ║  │                                                       │              ║
  ║  │  back_proj: Linear(D → D_enc, bias=False)             │              ║
  ║  │    Maps concept-space → LLM hidden space              │              ║
  ║  │                                                       │              ║
  ║  │  level_queries: ParameterList                         │              ║
  ║  │    K parameters, each [L_k, D_enc]                    │              ║
  ║  │    L_k ∈ level_lengths (e.g., [1,2,4,8,16,32])        │              ║
  ║  │                                                       │              ║
  ║  │  extract_attn: MultiheadAttention(D_enc, num_heads)   │              ║
  ║  │    + query_norm, context_norm, post_norm (LayerNorm)   │              ║
  ║  │    Cross-attention: queries extract from f_hat context │              ║
  ║  │                                                       │              ║
  ║  │  lvl_embed: Embedding(K, D_enc)                       │              ║
  ║  │    Unique per-level identity tag added to each token   │              ║
  ║  │                                                       │              ║
  ║  │  concept_head: Sequential(                            │              ║
  ║  │      Linear(D_enc, D_enc), GELU, Linear(D_enc, D))    │              ║
  ║  │    Maps LLM hidden output → concept space prediction  │              ║
  ║  │                                                       │              ║
  ║  └───────────────────────────────────────────────────────┘              ║
  ║                                                                         ║
  ╚═══════════════════════════════════════════════════════════════════════════╝

  Dimension glossary:
    B       = batch size
    D       = concept_dim (pyramid hidden_dim, e.g., 768)
    D_enc   = LLM hidden_size (reason_model.config.hidden_size)
    V       = LLM vocab size
    K       = num_levels (e.g., 6)
    L_k     = level_lengths[k] (e.g., 1, 2, 4, 8, 16, 32)
    total_C = Σ L_k (e.g., 63)
    L_canvas= inference_canvas_length (e.g., 128)
    T       = packed sequence length (varies per batch)

===============================================================================
4. TRAINING DATA FLOW — forward(batch)
===============================================================================

  batch: BuilderInput(questions, chain_of_thoughts, solutions)
                │
                ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ Phase 1: Input Preparation                                             │
  ├──────────────────────────────────────────────────────────────────────────┤
  │                                                                        │
  │  builder(batch) ──► PyramidOutput                                      │
  │    gt_concepts = [C_0 [B,L_0,D], ..., C_{K-1} [B,L_{K-1},D]]          │
  │    gt_f_hats   = [f_hat_0 [B,L,D], ..., f_hat_{K-1} [B,L,D]]          │
  │    (all detached — no gradient flows back to builder)                   │
  │                                                                        │
  │  tokenizer(questions) ──► question_ids [B, L_Q_pad], q_mask [B, L_Q_pad]│
  │  tokenizer(solutions) ──► solution_ids [B, L_S_pad], s_mask [B, L_S_pad]│
  │                                                                        │
  └──────────────────────────────────────────────────────────────────────────┘
                │
                ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ Phase 2: Single Packed Forward                                         │
  ├──────────────────────────────────────────────────────────────────────────┤
  │                                                                        │
  │  Step 2.1: Embed Q and S                                               │
  │  ──────────────────────                                                │
  │    Q_embeds = embed_layer(question_ids)           [B, L_Q_pad, D_enc]  │
  │    S_embeds = embed_layer(solution_ids)           [B, L_S_pad, D_enc]  │
  │                                                                        │
  │  Step 2.2: Construct approx tokens (pre-LLM)                          │
  │  ────────────────────────────────────────────                          │
  │    for k in [0, K):                                                    │
  │      approx_tokens_k = _construct_approx_tokens(k, gt_f_hats[k])       │
  │                                              → [B, L_k, D_enc]         │
  │    approx_tokens = cat(approx_token_list)    → [B, total_C, D_enc]     │
  │                                                                        │
  │  Step 2.3: Per-row packing (RoPE-safe)                                 │
  │  ──────────────────────────────────────                                │
  │    pack = pack_qcs_sequences(Q_embeds, q_mask, approx_tokens,           │
  │                              S_embeds, s_mask)                          │
  │                                                                        │
  │    Row i layout (no padding in middle):                                │
  │    ┌──────────┬─────────────────────┬──────────┬──────────┐            │
  │    │ real_Q_i │ approx_tokens (all K)│ real_S_i │ tail_pad │            │
  │    │ q_len[i] │      total_C        │ s_len[i] │ padding  │            │
  │    └──────────┴─────────────────────┴──────────┴──────────┘            │
  │    ◄──────────────────── T (packed length) ──────────────────►          │
  │                                                                        │
  │  Step 2.4: Build scale-causal 4D mask [B, 1, T, T]                     │
  │  ──────────────────────────────────────────────                        │
  │    (see §6 ATTENTION MASK for detailed layout)                         │
  │                                                                        │
  │  Step 2.5: Single LLM forward                                          │
  │  ─────────────────────────────                                         │
  │    model_out = reason_model(inputs_embeds=packed, attention_mask=mask4d,│
  │                             output_hidden_states=True)                  │
  │    hidden = model_out.hidden_states[-1]    [B, T, D_enc]               │
  │    logits = model_out.logits               [B, T, V]                   │
  │                                                                        │
  │  Step 2.6: Concept readout                                             │
  │  ──────────────────────────                                            │
  │    For row i, approx-token positions are:                              │
  │      col = q_len[i] + j,  j ∈ [0, total_C)                            │
  │    approx_hidden = hidden[i, q_len[i]:q_len[i]+total_C]  [B,total_C,D_enc]│
  │    C_hat_k = concept_head(approx_hidden[:, offset_k:offset_k+L_k])     │
  │                                                → [B, L_k, D]           │
  │                                                                        │
  │  Step 2.7: Reasoning NTP supervision                                   │
  │  ───────────────────────────────────                                   │
  │    reasoning_logits    = gather_solution_logits(logits, pack)           │
  │    reasoning_target_ids = build_solution_targets(solution_ids, s_mask)  │
  │                                                                        │
  └──────────────────────────────────────────────────────────────────────────┘
                │
                ▼
  PredictorOutput:
    predicted_concepts:   [C_hat_0, ..., C_hat_{K-1}]    (for concept loss)
    gt_concepts:          [C_0, ..., C_{K-1}]            (from builder)
    reasoning_logits:     [B, L_sol, V]                  (for CE loss)
    reasoning_target_ids: [B, L_sol]                     (shifted targets)

===============================================================================
5. INFERENCE DATA FLOW — _forward_inference(question_ids, q_mask)
===============================================================================

  Unlike training (single forward), inference runs K sequential passes
  because each f_hat depends on the previous level's prediction.

  ┌──────────────────────────────────────────────────────────────────────────┐
  │ Initialisation                                                         │
  │   f_hat = zeros [B, L_canvas, D]                                       │
  │   Q_embeds = embed_layer(question_ids)                                 │
  │   approx_token_list = []                                               │
  └──────────────────────────────────────────────────────────────────────────┘
                │
                ▼
  ┌══════════════════════════════════════════════════════════════════════════┐
  ║ Loop: for k = 0, 1, ..., K-1                                          ║
  ╠══════════════════════════════════════════════════════════════════════════╣
  ║                                                                        ║
  ║  ① _construct_approx_tokens(k, f_hat)                                  ║
  ║     → approx_tokens_k [B, L_k, D_enc]                                 ║
  ║     → attn_weights α_k [B, L_k, L_canvas]                             ║
  ║     approx_token_list.append(approx_tokens_k)                          ║
  ║                                                                        ║
  ║  ② Pack [real_Q_i | approx_tokens_0..k | tail_pad]                     ║
  ║     (growing sequence: level by level adds L_k tokens)                 ║
  ║                                                                        ║
  ║  ③ Build scale-causal mask + backbone forward → hidden                 ║
  ║                                                                        ║
  ║  ④ Readout level k's approx-token positions:                           ║
  ║     col[i] = q_len[i] + Σ_{j<k} L_j + arange(L_k)                     ║
  ║     hidden_k = hidden[row_idx, col_idx]        [B, L_k, D_enc]         ║
  ║     C_hat_k = concept_head(hidden_k)           [B, L_k, D]             ║
  ║                                                                        ║
  ║  ⑤ Reconstruct & accumulate:                                           ║
  ║     R_k = α_k^T @ C_hat_k                     [B, L_canvas, D]        ║
  ║     f_hat = f_hat + R_k                                                ║
  ║                                                                        ║
  ╚══════════════════════════════════════════════════════════════════════════╝
                │
                ▼
  PredictorOutput(predicted_concepts=[C_hat_0..C_hat_{K-1}], gt_concepts=None)

  Solution generation: call generate_solution() separately after prediction.

===============================================================================
6. ATTENTION MASK — Scale-Causal 4D Layout
===============================================================================

  The mask implements VAR-style scale-causal attention:
    - Within a level:  BIDIRECTIONAL (approx tokens of level k see each other)
    - Across levels:   CAUSAL (level k can see all levels j ≤ k)
    - Q region:        standard token-causal (autoregressive)
    - S region:        token-causal; sees everything to its left (Q + approx)

  Packed row i (T positions):
  ┌───────────────┬───┬───┬───┬─────┬───┬───────────────┬─────────────────┐
  │ Q (causal)    │ L0│ L1│ L2│ ... │L_{K-1}│ S (causal)    │ PAD (masked)    │
  │ scale_id = 0  │ 1 │ 2 │ 3 │     │ K     │ scale_id=K+1  │ scale_id = -1   │
  └───────────────┴───┴───┴───┴─────┴───────┴───────────────┴─────────────────┘

  Visibility matrix (scale_q, scale_k):
  ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
  │  q╲k    │   Q(0)  │  Lv1(1) │  Lv2(2) │   ...   │ LvK(K)  │  S(K+1) │
  ├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
  │  Q(0)   │ causal  │    ✗    │    ✗    │    ✗    │    ✗    │    ✗    │
  │  Lv1(1) │    ✓    │  bidir  │    ✗    │    ✗    │    ✗    │    ✗    │
  │  Lv2(2) │    ✓    │    ✓    │  bidir  │    ✗    │    ✗    │    ✗    │
  │  ...    │    ✓    │    ✓    │    ✓    │  bidir  │    ✗    │    ✗    │
  │  LvK(K) │    ✓    │    ✓    │    ✓    │    ✓    │  bidir  │    ✗    │
  │  S(K+1) │    ✓    │    ✓    │    ✓    │    ✓    │    ✓    │ causal  │
  └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

  Key properties:
    - "causal"  = token-level causal (pos_q >= pos_k)
    - "bidir"   = fully bidirectional within same level
    - "✓"       = full visibility (cross-level, lower scale → visible)
    - "✗"       = masked (cannot see future scales)
    - PAD keys  = masked for all queries (except diagonal NaN safeguard)

  Implementation: _build_scale_causal_mask_packed() returns [B, 1, T, T]
  additive mask (0.0 = attend, -inf = mask) compatible with HuggingFace
  transformers' attention_mask interface.

===============================================================================
7. APPROX-TOKEN CONSTRUCTION — _construct_approx_tokens(k, f_hat_k)
===============================================================================

  Purpose: Extract L_k fixed-length tokens from variable-length f_hat_k.
  Text-domain analog of VAR's area-downsample(f_hat, pn×pn) → pn² tokens.

  Data flow:

    f_hat_k [B, L, D]                  level_queries[k] [L_k, D_enc]
         │                                      │
         ▼                                      ▼
    back_proj(f_hat_k)              queries.expand(B, L_k, D_enc)
    [B, L, D_enc]                   [B, L_k, D_enc]
         │                                      │
         ▼                                      ▼
    context_norm ──────────────►  query_norm ────┤
    [B, L, D_enc] (K, V)         [B, L_k, D_enc] (Q)
                                        │
                                        ▼
                              ┌─────────────────────┐
                              │   extract_attn      │
                              │  (cross-attention)  │
                              └─────────┬───────────┘
                                        │
                              attn_out [B, L_k, D_enc]
                              attn_w   [B, L_k, L]   (α_k)
                                        │
                                        ▼
                              post_norm(attn_out + queries)   ← residual
                                        │
                                        ▼
                              + lvl_embed[k]                  ← level tag
                                        │
                                        ▼
                              approx_tokens_k [B, L_k, D_enc]

  The attention weights α_k are retained at inference for reconstruction:
    R_k = α_k^T @ C_hat_k    [B, L_canvas, D]

===============================================================================
8. GENERATE SOLUTION — generate_solution(predicted_concepts, ...)
===============================================================================

  After inference predicts [C_hat_0..C_hat_{K-1}], generate free-form
  solution text autoregressively:

    predicted_concepts ──► cat ──► back_decode ──► concept_embeds [B, total_C, D_enc]
    question_ids ──► embed_layer ──► Q_embeds [B, L_Q, D_enc]

    Pack [real_Q_i | concept_embeds | tail_pad]  (no S — generating it)
    reason_model.generate(inputs_embeds=packed, max_new_tokens=256)
    → List[str]

===============================================================================
9. CONFIGURATION KEYS
===============================================================================

  config["model"]["pyramid"]:
    num_levels:       K (number of pyramid levels)
    hidden_dim:       D (concept space dimension)
    num_heads:        attention heads for extract_attn
    level_lengths:    [L_0, L_1, ..., L_{K-1}] (tokens per level)
    max_seq_len:      max tokenization length for Q and S

  config["model"]["predictor"]:
    model_name:              HuggingFace model ID for reason_model
    dropout:                 dropout for extract_attn
    inference_canvas_length: L_canvas for f_hat at inference (default 128)
    num_layers:              optional layer truncation (None = use all)

  config["training"]["predictor"]:
    freeze:       bool — freeze reason_model base weights
    lora:         dict (r, lora_alpha, target_modules, lora_dropout, bias)

===============================================================================
10. EXTERNAL INTERFACES
===============================================================================

    forward(batch: BuilderInput) → PredictorOutput
        Training: single packed forward producing both concept predictions
        and reasoning_logits/target_ids.

    _forward_inference(question_ids, question_attention_mask)
        Inference: K sequential packed passes with self-maintained f_hat.
        Returns PredictorOutput(predicted_concepts, gt_concepts=None).

    generate_solution(predicted_concepts, question_ids, ...)
        → List[str]   (free autoregressive generation post-prediction)

REFERENCES:
    - concept_builder.py:  Builder that produces gt_concepts + f_hats
    - lcp.utils.pack_qcs_sequences: per-row packing contract (PackedQCS)
    - lcp.utils.gather_solution_logits / build_solution_targets: NTP helpers
    - VAR third-part/VAR-main/models/var.py: analogous multi-scale paradigm
    - docs/VAR.md §5.3: training pipeline and scale-causal mask
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from lcp.data_loader import BuilderInput
from lcp.utils import (
    build_solution_targets,
    gather_solution_logits,
    pack_qcs_sequences,
)


@dataclass
class PredictorOutput:
    """Full output of ConceptPredictor.forward()."""

    predicted_concepts: List[torch.Tensor]
    gt_concepts: Optional[List[torch.Tensor]] = None
    num_levels: int = 0
    level_lengths: List[int] = field(default_factory=list)
    reasoning_logits: Optional[torch.Tensor] = None
    reasoning_target_ids: Optional[torch.Tensor] = None
    reasoning_texts: Optional[List[str]] = None
    generation_texts: Optional[List[str]] = None


class ConceptPredictor(nn.Module):
    """Concept predictor: VAR-like single packed sequence with pre-LLM approx tokens.

    Takes a frozen Builder as a component. During training:
        Phase 1: builder(batch) → PyramidOutput (gt_concepts + f_hats)
        Phase 2: Pack [real_Q | approx_tokens_0..K-1 | real_S | tail_pad] per row
                 → single LLM forward → PredictorOutput
                 (concept predictions + reasoning NTP supervision).
    """

    def __init__(self, config: dict, builder: nn.Module):
        super().__init__()
        self.config = config
        self.pyramid_cfg = config["model"]["pyramid"]
        self.predictor_cfg = config["model"]["predictor"]

        num_levels = self.pyramid_cfg["num_levels"]
        concept_dim = self.pyramid_cfg["hidden_dim"]
        num_heads = self.pyramid_cfg["num_heads"]
        level_lengths = list(self.pyramid_cfg["level_lengths"])

        self._level_lengths = level_lengths
        self._num_levels = num_levels
        self._concept_dim = concept_dim
        self._total_concepts = sum(level_lengths)
        self._inference_canvas_length = self.predictor_cfg.get(
            "inference_canvas_length", 128
        )

        # Frozen builder — Phase 1 only (GT concepts + f_hats)
        self.builder = builder
        for p in self.builder.parameters():
            p.requires_grad = False

        # Predictor's OWN decoder-only model + tokenizer
        self.reason_model, self.tokenizer, self.reason_model_hidden_dim = (
            self._init_reason_model(self.predictor_cfg, config["training"]["predictor"])
        )

        D_enc = self.reason_model_hidden_dim

        # Predictor's OWN back_proj: concept-space → LLM hidden space
        self.back_proj = nn.Linear(concept_dim, D_enc, bias=False)

        # Per-level learnable queries [L_k, D_enc] — will attend to f_hat_k
        self.level_queries = nn.ParameterList(
            [nn.Parameter(torch.randn(Lk, D_enc)) for Lk in level_lengths]
        )

        # Approx-token construction cross-attention (pre-LLM)
        # Queries extract L_k tokens from variable-length back_proj(f_hat_k)
        self.query_norm = nn.LayerNorm(D_enc)
        self.context_norm = nn.LayerNorm(D_enc)
        self.extract_attn = nn.MultiheadAttention(
            embed_dim=D_enc,
            num_heads=num_heads,
            dropout=self.predictor_cfg["dropout"],
            batch_first=True,
        )
        self.post_norm = nn.LayerNorm(D_enc)

        # Level embedding: each pyramid level gets a unique embedding vector
        self.lvl_embed = nn.Embedding(num_levels, D_enc)

        # Concept head: D_enc → D (maps LLM output to concept space)
        self.concept_head = nn.Sequential(
            nn.Linear(D_enc, D_enc),
            nn.GELU(),
            nn.Linear(D_enc, concept_dim),
        )

        self._init_weights()

    # ------------------------------------------------------------------ #
    #  helpers                                                           #
    # ------------------------------------------------------------------ #

    def _init_reason_model(self, pred_cfg, train_cfg):
        """Load predictor's own decoder-only model from config."""
        model_name = pred_cfg["model_name"]
        reason_model = AutoModelForCausalLM.from_pretrained(model_name)
        hidden_dim = reason_model.config.hidden_size
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # LoRA
        lora_cfg = train_cfg.get("lora")
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
        # Freeze base weights (LoRA adapters remain trainable)
        if train_cfg.get("freeze", False):
            for p in reason_model.parameters():
                p.requires_grad = False
            if lora_cfg is not None:
                for n, p in reason_model.named_parameters():
                    if "lora_" in n:
                        p.requires_grad = True
        # Optional layer truncation
        num_layers = pred_cfg.get("num_layers")
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

    def _get_backbone(self):
        """Return underlying Transformer backbone (handles PEFT wrap)."""
        if hasattr(self.reason_model, "base_model"):
            inner = self.reason_model.base_model
            if hasattr(inner, "model"):
                return inner.model
            return inner
        if hasattr(self.reason_model, "model"):
            return self.reason_model.model
        return self.reason_model

    def _init_weights(self):
        for q in self.level_queries:
            nn.init.normal_(q, std=0.02)
        nn.init.normal_(self.lvl_embed.weight, std=0.02)
        for m in self.concept_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.back_proj.weight)

    def back_decode(self, concept_space_tensor):
        """Lift concept-space to encoder space: [.., D] -> [.., D_enc]."""
        return self.back_proj(concept_space_tensor)

    def _embed_questions(self, question_ids):
        """Embed question token ids: [B, L_Q] -> [B, L_Q, D_enc]."""
        return self._get_backbone().get_input_embeddings()(question_ids)

    def _build_scale_causal_mask_packed(
        self, q_len, s_len, level_lengths, T, dtype, device
    ):
        """Per-row scale-causal 4D additive mask for packed [Q|approx|S|pad].

        Mask semantics (per row i, with q_i = q_len[i], s_i = s_len[i]):

            scale assignment along the packed axis (pos t):
              0          if  0 <= t < q_i                          (Q)
              k+1        if  q_i + Σ_{j<k} L_j  <= t < q_i + Σ_{j≤k} L_j  (level-k approx)
              K+1        if  q_i + total_C <= t < q_i + total_C + s_i     (S)
              -1         otherwise (pad)

            Visibility rule for query t_q, key t_k:
              if either is pad → mask
              if same scale:
                scale 0 (Q)   → token-causal (t_q >= t_k)
                scales 1..K   → BIDIRECTIONAL within level
                scale K+1 (S) → token-causal (t_q >= t_k)
              else:
                scale_q >= scale_k  (scale-causal)

            Diagonal is always allowed to avoid all-masked rows on pad
            queries (NaN safeguard).

        Args:
            q_len:         [B] long, real Q length per row.
            s_len:         [B] long, real S length per row (zeros if no S).
            level_lengths: list of per-level approx-token counts (length K).
            T:             packed sequence length.
            dtype:         additive-mask float dtype.
            device:        torch device.

        Returns:
            [B, 1, T, T] additive mask: 0 = attend, finfo(dtype).min = mask.
        """
        B = int(q_len.shape[0])
        K = len(level_lengths)
        total_C = sum(level_lengths)

        # Position grid [B, T]
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)

        # Per-row region boundaries [B, 1]
        q_end = q_len.to(device).unsqueeze(1)
        sol_start = (q_len.to(device) + total_C).unsqueeze(1)
        sol_end = (q_len.to(device) + total_C + s_len.to(device)).unsqueeze(1)

        # scale_id [B, T]: -1 default (pad)
        scale_id = torch.full((B, T), -1, dtype=torch.long, device=device)
        # Q region: 0
        in_q = pos < q_end
        scale_id = torch.where(in_q, torch.zeros_like(scale_id), scale_id)
        # Per-level approx-token regions: k+1
        cum = 0
        for k, L_k in enumerate(level_lengths):
            lvl_start = q_end + cum
            lvl_end = q_end + cum + L_k
            in_lvl = (pos >= lvl_start) & (pos < lvl_end)
            scale_id = torch.where(in_lvl, torch.full_like(scale_id, k + 1), scale_id)
            cum += L_k
        # S region: K+1
        in_s = (pos >= sol_start) & (pos < sol_end)
        scale_id = torch.where(in_s, torch.full_like(scale_id, K + 1), scale_id)

        # Scale-causal visibility
        s_q = scale_id.unsqueeze(2)  # [B, T, 1]
        s_k = scale_id.unsqueeze(1)  # [B, 1, T]
        same = s_q == s_k
        cross_ok = s_q >= s_k

        pos_q = pos.unsqueeze(2)  # [B, T, 1]
        pos_k = pos.unsqueeze(1)  # [B, 1, T]
        # Token-causal applies inside Q (scale 0) and S (scale K+1)
        is_token_causal_scale = (s_q == 0) | (s_q == K + 1)
        same_token_causal = pos_q >= pos_k
        same_bidir = torch.ones_like(same_token_causal)
        same_ok = torch.where(is_token_causal_scale, same_token_causal, same_bidir)
        can_see = torch.where(same, same_ok, cross_ok)

        # Mask out pad keys (broadcast over query)
        pad_k = (scale_id < 0).unsqueeze(1)  # [B, 1, T]
        can_see = can_see & (~pad_k)

        # NaN safeguard: always allow self-attention (diagonal)
        eye = torch.eye(T, dtype=torch.bool, device=device).unsqueeze(0)
        can_see = can_see | eye

        neg_inf = torch.finfo(dtype).min
        mask_4d = torch.zeros(B, 1, T, T, dtype=dtype, device=device)
        mask_4d = mask_4d.masked_fill((~can_see).unsqueeze(1), neg_inf)
        return mask_4d

    # ------------------------------------------------------------------ #
    #  approx-token construction: pre-LLM level_queries × back_proj(f_hat)
    # ------------------------------------------------------------------ #

    def _construct_approx_tokens(self, level_idx, f_hat_k):
        """Construct approximation tokens for one pyramid level.

        Text-domain analog of VAR's spatial downsampling: extracts
        f_hat_k [B, L, D] → L_k approx tokens [B, L_k, D_enc] via
        learnable queries cross-attending to back_proj(f_hat_k).

        Args:
            level_idx: Pyramid level k in [0, K).
            f_hat_k: Cumulative reconstruction [B, ctx, D].

        Returns:
            (approx_tokens [B, L_k, D_enc], attn_weights [B, L_k, ctx]).
        """
        B = f_hat_k.shape[0]
        context = self.back_proj(f_hat_k)  # [B, ctx, D_enc]
        queries = self.level_queries[level_idx].unsqueeze(0).expand(B, -1, -1)
        if queries.dtype != context.dtype:
            queries = queries.to(context.dtype)

        q_n = self.query_norm(queries)
        c_n = self.context_norm(context)
        attn_out, attn_w = self.extract_attn(
            query=q_n,
            key=c_n,
            value=c_n,
            need_weights=True,
            average_attn_weights=True,
        )
        approx_tokens = self.post_norm(attn_out + queries)  # residual
        approx_tokens = approx_tokens + self.lvl_embed.weight[level_idx]
        return approx_tokens, attn_w

    # ------------------------------------------------------------------ #
    #  forward — single packed forward (training)                        #
    # ------------------------------------------------------------------ #

    def forward(self, batch: BuilderInput) -> PredictorOutput:
        """Training forward: BuilderInput → PredictorOutput.

        Phase 1: builder(batch) → gt_concepts + f_hats; tokenize Q (and S).
        Phase 2: Build approx tokens, pack [Q | approx tokens | S], single
                 LLM forward, read concept predictions at approx-token
                 positions and reasoning logits at solution-predicting
                 positions.
        """
        device = next(self.parameters()).device
        max_length = self.pyramid_cfg["max_seq_len"]
        K = self._num_levels
        total_C = self._total_concepts

        # ============================================================== #
        # Phase 1: Input preparation                                     #
        # ============================================================== #
        with torch.no_grad():
            pyramid = self.builder(batch)
        gt_concepts = [c.detach() for c in pyramid.concepts]
        gt_f_hats = [f.detach() for f in pyramid.f_hat_per_level]

        q_tokens = self.tokenizer(
            batch.questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        question_ids = q_tokens["input_ids"].to(device)
        q_mask = q_tokens["attention_mask"].to(device)

        if batch.has_solution:
            s_tokens = self.tokenizer(
                batch.solutions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            solution_ids = s_tokens["input_ids"].to(device)
            s_mask = s_tokens["attention_mask"].to(device)
        else:
            solution_ids = None
            s_mask = None

        B = question_ids.shape[0]

        # ============================================================== #
        # Phase 2: Single packed predictor forward                       #
        # ============================================================== #
        embed_layer = self._get_backbone().get_input_embeddings()
        Q_embeds = embed_layer(question_ids)  # [B, L_Q_pad, D_enc]

        # Build approximation tokens for all levels and concat
        approx_token_list = []
        for k in range(K):
            approx_tokens_k, _ = self._construct_approx_tokens(k, gt_f_hats[k])
            approx_token_list.append(approx_tokens_k)
        approx_tokens = torch.cat(approx_token_list, dim=1)  # [B, total_C, D_enc]
        if approx_tokens.dtype != Q_embeds.dtype:
            approx_tokens = approx_tokens.to(Q_embeds.dtype)

        # Solution embeddings (when supplied)
        if solution_ids is not None:
            S_embeds = embed_layer(solution_ids)
            if S_embeds.dtype != Q_embeds.dtype:
                S_embeds = S_embeds.to(Q_embeds.dtype)
        else:
            S_embeds = None

        # Pack per row: [real_Q_i | approx_tokens_0..K-1 | real_S_i | tail_pad]
        pack = pack_qcs_sequences(
            Q_embeds=Q_embeds,
            q_mask=q_mask,
            concept_embeds=approx_tokens,
            S_embeds=S_embeds,
            s_mask=s_mask,
        )

        # Per-row scale-causal 4D mask
        attention_mask_4d = self._build_scale_causal_mask_packed(
            q_len=pack.q_len,
            s_len=pack.s_len,
            level_lengths=self._level_lengths,
            T=pack.T,
            dtype=pack.packed_embeds.dtype,
            device=device,
        )

        # Single LLM forward (full reason_model: hidden + logits)
        model_out = self.reason_model(
            inputs_embeds=pack.packed_embeds,
            attention_mask=attention_mask_4d,
            output_hidden_states=True,
        )
        # Last-layer hidden states for concept readout
        if hasattr(model_out, "hidden_states") and model_out.hidden_states is not None:
            hidden = model_out.hidden_states[-1]
        else:
            hidden = model_out[0]
        logits = model_out.logits  # [B, T, V]

        # Concept readout AT approx-token positions: pos = q_len[i] + j (j in [0, total_C))
        arange_c = torch.arange(total_C, device=device)
        approx_row_idx = (
            torch.arange(B, device=device).unsqueeze(1).expand(B, total_C).contiguous()
        )
        approx_col_idx = pack.q_len.unsqueeze(1) + arange_c.unsqueeze(0)  # [B, total_C]
        approx_hidden = hidden[approx_row_idx, approx_col_idx]  # [B, total_C, D_enc]

        # Slice per level and project to concept space
        predicted_concepts: List[torch.Tensor] = []
        offset = 0
        for k in range(K):
            L_k = self._level_lengths[k]
            c_hat_k = self.concept_head(
                approx_hidden[:, offset : offset + L_k, :]
            )  # [B, L_k, D]
            predicted_concepts.append(c_hat_k)
            offset += L_k

        out = PredictorOutput(
            predicted_concepts=predicted_concepts,
            gt_concepts=gt_concepts,
            num_levels=K,
            level_lengths=list(self._level_lengths),
        )

        # Reasoning NTP supervision (S in input from the start)
        if pack.solution_col_idx is not None:
            out.reasoning_logits = gather_solution_logits(logits, pack)
            out.reasoning_target_ids = build_solution_targets(
                solution_ids, s_mask, pack
            )
            with torch.no_grad():
                predicted_ids = out.reasoning_logits.argmax(dim=-1)
                out.reasoning_texts = self.tokenizer.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )

        return out

    # ------------------------------------------------------------------ #
    #  inference — K sequential packed passes, self-maintained f_hat     #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _forward_inference(self, question_ids, question_attention_mask):
        """Autoregressive over levels with self-maintained f_hat.

        At each level k:
            1. Construct approx_tokens_k from back_proj(f_hat) via extract_attn
            2. Pack [real_Q_i | approx_tokens_0..k] per row, build per-row
               scale-causal mask, single backbone forward → hidden
            3. concept_head(hidden at level-k approx-token positions) → C_hat_k
            4. R_k = attn_weights^T @ C_hat_k; f_hat += R_k

        Args:
            question_ids: [B, L_Q].
            question_attention_mask: [B, L_Q] or None.

        Returns:
            PredictorOutput with predicted_concepts; gt_concepts=None.
        """
        B = question_ids.shape[0]
        device = question_ids.device
        L_canvas = self._inference_canvas_length

        if question_attention_mask is None:
            q_mask = torch.ones(
                B, question_ids.shape[1], device=device, dtype=torch.long
            )
        else:
            q_mask = question_attention_mask

        # Initialize f_hat canvas: [B, L_canvas, D]
        f_hat = torch.zeros(
            B,
            L_canvas,
            self._concept_dim,
            device=device,
            dtype=next(self.parameters()).dtype,
        )

        embed_layer = self._get_backbone().get_input_embeddings()
        Q_embeds = embed_layer(question_ids)  # [B, L_Q_pad, D_enc]
        backbone = self._get_backbone()

        predicted_concepts: List[torch.Tensor] = []
        approx_token_list: List[torch.Tensor] = []

        for k in range(self._num_levels):
            # Construct approx tokens for level k using current f_hat
            approx_tokens_k, attn_weights = self._construct_approx_tokens(k, f_hat)
            # approx_tokens_k: [B, L_k, D_enc]
            # attn_weights:    [B, L_k, L_canvas]
            approx_token_list.append(approx_tokens_k)

            # Concat approx_tokens_0..k for packing
            approx_tokens_0k = torch.cat(
                approx_token_list, dim=1
            )  # [B, Σ_{j≤k} L_j, D_enc]
            if approx_tokens_0k.dtype != Q_embeds.dtype:
                approx_tokens_0k = approx_tokens_0k.to(Q_embeds.dtype)

            level_lengths_in_seq = list(self._level_lengths[: k + 1])

            # Pack [real_Q_i | approx_tokens_0..k | tail_pad]
            pack = pack_qcs_sequences(
                Q_embeds=Q_embeds,
                q_mask=q_mask,
                concept_embeds=approx_tokens_0k,
                S_embeds=None,
                s_mask=None,
            )

            # Per-row scale-causal mask (no S region — s_len = 0)
            attention_mask_4d = self._build_scale_causal_mask_packed(
                q_len=pack.q_len,
                s_len=pack.s_len,
                level_lengths=level_lengths_in_seq,
                T=pack.T,
                dtype=pack.packed_embeds.dtype,
                device=device,
            )

            # Backbone forward (request hidden states; the PEFT-wrapped
            # AutoModelForCausalLM returns CausalLMOutputWithPast which has no
            # `last_hidden_state` attribute — only `hidden_states` when
            # `output_hidden_states=True`).
            backbone_out = backbone(
                inputs_embeds=pack.packed_embeds,
                attention_mask=attention_mask_4d,
                output_hidden_states=True,
            )
            if (
                hasattr(backbone_out, "hidden_states")
                and backbone_out.hidden_states is not None
            ):
                hidden = backbone_out.hidden_states[-1]
            elif hasattr(backbone_out, "last_hidden_state"):
                hidden = backbone_out.last_hidden_state
            else:
                hidden = backbone_out[0]

            # Extract C_hat_k at level-k approx-token positions:
            #   pos[i] = q_len[i] + Σ_{j<k} L_j ... + L_k - 1
            L_k = self._level_lengths[k]
            prev = sum(self._level_lengths[j] for j in range(k))
            arange_lk = torch.arange(L_k, device=device)
            row_idx = (
                torch.arange(B, device=device).unsqueeze(1).expand(B, L_k).contiguous()
            )
            col_idx = pack.q_len.unsqueeze(1) + prev + arange_lk.unsqueeze(0)
            hidden_k = hidden[row_idx, col_idx]  # [B, L_k, D_enc]
            c_hat_k = self.concept_head(hidden_k)  # [B, L_k, D]
            predicted_concepts.append(c_hat_k)

            # Reconstruct and accumulate into f_hat
            R_k = torch.bmm(attn_weights.transpose(1, 2), c_hat_k)
            f_hat = f_hat + R_k

        return PredictorOutput(
            predicted_concepts=predicted_concepts,
            gt_concepts=None,
            num_levels=self._num_levels,
            level_lengths=list(self._level_lengths),
        )

    # ------------------------------------------------------------------ #
    #  generate_solution — free autoregressive text generation            #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate_solution(
        self,
        predicted_concepts,
        question_ids,
        question_attention_mask=None,
        max_new_tokens=256,
    ):
        """Free autoregressive generation of solution from [Q, Concepts].

        Uses HuggingFace .generate() with inputs_embeds.

        Args:
            predicted_concepts: List of K tensors, each [B, L_k, D].
            question_ids: [B, L_Q].
            question_attention_mask: [B, L_Q] or None.
            max_new_tokens: Max tokens to generate.

        Returns:
            List of B generated strings.
        """
        B = question_ids.shape[0]
        device = question_ids.device

        concepts_flat = torch.cat(predicted_concepts, dim=1)  # [B, total_C, D]
        concept_embeds = self.back_decode(concepts_flat)  # [B, total_C, D_enc]

        Q_embeds = self._embed_questions(question_ids)
        if concept_embeds.dtype != Q_embeds.dtype:
            concept_embeds = concept_embeds.to(Q_embeds.dtype)

        if question_attention_mask is None:
            q_mask = torch.ones(
                B, question_ids.shape[1], device=device, dtype=torch.long
            )
        else:
            q_mask = question_attention_mask

        pack = pack_qcs_sequences(
            Q_embeds=Q_embeds,
            q_mask=q_mask,
            concept_embeds=concept_embeds,
            S_embeds=None,
            s_mask=None,
        )

        generated_ids = self.reason_model.generate(
            inputs_embeds=pack.packed_embeds,
            attention_mask=pack.packed_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,
        )

        input_len = pack.packed_embeds.shape[1]
        if generated_ids.shape[1] > input_len:
            new_ids = generated_ids[:, input_len:]
        else:
            new_ids = generated_ids
        return self.tokenizer.batch_decode(new_ids, skip_special_tokens=True)
