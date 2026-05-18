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
  ║  │  sb_query_head: Sequential(                           │              ║
  ║  │      Linear(D_enc, D_enc), GELU, Linear(D_enc, D))    │              ║
  ║  │    Parallel head to concept_head (V2 design):         │              ║
  ║  │    h_k → sb_q_k → Canvas.predict_soft_boundaries.     │              ║
  ║  │    Decouples sb prediction from concept prediction —  │              ║
  ║  │    breaks causal inversion + train/infer dist shift.  │              ║
  ║  │                                                       │              ║
  ║  │  canvas: Canvas(D, K, L_canvas) — VAR-style upsample  │              ║
  ║  │    • pad_f_hat:        L_real → L_canvas (zero-pad)   │              ║
  ║  │    • pad_soft_boundaries: gt_sb key-dim → L_canvas    │              ║
  ║  │    • predict_soft_boundaries(queries, k) → pred_sb    │              ║
  ║  │      (queries from sb_query_head; keys = canvas_k_proj │              ║
  ║  │       (sinusoidal_pos + lvl_embed[k]); + level_pos_bias)│              ║
  ║  │    • reconstruct: R_k = sb^T @ concepts (builder fmla)│              ║
  ║  │    Trained analog of builder's soft_boundaries; closes│              ║
  ║  │    train/inference gap on f_hat updates (no CoT at    │              ║
  ║  │    inference → no builder soft_boundaries available). │              ║
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
    L_canvas= canvas_length (sized per dataset CoT distribution; e.g., 256 for GSM8K, 512 for MATH)
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
  │    gt_sb_k     = pyramid.level_outputs[k].attention_weights            │
  │                  [B, L_k, L_CoT]   (builder's soft_boundaries)        │
  │    (all detached — no gradient flows back to builder)                   │
  │                                                                        │
  │  canvas.pad_f_hat(gt_f_hats[k], f_hat_mask)                            │
  │    → padded_gt_f_hats[k] [B, L_canvas, D], f_hat_mask_padded            │
  │  canvas.pad_soft_boundaries(gt_sb_k)                                   │
  │    → gt_sb_padded[k]    [B, L_k, L_canvas]                              │
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
  │  Step 2.6: Concept readout & per-level dual heads (V2)                  │
  │  ──────────────────────────                                            │
  │    For row i, approx-token positions are:                              │
  │      col = q_len[i] + j,  j ∈ [0, total_C)                            │
  │    approx_hidden = hidden[i, q_len[i]:q_len[i]+total_C]  [B,total_C,D_enc]│
  │    For each level k (offset = Σ_{j<k} L_j):                             │
  │      h_k     = approx_hidden[:, offset:offset+L_k, :]   [B, L_k, D_enc]  │
  │      C_hat_k = concept_head(h_k)                         [B, L_k, D]    │
  │      sb_q_k  = sb_query_head(h_k)                        [B, L_k, D]    │
  │    Concept and sb queries are SIBLINGS of h_k — no parent-child chain.  │
  │                                                                        │
  │  Step 2.7: Reasoning NTP supervision                                   │
  │  ───────────────────────────────────                                   │
  │    reasoning_logits    = gather_solution_logits(logits, pack)           │
  │    reasoning_target_ids = build_solution_targets(solution_ids, s_mask)  │
  │                                                                        │
  │  Step 2.8: Canvas soft_boundaries prediction (V2 multi-task)            │
  │  ─────────────────────────────────────────────────────────────   │
  │    For each level k, predict soft_boundaries from h_k (NOT concepts):   │
  │      sb_q_k    = sb_query_head(h_k)             [B, L_k, D]             │
  │      pred_sb_k = canvas.predict_soft_boundaries(sb_q_k, k)              │
  │                                       [B, L_k, L_canvas]               │
  │    Loss: MSE(pred_sb_k, gt_sb_padded[k]) on the FULL canvas —           │
  │    zero-padded gt drives self-suppression of the tail at inference.    │
  │    Train↔Inference parity: h_k is computed identically in both phases  │
  │    (both feed f_hat → approx_tokens → LLM); no gt_concepts on this path.│
  │                                                                        │
  └──────────────────────────────────────────────────────────────────────────┘
                │
                ▼
  PredictorOutput:
    predicted_concepts:    [C_hat_0, ..., C_hat_{K-1}]    (for concept loss)
    gt_concepts:           [C_0, ..., C_{K-1}]            (from builder)
    pred_soft_boundaries:  [pred_sb_0, ..., pred_sb_{K-1}] (canvas prediction)
    gt_soft_boundaries:    [gt_sb_0, ..., gt_sb_{K-1}]     (zero-padded gt)
    reasoning_logits:      [B, L_sol, V]                  (for CE loss)
    reasoning_target_ids:  [B, L_sol]                     (shifted targets)

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
  ║     (extract_attn's attn_weights are NOT used for reconstruction;     ║
  ║      they only carry the f_hat → approx_tokens routing for the LLM)  ║
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
  ║     C_hat_k  = concept_head(hidden_k)          [B, L_k, D]             ║
  ║     sb_q_k   = sb_query_head(hidden_k)         [B, L_k, D]             ║
  ║     (V2: dual heads share h_k=hidden_k; no concepts → sb chain.)        ║
  ║                                                                        ║
  ║  ⑤ Canvas reconstruct & accumulate (replaces builder's H_rest path):    ║
  ║     pred_sb_k = canvas.predict_soft_boundaries(sb_q_k, k)                ║
  ║                                                  [B, L_k, L_canvas]    ║
  ║     R_k       = Canvas.reconstruct(C_hat_k, pred_sb_k)                  ║
  ║                = pred_sb_k^T @ C_hat_k         [B, L_canvas, D]         ║
  ║     f_hat = f_hat + R_k                                                 ║
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

  The attention weights α_k from extract_attn are NOT used for f_hat
  reconstruction. Reconstruction goes through the dedicated Canvas module
  whose queries come from sb_query_head(h_k) (a parallel head to
  concept_head, both consuming the same LLM hidden state). Canvas is
  supervised by builder's gt_sb during training:
      sb_q_k    = sb_query_head(h_k)
      pred_sb_k = canvas.predict_soft_boundaries(sb_q_k, k)
      R_k       = pred_sb_k^T @ concepts_k     [B, L_canvas, D]

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
    canvas_length:           REQUIRED. L_canvas for f_hat (used in train + inference).
                             MUST be sized per dataset CoT distribution to avoid silent
                             truncation in Canvas.pad_f_hat / pad_soft_boundaries.
                             Recommended: GSM8K=256, MATH=512.
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

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from lcp.data_loader import BuilderInput
from lcp.concept_builder import PyramidOutput
from lcp.utils import (
    build_solution_targets,
    gather_solution_logits,
    pack_qcs_sequences,
)


@dataclass
class PredictorOutput:
    """Full output of ConceptPredictor.forward().

    Canvas (VAR-style upsampling) tensors are populated when the
    Canvas module is in use:
        pred_soft_boundaries: List of [B, L_k, L_canvas] per level
            — predictor's learned spatial routing (concept → canvas).
        gt_soft_boundaries:   List of [B, L_k, L_canvas] per level
            — builder's soft_boundaries, zero-padded on the tail to
            L_canvas. Supervision target for canvas loss (MSE on full
            canvas drives self-suppressing tail behavior).

    Both default to None at inference (no builder pyramid available).
    """

    predicted_concepts: List[torch.Tensor]
    gt_concepts: Optional[List[torch.Tensor]] = None
    num_levels: int = 0
    level_lengths: List[int] = field(default_factory=list)
    reasoning_logits: Optional[torch.Tensor] = None
    reasoning_target_ids: Optional[torch.Tensor] = None
    reasoning_texts: Optional[List[str]] = None
    generation_texts: Optional[List[str]] = None
    # Canvas supervision (VAR upsampling) — present when Canvas is active.
    pred_soft_boundaries: Optional[List[torch.Tensor]] = None
    gt_soft_boundaries: Optional[List[torch.Tensor]] = None


def _build_sinusoidal_pos(L: int, D: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding [L, D] (length-agnostic).

    Length-agnostic in the sense that it is precomputed up to L and works
    for any sub-length on subsequent calls (no learnable parameters tied
    to the seq dimension).
    """
    position = torch.arange(L, dtype=torch.float32).unsqueeze(1)  # [L, 1]
    div_term = torch.exp(
        torch.arange(0, D, 2, dtype=torch.float32) * -(math.log(10000.0) / D)
    )  # [D/2]
    pe = torch.zeros(L, D)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class Canvas(nn.Module):
    """VAR-style canvas operations for the predictor.

    PURPOSE — close the train/inference gap on f_hat updates.
        The Builder updates f_hat via
            R_k = soft_boundaries^T @ concepts   (concept_builder.py L1248-1249)
        where soft_boundaries is computed from the residual CoT (H_rest).
        At Predictor inference there is NO CoT, so soft_boundaries cannot
        be reproduced. The Canvas module fills this gap with a TRAINED
        concepts → spatial-distribution mapping that is supervised against
        the Builder's gt_soft_boundaries during training.

    SCOPE — all canvas-related operations live here:
        * pad variable-length f_hats and gt_soft_boundaries to L_canvas
        * predict pred_soft_boundaries from concepts (the missing
          "upsample" operator that mirrors builder's spatial routing)
        * deterministically reconstruct R_k via builder's formula
          R_k = pred_sb^T @ concepts (identical arithmetic; the only
          difference is sb is predicted instead of computed from H_rest)
        * provide an MSE-based canvas loss helper used by the loss module

    TRAIN/INFERENCE UNIFICATION:
        L_canvas is fixed at construction; both phases run on this length.
        Training: gt_f_hats and gt_sb (variable L_CoT from builder) are
            zero-padded on tail to L_canvas. cross-attention's
            key_padding_mask correctly ignores padding positions inside
            the predictor's _construct_approx_tokens.
        Inference: f_hat starts as zeros [B, L_canvas, D] and is updated
            via R_k = canvas.reconstruct(c_hat_k, canvas.predict_soft_boundaries(...)).
        The Canvas module is the SAME in both phases — same parameters,
        same forward path, same shape.

    SELF-SUPPRESSING TAIL:
        gt_soft_boundaries (zero-padded to L_canvas on the tail) gives
        zero target mass on positions beyond the real L_CoT. MSE loss on
        the FULL canvas (no masking out of padding) pushes pred_sb to
        also output zero at tail positions. At inference, the trained
        module emits ~zero pred_sb on the tail → ~zero R_k contribution
        → effectively ignores the excess canvas length. The model thus
        learns to allocate spatial information only where needed,
        regardless of the configured L_canvas.

    DESIGN (V2 — multi-task heads from LLM hidden states):
        Canvas is QUERY-AGNOSTIC. The query bank is produced upstream by
        the predictor's `sb_query_head` (a parallel head to `concept_head`)
        operating on the LLM hidden states `h_k` at level-k approx-token
        positions. Both train and inference feed the SAME h_k distribution
        into Canvas — no teacher-forcing distribution shift on this path.

        keys[t]    = canvas_k_proj(sinusoidal_pos[t] + lvl_embed[k])
        queries    = sb_query_head(h_k)             (computed by predictor)
        logits     = queries @ keys^T / sqrt(D)
        pred_sb    = softmax(logits + level_pos_bias[k])   # over L_canvas
        R_k        = pred_sb^T @ concepts                  # builder formula

        `level_pos_bias[k]` is a learnable per-level positional bias on the
        canvas axis (analogous to T5/ALiBi relative bias). It encodes the
        per-level routing prior — coarse levels can drift toward broad
        distributions, fine levels toward concentrated ones — without
        forcing the dot-product head to learn this from scratch.

    Args:
        concept_dim: D, the concept-space dimension (== pyramid hidden_dim).
        num_levels:  K, the number of pyramid levels.
        L_canvas:    fixed canvas length used in both training and inference.
    """

    # One-shot warning latch for the silent-truncation branch in
    # ``pad_f_hat`` (when L_real > L_canvas). Class-level so multiple
    # Canvas instances share the same suppression state across a run.
    _truncation_warned: bool = False

    def __init__(
        self,
        concept_dim: int,
        num_levels: int,
        L_canvas: int,
    ):
        super().__init__()
        if L_canvas <= 0:
            raise ValueError(f"L_canvas must be > 0, got {L_canvas}")
        self.concept_dim = concept_dim
        self.num_levels = num_levels
        self.L_canvas = L_canvas
        D = concept_dim
        self.scale = D**-0.5

        # Per-level identity tag added to canvas keys (mirrors builder's
        # implicit per-level distinction via separate query banks).
        self.lvl_embed = nn.Embedding(num_levels, D)

        # Canvas key projection (single-head attention K-side).
        # The Q-side projection is owned by the predictor (sb_query_head),
        # not by Canvas — this is a deliberate decoupling so Canvas does
        # not assume any particular query origin.
        self.canvas_k_proj = nn.Linear(D, D, bias=False)

        # Per-level learnable positional bias on canvas axis (V2 add-on).
        # Shape: [num_levels, L_canvas]. Zero-init so the softmax is purely
        # Q·K driven at start; the bias adapts during training to encode
        # per-level routing priors (broad vs. concentrated).
        self.level_pos_bias = nn.Embedding(num_levels, L_canvas)
        nn.init.zeros_(self.level_pos_bias.weight)

        # Sinusoidal canvas position embeddings (length-agnostic, no params).
        self.register_buffer(
            "_canvas_pos_embed",
            _build_sinusoidal_pos(L_canvas, D),
            persistent=False,
        )

    # ── Padding utilities ────────────────────────────────────────────

    def pad_f_hat(
        self,
        f_hat: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zero-pad / truncate f_hat to L_canvas; extend mask correspondingly.

        Args:
            f_hat: [B, L_real, D] cumulative-reconstruction snapshot from
                builder (variable L_real == L_CoT_batch_max).
            mask:  [B, L_real] valid-token mask (1=valid, 0=pad) or None.

        Returns:
            (f_hat_padded [B, L_canvas, D], mask_padded [B, L_canvas]).
            mask_padded is always a long tensor with 1=valid, 0=pad.
        """
        B, L_real, _ = f_hat.shape
        L = self.L_canvas
        device = f_hat.device

        if L_real == L:
            mask_padded = (
                mask
                if mask is not None
                else torch.ones(B, L, device=device, dtype=torch.long)
            )
            return f_hat, mask_padded

        if L_real > L:
            # Truncate (rare; happens only when CoT exceeds L_canvas).
            # Silent truncation is a real correctness loss — the tail of
            # the reasoning is discarded and gt_concepts / canvas
            # supervision no longer cover the original CoT. Emit a one-
            # shot RuntimeWarning so operators see this in the log
            # without spamming every batch.
            if not Canvas._truncation_warned:
                Canvas._truncation_warned = True
                import warnings

                warnings.warn(
                    f"Canvas.pad_f_hat: observed f_hat length L_real={L_real} "
                    f"> L_canvas={L}; tail positions [{L}:{L_real}] are being "
                    f"discarded (silent information loss). Increase "
                    f"'model.canvas.canvas_length' in the predictor YAML to "
                    f"match the dataset's typical CoT length distribution. "
                    f"(Suppressing further occurrences.)",
                    RuntimeWarning,
                    stacklevel=2,
                )
            f_hat_padded = f_hat[:, :L, :].contiguous()
            mask_padded = (
                mask[:, :L].contiguous()
                if mask is not None
                else torch.ones(B, L, device=device, dtype=torch.long)
            )
            return f_hat_padded, mask_padded

        # L_real < L: zero-pad along seq dim
        pad_zeros = torch.zeros(
            B, L - L_real, f_hat.shape[-1], device=device, dtype=f_hat.dtype
        )
        f_hat_padded = torch.cat([f_hat, pad_zeros], dim=1)
        if mask is not None:
            mask_pad = torch.zeros(B, L - L_real, device=device, dtype=mask.dtype)
            mask_padded = torch.cat([mask, mask_pad], dim=1)
        else:
            # Synthesize a [valid_real | pad] mask from L_real boundary.
            mask_padded = torch.cat(
                [
                    torch.ones(B, L_real, device=device, dtype=torch.long),
                    torch.zeros(B, L - L_real, device=device, dtype=torch.long),
                ],
                dim=1,
            )
        return f_hat_padded, mask_padded

    def pad_soft_boundaries(self, sb: torch.Tensor) -> torch.Tensor:
        """Zero-pad / truncate gt_soft_boundaries to L_canvas on key dim.

        Args:
            sb: [B, L_k, L_real] builder's soft attention. Mass is already
                concentrated on valid CoT positions (builder's softmax
                respects key_padding_mask), so zero-padding on the tail
                is a clean truncation of the support set.

        Returns:
            sb_padded: [B, L_k, L_canvas] with zeros on positions beyond
            L_real. Sums (over the canvas axis) are preserved at ~1.0
            on rows that have any valid keys, and 0.0 on fully-masked rows.
        """
        B, L_k, L_real = sb.shape
        L = self.L_canvas
        if L_real == L:
            return sb
        if L_real > L:
            return sb[:, :, :L].contiguous()
        pad = torch.zeros(B, L_k, L - L_real, device=sb.device, dtype=sb.dtype)
        return torch.cat([sb, pad], dim=-1)

    # ── Predict pred_soft_boundaries ─────────────────────────────────

    def predict_soft_boundaries(
        self,
        queries: torch.Tensor,
        level_idx: int,
    ) -> torch.Tensor:
        """Predict pred_soft_boundaries [B, L_k, L_canvas] from queries.

        Single-head attention: queries come from the predictor's
        `sb_query_head(h_k)` (LLM hidden states at level-k approx-token
        positions), keys are projected (sinusoidal_pos + lvl_embed[k]).
        Softmax over the L_canvas key axis yields a per-position
        distribution over canvas slots — the predictor's learned analog
        of builder's soft_boundaries.

        Why queries come from h_k (not from concepts):
            (1) breaks the causal inversion (concepts are an EFFECT of
                soft_boundaries in builder; predicting cause from effect
                is the wrong direction).
            (2) eliminates compounding errors at inference: a bad C_hat_k
                no longer pollutes pred_sb_k since both are siblings of
                h_k, not parent-child.
            (3) eliminates teacher-forcing distribution shift: h_k is
                computed identically at train and inference (both feed
                f_hat → approx_tokens → LLM).
            (4) h_k is strictly richer than C_hat_k (it carries Q + prior
                levels + RoPE positional context via causal attention).

        Args:
            queries: [B, L_k, D] precomputed queries in concept space
                (output of the predictor's sb_query_head).
            level_idx: int in [0, num_levels). Selects lvl_embed and
                level_pos_bias rows.

        Returns:
            pred_sb: [B, L_k, L_canvas] softmax distribution per query
                over canvas positions.
        """
        if not (0 <= level_idx < self.num_levels):
            raise IndexError(
                f"level_idx {level_idx} out of range [0, {self.num_levels})"
            )
        # Build canvas keys in the dtype of queries (handle bf16/fp16).
        pos = self._canvas_pos_embed.to(dtype=queries.dtype)  # [L, D]
        lvl = self.lvl_embed.weight[level_idx].to(dtype=queries.dtype)  # [D]
        canvas_keys = self.canvas_k_proj(pos + lvl)  # [L_canvas, D]

        # logits: [B, L_k, L_canvas]
        logits = torch.matmul(queries, canvas_keys.transpose(0, 1)) * self.scale
        # Add per-level positional bias on canvas axis (broadcasts over B, L_k).
        bias = self.level_pos_bias.weight[level_idx].to(dtype=queries.dtype)
        logits = logits + bias  # [B, L_k, L_canvas]
        return F.softmax(logits, dim=-1)

    # ── Reconstruction (builder formula, deterministic) ──────────────

    @staticmethod
    def reconstruct(
        concepts: torch.Tensor,
        soft_boundaries: torch.Tensor,
    ) -> torch.Tensor:
        """R_k = soft_boundaries^T @ concepts. Mirrors builder L1248-1249.

        Args:
            concepts:        [B, L_k, D]
            soft_boundaries: [B, L_k, L_canvas]

        Returns:
            R_k: [B, L_canvas, D]
        """
        return torch.bmm(soft_boundaries.transpose(1, 2), concepts)

    # ── Loss helper ──────────────────────────────────────────────────

    @staticmethod
    def canvas_mse(
        pred_sb: torch.Tensor,
        gt_sb_padded: torch.Tensor,
    ) -> torch.Tensor:
        """MSE on the FULL canvas (zero-padded gt drives tail suppression).

        Crucial choice: NO masking. By including padding positions
        (gt_sb=0 there) in the MSE, we explicitly train pred_sb to output
        zero on the tail. Masking those positions out would leave tail
        behavior undefined — exactly the failure mode this module fixes.
        """
        return F.mse_loss(pred_sb, gt_sb_padded)


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
        self._canvas_length = self.predictor_cfg["canvas_length"]

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

        # Soft-boundaries query head: D_enc → D (parallel to concept_head).
        # Operates on the SAME LLM hidden state h_k at level-k approx-token
        # positions, producing a query bank for Canvas.predict_soft_boundaries.
        # This is the V2 multi-task head design — concepts and soft_boundaries
        # are siblings of h_k (not parent-child), which:
        #   - breaks the causal inversion (concepts are an effect of sb,
        #     so predicting sb from concepts is the wrong direction),
        #   - eliminates compounding errors at inference (no C_hat_k → sb chain),
        #   - eliminates teacher-forcing distribution shift (h_k is identical
        #     at train and inference; gt_concepts no longer enters this path).
        self.sb_query_head = nn.Sequential(
            nn.Linear(D_enc, D_enc),
            nn.GELU(),
            nn.Linear(D_enc, concept_dim),
        )

        # Canvas: VAR-style upsampling module (predict soft_boundaries)
        self.canvas = Canvas(
            concept_dim=concept_dim,
            num_levels=num_levels,
            L_canvas=self._canvas_length,
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
        if train_cfg["freeze"]:
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
        for m in self.sb_query_head:
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

    def _construct_approx_tokens(
        self, level_idx, f_hat_k, f_hat_mask=None, return_weights: bool = False
    ):
        """Construct approximation tokens for one pyramid level.

        Text-domain analog of VAR's spatial downsampling: extracts
        f_hat_k [B, L, D] → L_k approx tokens [B, L_k, D_enc] via
        learnable queries cross-attending to back_proj(f_hat_k).

        PADDING SAFETY:
            f_hat at padding positions is zero (builder masks guarantee this).
            However, context_norm(back_proj(zeros)) ≠ 0 due to LayerNorm bias.
            Without key_padding_mask, the cross-attention would attend to these
            spurious constant-vector positions, causing information leakage.
            When f_hat_mask is provided, padding positions are excluded from
            key/value attention computation.

        Args:
            level_idx: Pyramid level k in [0, K).
            f_hat_k: Cumulative reconstruction [B, ctx, D].
            f_hat_mask: Optional [B, ctx] mask where 1=valid, 0=padding.
                At inference (self-maintained canvas) this is None (all valid).
            return_weights: If True, run MHA with need_weights=True and return
                the (B, L_k, ctx) attention probabilities. Production callers
                (training forward / inference loop) pass False since the
                Canvas module owns the soft-boundary path. Diagnostic / unit
                tests pass True to verify cross-attention correctness.

        Returns:
            (approx_tokens [B, L_k, D_enc], attn_weights or None).
            attn_weights is None when return_weights=False, else
            [B, L_k, ctx] (averaged over heads).
        """
        B = f_hat_k.shape[0]
        context = self.back_proj(f_hat_k)  # [B, ctx, D_enc]
        queries = self.level_queries[level_idx].unsqueeze(0).expand(B, -1, -1)
        if queries.dtype != context.dtype:
            queries = queries.to(context.dtype)

        q_n = self.query_norm(queries)
        c_n = self.context_norm(context)

        # key_padding_mask: True = IGNORE position (PyTorch MHA convention).
        # This prevents cross-attention from attending to padding positions
        # where context_norm produces a non-zero constant (LayerNorm bias).
        kpm = None
        if f_hat_mask is not None:
            kpm = f_hat_mask == 0  # [B, ctx] -> True at padding positions

        # need_weights gated by return_weights: production callers skip
        # weight materialization (Canvas owns the soft-boundary path),
        # avoiding the O(B * L_k * ctx) probability tensor and the MHA
        # backward through it. Tests/diagnostics opt in via return_weights.
        attn_out, attn_w = self.extract_attn(
            query=q_n,
            key=c_n,
            value=c_n,
            key_padding_mask=kpm,
            need_weights=return_weights,
            average_attn_weights=True,
        )
        approx_tokens = self.post_norm(attn_out + queries)  # residual
        approx_tokens = approx_tokens + self.lvl_embed.weight[level_idx]
        return approx_tokens, attn_w

    # ------------------------------------------------------------------ #
    #  forward — single packed forward (training)                        #
    # ------------------------------------------------------------------ #

    def forward(
        self, batch: BuilderInput, *, pyramid: Optional[PyramidOutput] = None
    ) -> PredictorOutput:
        """Training forward: BuilderInput → PredictorOutput.

        Phase 1: builder(batch) → gt_concepts + f_hats; tokenize Q (and S).
        Phase 2: Build approx tokens, pack [Q | approx tokens | S], single
                 LLM forward, read concept predictions at approx-token
                 positions and reasoning logits at solution-predicting
                 positions.

        Args:
            batch: BuilderInput with questions, cot_answers, solutions.
            pyramid: Optional pre-computed PyramidOutput from a frozen
                builder call (e.g., builder(_strip_solutions(batch))).
                When provided, the internal builder call is skipped.
                This is the recommended usage in training loops where the
                builder is invoked separately for efficiency (avoids
                running _prepare_reasoning on solutions the builder
                doesn't need).
        """
        device = next(self.parameters()).device
        max_length = self.pyramid_cfg["max_seq_len"]
        K = self._num_levels
        total_C = self._total_concepts

        # ============================================================== #
        # Phase 1: Input preparation                                     #
        # ============================================================== #
        if pyramid is None:
            with torch.no_grad():
                pyramid = self.builder(batch)
        gt_concepts = [c.detach() for c in pyramid.concepts]
        gt_f_hats = [f.detach() for f in pyramid.f_hat_per_level]
        # CoT padding mask from builder: [B, L_CoT] where 1=valid, 0=pad.
        # Passed to _construct_approx_tokens so cross-attention ignores padding
        # positions in f_hat (LayerNorm turns zeros into non-zero constants).
        f_hat_mask = pyramid.attention_mask  # [B, L_CoT] or None

        # ── Canvas: pad gt_f_hats to L_canvas for train/infer unification ──
        # Also extract gt_soft_boundaries from builder's level_outputs.
        gt_sb_list: List[torch.Tensor] = []  # padded gt_soft_boundaries per level
        pred_sb_list: List[torch.Tensor] = []  # predicted soft_boundaries per level
        padded_gt_f_hats: List[torch.Tensor] = []
        f_hat_mask_padded: Optional[torch.Tensor] = None
        for k in range(K):
            # Pad f_hat_k to L_canvas
            f_hat_k_padded, mask_k_padded = self.canvas.pad_f_hat(
                gt_f_hats[k], f_hat_mask
            )
            padded_gt_f_hats.append(f_hat_k_padded)
            if k == 0:
                f_hat_mask_padded = mask_k_padded
            # Extract and pad gt_soft_boundaries from builder
            gt_sb_k = pyramid.level_outputs[
                k
            ].attention_weights.detach()  # [B, L_k, L_CoT]
            gt_sb_k_padded = self.canvas.pad_soft_boundaries(
                gt_sb_k
            )  # [B, L_k, L_canvas]
            gt_sb_list.append(gt_sb_k_padded)

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

        # Build approximation tokens for all levels and concat.
        # f_hat_mask_padded ensures cross-attention ignores padding positions.
        approx_token_list = []
        for k in range(K):
            approx_tokens_k, _ = self._construct_approx_tokens(
                k, padded_gt_f_hats[k], f_hat_mask_padded
            )
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
        # (V2) Two parallel heads consume the SAME h_k:
        #   * concept_head    → C_hat_k       (concept prediction)
        #   * sb_query_head   → sb_q_k        (queries for Canvas.pred_sb)
        # Both heads back-propagate into h_k — multi-task regularization.
        predicted_concepts: List[torch.Tensor] = []
        offset = 0
        for k in range(K):
            L_k = self._level_lengths[k]
            h_k = approx_hidden[:, offset : offset + L_k, :]  # [B, L_k, D_enc]
            c_hat_k = self.concept_head(h_k)  # [B, L_k, D]
            predicted_concepts.append(c_hat_k)

            # Canvas: predict soft_boundaries from h_k via sb_query_head.
            # No teacher-forcing distribution shift on this path: h_k is
            # computed identically at train and inference time.
            sb_q_k = self.sb_query_head(h_k)  # [B, L_k, D]
            pred_sb_k = self.canvas.predict_soft_boundaries(
                sb_q_k, level_idx=k
            )  # [B, L_k, L_canvas]
            pred_sb_list.append(pred_sb_k)

            offset += L_k

        out = PredictorOutput(
            predicted_concepts=predicted_concepts,
            gt_concepts=gt_concepts,
            num_levels=K,
            level_lengths=list(self._level_lengths),
            pred_soft_boundaries=pred_sb_list,
            gt_soft_boundaries=gt_sb_list,
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
            3. Slice hidden_k at level-k approx-token positions; the SAME
               h_k feeds two parallel heads (V2 multi-task design):
                  C_hat_k = concept_head(hidden_k)
                  sb_q_k  = sb_query_head(hidden_k)
            4. pred_sb_k = canvas.predict_soft_boundaries(sb_q_k, k)
               R_k       = Canvas.reconstruct(C_hat_k, pred_sb_k)
               f_hat    += R_k
               (extract_attn's attn_weights are NOT used here; reconstruction
                goes through the dedicated Canvas module which is supervised
                during training against builder's gt_soft_boundaries. Queries
                come from h_k via sb_query_head — NOT from C_hat_k — so the
                inference path has no compounding error chain from concept
                prediction into soft_boundaries prediction.)

        Args:
            question_ids: [B, L_Q].
            question_attention_mask: [B, L_Q] or None.

        Returns:
            PredictorOutput with predicted_concepts; gt_concepts=None.
        """
        B = question_ids.shape[0]
        device = question_ids.device
        L_canvas = self._canvas_length

        if question_attention_mask is None:
            q_mask = torch.ones(
                B, question_ids.shape[1], device=device, dtype=torch.long
            )
        else:
            q_mask = question_attention_mask

        # Initialize f_hat canvas: [B, L_canvas, D]
        # No f_hat_mask needed: canvas is self-maintained with no padding
        # (all L_canvas positions are valid workspace for accumulation).
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
            approx_tokens_k, _ = self._construct_approx_tokens(k, f_hat)
            # approx_tokens_k: [B, L_k, D_enc]
            # (extract_attn weights intentionally discarded; the Canvas
            # module owns the f_hat update path via predict_soft_boundaries.)
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
            # (V2) Two parallel heads consume the SAME h_k = hidden_k:
            #   * concept_head  → C_hat_k
            #   * sb_query_head → sb_q_k (queries for Canvas.predict_sb)
            c_hat_k = self.concept_head(hidden_k)  # [B, L_k, D]
            predicted_concepts.append(c_hat_k)

            sb_q_k = self.sb_query_head(hidden_k)  # [B, L_k, D]
            pred_sb_k = self.canvas.predict_soft_boundaries(sb_q_k, level_idx=k)
            R_k = Canvas.reconstruct(c_hat_k, pred_sb_k)  # [B, L_canvas, D]
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
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
    ):
        """Free autoregressive generation of solution from [Q, Concepts].

        Uses HuggingFace .generate() with inputs_embeds.

        Args:
            predicted_concepts: List of K tensors, each [B, L_k, D].
            question_ids: [B, L_Q].
            question_attention_mask: [B, L_Q] or None.
            max_new_tokens: Max tokens to generate.
            do_sample: If True, sample with temperature/top_k/top_p; if
                False, use greedy decoding (sampling kwargs are ignored).
            temperature: Softmax temperature for sampling. Used iff
                ``do_sample`` is True.
            top_k: Top-k truncation size for sampling. Used iff
                ``do_sample`` is True.
            top_p: Nucleus sampling probability. Used iff ``do_sample``
                is True.

        All five generation knobs are REQUIRED (keyword-only, no
        defaults) so they must be supplied from the YAML config; this
        prevents hidden defaults from drifting between train and eval.

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

        # Sampling kwargs are conditional on do_sample to avoid HF
        # warnings about "temperature/top_k/top_p set with do_sample=False".
        gen_kwargs = {
            "inputs_embeds": pack.packed_embeds,
            "attention_mask": pack.packed_mask,
            "max_new_tokens": max_new_tokens,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_k"] = top_k
            gen_kwargs["top_p"] = top_p
        generated_ids = self.reason_model.generate(**gen_kwargs)

        input_len = pack.packed_embeds.shape[1]
        if generated_ids.shape[1] > input_len:
            new_ids = generated_ids[:, input_len:]
        else:
            new_ids = generated_ids
        return self.tokenizer.batch_decode(new_ids, skip_special_tokens=True)
