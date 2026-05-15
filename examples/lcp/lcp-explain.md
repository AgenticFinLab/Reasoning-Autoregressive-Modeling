# Concept Pyramid Architecture: From CoT to Hierarchical Concepts

## 1. Notation and Concepts

### 1.1 Indexing Convention

We use a two-level subscript **C_{k,j}** to unambiguously distinguish inter-level from intra-level concepts:

| Symbol      | Meaning                                                       | Example                            |
|-------------|---------------------------------------------------------------|------------------------------------|
| **C_{k,j}** | The j-th concept at level k                                   | C_{5,17} = 18th concept at level 5 |
| **C_k**     | All concepts at level k: [C_{k,0}, C_{k,1}, ..., C_{k,L_k-1}] | C_5 has shape [B, 32, D]           |
| **j**       | Intra-level concept index within level k                      | j ∈ {0, 1, ..., L_k-1}             |
| **k**       | Level index (inter-level)                                     | k ∈ {0, 1, ..., K-1}               |
| **K**       | Total number of levels                                        | K=6 (levels 0 to 5)                |
| **L_k**     | Number of concepts at level k                                 | L_k = 2^k for k < K                |

Level configuration (K=6): L_0=1, L_1=2, L_2=4, L_3=8, L_4=16, L_5=32 (total: 63 concepts)

### 1.1.1 Notation Convention

Throughout this document, we use **our LCP notation** C_{k,j} consistently, even
when describing other methods. When referencing DLCM's single-level concepts
(written as c_k in the DLCM paper), we write them as C_{k,j} and add a note
explaining the mapping. This is because:

- DLCM's c_1, c_2, c_3, ... correspond to our C_{k,0}, C_{k,1}, C_{k,2}, ...
  at any given level k
- DLCM has no inter-level dimension — it only partitions the CoT at one
  granularity, so its concept index maps directly to our intra-level index j
- Our C_{k,j} **subsumes** DLCM's c_j by adding the level dimension k

### 1.2 Key Variables (following VAR.md Section 5.2.2)

| Variable    | VAR Image Domain                | Our Text Domain                                  | Physical Meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|-------------|---------------------------------|--------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **H_proj**  | z = Encoder(image)              | H_proj = Linear(Encoder(CoT))                    | CoT information to decompose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **H_rest**  | f_rest = "still needs encoding" | H_rest_k = H_proj - Σ_{i<k} R_i                  | Residual at level k                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| **H_hat**   | f_hat = "already encoded"       | H_hat_k = Σ_{i<k} R_i                            | Accumulated reconstruction                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **Q_{k,j}** | —                               | `concept_queries[k][j]` ∈ ℝ^D, learnable         | **Query expansion mechanism** (`concept_builder.py:450-468`). `concept_queries` is an `nn.ParameterList` with K entries of shape `[L_k, D]` where **L_k = 2^k** (1 → 2 → 4 → 8 → 16 → 32 for K=6). Each level k owns its own bank of L_k query vectors; the per-level growth in L_k is precisely what makes the concept count expand layer-by-layer and produces the pyramid's inter-level granularity. Q_{k,j} is the j-th query at level k and learns what structural segment to attend to. LCP has **no codebook / no VQ step** — concepts are produced by pure continuous soft attention over H_rest_k. |
| **τ**       | —                               | `temperature` ∈ ℝ, learnable scalar              | Attention sharpness in the softmax denominator (`√D · τ`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **A_{k,j}** | —                               | A_{k,j} = softmax(Q_{k,j} @ H_rest_k^T / (√D·τ)) | Attention weights for C_{k,j} (continuous, no discretization).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| **C_{k,j}** | —                               | C_{k,j} = level_proj(A_{k,j} @ H_rest_k)         | Concept (purely residual; no codebook lookup).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| **R_k**     | f_hat += h_k_up                 | R_k = A_k^T @ C_k                                | Reconstruction from level k                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |

`Q_{k,j}` and `τ` are the **only learnable parameters that drive the attention** — every other variable in this table is either a derived tensor (`A`, `C`, `R`, `H_*`) or a fixed input (`H_proj`).

### 1.3 Two Structural Dimensions

The concept pyramid has two orthogonal structural dimensions:

**Inter-level (coarse-to-fine granularity)** — all levels look at the SAME CoT, but at different resolutions:

```
CoT: "Let me solve this. First, 2+3=5. Then, 5×4=20. So the answer is 20."

Level 0 (1 concept):  [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■]
                       └─────────── entire CoT compressed to 1 concept ──────┘

Level 1 (2 concepts): [■■■■■■■■■■■■■■■■■■■■■|■■■■■■■■■■■■■■■■■■■■■■■■■■■■]
                       └─ first half ──┘└──── second half ──────────────┘

Level 2 (4 concepts): [■■■■■■■■■|■■■■■■■■■|■■■■■■■■■|■■■■■■■■■]
                       └ 1st qtr ┘└ 2nd qtr ┘└ 3rd qtr ┘└ 4th qtr ┘

... (each level divides the SAME CoT into finer segments)

Level 5 (32 concepts): [■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■]
                        └each tiny segment compressed to 1 concept┘
```

Key: Level 2 does NOT come "after" Level 1. Level 2 covers the SAME CoT,
just with finer segmentation. This is granularity, not sequential ordering.

**Intra-level (positional ordering within each level)** — within a single level,
concepts are ordered from early to late CoT positions:

```
Level 5 = [C_{5,0},  C_{5,1},  ...,  C_{5,31}]
            ↑         ↑               ↑
       earliest   middle          latest
       segment    segment         segment
```

**Inter-level** governs **what granularity** of information is captured.
**Intra-level** governs **which segment** of the CoT is captured at that granularity.

Both dimensions are mediated by the **learnable query bank** `concept_queries`: each `Q_{k,j}` is the parameter that *picks out* segment j at granularity k by competing for soft-attention mass over `H_rest_k`. Increasing k means more queries (`L_k = 2^k`) competing for the same residual, which is what produces finer segmentation. The query bank is therefore the trainable instrument that realises the two structural dimensions above; without it, residual decomposition would have no controllable read-out.

### 1.4 Overall Architecture: From CoT to Concept Pyramid to Solution

This section provides a high-level overview of how the hybrid design achieves the research goal: **compressing CoT into a hierarchical concept pyramid for efficient reasoning**.

#### 1.4.1 The Two-Stage Pipeline

LCP is organised as **two sequential training stages** and an
autoregressive inference path. The two stages share a common notion of a
"concept pyramid" C = [C_0, C_1, ..., C_{K-1}], but train disjoint modules
with disjoint objectives. Stage 2 is implemented by **one canonical
predictor** (`examples/lcp/concept_predictor.py`): a VAR-faithful
single-sequence architecture that constructs all approximation tokens
*before* the LLM, packs them into a single sequence with `Q` and `S`, and
trains both the concept pyramid and the reasoning NTP loss in **one**
packed forward.

**Bird's-eye view of the whole pipeline**

```
═══════════════════════════════════════════════════════════════════════════════
                    STAGE 1 — ConceptPyramidBuilder (TRAIN)
═══════════════════════════════════════════════════════════════════════════════

   Input: (Q, CoT, S)                                   [CoT is visible here]
                 │
                 ▼
    ┌────────────────────────────────────────────────────────────────────┐
    │ reason_model backbone(CoT)        → H_CoT  [B, L, D_e]  (FROZEN + opt LoRA)
    │ H_proj = input_proj_norm(input_proj(H_CoT))         [B, L, D]
    │            └─ input_proj         : Linear(D_e → D)            (TRAINABLE)
    │            └─ input_proj_norm    : LayerNorm(D)               (TRAINABLE)
    └────────────────────────────────────────────────────────────────────┘
                 │
                 ▼                       ┌────── residual decomposition ──────┐
    ┌────────────────────────────────────────────┐ │ H_hat_0  = 0                         │
    │ K = 6 levels, L_k = 2^k ∈ {1,2,4,8,16,32}   │ │ H_rest_0 = H_proj                    │
    │ TRAINABLE attention parameters:             │ │                                      │
    │   Q_k = concept_queries[k]   ∈ [L_k, D]     │ │ for each level k:                    │
    │   τ   = temperature          ∈ ℝ            │ │   H_hat_{k+1}  = H_hat_k  + R_k       │
    │   level_projs[k] : Linear(D → D)            │◄│   H_rest_{k+1} = H_rest_k − R_k       │
    │ for k in 0..K-1:                            │ │   R_k = A_kᵀ · C_k    (rank ≤ L_k)    │
    │   A_k = softmax(Q_k · H_restᵀ / (√D · τ))   │ └──────────────────────────────────────┘
    │   C_k = level_projs[k](A_k · H_rest)        │
    └────────────────────────────────────────────┘
                 │ produces pyramid: [C_0, C_1, ..., C_{K-1}]  ← groundtruth for Stage 2
                 │ also exports f_hat_per_level = [f_hat_0, ..., f_hat_{K-1}]   (canvas TF input)
                 │
     ┌───────────┴───────────┬────────────────────┬─────────────────────────┐
     ▼                       ▼                    ▼                         ▼
  L_recon               L_ordering            L_residual              L_reasoning
  ‖back_proj(H_hat_K)   exp_pos[C_{k,j}]      ‖H_rest_K‖₁         CE on S via
    − H_CoT‖²            < exp_pos[C_{k,j+1}]                     reason_model fed
  (back_proj : Linear(D → D_e), TRAINABLE)                        [Q; back_proj(C); S]

═══════════════════════════════════════════════════════════════════════════════
      STAGE 2 — ConceptPredictor (TRAIN) — single packed forward (canonical)
═══════════════════════════════════════════════════════════════════════════════

          (Builder frozen) ──► groundtruth pyramid C_gt = [C_0, ..., C_{K-1}]
                              + canvas f_hats   = [f_hat_0, ..., f_hat_{K-1}]
                                          │ detach()
          Input: (Q, f_hats, S)   [Q, f_hats, S all teacher-forced; no CoT used]

  ┌─────── Pre-LLM approx-token construction (per level, K times) ────────┐
  │   for k in 0..K-1:                                                    │
  │     context_k   = back_proj(f_hat_k)                  [B, L, D_enc]   │
  │     queries_k   = level_queries[k]                    [L_k, D_enc]    │
  │     attn_k, α_k = extract_attn(query_norm(queries_k),                 │
  │                                context_norm(context_k))               │
  │     approx_k    = post_norm(attn_k + queries_k) + lvl_embed[k]        │
  │   approx_tokens = cat(approx_0..approx_{K-1})    [B, total_C, D_enc]  │
  └───────────────────────────────────────────────────────────────────────┘

  ┌─────────────── Per-row packing (RoPE-safe, no inner padding) ─────────┐
  │  ┌──────────┬──────────────────────────┬──────────┬──────────┐        │
  │  │ real_Q_i │ approx_tokens (all K)    │ real_S_i │ tail_pad │        │
  │  │ q_len[i] │         total_C          │ s_len[i] │ padding  │        │
  │  └──────────┴──────────────────────────┴──────────┴──────────┘        │
  │  ◄──────────────────── T (packed length) ──────────────────►          │
  └───────────────────────────────────────────────────────────────────────┘

  ┌─────────────── Single LLM forward + scale-causal 4D mask ─────────────┐
  │  reason_model(inputs_embeds=packed, attention_mask=mask4d,            │
  │               output_hidden_states=True)                              │
  │                                                                       │
  │  Mask layout (per row):                                               │
  │    Q-region   : token-causal                                          │
  │    level-k    : bidirectional within, sees all levels j ≤ k           │
  │    S-region   : token-causal, sees Q + all approx tokens              │
  │    PAD keys   : masked everywhere (NaN-safe diagonal allowed)         │
  │                                                                       │
  │  Outputs: hidden [B, T, D_enc],  logits [B, T, V]                     │
  └───────────────────────────────────────────────────────────────────────┘

  ┌─────────────── Two readouts from the SAME hidden states ──────────────┐
  │  ① concept_head at approx-token positions  → [Ĉ_0, ..., Ĉ_{K-1}]      │
  │  ② lm_head      at solution-token positions → reasoning_logits         │
  └───────────────────────────────────────────────────────────────────────┘

  Losses (computed in losses.py from PredictorOutput):
          L_concept  = (1/K) Σ_k loss_fn(Ĉ_k, C_k.detach())
          L_reasoning = CE(reasoning_logits, reasoning_target_ids)

═══════════════════════════════════════════════════════════════════════════════
              INFERENCE — AR generation from Q only (K passes)
═══════════════════════════════════════════════════════════════════════════════

                          Input: Q      (no CoT, no S)

  ┌─────── Self-maintained f_hat canvas, K sequential packed passes ──────┐
  │                                                                       │
  │   f_hat ← zeros [B, L_canvas, D]                                      │
  │   for k in 0..K-1:                                                    │
  │     approx_k, α_k = _construct_approx_tokens(k, f_hat)                │
  │     pack ← [real_Q_i | approx_0..k | tail_pad]                        │
  │     hidden = reason_model(pack, scale_causal_mask).hidden_states[-1]  │
  │     Ĉ_k = concept_head(hidden at level-k approx positions)            │
  │     R_k = α_kᵀ @ Ĉ_k                                  [B, L_canvas, D]│
  │     f_hat = f_hat + R_k                                               │
  │                                                                       │
  │   Output: [Ĉ_0, Ĉ_1, ..., Ĉ_{K-1}]                                    │
  │   Solution: generate_solution(predicted_concepts, Q) — separate call  │
  └───────────────────────────────────────────────────────────────────────┘
```

The diagram above captures the three operating modes of the codebase:
`train_builder.py` (Stage 1), `train_predictor.py` (Stage 2 — one packed
forward per batch), and `predictor._forward_inference` (Inference — K
sequential passes with a self-maintained `f_hat` canvas). The arrows that
cross stage boundaries are the **only** places where gradients do NOT
flow: `C_gt` and `f_hats` are `detach()`ed and the Builder is frozen
during Stage 2.

**Stage 1 — ConceptPyramidBuilder** (`examples/lcp/concept_builder.py`)

Trainable parameters (everything else is frozen, except optional LoRA adapters on `reason_model`):

| Parameter         | Type / shape                              | Created by `__init__` line in `concept_builder.py` |
|-------------------|-------------------------------------------|----------------------------------------------------|
| `input_proj`      | `nn.Linear(D_encoder, D)` (with bias)     | `self.input_proj = nn.Linear(...)`                 |
| `input_proj_norm` | `nn.LayerNorm(D)`                         | `self.input_proj_norm = nn.LayerNorm(...)`         |
| `concept_queries` | `nn.ParameterList` of K, each `[L_k, D]`  | `self.concept_queries = nn.ParameterList([...])`   |
| `temperature` (τ) | `nn.Parameter(torch.ones(1))`             | `self.temperature = nn.Parameter(torch.ones(1))`   |
| `level_projs`     | `nn.ModuleList` of K, each `Linear(D, D)` | `self.level_projs = nn.ModuleList([...])`          |
| `back_proj`       | `nn.Linear(D, D_encoder, bias=False)`     | `self.back_proj = nn.Linear(...)`                  |

```
Input : (Q, CoT, S)                         # Q = question, S = solution
Forward :
    H_CoT   = reason_model.embed(CoT)       # frozen embedding lookup
    H_proj  = LayerNorm(Linear(H_CoT))      # encode CoT into concept space
    for k in 0..K-1:                        # K = num_levels (6 for GSM8K)
        A_k = softmax(Q_k @ H_rest_k / √D / τ)
        C_k = level_proj_k(A_k @ H_rest_k)
        R_k = A_kᵀ @ C_k
        H_hat_{k+1} = H_hat_k + R_k
        H_rest_{k+1} = H_rest_k - R_k
    H_recon = back_proj(H_hat_K)            # map back to encoder space
    # Reasoning probe: run frozen reason_model on [Q; concepts; S]
Outputs : PyramidOutput {
    concepts = [C_0, ..., C_{K-1}],         # the groundtruth pyramid
    H_recon,                                # for reconstruction loss
    final_residual = H_rest_K,              # for residual loss
    exp_positions,                          # for ordering loss
    reasoning_logits / reasoning_target_ids # for NTP reasoning loss
}
Loss    : L_builder = w_recon·L_recon + w_order·L_order
                    + w_residual·L_residual + w_reasoning·L_reasoning
```

Only the encode/attend/residual modules and `back_proj` are trainable. The
backbone `reason_model` is **frozen** throughout Stage 1 (optionally LoRA-adapted).
Concretely, the trainable Stage-1 parameters are: `input_proj`, `input_proj_norm`, **`concept_queries` (the K-entry `nn.ParameterList`, `[L_k, D]` per level)**, **`temperature` (τ)**, `level_projs`, `back_proj` — plus optional LoRA adapters on `reason_model`.

**Stage 2 — ConceptPredictor (canonical, single-sequence)**

The predictor (`examples/lcp/concept_predictor.py`) implements one
VAR-faithful architecture: it constructs all approximation tokens
*before* the LLM via cross-attention from `level_queries[k]` over
`back_proj(f_hat_k)`, packs them into a single sequence with `Q` and `S`,
and reads BOTH the concept pyramid and the reasoning logits from a
single scale-causal forward pass.

```
Input : (Q, gt_f_hats = [f_hat_0, ..., f_hat_{K-1}], S)   # f_hats from frozen Builder
Forward (teacher-forced, ONE backbone pass):
    # 1. Pre-LLM approx-token construction (per level)
    for k in 0..K-1:
        context_k     = back_proj(f_hat_k)                       # (B, L, D_enc)
        approx_k, α_k = extract_attn(query_norm(level_queries[k]),
                                     context_norm(context_k))    # cross-attn
        approx_k      = post_norm(approx_k + level_queries[k]) + lvl_embed[k]
    approx_tokens = cat(approx_0..approx_{K-1})                  # (B, total_C, D_enc)
    # 2. Per-row pack [real_Q | approx_tokens | real_S | tail_pad]
    pack = pack_qcs_sequences(Q_embeds, q_mask, approx_tokens, S_embeds, s_mask)
    # 3. Build per-row scale-causal 4D mask  [B, 1, T, T]
    mask4d = _build_scale_causal_mask_packed(pack.q_len, pack.s_len, level_lengths, T)
    # 4. Single LLM forward (full reason_model: hidden + logits)
    out = reason_model(inputs_embeds=pack.packed_embeds, attention_mask=mask4d,
                       output_hidden_states=True)
    hidden = out.hidden_states[-1]                               # (B, T, D_enc)
    logits = out.logits                                          # (B, T, V)
    # 5. Two readouts from the SAME hidden states
    Ĉ_k             = concept_head(hidden[approx-token positions of level k])
    reasoning_logits = gather_solution_logits(logits, pack)
Outputs : PredictorOutput { predicted_concepts, gt_concepts,
                             reasoning_logits, reasoning_target_ids }
Inference: K sequential packed passes, self-maintained f_hat canvas.
           Each pass: build approx_0..k from current f_hat → LLM forward →
                      Ĉ_k = concept_head(hidden_k); R_k = α_kᵀ @ Ĉ_k; f_hat += R_k.
```

**Trainable components**: `reason_model` (own copy + optional LoRA),
`back_proj`, `level_queries` (K-entry `nn.ParameterList`, `[L_k, D_enc]`
per level), `query_norm` / `context_norm` / `post_norm`, `extract_attn`
(MultiheadAttention), `lvl_embed` (`Embedding(K, D_enc)`), `concept_head`
(`Linear(D_enc, D_enc) → GELU → Linear(D_enc, D)`). The Builder is held
fully frozen.

End-to-end flow (what actually changes between stages):

| Stage     | Trainable params                                                                                                            | Backbone         | Sees CoT?   | Loss                                         |
|-----------|-----------------------------------------------------------------------------------------------------------------------------|------------------|-------------|----------------------------------------------|
| Builder   | input_proj, input_proj_norm, **concept_queries**, **temperature**, level_projs, back_proj (+ LoRA)                          | frozen (+LoRA)   | Yes         | L_recon + L_order + L_residual + L_reasoning |
| Predictor | reason_model (+ LoRA), back_proj, level_queries, query_norm, context_norm, extract_attn, post_norm, lvl_embed, concept_head | own copy (+LoRA) | No (Q only) | L_concept + L_reasoning                      |

#### 1.4.2 Key Design Principles

**1. Builder-Predictor Separation**
- **Builder**: Uses soft attention + residual flow to extract groundtruth from CoT.
- **Predictor**: Generates the same pyramid from `Q` alone via the
  canonical single-sequence architecture (pre-LLM cross-attention over
  the cumulative `f_hat` canvas, then ONE packed LLM forward with a
  scale-causal 4D mask).
- **Rationale**: Builder defines "what is a good pyramid"; Predictor
  learns "how to generate it".

**2. Preserved Core Mechanisms**
All mechanisms from Section 1.3 are retained:
- **Query expansion**: 1→2→4→8→16→32 learnable queries per level (now
  `level_queries`, used as cross-attention queries over `f_hat`).
- **Soft attention (soft boundaries)**: Competition-based segment-concept correspondence.
- **Residual reconstruction**: Coarse-to-fine information decomposition;
  level k conditions only on `f_hat_k = Σ_{j<k} R_j` (cumulative canvas),
  not on raw concept stacks.
- **Intra-level ordering**: Concepts ordered by CoT position (Builder loss).
- **Purely residual**: No cross-scale conditioning in the builder (VAR.md principle).

**3. Training-Inference Alignment**
- Training: Predictor sees the Builder's `gt_f_hats` (teacher forcing on the canvas).
- Inference: Predictor self-maintains `f_hat` via
  `R_k = α_kᵀ @ Ĉ_k` after each level.
- Both use the same scale-causal structure: level k depends only on levels j ≤ k.

#### 1.4.3 Why This Design Works

**Efficiency**: At inference, we bypass CoT generation:
```
Traditional: Q → [long CoT text] → Solution  (slow, many tokens)
Ours:        Q → [Σ_{k=0}^{K-1} L_k concepts] → Solution   (fast, hierarchical)
```

**Effectiveness**: The concept pyramid preserves CoT's reasoning structure:
- Level 0 (1 concept): Global reasoning strategy
- Level 3 (8 concepts): Key reasoning steps
- Level 5 (32 concepts): Fine-grained details

**Learnability**: Two-phase design provides clear training signals:
- Builder ensures good pyramid structure exists
- Predictor learns to generate this structure from Q alone

---

## 2. Inter-Level Analysis: Coarse-to-Fine Hierarchy

### 2.1 The Rank Bottleneck Guarantee

At each level k, the reconstruction R_k = A_k^T @ C_k has rank at most L_k:

```
R_k = A_k^T @ C_k
    = [B, L, L_k] @ [B, L_k, D]
```

This means:
- **Level 0** (L_0=1): R_0 has rank 1 → can only capture **one global direction** of H_proj
- **Level 1** (L_1=2): R_1 has rank 2 → can capture **two independent directions** of H_rest_1
- **Level 5** (L_5=32): R_5 has rank 32 → can capture **32 independent directions**

This rank bottleneck is the mathematical guarantee of coarse-to-fine behavior. Regardless of how expressive `level_proj` is, the reconstruction R_k cannot exceed rank L_k. Level 0 is physically incapable of capturing fine details — it must focus on the dominant global pattern.

**Intuitive example**: Think of drawing a portrait:
```
Level 0 (1 concept):  One broad stroke — just the overall face shape and skin tone
Level 1 (2 concepts): Two strokes — left side vs right side of the face
Level 2 (4 concepts): Four strokes — forehead, eyes, nose, mouth regions
...
Level 5 (32 concepts): 32 fine strokes — individual eyelashes, pores, wrinkles
```
Each level CAN ONLY ADD at most L_k independent details. You can't paint
eyelashes with a single broad stroke (rank 1). The rank bottleneck is the
mathematical reason why coarse levels capture coarse structure.

### 2.2 Analogy with VAR Scale Bottleneck

| VAR Scale | Tokens | Information Capacity  | LCP Level | Concepts | Information Capacity  |
|-----------|--------|-----------------------|-----------|----------|-----------------------|
| 1×1       | 1      | Global color/tone     | Level 0   | 1        | Global CoT structure  |
| 2×2       | 4      | Coarse spatial layout | Level 1   | 2        | Two major segments    |
| 4×4       | 16     | Medium structure      | Level 2   | 4        | Four sub-segments     |
| ...       | ...    | ...                   | ...       | ...      | ...                   |
| 32×32     | 1024   | Fine details          | Level 5   | 32       | Fine-grained segments |

In VAR, each scale is independently quantized (VQ lookup), which naturally partitions information across scales. In our design, the residual flow serves the same purpose: H_rest_{k+1} = H_rest_k - R_k ensures that information captured at level k is no longer available at level k+1.

**Same image, different resolutions** (VAR):
```
An image of a cat:

1×1:  [██]             — just "orange blob" (1 token)
2×2:  [██|██]           — "orange blob, left/right half differ" (4 tokens)
4×4:  [████|████]       — "ears on top, face in middle" (16 tokens)
32×32: [detailed cat]   — whiskers, eyes, fur texture (1024 tokens)

All scales describe THE SAME cat, just at different pixel resolutions.
```

**Same CoT, different segmentations** (LCP):
```
CoT: "Let me solve this. First, 2+3=5. Then, 5×4=20. So the answer is 20."

Level 0: [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■] — "solving a math problem"
Level 1: [■■■■■■■■■■■■■■■■■■■■■|■■■■■■■■■■■■■■■■■■■■■■■■■■■■] — "setup | computation"
Level 2: [■■■■■■■■■|■■■■■■■■■|■■■■■■■■■|■■■■■■■■■] — "intro|step1|step2|answer"
Level 5: [■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■|■]

All levels describe THE SAME CoT, just at different segment granularities.
```

### 2.3 Purely Residual Decomposition (No Cross-Scale Conditioning)

**Design principle (VAR.md)**: The builder must be purely residual — each level only sees the current residual `H_rest_k`, with NO conditioning on previous levels' concepts. Cross-scale conditioning (e.g., cross-attention refinement using `[C_0, ..., C_{k-1}]`) belongs to Stage 2 (the Predictor), not Stage 1 (the Builder).

**Why no cross-attention in the builder?**
1. **VAR alignment**: VAR's VQ-VAE Stage 1 uses purely residual decomposition — each scale only encodes `f_rest`, with no knowledge of previous scales' codebook entries. Cross-scale conditioning only appears in Stage 2 (the Transformer).
2. **Clean gradient flow**: Every parameter in the builder is trained by `recon_loss` + `reasoning_loss`. Cross-attention on previous concepts would create parameters that only the predictor's loss could train — dead weights in Stage 1.
3. **Separation of concerns**: The builder extracts ground truth concepts from CoT. The predictor learns cross-level dependencies from Q alone. Mixing these concerns in the builder violates the two-stage design.

```
Level k processing (purely residual):

  ┌─────────────────────────────────────────────────┐
  │   C_{k,j} = level_proj(A_{k,j} @ H_rest_k)      │
  │   R_k = A_k^T @ C_k                             │
  │   H_hat += R_k        ← "what has been encoded"   │
  │   H_rest -= R_k       ← "what still needs encoding"│
  │                                                    │
  │   C_k is the FINAL concept — no refinement step.  │
  │   Cross-level dependencies are learned by the      │
  │   Predictor (Stage 2), not the Builder.            │
  └─────────────────────────────────────────────────┘
```

### 2.4 Potential Issue: Greedy Early Levels

**Concern**: Level 0 (1 concept) might extract too much information, leaving H_rest_1 nearly empty for levels 1 to K-1.

**Analysis**: This is constrained by the rank bottleneck. R_0 has rank 1 — even if C_{0,0}_base contains a lot of "energy", the reconstruction A_0^T @ C_{0,0}_base is still rank 1. It can only capture one linear direction of H_proj. The remaining directions are preserved in H_rest_1.

However, `level_proj` is a linear layer that can amplify the magnitude of C_{0,0}_base. If C_{0,0}_base has very large norm, then R_0 = A_0^T @ C_{0,0}_base can "absorb" a disproportionate share of H_proj's magnitude, leaving H_rest_1 with small but informationally rich residuals.

**Is this actually a problem?** The reconstruction loss provides a corrective signal: if levels 1 to K-1 cannot reconstruct H_rest (because it's nearly zero), the total loss increases. The model is incentivized to balance extraction across levels. But the incentive is indirect — the loss only measures total coverage, not per-level balance.

**Mitigation strategies** (for future consideration, not current implementation):
1. Per-level reconstruction loss: L_balanced = Σ_k ||R_k||² / ||H_proj||² (encourage each level to contribute)
2. Information-proportional initialization (already available via `use_positional_query_init`)
3. End-to-end NTP loss from the full LCP pipeline (strongest signal)

**Current assessment**: The greedy extraction concern is theoretically valid but likely manageable in practice. The rank bottleneck provides a hard constraint, and the full LCP training pipeline with NTP loss will provide the strongest corrective signal.

---

## 2.5 Deep Dive: The Rank-Constrained Residual Decomposition Principle

This section synthesizes §2.1–§2.4 and the VAR comparison of §7 into a single, mechanistic statement of what the Builder actually does. It is the most important section of this document — every downstream design choice (Predictor teacher forcing, loss weights, level schedule) flows from here. It is the lcp counterpart of `docs/VAR.md §5.3.2.1` (which established the dual fact for VAR: *codebook entries are residuals*).

### 2.5.0 Relationship to VAR.md §6 — No Contradiction, Two Layers of Description

Readers coming from [docs/VAR.md](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/docs/VAR.md) §6 — which declared that lcp's Builder "follows VAR's residual philosophy" and that `C_k` "expresses the semantic remainder scales 0..k-1 cannot cover" — may wonder whether §2.5's emphasis on a *rank-bounded softmax bottleneck* (contrasted with VAR's *discrete codebook bottleneck*) contradicts that claim, **or** whether §2.5's phrase "`C_k` is the best low-rank expression of the residual" is a third, different statement. **Neither is a contradiction.** The three statements operate at three different layers of abstraction and are mutually consistent. This subsection makes the layering explicit.

#### Two layers of architectural description

| Layer           | What it describes                                            | Same in VAR and lcp?  | Discussed in                                 |
|-----------------|--------------------------------------------------------------|-----------------------|----------------------------------------------|
| **Outer loop**  | The `H_rest / H_hat` residual-accumulation skeleton          | ✅ **YES — identical** | VAR.md §6; lcp-explain.md §2.5.5             |
| **Inner joint** | How each level produces its per-level output from `H_rest_k` | ❌ **NO — different**  | VAR.md §5.3.2.1; lcp-explain.md §2.5.2–2.5.6 |

```
┌─── OUTER LOOP (shared by VAR and lcp) ─────────────────────────────┐
│  for k in 0..K-1:                                                      │
│      level-k output  ←──── [INNER JOINT: differs] ────  H_rest_k       │
│      R_k             ←  smear level-k output to sequence length        │
│      H_hat_{k+1}     =  H_hat_k  + R_k      (canvas grows)             │
│      H_rest_{k+1}    =  H_rest_k - R_k      (residual shrinks)         │
│                                                                         │
│    ┌── INNER JOINT (differs) ────────────────────────────────┐         │
│    │  VAR:     level-k output  =  embedding(argmin_V ‖·‖)     │         │
│    │           (discrete codebook lookup, V hard options)     │         │
│    │  lcp:  level-k output  =  level_proj(A_k @ H_rest_k)  │         │
│    │           (rank-L_k soft summary, softmax weights)       │         │
│    └───────────────────────────────────────────────────────────┘        │
└────────────────────────────────────────────────────────────────────────┘
```

**VAR.md §6** is a statement about the **outer loop** — it's why the Predictor must replay the cumulative canvas `H_hat_k` (identical requirement in both systems).  
**lcp-explain.md §2.5** is a zoom-in on the **inner joint** — it explains that we swap discrete-argmin for rank-bounded-softmax while leaving the outer loop untouched.

#### Reconciling "residual in nature" vs "best low-rank summary"

These two phrasings describe the **same mathematical object** (`C_k`) from two different vocabularies:

| Phrasing (source)                                                            | Vocabulary       | What exactly it claims                                                           |
|------------------------------------------------------------------------------|------------------|----------------------------------------------------------------------------------|
| "`C_k` is residual in nature / expresses what prior can't cover" (VAR.md §6) | **Semantic**     | `C_k`'s information source is `H_rest_k`, not raw `H_proj`                       |
| "`C_k` is the best rank-`L_k` low-rank summary of `H_rest_k`" (§2.5.3)       | **Mathematical** | `C_k` approximates `H_rest_k` at rank ≤ `L_k`, optimally under training pressure |

The equivalence chain:

```
  H_rest_k  =  H_proj - Σ_{j<k} R_j    ← by construction
            =  "what scales 0..k-1 have not yet covered"

  C_k       =  level_proj( A_k @ H_rest_k )
            =  best rank-L_k summary of H_rest_k       (§2.5.3)
            =  best rank-L_k summary of what scales 0..k-1 have not yet covered
            =  "residual in nature"                     (VAR.md §6)
```

The VAR.md phrasing is the semantic-level consequence of the §2.5 mathematical-level statement. They are the same claim at two zoom levels.

#### Critical subtlety: `C_k ≠ H_rest_k`

It is tempting (and a common source of confusion) to read "`C_k` is residual in nature" as "`C_k` equals the residual tensor." **This is wrong.** `C_k` is a *rank-`L_k` lossy compression* of `H_rest_k`, not `H_rest_k` itself:

```
 Shape of H_rest_k :  [B, L,   D]     ← uncompressed residual (L positions)
 Shape of C_k      :  [B, L_k, D]     ← rank-L_k compressed summary (L_k ≪ L)
 Shape of R_k      :  [B, L,   D]     ← smeared-back rank-L_k reconstruction

 Relation:
   C_k  =  level_proj(A_k @ H_rest_k)     # compress: L → L_k
   R_k  =  A_k^T @ C_k                     # smear:    L_k → L
   H_rest_{k+1}  =  H_rest_k  −  R_k       # subtract R_k (NOT C_k) from residual
   H_hat_{k+1}   =  H_hat_k   +  R_k       # add       R_k (NOT C_k) to canvas
```

So three distinct tensors are about the residual, each playing a different role:

| Tensor     | Shape         | Role                                                        | Synonyms in literature                           |
|------------|---------------|-------------------------------------------------------------|--------------------------------------------------|
| `H_rest_k` | `[B, L, D]`   | The residual itself — what remains uncovered                | "uncovered information," "current state"         |
| `C_k`      | `[B, L_k, D]` | Rank-`L_k` **compressed summary** of the residual           | "concepts," "level-k latents," "codes"           |
| `R_k`      | `[B, L, D]`   | Smeared-back, rank-`L_k` **reconstruction** of the residual | "level-k reconstruction," "h_k" in VAR, "stroke" |

- `C_k` is what the **Predictor** predicts (and what `reason_model` sees after `back_proj`).
- `R_k` is what the **outer loop** debits from `H_rest` and adds to `H_hat`.
- `H_rest_k` is what the **inner joint at level k** reads as input.

"`C_k` is residual in nature" means: **`C_k`'s informational content comes from `H_rest_k`**, hence it inherits the property of being "what prior levels couldn't cover." It does **not** mean `C_k = H_rest_k` literally.

#### Summary table — which statement lives at which layer

| Claim                                                           | Layer        | Tensor level | Relationship to other claims           |
|-----------------------------------------------------------------|--------------|--------------|----------------------------------------|
| "Predictor must replay cumulative `H_hat_k`" (VAR.md §6)        | Outer loop   | `H_hat`      | Shared by VAR and lcp                  |
| "VAR uses discrete codebook, lcp uses rank bottleneck" (§2.5.6) | Inner joint  | per-level op | The only structural difference         |
| "`C_k` is residual in nature" (VAR.md §6)                       | Semantic     | `C_k`        | Equivalent to §2.5.3 at semantic zoom  |
| "`C_k` is best rank-`L_k` summary of `H_rest_k`" (§2.5.3)       | Mathematical | `C_k`        | The precise form of the semantic claim |
| "`R_k` is subtracted from `H_rest_k`" (both docs)               | Operational  | `R_k`        | The canvas-debit step; shared in both  |

All five statements are simultaneously true. They describe different faces of the same architecture.

---

### 2.5.1 The Core Sentence (核心一句话)

> **At each level, the Builder takes the current residual `H_rest_k`, uses `L_k` learnable queries to construct a rank-`L_k`-bounded best low-rank summary `C_k`, smears it back to sequence length as `R_k`, adds `R_k` onto the canvas `H_hat` and subtracts it from the residual, then hands whatever remains to the next level whose `2×`-wider query bank fishes again.**
>
> **我们每一层都基于当前残差 `H_rest_k`，用 `L_k` 条可学习查询构造一个秩受 `L_k` 约束的最佳低秩摘要 `C_k`，然后把它 smear 回序列长度得到 `R_k`，加入画布、从残差里扣掉，留下的信息交给下一层用 2 倍宽的查询再捞一次。**

Every clause in this sentence corresponds to an architectural commitment that can be read directly off the code in [concept_builder.py](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/examples/lcp/concept_builder.py). The rest of §2.5 unpacks it.

### 2.5.2 The Rank Inequality as lcp's "Invisible Codebook"

VAR bottlenecks information flow with a **discrete codebook** (hard argmin lookup against V learned centroids). lcp has no codebook — so what prevents the model from cheating and dumping all information into a single level? Answer: **a linear-algebraic rank constraint** just as unforgiving as a codebook, only expressed in the language of matrix factorization rather than nearest-neighbor search.

Formal statement. At level `k`, the reconstruction is assembled by matmul:

```
R_k  =   A_k^T   @   C_k
         [L,L_k]     [L_k,D]
         ────────    ────────
         smear       summary
```

and the summary itself is built from the attention-weighted residual:

```
C_k  =  level_proj(  A_k   @   H_rest_k  )
                     [L_k,L]    [L,D]
```

Hence `R_k` factors through `R^{L_k × D}`. Therefore:

```
rank(R_k)  ≤  L_k     (since L_k ≪ L and L_k ≪ D by construction)
```

This inequality is **strict and mechanical** — no clever initialization or loss can raise it. It is enforced at graph-construction time by setting `num_queries = L_k`. The rank upper bound **is** the bottleneck.

**Why this equals "a codebook of invisible size"**: VAR's codebook has `V` entries of dimension `Cvae`; `embedding(idx_k)` at each spatial position is one of at most `V` possible vectors. lcp's level-k output lives in a continuous rank-`L_k` subspace of `R^{L×D}`; `R_k` is one of infinitely many tensors in this subspace. Both are information-capacity ceilings, merely expressed in different bases:

| Bottleneck shape | VAR                                 | lcp                               |
|------------------|-------------------------------------|-----------------------------------|
| Capacity unit    | Discrete codebook entry (V options) | Continuous rank-1 direction       |
| Budget per level | `L_k^2` patches × V choices each    | `L_k` ranks, continuous           |
| Nature           | **Hard discrete** (argmin)          | **Hard on rank, soft on weights** |
| Differentiable?  | No (STE workaround)                 | Yes (softmax is smooth)           |

### 2.5.3 "Best Low-Rank Summary" — Why `C_k` is Optimal

The softmax weights `A_k = softmax(Q_k H_rest_k^T / (√D · τ))` are not arbitrary — they are the **gradient-descent optimum** of a scalar objective balancing two pressures:

1. **Coverage pressure**: `L_recon = ‖back_proj(Σ_j R_j) − H_CoT‖²` penalizes any residual that never gets captured.
2. **Rank pressure**: `R_k` is forced to rank ≤ `L_k`, so `C_k` cannot be all of `H_rest_k` — it must be a **lossy compression** that prioritizes the dominant directions of the residual.

Under these two pressures, training drives `A_k^T @ A_k @ H_rest_k` toward a rank-`L_k` approximation of the residual that preserves the most reconstructable energy. This is the learnable, non-linear, position-aware analog of the **Eckart–Young theorem**: the best rank-`L_k` approximation of a matrix is its top-`L_k` SVD reconstruction. Softmax attention is a cousin of SVD (with the budget constraint `Σ_j A_{k,j}(t) = 1` replacing orthonormality), and `level_proj` adds a learned feature transform on top.

Therefore the phrase "best low-rank summary" in §2.5.1 is not rhetoric — it is a statement about the loss landscape's optimum.

### 2.5.4 "Smear" — `R_k = A_k^T @ C_k` as a Rank-Bounded Broadcast

Multiplying `A_k^T ∈ R^{L×L_k}` by `C_k ∈ R^{L_k × D}` produces `R_k ∈ R^{L×D}`:

- Each of `L` sequence positions receives a convex-like combination of the `L_k` concepts, weighted by how much that position attended to each concept.
- If position `t` was claimed primarily by `C_{k,j}`, then `R_k[t] ≈ C_{k,j}`.
- If position `t` is on the boundary between two concepts, `R_k[t]` is a soft blend.

The composition `A_k^T @ A_k ∈ R^{L×L}` is a **rank-`L_k` soft-clustering smoother**: it replaces each position's feature with a soft-cluster-mean of its neighbors. Analogous operations across fields:

| Field       | Compression step       | Smear-back step              | Rank bound             |
|-------------|------------------------|------------------------------|------------------------|
| PCA         | project to top-k axes  | reconstruct via `V_k V_k^T`  | rank ≤ k               |
| K-means     | assign to centroid     | broadcast centroid to points | rank ≤ K               |
| lcp Builder | `A_k @ H_rest_k`       | `A_k^T @ C_k`                | rank ≤ L_k             |
| VAR VQ-VAE  | `argmin` over codebook | `embedding(idx_k)` + upscale | ≤ V discrete centroids |

### 2.5.5 "Paint on the Canvas, Subtract from the Residual" — The Two Ledgers

The Builder maintains two tensors that serve as accounting ledgers:

```
H_hat_k   = Σ_{j<k} R_j          — "what has already been painted onto the canvas"
H_rest_k  = H_proj - H_hat_k     — "what is still left to paint"
```

**Invariant**: at every level, `H_hat_k + H_rest_k = H_proj`. Both live in `R^{L×D}`.

After level k executes:

```
H_hat_{k+1}  = H_hat_k  + R_k     # add rank-L_k stroke to canvas
H_rest_{k+1} = H_rest_k - R_k     # debit the residual
```

Crucially, `H_rest_{k+1}` is **exactly the part of `H_proj` not spanned (in the rank-reduction sense) by everything captured so far**. When level `k+1` attends against `H_rest_{k+1}`, the directions it can discover are precisely those orthogonal (in the residual sense) to `R_0, ..., R_k`. **The residual itself performs the non-overlap enforcement that VAR achieves via codebook separation** — the mechanism is different (subtraction vs. discrete partition), but the net effect is equivalent: no level can redundantly re-capture information already booked by a coarser level.

Flow diagram of the ledger dynamics (K=6 levels, `L_k = 2^k`):

```
               level 0         level 1        level 2         ...    level 5
               (L_0=1)         (L_1=2)        (L_2=4)                (L_5=32)

H_proj ─► H_rest_0 ─► H_rest_1 ─► H_rest_2 ─► H_rest_3 ─► H_rest_4 ─► H_rest_5
              │           │            │                                 │
          Q_0/A_0/C_0  Q_1/A_1/C_1  Q_2/A_2/C_2                     Q_5/A_5/C_5
              │           │            │                                 │
              R_0         R_1          R_2                               R_5
              │           │            │                                 │
              ▼           ▼            ▼                                 ▼
H_hat: 0 ──► H_hat_1 ──► H_hat_2 ──► H_hat_3 ──► ... ──► H_hat_6 ≈ H_proj

rank(R_k):        1    ≤   2       ≤   4      ≤   8    ≤  16   ≤  32
cum. rank(H_hat): 1    ≤   3       ≤   7      ≤  15    ≤  31   ≤  63

(Σ L_k = 2^K - 1 = 63 concepts total, matching min(L, D) for typical L=128, D=64.)
```

### 2.5.6 Side-by-Side with VAR's Hard Codebook Bottleneck

The statement "VAR has a codebook, we don't" is true but misses the structural parallel. Here is the precise correspondence:

| Aspect                    | VAR Stage-1 (VQ-VAE)                              | lcp Builder                                      |
|---------------------------|---------------------------------------------------|--------------------------------------------------|
| Residual tensor           | `f_rest`, shape `[B, Cvae, H, W]`                 | `H_rest`, shape `[B, L, D]`                      |
| Canvas tensor             | `f_hat`                                           | `H_hat`                                          |
| Per-level atomic output   | `embedding(idx_k)` — codebook lookup (residual!)  | `C_k` — attention summary of residual            |
| Bottleneck mechanism      | Discrete lookup in V-entry codebook               | Rank-`L_k` matrix factorization                  |
| Bottleneck strength       | **Hard discrete** (argmin)                        | **Hard rank** (matmul-imposed)                   |
| Coefficients nature       | Binary indicator (one-hot codebook index)         | Continuous softmax weights                       |
| Capacity at level k       | `V^{L_k^2}` discrete patches (enormous but fixed) | Continuous rank-`L_k` subspace of `R^{L×D}`      |
| Reconstruction operator   | `φ_k(upsample(embedding(idx_k)))`                 | `A_k^T @ C_k`                                    |
| Non-overlap mechanism     | Each scale quantizes its own residual             | Each level subtracts its own `R_k` from residual |
| Coarse-to-fine guarantee  | Small spatial patch count at coarse scales        | Small `L_k` at coarse levels                     |
| Differentiability         | **Non-diff** (argmin); needs STE                  | **Fully differentiable** (softmax all the way)   |
| Training loss shape       | CE over indices + VQ + reconstruction             | MSE/NTP + ordering + residual + reasoning        |
| Failure mode              | Codebook collapse (few entries used)              | Attention collapse (queries attend uniformly)    |
| Zero residual achievable? | In practice yes (codebook spans the space)        | Yes iff `Σ L_k ≥ min(L, D)`                      |

**Key insight**: VAR's and lcp's bottlenecks are **duals of each other in information-capacity space** — different shapes of the same constraint. VAR trades differentiability for a crisp discrete vocabulary; lcp trades the discrete vocabulary for end-to-end differentiable training. Neither is strictly more powerful; they are two fixed points on a bottleneck-shape axis:

```
         hard discrete            soft continuous
         ┌────────────┐          ┌────────────────┐
         │ VAR VQ-VAE │ ◄──────► │ lcp Builder │
         │  codebook  │          │ rank-bounded   │
         │  (V entries│          │ attention      │
         │   per pos) │          │ (L_k ranks)    │
         └────────────┘          └────────────────┘
               │                         │
               │                         │
         non-differentiable          fully differentiable
         sparse codes                dense low-rank codes
         CE loss over indices        MSE/NTP over vectors
```

### 2.5.7 Numerical Walk-Through

Take `L_0,…,L_5 = 1, 2, 4, 8, 16, 32`, sequence length `L = 128`, concept dim `D = 64`, batch `B = 1`:

```
Level 0 (L_0=1):
  H_rest_0 : [1, 128, 64]                  # full CoT information
  Q_0      : [1, 64]                        # 1 learnable query
  A_0      : [1, 1, 128]                    # softmax over 128 positions
  C_0      : [1, 1, 64]                     # 1 concept, rank-1 summary
  R_0      : A_0^T @ C_0 : [1, 128, 64]     # rank(R_0) ≤ 1
  → all 128 positions share one globally-dominant direction

Level 1 (L_1=2):
  H_rest_1 = H_rest_0 − R_0 : [1, 128, 64]  # rank-1 direction removed
  Q_1      : [2, 64]                        # 2 independent queries
  A_1      : [1, 2, 128]                    # softmax forces queries to partition
  C_1      : [1, 2, 64]                     # rank ≤ 2 summary
  R_1      : [1, 128, 64], rank ≤ 2
  → positions split into ≈2 clusters by dominant residual direction

Level 5 (L_5=32):
  H_rest_5 : [1, 128, 64]                   # 1+2+4+8+16 = 31 ranks already removed
  Q_5      : [32, 64]
  A_5      : [1, 32, 128]
  C_5      : [1, 32, 64], rank ≤ 32
  R_5      : [1, 128, 64], rank ≤ 32
  → fine-grained detail captured in remaining 33 ranks of residual space

Cumulative rank at the end:
  Σ L_k = 1 + 2 + 4 + 8 + 16 + 32 = 63 ≈ min(L, D) = 64
```

**Observation**: `Σ L_k = 2^K − 1` is intentionally sized to match `min(L, D)`. More ranks would be redundant; fewer would leave information uncaptured. The doubling schedule `L_k = 2^k` is not arbitrary — it is the **geometric partitioning of the rank budget** that, combined with residual subtraction, gives the sharpest coarse-to-fine spectral staircase.

### 2.5.8 Why Doubling `L_k`? — The Exponential Rank Schedule

The clause "hand to the next level whose `2×`-wider query bank fishes again" encodes the doubling `L_k = 2 L_{k-1}`. Three independent alignments justify it:

1. **Dyadic segmentation**: each level halves the segment width, doubling the segment count. This matches the DLCM intra-level correspondence (§3).
2. **Geometric residual decay**: after a rank-`L_k` pursuit, the residual's L2 energy decays geometrically. The next level needs proportionally more ranks to keep up with the thinner residual.
3. **VAR alignment**: VAR's token counts per scale `{1, 4, 16, 64, 256, 1024}` grow by `4×` (which is `2×` along each spatial axis). Our `L_k = 2^k` is the 1-D analog.

The nonlinear contraction is:

```
H_rest_{k+1}  =  H_rest_k  −  A_k^T @ level_proj(A_k @ H_rest_k)
```

Iterating it K times with doubling `L_k` removes cumulatively rank `Σ L_k = 2^K − 1` — an exponential rank coverage per level, versus a linear coverage `K` that uniform rank-1 pursuit would give. The doubling schedule is an order-of-magnitude faster coverage than uniform matching pursuit.

### 2.5.9 Implications for the Predictor (Stage 2)

The §2.5 principle has a direct, non-negotiable consequence for ConceptPredictor teacher-forcing. This is the lcp analog of the warning in [docs/VAR.md §5.3.2.1](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/docs/VAR.md) about using `embedding(idx_k)` naively.

**Rule**: when predicting level k given levels `<k`, the context fed to the Predictor must represent the cumulative canvas `H_hat_k = Σ_{j<k} R_j`, **not raw concept stacks `[C_0, ..., C_{k-1}]`**.

Why? Because `H_hat_k` is the *position-aware, smeared* accumulation that captures what "has been painted" at every sequence position. A naked `C_{k-1} ∈ R^{L_{k-1}×D}` is missing:

1. The smearing operator `A_{k-1}^T` that maps `L_{k-1}` concepts back to `L` sequence positions.
2. All prior levels' `R_0, ..., R_{k-2}` that together constitute the canvas.
3. The cross-level index alignment (because `L_k = 2 L_{k-1}` — concept `C_{k,2j}` and `C_{k,2j+1}` are both children of `C_{k-1,j}`, a fact lost if we stack raw `C_j` tensors).

**Two admissible Predictor designs**:

| Design               | Input shape at level k                            | Faithful to §2.5?                            |
|----------------------|---------------------------------------------------|----------------------------------------------|
| Canvas-based         | `downsample(H_hat_k, to=L_k)`                     | ✅ Direct VAR analog (`idxBl_to_var_input`)   |
| Concept-stack + attn | `[C_0, ..., C_{k-1}]` + cross-attn over `H_hat_k` | ✅ Only if cross-attention truly reads canvas |
| Concept-stack alone  | `[C_0, ..., C_{k-1}]` (no canvas)                 | ❌ Loses smearing, ancestor alignment         |

Any Predictor that stacks concepts alone (without canvas reconstruction or a proxy for it) silently violates the rank-accumulation invariant and will need to re-learn `A_j^T` internally for every `j < k` — an expensive waste of parameters.

**How `concept_predictor.py` honours this rule.** The current (and only)
predictor implements the **canvas-based** design directly: the function
`_construct_approx_tokens(k, f_hat_k)` cross-attends `level_queries[k]`
over `back_proj(f_hat_k)`, where `f_hat_k = Σ_{j<k} R_j` is the cumulative
reconstruction canvas (taken from the frozen Builder during training and
self-maintained on a fixed `[B, L_canvas, D]` canvas during inference).
The LLM therefore **never sees raw concept stacks** — it only ever sees
the pre-LLM-extracted approximation tokens that already represent
`f_hat_k` in canvas form. This is the single most important Predictor
correctness property inherited from §2.5.

### 2.5.10 One-Line Mnemonic (For Everyday Use)

> **VAR constrains via a discrete codebook; lcp constrains via matrix rank. Both iteratively peel a residual, both enforce non-overlap through subtraction, both produce a coarse-to-fine pyramid. The only real difference is which algebraic structure (finite set vs. rank-bounded subspace) plays the role of "information capacity ceiling" at each level.**

---

## 3. Intra-Level Analysis: Segment-Concept Correspondence

### 3.1 The DLCM Principle

From dlcm.md Section 3.2, DLCM establishes a **hard segment-concept correspondence**:

```
CoT: "Q: What is 2+3? A: Let me solve this step by step. 2+3=5. Answer: 5"
      └── Segment 0 ──┘└──── Segment 1 ────┘└── Seg 2 ──┘└Seg 3┘└Seg 4┘
            ↓                    ↓                ↓           ↓       ↓
         C_{k,0}              C_{k,1}         C_{k,2}    C_{k,3}  C_{k,4}
```

> **Notation mapping**: DLCM only has a single-level concept partition, so its
> c_1, c_2, c_3, ... correspond to our C_{k,0}, C_{k,1}, C_{k,2}, ... at any
> given level k. DLCM has no inter-level dimension — it only partitions at one
> granularity. Our C_{k,j} generalizes DLCM's c_j by adding the level index k.

Each concept C_{k,j} = mean(S_j) contains information from exactly one disjoint segment. This guarantees:
- **Non-overlap**: No position belongs to two concepts
- **Coverage**: Every position belongs to some concept
- **Ordering**: Segments (and therefore concepts) are sequentially ordered

### 3.2 How Soft Attention Achieves Segment Correspondence

Our design uses **soft attention** rather than hard segmentation. Three mechanisms jointly create segment-like behavior:

#### Mechanism 1: Softmax Competition

```
A_k = softmax(Q_k @ H_rest_k^T / (√D × τ))   shape: [B, L_k, L]
```

For a fixed position t, softmax enforces: Σ_j A_{k,j}(t) = 1. This means concepts **compete** for each position. If C_{5,0} strongly attends to position [0, L/32], then A_{5,0}(t) is large for t ∈ [0, L/32], forcing A_{5,1}(t), ..., A_{5,31}(t) to be small for those positions. This pushes later concepts toward later positions.

#### Mechanism 2: Residual Flow

After level k extracts R_k from H_rest_k, the extracted information is removed:

```
H_rest_{k+1} = H_rest_k - R_k
```

At level k=K-1, H_rest_{K-1} = H_proj - Σ_{i=0}^{K-2} R_i. The residual flow means:
- Positions whose information was already captured by earlier levels have diminished representation in H_rest_5
- C_{5,j} = A_{5,j} @ H_rest_5 can only extract what remains
- This creates a natural "soft boundary" effect: concepts at level 5 physically cannot attend to information already claimed by coarser levels

**Intuitive example**: Think of peeling layers of an onion:
```
H_proj = full information of the CoT

Level 0 extracts: R_0 = "global theme" (e.g., "this is a math calculation")
  → H_rest_1 = H_proj - R_0 = everything EXCEPT the global theme

Level 1 extracts: R_1 = "two major segments" (e.g., "setup | computation")
  → H_rest_2 = H_rest_1 - R_1 = everything EXCEPT global theme and major segments

Level 5 extracts: R_5 = "32 fine-grained details" (e.g., individual step details)
  → H_rest_6 ≈ 0 (almost everything has been accounted for)

Each level can only "see" what coarser levels haven't already taken.
This is why finer levels naturally capture finer details — the coarse
structure has already been subtracted out.
```

**Comparison with DLCM**:
- DLCM: hard boundary, C_{k,j} = mean(S_j), segments are disjoint sets
- Our design: soft boundary, C_{k,j} = A_{k,j} @ H_rest_k, concepts attend to different (mostly non-overlapping) regions because residual removes claimed information

#### Mechanism 3: Ordering Loss (Intra-Level Only)

```
L_order = Σ_k Σ_j ReLU(exp_pos[C_{k,j}] - exp_pos[C_{k,j+1}] + margin)
```

where exp_pos[C_{k,j}] = Σ_t A_{k,j}(t) × t is the expected CoT position that concept C_{k,j} attends to.

This loss enforces:
- **Intra-level ordering**: C_{k,0} attends to earlier positions than C_{k,1}, which attends earlier than C_{k,2}, etc.

Concrete example for Level 5 (32 concepts):
```
CoT: "Let me solve this. First, 2+3=5. Then, 5×4=20. So the answer is 20."

Without ordering loss:            With ordering loss:
  C_{5,0} → "5×4=20" (pos 18)      C_{5,0} → "Let me"      (pos 0)
  C_{5,1} → "Let me"   (pos 0)      C_{5,1} → "solve"      (pos 4)
  C_{5,2} → "2+3=5"   (pos 12)     C_{5,2} → "this."       (pos 8)
  ...  (chaotic, no structure)       ...  (ordered, segment-like)
```

The ordering loss ensures each concept "owns" a contiguous, ordered segment
of the CoT, just like DLCM's hard segmentation — but enforced softly via loss.

> **Why no inter-level ordering?** Inter-level ordering (e.g., "last concept of
> level k attends to earlier positions than first concept of level k+1") is
> **incorrect and unnecessary**.
>
> Remember: each level covers the SAME CoT at a different granularity.
> Level k+1 is a finer partition of the SAME space, not a continuation of it.
>
> Concrete example — a CoT with 100 tokens:
> ```
> Level 1 (2 concepts):  C_{1,0} ~ tokens [0, 50),   C_{1,1} ~ tokens [50, 100)
> Level 2 (4 concepts):  C_{2,0} ~ tokens [0, 25),   C_{2,1} ~ tokens [25, 50),
>                         C_{2,2} ~ tokens [50, 75),  C_{2,3} ~ tokens [75, 100)
> ```
>
> Inter-level ordering would demand: exp_pos[C_{1,1}] < exp_pos[C_{2,0}]
>                                            75          <           12
> This is impossible! C_{1,1} covers the 2nd half of CoT, C_{2,0} covers
> the 1st quarter. There is no sequential relationship between them — they
> are different granularities of the same CoT.
>
> The coarse-to-fine structure is already guaranteed by:
> 1. **Rank bottleneck**: Level 0 can only capture 1 direction, level 5 can
>    capture 32 directions — finer levels have more capacity by construction.
> 2. **Residual flow**: H_rest_{k+1} = H_proj - R_0 - ... - R_k. Each level
>    picks up what coarser levels left behind. Finer levels naturally capture
>    finer residual details.

### 3.3 Soft vs Hard Segmentation: Theoretical Comparison

| Property              | DLCM (Hard)                       | Our Design (Soft)                                    | Verdict                               |
|-----------------------|-----------------------------------|------------------------------------------------------|---------------------------------------|
| Non-overlap           | Guaranteed by disjoint segments   | Soft — concepts can have overlapping attention tails | Soft is less strict but more flexible |
| Coverage              | Guaranteed by partition           | Guaranteed by recon loss ‖H_hat - H_proj‖²           | Both guarantee                        |
| Ordering              | Guaranteed by sequential segments | Enforced by ordering loss                            | Both achieve                          |
| Adaptive boundaries   | Similarity threshold τ            | Learned via concept_queries                          | Soft is more adaptive                 |
| Boundary sharpness    | Binary (boundary or not)          | Gradual (attention weights decay smoothly)           | Soft handles fuzzy boundaries better  |
| Multi-scale hierarchy | None (single granularity)         | K levels, coarse-to-fine                             | Soft is strictly superior             |
| Differentiability     | Threshold not differentiable      | Fully differentiable                                 | Soft is strictly superior             |

**Key insight**: DLCM's hard segmentation is a special case of soft attention where attention weights are binary (0 or 1). Our soft attention can learn to approximate hard segmentation when appropriate, but also allows smooth transitions where semantic boundaries are fuzzy. This is **strictly more expressive** than hard segmentation.

### 3.4 Why Soft Attention Is Sufficient for Segment Correspondence

The concern is: "Can soft attention actually learn focused, segment-like patterns, or will it remain diffuse?"

**Argument for sufficiency**:

1. **Competition forces focus**: In level 5 with 32 concepts, if C_{5,0} and C_{5,1} both attend diffusely to [0, L/2], they would produce nearly identical concepts. The NTP loss (from the decoder) would penalize redundancy — if two concepts carry the same information, one is wasted. The model is incentivized to differentiate concepts by attending to different positions.

2. **Residual flow prevents overlap**: Even without ordering loss, the residual flow naturally creates soft boundaries. If C_{5,0} extracts information from positions [0, L/32], that information is subtracted from H_rest for subsequent concepts.

3. **Ordering loss provides explicit pressure**: The ordering loss directly pushes concepts toward sequential, non-overlapping attention patterns.

4. **Positional query initialization**: When `use_positional_query_init=True`, concept queries start with positional priors that bias C_{k,j} toward the j-th segment of the sequence. This accelerates the discovery of segment structure.

**Potential failure mode**: If the temperature τ is too high, attention becomes too diffuse (close to uniform). The learnable temperature parameter addresses this — the model can lower τ to sharpen attention. However, if initialization is poor, the model may get stuck in a diffuse-attention local minimum.

**Mitigation**: Positional query initialization (`use_positional_query_init=True`) provides a strong starting point that avoids this failure mode.

### 3.5 Concept Position vs Concept Content

A subtle but important distinction:

**Concept Position** (where does C_{k,j} attend?): Determined by A_{k,j} — which positions contribute to C_{k,j}. This is governed by ordering loss and softmax competition.

**Concept Content** (what does C_{k,j} contain?): Determined by A_{k,j} @ H_rest_k — what information is extracted from those positions. This is governed by the encoder representations and the level_proj transformation.

In DLCM: C_{k,j} = mean(S_j). The content is simply the average of token representations in segment S_j. The position is determined by the segment boundaries.

> **Notation**: DLCM uses c_k for its single-level concepts. Since DLCM only has one
> level of segmentation, DLCM's c_k ≡ our C_{k,j} at whichever single level DLCM
> operates. Our notation C_{k,j} subsumes DLCM's by adding the level dimension.

In our design: C_{k,j} = level_proj(A_{k,j} @ H_rest_k). The content is a learned, weighted combination of residual representations. The position emerges from attention patterns.

Our design is strictly more expressive because:
1. **Weighted** combination (not just mean) — more important positions get higher weight
2. **level_proj** transformation — can extract task-relevant features from the pooled representation
3. **Residual input** — at level k, the input is H_rest_k (what hasn't been captured yet), not the original H. This means C_{k,j} contains genuinely new information, not redundant overlap with coarser concepts.

---

## 4. Two-Phase Architecture: Builder and Predictor

Following VAR's design principle, we explicitly separate **concept extraction** from **concept generation**:

### 4.1 ConceptPyramidBuilder (Phase 1: Extraction)

The Builder constructs the groundtruth concept pyramid from CoT using soft attention and residual decomposition. This subsection enumerates **every** component declared in `ConceptPyramidBuilder.__init__` (see [examples/lcp/concept_builder.py](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/examples/lcp/concept_builder.py)) and gives the design reason for each, followed by the forward-pass pipeline and the output dataclasses.

**Input**: `BuilderInput(questions, cot_answers, solutions)`
- **CoT**: core source for building the concept pyramid (encoded by `reason_model`).
- **Q**: context/prior used only by the reasoning loss; does **not** enter the pyramid.
- **Solution**: target for `L_reasoning`; concepts must reconstruct enough information to predict it.

**Output**: `PyramidOutput` containing the full pyramid `[C_0, ..., C_{K-1}]` plus all intermediate tensors needed for external loss computation.

#### 4.1.1 Components (`ConceptPyramidBuilder.__init__`)

| Component         | Shape / Type                                | Role and design reason                                                                                                                                                                                                                                                                                                                                                                                                   |
|-------------------|---------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `reason_model`    | `AutoModelForCausalLM` (e.g., Qwen2.5-0.5B) | One model, two roles: (1) backbone produces `H_CoT` for concept extraction; (2) `lm_head` computes the reasoning loss on `[Q; concepts; S]`. Loaded as Causal LM so a separate solution decoder is unnecessary.                                                                                                                                                                                                          |
| `tokenizer`       | `AutoTokenizer` paired with `reason_model`  | Tokenizes CoT / Q / S; `pad_token` falls back to `eos_token` when the model has none.                                                                                                                                                                                                                                                                                                                                    |
| `input_proj`      | `Linear(D_encoder, D)` (with bias)          | Maps reason_model hidden states from encoder space `D_encoder` to concept space `D`. When `D == D_encoder`, this is a same-dim learned rotation, mirroring VAR's `quant_conv` (preserves dimension to keep `back_proj` a faithful inverse).                                                                                                                                                                              |
| `input_proj_norm` | `LayerNorm(D)`                              | Normalises `H_proj`. Reason: raw Qwen2.5 hidden states have `std ≈ 10`, `max ≈ 200`; without LayerNorm the random pyramid explodes (reconstructed `std ≈ 200` vs. projected `std ≈ 12`, giving `recon_loss ≈ 4.4e4`). LayerNorm makes recon loss start at a sane magnitude.                                                                                                                                              |
| `concept_queries` | `ParameterList` of K, each `[L_k, D]`       | Learnable queries that define *what to attend to* at level k. Query-expansion schedule `L_k = 2^k`, i.e. `1 → 2 → 4 → 8 → 16 → 32` for K=6. Functionally replace VAR's discrete codebook with a continuous, level-specific query bank.                                                                                                                                                                                   |
| `temperature`     | `Parameter(torch.ones(1))`, scalar τ        | Learnable attention sharpness in `A_k = softmax(Q_k H_rest_k^⊤ / (√D · τ))`. Too large → diffuse attention; too small → sharp but inflexible. Letting τ be learnable lets the model anneal sharpness during training.                                                                                                                                                                                                    |
| `level_projs`     | `ModuleList` of K, each `Linear(D, D)`      | Per-level output projection `C_k = level_proj_k(A_k @ H_rest_k)`. Reason for *per-level* (not shared) projection: each level operates on a different residual `H_rest_k` whose statistics shift as coarse content is removed; an independent projection per level lets the model adapt to that drift.                                                                                                                    |
| `back_proj`       | `Linear(D, D_encoder, bias=False)`          | Maps concept-space tensors back to encoder space. Used in two places: (i) `L_recon = ‖back_proj(H_hat_K) − H_CoT‖²` so reconstruction is measured against the *stable* encoder output, not the projected one; (ii) `_prepare_reasoning` feeds `back_proj(concepts)` into `reason_model` for `L_reasoning`. Initialised as `input_proj.weight.T` (pseudo-inverse) so it starts as an approximate inverse of `input_proj`. |

Training-strategy flags (read from `config["training"]["reason_model"]`):

| Flag                                              | Effect on `reason_model`                                                                                                           |
|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| `freeze: true`                                    | All `reason_model` parameters get `requires_grad=False`. Mirrors VAR's frozen VQ-VAE encoder — stable target.                      |
| `lora: {r, alpha, target_modules, dropout, bias}` | PEFT LoRA adapters injected into `target_modules` (default `q_proj`, `v_proj`); only LoRA params are trainable when `freeze=true`. |
| `reason_model_num_layers: N` (>0)                 | Truncates the backbone to its first `N` Transformer layers (works for both plain and PEFT-wrapped models). `-1` disables pruning.  |

**There is no separate `solution_decoder`**, no `concept_transformer`, and no `start_token`. The model around which Stage 1 is built is exactly `reason_model`; everything else (`input_proj`, `input_proj_norm`, `concept_queries`, `temperature`, `level_projs`, `back_proj`) is the *trainable shell* that turns it into a pyramid extractor.

**Dimension-consistency warning (runtime).** `__init__` checks `pyramid.hidden_dim == reason_model.config.hidden_size` and emits a `UserWarning` when they differ. Reason: VAR's `quant_conv` preserves channel count so the inverse `post_quant_conv` is faithful; if our `D ≠ D_encoder`, then `input_proj` becomes a lossy compression and `back_proj` cannot perfectly invert it, putting a non-zero floor on `L_recon` that is unrelated to the pyramid's capacity. Set `D = D_encoder` for VAR-faithful, lossless projection.

**Config caches (bookkeeping, not learnable).** `self.config`, `self.reason_cfg = config["model"]["reason_model"]`, `self.pyramid_cfg = config["model"]["pyramid"]`, `self.builder_cfg = config["model"]["builder"]`, `self.use_positional_query_init`, and `self.train_rm_cfg = config["training"]["reason_model"]` are cached at construction time to avoid repeated deep-dict lookups in the hot forward path. They store no parameters.

#### 4.1.2 Output dataclasses

Each forward stage returns a typed dataclass instead of a loose `dict`, so downstream losses access fields by name. Defined in `concept_builder.py`:

| Dataclass       | Returned by                       | Fields (shapes)                                                                                                                                                                                                                                                                                                                                                                                                                   |
|-----------------|-----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `EncoderOutput` | `encode_cot`                      | `hidden_states [B, L, D_encoder]`, `attention_mask [B, L]`                                                                                                                                                                                                                                                                                                                                                                        |
| `LevelOutput`   | per-level inside `_build_pyramid` | `concepts [B, L_k, D]` (= `C_k`), `attention_weights [B, L_k, L]` (= `A_k`), `reconstruction [B, L, D]` (= `R_k`)                                                                                                                                                                                                                                                                                                                 |
| `PyramidOutput` | `_build_pyramid` / `forward`      | `concepts: List[Tensor]` (= `[C_0, ..., C_{K-1}]`), `level_outputs: List[LevelOutput]`, `encoder_hidden_states` (`H_CoT`), `projected_hidden` (`H_proj`), `reconstructed_hidden` (`H_hat_K`), `reconstructed_encoder_hidden` (`back_proj(H_hat_K)`), `residual_hidden` (`H_rest_K`), `num_levels`, `level_lengths`, `attention_mask`, optional `reasoning_logits`, `reasoning_target_ids`, `reasoning_texts`, `generation_texts`. |

`PyramidOutput` exposes three convenience accessors used by the loss layer: `total_concepts → Σ L_k`, `all_attentions → [A_0, ..., A_{K-1}]`, `all_reconstructions → [R_0, ..., R_{K-1}]`, and `cat_concepts() → [B, Σ L_k, D]`.

#### 4.1.3 Forward-pass pipeline

`forward(batch: BuilderInput) → PyramidOutput` is a three-step pipeline:

**Step 1 — `encode_cot(cot_answers)`** (returns `EncoderOutput`)
```
backbone = self._get_backbone()           # the Transformer backbone, NOT the lm_head
H_CoT    = backbone(input_ids=tok(CoT), attention_mask=...).last_hidden_state
                                          # [B, L, D_encoder]
```
The `lm_head` is deliberately **skipped** here; it is reserved for Step 3 (reasoning loss). The helper `_get_backbone()` returns the right inner module regardless of whether `reason_model` is plain (`reason_model.model`) or PEFT-wrapped (`reason_model.base_model.model`); all later embedding and forward calls go through it.

**Step 2 — `_build_pyramid(H_CoT, attention_mask)`** (returns `PyramidOutput` with empty reasoning fields)
```
H_proj   = input_proj_norm(input_proj(H_CoT))             # [B, L, D]
H_rest   = H_proj.clone()                                   # H_rest_0
H_hat    = zeros_like(H_proj)                               # H_hat_0
for k in 0..K-1:
    Q_k       = concept_queries[k]                          # [L_k, D]
    scores    = (Q_k_batched @ H_rest^⊤) / (√D · τ)         # [B, L_k, L]
    scores    = scores.masked_fill(pad_mask == 0, -inf)     # ignore padding
    A_k       = softmax(scores, dim=-1)                     # [B, L_k, L]
    C_k       = level_projs[k](A_k @ H_rest)                # [B, L_k, D]
    R_k       = A_k^⊤ @ C_k                                  # [B, L, D]
    H_hat    += R_k
    H_rest   -= R_k
H_recon = back_proj(H_hat)                                  # [B, L, D_encoder]
```
Key points:
- The padding mask is applied **before** softmax, then `nan_to_num` cleans up any all-`-inf` rows (concepts whose context is fully masked).
- The decomposition is **purely residual**: each level only sees `H_rest_k`, never previous concepts directly.
- `level_projs[k]` is per-level (not shared across k) because the residual statistics drift as coarse content is removed.

**Step 3 — `_prepare_reasoning(pyramid, q_ids, q_mask, sol_ids, sol_mask)`** (mutates `pyramid` in place, only when `batch.has_solution`)
```
concept_embeds = back_proj(pyramid.cat_concepts())             # [B, Σ L_k, D_encoder]
Q_embeds       = embed_tokens(q_ids)                             # [B, L_Q, D_encoder]
S_embeds       = embed_tokens(sol_ids)                           # [B, L_S, D_encoder]
seq            = cat([Q_embeds, concept_embeds, S_embeds], dim=1)
mask           = cat([q_mask, ones(Σ L_k), sol_mask], dim=1)
logits         = reason_model(inputs_embeds=seq, attention_mask=mask).logits
solution_logits = logits[:, L_Q + Σ L_k - 1 : L_Q + Σ L_k + L_S - 1, :]
targets        = sol_ids.clone();  targets[sol_mask == 0] = -100
pyramid.reasoning_logits     = solution_logits
pyramid.reasoning_target_ids = targets
pyramid.reasoning_texts      = tokenizer.batch_decode(solution_logits.argmax(-1))
```
This runs the full `reason_model` (backbone + lm_head) on `[Q ; back_proj(concepts) ; S]` to validate that the pyramid retains enough information to bridge `Q → S`.

A companion method `generate_solution(pyramid, q_ids, q_mask, max_new_tokens)` performs *free* autoregressive generation on `[Q ; back_proj(concepts)]` (no solution input) and returns decoded strings — used at evaluation time to compare teacher-forced vs. autoregressive quality. To avoid feeding right-padding tokens into `reason_model.generate`, it calls `pack_qcs_sequences` (from `lcp.utils`) which re-packs each row as `[real_Q_i | concepts | tail_pad]` so the prompt has no internal padding.

`back_decode(x)` is a thin wrapper around `back_proj(x)` kept as a separate method so it can later evolve into a fuller decoder (LayerNorm, MLPs) without changing call sites. `_get_backbone()` is the analogous helper for backbone access (plain vs. PEFT-wrapped).

#### 4.1.4 Initialisation (`_init_weights`)

| Component         | Init scheme                                                                                                                                          | Reason                                                                                                                                                                                                 |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `input_proj`      | Xavier-uniform weight, zero bias                                                                                                                     | Standard linear init.                                                                                                                                                                                  |
| `concept_queries` | If `use_positional_query_init=true`: `xavier_uniform + α · PE(j/L_k)` with sinusoidal PE at normalised positions `j/L_k`. Else: pure xavier-uniform. | The positional bias gives query `Q_{k,j}` a prior on segment `j` of the CoT, accelerating discovery of segment-concept correspondence (Section 6.2). `α` is read from `builder.positional_init_alpha`. |
| `level_projs`     | Xavier-uniform weight, zero bias                                                                                                                     | Standard linear init.                                                                                                                                                                                  |
| `back_proj`       | `back_proj.weight ← input_proj.weight.T` (no bias)                                                                                                   | Pseudo-inverse start: if `input_proj` maps `H_CoT → H_proj`, then `back_proj` initially maps `H_proj ≈ H_CoT`. Both layers remain free to learn.                                                       |
| `temperature`     | `ones(1)`                                                                                                                                            | Neutral starting sharpness (τ = 1).                                                                                                                                                                    |

#### 4.1.5 Loss hooks (computed externally in `losses.py`)

The Builder module itself does **not** compute losses; it returns a `PyramidOutput` whose fields are exactly what `compute_builder_loss` consumes. The four-term objective is:

```
L_builder = w_recon · L_recon + w_order · L_order + w_residual · L_residual + w_reasoning · L_reasoning

L_recon     = ‖reconstructed_encoder_hidden − encoder_hidden_states‖²          # back_proj(H_hat_K) vs. H_CoT
L_order     = ordering loss over [A_0, ..., A_{K-1}] (Section 3.2)
L_residual  = ‖residual_hidden‖₁                                              # ‖H_rest_K‖₁ — concept-space sparsity prior
L_reasoning = CE(reasoning_logits, reasoning_target_ids)                      # NTP on solution tokens
```

**Mechanism (one-line summary)**: at each level `k`, the Builder takes the current residual `H_rest_k`, uses `L_k` learnable queries to extract a rank-`L_k` summary `C_k`, broadcasts it back to sequence length as `R_k = A_k^⊤ C_k`, adds `R_k` to `H_hat` and subtracts it from `H_rest`, and hands the remainder to the next level (whose `2×`-wider query bank attends again).

**Key properties**:
- The Builder is used **only during training**; at inference time the Predictor takes over and Builder weights are not loaded into memory.
- Only the *trainable shell* (`input_proj`, `input_proj_norm`, `concept_queries`, `temperature`, `level_projs`, `back_proj`) plus optionally `reason_model`'s LoRA adapters are updated; the backbone itself is frozen by default.
- `PyramidOutput` is the single contract between Builder and Predictor: the Predictor consumes `pyramid.concepts` (detached) as its training targets.

### 4.2 ConceptPredictor (Phase 2: Generation)

The Predictor learns to generate the concept pyramid from `Q` alone,
using a causal decoder-only LLM (`reason_model`) as its backbone. It is
**single-sequence and VAR-faithful**: a fixed packed layout per row is
fed through ONE LLM forward at training time, and concept prediction +
reasoning NTP are read out from the same hidden states. There are no
alternative modes (no flat-AR variant, no shared/independent toggle).
The key idea is that **per-level approximation tokens are constructed
before entering the LLM**, so the LLM sees a compact sequence whose
length does not depend on the (potentially large) raw context length
of `f_hat_k`.

#### 4.2.1 Components (`ConceptPredictor.__init__`)

| Component       | Shape / Type                                             | Role                                                                                                             |
|-----------------|----------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `builder`       | frozen `ConceptPyramidBuilder`                           | Produces GT concepts and per-level cumulative `f_hat_k` during training. All parameters frozen.                  |
| `reason_model`  | own `AutoModelForCausalLM` (+ optional LoRA via PEFT)    | Predictor's own backbone (loaded from `predictor_cfg.model_name`). Base weights frozen; LoRA adapters trainable. |
| `tokenizer`     | own `AutoTokenizer`                                      | Tokenises questions and (when present) solutions for the predictor's own vocabulary.                             |
| `back_proj`     | `Linear(D, D_enc, bias=False)`                           | Lifts concept-space (`D`) to encoder/embedding-space (`D_enc`) for `f_hat` and predicted concepts.               |
| `level_queries` | `ParameterList([Tensor[L_k, D_enc] for k in 0..K-1])`    | Per-level learnable queries. Cross-attend to `back_proj(f_hat_k)` to extract `L_k` approximation tokens.         |
| `query_norm`    | `LayerNorm(D_enc)`                                       | Pre-norm on query side of pre-LLM cross-attention.                                                               |
| `context_norm`  | `LayerNorm(D_enc)`                                       | Pre-norm on context (KV) side of pre-LLM cross-attention.                                                        |
| `extract_attn`  | `MultiheadAttention(D_enc, num_heads, batch_first=True)` | Pre-LLM cross-attention shared across all levels; constructs approximation tokens.                               |
| `post_norm`     | `LayerNorm(D_enc)`                                       | Post-norm after attention + residual on raw queries.                                                             |
| `lvl_embed`     | `Embedding(K, D_enc)`                                    | Per-level identity tag added to every approximation token at level `k` (analogous to VAR's `lvl_emb`).           |
| `concept_head`  | `Linear(D_enc, D_enc) → GELU → Linear(D_enc, D)`         | Maps backbone hidden state at approx-token positions back to concept space to produce Ĉ_k.                       |

The `_inference_canvas_length` (default 128) sets the fixed canvas size
used at inference time when the Builder is unavailable. There is **no**
`level_embeddings` table on individual concept tokens, no
`position_embeddings`, no `q_proj`, no `start_token`, and no
`use_shared_model` toggle: the predictor always owns its own copy of
`reason_model`, `back_proj`, and `tokenizer`.

#### 4.2.2 Pre-LLM approximation-token construction (`_construct_approx_tokens`)

For each level `k`, the predictor first turns the cumulative reconstruction
`f_hat_k ∈ ℝ^{B × ctx × D}` (from the Builder during training, or self-
maintained at inference) into exactly `L_k` approximation tokens of size
`D_enc`, **before** anything reaches the LLM:

```
f_hat_k             [B, ctx, D]                      # cumulative canvas
   │
   ▼ back_proj  (D → D_enc)
context             [B, ctx, D_enc]
   │
   │  queries = level_queries[k]                    [L_k, D_enc]
   │  q_n = query_norm(queries.expand(B, L_k, D_enc))
   │  c_n = context_norm(context)
   ▼
extract_attn(q_n, c_n, c_n)  → attn_out [B, L_k, D_enc],
                              attn_w   [B, L_k, ctx]
   │
   ▼ residual on RAW queries + post-norm
approx_tokens   = post_norm(attn_out + queries)      [B, L_k, D_enc]
approx_tokens  += lvl_embed.weight[k]                # level identity
```

The context length `ctx` is whatever the Builder produced (training) or
the canvas length `L_canvas` (inference). The output shape `[B, L_k,
D_enc]` is **fixed by the level**, not by `ctx` — this is what makes the
downstream LLM input a compact `[L_Q + Σ L_k]` sequence regardless of how
long the underlying CoT / canvas is.

#### 4.2.3 Per-row packed layout and scale-causal mask

After approximation tokens for all `K` levels are concatenated, each row
is packed (via `pack_qcs_sequences`) into:

```
[real_Q_i | approx_tokens_0 | approx_tokens_1 | ... | approx_tokens_{K-1} | real_S_i | tail_pad]
  q_len[i]    L_0              L_1                    L_{K-1}              s_len[i]    …
```

This layout has four scale regions per row:

- **scale 0** (`Q`): `q_len[i]` real question tokens (left-aligned).
- **scale 1..K** (`approx`): `L_k` approximation tokens for each level
  `k`, contiguous in level order.
- **scale K+1** (`S`): `s_len[i]` real solution tokens (training only).
- **pad**: anywhere outside the above ranges.

`_build_scale_causal_mask_packed` materialises a per-row 4D additive mask
that encodes the following visibility rule (let `t_q` be the query, `t_k`
the key, both packed positions):

| Same scale?      | Visibility within scale        |
|------------------|--------------------------------|
| `Q`              | token-causal (`t_q ≥ t_k`)     |
| `approx` level k | **bidirectional** within level |
| `S`              | token-causal (`t_q ≥ t_k`)     |

| Cross-scale (scale_q vs scale_k)    | Visibility                         |
|-------------------------------------|------------------------------------|
| `Q` → anything later                | masked                             |
| `approx` level k → `Q`              | visible                            |
| `approx` level k → `approx` level j | visible iff `j ≤ k` (scale-causal) |
| `approx` level k → `S`              | masked                             |
| `S` → `Q`, `S` → `approx` (any k)   | visible                            |
| anything → pad                      | masked                             |

This matches VAR's scale-causal pattern at the level granularity, while
allowing the `L_k` approximation tokens within a single level to fully
exchange information — they describe the *same* level and have no
intrinsic order.

#### 4.2.4 Training forward pass (`forward`, single packed pass)

Given a `BuilderInput` batch with questions and solutions:

```
# Phase 1 — frozen Builder produces GT pyramid (no_grad, detached)
pyramid       = builder(batch)
gt_concepts   = [c.detach() for c in pyramid.concepts]            # K × [B, L_k, D]
gt_f_hats     = [f.detach() for f in pyramid.f_hat_per_level]     # K × [B, ctx, D]

# Tokenise Q (and S if present) with the predictor's OWN tokenizer
question_ids, q_mask = tokenizer(batch.questions, ...)
if batch.has_solution:
    solution_ids, s_mask = tokenizer(batch.solutions, ...)

# Phase 2 — build approx tokens for all levels and pack
Q_embeds = backbone.get_input_embeddings()(question_ids)          # [B, L_Q, D_enc]
approx_tokens = cat([_construct_approx_tokens(k, gt_f_hats[k])[0]
                     for k in range(K)], dim=1)                    # [B, Σ L_k, D_enc]
S_embeds = backbone.get_input_embeddings()(solution_ids)           # or None

pack = pack_qcs_sequences(Q_embeds, q_mask,
                          approx_tokens, S_embeds, s_mask)

attn_4d = _build_scale_causal_mask_packed(
    q_len=pack.q_len, s_len=pack.s_len,
    level_lengths=level_lengths, T=pack.T, ...)

# ONE LLM forward (concept + reasoning paths share these hidden states)
out    = reason_model(inputs_embeds=pack.packed_embeds,
                      attention_mask=attn_4d,
                      output_hidden_states=True)
hidden = out.hidden_states[-1]                                    # [B, T, D_enc]
logits = out.logits                                               # [B, T, V]

# Readout A — concept predictions at approx-token positions
#   col = q_len[i] + j   for j in [0, Σ L_k)
approx_hidden = gather(hidden, rows=B, cols=q_len[:, None] + arange(ΣL_k))
predicted_concepts = split(concept_head(approx_hidden), level_lengths)

# Readout B — reasoning NTP logits at solution-predicting positions
out.reasoning_logits     = gather_solution_logits(logits, pack)
out.reasoning_target_ids = build_solution_targets(solution_ids, s_mask, pack)
```

**Visual: the packed sequence and its two readouts** (K=3, L=[1,2,4]):

```
 position : 0 1 2 ... q-1 | q  q+1  q+2  q+3  q+4  q+5  q+6 | q+7 q+8 ... q+7+L_S-1
 kind     : Q Q Q ...  Q  | a  a    a    a    a    a    a   |  S   S  ...    S
 level k  : .   .  ...  . | 0  1    1    2    2    2    2   |  .   .  ...    .
                            │  │         │
                            │  │         └─ lvl_embed[k=2] added to all 4 approx tokens of level 2
                            │  └────────── lvl_embed[k=1] added to both approx tokens of level 1
                            └───────────── lvl_embed[k=0] added to single approx token of level 0

            SCALE-CAUSAL 4D MASK (per row, see §4.2.3)

 READOUT A (concept_head)                  READOUT B (reason_model.lm_head)
  hidden[approx_token positions]            logits[solution-predicting positions]
  → Ĉ_0, Ĉ_1, ..., Ĉ_{K-1}                  → reasoning_logits
              ↘                          ↙
              both come from ONE forward over `hidden`
```

Key properties:

- **Single LLM forward, two losses.** Concept MSE (via `concept_head`)
  and reasoning CE (via `reason_model.lm_head`) are produced from the
  same `hidden` / `logits` tensors. No second pass.
- **Pre-LLM cross-attention compresses `f_hat_k` to `L_k` tokens.** The
  LLM sees a sequence of length `L_Q + Σ L_k (+ L_S)`, independent of
  `ctx`. This is the text-domain analog of VAR's spatial downsampling.
- **Scale-causal visibility.** Level-`k` approximation tokens can attend
  to `Q` plus all earlier levels (and to each other within level `k`),
  but cannot peek at later levels or at `S`. Solution tokens see `Q` and
  all approximation tokens, then are token-causal among themselves.
- **Per-level identity via `lvl_embed`.** Within a level, the `L_k`
  approximation tokens are made distinguishable by the learned
  `level_queries[k]`; identity *across* levels is provided additively by
  `lvl_embed[k]`. Position embeddings on individual concepts are not
  needed because intra-level visibility is bidirectional.
- **Teacher forcing through `f_hat`.** At training time `gt_f_hats[k]` is
  detached from the Builder, so the predictor sees the same cumulative
  canvas at level `k` that the Builder constructed from GT CoT, exactly
  matching the inference protocol.

#### 4.2.5 Inference forward pass (`_forward_inference`, K sequential passes)

At test time the Builder is not used. The predictor self-maintains the
`f_hat` canvas of fixed length `L_canvas` and runs `K` sequential packed
passes, growing the approx-token segment by `L_k` tokens per level:

```
f_hat = zeros [B, L_canvas, D]
Q_embeds = backbone.get_input_embeddings()(question_ids)   # [B, L_Q, D_enc]
approx_token_list = []

for k in 0 .. K-1:
    # 1. Pre-LLM approximation-token construction from current f_hat
    approx_tokens_k, attn_w = _construct_approx_tokens(k, f_hat)
    #     approx_tokens_k: [B, L_k, D_enc]
    #     attn_w:          [B, L_k, L_canvas]
    approx_token_list.append(approx_tokens_k)
    approx_tokens_0k = cat(approx_token_list, dim=1)        # [B, Σ_{j≤k} L_j, D_enc]

    # 2. Pack [real_Q | approx_tokens_0..k | tail_pad] (no S region)
    pack = pack_qcs_sequences(Q_embeds, q_mask,
                              approx_tokens_0k, None, None)
    attn_4d = _build_scale_causal_mask_packed(
        q_len=pack.q_len, s_len=pack.s_len,
        level_lengths=level_lengths[: k+1], T=pack.T, ...)

    # 3. Single backbone forward; gather hidden at level-k approx positions
    hidden = backbone(inputs_embeds=pack.packed_embeds,
                      attention_mask=attn_4d,
                      output_hidden_states=True).hidden_states[-1]
    prev   = sum(level_lengths[:k])
    cols   = pack.q_len[:, None] + prev + arange(L_k)
    Ĉ_k   = concept_head(hidden[rows, cols])                # [B, L_k, D]
    predicted_concepts.append(Ĉ_k)

    # 4. Reconstruction: lift attention weights from queries onto canvas
    R_k    = bmm(attn_w.transpose(1,2), Ĉ_k)                # [B, L_canvas, D]
    f_hat += R_k
```

This is the predictor-side analogue of the Builder's
`f_hat = Σ_{j<k} A_j^T @ C_j`: at inference the contribution of level `k`
to the canvas is `R_k = α_k^T @ Ĉ_k`, where `α_k` are the cross-attention
weights produced by `_construct_approx_tokens(k, f_hat)`.

Key properties:

- **Fixed-length canvas.** `f_hat` always has shape `[B, L_canvas, D]`;
  only its content evolves as more levels are predicted.
- **No KV cache.** Because `f_hat` (and therefore the approximation
  tokens for previously processed levels) changes between passes, the
  packed sequence is fully recomputed at each pass.
- **Same scale-causal mask as training.** Each pass uses
  `_build_scale_causal_mask_packed` over the partial layout
  `[real_Q | approx_tokens_0..k]`, ensuring the model never attends to
  levels it has not yet predicted.
- **Self-consistent with training.** The same `_construct_approx_tokens`
  module that produced approx tokens from GT `f_hat_k` during training
  is reused at inference — no module is trained-only or inference-only.

#### 4.2.6 Solution generation (`generate_solution`)

After `_forward_inference` produces `predicted_concepts`, free
autoregressive solution generation is delegated to `reason_model.generate`
over a packed `[real_Q | back_proj(Σ Ĉ) ]` prefix:

```
concepts_flat   = cat(predicted_concepts, dim=1)              # [B, Σ L_k, D]
concept_embeds  = back_proj(concepts_flat)                    # [B, Σ L_k, D_enc]
pack            = pack_qcs_sequences(Q_embeds, q_mask,
                                     concept_embeds, None, None)
generated_ids   = reason_model.generate(
    inputs_embeds=pack.packed_embeds,
    attention_mask=pack.packed_mask,
    max_new_tokens=max_new_tokens,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    do_sample=False)
new_ids         = generated_ids[:, pack.packed_embeds.shape[1] :]
return tokenizer.batch_decode(new_ids, skip_special_tokens=True)
```

Note that `generate_solution` re-uses the **predicted concepts as raw
back-projected embeddings** (no `_construct_approx_tokens`, no `lvl_embed`
added). This is purely a reasoning prefix for `.generate` and does not
feed back into the concept-prediction loop.

#### 4.2.7 Initialisation (`_init_weights`) and trainable parameters

```
level_queries      : N(0, 0.02)               # one per level, [L_k, D_enc]
lvl_embed.weight   : N(0, 0.02)               # [K, D_enc]
back_proj.weight   : Xavier uniform           # [D_enc, D]
concept_head       : Xavier uniform on each Linear, zeros on bias
extract_attn       : PyTorch defaults
reason_model base  : FROZEN
reason_model LoRA  : trainable (when configured via predictor_cfg.lora)
```

The set of always-trainable parameters is therefore: `back_proj`,
`level_queries`, `query_norm`, `context_norm`, `extract_attn`,
`post_norm`, `lvl_embed`, `concept_head`, plus optional LoRA adapters on
`reason_model`.

#### 4.2.8 Summary

> **The predictor compresses each level's cumulative canvas `f_hat_k`
> into exactly `L_k` approximation tokens via a pre-LLM cross-attention
> head, packs them into a single `[Q | approx | S]` sequence with a
> per-row scale-causal 4D mask, runs ONE LLM forward, and reads concept
> predictions and reasoning logits off the same hidden states. At
> inference, the same machinery is run `K` times, each pass extending the
> approx segment and updating a fixed-length `f_hat` canvas via
> attention-transpose reconstruction.**

### 4.4 Why This Separation?

**VAR's Lesson**: VQ-VAE (extraction) and Transformer (generation) are separate because:
1. Extraction requires seeing the full information
2. Generation requires predicting without seeing the target

**Our Design**:
- **Builder**: Has access to CoT, uses soft attention to extract hierarchical structure
- **Predictor**: Only sees Q, learns to generate the same structure autoregressively

**Benefits**:
1. **Clear training signal**: Builder provides high-quality groundtruth
2. **Aligned inference**: Predictor mimics Builder's output distribution
3. **Efficient inference**: No need to generate CoT, directly predict concepts

### 4.5 Relationship to VAR

| VAR Component             | Our Equivalent        | Role                                            |
|---------------------------|-----------------------|-------------------------------------------------|
| VQ-VAE (Phase 1)          | ConceptPyramidBuilder | Extract groundtruth from full information       |
| VAR Transformer (Phase 2) | ConceptPredictor      | Generate autoregressively from condition        |
| Multi-scale indices       | Concept pyramid       | Hierarchical discrete/continuous representation |
| VAE Decoder               | Solution Decoder      | Decode final output from concepts               |

**Key Difference**: VAR predicts discrete indices; we predict continuous concepts. This is because:
- Our Builder uses soft attention (continuous)
- We want to preserve gradient flow end-to-end
- Continuous concepts are more expressive for text reasoning

---

## 5. Loss Function Analysis

We have two separate loss functions for the two phases.

### 5.1 ConceptPyramidBuilder Loss

The Builder's loss ensures high-quality groundtruth concept pyramid extraction.

#### 5.1.1 Reconstruction Loss

```
L_recon = ||back_proj(H_hat_K) - H_CoT||²
```

Ensures the concept pyramid **preserves all information** from CoT. The reconstruction is compared in encoder space via `back_proj` (maps concept space D back to encoder space D_encoder).

**What it guarantees**: If L_recon → 0, then back_proj(Σ_k A_k^T @ C_k) ≈ H_CoT. Every position in H_CoT is reconstructable from the concept pyramid.

#### 5.1.2 Ordering Loss (Intra-Level Only)

```
L_order = Σ_k Σ_j ReLU(exp_pos[C_{k,j}] - exp_pos[C_{k,j+1}] + margin)
```

where exp_pos[C_{k,j}] = Σ_t A_{k,j}(t) × t is the expected CoT position.

Ensures concepts within each level are ordered by CoT position (Section 3.2).

**Why no inter-level ordering**: Levels cover the SAME CoT at different granularities, not sequential segments (Section 3.2).

#### 5.1.3 Reasoning Loss (NTP)

```
L_reasoning = CrossEntropy(reason_model([Q_embeds; concept_embeds; S_embeds]), solution_tokens)
```

Validates that the concept pyramid supports reasoning. The input sequence
follows the causal ordering [Q, Concepts, S] — mirroring the original
Q -> CoT -> Solution flow. Question and solution tokens are embedded via
the frozen embed_tokens, concepts are back-projected to encoder space via
back_proj, and the concatenated sequence is fed through the frozen
reason_model (including lm_head) with teacher-forcing. Cross-entropy loss
on solution-position logits ensures the pyramid is useful for reasoning,
not just reconstruction.

#### 5.1.4 Total Builder Loss

```
L_builder = L_recon + λ_order × L_order + λ_residual × L_residual + λ_reasoning × L_reasoning
```

### 5.2 ConceptPredictor Loss

The Predictor optimises **two** losses drawn from the same forward pass:
a concept regression loss (Ĉ_k vs frozen groundtruth C_k) and a reasoning
cross-entropy loss (NTP over solution tokens). See
[`losses.py`](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/examples/lcp/losses.py)
`compute_predictor_loss` for the authoritative implementation.

#### 5.2.1 Concept Loss (per-level, averaged over K)

```
L_concept = (1/K) · Σ_{k=0}^{K-1} loss_fn(Ĉ_k, C_k.detach())

loss_fn ∈ { mse, cosine }       # selected by loss.concept_loss_type
  mse    : F.mse_loss(Ĉ_k, C_k)
  cosine : 1 - F.cosine_similarity(Ĉ_k, C_k, dim=-1).mean()
```

Properties:

- **Per-level averaging** prevents fine-grained levels (which have more concepts,
  e.g. L_5 = 32) from dominating coarse levels (L_0 = 1) simply by sample count.
- **Groundtruth is detached** from the Builder graph; the Predictor never
  back-propagates into Builder weights.
- `compute_predictor_concept_loss` also returns a `per_level` dict
  `{level_0_loss: ..., level_5_loss: ...}` for diagnostic logging.

#### 5.2.2 Reasoning Loss (NTP on solution tokens)

```
L_reasoning = F.cross_entropy(
    reasoning_logits.reshape(-1, V),      # (B·T_S, V)
    reasoning_target_ids.reshape(-1),     # shifted S tokens; pad = -100
    ignore_index=-100,
)
```

This is computed on **solution-position logits** extracted via
`gather_solution_logits` from the same packed hidden states `H` used for
`L_concept` — i.e. **one** backbone forward powers both losses. The
logits come from `reason_model.lm_head`, whose parameters are frozen.
Non-solution positions and padding are masked out via `-100` in the target.

`L_reasoning` plays two roles:

1. It validates that the Predictor's generated pyramid, embedded in context,
   still carries the information needed to produce the correct solution.
2. It provides a text-space training signal that is typically less noisy
   than the concept regression signal, stabilising training (see
   [loss-desien-analysis.md](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/examples/lcp/loss-desien-analysis.md) §6.2).

#### 5.2.3 Total Predictor Loss

```
L_predictor = w_concept · L_concept + w_reasoning · L_reasoning
```

Weights come from `training.loss_weights` in the YAML config. Defaults used
by the provided configs are `w_concept = 1.0`, `w_reasoning = 1.0`.

Note that the four Builder losses (recon / ordering / residual / reasoning)
are **not** part of `L_predictor`; the Builder is frozen during Stage 2.

### 5.3 Interaction Between Builder and Predictor

```
Builder (with CoT) ──→ Groundtruth [C_0, ..., C_{K-1}] ──→ Predictor (with Q only)
       ↑                                                    ↓
       └────────────── Training Signal ←────────────────────┘
```

1. **Builder defines "what is good"**: Uses full CoT to extract optimal pyramid
2. **Predictor learns "how to generate"**: Mimics Builder's output from Q alone
3. **End-to-end flow**: Builder's output serves as Predictor's training targets

### 5.4 Optional: Per-Level Weighting

For the Predictor, we can add per-level weights:

```
L_predictor_weighted = Σ_{k=0}^{K-1} w_k × MSE(Ĉ_k, C_k.detach())
```

Weighting strategies:
- **Uniform**: w_k = 1/K (default)
- **Progressive**: w_k increases with k (more weight on fine-grained levels)
- **Adaptive**: Learn w_k based on training dynamics

This is an experimental option for future exploration.

**Loss interaction example**:
```
Scenario 1: λ_order = 0 (no ordering pressure)
  → All concepts attend uniformly to the whole CoT
  → L_recon ≈ 0 (good coverage)
  → But concepts are redundant — C_{5,0} ≈ C_{5,1} ≈ ... ≈ C_{5,31}
  → Decoder cannot distinguish segments → poor NTP quality

Scenario 2: λ_order = ∞ (ordering dominates)
  → Concepts perfectly ordered but may miss information at segment boundaries
  → L_recon > 0 (some information lost at boundaries)
  → But each concept clearly "owns" its segment → good NTP quality

Scenario 3: Balanced λ_order
  → Concepts are mostly ordered with some overlap at boundaries
  → L_recon ≈ 0 (good coverage including boundaries)
  → L_order ≈ 0 (mostly ordered)
  → Best of both worlds: structured AND comprehensive
```

---

## 6. Positional Query Initialization (Builder)

Positional initialization is a training technique for the ConceptPyramidBuilder to accelerate convergence.

### 6.1 Motivation

With random (Xavier uniform) initialization, all concept queries start as random vectors. At the beginning of training:

```
C_{5,0}:  attends ~uniformly to [0, L]   (no positional preference)
C_{5,15}: attends ~uniformly to [0, L]   (no positional preference)
C_{5,31}: attends ~uniformly to [0, L]   (no positional preference)
```

The ordering loss must gradually push these toward:

```
C_{5,0}:  focuses on [0, L/32]           (first segment)
C_{5,15}: focuses on [15L/32, 16L/32]   (middle segment)
C_{5,31}: focuses on [31L/32, L]         (last segment)
```

This is possible but inefficient — the model must discover position structure entirely from loss gradients.

### 6.2 Positional Initialization

When `use_positional_query_init=True`:

```
Q_{k,j} = xavier_uniform(j, D) + α × PE(j / L_k)
```

where PE(p) is sinusoidal positional encoding at normalized position p, and α=0.5 controls the signal strength.

This provides a **starting point** where:
- C_{k,0} is biased toward attending to the **beginning** of the sequence
- C_{k,L_k-1} is biased toward attending to the **end** of the sequence
- Concepts in between are biased toward their corresponding segments

The queries remain fully learnable — training can override the positional prior. But the prior accelerates convergence by providing a reasonable initialization that aligns with the DLCM segment-concept correspondence principle.

### 6.3 Ablation Value

This is an **experimental option** for the Builder, not an architectural requirement. Comparing `use_positional_query_init=True` vs `False` allows us to measure:

1. **Convergence speed**: Does positional init reach good ordering faster?
2. **Final quality**: Does positional init lead to better segment locality at convergence?
3. **Training stability**: Does positional init avoid the diffuse-attention local minimum?

Note: The Predictor may also benefit from level embeddings initialized from the Builder's trained concept_queries.

---

## 7. Relationship to VAR Pipeline

### 7.1 Two-Phase Architecture Mapping

VAR explicitly separates extraction (VQ-VAE) from generation (Transformer). We follow the same principle:

| VAR Component                | Our Equivalent                              | Role                                            |
|------------------------------|---------------------------------------------|-------------------------------------------------|
| **Phase 1: VQ-VAE**          | **ConceptPyramidBuilder**                   | Extract groundtruth from full information (CoT) |
| Encoder                      | `reason_model.embed` + encode MLP           | Encode CoT to hidden states                     |
| Multi-scale quantizer        | Soft attention + residual                   | Extract hierarchical concepts                   |
| Codebook                     | `concept_queries`                           | Learnable "vocabulary" of concept patterns      |
| f_hat / f_rest               | H_hat / H_rest                              | Residual decomposition                          |
| **Phase 2: VAR Transformer** | **ConceptPredictor**                        | Generate autoregressively from condition        |
| Decoder-only Transformer     | `reason_model` (own copy + LoRA)            | Single packed forward over [Q                   |
| Scale embeddings             | `lvl_embed` (per-level identity tag)        | Marks level k for each approx token             |
| Pre-LLM scale extraction     | `level_queries` × `extract_attn` over f_hat | L_k tokens per level via cross-attention        |
| Prediction head              | `concept_head` MLP (D_enc → D)              | Project LLM hidden at approx positions to C_hat |
| VAE Decoder                  | `reason_model.lm_head` reused on solution   | Decode final output tokens from concepts        |

### 7.2 Key Differences

**VAR**: Predicts discrete indices (categorical distribution)
- Uses cross-entropy loss
- Hard codebook bottleneck
- Clear probability modeling

**Ours**: Predicts continuous concepts (regression)
- Uses MSE loss
- No codebook bottleneck
- More expressive but harder to model

**Why continuous?** 
- Builder uses soft attention (naturally continuous)
- Avoids VQ information loss
- End-to-end gradient flow

### 7.3 What We Gain from VAR

1. **Two-phase separation**: Clear distinction between extraction and generation
2. **f_hat + f_rest decomposition**: Mathematically principled coarse-to-fine
3. **Scale-level causality**: Level-by-level generation with parallel intra-level computation
4. **Teacher forcing training**: Groundtruth concepts guide Predictor learning

### 7.4 What We Adapt for Text

1. **Builder uses CoT, Predictor uses Q**: Training-inference asymmetry like VAR's VQ-VAE always seeing full images
2. **Soft attention replaces quantization**: Continuous concept extraction
3. **Learnable queries replace codebook**: Query expansion 1→2→4→8→16→32
4. **Ordering loss replaces spatial structure**: Enforce segment-concept correspondence

---

## 8. Summary of Design Validity

### 8.1 What Is Guaranteed by Construction (Builder)

| Guarantee                 | Mechanism                                       | Strength                           |
|---------------------------|-------------------------------------------------|------------------------------------|
| Coarse-to-fine hierarchy  | Rank bottleneck (L_k concepts) + residual flow  | **Hard** (mathematically provable) |
| Full information coverage | Reconstruction loss ‖back_proj(H_hat) - H_CoT‖² | **Soft** (loss-driven)             |
| Clean residual flow       | Purely residual (no cross-scale conditioning)   | **Hard** (architectural)           |
| Intra-level ordering      | Ordering loss L_order                           | **Soft** (loss-driven)             |

### 8.2 What Is Guaranteed by Construction (Predictor)

| Guarantee                                              | Mechanism                                                                                                                 | Strength                  |
|--------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|---------------------------|
| Scale-causal visibility (level k sees only levels j≤k) | Per-row 4D additive mask in `_build_scale_causal_mask_packed` (Q-causal, intra-level bidir, cross-level causal, S-causal) | **Hard** (architectural)  |
| Intra-level concept identity (Ĉ_{k,0} ≠ Ĉ_{k,1})       | `level_queries[k]` are L_k distinct learnable vectors; queries cross-attend independently over f_hat_k                    | **Hard** (architectural)  |
| Per-level identity tag                                 | `lvl_embed[k]` added to every approx token of level k after the post-norm residual                                        | **Hard** (architectural)  |
| VAR-faithful conditioning                              | LLM only ever sees `back_proj(f_hat_k)` (cumulative canvas), never raw concept stacks                                     | **Hard** (architectural)  |
| Teacher forcing alignment                              | `gt_f_hats` and `gt_concepts` taken from frozen Builder, detached                                                         | **Hard** (training setup) |
| Single-pass training                                   | One packed `[real_Q                                                                                                       | approx_0..K-1             |
| Inference fixed-canvas consistency                     | f_hat lives on `[B, L_canvas, D]` canvas; reconstruction `R_k = α_k^T @ Ĉ_k` updates the same canvas                      | **Hard** (architectural)  |

### 8.3 What Is Encouraged but Not Guaranteed

| Property                      | Mechanism                           | Strength                      |
|-------------------------------|-------------------------------------|-------------------------------|
| Segment locality (Builder)    | Ordering loss + softmax competition | **Soft** (inductive bias)     |
| Balanced extraction           | Rank bottleneck + recon loss        | **Soft** (indirect)           |
| Predictor matches Builder     | MSE loss + sufficient capacity      | **Soft** (training dependent) |
| Q-only → CoT-quality concepts | End-to-end training                 | **Soft** (emergent)           |

### 8.4 Open Questions for Experimental Validation

1. **Builder quality**: Does the Builder extract meaningful hierarchical structure? Visualize attention maps A_{k,j} and reconstructions.

2. **Predictor fidelity**: Does the Predictor accurately mimic the Builder? Compare Ĉ_k vs C_k across levels.

3. **Inference quality**: Do predicted concepts enable accurate Solution generation? Evaluate end-to-end accuracy.

4. **Ablation studies**:
   - Positional query initialization: Does it help convergence?
   - Per-level weighting: Does progressive weighting improve fine-grained prediction?
   - Solution loss in Builder: Does it improve downstream performance?

5. **Scalability**: How does performance vary with concept dimension D, number of levels, or query expansion pattern?

---

## 9. Conclusion

The Concept Pyramid design is architecturally sound. The ConceptPyramidBuilder uses soft attention (soft boundaries) with learnable query expansion to extract hierarchical concepts from CoT via purely residual decomposition — no cross-scale conditioning, following VAR's VQ-VAE Stage 1 principle. The ConceptPredictor learns to autoregressively generate these concepts from `Q` alone by reusing a causal decoder-only LLM (`reason_model`) as its backbone: concepts are back-projected into the embedding space, tagged with `level_embeddings` + `position_embeddings`, packed into `[Q; C; S]`, and consumed by the LLM's native causal attention. A lightweight `concept_head` MLP reads Ĉ_k out of the backbone hidden states; the same forward pass also produces solution logits for the reasoning CE loss. This yields VAR's two-phase separation and level-by-level causality without introducing a second Transformer. The rank bottleneck in the Builder provides a hard guarantee of coarse-to-fine hierarchy. The combination of softmax competition, residual flow, and ordering loss creates sufficient inductive bias for DLCM-style segment-concept correspondence without requiring hard segmentation.

The main limitations — soft segment locality, potential extraction imbalance, and Q-only generalization — are inherent trade-offs of the soft attention approach. They are acceptable for our research goals because:
1. The soft approach is strictly more expressive than hard segmentation
2. The full LCP training pipeline (with NTP loss) provides strong corrective signals
3. The design is fully differentiable and end-to-end trainable

These limitations should be monitored during experiments but do not warrant architectural changes at this stage.
