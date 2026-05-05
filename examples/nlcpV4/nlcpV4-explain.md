# Concept Pyramid Architecture: From CoT to Hierarchical Concepts

## 1. Notation and Concepts

### 1.1 Indexing Convention

We use a two-level subscript **C_{k,j}** to unambiguously distinguish inter-level from intra-level concepts:

| Symbol      | Meaning                                                       | Example                            |
|-------------|---------------------------------------------------------------|------------------------------------|
| **K**       | Total number of levels                                        | K=6 (levels 0 to 5)                |
| **k**       | Level index (inter-level)                                     | k ∈ {0, 1, ..., K-1}               |
| **j**       | Concept slot within level k (intra-level)                     | j ∈ {0, 1, ..., L_k-1}             |
| **C_{k,j}** | The j-th concept at level k                                   | C_{5,17} = 18th concept at level 5 |
| **L_k**     | Number of concept slots at level k                            | L_k = 2^k for k < K                |
| **C_k**     | All concepts at level k: [C_{k,0}, C_{k,1}, ..., C_{k,L_k-1}] | C_5 has shape [B, 32, D]           |

Level configuration (K=6): L_0=1, L_1=2, L_2=4, L_3=8, L_4=16, L_5=32 (total: 63 concepts)

### 1.1.1 Notation Convention

Throughout this document, we use **our NLCP notation** C_{k,j} consistently, even
when describing other methods. When referencing DLCM's single-level concepts
(written as c_k in the DLCM paper), we write them as C_{k,j} and add a note
explaining the mapping. This is because:

- DLCM's c_1, c_2, c_3, ... correspond to our C_{k,0}, C_{k,1}, C_{k,2}, ...
  at any given level k
- DLCM has no inter-level dimension — it only partitions the CoT at one
  granularity, so its concept index maps directly to our intra-level index j
- Our C_{k,j} **subsumes** DLCM's c_j by adding the level dimension k

### 1.2 Key Variables (following VAR.md Section 5.2.2)

| Variable    | VAR Image Domain                | Our Text Domain                          | Physical Meaning              |
|-------------|---------------------------------|------------------------------------------|-------------------------------|
| **H_proj**  | z = Encoder(image)              | H_proj = Linear(Encoder(CoT))            | CoT information to decompose  |
| **H_rest**  | f_rest = "still needs encoding" | H_rest_k = H_proj - Σ_{i<k} R_i          | Residual at level k           |
| **H_hat**   | f_hat = "already encoded"       | H_hat_k = Σ_{i<k} R_i                    | Accumulated reconstruction    |
| **A_{k,j}** | (implicit in VQ)                | A_{k,j} = softmax(Q_{k,j} @ H_rest_k^T)  | Attention weights for C_{k,j} |
| **C_{k,j}** | h_k = codebook[idx_k]           | C_{k,j} = level_proj(A_{k,j} @ H_rest_k) | Concept (purely residual)     |
| **R_k**     | f_hat += h_k_up                 | R_k = A_k^T @ C_k                        | Reconstruction from level k   |

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

### 1.4 Overall Architecture: From CoT to Concept Pyramid to Solution

This section provides a high-level overview of how the hybrid design achieves the research goal: **compressing CoT into a hierarchical concept pyramid for efficient reasoning**.

#### 1.4.1 The Two-Stage Pipeline

NLCP V4 is organised as **two sequential training stages** and a single
autoregressive inference path. The two stages share a common notion of a
"concept pyramid" C = [C_0, C_1, ..., C_{K-1}], but train disjoint modules
with disjoint objectives.

**Bird's-eye view of the whole pipeline**

```
═══════════════════════════════════════════════════════════════════════════════
                    STAGE 1 — ConceptPyramidBuilder (TRAIN)
═══════════════════════════════════════════════════════════════════════════════

   Input: (Q, CoT, S)                                   [CoT is visible here]
                 │
                 ▼
    ┌────────────────────────────────────────────────────────┐
    │ frozen reason_model.embed(CoT)  →  H_CoT   [B, L, D_e] │
    │ encode: LayerNorm(Linear) → H_proj         [B, L, D]   │
    └────────────────────────────────────────────────────────┘
                 │
                 ▼                       ┌─────────────── residual ledger ──────────────┐
    ┌─────────────────────────────┐      │ H_hat_0 = 0        H_rest_0 = H_proj          │
    │ K = 6 levels, L_k = 1..32   │      │ H_hat_{k+1} = H_hat_k + R_k                    │
    │ for k in 0..K-1:            │◄─────┤ H_rest_{k+1}= H_rest_k − R_k                   │
    │   A_k = softmax(Q_k·H_restᵀ) │      │ R_k = A_kᵀ · C_k          (rank ≤ L_k)         │
    │   C_k = level_proj(A_k·H_rest)│     └────────────────────────────────────────────────┘
    └─────────────────────────────┘
                 │ produces pyramid: [C_0, C_1, ..., C_{K-1}]  ← groundtruth for Stage 2
                 │
     ┌───────────┴───────────┬────────────────────┬─────────────────────────┐
     ▼                       ▼                    ▼                         ▼
  L_recon               L_ordering            L_residual              L_reasoning
  ‖back_proj(H_hat_K)   exp_pos[C_{k,j}]      ‖H_rest_K‖₁         CE on S via
    − H_CoT‖²            < exp_pos[C_{k,j+1}]                     frozen reason_model
                                                                   fed [Q; C; S]

═══════════════════════════════════════════════════════════════════════════════
                    STAGE 2 — ConceptPredictor (TRAIN)
═══════════════════════════════════════════════════════════════════════════════

          (Builder frozen) ──► groundtruth pyramid C_gt = [C_0, ..., C_{K-1}]
                                          │ detach()
          Input: (Q, C_gt, S)   [Q and C_gt are teacher-forced; no CoT used]
                 │                        │                       │
                 ▼                        ▼                       ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  back_proj(C_gt)  +  level_emb  +  position_emb        (slot markers)│
    │                                                                     │
    │  pack_qcs_sequences → one contiguous row per sample:                │
    │                                                                     │
    │  ┌──────────────┬───────────────────────────────────┬────────────┐  │
    │  │   Q tokens   │ C_0 C_1,C_1 C_2,C_2,C_2,C_2 ... │  S tokens  │  │
    │  │  (real Q_len)│         Σ L_k slots              │ (solution) │  │
    │  └──────────────┴───────────────────────────────────┴────────────┘  │
    │                                                                     │
    │  reason_model (causal LM, SHARED or INDEPENDENT) — one pass         │
    │                                                                     │
    │     hidden states H [B, T, D_enc]                                   │
    │              ├── gather concept positions → concept_head → Ĉ_k      │
    │              └── gather solution positions → lm_head → logits_S     │
    └─────────────────────────────────────────────────────────────────────┘
                 │                                       │
                 ▼                                       ▼
          L_concept  = (1/K) Σ_k MSE(Ĉ_k, C_k)   L_reasoning = CE(logits_S, S)

═══════════════════════════════════════════════════════════════════════════════
                    INFERENCE — AR generation from Q only
═══════════════════════════════════════════════════════════════════════════════

                          Input: Q      (no CoT, no S)
                                │
                                ▼
    ┌────── Step 0: prime KV cache ──────────────────────────────┐
    │ h = reason_model(embed(Q))                                 │
    │ Ĉ_0 = concept_head(h[last_real_Q])                          │
    │ cache = past_kv                                             │
    └────────────────────────────────────────────────────────────┘
                                │
           ┌────────────────────┴──── loop t = 1..Σ L_k − 1 ────────────┐
           ▼                                                            │
    ┌────── Step t: one concept per step ─────────────────────────┐     │
    │ x = back_proj(Ĉ_{t-1}) + level_emb[t-1] + position_emb[t-1] │     │
    │ h = reason_model(x, past_kv=cache, position_ids=q_len+t-1)  │─────┘
    │ Ĉ_t = concept_head(h[-1]);   cache ← updated KV             │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
                 Output: [Ĉ_0, Ĉ_1, ..., Ĉ_{K-1}]
              (downstream decoding of S not part of this document)
```

The three boxes above correspond to the three operating modes of the codebase:
`train_builder.py` (top), `train_predictor.py` (middle), and
`predictor._forward_inference` (bottom). The arrows that cross stage
boundaries are the **only** places where gradients do NOT flow:
`C_gt` is `detach()`ed and the Builder is frozen during Stage 2.

**Stage 1 — ConceptPyramidBuilder** (`examples/nlcpV4/concept_builder.py`)

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

**Stage 2 — ConceptPredictor** (`examples/nlcpV4/concept_predictor.py`)

```
Input : (Q, C_gt = [C_0, ..., C_{K-1}], S)  # C_gt comes from frozen Builder
Forward (teacher-forced, ONE backbone pass):
    # 1. back-decode groundtruth concepts to encoder space
    X_cat   = back_proj(concat(C_0, ..., C_{K-1}))           # (B, Σ L_k, D_enc)
    # 2. add slot markers
    X_cat  += level_embeddings(_level_ids_flat)
    X_cat  += position_embeddings(_pos_ids_flat)
    # 3. pack [Q_b; X_cat_b; S_b] row-by-row (no internal padding)
    packed, masks, positions = pack_qcs_sequences(...)
    # 4. run the (shared or independent) backbone ONCE
    H = reason_model(inputs_embeds=packed,
                     attention_mask=masks.attention_mask,
                     output_hidden_states=True).hidden_states[-1]
    # 5. two readouts from the SAME hidden states
    Ĉ_k  = concept_head(gather_concept_readout(H, masks, k))  # for k=0..K-1
    logits_S = lm_head(gather_solution_logits(H, masks))
Outputs : PredictorOutput {
    predicted_concepts = [Ĉ_0, ..., Ĉ_{K-1}],   # continuous regression targets
    gt_concepts        = [C_0, ..., C_{K-1}],
    reasoning_logits   = logits_S,
    reasoning_target_ids                         # S shifted by 1, pad = -100
}
Loss    : L_predictor = w_concept·L_concept + w_reason·L_reasoning
```

`concept_head` is a small `Linear(D_enc) → GELU → Linear(D)` MLP; the
backbone can be **SHARED** with the Builder (aliased `reason_model` /
`back_proj`, LoRA forbidden) or **INDEPENDENT** (own copy with optional LoRA).

**Inference** (autoregressive, Q-only)

```
Input : Q
Step 0 : run reason_model on embed(Q) → hidden h_last → Ĉ_0
         cache KV, remember q_len
Step t :                          # for t = 1 .. Σ L_k − 1
    x = back_proj(Ĉ_{t-1}) + level_emb[lvl(t-1)] + pos_emb[pos(t-1)]
    h = reason_model(inputs_embeds=x,
                     past_key_values=cache,
                     position_ids=[q_len + t − 1])          # explicit RoPE id
    Ĉ_t = concept_head(h[-1])
    cache = updated KV
Output : [Ĉ_0, Ĉ_1, ..., Ĉ_{K-1}]    # no explicit CoT is ever generated
```

End-to-end flow (what actually changes between stages):

| Stage                   | Trainable params                                          | Backbone         | Sees CoT?   | Loss                                         |
|-------------------------|-----------------------------------------------------------|------------------|-------------|----------------------------------------------|
| Builder                 | encode / queries / level_proj / back_proj (+ LoRA)        | frozen (+LoRA)   | Yes         | L_recon + L_order + L_residual + L_reasoning |
| Predictor — SHARED      | level_emb, position_emb, concept_head                     | aliased, frozen  | No (Q only) | L_concept + L_reasoning                      |
| Predictor — INDEPENDENT | back_proj, level_emb, position_emb, concept_head (+ LoRA) | own copy (+LoRA) | No (Q only) | L_concept + L_reasoning                      |

#### 1.4.2 Key Design Principles

**1. Builder-Predictor Separation**
- **Builder**: Uses soft attention + residual flow to extract groundtruth from CoT
- **Predictor**: Uses decoder-only Transformer to autoregressively generate concepts
- **Rationale**: Builder defines "what is a good pyramid", Predictor learns "how to generate it"

**2. Preserved Core Mechanisms**
All mechanisms from Section 1.3 are retained:
- **Query expansion**: 1→2→4→8→16→32 learnable queries per level
- **Soft attention (soft boundaries)**: Competition-based segment-concept correspondence
- **Residual reconstruction**: Coarse-to-fine information decomposition
- **Intra-level ordering**: Concepts ordered by CoT position
- **Purely residual**: No cross-scale conditioning in the builder (VAR.md principle)

**3. Training-Inference Alignment**
- Training: Predictor sees groundtruth concepts (teacher forcing)
- Inference: Predictor generates concepts step-by-step
- Both use same causal structure: level k depends on levels < k

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

| VAR Scale | Tokens | Information Capacity  | NLCP Level | Concepts | Information Capacity  |
|-----------|--------|-----------------------|------------|----------|-----------------------|
| 1×1       | 1      | Global color/tone     | Level 0    | 1        | Global CoT structure  |
| 2×2       | 4      | Coarse spatial layout | Level 1    | 2        | Two major segments    |
| 4×4       | 16     | Medium structure      | Level 2    | 4        | Four sub-segments     |
| ...       | ...    | ...                   | ...        | ...      | ...                   |
| 32×32     | 1024   | Fine details          | Level 5    | 32       | Fine-grained segments |

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

**Same CoT, different segmentations** (NLCP):
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
3. End-to-end NTP loss from the full NLCP pipeline (strongest signal)

**Current assessment**: The greedy extraction concern is theoretically valid but likely manageable in practice. The rank bottleneck provides a hard constraint, and the full NLCP training pipeline with NTP loss will provide the strongest corrective signal.

---

## 2.5 Deep Dive: The Rank-Constrained Residual Decomposition Principle

This section synthesizes §2.1–§2.4 and the VAR comparison of §7 into a single, mechanistic statement of what the Builder actually does. It is the most important section of this document — every downstream design choice (Predictor teacher forcing, loss weights, level schedule) flows from here. It is the nlcpV4 counterpart of `docs/VAR.md §5.3.2.1` (which established the dual fact for VAR: *codebook entries are residuals*).

### 2.5.0 Relationship to VAR.md §6 — No Contradiction, Two Layers of Description

Readers coming from [docs/VAR.md](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/docs/VAR.md) §6 — which declared that nlcpV4's Builder "follows VAR's residual philosophy" and that `C_k` "expresses the semantic remainder scales 0..k-1 cannot cover" — may wonder whether §2.5's emphasis on a *rank-bounded softmax bottleneck* (contrasted with VAR's *discrete codebook bottleneck*) contradicts that claim, **or** whether §2.5's phrase "`C_k` is the best low-rank expression of the residual" is a third, different statement. **Neither is a contradiction.** The three statements operate at three different layers of abstraction and are mutually consistent. This subsection makes the layering explicit.

#### Two layers of architectural description

| Layer           | What it describes                                            | Same in VAR and nlcpV4? | Discussed in                                    |
|-----------------|--------------------------------------------------------------|-------------------------|-------------------------------------------------|
| **Outer loop**  | The `H_rest / H_hat` residual-accumulation skeleton          | ✅ **YES — identical**   | VAR.md §6; nlcpV4-explain.md §2.5.5             |
| **Inner joint** | How each level produces its per-level output from `H_rest_k` | ❌ **NO — different**    | VAR.md §5.3.2.1; nlcpV4-explain.md §2.5.2–2.5.6 |

```
┌─── OUTER LOOP (shared by VAR and nlcpV4) ─────────────────────────────┐
│  for k in 0..K-1:                                                      │
│      level-k output  ←──── [INNER JOINT: differs] ────  H_rest_k       │
│      R_k             ←  smear level-k output to sequence length        │
│      H_hat_{k+1}     =  H_hat_k  + R_k      (canvas grows)             │
│      H_rest_{k+1}    =  H_rest_k - R_k      (residual shrinks)         │
│                                                                         │
│    ┌── INNER JOINT (differs) ────────────────────────────────┐         │
│    │  VAR:     level-k output  =  embedding(argmin_V ‖·‖)     │         │
│    │           (discrete codebook lookup, V hard options)     │         │
│    │  nlcpV4:  level-k output  =  level_proj(A_k @ H_rest_k)  │         │
│    │           (rank-L_k soft summary, softmax weights)       │         │
│    └───────────────────────────────────────────────────────────┘        │
└────────────────────────────────────────────────────────────────────────┘
```

**VAR.md §6** is a statement about the **outer loop** — it's why the Predictor must replay the cumulative canvas `H_hat_k` (identical requirement in both systems).  
**nlcpV4-explain.md §2.5** is a zoom-in on the **inner joint** — it explains that we swap discrete-argmin for rank-bounded-softmax while leaving the outer loop untouched.

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

| Claim                                                              | Layer        | Tensor level | Relationship to other claims           |
|--------------------------------------------------------------------|--------------|--------------|----------------------------------------|
| "Predictor must replay cumulative `H_hat_k`" (VAR.md §6)           | Outer loop   | `H_hat`      | Shared by VAR and nlcpV4               |
| "VAR uses discrete codebook, nlcpV4 uses rank bottleneck" (§2.5.6) | Inner joint  | per-level op | The only structural difference         |
| "`C_k` is residual in nature" (VAR.md §6)                          | Semantic     | `C_k`        | Equivalent to §2.5.3 at semantic zoom  |
| "`C_k` is best rank-`L_k` summary of `H_rest_k`" (§2.5.3)          | Mathematical | `C_k`        | The precise form of the semantic claim |
| "`R_k` is subtracted from `H_rest_k`" (both docs)                  | Operational  | `R_k`        | The canvas-debit step; shared in both  |

All five statements are simultaneously true. They describe different faces of the same architecture.

---

### 2.5.1 The Core Sentence (核心一句话)

> **At each level, the Builder takes the current residual `H_rest_k`, uses `L_k` learnable queries to construct a rank-`L_k`-bounded best low-rank summary `C_k`, smears it back to sequence length as `R_k`, adds `R_k` onto the canvas `H_hat` and subtracts it from the residual, then hands whatever remains to the next level whose `2×`-wider query bank fishes again.**
>
> **我们每一层都基于当前残差 `H_rest_k`，用 `L_k` 条可学习查询构造一个秩受 `L_k` 约束的最佳低秩摘要 `C_k`，然后把它 smear 回序列长度得到 `R_k`，加入画布、从残差里扣掉，留下的信息交给下一层用 2 倍宽的查询再捞一次。**

Every clause in this sentence corresponds to an architectural commitment that can be read directly off the code in [concept_builder.py](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/examples/nlcpV4/concept_builder.py). The rest of §2.5 unpacks it.

### 2.5.2 The Rank Inequality as nlcpV4's "Invisible Codebook"

VAR bottlenecks information flow with a **discrete codebook** (hard argmin lookup against V learned centroids). nlcpV4 has no codebook — so what prevents the model from cheating and dumping all information into a single level? Answer: **a linear-algebraic rank constraint** just as unforgiving as a codebook, only expressed in the language of matrix factorization rather than nearest-neighbor search.

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

**Why this equals "a codebook of invisible size"**: VAR's codebook has `V` entries of dimension `Cvae`; `embedding(idx_k)` at each spatial position is one of at most `V` possible vectors. nlcpV4's level-k output lives in a continuous rank-`L_k` subspace of `R^{L×D}`; `R_k` is one of infinitely many tensors in this subspace. Both are information-capacity ceilings, merely expressed in different bases:

| Bottleneck shape | VAR                                 | nlcpV4                            |
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

- Each of `L` sequence positions receives a convex-like combination of the `L_k` concepts, weighted by how much that position attended to each concept slot.
- If position `t` was claimed primarily by `C_{k,j}`, then `R_k[t] ≈ C_{k,j}`.
- If position `t` is on the boundary between two concepts, `R_k[t]` is a soft blend.

The composition `A_k^T @ A_k ∈ R^{L×L}` is a **rank-`L_k` soft-clustering smoother**: it replaces each position's feature with a soft-cluster-mean of its neighbors. Analogous operations across fields:

| Field          | Compression step       | Smear-back step              | Rank bound             |
|----------------|------------------------|------------------------------|------------------------|
| PCA            | project to top-k axes  | reconstruct via `V_k V_k^T`  | rank ≤ k               |
| K-means        | assign to centroid     | broadcast centroid to points | rank ≤ K               |
| nlcpV4 Builder | `A_k @ H_rest_k`       | `A_k^T @ C_k`                | rank ≤ L_k             |
| VAR VQ-VAE     | `argmin` over codebook | `embedding(idx_k)` + upscale | ≤ V discrete centroids |

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

| Aspect                    | VAR Stage-1 (VQ-VAE)                              | nlcpV4 Builder                                   |
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

**Key insight**: VAR's and nlcpV4's bottlenecks are **duals of each other in information-capacity space** — different shapes of the same constraint. VAR trades differentiability for a crisp discrete vocabulary; nlcpV4 trades the discrete vocabulary for end-to-end differentiable training. Neither is strictly more powerful; they are two fixed points on a bottleneck-shape axis:

```
         hard discrete            soft continuous
         ┌────────────┐          ┌────────────────┐
         │ VAR VQ-VAE │ ◄──────► │ nlcpV4 Builder │
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

The §2.5 principle has a direct, non-negotiable consequence for ConceptPredictor teacher-forcing. This is the nlcpV4 analog of the warning in [docs/VAR.md §5.3.2.1](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/docs/VAR.md) about using `embedding(idx_k)` naively.

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

**Actionable check**: `concept_predictor.py`'s level-conditioning path (e.g., `_upsample_prev_to_level` or analogous) must either reconstruct `H_hat_k` explicitly or provide positional/level embeddings rich enough that the Transformer can reconstruct it in-attention. This is the single most important Predictor correctness property inherited from §2.5.

### 2.5.10 One-Line Mnemonic (For Everyday Use)

> **VAR constrains via a discrete codebook; nlcpV4 constrains via matrix rank. Both iteratively peel a residual, both enforce non-overlap through subtraction, both produce a coarse-to-fine pyramid. The only real difference is which algebraic structure (finite set vs. rank-bounded subspace) plays the role of "information capacity ceiling" at each level.**

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

For a fixed position t, softmax enforces: Σ_j A_{k,j}(t) = 1. This means concept slots **compete** for each position. If C_{5,0} strongly attends to position [0, L/32], then A_{5,0}(t) is large for t ∈ [0, L/32], forcing A_{5,1}(t), ..., A_{5,31}(t) to be small for those positions. This pushes later concept slots toward later positions.

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

The ordering loss ensures each concept slot "owns" a contiguous, ordered segment
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

1. **Competition forces focus**: In level 5 with 32 concept slots, if C_{5,0} and C_{5,1} both attend diffusely to [0, L/2], they would produce nearly identical concepts. The NTP loss (from the decoder) would penalize redundancy — if two concepts carry the same information, one is wasted. The model is incentivized to differentiate concepts by attending to different positions.

2. **Residual flow prevents overlap**: Even without ordering loss, the residual flow naturally creates soft boundaries. If C_{5,0} extracts information from positions [0, L/32], that information is subtracted from H_rest for subsequent concepts.

3. **Ordering loss provides explicit pressure**: The ordering loss directly pushes concept slots toward sequential, non-overlapping attention patterns.

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

The Builder constructs the groundtruth concept pyramid from CoT using soft attention and residual reconstruction.

**Input**: (Q, CoT, Solution)
- **CoT**: Core source for building the concept pyramid
- **Q**: Context/prior (conditions the extraction but doesn't enter pyramid)
- **Solution**: Used for auxiliary loss (validating pyramid's reasoning capability)

**Mechanism** (purely residual — no cross-scale conditioning):
```
H_CoT = Encoder(CoT)                      # Encode CoT to hidden states
H_proj = LayerNorm(Linear(H_CoT))         # Project to concept space
H_rest_0 = H_proj

for k in range(K):  # K levels
    # Soft boundaries via learnable queries
    A_k = softmax(Q_k @ H_rest_k^T / (sqrt(D) × τ))     # [B, L_k, L]
    C_k = level_proj(A_k @ H_rest_k)                     # [B, L_k, D]
    
    # Residual update (VAR f_hat/f_rest)
    R_k = A_k^T @ C_k                                    # [B, L, D]
    H_hat_{k+1} = H_hat_k + R_k
    H_rest_{k+1} = H_rest_k - R_k

# Back-project to encoder space for reconstruction loss
H_recon = back_proj(H_hat_K)              # [B, L, D_encoder]
L_recon = ||H_recon - H_CoT||²
```

**Output**: Groundtruth concept pyramid [C_0, C_1, ..., C_{K-1}]

**Loss** (Stage 1 dual objectives):
```
L_builder = L_recon + λ_order × L_ordering + λ_residual × L_residual + λ_reasoning × L_reasoning

- L_recon: ||back_proj(H_hat_K) - H_CoT||²  (reconstruction in encoder space)
- L_ordering: Intra-level concept ordering (Section 3.2)
- L_residual: L1 norm of final residual ||f_rest_K||  (concept-space regularization)
- L_reasoning: NTP cross-entropy — teacher-forced [Q, concepts, S] → predict Solution tokens
    (ensures pyramid is useful for reasoning, not just reconstruction)
```

**Key Properties**:
- Builder is only used during training to generate groundtruth
- All mechanisms from Sections 2-3 are employed (soft attention, residual flow, query expansion)
- The output serves as training targets for the Predictor

### 4.2 ConceptPredictor (Phase 2: Generation)

The Predictor learns to autoregressively generate the concept pyramid
from `Q` alone, mimicking the Builder's output. It **reuses a causal
decoder-only LLM (`reason_model`) as its backbone** rather than introducing
a separate Transformer; the entire concept pyramid is materialised inside
the same sequence that the LLM already natively consumes.

#### 4.2.1 Components (`ConceptPredictor.__init__`)

| Component             | Shape / Type                                     | Role                                                                                     |
|-----------------------|--------------------------------------------------|------------------------------------------------------------------------------------------|
| `reason_model`        | HuggingFace causal LM                            | Shared backbone; processes `[Q; back-decoded concepts; S]` as ordinary `inputs_embeds`.  |
| `back_proj`           | `Linear(D, D_enc, bias=False)`                   | Maps concept-space D to encoder/embedding-space D_enc so concepts can be fed to the LLM. |
| `level_embeddings`    | `nn.Embedding(K, D_enc)`                         | Per-level marker k added to each concept slot; analogous to VAR's `lvl_emb`.             |
| `position_embeddings` | `nn.Embedding(max_k L_k, D_enc)`                 | Within-level slot marker j; lets the backbone distinguish `C_{k,0}` from `C_{k,1}`.      |
| `concept_head`        | `Linear(D_enc, D_enc) → GELU → Linear(D_enc, D)` | Maps backbone hidden state back to concept space to produce Ĉ.                           |
| `_level_ids_flat`     | buffer `int64 [Σ L_k]`                           | Precomputed flat level ids `[0, 1,1, 2,2,2,2, ...]` for slot marker lookup.              |
| `_pos_ids_flat`       | buffer `int64 [Σ L_k]`                           | Precomputed flat within-level ids `[0, 0,1, 0,1,2,3, ...]`.                              |

There is **no** `q_proj`, no `q_proj_norm`, no separate `concept_transformer`,
and no `start_token`; the question is injected simply by running its token
embeddings through `reason_model` as the first part of the packed sequence.

#### 4.2.2 Backbone modes: SHARED vs INDEPENDENT (`use_shared_model`)

The Predictor supports two mutually exclusive backbone configurations,
selected via `model.predictor.use_shared_model` in the YAML config:

| Aspect                  | SHARED (`use_shared_model: true`)                                     | INDEPENDENT (`use_shared_model: false`)                                               |
|-------------------------|-----------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| `reason_model`          | **Aliased** to `builder.reason_model` (no new weights)                | **Own copy**, loaded from `predictor_model_name`                                      |
| `back_proj`             | **Aliased** to `builder.back_proj`                                    | **Own copy** (fresh, trainable)                                                       |
| `tokenizer`             | **Aliased** to `builder.tokenizer`                                    | **Own instance** (from `predictor_model_name`)                                        |
| LoRA on `reason_model`  | **Forbidden** (fail-fast: raises at init)                             | **Allowed** (default target_modules `[q_proj, v_proj]`, r=16, α=32)                   |
| Extra trainable weights | `level_embeddings`, `position_embeddings`, `concept_head` only        | `back_proj`, `level_embeddings`, `position_embeddings`, `concept_head`, LoRA adapters |
| Config example          | `configs/nlcpV4/GSM8K/train_predictor_Qwen2.5-0.5B_2level_shared.yml` | `configs/nlcpV4/GSM8K/train_predictor_Qwen2.5-0.5B_2level_independent.yml`            |

**Design intent.** SHARED mode tests whether the *same* LLM that built the
pyramid can also predict it, using only a small MLP head on top. INDEPENDENT
mode decouples Builder and Predictor capacities: the Predictor may use a
larger backbone and/or LoRA to compensate for the absence of CoT.

**Visual: what is shared vs what is owned**

```
           SHARED MODE (use_shared_model=True)                INDEPENDENT MODE (use_shared_model=False)
  ┌───────────────────────────────────────┐       ┌─────────────────────────────────────┐
  │          ConceptPyramidBuilder         │       │          ConceptPyramidBuilder         │
  │  reason_model (frozen) ●────────────┐ │       │  reason_model (frozen)                │
  │  back_proj            ●──────────┐│ │       │  back_proj                            │
  │  concept_queries, level_proj, ...      ││ │       │  concept_queries, level_proj, ...      │
  └────────────────────────────────────────┘│ │       └─────────────────────────────────────┘
                            alias◄──────┘│ │
                            alias◄────────┘ │       (no alias; Predictor owns its copy)
                                                 │
  ┌────────────────────────────────────────┘       ┌─────────────────────────────────────┐
  │         ConceptPredictor               │       │         ConceptPredictor               │
  │  reason_model   ◁── aliased (same obj)   │       │  reason_model    (NEW instance)        │
  │  back_proj      ◁── aliased               │       │  back_proj       (NEW instance)        │
  │  tokenizer      ◁── aliased               │       │  tokenizer       (NEW instance)        │
  │                                         │       │                                         │
  │  level_embeddings    ◆ trainable         │       │  level_embeddings    ◆ trainable         │
  │  position_embeddings ◆ trainable         │       │  position_embeddings ◆ trainable         │
  │  concept_head        ◆ trainable         │       │  concept_head        ◆ trainable         │
  │                                         │       │  back_proj           ◆ trainable         │
  │  LoRA                ✖ FORBIDDEN         │       │  LoRA on q_proj/v_proj ◆ optional adapters│
  └────────────────────────────────────────┘       └─────────────────────────────────────┘

                   ●  shared object (Python alias, literally the same tensor)
                   ◆  owned & trainable by Predictor
                   ✖  construction fails fast if configured
```

#### 4.2.3 Training forward pass (`_forward_training`, unified single pass)

Given a batch of `(Q, C_gt = [C_0, ..., C_{K-1}], S)`, the Predictor does
**one** teacher-forced pass through `reason_model`:

```
# 1. Back-decode groundtruth concepts to encoder space and add slot markers
C_flat   = concat_along_slots([C_0, ..., C_{K-1}])          # (B, Σ L_k, D)
X_concept= back_proj(C_flat)                                 # (B, Σ L_k, D_enc)
X_concept+= level_embeddings(_level_ids_flat)                # broadcast over batch
X_concept+= position_embeddings(_pos_ids_flat)

# 2. Per-row packing: remove any internal padding between Q, C, S
packed, masks, positions = pack_qcs_sequences(
    q_embeds=embed_tokens(Q),        q_mask,
    c_embeds=X_concept,              c_mask,
    s_embeds=embed_tokens(S),        s_mask,
)
# masks carries per-row slot indices so we can gather later

# 3. ONE pass through the (shared or independent) backbone
out = reason_model(
    inputs_embeds=packed,
    attention_mask=masks.attention_mask,
    position_ids=positions,
    output_hidden_states=True,
)
H = out.hidden_states[-1]                                     # (B, T, D_enc)

# 4. Two independent readouts from the SAME H
#    — concept readout: positions that correspond to concept slots
H_concepts = gather_concept_readout(H, masks)                 # (B, Σ L_k, D_enc)
predicted  = concept_head(H_concepts)                         # (B, Σ L_k, D)
[Ĉ_0, ..., Ĉ_{K-1}] = split_levels(predicted, level_lengths)

#    — reasoning readout: positions that correspond to solution tokens
reasoning_logits = gather_solution_logits(H, masks, lm_head=reason_model.lm_head)
reasoning_target_ids = build_solution_targets(S_ids, s_mask, pad_id=-100)
```

Key properties of this forward path:

**Visual: the packed sequence and its two readouts**

For a single row `b` in the batch (K=3, L = [1, 2, 4] for illustration):

```
 position : 0 1 2 ... q_len-1 | q_len      ...  q_len+6 | q_len+7 ... q_len+7+L_S-1
 slot kind: Q Q Q ...    Q    | C_{0,0} C_{1,0} C_{1,1} C_{2,0} C_{2,1} C_{2,2} C_{2,3} | S S ...
 level k  : · · · ...    ·    |   0       1       1       2       2       2       2    | · · ...
 pos   j  : · · · ...    ·    |   0       0       1       0       1       2       3    | · · ...
                                    │         │       │
                                    │         │       └─ position_embeddings(j=1)
                                    │         └───── level_embeddings(k=1)
                                    └───────── back_proj(C_{0,0}) + …

                                          CAUSAL MASK (from the backbone)
                                   every position sees only itself + all earlier

 READOUT A (concept head)                     READOUT B (lm_head)
  positions ↑ predict the NEXT concept         positions ↑ predict NEXT solution token
  (take q_len−1 → predicts C_{0,0}; then        (take last_C → predicts S_0; then
   each concept’s hidden predicts the next)     each S_t → predicts S_{t+1})

                           ↓ both readouts come from the SAME
                             hidden states H produced by ONE pass
```

Under the causal mask the dependency structure is:

```
       Q ────────────────────────────────────────────────────┐
                                                                       ▼
       Q,C_{0,0} ─────────────────────────────────────────────► C_{1,0}
       Q,C_{0,0},C_{1,0} ──────────────────────────────────► C_{1,1}
       Q,C_{0,0},C_{1,0},C_{1,1} ───────────────────────► C_{2,0}
       …
       Q,C_{0,0},…,C_{2,3} ────────────────────────────────────► S_0
       …
```

So **inter-level causality (k depends on < k) and intra-level right-to-left
ordering (j depends on < j at the same level) both fall out of the backbone's
natural causal mask** — no custom "scale-level mask" is built anywhere in the
code.

- **One backbone pass powers two losses.** The same `H` produces both the
  concept MSE (via `concept_head`) and the reasoning CE (via the frozen
  `lm_head`). No separate forward is needed.
- **No internal padding inside a row.** `pack_qcs_sequences` concatenates
  `[Q_b; C_b; S_b]` for each row `b`, yielding a contiguous sequence that
  the LLM sees as a single natural utterance. Padding, if any, is pushed to
  the right edge of the batch.
- **Teacher forcing at the concept slots.** Because `X_concept` is built
  from **groundtruth** `C_gt` (detached from Builder), every concept slot
  sees only past slots plus `Q`, exactly like NTP on text tokens.
- **Slot identity via additive markers.** Inside a level, slots share the
  same causal context; the only way the backbone can distinguish
  `C_{k,0}` from `C_{k,1}` is through `position_embeddings`. Level identity
  across K levels is provided analogously by `level_embeddings`.
- **Causality is natural, not specialised.** We do **not** build a
  scale-level causal mask by hand. The backbone's own causal mask, applied
  to the packed `[Q; C; S]` sequence, yields the correct dependency
  pattern: every concept slot sees `Q` plus earlier concept slots only; the
  solution tokens see `Q` plus all concept slots.

#### 4.2.4 Inference forward pass (`_forward_inference`, autoregressive with KV cache)

At test time the Builder is not used; the Predictor generates
`[Ĉ_0, ..., Ĉ_{K-1}]` from `Q` alone:

```
# Step 0 — prime the KV cache with Q
h_Q, past_kv = reason_model(inputs_embeds=embed_tokens(Q),
                             attention_mask=q_mask,
                             use_cache=True,
                             output_hidden_states=True)
q_len        = q_mask.sum(dim=-1)                              # per-row real length
last_real_idx= (q_mask * arange_like(q_mask)).argmax(-1)       # side-agnostic
Ĉ_0         = concept_head(h_Q[:, last_real_idx, :])          # first concept

# Step t = 1 .. Σ L_k - 1 — feed one concept at a time
for t in range(1, total_slots):
    x = back_proj(Ĉ_{t-1}) \
        + level_embeddings(level_ids[t-1]) \
        + position_embeddings(pos_ids[t-1])
    out = reason_model(
        inputs_embeds=x,
        past_key_values=past_kv,
        position_ids=torch.tensor([q_len + t - 1]),            # explicit RoPE id
        use_cache=True,
        output_hidden_states=True,
    )
    past_kv = out.past_key_values
    Ĉ_t     = concept_head(out.hidden_states[-1][:, -1, :])
```

- **KV-cache reuse:** after priming on `Q`, every new concept costs only
  `O(1)` transformer steps. The cache is updated in place.
- **Explicit `position_ids = q_len + t − 1`:** required for RoPE
  consistency; without this, the model would re-interpret step `t` as
  position `0`, desynchronising the positional encoding.
- **Side-agnostic length via argmax:** `last_real_idx` works for both
  left-padded and right-padded Q batches.

### 4.3 Why This Separation?

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

### 4.4 Relationship to VAR

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
[`losses.py`](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/examples/nlcpV4/losses.py)
`compute_predictor_loss` for the authoritative implementation.

#### 5.2.1 Concept Loss (per-level, averaged over K)

```
L_concept = (1/K) · Σ_{k=0}^{K-1} loss_fn(Ĉ_k, C_k.detach())

loss_fn ∈ { mse, cosine }       # selected by loss.concept_loss_type
  mse    : F.mse_loss(Ĉ_k, C_k)
  cosine : 1 - F.cosine_similarity(Ĉ_k, C_k, dim=-1).mean()
```

Properties:

- **Per-level averaging** prevents fine-grained levels (which have more slots,
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
   [loss-desien-analysis.md](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/examples/nlcpV4/loss-desien-analysis.md) §6.2).

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

| VAR Component                | Our Equivalent                             | Role                                            |
|------------------------------|--------------------------------------------|-------------------------------------------------|
| **Phase 1: VQ-VAE**          | **ConceptPyramidBuilder**                  | Extract groundtruth from full information (CoT) |
| Encoder                      | `reason_model.embed` + encode MLP          | Encode CoT to hidden states                     |
| Multi-scale quantizer        | Soft attention + residual                  | Extract hierarchical concepts                   |
| Codebook                     | `concept_queries`                          | Learnable "vocabulary" of concept patterns      |
| f_hat / f_rest               | H_hat / H_rest                             | Residual decomposition                          |
| **Phase 2: VAR Transformer** | **ConceptPredictor**                       | Generate autoregressively from condition        |
| Decoder-only Transformer     | `reason_model` (shared or independent LLM) | Predict next concept slot given previous        |
| Scale embeddings             | `level_embeddings` + `position_embeddings` | Mark level k and within-level slot j            |
| Prediction head              | `concept_head` MLP                         | Project backbone hidden state to concept space  |
| VAE Decoder                  | `reason_model.lm_head` reused on solution  | Decode final output tokens from concepts        |

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

| Guarantee                                             | Mechanism                                                                                 | Strength                  |
|-------------------------------------------------------|-------------------------------------------------------------------------------------------|---------------------------|
| Inter-level causality (level k depends on levels < k) | Packed `[Q; C; S]` sequence + backbone's native causal mask                               | **Hard** (architectural)  |
| Intra-level slot identity (C_{k,0} ≠ C_{k,1})         | `position_embeddings` added to every slot                                                 | **Hard** (architectural)  |
| Backbone reuse without new Transformer                | `reason_model` processes concepts as `inputs_embeds`; `concept_head` is the only new head | **Hard** (architectural)  |
| Teacher forcing alignment                             | Groundtruth `C_gt` from frozen Builder, detached                                          | **Hard** (training setup) |
| SHARED-mode weight integrity                          | Predictor aliases Builder's `reason_model` / `back_proj`; LoRA forbidden (fail-fast)      | **Hard** (architectural)  |
| Inference-time RoPE consistency                       | Explicit `position_ids = q_len + t − 1` passed on every AR step                           | **Hard** (architectural)  |

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
2. The full NLCP training pipeline (with NTP loss) provides strong corrective signals
3. The design is fully differentiable and end-to-end trainable

These limitations should be monitored during experiments but do not warrant architectural changes at this stage.
