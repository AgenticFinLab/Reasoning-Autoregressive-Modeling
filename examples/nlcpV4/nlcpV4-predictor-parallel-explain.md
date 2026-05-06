# Concept Predictor Parallel (Option Y): Per-Level Queries + Cross-Attention

## 1. Overview and Motivation

### 1.1 What Problem Does Option Y Solve?

The original ConceptPredictor (Option X) generates the concept pyramid **one concept at a time** — a flat autoregressive loop of 63 steps for K=6 levels. Option Y introduces a **per-level parallel** architecture: all concepts within a single level are generated **simultaneously** via cross-attention, reducing the inference loop from 63 steps to just K=6 passes.

```
Option X (flat AR):     63 sequential LLM steps  (one concept per step)
Option Y (per-level):    6 sequential LLM passes  (one LEVEL per pass, all L_k concepts at once)
                         ─────────────────────────
                         10.5× fewer sequential LLM calls
```

### 1.2 Architectural Positioning

```
                    NLCP V4 Two-Stage Pipeline
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: ConceptPyramidBuilder                               │
│   Input: (Q, CoT, S)  →  Output: C_gt = [C_0, ..., C_{K-1}] │
│   (frozen during Stage 2)                                    │
└─────────────────────────────────────────────────────────────┘
                              │ detach()
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: ConceptPredictor                                    │
│                                                              │
│   ┌─────────────────────────┐  ┌──────────────────────────┐ │
│   │ Option X: Flat AR       │  │ Option Y: Per-Level Query │ │
│   │ (concept_predictor.py)  │  │ (concept_predictor_       │ │
│   │                         │  │      parallel.py)         │ │
│   │ 63 steps, 1 concept/step│  │ K steps, L_k concepts/   │ │
│   │                         │  │     step via cross-attn   │ │
│   └─────────────────────────┘  └──────────────────────────┘ │
│                                                              │
│   Both produce identical PredictorOutput                     │
│   Both share the same losses.py                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Key Notation (inherited from nlcpV4-explain.md)

| Symbol      | Meaning                              | Default Value       |
|-------------|--------------------------------------|---------------------|
| **K**       | Number of pyramid levels             | 6                   |
| **L_k**     | Concepts at level k                  | 2^k (1,2,4,8,16,32) |
| **D**       | Concept space dimension              | 896                 |
| **D_enc**   | Encoder/LLM hidden dimension         | 896                 |
| **B**       | Batch size                           | 4                   |
| **L_Q**     | Question token count                 | 40                  |
| **total_C** | Total concept slots: Σ L_k           | 63                  |
| **C_k**     | All concepts at level k: [B, L_k, D] |                     |
| **Ĉ_k**     | Predicted concepts at level k        |                     |

---

## 2. Core Idea: Two-Stage Internal Architecture

Option Y separates the forward pass into two internal stages within a single model:

```
┌─────────────────────────────────────────────────────────────────────┐
│  INTERNAL Stage 1: Content Backbone (LLM)                           │
│                                                                     │
│  Purpose: Contextualise ALL input content into rich hidden states   │
│  Input:   [Q_embeds, back_decode(C_0..C_{K-1}) + slot_markers]      │
│  Output:  Hidden states H [B, L_Q + 63, D_enc]                      │
│                                                                     │
│  ✓ Real content only (question tokens + concept embeddings)         │
│  ✗ NO learnable queries in the LLM sequence                        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ H (LLM hidden states)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  INTERNAL Stage 2: Per-Level Cross-Attention Head                    │
│                                                                     │
│  Purpose: EXTRACT predictions for each level from H                 │
│  Mechanism: Learnable level_queries[k] cross-attend to H prefix     │
│  Output:  Ĉ_k for each k ∈ [0, K)                                  │
│                                                                     │
│  ✓ All K levels run in PARALLEL (no sequential dependency)          │
│  ✓ Each level sees only Q + levels < k (information consistency)    │
└─────────────────────────────────────────────────────────────────────┘
```

**The fundamental principle**: Learnable parameters (queries) and real content (Q tokens, concept embeddings) NEVER share the same LLM input sequence. The LLM processes only content; the queries live in a separate cross-attention head.

---

## 3. Architectural Symmetry with the Builder

Option Y mirrors the Builder's design philosophy:

```
    Builder (Stage 1)                     Predictor Option Y (Stage 2)
    ─────────────────                     ──────────────────────────────
    
    encoder(CoT)                          reason_model([Q, back_decode(C_<k)])
         │                                           │
         ▼                                           ▼
    H_CoT [B, L, D_enc]                  H [B, prefix_len_k, D_enc]
         │                                           │
    concept_queries[k] @ H_CoT           level_queries[k] @ H_prefix_k
         │                                           │
         ▼                                           ▼
    C_k [B, L_k, D]                      Ĉ_k [B, L_k, D]
    (concept space)                       (concept space)
```

Both use **learnable queries** to extract per-level outputs via attention. The Builder attends over CoT hidden states; the Predictor attends over LLM hidden states of [Q + previous levels].

---

## 4. Detailed Component Analysis

### 4.1 Component Table

| Component             | Shape                                | Role                                                         |
|-----------------------|--------------------------------------|--------------------------------------------------------------|
| `reason_model`        | HuggingFace causal LM                | Content backbone; produces hidden states H                   |
| `back_proj`           | Linear(D → D_enc)                    | Lifts concept-space to encoder-space for LLM input           |
| `level_embeddings`    | Embedding(K, D_enc)                  | Per-slot level marker (k=0..5)                               |
| `position_embeddings` | Embedding(max(L_k), D_enc)           | Per-slot intra-level marker (j=0..L_k-1)                     |
| **`level_queries`**   | **ParameterList of K: [L_k, D_enc]** | **Core of Option Y — learnable queries for cross-attention** |
| `query_norm`          | LayerNorm(D_enc)                     | Pre-norm on query side of cross-attention                    |
| `context_norm`        | LayerNorm(D_enc)                     | Pre-norm on context (KV) side of cross-attention             |
| `cross_attn`          | MultiheadAttention(D_enc, 8 heads)   | Shared cross-attention module across all levels              |
| `post_norm`           | LayerNorm(D_enc)                     | Post-norm after attention + residual                         |
| `concept_head`        | Linear→GELU→Linear (D_enc → D)       | Projects attention output to concept space                   |

### 4.2 Level Queries — The Core Innovation

```
level_queries[k] ∈ ℝ^{L_k × D_enc}

    level_queries[0] : [ 1, 896]    ← 1 learnable vector
    level_queries[1] : [ 2, 896]    ← 2 learnable vectors
    level_queries[2] : [ 4, 896]    ← 4 learnable vectors
    level_queries[3] : [ 8, 896]    ← 8 learnable vectors
    level_queries[4] : [16, 896]    ← 16 learnable vectors
    level_queries[5] : [32, 896]    ← 32 learnable vectors
    ─────────────────────────────
    Total: 63 × 896 ≈ 56,448 parameters
```

Each `level_queries[k]` learns "what information to extract" from the LLM's context for level k. They function like **DETR-style object queries** — each query slot "asks" for a specific concept from the contextualised hidden states.

### 4.3 Cumulative Lengths and Context Windows

```python
cum_lengths = [0, 1, 3, 7, 15, 31, 63]

# For level k, the context prefix includes Q + all concepts from levels < k:
prefix_len_k = L_Q + cum_lengths[k]
```

| Level k | cum_lengths[k] | prefix_len_k | Context includes           |
|---------|----------------|--------------|----------------------------|
| 0       | 0              | 40           | Q only                     |
| 1       | 1              | 41           | Q + C_0 (1 concept)        |
| 2       | 3              | 43           | Q + C_0 + C_1 (3 concepts) |
| 3       | 7              | 47           | Q + C_0..C_2 (7 concepts)  |
| 4       | 15             | 55           | Q + C_0..C_3 (15 concepts) |
| 5       | 31             | 71           | Q + C_0..C_4 (31 concepts) |

**Critical**: Level k's context **excludes** level k itself — the prediction must not see its own groundtruth.

---

## 5. Training Forward Pass — Full Flow

### 5.1 High-Level Pipeline

```
═══════════════════════════════════════════════════════════════════════════
TRAINING: Single LLM Pass + K Parallel Cross-Attentions
═══════════════════════════════════════════════════════════════════════════

Input: question_ids [B, L_Q], gt_concepts = [C_0, ..., C_{K-1}]

┌────────────── Internal Stage 1: Content Backbone ──────────────────┐
│                                                                     │
│  Step 1: Prepare concept embeddings                                 │
│     concepts_flat = torch.cat(gt_concepts, dim=1)  → [B, 63, D]    │
│     concept_embeds = back_proj(concepts_flat)       → [B, 63, D_enc]│
│     concept_embeds += level_embeddings(level_ids)                   │
│     concept_embeds += position_embeddings(pos_ids)                  │
│                                                                     │
│  Step 2: Prepare question embeddings                                │
│     Q_embeds = embed_tokens(question_ids)           → [B, 40, D_enc]│
│                                                                     │
│  Step 3: Concatenate and run LLM                                    │
│     inputs_embeds = cat([Q_embeds, concept_embeds]) → [B, 103, D_enc]│
│     H = reason_model(inputs_embeds)                 → [B, 103, D_enc]│
│                                                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  Internal Stage 2: Per-Level Cross-Attention (PARALLEL)              │
│                                                                     │
│  for k = 0, 1, 2, 3, 4, 5:                                         │
│      prefix_end = L_Q + cum_lengths[k]                              │
│      context_k  = H[:, :prefix_end, :]    (truncated hidden prefix) │
│      Ĉ_k       = _extract_level(k, context_k)                      │
│                                                                     │
│  Result:                                                            │
│      Ĉ_0 [B,  1, D]  from context [B, 40, D_enc]                   │
│      Ĉ_1 [B,  2, D]  from context [B, 41, D_enc]                   │
│      Ĉ_2 [B,  4, D]  from context [B, 43, D_enc]                   │
│      Ĉ_3 [B,  8, D]  from context [B, 47, D_enc]                   │
│      Ĉ_4 [B, 16, D]  from context [B, 55, D_enc]                   │
│      Ĉ_5 [B, 32, D]  from context [B, 71, D_enc]                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Concrete Numerical Example

Let B=4, L_Q=40, K=6, level_lengths=[1,2,4,8,16,32], D=D_enc=896.

**Step-by-step with actual tensor shapes:**

```
1. gt_concepts input:
     C_0: [4,  1, 896]
     C_1: [4,  2, 896]
     C_2: [4,  4, 896]
     C_3: [4,  8, 896]
     C_4: [4, 16, 896]
     C_5: [4, 32, 896]

2. concepts_flat = cat(gt_concepts, dim=1):  [4, 63, 896]

3. back_proj(concepts_flat):                  [4, 63, 896]
   (D=D_enc=896, so this is a learned rotation, not a dim change)

4. Slot markers added:
     level_ids_flat = [0, 1,1, 2,2,2,2, 3,3,3,3,3,3,3,3, 4×16, 5×32]
     pos_ids_flat   = [0, 0,1, 0,1,2,3, 0,1,2,3,4,5,6,7, 0..15, 0..31]
     
     concept_embeds += level_embeddings(level_ids_flat)   [63, 896] → broadcast
     concept_embeds += position_embeddings(pos_ids_flat)  [63, 896] → broadcast

5. Q_embeds = embed_tokens(question_ids):     [4, 40, 896]

6. inputs_embeds = cat([Q, concepts]):         [4, 103, 896]
                                                    ↑
                                              40 + 63 = 103

7. H = reason_model.backbone(inputs_embeds):   [4, 103, 896]
   (causal attention: each position sees only previous positions)

8. Cross-attention extraction (per level):
     k=0: context = H[:, :40, :]   → [4, 40, 896]
           level_queries[0] [1, 896] → expand → [4, 1, 896]
           Ĉ_0 = extract(0, context) → [4, 1, 896]
     
     k=1: context = H[:, :41, :]   → [4, 41, 896]
           level_queries[1] [2, 896] → expand → [4, 2, 896]
           Ĉ_1 = extract(1, context) → [4, 2, 896]
     
     ... (k=2..5 analogous)
```

### 5.3 Why Training is "Parallel" at Stage 2

After the single LLM pass produces H, **all K cross-attentions are independent** — they share no state with each other. Level 3's cross-attention does not depend on level 2's output. They each simply read a different slice of the **same** H tensor:

```
H = [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||]
     ↑                                      ↑                                                            ↑
     position 0                        position 40                                               position 102
     (first Q token)                   (first concept slot)                                 (last concept slot)

Level 0 reads: H[:, 0:40, :]                    ← Q only
Level 1 reads: H[:, 0:41, :]                    ← Q + 1 concept position
Level 2 reads: H[:, 0:43, :]                    ← Q + 3 concept positions
Level 3 reads: H[:, 0:47, :]                    ← Q + 7 concept positions
Level 4 reads: H[:, 0:55, :]                    ← Q + 15 concept positions
Level 5 reads: H[:, 0:71, :]                    ← Q + 31 concept positions

All are READ-ONLY slices of H. No write dependency between levels.
→ Can be computed in parallel (or in any order).
```

---

## 6. The Cross-Attention Mechanism (`_extract_level`)

### 6.1 Full Data Flow

For level k=3 (L_k=8, prefix_len=47, D_enc=896, B=4, num_heads=8):

```
┌─────────────────────────────────────────────────────────────────┐
│  _extract_level(level_idx=3, context=[4, 47, 896])              │
│                                                                  │
│  1. Expand queries:                                              │
│     queries = level_queries[3]              [8, 896]             │
│     queries = queries.unsqueeze(0).expand() [4, 8, 896]         │
│                                                                  │
│  2. Pre-LayerNorm (stabilises training):                         │
│     q_normed = query_norm(queries)          [4, 8, 896]         │
│     c_normed = context_norm(context)        [4, 47, 896]        │
│                                                                  │
│  3. Multi-head cross-attention:                                  │
│     attn_out = cross_attn(                                       │
│         query=q_normed,     # Q: [4, 8, 896]                    │
│         key=c_normed,       # K: [4, 47, 896]                   │
│         value=c_normed      # V: [4, 47, 896]                   │
│     )                       → attn_out: [4, 8, 896]             │
│                                                                  │
│     Internal to cross_attn (8 heads, d_head=112):                │
│       per-head Q: [4, 8, 8, 112]                                 │
│       per-head K: [4, 8, 47, 112]                                │
│       scores:     [4, 8, 8, 47]  = Q @ K^T / √112               │
│       probs:      [4, 8, 8, 47]  = softmax(scores)               │
│       per-head V: [4, 8, 47, 112]                                │
│       raw_out:    [4, 8, 8, 112] = probs @ V                     │
│       concat:     [4, 8, 896]    = reshape + out_proj            │
│                                                                  │
│  4. Residual connection:                                         │
│     out = attn_out + queries  (NOT normed queries)  [4, 8, 896] │
│                                                                  │
│  5. Post-LayerNorm:                                              │
│     out = post_norm(out)                            [4, 8, 896] │
│                                                                  │
│  6. Concept head (MLP):                                          │
│     out = Linear(896→896) → GELU → Linear(896→896) [4, 8, 896] │
│     = Ĉ_3                                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Visual: What the Cross-Attention "Sees"

```
For level k=3, each of the 8 query vectors attends over 47 context positions:

        Query 0  Query 1  Query 2  ...  Query 7      (8 learnable queries)
           │        │        │              │
           ▼        ▼        ▼              ▼
    ┌──────────────────────────────────────────────┐
    │   Context positions 0..46                     │
    │                                              │
    │   [Q_0, Q_1, ..., Q_39, C_{0,0}, C_{1,0},   │
    │    C_{1,1}, C_{2,0}, C_{2,1}, C_{2,2},       │
    │    C_{2,3}]                                   │
    │                                              │
    │   = 40 Q tokens + 7 concept positions        │
    │     (C_0: 1 slot, C_1: 2 slots, C_2: 4 slots)│
    └──────────────────────────────────────────────┘
           │        │        │              │
           ▼        ▼        ▼              ▼
    attn_out_0  attn_out_1  attn_out_2 ... attn_out_7

    Each attn_out_i = weighted sum of context positions
    Weights = softmax(query_i @ context / √d_head)

    NO causal mask on cross-attention — queries see ALL context positions.
    (The causal restriction is already in the LLM's own processing of H.)
```

### 6.3 Why the Residual Add (`attn_out + queries`)

```
out = post_norm(attn_out + queries)    ← residual on RAW queries (not normed)

Purpose: If the cross-attention head is near-zero at initialisation (common
with random init), the residual ensures the concept_head still receives a
meaningful signal from the learnable queries themselves.

Without residual:  concept_head(≈ 0) → degenerate Ĉ_k at init
With residual:     concept_head(queries) → non-zero, query-seeded Ĉ_k at init

This stabilises early training — the model can always produce a baseline
prediction from its queries, then gradually improve by attending to context.
```

---

## 7. Inference Forward Pass — K Sequential Passes

### 7.1 Why Inference Cannot Be Fully Parallel

During **training**, gt_concepts are available (teacher forcing), so the LLM processes all 63 concept positions in one pass. During **inference**, we don't have gt_concepts — we must generate them level-by-level, feeding each level's prediction as input for the next LLM pass.

```
Training:  gt available → 1 LLM pass + K parallel cross-attentions
Inference: no gt        → K LLM passes (growing KV cache) + K cross-attentions
```

### 7.2 Full Inference Flow Diagram

```
═══════════════════════════════════════════════════════════════════════════
INFERENCE: K=6 Sequential LLM Passes with KV Cache
═══════════════════════════════════════════════════════════════════════════

Pass 0 (prime with Q):
    ┌──────────────────────────────────────────────────────────┐
    │ x = embed_tokens(Q)                    [4, 40, 896]      │
    │ out = LLM(x, use_cache=True)                             │
    │ pkv = out.past_key_values              (KV cache: 40 pos)│
    │ context = out.last_hidden_state        [4, 40, 896]      │
    │                                                          │
    │ Ĉ_0 = _extract_level(0, context)      [4,  1, 896]      │
    └──────────────────────────────────────────────────────────┘
         │ Ĉ_0

Pass 1 (feed Ĉ_0, predict level 1):
    ┌──────────────────────────────────────────────────────────┐
    │ x = back_proj(Ĉ_0) + lvl_emb(0) + pos_emb(0) [4,1,896] │
    │ out = LLM(x, past_key_values=pkv)                        │
    │ pkv updated                            (KV cache: 41 pos)│
    │ context = cat(context, out.hidden)     [4, 41, 896]      │
    │                                                          │
    │ Ĉ_1 = _extract_level(1, context)      [4,  2, 896]      │
    └──────────────────────────────────────────────────────────┘
         │ Ĉ_1

Pass 2 (feed Ĉ_1 = 2 positions, predict level 2):
    ┌──────────────────────────────────────────────────────────┐
    │ x = back_proj(Ĉ_1) + markers          [4, 2, 896]       │
    │ out = LLM(x, past_key_values=pkv)                        │
    │ pkv updated                            (KV cache: 43 pos)│
    │ context = cat(context, out.hidden)     [4, 43, 896]      │
    │                                                          │
    │ Ĉ_2 = _extract_level(2, context)      [4,  4, 896]      │
    └──────────────────────────────────────────────────────────┘
         │ Ĉ_2

Pass 3 (feed Ĉ_2 = 4 positions, predict level 3):
    ┌──────────────────────────────────────────────────────────┐
    │ x = back_proj(Ĉ_2) + markers          [4, 4, 896]       │
    │ out = LLM(x, past_key_values=pkv)                        │
    │ pkv updated                            (KV cache: 47 pos)│
    │ context = cat(context, out.hidden)     [4, 47, 896]      │
    │                                                          │
    │ Ĉ_3 = _extract_level(3, context)      [4,  8, 896]      │
    └──────────────────────────────────────────────────────────┘

Pass 4 (feed Ĉ_3 = 8 positions):   context → [4, 55, 896],  Ĉ_4 [4, 16, 896]
Pass 5 (feed Ĉ_4 = 16 positions):  context → [4, 71, 896],  Ĉ_5 [4, 32, 896]
```

### 7.3 KV Cache Growth Table

| Pass | New tokens fed to LLM | Cumulative KV cache | Cross-attn context size | Output       |
|------|-----------------------|---------------------|-------------------------|--------------|
| 0    | 40 (Q)                | 40                  | [B, 40, 896]            | Ĉ_0 [B,1,D]  |
| 1    | 1 (Ĉ_0)               | 41                  | [B, 41, 896]            | Ĉ_1 [B,2,D]  |
| 2    | 2 (Ĉ_1)               | 43                  | [B, 43, 896]            | Ĉ_2 [B,4,D]  |
| 3    | 4 (Ĉ_2)               | 47                  | [B, 47, 896]            | Ĉ_3 [B,8,D]  |
| 4    | 8 (Ĉ_3)               | 55                  | [B, 55, 896]            | Ĉ_4 [B,16,D] |
| 5    | 16 (Ĉ_4)              | 71                  | [B, 71, 896]            | Ĉ_5 [B,32,D] |

Total tokens processed: 40 + 1 + 2 + 4 + 8 + 16 = 71 (not 40 + 63 = 103, because Ĉ_5 is never fed back).

### 7.4 State Variables Across Passes

Three pieces of state are maintained across passes:

```
pkv (past_key_values):
    The LLM's KV cache. Grows by L_{k-1} entries per pass.
    Used by the LLM for self-attention over all previously processed positions.

running_mask:
    Attention mask covering all positions in the cache.
    Extended by L_{k-1} ones per pass.
    Shape: [B, cumulative_positions]

context:
    Running concatenation of LLM hidden states.
    Used ONLY by the cross-attention head (NOT by the LLM, which uses its own KV cache).
    Shape: [B, cumulative_positions, D_enc]
    
    IMPORTANT: The LLM never reads `context` — it uses pkv.
    The cross-attention head never reads pkv — it uses `context`.
    Two separate state streams for two separate purposes.
```

---

## 8. Comparison: Option X vs Option Y

### 8.1 Architecture Comparison

```
┌────────────────────────────────────────────────────────────────────────┐
│  Option X (Flat AR)                                                     │
│                                                                        │
│  Training:                                                             │
│    [Q; C_0; C_1,0; C_1,1; ...; C_5,31; S]  ← packed into ONE sequence │
│    ONE backbone pass → concept_head at each concept position           │
│    Causal mask naturally enforces inter/intra-level dependencies        │
│                                                                        │
│  Inference:                                                            │
│    Step 0: LLM(Q)         → Ĉ_{0,0}                                   │
│    Step 1: LLM(Ĉ_{0,0})  → Ĉ_{1,0}                                   │
│    Step 2: LLM(Ĉ_{1,0})  → Ĉ_{1,1}                                   │
│    ...                                                                 │
│    Step 62: LLM(Ĉ_{5,30})→ Ĉ_{5,31}                                  │
│                                                                        │
│    Total: 63 sequential LLM forward calls (1 concept per call)         │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│  Option Y (Per-Level Parallel)                                          │
│                                                                        │
│  Training:                                                             │
│    [Q; back_decode(all C) + markers]  ← content backbone               │
│    ONE backbone pass → K parallel cross-attentions                     │
│                                                                        │
│  Inference:                                                            │
│    Pass 0: LLM(Q)         → cross_attn → Ĉ_0 (1 concept)             │
│    Pass 1: LLM(Ĉ_0)      → cross_attn → Ĉ_1 (2 concepts at once)    │
│    Pass 2: LLM(Ĉ_1)      → cross_attn → Ĉ_2 (4 concepts at once)    │
│    ...                                                                 │
│    Pass 5: LLM(Ĉ_4)      → cross_attn → Ĉ_5 (32 concepts at once)   │
│                                                                        │
│    Total: 6 sequential LLM forward calls (1 level per call)            │
└────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Detailed Comparison Table

| Aspect                      | Option X (Flat AR)                  | Option Y (Per-Level Parallel)       |
|-----------------------------|-------------------------------------|-------------------------------------|
| **Inference steps**         | Σ L_k = 63                          | K = 6                               |
| **Intra-level dependency**  | Sequential (C_{k,j} depends on j-1) | None (all L_k concepts in parallel) |
| **Inter-level dependency**  | Inherent via sequence order         | Explicit via context window cutoff  |
| **Learnable queries**       | None (backbone hidden → head)       | level_queries[k]: [L_k, D_enc]      |
| **Extra parameters**        | ~0 (just concept_head)              | ~56k (queries) + cross_attn weights |
| **Training architecture**   | Pure causal LM                      | Causal LM + cross-attention head    |
| **Concept differentiation** | Position in sequence + markers      | Separate query identity             |
| **VAR analogy**             | Token-level AR                      | Scale-level AR (like VAR itself!)   |

### 8.3 The VAR Alignment Insight

```
VAR Generation Process:
    Scale 1×1:  generate 1 token     (1 step)
    Scale 2×2:  generate 4 tokens    (1 step, parallel)
    Scale 4×4:  generate 16 tokens   (1 step, parallel)
    ...
    Scale 32×32: generate 1024 tokens (1 step, parallel)
    → K steps total, each step generates all tokens at one scale simultaneously

Option Y Generation Process:
    Level 0:  generate 1 concept     (1 LLM pass + 1 cross-attn)
    Level 1:  generate 2 concepts    (1 LLM pass + 1 cross-attn, parallel)
    Level 2:  generate 4 concepts    (1 LLM pass + 1 cross-attn, parallel)
    ...
    Level 5:  generate 32 concepts   (1 LLM pass + 1 cross-attn, parallel)
    → K passes total, each pass generates all concepts at one level simultaneously

Option Y IS the direct textual analog of VAR's scale-by-scale generation!
```

---

## 9. Why Per-Level Parallelism Works (Theoretical Justification)

### 9.1 Information Independence Within a Level

From nlcpV4-explain.md §2.3:

> The builder must be purely residual — each level only sees the current residual H_rest_k, with NO conditioning on previous levels' concepts.

This means concepts within the same level are extracted from the **same** residual H_rest_k, with no cross-dependency among them. They are independent projections of the same source:

```
C_{k,0} = level_proj(A_{k,0} @ H_rest_k)
C_{k,1} = level_proj(A_{k,1} @ H_rest_k)
...
C_{k,L_k-1} = level_proj(A_{k,L_k-1} @ H_rest_k)

These are L_k INDEPENDENT readouts from the same tensor.
→ No inherent sequential dependency among them.
→ A model that predicts all L_k simultaneously is architecturally valid.
```

### 9.2 The Cross-Attention as a Multi-Slot Soft Readout

The cross-attention mechanism naturally handles multiple simultaneous predictions:

```
For level k with L_k queries:

    scores[i, j] = query_i @ context_j / √d      for all (i, j)
    probs[i, :]  = softmax(scores[i, :])          per query, over context
    out[i]       = Σ_j probs[i, j] × context[j]   per query, weighted sum

Each of the L_k queries independently computes its own attention weights over
the shared context. No query "steals" from another — they all see the same
context, but learn to attend to different parts of it.

This is exactly how DETR's object queries work:
    - 100 queries, each detects one object independently
    - All attend to the same image features
    - Hungarian matching assigns GT to queries

Our setting:
    - L_k queries, each predicts one concept independently
    - All attend to the same LLM hidden prefix
    - Position correspondence assigns GT to queries (by slot order)
```

### 9.3 What Option Y Loses vs Option X

Option X's flat AR generates C_{k,j} conditioned on C_{k,j-1} (intra-level autoregression). Option Y generates all C_{k,0..L_k-1} simultaneously — they cannot condition on each other.

```
Option X intra-level dependency:
    C_{k,0} → C_{k,1} → C_{k,2} → ... → C_{k,L_k-1}
    (each concept sees all previous concepts at the same level)

Option Y intra-level dependency:
    C_{k,0}   C_{k,1}   C_{k,2}   ...   C_{k,L_k-1}
    (each concept sees only Q + levels < k, NOT siblings)
```

**Is this a problem?** Likely not, because:

1. The Builder's groundtruth concepts are already independently extracted (no intra-level conditioning)
2. The learnable queries provide per-slot identity — Query_0 ≠ Query_1 even without seeing each other's output
3. The context from the LLM already contains rich representations of Q + prior levels
4. VAR itself uses this exact pattern (parallel intra-scale) with great success

---

## 10. The Reasoning Loss Path (`_prepare_reasoning`)

After concept prediction, Option Y can optionally compute reasoning CE loss:

```
Sequence layout for reasoning:
    [Q_embeds | back_decode(predicted_concepts) | S_embeds]
    [B, 40]   [B, 63]                           [B, L_S]
    ─────────────────────────────────────────────────────
    Total: [B, 40 + 63 + L_S, D_enc]

    logits from position (40+63-1) to (40+63+L_S-1) predict solution tokens.
    
    Gradient path:
        predicted_concepts → back_decode → reason_model → CE loss
        (backprop flows through the concept predictions)
```

This is identical to Option X's reasoning path — ensuring both options produce the same PredictorOutput and use the same losses.py code.

---

## 11. Worked Example: GSM8K Question

### 11.1 Setup

```
Question: "If a bag has 5 red balls and 3 blue balls, how many balls total?"
Q tokenized: ["If", "a", "bag", "has", "5", "red", "balls", "and", "3", "blue", ...]
L_Q = 40 tokens (after padding)

Builder produces gt_concepts:
    C_0 [1, 1, 896]:  "arithmetic addition problem"      (global theme)
    C_1 [1, 2, 896]:  ["setup: quantities", "compute: sum"]
    C_2 [1, 4, 896]:  ["5 red", "3 blue", "addition op", "result 8"]
    C_3 [1, 8, 896]:  (finer decomposition...)
    C_4 [1, 16, 896]: (even finer...)
    C_5 [1, 32, 896]: (finest-grained details)
```

### 11.2 Training Pass (Teacher-Forced)

```
1. All 63 gt concepts → back_proj + markers → 63 embeddings
2. Concatenate with Q: [Q(40) | concepts(63)] = 103 positions
3. LLM processes all 103 positions (causal mask):
     Position 0-39:   Q context builds up
     Position 40:     C_{0,0} sees Q
     Position 41-42:  C_{1,0}, C_{1,1} see Q + C_0
     Position 43-46:  C_{2,0..3} see Q + C_0 + C_1
     ...

4. Hidden H [1, 103, 896] extracted

5. Level 0 cross-attention:
     level_queries[0] (1 vector) attends to H[:, :40, :] (Q only)
     → Ĉ_0 should predict "arithmetic addition problem"

6. Level 2 cross-attention:
     level_queries[2] (4 vectors) attends to H[:, :43, :] (Q + C_0 + C_1)
     Query 0 learns to extract: "5 red"
     Query 1 learns to extract: "3 blue"
     Query 2 learns to extract: "addition op"
     Query 3 learns to extract: "result 8"
```

### 11.3 Inference Pass (No GT)

```
Pass 0: LLM processes Q (40 tokens)
    → context [1, 40, 896]
    → Ĉ_0 = cross_attn(level_queries[0], context)
    → Ĉ_0 ≈ "arithmetic addition" [1, 1, 896]

Pass 1: Feed back_proj(Ĉ_0) + markers (1 new token)
    → context grows to [1, 41, 896]
    → Ĉ_1 = cross_attn(level_queries[1], context)
    → Ĉ_1 ≈ ["setup", "compute"] [1, 2, 896]

Pass 2: Feed back_proj(Ĉ_1) + markers (2 new tokens)
    → context grows to [1, 43, 896]
    → Ĉ_2 = cross_attn(level_queries[2], context)
    → Ĉ_2 ≈ ["5 red", "3 blue", "add", "8"] [1, 4, 896]

... (passes 3-5 analogous, producing progressively finer concepts)
```

---

## 12. Implementation Details

### 12.1 Shared vs Independent Model (Same as Option X)

Option Y supports the same two backbone modes:

```
SHARED (use_shared_model=True):
    predictor.reason_model = builder.reason_model  (alias)
    predictor.back_proj    = builder.back_proj      (alias)
    → Only level_queries, cross_attn, concept_head, norms are new parameters

INDEPENDENT (use_shared_model=False):
    predictor.reason_model = new AutoModelForCausalLM  (own copy)
    predictor.back_proj    = new Linear(D, D_enc)      (own copy)
    → All parameters are independent; LoRA optional
```

### 12.2 Weight Initialization

```python
level_queries:         N(0, 0.02)   # Small init, symmetry-breaking
level_embeddings:      N(0, 0.02)
position_embeddings:   N(0, 0.02)
concept_head (Linear): Xavier uniform
back_proj (if owned):  Xavier uniform
cross_attn:            PyTorch default (Xavier uniform for in_proj, zeros for bias)
```

### 12.3 Memory and Compute Comparison

```
Option X training (B=4, L_Q=40, total_C=63, K=6):
    LLM input: [B, ~103+L_S, D_enc] (one pass, includes solution)
    No cross-attention module
    Extra params: concept_head only (~1.6M)

Option Y training (B=4, L_Q=40, total_C=63, K=6):
    LLM input: [B, 103, D_enc] (one pass, no solution in this path)
    + 6 cross-attention calls (varying context sizes)
    Extra params: concept_head + cross_attn + norms + level_queries (~5M)

Option X inference:
    63 LLM forward calls (1 token each, after initial Q pass)
    Total KV growth: 63 positions

Option Y inference:
    5 LLM forward calls (1, 2, 4, 8, 16 tokens each, after initial Q pass)
    Total KV growth: 31 positions (C_5 is predicted but never fed back)
    + 6 cross-attention calls
    
    Speedup: 63/6 ≈ 10.5× fewer LLM calls
    (cross-attention is much cheaper than a full LLM forward)
```

---

## 13. Summary

### 13.1 One-Line Summary

> **Option Y replaces flat autoregressive concept generation (63 steps) with per-level cross-attention readout (6 passes): the LLM contextualises content only, and a separate learned query bank extracts all L_k concepts at each level simultaneously — directly mirroring VAR's scale-parallel token generation.**

### 13.2 Key Design Properties

| Property                            | Mechanism                                                     |
|-------------------------------------|---------------------------------------------------------------|
| Intra-level parallelism             | Cross-attention with L_k independent queries                  |
| Inter-level autoregression          | K sequential LLM passes, each feeding previous level's output |
| Information consistency             | Context truncated to exclude level k itself                   |
| No learnable params in LLM sequence | Queries live in separate cross-attention head                 |
| VAR alignment                       | Scale-by-scale generation (K steps, not Σ L_k steps)          |
| Builder symmetry                    | Both use learnable queries to extract from context            |
| Loss compatibility                  | Identical PredictorOutput → same losses.py                    |
| Inference efficiency                | 10.5× fewer sequential LLM calls than Option X                |
