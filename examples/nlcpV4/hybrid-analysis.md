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

#### 1.4.1 The Two-Phase Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PHASE                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: (Q, CoT, Solution)                                                   │
│         │   │       │                                                        │
│         │   │       └── Used to validate pyramid's reasoning capability      │
│         │   └── Core source for building concept pyramid                     │
│         └── Prior/context (not part of pyramid)                              │
│                                                                              │
│  Step 1: ConceptPyramidBuilder                                               │
│          ├── Encodes CoT → H_CoT                                            │
│          ├── Applies soft attention with learnable queries (1→2→4→8→16→32)   │
│          ├── Uses residual reconstruction for coarse-to-fine decomposition   │
│          └── Outputs: Groundtruth [C_0, C_1, ..., C_{K-1}]  (K=6 levels)    │
│                                                                              │
│  Step 2: ConceptPredictor (Decoder-only Transformer)                         │
│          ├── Input sequence: [Q, C_0, C_1, ..., C_{K-1}, Solution]          │
│          ├── Training: Teacher forcing with causal masking                   │
│          ├── Learns: Given Q and previous concepts, predict next level       │
│          └── Output: Predicted concepts matching Builder's groundtruth       │
│                                                                              │
│  Loss: L_recon + L_ordering + L_reasoning (Builder)                      │
│        L_prediction (Predictor, MSE vs frozen Builder GT)                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
                                    ↓ Trained models
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PHASE                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: Q only (no CoT, no Solution)                                         │
│                                                                              │
│  Step 1: ConceptPredictor autoregressively generates                         │
│          Q → Ĉ_0 → Ĉ_1 → Ĉ_2 → Ĉ_3 → Ĉ_4 → Ĉ_5                              │
│                                                                              │
│  Step 2: Solution Decoder                                                    │
│          [Q, Ĉ_0, Ĉ_1, Ĉ_2, Ĉ_3, Ĉ_4, Ĉ_5] → Solution                       │
│                                                                              │
│  Output: Solution (without explicit CoT generation)                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

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
L_builder = L_recon + λ_order × L_ordering + λ_reasoning × L_reasoning

- L_recon: ||back_proj(H_hat_K) - H_CoT||²  (reconstruction in encoder space)
- L_ordering: Intra-level concept ordering (Section 3.2)
- L_reasoning: NTP cross-entropy — [concepts + Q] → predict Solution tokens
    (ensures pyramid is useful for reasoning, not just reconstruction)
```

**Key Properties**:
- Builder is only used during training to generate groundtruth
- All mechanisms from Sections 2-3 are employed (soft attention, residual flow, query expansion)
- The output serves as training targets for the Predictor

### 4.2 ConceptPredictor (Phase 2: Generation)

The Predictor learns to autoregressively generate the concept pyramid from Q alone, mimicking the Builder's output.

**Architecture**: Concept Transformer with scale-level causal masking.
The backbone model can be configured to either:
- Reuse the Builder's `reason_model` (shared weights, `use_shared_model: true`)
- Load its own separate model (`use_shared_model: false`)

**Components**:
- `q_proj + q_proj_norm`: Project question hidden states to concept space
- `level_embeddings`: Learnable per-level embeddings [K, D] (analogous to VAR's `lvl_emb`)
- `position_embeddings`: Within-level positional encoding
- `concept_transformer`: Transformer blocks with scale-level causal mask
- `concept_head`: 2-layer MLP predicting concept vectors
- `start_token`: Learnable [1, D] token injected with question context

**Training** (Teacher Forcing):
```
Input:  [start_token + Q_context, C_0_gt, C_1_gt, ..., C_{K-2}_gt]
Target: [C_0_gt, C_1_gt, ..., C_{K-1}_gt]

start_token output → predicts C_0
C_0 output positions → predicts C_1
C_1 output positions → predicts C_2
...
C_{K-2} output positions → predicts C_{K-1}
```

**Scale-Level Causal Masking** (VAR.md Section 5.3.1):
- Start token: visible to all (like VAR's class embedding)
- Within a level: full visibility (parallel prediction)
- Across levels: strict causality — level k attends to levels < k only
- `mask[i,j] = 1 if level[i] >= level[j] else 0`

**Loss**:
```
L_predictor = (1/K) × Σ_{k=0}^{K-1} MSE(Ĉ_k, C_k.detach())

Where C_k are groundtruth concepts from frozen Builder
```

**Inference** (Autoregressive Generation):
```
Step 0: [start + Q_context] → Ĉ_0
Step 1: [start + Q_context, Ĉ_0] → Ĉ_1
Step 2: [start + Q_context, Ĉ_0, Ĉ_1] → Ĉ_2
...
Step K-1: [start + Q_context, Ĉ_0, ..., Ĉ_{K-2}] → Ĉ_{K-1}
```

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
L_reasoning = CrossEntropy(reason_model([concept_embeds; Q_embeds]), solution_tokens)
```

Validates that the concept pyramid supports reasoning. Concepts are back-projected to encoder space, concatenated with question embeddings, and fed through the reason_model's lm_head. Cross-entropy loss on solution tokens ensures the pyramid is useful for reasoning, not just reconstruction.

#### 5.1.4 Total Builder Loss

```
L_builder = L_recon + λ_order × L_order + λ_reasoning × L_reasoning
```

### 5.2 ConceptPredictor Loss

The Predictor's loss ensures accurate autoregressive generation of concepts.

```
L_predictor = (1/K) × Σ_{k=0}^{K-1} MSE(Ĉ_k, C_k.detach())

Where:
- Ĉ_k: Predicted concepts at level k
- C_k: Groundtruth concepts from frozen Builder (detached)
```

**Training**: Teacher forcing with groundtruth concepts from frozen Builder
**Inference**: Autoregressive generation level by level

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

| VAR Component                | Our Equivalent             | Role                                            |
|------------------------------|----------------------------|-------------------------------------------------|
| **Phase 1: VQ-VAE**          | **ConceptPyramidBuilder**  | Extract groundtruth from full information (CoT) |
| Encoder                      | Encoder(CoT)               | Encode CoT to hidden states                     |
| Multi-scale quantizer        | Soft attention + residual  | Extract hierarchical concepts                   |
| Codebook                     | concept_queries            | Learnable "vocabulary" of concept patterns      |
| f_hat / f_rest               | H_hat / H_rest             | Residual decomposition                          |
| **Phase 2: VAR Transformer** | **ConceptPredictor**       | Generate autoregressively from condition        |
| Decoder-only Transformer     | Decoder-only Transformer   | Predict next level given previous               |
| Scale embeddings             | Level queries / embeddings | Mark current generation level                   |
| VAE Decoder                  | Solution Decoder           | Decode final output from concepts               |

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

| Guarantee                 | Mechanism                     | Strength                  |
|---------------------------|-------------------------------|---------------------------|
| Level-level causality     | Causal masking in Transformer | **Hard** (architectural)  |
| Intra-level parallelism   | Level-wise attention mask     | **Hard** (architectural)  |
| Teacher forcing alignment | Groundtruth from Builder      | **Hard** (training setup) |

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

The Concept Pyramid design is architecturally sound. The ConceptPyramidBuilder uses soft attention (soft boundaries) with learnable query expansion to extract hierarchical concepts from CoT via purely residual decomposition — no cross-scale conditioning, following VAR's VQ-VAE Stage 1 principle. The ConceptPredictor learns to autoregressively generate these concepts from Q alone using scale-level causal attention, following VAR's Transformer Stage 2 principle. The rank bottleneck provides a hard guarantee of coarse-to-fine hierarchy. The combination of softmax competition, residual flow, and ordering loss creates sufficient inductive bias for DLCM-style segment-concept correspondence without requiring hard segmentation.

The main limitations — soft segment locality, potential extraction imbalance, and Q-only generalization — are inherent trade-offs of the soft attention approach. They are acceptable for our research goals because:
1. The soft approach is strictly more expressive than hard segmentation
2. The full NLCP training pipeline (with NTP loss) provides strong corrective signals
3. The design is fully differentiable and end-to-end trainable

These limitations should be monitored during experiments but do not warrant architectural changes at this stage.
