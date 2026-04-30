# Concept Pyramid Reconstruction: Architectural Concerns

This document discusses two fundamental architectural concerns in the current
concept pyramid reconstruction path, with detailed analysis, code references,
and concrete numerical examples.

## Background: The Reconstruction Path

The full reconstruction pipeline:

```
H_CoT = Encoder(CoT)                                       [B, L, D_enc]
H_proj = LayerNorm(input_proj(H_CoT))                      [B, L, D]
f_rest_0 = H_proj,  f_hat_0 = 0

for k = 0..K-1:
    A_k       = softmax(Q_k @ f_rest_k^T / (√D × τ))      [B, L_k, L]
    C_k_base  = level_proj_k(A_k @ f_rest_k)               [B, L_k, D]
    R_k       = A_k^T @ C_k_base                           [B, L, D]
    f_hat_{k+1}  = f_hat_k  + R_k
    f_rest_{k+1} = f_rest_k - R_k

recon = back_proj(f_hat_K)                                  [B, L, D_enc]
L_recon = MSE(recon, H_CoT)
```

Code: `concept_hybrid_builder.py` L1082–1214, `train_builder.py` L138–155.

---

## Concern 1: Reconstruction Only Uses C_k_base, Not Refined C_k

### The Two Paths

At each pyramid level k, the code produces **two different concept vectors**
that diverge after a shared starting point:

**Path A — Base concept (enters reconstruction):**

```python
# concept_hybrid_builder.py L1131–1137
level_concepts_base = torch.bmm(level_attention, residual_hidden)
level_concepts_base = self.level_projs[level_idx](level_concepts_base)

# concept_hybrid_builder.py L1144–1146
reconstruction = torch.bmm(level_attention.transpose(1, 2), level_concepts_base)
```

This is the COMMIT path. `C_k_base` is scattered back via `A_k^T @ C_k_base` to
update the running reconstruction `f_hat` and residual `f_rest`.

**Path B — Refined concept (goes to reasoning):**

```python
# concept_hybrid_builder.py L1169–1185
if level_idx > 0:
    context = torch.cat([projected_hidden, prev_concepts_cat], dim=1)
    refined_concepts, _ = self.level_attn[level_idx](
        expanded_queries, context, context
    )
    level_concepts = level_concepts_base + refined_concepts
```

This is the REFINEMENT path. `C_k = C_k_base + refined_k` is the concept
that eventually gets sent to the reasoning/decoder pipeline. The cross-attention
reads from `[H_proj, C_0, ..., C_{k-1}]` to add context.

### The Divergence

The key point: **`refined_k` is never part of reconstruction.**

```
Reconstruction:  R_k = A_k^T @ C_k_base           ← only base
Reasoning uses:  C_k = C_k_base + refined_k        ← base + cross-attn output
```

The information added by cross-attention (`refined_k`) goes to the reasoning
pipeline but is completely invisible to `f_hat`, `f_rest`, and `recon_loss`.

### Numerical Walk-Through

Suppose level k=2, with L_k=4 concepts, D=896 (Qwen2.5-0.5B):

```
Step 1 — Pool and project (shared):
  raw_pooled = A_2 @ f_rest_2                # [B, 4, 896]
  C_2_base = level_proj_2(raw_pooled)        # [B, 4, 896]
  Example C_2_base[0,0] = [0.1, -0.5, 0.3, ..., 0.2]   (896 values)

Step 2a — Reconstruction (Path A):
  R_2 = A_2^T @ C_2_base                    # [B, L, 896]
  f_hat_3 = f_hat_2 + R_2
  → The reconstruction ONLY sees [0.1, -0.5, 0.3, ..., 0.2]

Step 2b — Refinement (Path B):
  context = [H_proj, C_0, C_1]              # [B, L+1+2, 896]
  refined_2 = CrossAttn(Q_2, context, context)  # [B, 4, 896]
  Example refined_2[0,0] = [0.2, -0.1, 0.8, ..., -0.3]

  C_2 = C_2_base + refined_2
  Example C_2[0,0] = [0.3, -0.6, 1.1, ..., -0.1]
  → The reasoning pipeline sees [0.3, -0.6, 1.1, ..., -0.1]
```

The two paths see **different vectors** for the same concept. The refinement
added `[0.2, -0.1, 0.8, ..., -0.3]` but this never enters `f_hat`.

### Consequence: Cross-Attention Gets Zero Gradient from recon_loss

Since `refined_k` is not part of the reconstruction path, the cross-attention
layers (`self.level_attn`) receive **zero gradient from `recon_loss`**.

Gradient flow:

```
recon_loss → back_proj → f_hat_K → Σ R_k → Σ A_k^T @ C_k_base
                                                ↓
                                           level_projs ✓
                                           concept_queries ✓
                                           temperature ✓
                                           input_proj ✓

                         level_attn (cross-attention) ← NO gradient from recon_loss
```

The cross-attention layers are ONLY trained by `reasoning_loss`. This means:

1. **If `reasoning_loss_weight = 0`** (as in most GSM8K configs currently):
   - `level_attn` gets zero gradients → weights stay at initialization
   - `refined_k` is effectively random noise
   - `C_k = C_k_base + noise` → the concepts sent to reasoning are corrupted
   - But `recon_loss` is unaffected (it only sees `C_k_base`)
   - **Result**: The pyramid learns to reconstruct well, but its concepts
     (which include refinement) are useless for downstream reasoning.

2. **If `reasoning_loss_weight > 0`**:
   - `level_attn` learns from reasoning supervision
   - `refined_k` becomes meaningful
   - But `recon_loss` still doesn't "know" about the refinement
   - **Result**: Two objectives train two different aspects of the same level —
     reconstruction trains `level_proj` and attention; reasoning trains
     `level_attn`. They share `C_k_base` but diverge after that.

### The Dual Personality

The pyramid has a **split brain**:

| Component                       | Trained by recon_loss | Trained by reasoning_loss           |
|---------------------------------|-----------------------|-------------------------------------|
| `input_proj`, `input_proj_norm` | Yes                   | Yes (through `back_proj(concepts)`) |
| `concept_queries`               | Yes                   | Yes                                 |
| `temperature`                   | Yes                   | Yes                                 |
| `level_projs`                   | Yes                   | Yes                                 |
| `level_attn` (cross-attention)  | **No**                | Yes                                 |
| `back_proj`                     | Yes                   | Yes                                 |

### Why This Design Exists

The commit-refinement separation was intentional (see hybrid-analysis.md
Section 2.3). The rationale:

- If refined concepts entered the residual flow (`f_rest`), cross-attention
  information would be scattered back and subtracted, potentially
  "double-counting" context that was already in `H_proj`.
- By keeping refinement OUT of `f_rest`, the residual decomposition stays
  clean: `f_hat + f_rest = H_proj` is an exact invariant.

### Open Questions

1. **Is the refinement actually helping?** If `reasoning_loss_weight = 0`,
   the cross-attention is wasted computation. If it IS used, how much does
   `refined_k` contribute vs. `C_k_base` alone?

2. **Should reconstruction also use refined concepts?** One could compute:
   `R_k = A_k^T @ C_k` (refined), but this breaks the `f_hat + f_rest = H_proj`
   invariant. Is that invariant more important than coherent gradient flow?

3. **Alternative: separate reconstruction and concept output entirely.**
   Instead of using the same `C_k_base` for both paths, one could have a
   dedicated reconstruction head and a separate reasoning head per level.

### How VAR Handles This: No Dual Path Problem

Let's examine the exact same question in VAR: does "next-scale construction"
(the analogue of our reconstruction) diverge from "Transformer prediction"
(the analogue of our reasoning)?

**Answer: No.** VAR's architecture fundamentally avoids this problem through
a **discrete bottleneck** design that couples the two paths.

#### VAR's Two Spaces

VAR operates in two distinct spaces:

| Space          | Dimension | What Lives Here                                    |
|----------------|-----------|----------------------------------------------------|
| **Cvae space** | 32        | Codebook embeddings, f_hat, f_rest, reconstruction |
| **C space**    | 1024      | Transformer hidden states, prediction logits       |

These are connected only through a **discrete bottleneck** (codebook indices).

#### VAR's Training Flow (Stage 2: Transformer)

Code reference: `var.py` L192–234, `quant.py` L169–184.

```
# Step 1: VQ-VAE (frozen) extracts ground truth
z = Encoder(image)                          [B, 32, 16, 16]  Cvae space
indices = Quantizer(z)                      [B, k×k] per scale  DISCRETE

# Step 2: Build teacher-forcing input
f_hat = 0
for k = 0..K-2:
    h = Codebook[indices[k]]                [B, Cvae, k, k]   Cvae space
    h = Φ_k(upsample(h))                    [B, Cvae, H, H]   Cvae space
    f_hat += h                              [B, Cvae, H, H]   Cvae space
    input_{k+1} = downsample(f_hat)         [B, Cvae, k', k'] Cvae space

# Step 3: Transformer predicts
x = word_embed(input)                       [B, L, C=1024]    C space
x = Transformer(x)                          [B, L, C=1024]    C space
logits = head(x)                            [B, L, V=4096]    C space

# Step 4: Loss
loss = CrossEntropy(logits, gt_indices)      scalar
```

#### VAR's Inference Flow

Code reference: `var.py` L127–190, `quant.py` L187–196.

```
f_hat = 0
for k = 0..K-1:
    x = word_embed(downsample(f_hat))       C space
    x = Transformer(x)                      C space
    logits = head(x)                        C space
    indices = sample(logits)                DISCRETE  ← the bridge
    h = Codebook[indices]                   Cvae space
    h = Φ_k(upsample(h))                   Cvae space
    f_hat += h                             Cvae space  ← reconstruction
```

#### Why VAR Has NO Dual Path Divergence

The crucial architectural difference:

```
VAR:
  Transformer (C space)  ──logits──→  sample()  ──indices──→  Codebook (Cvae space)  ──→  f_hat
                            C=1024       DISCRETE, V=4096        Cvae=32

Our Pyramid:
  C_k_base ──────────────────→ A_k^T @ C_k_base ──→ f_hat        (Path A: reconstruction)
       └──→ + CrossAttn ──→ C_k ─────────────────→ reasoning     (Path B: reasoning)
                                  SAME CONTINUOUS SPACE, D=896
```

**In VAR**, the Transformer operates in a completely separate space (C=1024)
from reconstruction (Cvae=32). The connection is through **discrete indices** —
the Transformer predicts a probability distribution over V=4096 codebook entries,
then the selected entries are looked up in the **same frozen codebook** that was
learned during VQ-VAE training. The Transformer's hidden states **never directly
enter f_hat**. Only the codebook embeddings do.

This means:
- There is exactly ONE representation for each scale: the codebook entry.
- The Transformer predicts WHICH entry to use (discrete choice).
- The reconstruction uses THAT SAME entry.
- No divergence possible.

**In our pyramid**, the concept `C_k_base` and the refined concept `C_k` are both
continuous vectors in the same D-dimensional space. The divergence happens within
the same space — `C_k = C_k_base + refined_k` is a different point in D-space from
`C_k_base`. Reconstruction uses `C_k_base`, reasoning uses `C_k`. They are two
different continuous representations with no discrete bottleneck to unify them.

#### The Role of Φ (phi) vs. Our level_proj

VAR's Φ is a **partial residual** (quant.py L199–206):

```python
def forward(self, h_BChw):
    return h_BChw.mul(1 - self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)
```

With `resi_ratio = 0.5`, this is:
```
Φ(h) = 0.5 × h + 0.5 × Conv(h)
```

Key properties:
- **Preserves most of the original signal** (50% passthrough)
- **Same Φ is used identically** in VQ-VAE training, teacher-forcing, and inference
- **No path divergence**: the Φ-transformed output goes to BOTH f_hat (reconstruction)
  AND becomes the next scale's input (via downsample(f_hat))

Our `level_proj` is an **unconstrained D×D linear layer** with no passthrough:
```
level_proj(h) = W @ h + b    # arbitrary rotation/scaling
```

- No signal preservation guarantee
- Only used in the reconstruction path (not in the reasoning path after refinement)

#### Concrete Side-by-Side Comparison: What Happens at Each Scale

**VAR Scale k**:
```
Quantizer:   f_rest → downsample → find nearest codebook entry → idx_k
                                                                  |
                                                                  v
             h_k = Codebook[idx_k]          ← SINGLE representation
             h_k = Φ(upsample(h_k))         ← same transform for both paths
             f_hat += h_k                   ← reconstruction uses h_k
             f_rest -= h_k                  ← residual uses h_k

Transformer: sees downsample(f_hat)         ← f_hat INCLUDES h_k
             predicts idx_{k+1}             ← predicts next scale's codebook entry
```

**Our Pyramid Level k**:
```
Pyramid:     f_rest → attention → pooled features
                                       |
                                       v
             C_k_base = level_proj(pooled)    ← transform
             R_k = A_k^T @ C_k_base           ← reconstruction uses C_k_base
             f_hat += R_k
             f_rest -= R_k

Refinement:  C_k = C_k_base + CrossAttn(...)  ← DIFFERENT representation
             C_k goes to reasoning pipeline    ← reasoning uses C_k ≠ C_k_base
```

The divergence is clear:
- **VAR**: ONE representation (codebook entry) serves BOTH reconstruction and
  downstream prediction. The Transformer predicts the next scale's representation
  based on the accumulated f_hat, which faithfully includes what was reconstructed.
- **Our Pyramid**: TWO representations (`C_k_base` for reconstruction, `C_k` for
  reasoning). The reasoning pipeline receives concepts that contain information
  (from cross-attention) that the reconstruction path has never seen.

#### What This Means for Architecture Design

VAR's discrete bottleneck serves as a **unifying constraint**: the codebook entry
IS the representation, period. It goes into f_hat, and f_hat feeds the next scale.
There's no room for divergence.

Our continuous architecture lacks this constraint. The cross-attention refinement
adds information that exists only in the reasoning path. This raises the question:

**Should we redesign so that the reconstruction path sees the SAME concepts
that the reasoning path uses?** This would mean:
- Either: reconstruction uses refined `C_k` (breaking the `f_hat + f_rest = H_proj`
  invariant)
- Or: reasoning uses only `C_k_base` (losing the benefit of cross-attention)
- Or: a VAR-like discrete bottleneck where concepts are quantized (losing
  gradient flow through the reconstruction path)

Each option has trade-offs. But the VAR comparison makes it clear that the
current dual-path design is a departure from the reference architecture's
philosophy of representational unity.

---

## Concern 2: level_proj Breaks Identity Reconstruction

### The Core Mechanism

At each level k, the per-level reconstruction is:

```python
# concept_hybrid_builder.py L1131–1146
C_k_base = level_proj_k(A_k @ f_rest_k)          # pool, then transform
R_k = A_k^T @ C_k_base                            # scatter back
```

Expanding: `R_k = A_k^T @ level_proj_k(A_k @ f_rest_k)`

### What Happens WITHOUT level_proj

If there were no `level_proj`, the reconstruction would be:

```
R_k = A_k^T @ (A_k @ f_rest_k) = (A_k^T @ A_k) @ f_rest_k
```

The matrix `P_k = A_k^T @ A_k` is an L×L matrix. It is a **projection operator**
— it projects `f_rest_k` onto the subspace "seen" by the L_k attention patterns.

**Example with L=8 tokens, L_k=2 concepts:**

Suppose the attention weights are clean (concept 0 attends to tokens 0-3,
concept 1 attends to tokens 4-7):

```
         tokens: t0   t1   t2   t3   t4   t5   t6   t7
A_k = [ [0.25 0.25 0.25 0.25 0.00 0.00 0.00 0.00],   ← concept 0
        [0.00 0.00 0.00 0.00 0.25 0.25 0.25 0.25] ]   ← concept 1
```

Then:
```
A_k @ f_rest_k:
  concept_0 = mean(f[t0], f[t1], f[t2], f[t3])     ← average of first half
  concept_1 = mean(f[t4], f[t5], f[t6], f[t7])     ← average of second half

A_k^T @ (A_k @ f_rest_k):
  R_k[t0] = 0.25 × concept_0 = 0.25 × mean(f[t0..t3])
  R_k[t1] = 0.25 × concept_0 = 0.25 × mean(f[t0..t3])
  ...
  R_k[t4] = 0.25 × concept_1 = 0.25 × mean(f[t4..t7])
  ...
```

Each token gets back the **average of its segment**, scaled by 0.25. This is
a projection — it captures the mean of each segment (the "global shape"), and
the within-segment variation is left in the residual.

After this level, `f_rest_{k+1} = f_rest_k - R_k` contains the **deviations
from segment means** — exactly the fine-grained detail that the next level
should capture. This is clean residual decomposition.

### What Happens WITH level_proj

With `level_proj` (a D×D linear layer), the reconstruction becomes:

```
R_k = A_k^T @ level_proj_k(A_k @ f_rest_k)
```

The `level_proj_k` transforms each concept BEFORE scattering it back.

**Continuing the example:**

```
raw concept_0 = mean(f[t0..t3]) = [0.1, -0.5, 0.3, ..., 0.2]

After level_proj:
  C_0_base = W_k @ raw_concept_0 + b_k
           = [5.0, -0.01, 0.0, ..., 8.3]       ← COMPLETELY DIFFERENT
```

The `level_proj` is a learned 896×896 matrix (for D=896). It can:
- Rotate the vector (change which directions have energy)
- Scale dimensions differently (amplify some, suppress others)
- Mix all dimensions together (each output is a linear combination of all inputs)

Then when we scatter back:
```
R_k[t0] = 0.25 × C_0_base = 0.25 × [5.0, -0.01, 0.0, ..., 8.3]
```

Token t0 gets back [5.0, ...] instead of the original segment-mean [0.1, ...].
The reconstruction `R_k` no longer resembles the original `f_rest_k`!

### Analogy: Lossy Image Compression with a Filter

Think of a 1D signal (e.g., audio):

```
Original signal: [1.0, 1.2, 0.9, 1.1, | 3.0, 2.8, 3.2, 3.1]
                  ←── segment A ──────→  ←── segment B ──────→
```

**Without level_proj**: Pool each segment, scatter back
```
Pool:        seg_A = 1.05,  seg_B = 3.025
Scatter:     [1.05, 1.05, 1.05, 1.05, 3.025, 3.025, 3.025, 3.025]
Residual:    [-0.05, 0.15, -0.15, 0.05, -0.025, -0.225, 0.175, 0.075]
```
The reconstruction captures the block means perfectly. The residual is the
within-block detail. This is a clean wavelet-like decomposition.

**With level_proj (e.g., multiply by 10)**:
```
Pool:        seg_A = 1.05,  seg_B = 3.025
Transform:   seg_A' = 10.5, seg_B' = 30.25
Scatter:     [10.5, 10.5, 10.5, 10.5, 30.25, 30.25, 30.25, 30.25]
```

Now `f_hat` has values ~10–30 while the original signal has values ~1–3.
The reconstruction is completely wrong, and the residual
`f_rest = original - f_hat` = [-9.5, -9.3, -9.6, -9.4, -27.25, ...]`
is **larger** than the original signal. The decomposition breaks down.

Of course, `level_proj` can also learn to be close to identity (scaling ≈ 1),
but nothing in the architecture guarantees this. The optimization must discover
this on its own through `recon_loss`.

### The Tension: Expressiveness vs. Reconstruction Fidelity

| Goal                       | What level_proj should do                                  |
|----------------------------|------------------------------------------------------------|
| Minimize recon_loss        | `level_proj ≈ Identity` (don't change the pooled features) |
| Maximize reasoning quality | `level_proj` = expressive transform (abstract features)    |

**Why level_proj helps reasoning:**

Raw pooled features `A_k @ f_rest_k` are just weighted averages of token
hidden states — they live in the same representational space as individual tokens.
This is not necessarily the best space for representing "concepts."

`level_proj` allows the model to learn a **concept-specific feature space**:
- Concept features can emphasize certain semantic directions
- Different dimensions can encode different concept properties
- The transform can learn to project away noise

**Why level_proj hurts reconstruction:**

For perfect reconstruction, we need `f_hat_K = H_proj`. Each `R_k` contributes
to `f_hat`. If `level_proj` transforms the pooled features, the scattered-back
`R_k` lives in a **different representational space** than `f_rest_k`. The
residual `f_rest_{k+1} = f_rest_k - R_k` is the difference between vectors
in two different spaces — it doesn't have a clean "remaining detail" interpretation.

### What This Means for Training

In practice, `recon_loss` will push `level_proj` toward something that makes
reconstruction work, while `reasoning_loss` (if active) will push it toward
something that makes concepts useful. The equilibrium depends on the loss
weight ratio `recon_loss_weight / reasoning_loss_weight`.

**Scenario 1: recon_loss_weight >> reasoning_loss_weight**
- `level_proj` learns to be close to identity or a scaled identity
- Reconstruction is good, but concepts are just weighted averages (not very abstract)

**Scenario 2: reasoning_loss_weight >> recon_loss_weight**
- `level_proj` learns expressive transforms
- Concepts are abstract and useful for reasoning, but reconstruction degrades
- Residual decomposition becomes noisy

**Scenario 3: Both weights balanced**
- `level_proj` must find a compromise
- The optimal compromise depends on the task

### Comparison with VAR: How Φ Preserves Residual Semantics

VAR's per-scale transform Φ (the closest analogue to our `level_proj`) is
designed to preserve residual decomposition semantics.

#### VAR's per-scale pipeline (quant.py L65–86)

```
for each scale k:
    rest_at_k = downsample(f_rest, k×k)     # extract residual at scale k
    idx = argmin ||rest_at_k - codebook||    # find nearest codebook entry
    h = Codebook[idx]                        # lookup (Cvae → Cvae, no dim change)
    h = upsample(h, H×H)                    # back to full resolution
    h = Φ(h)                                 # partial residual adjustment
    f_hat += h                               # accumulate
    f_rest -= h                              # subtract what was captured
```

The critical transform chain: `Codebook → upsample → Φ → goes to BOTH f_hat AND f_rest`.

#### Why Φ preserves residual semantics

Φ is defined as (quant.py L199–206):
```python
class Phi(nn.Conv2d):
    def forward(self, h_BChw):
        return h_BChw.mul(1 - self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)
```

With `resi_ratio = 0.5`:
```
Φ(h) = 0.5 × h + 0.5 × Conv3×3(h)
```

This has three key properties that `level_proj` lacks:

| Property            | Φ (VAR)                                                     | level_proj (Ours)                     |
|---------------------|-------------------------------------------------------------|---------------------------------------|
| Signal passthrough  | 50% of input passes through unchanged                       | 0% — fully transformed                |
| Transform type      | Conv2d (local, spatially smooth)                            | Dense linear (global, arbitrary)      |
| Shared across paths | Yes — same Φ(h) goes to f_hat, f_rest, and next-scale input | No — only enters f_hat, not reasoning |
| Dimension           | Cvae → Cvae (same dim, typically 32)                        | D → D (same dim, but unconstrained)   |

The 50% passthrough means `Φ(h)` is always close to `h`. The residual
`f_rest -= Φ(h)` still means "what remains after removing something close to
the codebook entry at this scale" — a semantically meaningful residual.

With our `level_proj`, the output can be arbitrarily far from the input.
`f_rest -= A_k^T @ level_proj(A_k @ f_rest)` subtracts something that may
have been completely rotated — the residual loses its "remaining detail" meaning.

#### The quant_conv is also same-dimension

VAR's `quant_conv` (vqvae.py L48) is:
```python
self.quant_conv = nn.Conv2d(self.Cvae, self.Cvae, ks, stride=1, padding=ks//2)
```

`in_channels == out_channels == Cvae = 32`. This is the encoder-to-quantizer
projection — it transforms features before quantization, but stays in the
same 32-dim space. No lossy dimension reduction.

### Possible Mitigations

1. **Initialize level_proj as identity:** Set `level_proj_k.weight = I`,
   `level_proj_k.bias = 0`. This starts with perfect reconstruction and lets
   training gradually deviate as needed for reasoning.

2. **Regularize level_proj toward identity:** Add a penalty
   `||W_k - I||^2` to keep it close to identity during training.

3. **Remove level_proj entirely:** Use raw pooled features for both
   reconstruction and reasoning. Simpler, guarantees cleaner residual
   decomposition, but less expressive concepts.

4. **Separate reconstruction and concept transforms:** Have two branches:
   - `C_k_recon = A_k @ f_rest_k` (no transform, for reconstruction)
   - `C_k_concept = level_proj_k(A_k @ f_rest_k)` (transformed, for reasoning)
   - `R_k = A_k^T @ C_k_recon` (clean reconstruction)
   - This decouples the two objectives completely.

---

## Summary

| Concern                                     | Effect                                                                                   | VAR's Solution                                                                                                  | Severity                           |
|---------------------------------------------|------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|------------------------------------|
| Cross-attention invisible to recon_loss     | `level_attn` gets no gradient when `reasoning_loss_weight=0`; refined concepts are noise | VAR has no dual path — discrete bottleneck ensures ONE representation serves both reconstruction and prediction | High when reasoning_loss disabled  |
| `level_proj` breaks identity reconstruction | Residual decomposition becomes semantically incoherent; recon_loss has non-zero floor    | VAR uses Φ with 50% passthrough, preserving residual semantics                                                  | Medium — optimization can mitigate |

Both concerns stem from the same root: **the reconstruction path and the
reasoning path share components but diverge**, creating tension between
reconstruction fidelity and concept quality.

The VAR comparison reveals that our dual-path continuous design is a fundamental
departure from VAR's philosophy of **representational unity through discrete
bottleneck**. In VAR, the codebook entry IS the representation — it goes into
f_hat, feeds the next scale, and is what the Transformer learns to predict.
There is no room for divergence. Our architecture allows the concept sent to
reasoning (`C_k`) to differ arbitrarily from the concept used for reconstruction
(`C_k_base`), breaking this unity.
