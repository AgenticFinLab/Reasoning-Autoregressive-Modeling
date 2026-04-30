# Concept Pyramid Builder: Architectural Concerns

This document discusses architectural concerns in the concept pyramid builder,
framed in the context of the **two-stage training architecture** (analogous to VAR):

- **Stage 1 (current):** Train Builder — extract groundtruth concept pyramid
  from CoT. The builder is **purely residual** (like VAR's VQ-VAE). Stage 1
  has two objectives that both operate on the same purely-residual concepts:
  - `recon_loss`: Can the pyramid reconstruct the CoT? (faithfulness)
  - `reasoning_loss`: Can the pyramid guide reasoning to produce the Solution?
    (utility — ensures the ground truth pyramid is a good target for Stage 2)
- **Stage 2 (future):** Train Predictor — a causal transformer predicts
  next-level concepts given previous levels (not yet implemented).
  "Condition on previous levels" belongs here, NOT in Stage 1.

## Background: The Correct Stage 1 Pipeline (Purely Residual)

Following VAR's VQ-VAE design, the builder should be purely residual:

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
    output C_k = C_k_base                                   ← NO refinement

recon = back_proj(f_hat_K)                                  [B, L, D_enc]
L_recon = MSE(recon, H_CoT)
L_reason = NTP(back_proj(concepts) → decoder → Solution)    ← optional
```

Both losses train the SAME purely-residual parameters (`input_proj`,
`level_projs`, `concept_queries`, `temperature`, `back_proj`).
No cross-scale conditioning. No cross-attention refinement.

Code: `concept_hybrid_builder.py` L1082–1234, `train_builder.py` L138–155.

---

## Concern 1: Cross-Attention Refinement Does Not Belong in the Builder

### The Design Error

The builder currently contains cross-attention layers (`level_attn`) that
condition each level's concept on previous levels' concepts:

```python
# concept_hybrid_builder.py L1169–1185
if level_idx > 0:
    context = torch.cat([projected_hidden, prev_concepts_cat], dim=1)
    refined_concepts, _ = self.level_attn[level_idx](
        expanded_queries, context, context
    )
    level_concepts = level_concepts_base + refined_concepts
```

This is wrong. "Condition current level on previous levels" is a Stage 2
operation. The builder (Stage 1) should be **purely residual** — each level
only sees the current residual `f_rest`, nothing else.

### VAR's Stage 1: Purely Residual, No Cross-Scale Conditioning

In VAR's VQ-VAE (Stage 1), each scale operates ONLY on `f_rest`:

```
f_rest = z.clone()        ← encoder output
f_hat = zeros

for scale k in [1, 2, 4, 8, 16, 32]:
    Step 1: downsample f_rest to k×k
    Step 2: find nearest codebook entry     ← ONLY looks at f_rest
    Step 3: lookup codebook vector
    Step 4: upsample to full resolution
    Step 5: apply Φ (partial residual)
    Step 6: f_hat += h,  f_rest -= h
```

(Reference: `docs/VAR.md` VQ-VAE Training Flow, `quant.py` L65–86)

No scale looks at what previous scales produced. No cross-scale attention.
The point of Stage 1 is to decompose the signal into independent layers
of residual detail — coarse first, then finer.

"Condition on previous scales" appears **ONLY in Stage 2** (the Transformer),
where the model sees `[scale_0, ..., scale_{k-1}]` and predicts `scale_k`.

### Why reasoning_loss Does Not Justify Cross-Attention

Stage 1 has two losses: `recon_loss` and `reasoning_loss`. Both exist to
make `C_k_base` better — not to add new components.

- `recon_loss` pushes `C_k_base` toward faithful CoT coverage
- `reasoning_loss` pushes `C_k_base` toward reasoning utility

Both train the **same purely-residual parameters**: `input_proj`,
`level_projs`, `concept_queries`, `temperature`, `back_proj`. The dual
objective makes the residual decomposition itself produce concepts that
are inherently useful for both reconstruction AND reasoning.

The cross-attention refinement is not needed for this. Adding `refined_k`
is not "making the decomposition better" — it is adding cross-scale
conditioning that fundamentally doesn't belong in Stage 1.

### What's Wrong in the Current Code

| Component                                         | Correct Role | Current Code |
|---------------------------------------------------|--------------|--------------|
| `C_k_base` (residual decomposition)               | Stage 1 ✓    | ✓            |
| `R_k = A_k^T @ C_k_base` (reconstruction)         | Stage 1 ✓    | ✓            |
| `f_hat += R_k`, `f_rest -= R_k` (residual update) | Stage 1 ✓    | ✓            |
| `level_attn` (condition on previous levels)       | **Stage 2**  | ✗ In builder |
| `C_k = C_k_base + refined_k` (cross-scale output) | **Stage 2**  | ✗ In builder |

### Resolution

Remove `level_attn` and `refined_k` from the builder entirely:

```python
# CORRECT: purely residual, no cross-scale conditioning
all_level_concepts.append(level_concepts_base)   # C_k_base only
```

The builder outputs `C_k_base` — purely from residual decomposition.
Stage 2's predictor will condition on previous levels when predicting
next-level concepts, which is where that logic belongs.

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

| Concern                      | Root Cause                                                                | VAR's Approach                                          | Resolution                                               |
|------------------------------|---------------------------------------------------------------------------|---------------------------------------------------------|----------------------------------------------------------|
| Cross-attention in Stage 1   | Stage 2 mechanism (condition on previous levels) placed in Stage 1 module | Stage 1 is purely residual; no cross-scale conditioning | Remove `level_attn` from builder; output `C_k_base` only |
| `level_proj` breaks identity | Unconstrained linear transform before scatter-back                        | Φ with 50% passthrough preserves residual semantics     | Initialize as identity or add passthrough                |

Core principle from VAR: **Stage 1 is purely residual decomposition —
no cross-scale conditioning. Stage 1's dual objectives (recon + reasoning)
both improve the same residual concepts. Cross-scale conditioning belongs
in Stage 2 (the predictor).**
