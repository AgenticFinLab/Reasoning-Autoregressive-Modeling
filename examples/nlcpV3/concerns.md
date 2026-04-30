# Concept Pyramid Reconstruction: Architectural Concerns

This document discusses architectural concerns in the concept pyramid,
framed in the context of the **two-stage training architecture** (analogous to VAR):

- **Stage 1 (current):** Train Builder — extract groundtruth concept pyramid
  from CoT using reconstruction loss. Cross-attention refinement exists in
  code but is NOT trained (reasoning_loss_weight = 0).
- **Stage 2 (future):** Train Predictor — a causal transformer predicts
  next-level concepts given previous levels (not yet implemented).

## Background: The Reconstruction Path (Stage 1)

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

    # Cross-attention refinement (exists but untrained in Stage 1):
    if k > 0:
        refined_k = CrossAttn(Q_k, [H_proj, C_0..C_{k-1}])
        C_k = C_k_base + refined_k              ← stored in PyramidOutput.concepts
    else:
        C_k = C_k_base

recon = back_proj(f_hat_K)                                  [B, L, D_enc]
L_recon = MSE(recon, H_CoT)
```

Code: `concept_hybrid_builder.py` L1082–1234, `train_builder.py` L138–155.

---

## Concern 1: Cross-Attention Is Dead in Stage 1, Contaminates Output for Stage 2

### The Problem: Two Stages, But Cross-Attention Belongs to Neither

The cross-attention refinement (`level_attn`) exists in the builder code, but:
- **Stage 1 (recon_loss)** does not train it — zero gradient (explained below)
- **Stage 2 (predictor)** is not yet implemented

So the cross-attention layers are dead weight in Stage 1, producing untrained
noise that contaminates the builder's output concepts.

### Why Cross-Attention Gets ZERO Gradient in Stage 1

To understand why, we need to trace the computation graph from `recon_loss`
back to the cross-attention parameters.

**The computation graph:**

```
                    ┌────────────────────────────────────────────────────────┐
                    │               RECON_LOSS COMPUTATION GRAPH              │
                    │                                                        │
  input_proj ──→ H_proj ──→ f_rest_0                                        │
                              │                                              │
                              ▼                                              │
  concept_queries ──→ A_k = softmax(Q_k @ f_rest_k^T / (√D × τ))           │
  temperature ───────┘        │                                              │
                              ▼                                              │
  level_projs ──→ C_k_base = level_proj(A_k @ f_rest_k)                     │
                              │                                              │
                              ├──→ R_k = A_k^T @ C_k_base ──→ f_hat ──→ ··· │
                              │    (enters recon_loss)                       │
                              │                                              │
                              └──→ C_k = C_k_base + CrossAttn(...)           │
                                   (stored in output, but NOT in recon_loss) │
                    └────────────────────────────────────────────────────────┘
```

The path from `recon_loss` backwards:
```
recon_loss → back_proj → f_hat_K → Σ R_k → Σ A_k^T @ C_k_base → level_projs ✓
                                                                → concept_queries ✓
                                                                → temperature ✓
                                                                → input_proj ✓
```

The cross-attention computation `refined_k = level_attn(Q, context, context)` is
a **dead branch** — its output `refined_k` is added to `C_k_base` to form `C_k`,
but `C_k` only goes into `PyramidOutput.concepts`. It does NOT flow into `R_k`,
`f_hat`, or `recon_loss`. The computation graph has no path from `recon_loss`
back to `level_attn` parameters.

### Concrete Gradient Example: Why Zero Gradient Means No Learning

Let's trace a minimal example to make "zero gradient" concrete.

Consider a simple 2-parameter network:
```python
W_proj = nn.Parameter(torch.tensor(2.0))    # level_proj (trained by recon_loss)
W_attn = nn.Parameter(torch.tensor(3.0))    # level_attn (NOT trained by recon_loss)

x = torch.tensor(1.0)                       # input (f_rest)

# Path A: reconstruction
C_base = W_proj * x                         # C_base = 2.0
R = C_base                                  # R = 2.0 (scatter back, simplified)
f_hat = R                                   # f_hat = 2.0

# Path B: refinement (dead branch for recon_loss)
refined = W_attn * x                        # refined = 3.0
C_k = C_base + refined                      # C_k = 5.0 (stored in output)

# Loss: only uses f_hat
target = torch.tensor(1.5)
loss = (f_hat - target) ** 2                 # (2.0 - 1.5)^2 = 0.25

loss.backward()
```

After `.backward()`:
```
W_proj.grad = d(loss)/d(W_proj)
            = d(loss)/d(f_hat) × d(f_hat)/d(R) × d(R)/d(C_base) × d(C_base)/d(W_proj)
            = 2×(2.0-1.5) × 1 × 1 × 1.0
            = 1.0  ← NON-ZERO, W_proj gets updated ✓

W_attn.grad = d(loss)/d(W_attn)
            = ??? There is NO path from loss to W_attn!
            = 0.0  ← ZERO, W_attn stays at initialization ✗
```

PyTorch's autograd traces the computation graph from the loss backwards.
Since `loss` depends on `f_hat → R → C_base → W_proj`, the gradient flows
to `W_proj`. But `loss` does NOT depend on `C_k` or `refined` or `W_attn` —
`C_k` is computed but never used in the loss. So `W_attn.grad` is exactly 0.

**After 1000 training steps:**
- `W_proj` has been updated 1000 times → learned to minimize recon_loss
- `W_attn` is STILL at its initialization value → its output is random noise

This is exactly what happens to our `level_attn` in Stage 1.

### The Impact on Stage 2

The builder returns `PyramidOutput.concepts = [C_0, C_1, ..., C_{K-1}]`
(code: `concept_hybrid_builder.py` L1222–1223):

```python
return PyramidOutput(
    concepts=all_level_concepts,  # [C_0, ..., C_{K-1}]
```

Where `C_k = C_k_base + refined_k` for k > 0 (L1185).

After Stage 1 training:
```
C_0 = C_0_base                              ← clean (no cross-attn at level 0)
C_1 = C_1_base + untrained_noise            ← contaminated
C_2 = C_2_base + untrained_noise            ← contaminated
C_3 = C_3_base + untrained_noise            ← contaminated
...
```

When Stage 2 (the causal predictor) uses these as ground truth targets,
it would be learning to predict `C_k_base + noise`. The predictor would
waste capacity trying to model random noise that carries no information.

### How VAR Avoids This: Clean Stage 1 Output

In VAR's Stage 1 (VQ-VAE training), the output is **codebook indices** —
clean, discrete, fully trained:

```
indices[0] = [42]              ← 1 codebook entry, chosen by argmin ||f_rest - codebook||
indices[1] = [7, 13, 91, 5]   ← 4 codebook entries, chosen the same way
...
```

Every index was selected by a supervised operation (nearest-neighbor lookup
against f_rest). No untrained component is added. When Stage 2 (Transformer)
uses these indices as ground truth targets, they are clean and meaningful.

There is NO "extra transform" in VAR's quantizer output that wasn't trained
by the VQ loss. Every part of the representation that Stage 2 sees was
optimized during Stage 1.

### The Design Question

The concern simplifies to: **What should the Builder output for Stage 2?**

| Option                                 | Stage 2 predicts                | Pro                                       | Con                                |
|----------------------------------------|---------------------------------|-------------------------------------------|------------------------------------|
| A: Output `C_k_base` only              | Reconstruction-trained concepts | Clean, fully supervised by recon_loss     | No cross-attention enrichment      |
| B: Output `C_k = C_k_base + refined_k` | Base + noise (Stage 1)          | Richer IF cross-attn is trained           | Contaminated when reasoning_loss=0 |
| C: Remove cross-attention entirely     | Just `C_k_base`                 | Simpler, no dead parameters, clean output | Less flexible architecture         |

If Stage 2's predictor should predict `C_k_base` (not the refined version),
then the cross-attention layers are unnecessary in Stage 1 and can be removed.

### VAR Comparison: Two-Stage Architecture

Our architecture mirrors VAR's two-stage design:

|                    | VAR                                                      | Our Concept Pyramid                                   |
|--------------------|----------------------------------------------------------|-------------------------------------------------------|
| **Stage 1**        | Train VQ-VAE (encoder → multi-scale quantizer → decoder) | Train Builder (encoder → concept pyramid → back_proj) |
| **Stage 1 Loss**   | Image reconstruction + VQ loss                           | `recon_loss` (MSE against frozen encoder output)      |
| **Stage 1 Output** | Codebook indices per scale                               | Concepts per level                                    |
| **Stage 2**        | Train Transformer (predict next-scale codebook indices)  | Train Predictor (predict next-level concepts)         |
| **Stage 2 Input**  | Clean codebook indices from frozen VQ-VAE                | ??? from frozen Builder                               |

The critical difference is what Stage 1 hands to Stage 2:

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

#### Why VAR's Stage 1 Output Is Clean

The crucial point: in VAR's Stage 1, **everything in the output was trained
by the Stage 1 loss**. The codebook entries were optimized by VQ loss +
reconstruction loss. The Φ transform was trained by reconstruction loss.
There is no component in the output that wasn't supervised.

In our Stage 1, the output `C_k = C_k_base + refined_k` contains `refined_k`
which was **never trained** (zero gradient from recon_loss). This is the
contamination problem.

#### Architecture Comparison

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

#### Implication for Two-Stage Design

VAR's discrete bottleneck serves as a **unifying constraint**: the codebook entry
IS the representation, period. Stage 1 trains it, Stage 2 predicts it. Clean.

Our architecture must decide: **What is the Stage 2 target?**
- If `C_k_base` → remove cross-attention from builder (it's dead weight in Stage 1)
- If `C_k = C_k_base + refined_k` → must train cross-attention in Stage 1
  (requires reasoning_loss_weight > 0 or a new supervision signal)
- If we add a discrete bottleneck (quantize concepts) → closer to VAR but
  loses gradient flow through reconstruction

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

| Concern                         | Stage 1 Impact                                    | Stage 2 Impact                                                                | VAR's Solution                                                       |
|---------------------------------|---------------------------------------------------|-------------------------------------------------------------------------------|----------------------------------------------------------------------|
| Cross-attention dead in Stage 1 | None — recon_loss unaffected                      | **High** — output concepts contain untrained noise, poisoning Stage 2 targets | No untrained components in output; codebook entries fully supervised |
| `level_proj` breaks identity    | Medium — optimization must discover near-identity | Medium — concepts are in a transformed space                                  | Φ with 50% passthrough preserves residual semantics                  |

Both concerns are about **what Stage 1 produces for Stage 2**:

- Concern 1: The builder output contains untrained noise (`refined_k`).
  **Fix**: Either remove cross-attention from the builder, or only output
  `C_k_base` for Stage 2.
- Concern 2: The builder output is in a space that may not preserve
  residual decomposition semantics. **Fix**: Initialize `level_proj` as
  identity, or add passthrough like VAR's Φ.

The core principle from VAR: **every component of the Stage 1 output that
Stage 2 will use must be fully supervised by the Stage 1 loss.**
