# VAR: Visual Autoregressive Modeling

## Overview

VAR (Visual Autoregressive Modeling) is a novel image generation approach that applies **next-scale autoregression** instead of traditional next-token autoregression.

**Key Innovation**: Generate image scale-by-scale (coarse-to-fine), not pixel-by-pixel or token-by-token.

---

## Architecture Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        VAR Architecture                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  VQ-VAE  │    │ Quantizer│    │Transformer│    │  Head    │  │
│  │ (Encoder)│───►│(Multi-   │───►│ (GPT-like)│───►│(Predictor│  │
│  │          │    │ Scale)   │    │           │    │  Logits) │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. VQ-VAE Encoder (basic_vae.py, vqvae.py)

**Purpose**: Compress image into discrete latent codes

```
Input Image [B, 3, H, W]
        │
        ▼
┌───────────────────┐
│ Conv Encoder      │  Downsampling: H×W → H/16 × W/16
│ (4 stages)        │
└─────────┬─────────┘
          │
          ▼
Latent Feature [B, C, H/16, W/16]
        │
        ▼
┌───────────────────┐
│ Quantizer         │  Multi-scale quantization
│ (see below)       │
└─────────┬─────────┘
          │
          ▼
Indices per Scale: [idx_0, idx_1, ..., idx_K]
```

### 2. Multi-Scale Quantizer (quant.py)

**Purpose**: Convert continuous features to discrete indices at multiple scales

```
Latent Feature [B, C, H, W]  (e.g., H=W=16 for 256×256 image)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Scale Quantization                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  f_rest = z.clone()                                          │
│  f_hat = zeros                                               │
│                                                              │
│  for scale k in [1, 2, 4, 8, 16, 32]:                       │
│      ┌─────────────────────────────────────────────────┐    │
│      │ Step 1: Downsample f_rest to k×k                 │    │
│      │         f_rest [B,C,16,16] → [B,C,k,k]           │    │
│      ├─────────────────────────────────────────────────┤    │
│      │ Step 2: Find nearest codebook entry              │    │
│      │         distances = ||f_rest - codebook||²       │    │
│      │         indices[k] = argmin(distances)           │    │
│      ├─────────────────────────────────────────────────┤    │
│      │ Step 3: Lookup codebook                          │    │
│      │         h_k = codebook[indices[k]]               │    │
│      ├─────────────────────────────────────────────────┤    │
│      │ Step 4: Apply φ (residual learning)              │    │
│      │         h_k = φ_k(h_k) * 0.5 + h_k * 0.5         │    │
│      ├─────────────────────────────────────────────────┤    │
│      │ Step 5: Upsample to full resolution              │    │
│      │         h_k_up = bicubic_upsample(h_k, 16×16)    │    │
│      ├─────────────────────────────────────────────────┤    │
│      │ Step 6: Accumulate and update                    │    │
│      │         f_hat += h_k_up                          │    │
│      │         f_rest -= h_k_up                         │    │
│      └─────────────────────────────────────────────────┘    │
│                                                              │
│  VQ Loss = β||f_hat - z||² + ||f_hat - sg[z]||²            │
│  STE: f_hat = (f_hat.detach() - z.detach()) + z            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
Output:
  - f_hat [B, C, H, W]        (quantized features)
  - indices_per_scale: List[[B,k×k] for k in scales]
  - vq_loss (scalar)
```

**Scale Configuration** (for 256×256 image):
```
Scale 0:  1×1   = 1    token   (global structure)
Scale 1:  2×2   = 4    tokens  (coarse structure)
Scale 2:  4×4   = 16   tokens  (medium structure)
Scale 3:  8×8   = 64   tokens  (fine structure)
Scale 4:  16×16 = 256  tokens  (fine details)
Scale 5:  32×32 = 1024 tokens  (finest details)
─────────────────────────────────────────────
Total: 1365 tokens (vs 65536 pixels)
```

### 3. Transformer (var.py, basic_var.py)

**Purpose**: Autoregressively predict next-scale indices

```
┌─────────────────────────────────────────────────────────────┐
│                    VAR Transformer                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input:                                                      │
│    - class_emb [B, D]           (class conditioning)        │
│    - prev_scale_tokens [B, L, D] (previous scale indices)   │
│                                                              │
│  Architecture:                                               │
│    - Embedding layer: indices → tokens                      │
│    - Position embedding (per-scale)                         │
│    - Level embedding (distinguish scales)                   │
│    - AdaLN Transformer blocks (class-conditional)           │
│    - Prediction head: tokens → logits                       │
│                                                              │
│  Key Components:                                             │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ AdaLNSelfAttn Block                                  │    │
│  │                                                       │    │
│  │  x ──► LayerNorm ──► Self-Attention ──► + ──► x      │    │
│  │         ↑              ↑             ↑               │    │
│  │         └──────── AdaLN(cond) ───────┘               │    │
│  │                                                       │    │
│  │  x ──► LayerNorm ──► FFN ──► + ──► x                 │    │
│  │         ↑              ↑                             │    │
│  │         └──────── AdaLN(cond) ───────┘               │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  AdaLN (Adaptive Layer Norm):                               │
│    scale, shift = Linear(cond)                              │
│    out = LayerNorm(x) * (1 + scale) + shift                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4. Attention Mask (Causal over Scales)

```
┌─────────────────────────────────────────────────────────────┐
│                 Causal Attention Mask                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Scale indices: [0, 1,1, 2,2,2,2, 3,3,3,3,3,3,3,3, ...]     │
│                                                              │
│  Mask matrix (L×L):                                          │
│                                                              │
│         scale:  0  1  2  3  ...                              │
│                 ↓  ↓  ↓  ↓                                   │
│         pos:   0 12 3456 7890...                             │
│                ┌─────────────────┐                           │
│        0 (s0) │1 0 0 0 0 0 0 0 │  ← scale 0 sees only 0    │
│        1 (s1) │1 1 0 0 0 0 0 0 │  ← scale 1 sees 0,1       │
│        2 (s1) │1 1 1 0 0 0 0 0 │                            │
│        3 (s2) │1 1 1 1 1 0 0 0 │  ← scale 2 sees 0,1,2     │
│        4 (s2) │1 1 1 1 1 0 0 0 │                            │
│        5 (s2) │1 1 1 1 1 0 0 0 │                            │
│        6 (s2) │1 1 1 1 1 0 0 0 │                            │
│        ...    │...              │                            │
│                └─────────────────┘                           │
│                                                              │
│  Code (var.py line 107-112):                                │
│    d = [0, 1,1, 2,2,2,2, ...]  # scale index per position  │
│    attn_mask = (d >= d.T) ? 0 : -inf                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Training Flow

### Two-Stage Training

```
┌─────────────────────────────────────────────────────────────┐
│                  Stage 1: VQ-VAE Training                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Image [B, 3, H, W]                                         │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────┐                                            │
│  │ Encoder     │                                            │
│  └──────┬──────┘                                            │
│         │ z [B, C, H', W']                                  │
│         ▼                                                    │
│  ┌─────────────┐                                            │
│  │ Quantizer   │──► indices_per_scale                       │
│  └──────┬──────┘                                            │
│         │ f_hat [B, C, H', W']                              │
│         ▼                                                    │
│  ┌─────────────┐                                            │
│  │ Decoder     │                                            │
│  └──────┬──────┘                                            │
│         │                                                    │
│         ▼                                                    │
│  Reconstructed Image [B, 3, H, W]                           │
│         │                                                    │
│         ▼                                                    │
│  Loss = L_recon + λ * L_vq                                  │
│         = ||img - img_rec||² + λ * (β||sg[z]-q||² + ||z-sg[q]||²) │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  Stage 2: VAR Transformer Training           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Image + Class Label                                        │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────┐                                            │
│  │ Frozen      │                                            │
│  │ VQ-VAE      │──► indices_per_scale                       │
│  │ Encoder     │    (DISCRETE integers)                     │
│  └─────────────┘                                            │
│         │                                                    │
│         │  indices_per_scale: List[[B, k×k]] integers      │
│         ▼                                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ CRITICAL: indices → embeddings conversion           │    │
│  │ ═════════════════════════════════════════════════    │    │
│  │                                                       │    │
│  │ indices[0:k-1]  ──► codebook.embedding               │    │
│  │ (DISCRETE)            │                               │    │
│  │                       ▼                               │    │
│  │              embeddings [B, L, Cvae]                 │    │
│  │              (CONTINUOUS float)                      │    │
│  │                       │                               │    │
│  │                       ▼                               │    │
│  │              word_embed (Linear)                     │    │
│  │                       │                               │    │
│  │                       ▼                               │    │
│  │              features [B, L, C]                      │    │
│  │              (CONTINUOUS float)                      │    │
│  │                                                       │    │
│  └─────────────────────────────────────────────────────┘    │
│         │                                                    │
│         │  + class_emb [B, C] + pos_emb + lvl_emb            │
│         ▼                                                    │
│  ┌─────────────┐                                            │
│  │ Transformer │                                            │
│  │ (with causal│                                            │
│  │  mask)      │                                            │
│  └──────┬──────┘                                            │
│         │                                                    │
│         ▼                                                    │
│  Logits [B, L, V]  (V = codebook_size)                      │
│         │                                                    │
│         ▼                                                    │
│  Loss = CrossEntropy(logits, target_indices)                │
│                                                              │
│  where target_indices = indices[k] (current scale)          │
│  NOTE: target_indices are DISCRETE, logits are CONTINUOUS   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### CRITICAL: Indices → Embeddings Conversion

```
┌─────────────────────────────────────────────────────────────────────┐
│     Transformer NEVER sees discrete indices, ONLY embeddings!       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Two-Step Embedding Process:                                         │
│                                                                      │
│  Step 1: Codebook Lookup (quant.py line 180)                        │
│  ─────────────────────────────────────────                          │
│  indices [B, k×k]  ──►  codebook.embedding(indices)                 │
│       │                        │                                     │
│       │                        ▼                                     │
│       │              embeddings [B, k×k, Cvae]                       │
│       │                  (e.g., Cvae = 32)                           │
│       │                                                              │
│       ▼                                                              │
│  Step 2: Word Embedding Projection (var.py line 206)               │
│  ────────────────────────────────────────────────                   │
│  embeddings [B, L, Cvae]  ──►  word_embed (Linear)                  │
│       │                            │                                 │
│       │                            ▼                                 │
│       │                   tokens [B, L, C]                           │
│       │                    (e.g., C = 1024)                          │
│       │                                                              │
│       ▼                                                              │
│  Step 3: Add Position + Level Embedding                             │
│  ─────────────────────────────────────                              │
│  tokens [B, L, C] + pos_emb + lvl_emb                               │
│       │                                                              │
│       ▼                                                              │
│  Transformer Input [B, L, C]  ← THIS IS WHAT TRANSFORMER SEES       │
│                                                                      │
│  ═══════════════════════════════════════════════════════════════    │
│  NOTE: Cvae ≠ C  (different dimensions!)                            │
│        Cvae = codebook dimension (e.g., 32)                         │
│        C = transformer hidden dimension (e.g., 1024)                │
│        word_embed = nn.Linear(Cvae, C)  ← learnable projection      │
│  ═══════════════════════════════════════════════════════════════    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Training Code Flow (train.py, trainer.py)

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Loop                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  for epoch in epochs:                                        │
│      for batch in dataloader:                               │
│          images, labels = batch                              │
│                                                              │
│          # Step 1: Get ground truth indices (discrete)       │
│          with torch.no_grad():                              │
│              z = vae.encoder(images)                        │
│              indices_per_scale = quantizer.f_to_idxBl(z)    │
│              # indices_per_scale: List of [B, k×k] tensors  │
│              # each tensor contains DISCRETE integers 0~V-1 │
│                                                              │
│          # Step 2: Convert indices → embeddings              │
│          # idxBl_to_var_input does:                          │
│          #   indices → codebook.lookup → upsample → f_hat   │
│          #   Returns CONTINUOUS features [B, L, Cvae]       │
│          tf_input = quantizer.idxBl_to_var_input(           │
│              indices_per_scale                               │
│          )                                                   │
│          # tf_input: [B, L, Cvae] CONTINUOUS features       │
│                                                              │
│          # Step 3: Forward pass                              │
│          # Inside var_model:                                 │
│          #   tf_input [B,L,Cvae] → word_embed → [B,L,C]     │
│          #   Then add pos_emb, lvl_emb                      │
│          #   Then transformer forward                        │
│          logits = var_model(labels, tf_input)               │
│          # logits: [B, L, V] predictions for all scales     │
│                                                              │
│          # Step 4: Compute loss                              │
│          loss = 0                                            │
│          for k, idx_k in enumerate(indices_per_scale):      │
│              # Get logits for scale k                        │
│              start, end = scale_ranges[k]                   │
│              logits_k = logits[:, start:end, :]             │
│              # Cross entropy loss (against DISCRETE indices)│
│              loss += CE(logits_k, idx_k)                    │
│                                                              │
│          # Step 5: Backward                                  │
│          loss.backward()                                     │
│          optimizer.step()                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Inference Flow

### Autoregressive Generation (var.py: autoregressive_infer_cfg)

```
┌─────────────────────────────────────────────────────────────┐
│                 VAR Inference (Generation)                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: class_label (e.g., "golden retriever" = 207)        │
│                                                              │
│  Step 0: Initialize                                          │
│  ────────────────────────────────────────────────           │
│  class_emb = class_embedding[class_label]  # [B, D]         │
│  f_hat = zeros [B, C, H, W]                                 │
│  kv_cache = empty                                           │
│                                                              │
│  for scale_idx, pn in enumerate([1,2,4,8,16,32]):          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Step 1: Prepare input for current scale              │    │
│  │                                                       │    │
│  │ if scale_idx == 0:                                   │    │
│  │     x = class_emb + pos_start  # [B, 1, C]           │    │
│  │     # First scale: only class embedding              │    │
│  │ else:                                                 │    │
│  │     # Use PREVIOUS scale's f_hat (CONTINUOUS!)       │    │
│  │     # f_hat was accumulated from previous scales     │    │
│  │     # Downsample f_hat to next scale size            │    │
│  │     x = downsample(f_hat, pn_next, pn_next)          │    │
│  │     x = x.view(B, C, -1).transpose(1,2)  # [B,pn²,Cvae]│   │
│  │     x = word_embed(x)  # [B, pn², C] ← PROJECT!      │    │
│  │     x = x + pos_emb + lvl_emb                         │    │
│  │                                                       │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │ Step 2: Transformer forward (with KV cache)          │    │
│  │                                                       │    │
│  │ for block in transformer_blocks:                     │    │
│  │     x = block(x, cond=class_emb, attn_bias=None)    │    │
│  │     # KV cache enables efficient inference           │    │
│  │                                                       │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │ Step 3: Predict logits for current scale             │    │
│  │                                                       │    │
│  │ logits = head(x)  # [B, pn*pn, vocab_size]           │    │
│  │                                                       │    │
│  │ # Optional: Classifier-Free Guidance                 │    │
│  │ if cfg > 1:                                          │    │
│  │     logits = (1+cfg) * logits_cond - cfg * logits_uncond │
│  │                                                       │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │ Step 4: Sample indices (RANDOMNESS SOURCE!)          │    │
│  │ ═════════════════════════════════════════════════    │    │
│  │                                                       │    │
│  │ probs = softmax(logits / temperature)                │    │
│  │ indices = sample_top_k_top_p(probs, k, p)            │    │
│  │ # indices: [B, pn*pn] ← DISCRETE integers 0~V-1     │    │
│  │                                                       │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │ Step 5: Convert indices → embeddings (CRITICAL!)     │    │
│  │ ═════════════════════════════════════════════════    │    │
│  │                                                       │    │
│  │ # 5a: Codebook lookup (DISCRETE → CONTINUOUS)        │    │
│  │ h = codebook.embedding(indices)  # [B, pn*pn, Cvae]  │    │
│  │ # indices are INTEGERS, h is FLOAT tensor            │    │
│  │                                                       │    │
│  │ # 5b: Reshape to 2D spatial                          │    │
│  │ h = h.transpose(1,2).reshape(B, Cvae, pn, pn)        │    │
│  │                                                       │    │
│  │ # 5c: Upsample to full resolution & apply phi        │    │
│  │ h_up = bicubic_upsample(h, H, W)  # [B, Cvae, H, W]  │    │
│  │ h_up = phi_scale(h_up)                               │    │
│  │                                                       │    │
│  │ # 5d: Accumulate to f_hat                            │    │
│  │ f_hat += h_up  # [B, Cvae, H, W]                     │    │
│  │                                                       │    │
│  │ # 5e: Prepare next scale input (if not last)         │    │
│  │ if scale_idx < num_scales - 1:                       │    │
│  │     next_input = downsample(f_hat, pn_next, pn_next) │    │
│  │     # This will be converted via word_embed next iter│    │
│  │                                                       │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Step 6: Decode to Image                                     │
│  ────────────────────────────────────────────────           │
│  f_hat [B, Cvae, H, W] → VAE decoder → image [B, 3, H, W]   │
│                                                              │
│  Output: Generated Image of class_label                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Classifier-Free Guidance (CFG)

```
┌─────────────────────────────────────────────────────────────┐
│              Classifier-Free Guidance (CFG)                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  During inference, run two forward passes:                  │
│                                                              │
│  1. Conditional: class_label = actual_class                 │
│     logits_cond = model(class_emb)                          │
│                                                              │
│  2. Unconditional: class_label = NULL_TOKEN                 │
│     logits_uncond = model(null_emb)                         │
│                                                              │
│  3. Combine:                                                 │
│     logits = logits_uncond + cfg * (logits_cond - logits_uncond) │
│                                                              │
│  cfg = 1.0  → no guidance (pure conditional)               │
│  cfg = 1.5  → mild guidance (default)                      │
│  cfg = 3.0  → strong guidance (more class-consistent)      │
│                                                              │
│  Code (var.py line 172-173):                                │
│    t = cfg * ratio  # ratio increases with scale            │
│    logits = (1+t) * logits[:B] - t * logits[B:]            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Code Locations

| Component                     | File           | Lines     |
|-------------------------------|----------------|-----------|
| VAR main class                | `var.py`       | 21-290    |
| Autoregressive inference      | `var.py`       | 127-190   |
| Training forward              | `var.py`       | 192-234   |
| Multi-scale quantizer         | `quant.py`     | 15-196    |
| Quantization forward          | `quant.py`     | 52-104    |
| idxBl_to_var_input            | `quant.py`     | 169-184   |
| get_next_autoregressive_input | `quant.py`     | 187-196   |
| AdaLN block                   | `basic_var.py` | 128-162   |
| Self-attention                | `basic_var.py` | 58-125    |
| VQ-VAE                        | `vqvae.py`     | full file |

---

## Comparison: VAR vs Other Methods

```
┌─────────────────────────────────────────────────────────────┐
│              Generation Paradigm Comparison                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Next-Token AR (GPT):                                    │
│     token_0 → token_1 → token_2 → ... → token_L            │
│     (L steps, sequential)                                   │
│                                                              │
│  2. Diffusion (DDPM):                                        │
│     noisy_image → denoise → denoise → ... → clean_image    │
│     (T steps, parallel denoising)                           │
│                                                              │
│  3. VAR (Next-Scale AR):                                    │
│     scale_0 → scale_1 → scale_2 → ... → scale_K            │
│     (K steps, K << L, scale-by-scale)                       │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Metric Comparison:                                          │
│                                                              │
│  Method        | Steps | Parallelism | Quality | Speed     │
│  ──────────────┼───────┼─────────────┼─────────┼───────────│
│  GPT-style AR  | ~1000 | None        | Good    | Slow      │
│  Diffusion     | ~1000 | Full        | Great   | Slow      │
│  VAR           | ~10   | Per-scale   | Great   | Fast      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Summary: VAR Key Points

1. **Two-Stage Training**:
   - Stage 1: Train VQ-VAE (encoder + quantizer + decoder)
   - Stage 2: Train VAR transformer (predict next-scale indices)

2. **Next-Scale Autoregression**:
   - Not next-token, but next-scale
   - Scale k depends on scales [0, k-1]
   - Causal attention mask enforces this dependency

3. **Class Conditioning**:
   - Class embedding added at every scale
   - AdaLN injects conditioning into transformer

4. **Randomness in Generation**:
   - Sample from predicted distribution at each scale
   - Temperature and top-k/p control diversity

5. **Efficient Inference**:
   - Only ~10 steps (vs ~1000 for diffusion)
   - KV cache for efficient autoregression
   - Classifier-free guidance for quality control

6. **CRITICAL: Indices Never Go Directly to Transformer**:
   - Transformer ALWAYS sees continuous embeddings, NEVER discrete indices
   - Conversion: indices → codebook.embedding → word_embed → transformer
   - Two different dimensions: Cvae (codebook) vs C (transformer)

---

## Quick Reference: Data Types Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│              What Data Type at Each Stage?                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Image                    →  CONTINUOUS [B, 3, H, W]               │
│       ↓                                                              │
│  Encoder output (z)       →  CONTINUOUS [B, Cvae, H', W']          │
│       ↓                                                              │
│  Quantizer indices        →  DISCRETE   List[[B, k×k] integers]    │
│       ↓ (codebook.embedding)                                        │
│  Codebook embeddings      →  CONTINUOUS [B, k×k, Cvae]             │
│       ↓ (accumulate f_hat)                                          │
│  f_hat                    →  CONTINUOUS [B, Cvae, H, W]            │
│       ↓ (downsample + word_embed)                                   │
│  Transformer input        →  CONTINUOUS [B, L, C]                  │
│       ↓                                                              │
│  Transformer output       →  CONTINUOUS [B, L, C]                  │
│       ↓ (head)                                                       │
│  Logits                   →  CONTINUOUS [B, L, V] (probabilities)  │
│       ↓ (sample)                                                     │
│  Sampled indices          →  DISCRETE   [B, k×k] integers          │
│       ↓ (loop back to codebook.embedding)                           │
│                                                                      │
│  ═══════════════════════════════════════════════════════════════    │
│  KEY INSIGHT:                                                        │
│  - DISCRETE indices exist ONLY at:                                  │
│    (1) Quantizer output (ground truth during training)              │
│    (2) Sampling output (during inference)                           │
│  - Transformer input/output is ALWAYS CONTINUOUS                    │
│  - The conversion DISCRETE → CONTINUOUS happens via codebook        │
│  ═══════════════════════════════════════════════════════════════    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```
