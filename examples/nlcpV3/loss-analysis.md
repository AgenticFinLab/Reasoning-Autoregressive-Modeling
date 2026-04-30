# NLCP V3 Loss Analysis

## Overview

Total loss is a weighted sum of four components:

```
total_loss = recon_w × L_recon + ordering_w × L_ordering + residual_w × L_residual + reasoning_w × L_reasoning
```

All four losses are **always computed** (no gating). Weights control gradient contribution; setting a weight to 0 disables gradient flow but the loss is still logged for monitoring.

**Code**: [`train_builder.py` L201–210](examples/nlcpV3/train_builder.py#L201-L210) (base 3 losses assembled in `compute_builder_loss`), [`train_builder.py` L379–407](examples/nlcpV3/train_builder.py#L379-L407) (reasoning loss added in training loop when `batch.has_solution`).

---

## 1. Reconstruction Loss (`recon_loss`)

### Formula

$$L_\text{recon} = \frac{1}{N_\text{valid} \times D_\text{encoder}} \sum_{b,t,d} \bigl(\text{back\_proj}(\hat{f}_K) - H_\text{CoT}\bigr)^2_{b,t,d} \cdot \mathbb{1}[\text{mask}(b,t)=1]$$

Equivalently: `MSE(back_proj(f_hat_K), H_CoT)` averaged over all valid elements (tokens × D_encoder).

### What it measures

How well the concept pyramid preserves the **original frozen encoder output** after a round-trip through concept space:

```
H_CoT [B,L,D_enc] → input_proj → H_proj [B,L,D] → pyramid → f_hat_K [B,L,D] → back_proj → recon [B,L,D_enc] → MSE(recon, H_CoT)
```

### Data flow (with code references)

| Step | Operation                     | Tensor / Shape                                               | Code Location                                                                                   |
|------|-------------------------------|--------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| 1    | Encode CoT                    | `H_CoT = backbone(CoT)` → `[B, L, D_encoder]`                | [`concept_hybrid_builder.py` L847–859](examples/nlcpV3/concept_hybrid_builder.py#L847-L859)     |
| 2    | Project to concept space      | `H_proj = LayerNorm(input_proj(H_CoT))` → `[B, L, D]`        | [`concept_hybrid_builder.py` L1024](examples/nlcpV3/concept_hybrid_builder.py#L1024)            |
| 3    | Init residual                 | `f_rest_0 = H_proj.clone()` → `[B, L, D]`                    | [`concept_hybrid_builder.py` L1034](examples/nlcpV3/concept_hybrid_builder.py#L1034)            |
| 4    | Init accumulator              | `f_hat_0 = zeros_like(H_proj)` → `[B, L, D]`                 | [`concept_hybrid_builder.py` L1037](examples/nlcpV3/concept_hybrid_builder.py#L1037)            |
| 5    | Per-level loop (k=0..K−1)     | See sub-steps below                                          | [`concept_hybrid_builder.py` L1055–1177](examples/nlcpV3/concept_hybrid_builder.py#L1055-L1177) |
| 5a   | Soft attention                | `A_k = softmax(Q_k @ f_rest_k^T / (√D × τ))` → `[B, L_k, L]` | [`concept_hybrid_builder.py` L1072–1097](examples/nlcpV3/concept_hybrid_builder.py#L1072-L1097) |
| 5b   | Base concepts                 | `C_k_base = level_proj_k(A_k @ f_rest_k)` → `[B, L_k, D]`    | [`concept_hybrid_builder.py` L1104–1109](examples/nlcpV3/concept_hybrid_builder.py#L1104-L1109) |
| 5c   | Per-level recon               | `R_k = A_k^T @ C_k_base` → `[B, L, D]`                       | [`concept_hybrid_builder.py` L1117–1122](examples/nlcpV3/concept_hybrid_builder.py#L1117-L1122) |
| 5d   | Accumulate                    | `f_hat_{k+1} = f_hat_k + R_k`                                | [`concept_hybrid_builder.py` L1130](examples/nlcpV3/concept_hybrid_builder.py#L1130)            |
| 5e   | Update residual               | `f_rest_{k+1} = f_rest_k - R_k`                              | [`concept_hybrid_builder.py` L1133](examples/nlcpV3/concept_hybrid_builder.py#L1133)            |
| 6    | Back-project to encoder space | `recon_enc = back_proj(f_hat_K)` → `[B, L, D_encoder]`       | [`concept_hybrid_builder.py` L1187](examples/nlcpV3/concept_hybrid_builder.py#L1187)            |
| 7    | Compute masked MSE            | See formula above                                            | [`train_builder.py` L138–156](examples/nlcpV3/train_builder.py#L138-L156)                       |

### Computation details

```python
# train_builder.py L143-155
mask = pyramid.attention_mask.unsqueeze(-1)                      # [B, L, 1]
recon_diff = (pyramid.reconstructed_encoder_hidden
              - pyramid.encoder_hidden_states) * mask            # [B, L, D_enc] masked
num_valid_elements = mask.sum() * pyramid.encoder_hidden_states.shape[-1]   # N_valid × D_enc
recon_loss = (recon_diff ** 2).sum() / num_valid_elements        # scalar
```

- **Numerator**: sum of squared differences over all (b,t,d) where token t is valid.
- **Denominator**: `N_valid_tokens × D_encoder` — total number of valid scalar elements. This matches `F.mse_loss(reduction='mean')` convention where the mean is over ALL elements, not just the token count.
- **Unmasked fallback**: `F.mse_loss(reconstructed_encoder_hidden, encoder_hidden_states)` (L153–155).

### Key design decisions

- **Target is `H_CoT`** (frozen encoder output), NOT `H_proj` (projected version). This follows VAR's principle: the quantizer reconstructs against the frozen encoder ([`quant.py` L95](third-part/VAR-main/models/quant.py#L95)).
- **Round-trip via `back_proj`**: Since pyramid operates in D space but target is in D_encoder space, `back_proj` must learn a meaningful inverse of `input_proj`.
- **`back_proj` initialization**: `back_proj.weight = input_proj.weight^T` ([`concept_hybrid_builder.py` L782](examples/nlcpV3/concept_hybrid_builder.py#L782)), providing a pseudo-inverse starting point.

### Gradient flow

```
L_recon
 → (recon_diff)² → back_proj.weight
   → reconstructed_accumulator (= f_hat_K = Σ R_k)
     → R_k = A_k^T @ C_k_base
       → A_k → concept_queries[k], temperature, f_rest_k
       → C_k_base → level_projs[k], A_k, f_rest_k
         → f_rest_k → ... → input_proj.weight, input_proj.bias, input_proj_norm
```

All Builder parameters receive gradients. `reason_model` is frozen — **no gradients flow through `H_CoT`**.

---

## 2. Ordering Loss (`ordering_loss`)

### Formula (margin variant, default)

$$L_\text{ordering} = \frac{1}{|\mathcal{K}|} \sum_{k \in \mathcal{K}} \sum_{j=0}^{L_k-2} \text{mean}_B\Bigl[\text{ReLU}\bigl(\text{exp\_pos}_k[j] - \text{exp\_pos}_k[j+1] + m\bigr)\Bigr]$$

where:
- $\text{exp\_pos}_k[j] = \sum_t A_{k,j}(t) \cdot t$ — expected CoT position for concept j at level k
- $\mathcal{K} = \{k : L_k > 1\}$ — levels with more than one concept (skips level 0)
- $m$ = margin (config: `ordering_margin`, default 1.0)

### What it measures

Enforces **intra-level positional ordering**: concept j should attend to earlier CoT positions than concept j+1. This ensures concepts within a level are ordered monotonically by their CoT position, not randomly distributed.

### Data flow

| Step | Operation               | Tensor / Shape                                                          | Code Location                                                                                    |
|------|-------------------------|-------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| 1    | Get attention weights   | `A_k` from pyramid level k                                              | Produced at [`concept_hybrid_builder.py` L1092](examples/nlcpV3/concept_hybrid_builder.py#L1092) |
| 2    | Create position indices | `positions = arange(L)` → `[L]`                                         | [`train_builder.py` L74](examples/nlcpV3/train_builder.py#L74)                                   |
| 3    | Expected position       | `exp_pos = (A_k × positions).sum(dim=-1)` → `[B, L_k]`                  | [`train_builder.py` L76](examples/nlcpV3/train_builder.py#L76)                                   |
| 4    | Margin violation        | `ReLU(exp_pos[:,j] - exp_pos[:,j+1] + margin).mean()` → scalar per pair | [`train_builder.py` L81–83](examples/nlcpV3/train_builder.py#L81-L83)                            |
| 5    | Sum over pairs          | `loss += violation` for j=0..L_k-2                                      | [`train_builder.py` L79–83](examples/nlcpV3/train_builder.py#L79-L83)                            |
| 6    | Average over levels     | `ordering_loss /= levels_with_ordering`                                 | [`train_builder.py` L184–185](examples/nlcpV3/train_builder.py#L184-L185)                        |

### Key details

- **Skips levels with L_k=1** (level 0 with 1 concept): no ordering to enforce.
- **Margin `m`**: minimum required gap in expected position between adjacent concepts. Larger margin → stricter ordering.
- **Alternative**: Gaussian target variant ([`train_builder.py` L88–115](examples/nlcpV3/train_builder.py#L88-L115)) — KL-divergence-like loss against Gaussian distributions centered at evenly-spaced segment midpoints. Selected via config `ordering_loss_type: "gaussian"` or `"both"`.
- **Level loop**: iterates over all `pyramid.level_outputs` ([`train_builder.py` L163–185](examples/nlcpV3/train_builder.py#L163-L185)).

### Gradient flow

```
L_ordering → exp_pos → A_k → attention_scores / (√D × τ)
  → concept_queries[k] (via Q_k @ f_rest_k^T)
  → temperature (via scaling)
  → f_rest_k → ... → input_proj, input_proj_norm
```

Does **not** flow through `back_proj`, `level_projs`, or `level_attn`.

---

## 3. Residual Loss (`residual_loss`)

### Formula

$$L_\text{residual} = \frac{1}{N_\text{valid} \times D} \sum_{b,t,d} |f\_rest_K|_{b,t,d} \cdot \mathbb{1}[\text{mask}(b,t)=1]$$

Equivalently: L1 mean of the final residual `f_rest_K`, averaged over all valid elements (tokens × D).

### What it measures

The magnitude of the **unexplained residual** after K levels of decomposition. Since `f_rest_K = H_proj - f_hat_K`, this measures how much of the projected CoT information the pyramid failed to capture. Ideally `f_rest_K → 0` for exact decomposition.

### Data flow

| Step | Operation      | Tensor / Shape                                                   | Code Location                                                                                                                   |
|------|----------------|------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| 1    | Final residual | `f_rest_K` after K iterations of `f_rest_{k+1} = f_rest_k - R_k` | [`concept_hybrid_builder.py` L1133](examples/nlcpV3/concept_hybrid_builder.py#L1133), stored in `PyramidOutput.residual_hidden` |
| 2    | Masked L1 mean | `(                                                               | f_rest_K                                                                                                                        |

### Computation details

```python
# train_builder.py L191-196
mask = pyramid.attention_mask.unsqueeze(-1)                          # [B, L, 1]
num_valid_elements = mask.sum() * pyramid.residual_hidden.shape[-1]  # N_valid × D
res_loss = (pyramid.residual_hidden.abs() * mask).sum() / num_valid_elements
```

- **L1 norm** (not L2): penalizes absolute value, encouraging sparse/small residuals.
- **Denominator**: `N_valid_tokens × D` — consistent per-element mean convention.
- **Operates in concept space D**, not encoder space D_encoder.

### Gradient flow

```
L_residual → |f_rest_K| → f_rest_K = H_proj - f_hat_K
  → f_hat_K = Σ R_k → A_k, C_k_base
    → concept_queries[k], level_projs[k], temperature
  → H_proj → input_proj, input_proj_norm
```

Does **not** flow through `back_proj` or `level_attn`.

### Relationship to recon_loss

`residual_loss` and `recon_loss` are **correlated but not identical**:

| Aspect              | recon_loss                                     | residual_loss                               |
|---------------------|------------------------------------------------|---------------------------------------------|
| Space               | D_encoder (via `back_proj`)                    | D (concept space)                           |
| Norm                | L2 (MSE)                                       | L1 (MAE)                                    |
| Measures            | Round-trip fidelity: H_CoT → D → D_encoder     | In-space decomposition: H_proj − f_hat_K    |
| Involves back_proj? | Yes                                            | No                                          |
| Unique signal       | Forces `back_proj` to learn meaningful inverse | Directly regularizes concept-space residual |

A small residual in D space does NOT guarantee small recon error in D_encoder space (because `back_proj` may not be a perfect inverse). Conversely, good D_encoder reconstruction may coexist with large D-space residual if `back_proj` compensates. If `recon_loss` alone suffices, `residual_loss` can be disabled by setting `residual_loss_weight: 0`.

---

## 4. Reasoning Loss (`reasoning_loss`)

### Formula

$$L_\text{reasoning} = \text{CrossEntropy}\bigl(\text{logits}_\text{solution},\; \text{solution\_ids}\bigr)$$

where `logits_solution` is produced by feeding `[back_proj(concepts); Q_embeds]` through the frozen `reason_model`.

### What it measures

Whether the extracted concepts, combined with the question, can **predict the correct solution tokens**. This is the only loss that validates the **semantic usefulness** of concepts (not just geometric reconstruction fidelity).

### Data flow

| Step | Operation                   | Tensor / Shape                                                       | Code Location                                                                               |
|------|-----------------------------|----------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| 1    | Concatenate all concepts    | `concepts = cat(C_0, ..., C_{K-1})` → `[B, total_C, D]`              | [`concept_hybrid_builder.py` L909](examples/nlcpV3/concept_hybrid_builder.py#L909)          |
| 2    | Back-project to encoder dim | `concept_embeds = back_proj(concepts)` → `[B, total_C, D_enc]`       | [`concept_hybrid_builder.py` L912](examples/nlcpV3/concept_hybrid_builder.py#L912)          |
| 3    | Embed question tokens       | `Q_embeds = embed_tokens(Q_ids)` → `[B, L_Q, D_enc]`                 | [`concept_hybrid_builder.py` L917](examples/nlcpV3/concept_hybrid_builder.py#L917)          |
| 4    | Concatenate input           | `input = [concept_embeds; Q_embeds]` → `[B, total_C+L_Q, D_enc]`     | [`concept_hybrid_builder.py` L921–923](examples/nlcpV3/concept_hybrid_builder.py#L921-L923) |
| 5    | Build attention mask        | `mask = [ones(total_C); Q_mask]` → `[B, total_C+L_Q]`                | [`concept_hybrid_builder.py` L927–935](examples/nlcpV3/concept_hybrid_builder.py#L927-L935) |
| 6    | Forward full reason_model   | `logits = reason_model(inputs_embeds=input)` → `[B, total_C+L_Q, V]` | [`concept_hybrid_builder.py` L939–943](examples/nlcpV3/concept_hybrid_builder.py#L939-L943) |
| 7    | Extract solution logits     | `sol_logits = logits[:, total_C:, :]` → `[B, L_Q, V]`                | [`concept_hybrid_builder.py` L954](examples/nlcpV3/concept_hybrid_builder.py#L954)          |
| 8    | Cross-entropy               | `CE(sol_logits[:,:L_min], sol_ids[:,:L_min])`                        | [`concept_hybrid_builder.py` L958–963](examples/nlcpV3/concept_hybrid_builder.py#L958-L963) |

### Training loop integration

```python
# train_builder.py L379-407
if batch.has_solution:
    q_tokens = builder.tokenizer(batch.questions, ...)
    sol_tokens = builder.tokenizer(batch.solutions, ...)
    reasoning_loss = builder.compute_reasoning_loss(pyramid, q_ids, q_mask, sol_ids)
    total_loss = total_loss + reasoning_loss_weight * reasoning_loss
```

- **Gated by data availability** (`batch.has_solution`), not by weight. Even with `reasoning_loss_weight > 0`, if the batch has no solutions, reasoning loss is simply not computed for that batch.
- **Tokenization**: both Q and S are tokenized on-the-fly using `builder.tokenizer` with `max_length = pyramid_cfg["max_seq_len"]`.

### Key details

- **Uses the FULL `reason_model`** (backbone + lm_head): `self.reason_model(inputs_embeds=...)` ([L939](examples/nlcpV3/concept_hybrid_builder.py#L939)), not just the backbone.
- **`reason_model` is frozen**: all params have `requires_grad=False`. Gradients flow through `inputs_embeds` (the concatenated concept+Q embeddings), reaching `back_proj` and upstream pyramid parameters.
- **`back_proj` is shared** with recon_loss: same layer maps concepts to D_encoder space for both reconstruction and reasoning.
- **`ignore_index=-100`**: standard HF padding token exclusion in cross-entropy (L962).
- **`L_min = min(L_Q, L_S)`**: handles length mismatch between question-position logits and solution tokens (L958).

### Gradient flow

```
L_reasoning → CE(logits, sol_ids)
  → logits = reason_model(inputs_embeds)  [frozen: grad passes through inputs_embeds only]
  → decoder_input_embeds = [concept_embeds; Q_embeds]
    → concept_embeds = back_proj(concepts)
      → back_proj.weight
      → concepts = cat(C_0, ..., C_{K-1})
        → C_k = C_k_base + refined_k (for k>0)
          → C_k_base → level_projs[k], A_k, f_rest_k → input_proj, input_proj_norm
          → refined_k → level_attn[k] (cross-attention), concept_queries[k]
    → Q_embeds = embed_tokens(Q_ids) [frozen: no grad]
```

Note: `reason_model` parameters do NOT receive gradients (frozen). `embed_tokens` is part of the frozen model, so Q_embeds also has no grad. Only `concept_embeds` carries gradients backward.

---

## Total Loss Assembly

### In `compute_builder_loss` (base 3 losses)

**Code**: [`train_builder.py` L201–210](examples/nlcpV3/train_builder.py#L201-L210)

```python
total_loss = (
    loss_weights["recon_loss_weight"]    * recon_loss       # L2 in D_encoder space
    + loss_weights["ordering_loss_weight"] * ordering_loss   # margin-based positional ordering
    + loss_weights["residual_loss_weight"] * res_loss        # L1 in D space
)
```

### In training loop (reasoning loss added)

**Code**: [`train_builder.py` L400–407](examples/nlcpV3/train_builder.py#L400-L407)

```python
if batch.has_solution:
    reasoning_loss = builder.compute_reasoning_loss(pyramid, q_ids, q_mask, sol_ids)
    total_loss = total_loss + loss_weights["reasoning_loss_weight"] * reasoning_loss
```

---

## Trainable Parameters Summary

All four losses share (subsets of) the same trainable parameter set:

| Parameter                    | Shape                 | Updated by which losses                   |
|------------------------------|-----------------------|-------------------------------------------|
| `input_proj.weight`          | `[D, D_encoder]`      | recon, ordering, residual, reasoning      |
| `input_proj.bias`            | `[D]`                 | recon, ordering, residual, reasoning      |
| `input_proj_norm.weight`     | `[D]`                 | recon, ordering, residual, reasoning      |
| `input_proj_norm.bias`       | `[D]`                 | recon, ordering, residual, reasoning      |
| `concept_queries[k]`         | `[L_k, D]` × K levels | recon, ordering, residual, reasoning      |
| `temperature`                | `[1]`                 | recon, ordering, residual, reasoning      |
| `level_projs[k].weight/bias` | `[D, D]` × K levels   | recon, residual, reasoning                |
| `level_attn[k]` (MHA params) | varies × K levels     | **reasoning only** (via refined concepts) |
| `back_proj.weight`           | `[D_encoder, D]`      | **recon, reasoning only**                 |

### Frozen parameters

| Parameter                      | Frozen?                                                         | Role                                                                          |
|--------------------------------|-----------------------------------------------------------------|-------------------------------------------------------------------------------|
| `reason_model` (all weights)   | Configurable via `training.reason_model.freeze` (default: true) | Encoding: backbone produces H_CoT; Decoding: lm_head produces solution logits |
| `reason_model` + LoRA adapters | LoRA params trainable if `training.reason_model.lora` is set    | Fine-tune backbone representation with PEFT                                   |

Freezing is controlled by `training.reason_model.freeze` (default: true for VAR-faithful behavior). LoRA config is at `training.reason_model.lora`. Code: [`concept_hybrid_builder.py` L641–654](examples/nlcpV3/concept_hybrid_builder.py#L641-L654).

---

## Architectural Comparison with VAR

| Aspect              | VAR (VQVAE Stage 1)                                       | NLCP V3                                                                   |
|---------------------|-----------------------------------------------------------|---------------------------------------------------------------------------|
| Encoder             | CNN encoder (trainable)                                   | Frozen LLM backbone                                                       |
| Target              | Encoder output `f_BChw`                                   | `H_CoT` from frozen backbone                                              |
| Decomposition space | Same dim as encoder (C=32)                                | Reduced dim D (e.g., 256) via `input_proj`                                |
| Dimension change?   | **No** — `quant_conv` is `Conv2d(C, C)`                   | **Yes** — D_encoder (e.g., 1536) → D (e.g., 256)                          |
| Reconstruction      | `F.mse_loss(f_hat, f_BChw)` in C space                    | `MSE(back_proj(f_hat), H_CoT)` round-trip through D                       |
| Quantization        | Hard (nearest codebook) + STE                             | Soft (attention-weighted pooling)                                         |
| Additional losses   | VQ commitment loss (`β × MSE`)                            | ordering, residual, reasoning                                             |
| Code ref            | [`quant.py` L95](third-part/VAR-main/models/quant.py#L95) | [`train_builder.py` L138–156](examples/nlcpV3/train_builder.py#L138-L156) |

### Key difference: round-trip reconstruction

VAR operates entirely in the encoder's native dimension — no dimension change, no back-projection:
```
f_BChw [B,C,H,W] → quantize → f_hat [B,C,H,W] → MSE(f_hat, f_BChw)
```

NLCP V3 must project down then back up because D_encoder ≠ D:
```
H_CoT [B,L,D_enc] → input_proj → H_proj [B,L,D] → pyramid → f_hat [B,L,D] → back_proj → [B,L,D_enc] → MSE(·, H_CoT)
```

This means `back_proj` must learn a meaningful inverse of `input_proj`. The initialization `back_proj.weight = input_proj.weight^T` ([`concept_hybrid_builder.py` L782`](examples/nlcpV3/concept_hybrid_builder.py#L782)) provides a starting point.

---

## Discussion Points

### 1. `back_proj` is critical and serves dual roles

`back_proj` is used in **two distinct paths**:

| Role           | Path                                                                                 | Loss             |
|----------------|--------------------------------------------------------------------------------------|------------------|
| Reconstruction | `back_proj(f_hat_K)` → compare against `H_CoT`                                       | `recon_loss`     |
| Reasoning      | `back_proj(cat(C_0,...,C_{K-1}))` → feed into `reason_model` for solution prediction | `reasoning_loss` |

These two objectives may **conflict**: recon wants `back_proj` to faithfully invert `input_proj` (geometric fidelity), while reasoning wants `back_proj` to produce embeddings the LM head can decode into correct solutions (semantic utility). The shared `back_proj` must balance both.

**Note**: The reconstruction path uses `f_hat_K` (accumulated base-concept reconstructions), while the reasoning path uses `cat(C_k)` (refined concepts including cross-attention output). These are **different tensors** — the reasoning path includes refinement information that the reconstruction path does not.

### 2. `residual_loss` and `recon_loss` are complementary, not redundant

Both measure reconstruction quality, but in different spaces with different norms:

- `recon_loss` (L2 in D_encoder) forces the full round-trip to preserve information. It's the only loss that trains `back_proj` toward a good inverse.
- `residual_loss` (L1 in D) directly regularizes the concept-space decomposition without involving `back_proj`. It provides a cleaner signal for the pyramid mechanics.

In principle, if `back_proj` is perfect, minimizing `residual_loss` implies minimizing `recon_loss`. In practice, the two provide complementary gradient signals. `residual_loss` can be disabled (`weight=0`) if `recon_loss` alone drives sufficient convergence.

### 3. Ordering loss is structurally independent

The ordering loss depends **only on attention weights `A_k`**, not on concept quality, reconstruction, or `back_proj`. It can be fully satisfied even if concepts are semantically meaningless — as long as they attend to monotonically increasing positions. This is by design: it's a structural regularizer that prevents degenerate attention patterns, not a quality metric.

### 4. Reasoning loss is the only semantic anchor

Without `reasoning_loss`, the pyramid could converge to a mathematically valid decomposition that is semantically empty — concepts that perfectly reconstruct H_CoT but carry no useful reasoning information. The reasoning loss forces the concept representation to support actual solution prediction, preventing degenerate geometric-only solutions.

### 5. `level_attn` (cross-attention refinement) is only trained by reasoning loss

The cross-attention refinement layers (`level_attn[k]`) produce `refined_k`, which is added to `C_k_base` to form the final concept `C_k`. Since only `C_k_base` (not `refined_k`) enters the reconstruction flow (`R_k = A_k^T @ C_k_base`), the refinement layers receive **no gradients from recon_loss, residual_loss, or ordering_loss**. They are trained exclusively through `reasoning_loss` (via the refined concepts entering the reasoning path). If `reasoning_loss_weight = 0`, the refinement layers are effectively dead parameters.
