# lcp Loss Analysis

## Overview

lcp has **two stages**, each with its own loss function. All loss logic lives in a single module ([`losses.py`](examples/lcp/losses.py)) so that the training scripts only invoke `compute_builder_loss` or `compute_predictor_loss` and never reimplement the math.

### Stage 1 — ConceptPyramidBuilder (`compute_builder_loss`)

A weighted sum of **four** components:

```
L_builder = recon_w    × L_recon
          + ordering_w × L_ordering
          + residual_w × L_residual
          + reasoning_w × L_reasoning      (added only if pyramid.reasoning_logits is populated)
```

- All four losses are **always computed and logged** when their inputs are available (no gating). Weights control gradient contribution; setting a weight to 0 disables gradient flow but the scalar is still logged for monitoring.
- `L_reasoning` is only assembled when the batch carries solution tokens (`batch.has_solution=True`). When the predictor-only path is exercised, it is simply absent.
- **Code**: [`losses.py` L103–210](examples/lcp/losses.py#L103-L210) (`compute_builder_loss`), [`train_builder.py`](examples/lcp/train_builder.py) (`builder(batch) → compute_builder_loss(...)` in the training loop).

### Stage 2 — ConceptPredictor (`compute_predictor_loss`)

A weighted sum of **two** components:

```
L_predictor = concept_w   × L_concept
            + reasoning_w × L_reasoning    (added only if reasoning_logits is populated)
```

- `L_concept` — per-level MSE (or cosine) averaged across the K pyramid levels, computed between the predictor's predicted concepts and the frozen Builder's ground-truth concepts.
- `L_reasoning` — next-token cross-entropy on solution tokens, produced by the **same unified teacher-forced forward** that yields `L_concept` (not a separate pass).
- Both components are optional: `compute_predictor_loss` gracefully skips whichever tensor the caller did not populate, so the same function serves training and evaluation.
- **Code**: [`losses.py` L262–305](examples/lcp/losses.py#L262-L305) (`compute_predictor_concept_loss`), [`losses.py` L308–378](examples/lcp/losses.py#L308-L378) (`compute_predictor_loss`), [`train_predictor.py`](examples/lcp/train_predictor.py) (`predictor(question_ids, ..., gt_concepts, solution_ids) → compute_predictor_loss(...)` in the training loop).

### Document structure

- §1–§4 cover the four **Builder** losses (recon, ordering, residual, reasoning).
- §5 covers total **Builder** loss assembly.
- §6 covers the two **Predictor** losses (concept, reasoning) and their assembly.
- §7 is the unified **Trainable-Parameter Summary** (Builder + Predictor SHARED + Predictor INDEPENDENT).
- §8 compares the full stack with VAR Stage-1 / Stage-2.
- §9 is the Discussion / gotcha list.

---

## 1. Reconstruction Loss (`recon_loss`) — Builder

### Formula

$$L_\text{recon} = \frac{1}{N_\text{valid} \times D_\text{encoder}} \sum_{b,t,d} \bigl(\text{back\_proj}(\hat{f}_K) - H_\text{CoT}\bigr)^2_{b,t,d} \cdot \mathbb{1}[\text{mask}(b,t)=1]$$

Equivalently: `MSE(back_proj(f_hat_K), H_CoT)` averaged over all valid elements (tokens × D_encoder).

### What it measures

How well the concept pyramid preserves the **original frozen encoder output** after a round-trip through concept space:

```
H_CoT [B,L,D_enc] → input_proj → H_proj [B,L,D] → pyramid → f_hat_K [B,L,D] → back_proj → recon [B,L,D_enc] → MSE(recon, H_CoT)
```

### Data flow (with code references)

| Step | Operation                     | Tensor / Shape                                               | Code Location                                                                  |
|------|-------------------------------|--------------------------------------------------------------|--------------------------------------------------------------------------------|
| 1    | Encode CoT                    | `H_CoT = backbone(CoT)` → `[B, L, D_encoder]`                | [`concept_builder.py` L708–788](examples/lcp/concept_builder.py#L708-L788)     |
| 2    | Project to concept space      | `H_proj = LayerNorm(input_proj(H_CoT))` → `[B, L, D]`        | [`concept_builder.py` L949](examples/lcp/concept_builder.py#L949)              |
| 3    | Init residual                 | `f_rest_0 = H_proj.clone()` → `[B, L, D]`                    | [`concept_builder.py` L959](examples/lcp/concept_builder.py#L959)              |
| 4    | Init accumulator              | `f_hat_0 = zeros_like(H_proj)` → `[B, L, D]`                 | [`concept_builder.py` L962](examples/lcp/concept_builder.py#L962)              |
| 5    | Per-level loop (k=0..K−1)     | See sub-steps below                                          | [`concept_builder.py` L980–1069](examples/lcp/concept_builder.py#L980-L1069)   |
| 5a   | Soft attention                | `A_k = softmax(Q_k @ f_rest_k^T / (√D × τ))` → `[B, L_k, L]` | [`concept_builder.py` L997–1022](examples/lcp/concept_builder.py#L997-L1022)   |
| 5b   | Base concepts                 | `C_k_base = level_proj_k(A_k @ f_rest_k)` → `[B, L_k, D]`    | [`concept_builder.py` L1029–1034](examples/lcp/concept_builder.py#L1029-L1034) |
| 5c   | Per-level recon               | `R_k = A_k^T @ C_k_base` → `[B, L, D]`                       | [`concept_builder.py` L1041](examples/lcp/concept_builder.py#L1041)            |
| 5d   | Accumulate                    | `f_hat_{k+1} = f_hat_k + R_k`                                | [`concept_builder.py` L1052](examples/lcp/concept_builder.py#L1052)            |
| 5e   | Update residual               | `f_rest_{k+1} = f_rest_k - R_k`                              | [`concept_builder.py` L1055](examples/lcp/concept_builder.py#L1055)            |
| 6    | Back-project to encoder space | `recon_enc = back_proj(f_hat_K)` → `[B, L, D_encoder]`       | [`concept_builder.py` L1079](examples/lcp/concept_builder.py#L1079)            |
| 7    | Compute masked MSE            | See formula above                                            | [`losses.py` L109–127](examples/lcp/losses.py#L109-L127)                       |

### Computation details

```python
# losses.py L114-122
mask = pyramid.attention_mask.unsqueeze(-1)                      # [B, L, 1]
recon_diff = (pyramid.reconstructed_encoder_hidden
              - pyramid.encoder_hidden_states) * mask            # [B, L, D_enc] masked
num_valid_elements = mask.sum() * pyramid.encoder_hidden_states.shape[-1]   # N_valid × D_enc
recon_loss = (recon_diff ** 2).sum() / num_valid_elements        # scalar
```

- **Numerator**: sum of squared differences over all (b,t,d) where token t is valid.
- **Denominator**: `N_valid_tokens × D_encoder` — total number of valid scalar elements. This matches `F.mse_loss(reduction='mean')` convention where the mean is over ALL elements, not just the token count.
- **Unmasked fallback**: `F.mse_loss(reconstructed_encoder_hidden, encoder_hidden_states)` (`losses.py` L124–126).

### Key design decisions

- **Target is `H_CoT`** (frozen encoder output), NOT `H_proj` (projected version). This follows VAR's principle: the quantizer reconstructs against the frozen encoder ([`quant.py` L95](third-part/VAR-main/models/quant.py#L95)).
- **Round-trip via `back_proj`**: Since pyramid operates in D space but target is in D_encoder space, `back_proj` must learn a meaningful inverse of `input_proj`.
- **`back_proj` initialization**: `back_proj.weight = input_proj.weight^T` ([`concept_builder.py` L706](examples/lcp/concept_builder.py#L706)), providing a pseudo-inverse starting point.

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

## 2. Ordering Loss (`ordering_loss`) — Builder

### Formula (margin variant, default)

$$L_\text{ordering} = \frac{1}{|\mathcal{K}|} \sum_{k \in \mathcal{K}} \sum_{j=0}^{L_k-2} \text{mean}_B\Bigl[\text{ReLU}\bigl(\text{exp\_pos}_k[j] - \text{exp\_pos}_k[j+1] + m\bigr)\Bigr]$$

where:
- $\text{exp\_pos}_k[j] = \sum_t A_{k,j}(t) \cdot t$ — expected CoT position for concept j at level k
- $\mathcal{K} = \{k : L_k > 1\}$ — levels with more than one concept (skips level 0)
- $m$ = margin (config: `ordering_margin`, default 1.0)

### What it measures

Enforces **intra-level positional ordering**: concept j should attend to earlier CoT positions than concept j+1. This ensures concepts within a level are ordered monotonically by their CoT position, not randomly distributed.

### Data flow

| Step | Operation               | Tensor / Shape                                                          | Code Location                                                                   |
|------|-------------------------|-------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| 1    | Get attention weights   | `A_k` from pyramid level k                                              | Produced at [`concept_builder.py` L1017](examples/lcp/concept_builder.py#L1017) |
| 2    | Create position indices | `positions = arange(L)` → `[L]`                                         | [`losses.py` L42](examples/lcp/losses.py#L42)                                   |
| 3    | Expected position       | `exp_pos = (A_k × positions).sum(dim=-1)` → `[B, L_k]`                  | [`losses.py` L44](examples/lcp/losses.py#L44)                                   |
| 4    | Margin violation        | `ReLU(exp_pos[:,j] - exp_pos[:,j+1] + margin).mean()` → scalar per pair | [`losses.py` L49–50](examples/lcp/losses.py#L49-L50)                            |
| 5    | Sum over pairs          | `loss += violation` for j=0..L_k-2                                      | [`losses.py` L47–50](examples/lcp/losses.py#L47-L50)                            |
| 6    | Average over levels     | `ordering_loss /= levels_with_ordering`                                 | [`losses.py` L155–156](examples/lcp/losses.py#L155-L156)                        |

### Key details

- **Skips levels with L_k=1** (level 0 with 1 concept): no ordering to enforce.
- **Margin `m`**: minimum required gap in expected position between adjacent concepts. Larger margin → stricter ordering.
- **Alternative**: Gaussian target variant ([`losses.py` L56–84](examples/lcp/losses.py#L56-L84)) — KL-divergence-like loss against Gaussian distributions centered at evenly-spaced segment midpoints. Selected via config `ordering_loss_type: "gaussian"` or `"both"`.
- **Level loop**: iterates over all `pyramid.level_outputs` ([`losses.py` L134–156](examples/lcp/losses.py#L134-L156)).

### Gradient flow

```
L_ordering → exp_pos → A_k → attention_scores / (√D × τ)
  → concept_queries[k] (via Q_k @ f_rest_k^T)
  → temperature (via scaling)
  → f_rest_k → ... → input_proj, input_proj_norm
```

Does **not** flow through `back_proj` or `level_projs`.

---

## 3. Residual Loss (`residual_loss`) — Builder

### Formula

$$L_\text{residual} = \frac{1}{N_\text{valid} \times D} \sum_{b,t,d} |f\_rest_K|_{b,t,d} \cdot \mathbb{1}[\text{mask}(b,t)=1]$$

Equivalently: L1 mean of the final residual `f_rest_K`, averaged over all valid elements (tokens × D).

### What it measures

The magnitude of the **unexplained residual** after K levels of decomposition. Since `f_rest_K = H_proj - f_hat_K`, this measures how much of the projected CoT information the pyramid failed to capture. Ideally `f_rest_K → 0` for exact decomposition.

### Data flow

| Step | Operation      | Tensor / Shape                                                   | Code Location                                                                                                  |
|------|----------------|------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| 1    | Final residual | `f_rest_K` after K iterations of `f_rest_{k+1} = f_rest_k - R_k` | [`concept_builder.py` L1055](examples/lcp/concept_builder.py#L1055), stored in `PyramidOutput.residual_hidden` |
| 2    | Masked L1 mean | `(                                                               | f_rest_K                                                                                                       |

### Computation details

```python
# losses.py L159–170
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

Does **not** flow through `back_proj`.

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

## 4. Reasoning Loss (`reasoning_loss`) — Builder

### Formula

$$L_\text{reasoning} = \text{CrossEntropy}\bigl(\text{logits}_\text{solution},\; \text{solution\_ids}\bigr)$$

where `logits_solution` is produced by feeding `[Q_embeds, back_proj(concepts), S_embeds]` through the frozen `reason_model` with teacher-forcing, then slicing the logits at solution-prediction positions.

### What it measures

Whether the extracted concepts, placed between Q and S in the autoregressive chain (replacing CoT), can **predict the correct solution tokens**. This is the only loss that validates the **semantic usefulness** of concepts (not just geometric reconstruction fidelity).

The original autoregressive flow is `Q -> CoT -> Solution`. With concepts replacing CoT, the reasoning loss validates the flow `Q -> Concepts -> Solution`.

### Data flow (teacher-forcing)

| Step | Operation                   | Tensor / Shape                                                                  |
|------|-----------------------------|---------------------------------------------------------------------------------|
| 1    | Concatenate all concepts    | `concepts = cat(C_0, ..., C_{K-1})` -> `[B, total_C, D]`                        |
| 2    | Back-project to encoder dim | `concept_embeds = back_proj(concepts)` -> `[B, total_C, D_enc]`                 |
| 3    | Embed question tokens       | `Q_embeds = embed_tokens(Q_ids)` -> `[B, L_Q, D_enc]`                           |
| 4    | Embed solution tokens       | `S_embeds = embed_tokens(S_ids)` -> `[B, L_S, D_enc]`                           |
| 5    | Concatenate input           | `input = [Q_embeds, concept_embeds, S_embeds]` -> `[B, L_Q+total_C+L_S, D_enc]` |
| 6    | Build attention mask        | `mask = [Q_mask, ones(total_C), S_mask]` -> `[B, L_Q+total_C+L_S]`              |
| 7    | Forward full reason_model   | `logits = reason_model(inputs_embeds=input)` -> `[B, L_Q+total_C+L_S, V]`       |
| 8    | Extract solution logits     | `sol_logits = logits[:, L_Q+total_C-1 : L_Q+total_C+L_S-1, :]` -> `[B, L_S, V]` |
| 9    | Build targets               | `targets = S_ids` with pad positions set to `-100`                              |
| 10   | Cross-entropy               | `CE(sol_logits, targets, ignore_index=-100)`                                    |

**Why this logit slice?** In a causal LM, logits at position `t` predict the token at position `t+1`. The last concept position `(L_Q + total_C - 1)` predicts `S_0`, and position `(L_Q + total_C + L_S - 2)` predicts `S_{L_S-1}`. So the slice `[L_Q+total_C-1, L_Q+total_C+L_S-1)` gives exactly `L_S` logits aligned with the solution tokens.

### Training loop integration

```python
# train_builder.py L349-354
pyramid = builder(batch)  # encode + pyramid + reasoning (if has_solution)
total_loss, loss_dict = compute_builder_loss(pyramid, loss_weights, ordering_loss_type)
```

- **Gated by data availability** (`batch.has_solution`), not by weight. Even with `reasoning_loss_weight > 0`, if the batch has no solutions, reasoning loss is simply not computed for that batch.
- **Tokenization**: both Q and S are tokenized internally by `forward()` using `self.tokenizer` with `max_length = pyramid_cfg["max_seq_len"]`.
- **All losses computed in one call**: `compute_builder_loss()` in `losses.py` handles recon, ordering, residual, and reasoning (if `pyramid.reasoning_logits` is populated).

### Key details

- **Uses the FULL `reason_model`** (backbone + lm_head): `self.reason_model(inputs_embeds=...)`, not just the backbone.
- **`reason_model` is frozen**: all params have `requires_grad=False`. Gradients flow through `inputs_embeds` (the concatenated Q+concept+S embeddings), reaching `back_proj` and upstream pyramid parameters.
- **`back_proj` is shared** with recon_loss: same layer maps concepts to D_encoder space for both reconstruction and reasoning.
- **`ignore_index=-100`**: padding positions in solution_ids are set to -100, excluded from cross-entropy.
- **Teacher-forced argmax decode**: after computing logits, `argmax(dim=-1)` + `tokenizer.batch_decode()` produces `reasoning_texts` (List[str]) stored in PyramidOutput for eval logging.

### Gradient flow

```
L_reasoning -> CE(sol_logits, targets)
  -> sol_logits = logits[:, L_Q+total_C-1 : L_Q+total_C+L_S-1, :]
  -> logits = reason_model(inputs_embeds)  [frozen: grad passes through inputs_embeds only]
  -> decoder_input_embeds = [Q_embeds, concept_embeds, S_embeds]
    -> Q_embeds = embed_tokens(Q_ids) [frozen: no grad]
    -> concept_embeds = back_proj(concepts)
      -> back_proj.weight
      -> concepts = cat(C_0, ..., C_{K-1})
        -> C_k = level_proj_k(A_k @ f_rest_k)  [purely residual]
          -> level_projs[k], A_k, f_rest_k -> input_proj, input_proj_norm
    -> S_embeds = embed_tokens(S_ids) [frozen: no grad]
```

Note: `reason_model` parameters do NOT receive gradients (frozen). `embed_tokens` is part of the frozen model, so both Q_embeds and S_embeds have no grad. Only `concept_embeds` carries gradients backward.

---

## 5. Stage 1 Total Loss Assembly

### In `compute_builder_loss` (all four losses)

**Code**: [`losses.py` L103–210](examples/lcp/losses.py#L103-L210)

```python
total_loss = (
    loss_weights["recon_loss_weight"]    * recon_loss       # L2 in D_encoder space
    + loss_weights["ordering_loss_weight"] * ordering_loss   # margin-based positional ordering
    + loss_weights["residual_loss_weight"] * res_loss        # L1 in D space
)
# Reasoning loss added if pyramid.reasoning_logits is populated
if pyramid.reasoning_logits is not None:
    total_loss += loss_weights["reasoning_loss_weight"] * reasoning_loss
```

All four losses are computed inside `compute_builder_loss()`. The training loop simply calls:

```python
pyramid = builder(batch)  # encode + pyramid + reasoning (if has_solution)
total_loss, loss_dict = compute_builder_loss(pyramid, loss_weights, ordering_loss_type)
```

---

## 6. Predictor Losses (Stage 2)

The ConceptPredictor is trained with a weighted sum of two components assembled in `compute_predictor_loss` ([`losses.py` L308–378](examples/lcp/losses.py#L308-L378)):

```
L_predictor = concept_loss_weight   × L_concept
            + reasoning_loss_weight × L_reasoning   (added only if reasoning_logits is populated)
```

Both components come from the **same unified teacher-forced forward** through the backbone over `[Q, C_gt, S]` (see [`concept_predictor.py` L768–956](examples/lcp/concept_predictor.py#L768-L956)). No separate pass is needed.

### 6.1 Concept Reconstruction Loss (`concept_loss`)

#### Formula

For a pyramid of `K` levels with per-level sizes `L_0, ..., L_{K-1}`:

$$L_\text{concept} = \frac{1}{K} \sum_{k=0}^{K-1} \ell\bigl(\hat{C}_k,\ \mathrm{sg}[C_k]\bigr)$$

where `sg[·]` is `.detach()` (stop-gradient into the frozen Builder) and `ℓ` is one of:

| `concept_loss_type` | Per-level loss                                           | Source                                                   |
|---------------------|----------------------------------------------------------|----------------------------------------------------------|
| `"mse"` (default)   | `F.mse_loss(Ĉ_k, C_k.detach())`                          | [`losses.py` L232–242](examples/lcp/losses.py#L232-L242) |
| `"cosine"`          | `mean(1 − cosine_similarity(Ĉ_k, C_k.detach(), dim=-1))` | [`losses.py` L245–259](examples/lcp/losses.py#L245-L259) |

Per-level losses are averaged uniformly across K levels; no per-level weighting is used in the current implementation. Per-level scalars are also exposed in `loss_dict["concept_per_level"]` for monitoring.

#### What it measures

The predictor's ability to **reproduce the frozen Builder's pyramid from `Q` alone**, level by level. Analogous to VAR Stage-2's next-scale token prediction loss, but operating directly in concept space (continuous vectors) rather than codebook indices.

#### Data flow (teacher-forced, unified single pass)

| Step | Operation                                                         | Tensor / Shape                                               | Code                                                                                                   |
|------|-------------------------------------------------------------------|--------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| 1    | Flatten GT concepts level-major                                   | `concepts_flat = cat(C_0, ..., C_{K-1})` → `[B, total_C, D]` | [`concept_predictor.py` L851](examples/lcp/concept_predictor.py#L851)                                  |
| 2    | Back-decode + slot markers                                        | `concept_embeds` → `[B, total_C, D_enc]`                     | [`concept_predictor.py` L639–701](examples/lcp/concept_predictor.py#L639-L701)                         |
| 3    | Embed Q (shared embed_tokens)                                     | `Q_embeds` → `[B, L_Q, D_enc]`                               | [`concept_predictor.py` L625–637](examples/lcp/concept_predictor.py#L625-L637)                         |
| 4    | Embed S (shared embed_tokens, optional)                           | `S_embeds` → `[B, L_S, D_enc]`                               | [`concept_predictor.py` L866–872](examples/lcp/concept_predictor.py#L866-L872)                         |
| 5    | Per-row pack `[Q, C, S]` (no internal pad)                        | `pack.packed_embeds` → `[B, T, D_enc]`                       | `pack_qcs_sequences` at [`concept_predictor.py` L885–891](examples/lcp/concept_predictor.py#L885-L891) |
| 6    | Forward full `reason_model` (causal, `output_hidden_states=True`) | `hidden` → `[B, T, D_enc]` + `logits` → `[B, T, V]`          | [`concept_predictor.py` L899–904](examples/lcp/concept_predictor.py#L899-L904)                         |
| 7    | Per-row gather at concept positions                               | `readout` → `[B, total_C, D_enc]`                            | `gather_concept_readout` at [`concept_predictor.py` L912](examples/lcp/concept_predictor.py#L912)      |
| 8    | `concept_head` MLP (`D_enc → D`)                                  | `flat_predicted` → `[B, total_C, D]`                         | [`concept_predictor.py` L913](examples/lcp/concept_predictor.py#L913)                                  |
| 9    | Split into K per-level tensors                                    | `predicted_concepts[k]` → `[B, L_k, D]`                      | [`concept_predictor.py` L918–922](examples/lcp/concept_predictor.py#L918-L922)                         |
| 10   | Per-level MSE (or cosine) vs `C_k.detach()`, then mean            | scalar                                                       | [`losses.py` L262–305](examples/lcp/losses.py#L262-L305)                                               |

#### Key design decisions

- **Teacher-forcing from GT concepts.** Level-k prediction is conditioned on *ground-truth* levels `0..k−1` rather than on predictions. This aligns Stage-2 training with VAR's teacher-forced next-scale regime.
- **Per-row packing** ([`pack_qcs_sequences`](examples/lcp/utils.py)) eliminates geometry bugs that arise when a legacy concat-then-slice concatenated right-padded Q with the concept block: under variable `L_Q`, the single batch-uniform slice offset would point into Q's padding region for short rows, reading concept hidden states off *pad tokens*. Per-row packing guarantees every row has no internal padding.
- **Slot markers (`level_embeddings + position_embeddings`)** are added on top of `back_proj(concepts_flat)` so the backbone can distinguish slot `(level=k, pos=j)` from any other slot purely from the input embedding, independent of absolute sequence position ([`concept_predictor.py` L684–701](examples/lcp/concept_predictor.py#L684-L701)).
- **Detach on target.** The target `C_k` is explicitly `.detach()`-ed inside `compute_predictor_concept_loss` as a defensive measure; the GT already comes from a frozen Builder.
- **Non-AR training but AR inference** — `_forward_training` does ONE parallel causal pass; `_forward_inference` is a 63-step loop with KV cache ([`concept_predictor.py` L962–L1123](examples/lcp/concept_predictor.py#L962-L1123)).

#### Gradient flow

```
L_concept → MSE/cosine(Ĉ_k, C_k.detach())
  → Ĉ_k = split(concept_head(readout))
    → concept_head [D_enc → D_enc → D]           (ALWAYS trainable)
    → readout = gather(hidden, pack)
      → hidden = reason_model(pack.packed_embeds).hidden_states[-1]
        → reason_model                            (frozen or LoRA-only)
        → pack.packed_embeds
          → Q_embeds = embed_tokens(Q_ids)        (frozen, no grad)
          → concept_embeds = back_decode(concepts_flat) + lvl_emb + pos_emb
            → back_decode ≡ back_proj             (shared+frozen OR independent+trainable)
            → level_embeddings                    (ALWAYS trainable)
            → position_embeddings                 (ALWAYS trainable)
          → S_embeds = embed_tokens(S_ids)        (unused for L_concept, fine to include)
```

The concept loss does **NOT** flow into `C_k` (detached) nor into the Builder's pyramid. Gradients reach the predictor's head, embeddings, and — only in INDEPENDENT mode — its own `back_proj` and LoRA adapters.

### 6.2 Reasoning Loss (`reasoning_loss`) — Predictor

#### Formula

$$L_\text{reasoning} = \text{CrossEntropy}\bigl(\text{reasoning\_logits},\ \text{reasoning\_target\_ids}\bigr)$$

with `ignore_index=-100` on solution-pad positions ([`losses.py` L354–368](examples/lcp/losses.py#L354-L368)).

#### What it measures

Whether concepts teacher-forced from the Builder, placed between `Q` and `S` in the unified causal chain, can **predict the correct solution tokens** *under the predictor's own backbone (possibly LoRA-adapted)*. For SHARED mode this is essentially an inherited Stage-1 objective; for INDEPENDENT mode it co-trains the predictor's own reason_model / LoRA adapters with its concept-generation path.

#### Data flow

Produced by the **same forward** as `L_concept`:

| Step | Operation                                 | Tensor / Shape                    | Code                                                                                                    |
|------|-------------------------------------------|-----------------------------------|---------------------------------------------------------------------------------------------------------|
| 1–5  | (same as §6.1 steps 1–5)                  | —                                 | —                                                                                                       |
| 6    | Full `reason_model` forward → `logits`    | `[B, T, V]`                       | [`concept_predictor.py` L937](examples/lcp/concept_predictor.py#L937)                                   |
| 7    | Per-row gather at solution positions      | `solution_logits` → `[B, L_S, V]` | `gather_solution_logits` [`concept_predictor.py` L938](examples/lcp/concept_predictor.py#L938)          |
| 8    | Build targets with `-100` on pad          | `targets` → `[B, L_S]`            | `build_solution_targets` [`concept_predictor.py` L942–944](examples/lcp/concept_predictor.py#L942-L944) |
| 9    | `F.cross_entropy(..., ignore_index=-100)` | scalar                            | [`losses.py` L356–360](examples/lcp/losses.py#L356-L360)                                                |

**Why the per-row gather?** In a right-padded batch, position-wise slicing would read a mix of Q pad tokens, concept tokens, and S pad tokens at the same offset across rows. The per-row gather uses each row's real `q_len[i]` so that row `i` reads logits at `q_len[i] + total_C − 1 + j` for `j = 0..L_S−1`, aligning logits with solution tokens under the causal "position t predicts t+1" rule.

#### Key design decisions

- **Unified forward, not a second pass.** Running reasoning CE in the same forward as the concept MSE saves one backbone forward per step and guarantees the hidden states / logits come from the same weights. Branch drift is eliminated.
- **Only defined in the teacher-forced path.** `_forward_inference` explicitly refuses `solution_ids` because during AR inference the concepts are predicted, not ground truth, and a reasoning CE on a still-forming pyramid is not a meaningful training signal.
- **`reasoning_texts`**. An `argmax` decode of `solution_logits` is also stored on `PredictorOutput.reasoning_texts` under `no_grad()` for qualitative inspection ([`concept_predictor.py` L950–954](examples/lcp/concept_predictor.py#L950-L954)).

#### Gradient flow

```
L_reasoning → CE(solution_logits, solution_ids)
  → logits = reason_model(pack.packed_embeds).logits
    → reason_model                           (frozen / LoRA in INDEPENDENT, frozen in SHARED)
    → pack.packed_embeds
      → Q_embeds (frozen)
      → concept_embeds = back_decode(C_gt) + lvl_emb + pos_emb
        → C_gt is .detach()-ed upstream so NO grad into Builder
        → back_decode ≡ back_proj             (shared+frozen OR independent+trainable)
        → level_embeddings / position_embeddings (trainable)
      → S_embeds (frozen)
```

Crucially, the reasoning CE does **NOT** flow into the `concept_head` MLP (the concept readout is never fed back into the packed input). `concept_head` is trained purely by `L_concept`.

### 6.3 Stage 2 Total Loss Assembly

**Code**: [`losses.py` L308–378](examples/lcp/losses.py#L308-L378)

```python
# Concept component (skipped if gt_concepts is None or empty)
total = concept_loss_weight * L_concept

# Reasoning component (skipped if reasoning_logits / target_ids absent)
total += reasoning_loss_weight * L_reasoning
```

The training loop ([`train_predictor.py`](examples/lcp/train_predictor.py)):

```python
output = predictor(
    question_ids=batch.question_ids,
    question_attention_mask=batch.question_attention_mask,
    gt_concepts=gt_concepts,              # from frozen builder, detached
    solution_ids=batch.solution_ids,
    solution_attention_mask=batch.solution_attention_mask,
)
loss, loss_dict = compute_predictor_loss(
    output,
    loss_weights=cfg["training"]["loss_weights"],
    concept_loss_type=cfg["training"].get("concept_loss_type", "mse"),
)
```

Default weights from the GSM8K configs: `concept_loss_weight: 1.0`, `reasoning_loss_weight: 1.0` ([`configs/lcp/GSM8K/train_predictor_Qwen2.5-0.5B_2level_shared.yml#L121-L123`](configs/lcp/GSM8K/train_predictor_Qwen2.5-0.5B_2level_shared.yml#L121-L123)).

---

## 7. Trainable Parameters Summary

### 7.1 Stage 1 (Builder)

All four Builder losses share (subsets of) the same trainable parameter set:

| Parameter                    | Shape                 | Updated by which losses              |
|------------------------------|-----------------------|--------------------------------------|
| `input_proj.weight`          | `[D, D_encoder]`      | recon, ordering, residual, reasoning |
| `input_proj.bias`            | `[D]`                 | recon, ordering, residual, reasoning |
| `input_proj_norm.weight`     | `[D]`                 | recon, ordering, residual, reasoning |
| `input_proj_norm.bias`       | `[D]`                 | recon, ordering, residual, reasoning |
| `concept_queries[k]`         | `[L_k, D]` × K levels | recon, ordering, residual, reasoning |
| `temperature`                | `[1]`                 | recon, ordering, residual, reasoning |
| `level_projs[k].weight/bias` | `[D, D]` × K levels   | recon, residual, reasoning           |
| `back_proj.weight`           | `[D_encoder, D]`      | **recon, reasoning only**            |

#### Frozen / LoRA-configurable parameters (Builder)

| Parameter                      | Frozen?                                                         | Role                                                                          |
|--------------------------------|-----------------------------------------------------------------|-------------------------------------------------------------------------------|
| `reason_model` (all weights)   | Configurable via `training.reason_model.freeze` (default: true) | Encoding: backbone produces H_CoT; Decoding: lm_head produces solution logits |
| `reason_model` + LoRA adapters | LoRA params trainable if `training.reason_model.lora` is set    | Fine-tune backbone representation with PEFT                                   |

Freezing is controlled by `training.reason_model.freeze` (default: true for VAR-faithful behaviour). LoRA config is at `training.reason_model.lora`. Code: [`concept_builder.py`](examples/lcp/concept_builder.py) (`_init_reason_model`).

### 7.2 Stage 2 (Predictor) — SHARED mode (`use_shared_model: true`)

In SHARED mode the predictor *aliases* the Builder's `reason_model`, `tokenizer`, and `back_proj` as module attributes (weight-tied, not copied). Only the three predictor-owned heads are trainable; the Builder's weights must remain strictly frozen, otherwise gt_concepts (produced each batch by the Builder) would start chasing a moving target.

| Parameter                     | Shape               | Updated by which losses | Notes                                |
|-------------------------------|---------------------|-------------------------|--------------------------------------|
| `level_embeddings.weight`     | `[K, D_enc]`        | concept, reasoning      | One marker per pyramid level         |
| `position_embeddings.weight`  | `[max(L_k), D_enc]` | concept, reasoning      | One marker per intra-level position  |
| `concept_head[0].weight/bias` | `[D_enc, D_enc]`    | **concept only**        | Linear → GELU → Linear MLP           |
| `concept_head[2].weight/bias` | `[D_enc, D]`        | **concept only**        | Maps backbone hidden → concept space |

| Aliased (frozen, unconditionally)     | Role                                                                                       |
|---------------------------------------|--------------------------------------------------------------------------------------------|
| `reason_model` = builder.reason_model | Backbone + lm_head (produces both hidden states and solution logits).                      |
| `tokenizer`    = builder.tokenizer    | Q / S tokenisation.                                                                        |
| `back_proj`    = builder.back_proj    | `D → D_enc` lift used in both the concept-input path and the Builder's own reasoning path. |

**Fail-fast constraint**: `use_shared_model=True` ⇒ `training.predictor.lora` **must be null**. The predictor's `__init__` enforces this because wrapping the shared `reason_model` with LoRA would leak gradients into the Builder's forward and violate the frozen-target invariant. Config comments: [`train_predictor_Qwen2.5-0.5B_2level_shared.yml#L98-L102`](configs/lcp/GSM8K/train_predictor_Qwen2.5-0.5B_2level_shared.yml#L98-L102).

### 7.3 Stage 2 (Predictor) — INDEPENDENT mode (`use_shared_model: false`)

In INDEPENDENT mode the predictor owns its own `reason_model` (loaded fresh from `model_name`) and its own `back_proj`. The Builder continues to run forward through its *own* module tree to produce `gt_concepts`, so the two backbones do not share parameters and LoRA updates on the predictor's backbone cannot corrupt the Builder.

| Parameter                     | Shape                          | Updated by which losses | Notes                                                                                                                                                    |
|-------------------------------|--------------------------------|-------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `level_embeddings.weight`     | `[K, D_enc]`                   | concept, reasoning      | Same as SHARED                                                                                                                                           |
| `position_embeddings.weight`  | `[max(L_k), D_enc]`            | concept, reasoning      | Same as SHARED                                                                                                                                           |
| `concept_head[0].weight/bias` | `[D_enc, D_enc]`               | **concept only**        | Same as SHARED                                                                                                                                           |
| `concept_head[2].weight/bias` | `[D_enc, D]`                   | **concept only**        | Same as SHARED                                                                                                                                           |
| `back_proj.weight`            | `[D_enc, D]`                   | concept, reasoning      | Predictor's **own** copy, learned from scratch. Bias-free linear.                                                                                        |
| `reason_model` LoRA adapters  | depends on `r, target_modules` | concept, reasoning      | Activated when `training.predictor.lora` is non-null. Base weights are frozen by `_init_reason_model` but `lora_*` parameters keep `requires_grad=True`. |

| Frozen (INDEPENDENT)               | Role                                            |
|------------------------------------|-------------------------------------------------|
| `reason_model` base weights        | Only LoRA adapters train (standard PEFT setup). |
| `embed_tokens` inside reason_model | Used for both Q and S embedding — no grad.      |

**Note**: the Builder is still loaded (frozen) even in INDEPENDENT mode because gt_concepts must be recomputed each batch from the same CoT encoder path. This is what produces the two "Loading weights" passes at startup in INDEPENDENT runs (one for the Builder's reason_model, one for the predictor's independent reason_model).

---

## 8. Architectural Comparison with VAR (Stage 1 + Stage 2)

### 8.1 Stage 1: Builder vs VQ-VAE

| Aspect              | VAR (VQVAE Stage 1)                                       | lcp Builder                                              |
|---------------------|-----------------------------------------------------------|----------------------------------------------------------|
| Encoder             | CNN encoder (trainable)                                   | Frozen LLM backbone                                      |
| Target              | Encoder output `f_BChw`                                   | `H_CoT` from frozen backbone                             |
| Decomposition space | Same dim as encoder (C=32)                                | Reduced dim D (e.g., 256) via `input_proj`               |
| Dimension change?   | **No** — `quant_conv` is `Conv2d(C, C)`                   | **Yes** — D_encoder (e.g., 1536) → D (e.g., 256)         |
| Reconstruction      | `F.mse_loss(f_hat, f_BChw)` in C space                    | `MSE(back_proj(f_hat), H_CoT)` round-trip through D      |
| Quantization        | Hard (nearest codebook) + STE                             | Soft (attention-weighted pooling)                        |
| Additional losses   | VQ commitment loss (`β × MSE`)                            | ordering, residual, reasoning                            |
| Code ref            | [`quant.py` L95](third-part/VAR-main/models/quant.py#L95) | [`losses.py` L103–210](examples/lcp/losses.py#L103-L210) |

#### Key difference: round-trip reconstruction

VAR operates entirely in the encoder's native dimension — no dimension change, no back-projection:
```
f_BChw [B,C,H,W] → quantize → f_hat [B,C,H,W] → MSE(f_hat, f_BChw)
```

lcp must project down then back up because D_encoder ≠ D:
```
H_CoT [B,L,D_enc] → input_proj → H_proj [B,L,D] → pyramid → f_hat [B,L,D] → back_proj → [B,L,D_enc] → MSE(·, H_CoT)
```

This means `back_proj` must learn a meaningful inverse of `input_proj`. The initialization `back_proj.weight = input_proj.weight^T` ([`concept_builder.py`](examples/lcp/concept_builder.py) in `__init__`) provides a starting point.

### 8.2 Stage 2: Predictor vs VAR Transformer

| Aspect                 | VAR Stage-2 Transformer                                          | lcp Predictor                                                     |
|------------------------|------------------------------------------------------------------|-------------------------------------------------------------------|
| Condition              | Class label embedding                                            | `Q` token sequence (embedded via shared `embed_tokens`)           |
| Target                 | Discrete codebook indices `idx_k` per scale                      | Continuous concept vectors `C_k` per level                        |
| Tokens per scale/level | `L_k²` (spatial)                                                 | `L_k` (1-D)                                                       |
| Primary loss           | Cross-entropy over codebook (`V` classes)                        | **MSE** (per-level, averaged) on concept vectors                  |
| Auxiliary loss         | —                                                                | **Reasoning CE** on solution tokens in same forward               |
| Teacher-forced input   | Scale-by-scale embedded indices                                  | Flat `back_proj(C_gt) + lvl_emb + pos_emb` over all `Σ L_k` slots |
| Causal structure       | Scale-level causal mask (within-scale full, across-scale causal) | 1-D token-level causal via backbone's standard causal mask        |
| Backbone               | Dedicated transformer with scale embeddings                      | Reused LLM backbone (shared with Builder OR independent + LoRA)   |
| Inference              | AR over scales with KV cache                                     | AR over 63 flat slots with KV cache + explicit `position_ids`     |

**Takeaway.** lcp Stage-2 is shaped like a continuous-valued VAR Stage-2, with two twists:
1. The "codebook" is replaced by a continuous concept space (hence MSE instead of CE for the primary loss).
2. An **auxiliary reasoning CE** is bolted onto the same forward so the predictor's concept tokens remain anchored to solution-generation utility — the same anchor Stage-1 already uses.

---

## 9. Discussion Points

### 9.1 `back_proj` is critical and serves dual roles (both stages)

`back_proj` is used in **two distinct paths** in Stage 1, and again as the input-lift in Stage 2:

| Role                       | Path                                                        | Loss                                  |
|----------------------------|-------------------------------------------------------------|---------------------------------------|
| Stage-1 reconstruction     | `back_proj(f_hat_K)` vs `H_CoT`                             | `recon_loss`                          |
| Stage-1 reasoning          | `back_proj(cat(C_0,...,C_{K-1}))` → `reason_model`          | `reasoning_loss` (S1)                 |
| Stage-2 concept input lift | `back_proj(cat(C_gt)) + lvl_emb + pos_emb` → `reason_model` | `concept_loss`, `reasoning_loss` (S2) |

In SHARED mode the **same tensor** carries all three roles; in INDEPENDENT mode the Stage-2 `back_proj` is a fresh linear layer that must learn `D → D_enc` from scratch while the Stage-1 `back_proj` stays frozen.

### 9.2 `residual_loss` and `recon_loss` are complementary, not redundant

Both measure Builder reconstruction quality, but in different spaces with different norms:

- `recon_loss` (L2 in D_encoder) forces the full round-trip to preserve information. It's the only loss that trains `back_proj` toward a good inverse.
- `residual_loss` (L1 in D) directly regularizes the concept-space decomposition without involving `back_proj`. It provides a cleaner signal for the pyramid mechanics.

In principle, if `back_proj` is perfect, minimizing `residual_loss` implies minimizing `recon_loss`. In practice, the two provide complementary gradient signals. `residual_loss` can be disabled (`weight=0`) if `recon_loss` alone drives sufficient convergence.

### 9.3 Ordering loss is structurally independent

The ordering loss depends **only on attention weights `A_k`**, not on concept quality, reconstruction, or `back_proj`. It can be fully satisfied even if concepts are semantically meaningless — as long as they attend to monotonically increasing positions. This is by design: it's a structural regulariser that prevents degenerate attention patterns, not a quality metric.

### 9.4 Reasoning loss is the only semantic anchor (both stages)

Without `reasoning_loss`, the pyramid could converge to a mathematically valid decomposition that is semantically empty — concepts that perfectly reconstruct H_CoT but carry no useful reasoning information. The Stage-1 reasoning loss forces the concept representation to support actual solution prediction. The Stage-2 reasoning loss, attached to the *same* unified forward as the concept MSE, ensures the predictor's own backbone (and LoRA in INDEPENDENT mode) stays aligned with solution generation while it learns to reproduce the pyramid.

### 9.5 All Builder parameters are trained by the same losses

Since the architecture is purely residual (no cross-attention refinement layers), every trainable parameter in the Builder receives gradients from at least two loss components. There are no dead parameters that depend on a single loss weight being non-zero. This simplifies hyperparameter tuning — setting any individual loss weight to zero only reduces gradient signal, it does not create untrained parameters.

### 9.6 Predictor concept_head is trained ONLY by the concept loss

Unlike the Builder, the Predictor has one parameter block (`concept_head`) that is driven by a **single** loss component (`L_concept`). The reasoning CE cannot reach `concept_head` because the concept readout is never fed back into the packed input during the unified training forward. If `concept_loss_weight` is set to 0, `concept_head` receives no gradient signal and will stay at its initial values. In practice both predictor weights should be kept strictly positive.

### 9.7 SHARED vs INDEPENDENT — picking between them

| Criterion                          | SHARED                                      | INDEPENDENT                                       |
|------------------------------------|---------------------------------------------|---------------------------------------------------|
| Parameter count trained in Stage 2 | tiny (`level_emb + pos_emb + concept_head`) | tiny + `back_proj` + LoRA (controlled by `r`)     |
| GPU memory                         | 1× backbone (shared alias)                  | 2× backbone (Builder + Predictor)                 |
| Startup disk cost                  | One `from_pretrained`                       | Two `from_pretrained` calls (Builder + Predictor) |
| Risk of Builder contamination      | None by construction; **LoRA disallowed**   | None (separate module tree); LoRA safe            |
| Adaptation capacity                | Only the heads adapt                        | Heads + own `back_proj` + LoRA adapt              |
| Typical use                        | Fast sanity check, pure pyramid-prediction  | Full capability, reasoning-aligned fine-tuning    |

The two variants are explicit A/B pairs in the config tree (e.g. `train_predictor_*_shared.yml` vs `train_predictor_*_independent.yml`).
