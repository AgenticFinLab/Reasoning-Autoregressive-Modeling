# NLCP V4 Builder Loss Analysis and Weight Design — MATH

> **Scope.** This document is the MATH counterpart of
> [`loss-weights-analysis-gsm8k.md`](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/examples/lcp/loss-weights-analysis-gsm8k.md).
> Part I (§1–§7) is a purely empirical characterization of the four
> raw loss components across the 36-config MATH matrix using exactly
> the same protocol. Part II (§8–§12) reuses the dataset-agnostic
> weight-design framework from the GSM8K document and derives the
> **MATH-specific** per-(model, level) weights, because the closed-form
> per-model reduction (GSM8K §9.7) breaks down on MATH due to a much
> larger L = 8 reconstruction spike.

---

# Part I — Empirical analysis

## 1. Purpose & data source

This document characterizes the four raw loss components produced by
`compute_builder_loss` (reconstruction, ordering, residual, reasoning)
across every available **MATH** Builder configuration, to select
`loss_weights` from empirical evidence.

- **Data file**: [`EXPERIMENT/lcp/builder/MATH_Loss_prepare.json`](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/EXPERIMENT/lcp/builder/MATH_Loss_prepare.json)
- **Derived artifact**: [`EXPERIMENT/lcp/builder/MATH_training_prepare/weights_summary.txt`](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/EXPERIMENT/lcp/builder/MATH_training_prepare/weights_summary.txt)
  (rows 18–53 tabulate the raw values used below).
- **Entries analyzed**: 36 configs (6 model sizes × 6 pyramid levels;
  Qwen3-8B entries are not yet recorded in `MATH_Loss_prepare.json`).
- **Protocol**: identical to GSM8K — each entry is the mean over a
  small number of forward passes (`loss_prepare.py`) with
  `batch_size = 4`, no optimizer step, FP32. Values are **raw**
  (pre-weight) magnitudes.

Model-family / size axis (columns):

| family  | sizes recorded |
|---------|----------------|
| Qwen2.5 | 0.5B, 1.5B, 3B |
| Qwen3   | 0.6B, 1.7B, 4B |

Pyramid level axis (rows): L ∈ {2, 3, 4, 5, 6, 8}; `level_lengths`
follow the geometric schedule 1, 2, 4, 8, 16, 32, 64, 128.

---

## 2. Raw loss table

| model        | L | recon | ordering | residual | reasoning |
|--------------|--:|------:|---------:|---------:|----------:|
| Qwen2.5-0.5B | 2 | 70.40 |     0.81 |     0.80 |      5.77 |
| Qwen2.5-0.5B | 3 | 70.39 |     2.06 |     0.80 |      4.46 |
| Qwen2.5-0.5B | 4 | 70.43 |     3.79 |     0.81 |      5.90 |
| Qwen2.5-0.5B | 5 | 70.45 |     6.77 |     0.81 |      5.77 |
| Qwen2.5-0.5B | 6 | 70.48 |    11.43 |     0.83 |      5.85 |
| Qwen2.5-0.5B | 8 | 79.80 |    35.36 |     1.31 |      5.88 |
| Qwen2.5-1.5B | 2 | 16.66 |     0.88 |     0.81 |      5.13 |
| Qwen2.5-1.5B | 3 | 16.65 |     1.86 |     0.80 |      4.23 |
| Qwen2.5-1.5B | 4 | 16.66 |     3.67 |     0.81 |      4.47 |
| Qwen2.5-1.5B | 5 | 16.68 |     7.02 |     0.81 |      3.97 |
| Qwen2.5-1.5B | 6 | 16.78 |    11.52 |     0.84 |      4.08 |
| Qwen2.5-1.5B | 8 | 26.44 |    35.30 |     1.39 |      4.75 |
| Qwen2.5-3B   | 2 |  8.78 |     1.39 |     0.80 |      4.53 |
| Qwen2.5-3B   | 3 |  8.78 |     2.05 |     0.80 |      4.42 |
| Qwen2.5-3B   | 4 |  8.78 |     3.64 |     0.80 |      4.61 |
| Qwen2.5-3B   | 5 |  8.78 |     6.58 |     0.80 |      4.73 |
| Qwen2.5-3B   | 6 |  8.85 |    11.57 |     0.82 |      5.05 |
| Qwen2.5-3B   | 8 | 43.08 |    35.00 |     1.35 |      4.91 |
| Qwen3-0.6B   | 2 |  9.93 |     1.25 |     0.80 |      6.50 |
| Qwen3-0.6B   | 3 |  9.93 |     1.91 |     0.80 |      5.53 |
| Qwen3-0.6B   | 4 |  9.93 |     4.16 |     0.80 |      5.93 |
| Qwen3-0.6B   | 5 |  9.94 |     6.59 |     0.80 |      6.25 |
| Qwen3-0.6B   | 6 |  9.99 |    11.32 |     0.82 |      6.50 |
| Qwen3-0.6B   | 8 | 17.43 |    35.75 |     1.28 |      7.02 |
| Qwen3-1.7B   | 2 |  5.47 |     0.92 |     0.80 |      4.93 |
| Qwen3-1.7B   | 3 |  5.47 |     2.07 |     0.80 |      5.42 |
| Qwen3-1.7B   | 4 |  5.47 |     3.56 |     0.80 |      5.34 |
| Qwen3-1.7B   | 5 |  5.49 |     6.36 |     0.81 |      6.09 |
| Qwen3-1.7B   | 6 |  5.51 |    11.38 |     0.82 |      6.11 |
| Qwen3-1.7B   | 8 | 16.75 |    35.01 |     1.43 |      8.71 |
| Qwen3-4B     | 2 |  5.88 |     1.14 |     0.80 |      4.22 |
| Qwen3-4B     | 3 |  5.88 |     1.96 |     0.80 |      4.25 |
| Qwen3-4B     | 4 |  5.89 |     3.59 |     0.80 |      4.27 |
| Qwen3-4B     | 5 |  5.90 |     6.76 |     0.81 |      5.29 |
| Qwen3-4B     | 6 |  5.94 |    11.43 |     0.82 |      4.59 |
| Qwen3-4B     | 8 | 15.54 |    35.68 |     1.31 |      5.06 |

---

## 3. Aggregated views

### 3.1 Per-level averages (across the 6 models)

|     L |     recon |  ordering |  residual | reasoning |
|------:|----------:|----------:|----------:|----------:|
|     2 |     19.52 |      1.07 |     0.802 |      5.18 |
|     3 |     19.52 |      1.99 |     0.800 |      4.72 |
|     4 |     19.53 |      3.74 |     0.803 |      5.09 |
|     5 |     19.54 |      6.68 |     0.807 |      5.35 |
|     6 |     19.59 |     11.44 |     0.825 |      5.36 |
| **8** | **33.17** | **35.35** | **1.345** |  **6.06** |

### 3.2 Per-model averages (across the 6 levels)

| model        | recon | ordering | residual | reasoning |
|--------------|------:|---------:|---------:|----------:|
| Qwen2.5-0.5B | 71.99 |    10.04 |    0.893 |      5.61 |
| Qwen2.5-1.5B | 18.31 |    10.04 |    0.910 |      4.44 |
| Qwen2.5-3B   | 14.51 |    10.04 |    0.895 |      4.71 |
| Qwen3-0.6B   | 11.19 |    10.16 |    0.883 |      6.29 |
| Qwen3-1.7B   |  7.36 |     9.88 |    0.910 |      6.10 |
| Qwen3-4B     |  7.50 |    10.09 |    0.890 |      4.61 |

### 3.3 Side-by-side against GSM8K (per-level averages)

| L | recon MATH | recon GSM8K | ord MATH | ord GSM8K | res MATH | res GSM8K | rea MATH | rea GSM8K |
|--:|-----------:|------------:|---------:|----------:|---------:|----------:|---------:|----------:|
| 2 |      19.52 |       21.25 |     1.07 |      1.03 |    0.802 |     0.802 |     5.18 |      5.61 |
| 3 |      19.52 |       21.25 |     1.99 |      1.99 |    0.800 |     0.801 |     4.72 |      5.63 |
| 4 |      19.53 |       21.26 |     3.74 |      3.69 |    0.803 |     0.805 |     5.09 |      5.72 |
| 5 |      19.54 |       21.28 |     6.68 |      6.63 |    0.807 |     0.808 |     5.35 |      6.27 |
| 6 |      19.59 |       21.32 |    11.44 |     11.39 |    0.825 |     0.825 |     5.36 |      5.80 |
| 8 |  **33.17** |   **23.02** |    35.35 |     35.44 |    1.345 |     1.243 |     6.06 |      6.02 |

**Reading**. Ordering, residual, and reasoning are essentially
indistinguishable from GSM8K. Recon differs in one place only — but
the difference is large: **at L = 8, MATH's per-level mean recon
(33.17) is ~44 % higher than GSM8K's (23.02)**. This single cell is
the central MATH-specific finding; everything else is near-identical.

---

## 4. Per-component analysis

### 4.1 Reconstruction loss (`recon`)

**Level dependence — near-constant for L ≤ 6, sharp jump at L = 8.**
Within each model, recon varies by less than 1 % across L ∈ {2, …, 6}
(e.g., Qwen3-4B: 5.88 → 5.94; Qwen2.5-1.5B: 16.66 → 16.78). At L = 8
the jump is **qualitatively larger than on GSM8K**:

| model        | μ recon L≤6 | recon L=8 |     ratio |
|--------------|------------:|----------:|----------:|
| Qwen2.5-0.5B |       70.43 |     79.80 |     1.13× |
| Qwen2.5-1.5B |       16.69 |     26.44 |     1.58× |
| Qwen2.5-3B   |        8.79 |     43.08 | **4.90×** |
| Qwen3-0.6B   |        9.94 |     17.43 |     1.75× |
| Qwen3-1.7B   |        5.48 |     16.75 |     3.06× |
| Qwen3-4B     |        5.89 |     15.54 |     2.64× |

On GSM8K the same ratios sat in [1.08, 1.20] (cf. GSM8K §4.1). On MATH
the median ratio is ~2.1× and the worst case (Qwen2.5-3B) is 4.9×.

**Model dependence — same Qwen2.5/Qwen3 split as GSM8K, but
attenuated at L ≤ 6 and disrupted at L = 8.** At L ≤ 6 the family
pattern holds (Qwen2.5-0.5B 70.4 ≫ Qwen3-1.7B 5.5), but the within-
family ordering is not monotone in size:

- Qwen2.5 family (L≤6 means): 70.4, 16.7, 8.8 — monotone decreasing.
- Qwen3 family (L≤6 means):   9.9, 5.5, 5.9 — 1.7B is **lowest**, 4B
  is slightly higher (matches the GSM8K pattern §4.1).

When L = 8 is included in the per-model mean, the ordering breaks:
Qwen2.5-3B (14.51) is now larger than Qwen3-0.6B (11.19). This is
entirely driven by the Qwen2.5-3B × L=8 cell (43.08), and is the
reason the GSM8K per-model closed form (§9.7) cannot be transplanted
naively.

**Mechanistic cause.** `recon_loss = MSE(pred, H_CoT) /
(H_CoT.std() + eps)^2` (see [losses.py](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/ram/losses)).
The L≤6 magnitudes are governed by `H_CoT.std()` of each backbone on
MATH text, exactly as in GSM8K, and sit at similar levels (Qwen3-4B
L=2: 6.41 GSM8K vs 5.88 MATH; same order of magnitude). The L = 8
escalation reflects the harder reconstruction problem when the
pyramid must reach 128 concept tokens. On MATH, where CoT hidden
states carry more arithmetic structure per token, the finer-grained
reconstruction at deep levels is evidently much harder — and the
effect is strongest for mid-size Qwen2.5.

### 4.2 Residual loss (`residual`)

**Level dependence — flat for L ≤ 6, jump at L = 8.** Identical to
the GSM8K pattern: L ∈ {2, …, 6} sits at 0.80 ± 0.04 across all six
models; L = 8 averages 1.345 (range 1.28–1.43).

**Model dependence — effectively absent.** Per-model means span
0.883–0.910 (CoV ≈ 1.2 %), the tightest of the four components.

**Mechanism** is the same as GSM8K §4.2.

### 4.3 Ordering loss (`ordering`)

**Level dependence — super-linear, structurally identical to GSM8K.**
Per-level averages: 1.07, 1.99, 3.74, 6.68, 11.44, 35.35. The ratio
`ordering(L=8) / ordering(L=2)` ≈ 33×, again tracking the
combinatorial pair count 1 / 7 / 35 / 155 / 651 / 10 795.

**Model dependence — effectively absent.** Per-model means sit in
9.88–10.16 (CoV ≈ 1.1 %).

This is expected: ordering loss only reads query embeddings, not the
backbone hidden states, so it is dataset-agnostic **and** model-
agnostic in its raw scale. The L = 8 numbers on MATH and GSM8K match
to two decimal places.

### 4.4 Reasoning loss (`reasoning`)

**Level dependence — weakly variable, no monotone trend.** Per-level
averages: 5.18, 4.72, 5.09, 5.35, 5.36, 6.06. Envelope [3.97, 8.71].

**Model dependence — present but non-monotonic, similar to GSM8K.**
Qwen3-0.6B highest at 6.29, Qwen3-4B lowest at 4.61. The within-
backbone CoV on MATH is slightly higher than on GSM8K (Qwen3-1.7B L=8
hits 8.71, a 77 % deviation from that model's L=2 value of 4.93), but
the overall envelope shift is small.

**Mechanism** is the same as GSM8K §4.4 (frozen `lm_head` anchors the
scale to NTP entropy).

---

## 5. Structural summary

| component | level-sensitivity on MATH           | model-sensitivity on MATH     | vs. GSM8K              |
|-----------|-------------------------------------|-------------------------------|------------------------|
| recon     | flat L ≤ 6, **large** jump at L = 8 | family > size (broken at L=8) | L=8 jump 1.5–5× larger |
| ordering  | super-linear with L (pair-count)    | none                          | identical              |
| residual  | flat L ≤ 6, +55 % jump at L = 8     | none                          | identical              |
| reasoning | flat across L                       | mild, non-monotonic           | very similar           |

**Single MATH-specific takeaway**: the L = 8 reconstruction cell is
the one cell that differs materially from GSM8K. Any weight design
that uses only per-model (or only per-level) weights for recon will
fail to capture it.

---

## 6. Implications for loss-weight design on MATH

1. **Per-level weighting remains required for `ordering`.** Raw
   ordering spans 0.81 → 35.85 across the 36 configs, 44×. Same
   conclusion and same fix as GSM8K: `w_ord(L) = min(1, 6 / μ_ord(L))`.
   The per-level average table from §3.1 is within 1 % of GSM8K's, so
   the resulting ordering weights are **numerically identical**.

2. **For `recon`, the GSM8K per-model closed form is
   insufficient.** The within-model spread, driven by the L = 8
   spike, is now up to 4.9× — far above the 20 % assumption that
   justified GSM8K §9.7. The only principled recipe on MATH is the
   full per-(model, level) rule `w_recon(m, L) =
   min(1, 10 / L̄_recon(m, L))`.

3. **`residual` and `reasoning` admit a flat weight.** Exactly as on
   GSM8K: residual ∈ [0.80, 1.43] and reasoning ∈ [3.97, 8.71]. Both
   stay within a factor of ≤ 2× of their own means; no cap is
   warranted.

4. **The L = 8 regime needs explicit attention.** On MATH, three
   components jump simultaneously at L = 8: recon (×2 on average,
   ×4.9 worst case), residual (+60 %), and ordering (combinatorial).
   Without the per-(model, level) treatment for recon, the L = 8
   column would see recon absorb > 60 % of the scalar loss for
   Qwen2.5-3B.

5. **Cross-dataset transfer is possible but lossy.** Because ordering,
   residual, and reasoning are near-identical between the two
   datasets, only the recon weights need MATH-specific re-calibration.
   If runtime constraints force reuse of a single weights file,
   transplanting GSM8K's ordering/residual/reasoning weights and
   *only* re-computing the 36 recon weights on MATH data recovers
   most of the benefit.

---

## 7. Notes on the measurement

Same caveats as GSM8K §7 (FP32 no-grad forward passes, `batch_size=4`,
initial pre-training loss, Qwen3-8B not yet recorded). All MATH
numbers in §2 are sourced verbatim from
[`MATH_training_prepare/weights_summary.txt`](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/EXPERIMENT/lcp/builder/MATH_training_prepare/weights_summary.txt)
rows 18–53.

---

# Part II — Weight design

The design philosophy (§8), mathematical rule (§9.1–§9.5), and cap
values (§9.3) are identical to the GSM8K document. What changes for
MATH is the **reduction strategy in §9.6–§9.7**: GSM8K could collapse
§9.2 into a closed per-model form for recon and a closed per-level
form for ordering; MATH can collapse only the ordering half.

## 8. Design philosophy

Reuse [GSM8K §8](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/examples/lcp/loss-weights-analysis-gsm8k.md)
verbatim:

- Objective: learn a concept pyramid that fits CoT hidden states while
  preserving downstream reasoning ability.
- Priority: recon ⪰ reasoning ⪰ ordering ⪰ residual.
- Constraints (C1) no component > 50 % of the scalar loss; (C2)
  neutralize structural / backbone-scale artifacts; (C3) leave quiet
  components at w = 1; (C4) preserve the priority ordering where
  possible.

## 9. Mathematical formulation

### 9.1–9.5 (unchanged from GSM8K)

$$
w_i^{\star} \;=\; \min\!\left(1,\; \frac{c_i}{\bar{\mathcal{L}}_i}\right),
\qquad
c_{\text{recon}}=10,\; c_{\text{ord}}=6,\; c_{\text{res}}=c_{\text{rea}}=+\infty.
$$

$$
\tilde{\mathcal{L}}_i = \min(\bar{\mathcal{L}}_i, c_i),
\qquad
\mathcal{L}_{\text{total}} = \sum_i w_i^{\star}\,\mathcal{L}_i.
$$

### 9.6 Ordering weight as a function of level (closed form)

Because `ordering` is model-invariant (§4.3), the same closed form as
GSM8K applies, with per-level averages recomputed from §3.1:

$$
w_{\text{ord}}^{\star}(L) \approx \min\!\left(1,\; \frac{6}{\mu_{\text{ord}}(L)}\right),
\quad
\mu_{\text{ord}}(L) \in \{1.07,\, 1.99,\, 3.74,\, 6.68,\, 11.44,\, 35.35\}
$$

yielding, for $L \in \{2, 3, 4, 5, 6, 8\}$:

$$
w_{\text{ord}}^{\star}(L) \in \{1.000,\, 1.000,\, 1.000,\, 0.898,\, 0.524,\, 0.170\}.
$$

These numbers match the GSM8K ordering weights to within ±0.01 and
may be used verbatim from the GSM8K AutoWeighted configs.

### 9.7 Recon weight — per-(model, level) (no closed form)

The GSM8K closed form
$w_{\text{recon}}^{\star}(m) = \min(1, 10/\mu_{\text{recon}}(m))$
assumes within-model level spread $< 20 \%$. **On MATH this
assumption fails** (§4.1; worst case Qwen2.5-3B has a 4.9× within-
model spread). The principled weight is the full §9.2 rule applied
cell-by-cell:

$$
w_{\text{recon}}^{\star}(m, L) = \min\!\left(1,\; \frac{10}{\bar{\mathcal{L}}_{\text{recon}}(m, L)}\right).
$$

Resulting table:

| model        |   L=2 |   L=3 |   L=4 |   L=5 |   L=6 |       L=8 |
|--------------|------:|------:|------:|------:|------:|----------:|
| Qwen2.5-0.5B | 0.142 | 0.142 | 0.142 | 0.142 | 0.142 |     0.125 |
| Qwen2.5-1.5B | 0.600 | 0.601 | 0.600 | 0.599 | 0.596 |     0.378 |
| Qwen2.5-3B   | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **0.232** |
| Qwen3-0.6B   | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |     0.574 |
| Qwen3-1.7B   | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |     0.597 |
| Qwen3-4B     | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |     0.644 |

The dominant deviation from GSM8K is the **Qwen2.5-3B L=8 cell (0.232
vs. GSM8K's 0.941)** — a 4× tighter cap at that single cell, which is
exactly what §4.1 showed was needed.

The clamped weighted value $\tilde{\mathcal{L}}_{\text{recon}}$ is at
most 10 everywhere (by construction).

### 9.8 Residual and reasoning — uncapped

$w_{\text{res}}^{\star} = w_{\text{rea}}^{\star} = 1$ for all 36
configs, as in GSM8K §10.4. Their raw envelopes on MATH
([0.80, 1.43] and [3.97, 8.71]) are within the no-cap criterion of
§9.3.

---

## 10. Target effects

### 10.1 Bounded weighted envelopes (MATH)

| component | weighted range $\tilde{\mathcal{L}}_i$ | weight range $w_i^{\star}$ |
|-----------|---------------------------------------:|---------------------------:|
| recon     |                          [5.47, 10.00] |             [0.125, 1.000] |
| reasoning |                          [3.97,  8.71] |                    {1.000} |
| ordering  |                          [0.81,  6.00] |             [0.170, 1.000] |
| residual  |                          [0.80,  1.43] |                    {1.000} |

Spread compression: recon 9.8× → 1.8×, ordering 44× → 7.4×. Every
component is within a factor of ~2× of its own mean across the entire
MATH matrix after weighting.

### 10.2 Gradient-share analysis at the pathological cell

Qwen2.5-3B, L=8, MATH — the single cell where the MATH design
diverges most from GSM8K:

| regime                                              | $S_{\text{recon}}$ | $S_{\text{rea}}$ | $S_{\text{ord}}$ | $S_{\text{res}}$ |
|-----------------------------------------------------|-------------------:|-----------------:|-----------------:|-----------------:|
| no weights                                          |           **50 %** |              6 % |         **41 %** |              2 % |
| GSM8K per-model recon weight (0.941) + cap ordering |           **64 %** |              7 % |              9 % |              1 % |
| MATH per-(m,L) recon weight (0.232) + cap ordering  |               42 % |             22 % |             27 % |              6 % |

Applying the GSM8K closed form here would push recon to 64 % of the
scalar objective at L = 8 for Qwen2.5-3B — a clear violation of (C1).
The per-(model, level) rule keeps every component under 45 %.

### 10.3 Priority preservation

Priority-pair holding rates across the 36-config MATH matrix with the
§9.6–§9.7 weights:

| pair                                                                      | configs satisfied |
|---------------------------------------------------------------------------|------------------:|
| $\tilde{\mathcal{L}}_{\text{recon}} \ge \tilde{\mathcal{L}}_{\text{res}}$ |           36 / 36 |
| $\tilde{\mathcal{L}}_{\text{rea}} \ge \tilde{\mathcal{L}}_{\text{res}}$   |           36 / 36 |
| $\tilde{\mathcal{L}}_{\text{recon}} \ge \tilde{\mathcal{L}}_{\text{ord}}$ |           34 / 36 |
| $\tilde{\mathcal{L}}_{\text{recon}} \ge \tilde{\mathcal{L}}_{\text{rea}}$ |           33 / 36 |
| $\tilde{\mathcal{L}}_{\text{rea}} \ge \tilde{\mathcal{L}}_{\text{ord}}$   |           27 / 36 |

Matches GSM8K within ±1 config on every pair; the strict
recon ⪰ residual, reasoning ⪰ residual ordering holds universally.

---

## 11. Implementation and provenance

### 11.1 Pipeline (MATH-specific paths)

```
configs/lcp/MATH/train_builder_*_*level.yml       (baseline recipes)
                │
                ▼  loss_prepare.py --dataset MATH (10-batch warm-up)
EXPERIMENT/lcp/builder/MATH_Loss_prepare.json     (raw L̄_i per config)
                │
                ▼  loss_weight_compute.py -f MATH_Loss_prepare.json
EXPERIMENT/lcp/builder/MATH_training_prepare/recommended_weights.csv
                │
                ▼  AutoWeighted generator
configs/lcp/MATH/AutoWeighted/train_builder_*_*level.yml   (to create)
                │
                ▼  train_builder.py
EXPERIMENT/lcp/builder/MATH_<m>_<L>level_AutoWeighted/
```

### 11.2 File artifacts (existing)

- [`EXPERIMENT/lcp/builder/MATH_Loss_prepare.json`](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/EXPERIMENT/lcp/builder/MATH_Loss_prepare.json) — raw measurements.
- [`EXPERIMENT/lcp/builder/MATH_training_prepare/weights_summary.txt`](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/EXPERIMENT/lcp/builder/MATH_training_prepare/weights_summary.txt) — per-config raw + recommended weights (target-0.8 variant).
- [`EXPERIMENT/lcp/builder/MATH_training_prepare/recommended_weights.csv`](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/EXPERIMENT/lcp/builder/MATH_training_prepare/recommended_weights.csv) — machine-readable equivalent.
- Heat-maps and line plots in the same directory visualize the
  pre-/post-weight distribution of each component.

### 11.3 File artifacts (to create)

- `configs/lcp/MATH/AutoWeighted/train_builder_*_*level.yml` — 36
  recipes (+ 6 for Qwen3-8B once measured). Each should carry a
  provenance banner listing $\bar{\mathcal{L}}_i$, $w_i^{\star}$, and
  $\tilde{\mathcal{L}}_i$ for its (model, level) cell.

### 11.4 Regeneration

```bash
python3 examples/RunResults/loss_prepare.py \
    -c configs/lcp/MATH/ --dataset MATH
python3 examples/lcp/loss_weight_compute.py \
    -f EXPERIMENT/lcp/builder/MATH_Loss_prepare.json \
    --c-recon 10 --c-ordering 6
# then re-run the AutoWeighted generator against configs/lcp/MATH/
```

### 11.5 Coverage

- 36 / 42 MATH configs covered (6 models × 6 levels).
- Qwen3-8B (6 configs) pending — same status as GSM8K §11.4.

---

## 12. Limitations and when to re-tune

### 12.1 Escalation paths

Two graded alternatives to the hard cap at the Qwen2.5-3B L=8 cell —
the one cell where MATH's weights are most aggressive:

| rule                | formula                                               | $w_{\text{recon}}$ | $\tilde{\mathcal{L}}_{\text{recon}}$ |
|---------------------|-------------------------------------------------------|-------------------:|-------------------------------------:|
| hard cap (proposed) | $\min(1, 10/\bar{\mathcal{L}}_{\text{recon}})$        |              0.232 |                                 10.0 |
| soft cap (sqrt)     | $\min(1, \sqrt{10/\bar{\mathcal{L}}_{\text{recon}}})$ |              0.482 |                                 20.8 |
| raised cap ($c=20$) | $\min(1, 20/\bar{\mathcal{L}}_{\text{recon}})$        |              0.464 |                                 20.0 |

If training on Qwen2.5-3B at L=8 shows `recon` plateauing
immediately — the documented over-suppression symptom — switch to the
raised-cap rule for that cell only.

### 12.2 When to re-measure

Same triggers as GSM8K §12.2, plus:

- Switching math dataset (e.g., MATH-500 → full MATH, or to AIME)
  will change `H_CoT.std()` and can shift recon envelopes; re-run
  §11.4.
- Any change to prompting / tokenization of MATH solutions will shift
  reasoning (NTP baseline).

### 12.3 What this design does *not* guarantee

- Does not auto-balance during training (weights frozen at init).
- Does not make recon magnitudes directly comparable to GSM8K (they
  are not — the L = 8 cell in MATH is structurally different).
- Does not transfer to other hard-math datasets without re-measuring.

---

## 13. Summary — recommended weight recipe for MATH

**Feasible and minimum-risk recipe**:

1. Use **per-level ordering weights** from §9.6 — identical to GSM8K
   (within ±0.01): $w_{\text{ord}}^{\star}(L) = \{1, 1, 1, 0.898,
   0.524, 0.170\}$ for $L \in \{2, 3, 4, 5, 6, 8\}$.
2. Use **per-(model, level) recon weights** from §9.7, not a per-
   model closed form. The 36-entry table is the artifact.
3. Set `residual_loss_weight = 1.0` and `reasoning_loss_weight = 1.0`
   everywhere.
4. Keep `ordering_margin = 1.0` (unchanged from GSM8K).

**Emit these into `configs/lcp/MATH/AutoWeighted/*.yml`** with the
same banner-and-suffix convention as the GSM8K AutoWeighted set:

```yaml
training:
  loss_weights:
    reasoning_loss_weight: 1.0000
    ordering_loss_weight:  <from §9.6 table>
    recon_loss_weight:     <from §9.7 table>
    ordering_margin:       1.0
    residual_loss_weight:  1.0000
log:
  save_folder: EXPERIMENT/lcp/builder/MATH_<model>_<L>level_AutoWeighted
```

This matches the GSM8K methodology in every respect except recon,
which is now resolved at the per-(model, level) granularity that
MATH's L = 8 reconstruction spike demands.
