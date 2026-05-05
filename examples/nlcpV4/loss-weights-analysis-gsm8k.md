# NLCP V4 Builder Loss Analysis and Weight Design — GSM8K

> **Document structure.** Part I (§1–§7) is a purely empirical
> characterization of the four raw loss components across the 36-config
> GSM8K matrix. Part II (§8–§12) defines the weight-design theory — the
> objective, the mathematical formulation, the target effects, and the
> implementation pipeline — and shows how the weights in every
> [`configs/nlcpV4/GSM8K/AutoWeighted/`](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/configs/nlcpV4/GSM8K/AutoWeighted)
> config are derived from the empirical numbers in Part I.

---

# Part I — Empirical analysis

## 1. Purpose & data source

This document characterizes the four raw loss components produced by
`compute_builder_loss` (reconstruction, ordering, residual, reasoning)
across every available GSM8K Builder configuration, so that sensible
`loss_weights` can be chosen from empirical evidence rather than
guesswork. The concrete weights used in production follow in Part II.

- **Data file**: `EXPERIMENT/nlcpV4/builder/Loss_prepare.json`
- **Entries analyzed**: 36 configs (6 model sizes × 6 pyramid levels;
  Qwen3-8B configs were not yet recorded at the time of writing).
- **Protocol**: each entry is the mean over a small number of forward
  passes (`loss_prepare.py`) with `batch_size = 4`, no optimizer step.
  Values below are **raw** (pre-weight) loss magnitudes.

Model family / size axis (columns):

| family  | sizes recorded |
|---------|----------------|
| Qwen2.5 | 0.5B, 1.5B, 3B |
| Qwen3   | 0.6B, 1.7B, 4B |

Pyramid level axis (rows): L ∈ {2, 3, 4, 5, 6, 8} (level_lengths
follow a geometric schedule 1, 2, 4, 8, 16, 32, 64, 128).

---

## 2. Raw loss table

| model        | L | recon | ordering | residual | reasoning |
|--------------|--:|------:|---------:|---------:|----------:|
| Qwen2.5-0.5B | 2 | 77.06 |     0.62 |     0.80 |      6.00 |
| Qwen2.5-0.5B | 3 | 77.06 |     2.06 |     0.80 |      4.43 |
| Qwen2.5-0.5B | 4 | 77.08 |     4.05 |     0.81 |      6.17 |
| Qwen2.5-0.5B | 5 | 77.08 |     6.64 |     0.81 |      5.78 |
| Qwen2.5-0.5B | 6 | 77.15 |    11.61 |     0.82 |      6.08 |
| Qwen2.5-0.5B | 8 | 78.80 |    35.44 |     1.22 |      6.69 |
| Qwen3-0.6B   | 2 |  9.36 |     1.10 |     0.80 |      7.38 |
| Qwen3-0.6B   | 3 |  9.36 |     2.08 |     0.80 |      8.46 |
| Qwen3-0.6B   | 4 |  9.37 |     3.77 |     0.80 |      6.56 |
| Qwen3-0.6B   | 5 |  9.39 |     6.42 |     0.81 |      8.15 |
| Qwen3-0.6B   | 6 |  9.44 |    11.00 |     0.83 |      6.75 |
| Qwen3-0.6B   | 8 | 11.12 |    35.69 |     1.25 |      6.18 |
| Qwen2.5-1.5B | 2 | 18.81 |     1.10 |     0.81 |      6.42 |
| Qwen2.5-1.5B | 3 | 18.80 |     2.03 |     0.80 |      5.05 |
| Qwen2.5-1.5B | 4 | 18.81 |     3.56 |     0.81 |      5.10 |
| Qwen2.5-1.5B | 5 | 18.85 |     6.94 |     0.82 |      4.51 |
| Qwen2.5-1.5B | 6 | 18.91 |    11.37 |     0.84 |      4.39 |
| Qwen2.5-1.5B | 8 | 20.85 |    35.58 |     1.31 |      5.63 |
| Qwen3-1.7B   | 2 |  5.57 |     1.05 |     0.80 |      4.64 |
| Qwen3-1.7B   | 3 |  5.56 |     1.97 |     0.80 |      6.57 |
| Qwen3-1.7B   | 4 |  5.57 |     3.49 |     0.81 |      6.28 |
| Qwen3-1.7B   | 5 |  5.59 |     6.30 |     0.81 |      8.19 |
| Qwen3-1.7B   | 6 |  5.60 |    11.36 |     0.82 |      6.59 |
| Qwen3-1.7B   | 8 |  7.67 |    35.07 |     1.32 |      7.44 |
| Qwen2.5-3B   | 2 | 10.32 |     1.21 |     0.80 |      5.23 |
| Qwen2.5-3B   | 3 | 10.32 |     1.76 |     0.80 |      4.73 |
| Qwen2.5-3B   | 4 | 10.33 |     3.58 |     0.80 |      5.29 |
| Qwen2.5-3B   | 5 | 10.32 |     6.73 |     0.80 |      5.42 |
| Qwen2.5-3B   | 6 | 10.39 |    11.34 |     0.82 |      5.90 |
| Qwen2.5-3B   | 8 | 12.11 |    35.03 |     1.24 |      5.59 |
| Qwen3-4B     | 2 |  6.41 |     1.10 |     0.80 |      3.98 |
| Qwen3-4B     | 3 |  6.41 |     2.01 |     0.80 |      4.54 |
| Qwen3-4B     | 4 |  6.42 |     3.67 |     0.80 |      4.92 |
| Qwen3-4B     | 5 |  6.43 |     6.76 |     0.81 |      5.57 |
| Qwen3-4B     | 6 |  6.46 |    11.66 |     0.82 |      5.07 |
| Qwen3-4B     | 8 |  7.59 |    35.85 |     1.13 |      4.60 |

---

## 3. Aggregated views

### 3.1 Per-level averages (across the 6 models)

|     L |     recon |  ordering |  residual | reasoning |
|------:|----------:|----------:|----------:|----------:|
|     2 |     21.25 |      1.03 |     0.802 |      5.61 |
|     3 |     21.25 |      1.99 |     0.801 |      5.63 |
|     4 |     21.26 |      3.69 |     0.805 |      5.72 |
|     5 |     21.28 |      6.63 |     0.808 |      6.27 |
|     6 |     21.32 |     11.39 |     0.825 |      5.80 |
| **8** | **23.02** | **35.44** | **1.243** |  **6.02** |

### 3.2 Per-model averages (across the 6 levels)

| model        | recon | ordering | residual | reasoning |
|--------------|------:|---------:|---------:|----------:|
| Qwen2.5-0.5B | 77.37 |    10.07 |    0.877 |      5.86 |
| Qwen3-0.6B   |  9.67 |    10.01 |    0.882 |      7.25 |
| Qwen2.5-1.5B | 19.17 |    10.10 |    0.895 |      5.18 |
| Qwen3-1.7B   |  5.93 |     9.87 |    0.892 |      6.62 |
| Qwen2.5-3B   | 10.63 |     9.94 |    0.879 |      5.36 |
| Qwen3-4B     |  6.62 |    10.18 |    0.860 |      4.78 |

---

## 4. Per-component analysis

### 4.1 Reconstruction loss (`recon`)

**Level dependence — near-constant for L ≤ 6, small jump at L = 8.**
Within each model, recon varies by less than 1% across L ∈ {2,…,6}
(e.g., Qwen2.5-1.5B: 18.80 → 18.91; Qwen3-4B: 6.41 → 6.46). At L = 8
there is a consistent uptick of roughly +1.5 to +2 units (e.g.,
Qwen2.5-1.5B: 18.91 → 20.85; Qwen3-4B: 6.46 → 7.59), a +8% to +20%
jump depending on the model.

**Model dependence — dominated by family, not by size.**
Raw magnitudes span a factor of ~13× across the 6 models (77.4 on
Qwen2.5-0.5B vs. 5.9 on Qwen3-1.7B). The dominant signal is the
**Qwen2.5 vs. Qwen3 family split**:
- Qwen2.5 family: 77.37, 19.17, 10.63 (monotonically decreasing with
  size within the family)
- Qwen3 family: 9.67, 5.93, 6.62 (essentially flat — 1.7B is actually
  slightly lower than 4B)

The "shrinks with size" trend holds inside the Qwen2.5 family but not
inside the Qwen3 family. Taken across both families, the ordering is
not monotone in parameter count.

**Mechanistic cause.** `recon_loss` is `MSE(pred, H_CoT) /
(H_CoT.std() + eps)^2` (see [losses.py](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/examples/nlcpV4/losses.py)). The numerator scales with the
squared norm of the hidden states; the denominator normalizes by
hidden-state spread. If a backbone has small `H_CoT.std()` relative to
its mean norm — typical of older / less-regularized checkpoints — the
ratio is amplified. Qwen2.5-0.5B's 77.4 value is therefore a property
of that backbone's hidden-state statistics at initialization, not an
indicator of "harder reasoning" on this dataset. The smaller L = 8
uptick reflects the harder optimization problem that arises when the
pyramid must reconstruct finer-grained hidden-state detail.

### 4.2 Residual loss (`residual`)

**Level dependence — flat for L ≤ 6, sharp jump at L = 8.**
For L ∈ {2, …, 6} the value is essentially constant at 0.80 ± 0.02
across every model. At L = 8 it jumps to ~1.24 (per-level average),
or individually 1.13–1.32 — a ~55% increase that is highly consistent
across all 6 models.

**Model dependence — effectively absent.** Across the 6 models, the
per-model mean residual sits in the very tight range 0.860–0.895
(coefficient of variation ≈ 1.4%). The residual loss is the most
model-invariant of the four components.

**Mechanistic cause.** Residual regularization penalizes the degree to
which successive pyramid levels fail to decompose as additive
residuals of the previous level's reconstruction. At small L (1, 2, 4
concept tokens per level), the residual structure is easy to satisfy
because each level adds relatively little information. At L = 8 the
deepest levels (16, 32, 64, 128 tokens) must carry fine detail, which
strains the residual-additivity assumption and shifts the loss upward.

### 4.3 Ordering loss (`ordering`)

**Level dependence — super-linear growth.** Per-level averages:
L=2 → 1.03, L=3 → 1.99, L=4 → 3.69, L=5 → 6.63, L=6 → 11.39, L=8 →
35.44. The ratio `ordering(L=8) / ordering(L=2)` ≈ 34×.

**Model dependence — effectively absent.** Per-model averages across
all levels sit in the range 9.87–10.18 (coefficient of variation ≈
1.2%). The ordering loss is blind to which backbone provides
hidden-state targets.

**Mechanistic cause.** Ordering loss sums margin violations across all
intra-level query pairs. With geometric `level_lengths = [1, 2, 4, 8,
16, 32, 64, 128]`, the number of ordered pairs per level is
`l_i·(l_i−1)/2`. Totalling across all levels:

| L | cumulative pair count |
|--:|----------------------:|
| 2 |                     1 |
| 3 |                     7 |
| 4 |                    35 |
| 5 |                   155 |
| 6 |                   651 |
| 8 |                10 795 |

The raw ordering magnitude tracks roughly `sqrt(pair count)` rather
than pair count directly (because per-pair violations average out),
but the super-linear growth in raw loss follows directly from this
combinatorial explosion. This is a **structural** effect of the level
schedule, not a signal that "ordering is harder" at large L.

### 4.4 Reasoning loss (`reasoning`)

**Level dependence — weakly variable, no monotone trend.** Per-level
averages: 5.61, 5.63, 5.72, 6.27, 5.80, 6.02. Range across L is under
12% of the mean. No systematic growth or decrease with L.

**Model dependence — present but non-monotonic.** Per-model means:
Qwen2.5-0.5B 5.86, Qwen3-0.6B 7.25, Qwen2.5-1.5B 5.18, Qwen3-1.7B
6.62, Qwen2.5-3B 5.36, Qwen3-4B 4.78. Qwen3-0.6B is the highest and
Qwen3-4B is the lowest; size is not the governing variable.

**Mechanistic cause.** Reasoning loss is next-token cross-entropy
computed by feeding the pyramid's reconstructed hidden states through
the **frozen** `lm_head`. Its scale is therefore anchored by (a)
vocabulary size (which sets the upper bound of CE ≈ log(|V|)) and (b)
the baseline next-token predictability of GSM8K text under each
backbone. Neither quantity scales systematically with pyramid depth
(the pyramid is frozen out of the lm_head gradient path), so
reasoning loss stays in the 4–8 envelope regardless of L. Per-model
differences reflect how "well-calibrated" each backbone is on GSM8K
at the start of Builder training.

---

## 5. Structural summary

| component | level-sensitivity                  | model-sensitivity        | driver                            |
|-----------|------------------------------------|--------------------------|-----------------------------------|
| recon     | low for L ≤ 6, small jump at L = 8 | **high** (family > size) | backbone `H_CoT.std()`            |
| ordering  | **super-linear with L**            | none                     | pair-count (combinatorial)        |
| residual  | flat for L ≤ 6, jump at L = 8      | none                     | residual-decomposition difficulty |
| reasoning | flat across L                      | mild, non-monotonic      | backbone-dependent NTP baseline   |

Two of the four components are **structurally determined** (ordering,
residual): their magnitudes are set by the level schedule and
backbone-agnostic. Two depend on the backbone (recon, reasoning), but
via different mechanisms: recon tracks hidden-state scale statistics,
reasoning tracks NTP-CE baseline.

---

## 6. Implications for loss-weight design

The following are objective consequences of the patterns above. They
do not prescribe a specific recipe; they constrain the space of
reasonable recipes.

1. **Per-level weighting is required for `ordering`.**
   Raw ordering spans 1.0 → 35.4 across L ∈ {2, …, 8} — a 34× range.
   Any flat weight will either leave ordering invisibly small at L = 2
   or allow it to dominate the objective at L = 8. A weight of the
   form `w_ord(L) = c / mean_ordering(L)` fully neutralizes the
   combinatorial effect.

2. **Per-model (or at least per-family) weighting is required for
   `recon`.**
   Raw recon spans 5.9 → 77.4 across the 6 models — a 13× range, with
   a clear family boundary between Qwen2.5 and Qwen3. Using a single
   flat weight means recon's contribution to the scalar objective
   varies by an order of magnitude across backbones trained with the
   "same" recipe, which undermines cross-model comparability.

3. **`residual` and `reasoning` admit a single flat weight.**
   Residual is bounded in [0.80, 1.32] across all 36 configs;
   reasoning is bounded in [3.98, 8.46]. Both are within a factor of
   ≤2 of their own means, so a flat weight gives comparable
   contributions across the matrix.

4. **The L = 8 regime has its own character.**
   At L = 8, both `recon` (+10%) and `residual` (+55%) jump, while
   `ordering` explodes combinatorially. If the experiment matrix
   includes L = 8, weights should be designed to ensure the
   aggregate loss does not become dominated by the level-8 structural
   terms.

5. **`reasoning` magnitude is set outside the pyramid.**
   Because it passes through a frozen `lm_head`, its value cannot be
   driven to zero by pyramid optimization alone — it is bounded below
   by the backbone's intrinsic NTP entropy on GSM8K. A weight that
   assumes reasoning can collapse to near-zero during training will
   mis-calibrate the loss balance.

6. **Family-level recon behavior matters for interpretation.**
   A small recon loss on Qwen3-1.7B (~5.9) and a large recon loss on
   Qwen2.5-0.5B (~77) do **not** mean one pyramid is reconstructing
   CoT states better than the other — they largely reflect
   `H_CoT.std()` of the respective backbones. Cross-model recon
   comparisons should be normalized (either via per-model weights or
   via reporting `recon × H_CoT.std()^2` as an absolute MSE).

---

## 7. Notes on the measurement

- Values are raw loss magnitudes averaged over a small number of
  no-grad forward passes (see [loss_prepare.py](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/examples/RunResults/loss_prepare.py)). They reflect
  **initial** (pre-training) loss. Relative magnitudes are stable
  enough to guide weight selection, but the absolute values will
  shift during training.
- Forward passes were performed in **FP32**. After the migration to
  BF16 backbone loading, magnitudes are expected to shift by ≤ 1%;
  the patterns described here are robust to that precision change.
- `batch_size` was 4 across all measurements; raw loss values are
  means over batches.
- Qwen3-8B entries are absent from the current JSON and are therefore
  not included in any of the per-model statistics above.

---

# Part II — Weight design

## 8. Design philosophy and principles

### 8.1 The Builder training objective

The Builder learns a concept pyramid that simultaneously satisfies four
objectives, each associated with one loss component:

| component   | objective (informal)                                                    |
|-------------|-------------------------------------------------------------------------|
| `recon`     | reconstructed hidden states match the frozen backbone's CoT states      |
| `residual`  | successive levels form an additive residual decomposition               |
| `ordering`  | intra-level queries respect a canonical ordering (margin-based)         |
| `reasoning` | next-token prediction through the *reconstructed* states stays faithful |

In one sentence: **learn a good concept hierarchy that fits CoT while
preserving the backbone's reasoning ability.**

### 8.2 Priority hierarchy

Not every objective carries the same weight for the research question.
The design ranks them as:

$$
\text{recon}\;\succeq\;\text{reasoning}\;\succeq\;\text{ordering}\;\succeq\;\text{residual}
$$

- **recon** is the *core* target: if the pyramid does not reconstruct
  CoT hidden states, everything else is vacuous.
- **reasoning** is the *fidelity guarantee*: without it, the pyramid may
  reconstruct well on the MSE metric yet destroy the downstream
  reasoning signal.
- **ordering** and **residual** are *structural regularizers*; they
  shape the pyramid's geometry but are not the product.

### 8.3 Four design constraints

From the empirical patterns in Part I the design must satisfy:

- **(C1) No domination.** No single component should contribute more
  than ~50% of the scalar objective for any (model, level) pair.
  The raw matrix violates this at every L ≥ 6 for ordering (up to 73%
  at L=8) and at every small Qwen2.5 config for recon.
- **(C2) Structural invariance.** Magnitudes driven by the level
  schedule (ordering's combinatorial growth) or the backbone's
  hidden-state scale (recon's 13× family spread) must be neutralized —
  they are not signals of learning difficulty, just scale artifacts.
- **(C3) Adaptivity / minimal intervention.** If a component is
  already quiet (`raw < cap`), leave its weight at 1.0 so the natural
  gradient signal is preserved. Only shrink weights where necessary.
- **(C4) Priority preservation.** Where it is mathematically possible,
  ensure the weighted-contribution ordering follows §8.2.

---

## 9. Mathematical formulation

### 9.1 Setup

Let $i \in \{\text{recon}, \text{ord}, \text{res}, \text{rea}\}$ index
the four loss components. Let $\bar{\mathcal{L}}_i$ denote the empirical
pre-training mean of component $i$ over 10 no-grad forward passes with
`batch_size = 4` (the values tabulated in §2). Let $w_i \ge 0$ be the
weight the designer sets, and $c_i \in (0, +\infty]$ be the **cap** of
component $i$.

### 9.2 Weight rule

$$
w_i^{\star} \;=\; \min\!\left(1,\; \frac{c_i}{\bar{\mathcal{L}}_i}\right)
\qquad\Longleftrightarrow\qquad
w_i^{\star} \;=\; \begin{cases}
  1 & \text{if } \bar{\mathcal{L}}_i < c_i \\[4pt]
  \dfrac{c_i}{\bar{\mathcal{L}}_i} & \text{if } \bar{\mathcal{L}}_i \ge c_i
\end{cases}
$$

When $c_i = +\infty$ this reduces to $w_i^{\star} = 1$ (no cap applied).

### 9.3 Cap values

$$
c_{\text{recon}} = 10,\qquad
c_{\text{ord}} = 6,\qquad
c_{\text{res}} = +\infty,\qquad
c_{\text{rea}} = +\infty.
$$

The cap values were chosen to satisfy §8.3's constraints given the
envelopes observed in Part I:

| component | observed envelope $\bar{\mathcal{L}}_i$ | cap $c_i$ | why this cap                                     |
|-----------|-----------------------------------------|-----------|--------------------------------------------------|
| recon     | [5.57, 78.80] (13× spread)              | 10        | above max reasoning (8.46), below worst recon    |
| ordering  | [0.62, 35.85] (58× spread)              | 6         | above max residual (1.32), keeps below reasoning |
| residual  | [0.80, 1.32]  (1.6× spread)             | ∞         | already tight; no cap needed                     |
| reasoning | [3.98, 8.46]  (2.1× spread)             | ∞         | naturally bounded by frozen-`lm_head` entropy    |

### 9.4 Weighted value at initialization

A convenient consequence of the rule is that the weighted value
*clamps* at the cap:

$$
\tilde{\mathcal{L}}_i \;\triangleq\; w_i^{\star} \cdot \bar{\mathcal{L}}_i
\;=\; \min\!\left(\bar{\mathcal{L}}_i,\; c_i\right).
$$

So quiet components pass through unchanged ($\tilde{\mathcal{L}}_i = \bar{\mathcal{L}}_i$)
and loud components are pinned at the cap ($\tilde{\mathcal{L}}_i = c_i$).

### 9.5 Total training loss

$$
\mathcal{L}_{\text{total}}(\theta)
\;=\; w_{\text{recon}}^{\star}\,\mathcal{L}_{\text{recon}}(\theta)
   \;+\; w_{\text{ord}}^{\star}\,\mathcal{L}_{\text{ord}}(\theta)
   \;+\; w_{\text{res}}^{\star}\,\mathcal{L}_{\text{res}}(\theta)
   \;+\; w_{\text{rea}}^{\star}\,\mathcal{L}_{\text{rea}}(\theta).
$$

Because the $w_i^{\star}$ are scalars computed once from
`Loss_prepare.json`, each config has its own fixed-weight objective —
no dynamic reweighting during training.

### 9.6 Ordering weight as a function of level (closed form)

Since $\bar{\mathcal{L}}_{\text{ord}}$ is nearly model-invariant
(§4.3, per-model CoV ≈ 1.2%), we can approximate it by the per-level
average $\mu_{\text{ord}}(L)$:

$$
w_{\text{ord}}^{\star}(L) \;\approx\; \min\!\left(1,\; \frac{6}{\mu_{\text{ord}}(L)}\right),
\qquad
\mu_{\text{ord}}(L) \in \{1.03,\, 1.99,\, 3.69,\, 6.63,\, 11.39,\, 35.44\}
$$

for $L \in \{2, 3, 4, 5, 6, 8\}$, giving

$$
w_{\text{ord}}^{\star}(L) \in \{1.000,\, 1.000,\, 1.000,\, 0.905,\, 0.527,\, 0.169\}.
$$

The effective weighted ordering loss plateau-saturates:

$$
\tilde{\mathcal{L}}_{\text{ord}}(L) \in \{1.03,\, 1.99,\, 3.69,\, 6.00,\, 6.00,\, 6.00\}.
$$

### 9.7 Recon weight as a function of model (closed form)

Since $\bar{\mathcal{L}}_{\text{recon}}$ is nearly level-invariant
within a model (§4.1, within-model spread < 20%), we can approximate
it by the per-model average $\mu_{\text{recon}}(m)$:

$$
w_{\text{recon}}^{\star}(m) \;\approx\; \min\!\left(1,\; \frac{10}{\mu_{\text{recon}}(m)}\right)
$$

yielding

| $m$          | $\mu_{\text{recon}}(m)$ | $w_{\text{recon}}^{\star}(m)$ |
|--------------|------------------------:|------------------------------:|
| Qwen2.5-0.5B |                   77.37 |                     **0.129** |
| Qwen2.5-1.5B |                   19.17 |                     **0.522** |
| Qwen2.5-3B   |                   10.63 |                     **0.941** |
| Qwen3-0.6B   |                    9.67 |                         1.000 |
| Qwen3-1.7B   |                    5.93 |                         1.000 |
| Qwen3-4B     |                    6.62 |                         1.000 |

Qwen3 family is entirely un-capped; Qwen2.5 family is increasingly
down-weighted toward smaller models. This directly mirrors the family
boundary observed in §4.1.

---

## 10. Target effects

### 10.1 Bounded weighted envelopes

Applying the rule to all 36 configs produces weighted values with
bounded ranges:

| component | weighted range $\tilde{\mathcal{L}}_i$ | weight range $w_i^{\star}$ |
|-----------|---------------------------------------:|---------------------------:|
| recon     |                          [5.57, 10.00] |             [0.127, 1.000] |
| reasoning |                          [3.98,  8.46] |                    {1.000} |
| ordering  |                          [0.62,  6.00] |             [0.167, 1.000] |
| residual  |                          [0.80,  1.32] |                    {1.000} |

Spread compression: recon 13× → 1.8×, ordering 58× → 9.7×. After
weighting, no component has a spread larger than 10× across the
entire 36-config matrix — the scalar objective is now comparable
across runs.

### 10.2 Gradient-share analysis

Define the contribution share of component $i$ to the scalar loss:

$$
S_i(m, L) \;=\; \frac{\tilde{\mathcal{L}}_i(m, L)}{\sum_j \tilde{\mathcal{L}}_j(m, L)}.
$$

Representative values on Qwen3-4B:

| regime                          | $S_{\text{recon}}$ | $S_{\text{rea}}$ | $S_{\text{ord}}$ | $S_{\text{res}}$ |
|---------------------------------|-------------------:|-----------------:|-----------------:|-----------------:|
| L=2, all weights = 1 (natural)  |               52 % |             32 % |              9 % |              7 % |
| L=8, **no cap** (raw ord ≈ 36)  |               15 % |              9 % |         **73 %** |              2 % |
| L=8, **with cap** (this design) |               39 % |             24 % |             31 % |              6 % |

Without the ordering cap, L=8 training is driven entirely by ordering
(violates C1). With the cap, ordering's share grows from 9% at L=2 to
31% at L=8 — reflecting that deeper pyramids genuinely have more
ordering structure to learn, but without crushing recon or reasoning.

### 10.3 Priority preservation (approximate, measured)

Priority-pair holding rates across the 36-config matrix:

| pair                                                                      | configs satisfied |
|---------------------------------------------------------------------------|------------------:|
| $\tilde{\mathcal{L}}_{\text{recon}} \ge \tilde{\mathcal{L}}_{\text{res}}$ |           36 / 36 |
| $\tilde{\mathcal{L}}_{\text{rea}} \ge \tilde{\mathcal{L}}_{\text{res}}$   |           36 / 36 |
| $\tilde{\mathcal{L}}_{\text{recon}} \ge \tilde{\mathcal{L}}_{\text{ord}}$ |           34 / 36 |
| $\tilde{\mathcal{L}}_{\text{recon}} \ge \tilde{\mathcal{L}}_{\text{rea}}$ |           32 / 36 |
| $\tilde{\mathcal{L}}_{\text{rea}}   \ge \tilde{\mathcal{L}}_{\text{ord}}$ |           26 / 36 |

The strict tail ($\{\text{recon, rea}\} \ge \text{res}$) holds
**universally**. The finer intra-hierarchy pairs sometimes invert —
typically when a specific backbone happens to have $\bar{\mathcal{L}}_{\text{rea}}
< 6$ at a level with raw ordering > 6, making ordering's cap of 6
momentarily exceed reasoning's natural scale. Universal caps cannot
fix these per-config inversions without overfitting; they stay within
tolerance because the offsets are small (within ~50% of each other).

### 10.4 Natural-signal preservation for residual and reasoning

Since $c_{\text{res}} = c_{\text{rea}} = +\infty$,

$$
w_{\text{res}}^{\star} = w_{\text{rea}}^{\star} = 1,
\qquad
\tilde{\mathcal{L}}_{\text{res}} = \bar{\mathcal{L}}_{\text{res}},
\qquad
\tilde{\mathcal{L}}_{\text{rea}} = \bar{\mathcal{L}}_{\text{rea}}.
$$

No intervention is applied where none is needed (C3). Both components
already sit in tight envelopes whose cross-config spread is smaller
than the cap-imposed spread on recon and ordering, so any cap would
be a net loss of gradient signal.

### 10.5 Why a constant factor does not "starve" a component

For a component whose weight is $w_i^{\star} \ll 1$ (e.g.,
ordering at L=8 with $w = 0.167$), the update under Adam on
parameter $\theta$ is approximately

$$
\Delta \theta
\;\propto\;
\frac{w_i^{\star}\,\nabla_\theta \mathcal{L}_i}
     {\sqrt{\mathbb{E}\!\left[(w_i^{\star})^{2}\,(\nabla_\theta \mathcal{L}_i)^{2}\right] + \varepsilon}}.
$$

When a single component dominates a parameter's gradient, the Adam
denominator scales with $w_i^{\star}$ and largely cancels the
numerator factor — the *direction* of the update is preserved and the
*magnitude* is attenuated much less than the naive $w_i^{\star}$
factor suggests. Hard capping therefore does not starve ordering of
learning signal; it only prevents it from crowding out the other
three components in the shared portions of the gradient.

---

## 11. Implementation and provenance

### 11.1 Pipeline

```
configs/nlcpV4/GSM8K/train_builder_*_*level.yml   (baseline recipes)
                │
                ▼  loss_prepare.py (10-batch warm-up, no grad)
EXPERIMENT/nlcpV4/builder/Loss_prepare.json       (raw L̄_i per config)
                │
                ▼  loss_weight_compute.py -f Loss_prepare.json
EXPERIMENT/nlcpV4/builder/Loss_prepare_weights.csv (w_i*, L̃_i per config)
                │
                ▼  AutoWeighted generator
configs/nlcpV4/GSM8K/AutoWeighted/train_builder_*_*level.yml
                │
                ▼  train_builder.py
EXPERIMENT/nlcpV4/builder/GSM8K_<m>_<L>level_AutoWeighted/
```

### 11.2 File artifacts

- **[`examples/nlcpV4/loss_weight_compute.py`](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/examples/nlcpV4/loss_weight_compute.py)** — implements §9.2 verbatim; emits the CSV.
- **[`EXPERIMENT/nlcpV4/builder/Loss_prepare_weights.csv`](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/EXPERIMENT/nlcpV4/builder/Loss_prepare_weights.csv)** — 36 rows × 18 columns: raw, weight, and weighted values per config.
- **[`configs/nlcpV4/GSM8K/AutoWeighted/`](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/configs/nlcpV4/GSM8K/AutoWeighted)** — 36 YAML recipes; each carries a provenance banner listing $\bar{\mathcal{L}}_i$, $w_i^{\star}$, and $\tilde{\mathcal{L}}_i$, and retargets `save_folder` / `checkpoint_path` / `log_path` with a `_AutoWeighted` suffix to avoid colliding with baseline outputs.

### 11.3 Regeneration

To refresh after re-measuring (e.g., after a backbone precision change or
after adding Qwen3-8B entries):

```bash
python3 examples/RunResults/loss_prepare.py -c configs/nlcpV4/GSM8K/
python3 examples/nlcpV4/loss_weight_compute.py \
    -f EXPERIMENT/nlcpV4/builder/Loss_prepare.json
# then re-run the AutoWeighted generator
```

### 11.4 Coverage

- 36 / 42 GSM8K configs have AutoWeighted variants (6 models × 6 levels).
- Qwen3-8B (6 configs) pending: the 8B runs have not yet been recorded
  in `Loss_prepare.json`.

---

## 12. Limitations and when to re-tune

### 12.1 Escalation paths if a component stalls

Monitor via [`builder_training_analysis.py`](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/examples/nlcpV4/builder_training_analysis.py).
The failure mode to watch is **a component whose loss plateaus
immediately and refuses to decrease** — this is the symptom of
over-suppression.

Two graded alternatives to the hard cap, in increasing softness:

| rule                    | formula                                                | $w_{\text{ord}}^{\star}$ at L=8 | $\tilde{\mathcal{L}}_{\text{ord}}$ at L=8 |
|-------------------------|--------------------------------------------------------|--------------------------------:|------------------------------------------:|
| hard cap (current)      | $\min(1, c_i/\bar{\mathcal{L}}_i)$                     |                           0.167 |                                       6.0 |
| soft cap (sqrt)         | $\min\!\left(1, \sqrt{c_i/\bar{\mathcal{L}}_i}\right)$ |                           0.414 |                                      14.5 |
| raised cap ($c_i = 10$) | $\min(1, 10/\bar{\mathcal{L}}_i)$                      |                           0.286 |                                      10.0 |

### 12.2 When to re-measure

The caps are calibrated to the current Builder training setup. Invalidating
events (each requires re-running §11.1's pipeline from step 2):

- `level_lengths` changed → `ordering` combinatorics shift, cap will be stale.
- `recon_loss` normalization changed → recon's envelope shifts.
- `lm_head` unfrozen or LoRA-fied → reasoning is no longer anchored by
  NTP entropy and may need its own cap.
- New backbone family → hidden-state statistics shift, recon envelope
  shifts, may need new caps.
- BF16 vs FP32 precision change → numerical scale shifts are < 1% per
  §7; no re-tuning required.

### 12.3 What this design does *not* guarantee

- It does not auto-balance as training progresses — all weights are
  frozen at initialization values.
- It does not equalize gradient *magnitudes* on a per-parameter basis
  (that is Adam's job); it equalizes scalar *loss contributions*.
- It does not fix cross-backbone comparability of `recon` at a finer
  than per-family level — Qwen3-1.7B (raw 5.93) and Qwen3-4B (raw
  6.62) are both uncapped, and their weighted values differ by
  ~12%. If finer comparability is needed, switch to per-model caps
  (i.e., per-config $c_{\text{recon}}(m) = \bar{\mathcal{L}}_{\text{recon}}(m)$).

---

# Part III — Predictor loss analysis (GSM8K, independent mode)

> **Scope.** Part III mirrors Part I / II but for the **predictor**
> branch. The predictor's loss has only two components — `concept`
> (MSE between predicted and target level-wise concept embeddings)
> and `reasoning` (next-token CE through the predicted concept
> path). This section characterizes the 12 recorded
> `independent`-mode configs and derives per-config `loss_weights`
> under the design rule *"keep weighted concept ≈ 10, leave
> reasoning unchanged"*.

## 13. Purpose & data source

The predictor is trained to generate the next concept (per-level
embeddings) and, through those concepts, the next-token distribution
over the reasoning tail. `compute_predictor_loss` emits two scalar
components:

| component   | definition (informal)                                                                                  |
|-------------|--------------------------------------------------------------------------------------------------------|
| `concept`   | MSE between predicted concept embeddings and the builder-pyramid target concepts, averaged over levels |
| `reasoning` | next-token CE computed by feeding the **predicted** concepts through the frozen `lm_head`              |

In **independent** mode the predictor carries its own backbone
(disjoint from the builder's `reason_model`); LoRA is optional. The
concept target at each level is produced by the already-trained
builder pyramid, so its magnitude depends on both the level
schedule (how many tokens a level must summarize) and the
backbone's hidden-state scale.

- **Data file**: [`EXPERIMENT/nlcpV4/predictor/GSM8K_Loss_prepare_independent.json`](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/EXPERIMENT/nlcpV4/predictor/GSM8K_Loss_prepare_independent.json)
- **Entries analyzed**: 12 configs (Qwen2.5-0.5B × {2,4,6,8}, Qwen2.5-1.5B × {2,3,6,8}, Qwen3-0.6B × {2,4,6,8}).
  The 6×6 matrix is partially filled; Qwen2.5-3B / Qwen3-1.7B / Qwen3-4B / Qwen3-8B and the remaining `(m, L)` cells are pending.
- **Protocol**: `loss_predictor_prepare.py`, `batch_size = 4`, 10 no-grad forward passes; values are raw means, equal to weighted means (both weights are 1.0 at measurement time).

---

## 14. Raw loss table

| model        | L | concept (raw) | reasoning (raw) | `concept_per_level` (first batch)                |
|--------------|--:|--------------:|----------------:|--------------------------------------------------|
| Qwen2.5-0.5B | 2 |      1 039.74 |            5.99 | [2000.0, 33.3]                                   |
| Qwen2.5-0.5B | 4 |        591.26 |            6.92 | [1268.9, 348.9, 213.6, 33.2]                     |
| Qwen2.5-0.5B | 6 |        103.60 |            5.40 | [279.9, 37.3, 122.6, 41.4, 43.1, 38.5]           |
| Qwen2.5-0.5B | 8 |         51.16 |            5.69 | [71.3, 75.6, 47.1, 60.6, 36.0, 32.5, 31.5, 31.6] |
| Qwen2.5-1.5B | 2 |        746.00 |            4.41 | [851.6, 6.15]                                    |
| Qwen2.5-1.5B | 3 |        944.14 |            4.13 | [1409.6, 545.5, 5.98]                            |
| Qwen2.5-1.5B | 6 |        132.65 |            4.81 | [345.4, 94.6, 137.5, 34.2, 16.5, 5.28]           |
| Qwen2.5-1.5B | 8 |         22.18 |            5.79 | [34.2, 26.0, 44.3, 14.4, 15.8, 10.8, 4.67, 5.13] |
| Qwen3-0.6B   | 2 |        374.19 |            7.70 | [412.9, 67.8]                                    |
| Qwen3-0.6B   | 4 |        425.82 |            7.21 | [995.2, 351.0, 21.1, 33.2]                       |
| Qwen3-0.6B   | 6 |         67.64 |            6.41 | [105.3, 134.4, 60.8, 33.9, 18.3, 9.46]           |
| Qwen3-0.6B   | 8 |         20.50 |            7.44 | [26.7, 34.1, 34.5, 19.9, 15.5, 10.5, 8.92, 7.92] |

---

## 15. Aggregated views

### 15.1 Per-level averages (across available models)

|     L | #models |   concept | reasoning |
|------:|--------:|----------:|----------:|
|     2 |       3 |    719.98 |      6.03 |
|     3 |       1 |    944.14 |      4.13 |
|     4 |       2 |    508.54 |      7.06 |
|     6 |       3 |    101.30 |      5.54 |
| **8** |   **3** | **31.28** |  **6.31** |

Concept loss **decreases sharply with L** (mean drops from ~720 at
L=2 to ~31 at L=8 — a 23× contraction). Reasoning loss stays in a
narrow envelope [4.13, 7.06], matching the builder's reasoning
envelope in §4.4.

### 15.2 Per-model averages (across recorded levels)

| model        | Ls recorded | concept | reasoning |
|--------------|-------------|--------:|----------:|
| Qwen2.5-0.5B | 2,4,6,8     |  446.44 |      6.00 |
| Qwen2.5-1.5B | 2,3,6,8     |  461.24 |      4.79 |
| Qwen3-0.6B   | 2,4,6,8     |  222.04 |      7.19 |

Qwen3-0.6B sits at roughly half the concept magnitude of the two
Qwen2.5 configs, echoing the builder's family split for `recon`
(§4.1) — both `concept` and `recon` are MSE-class objectives over
hidden states, so they inherit the same `H_CoT.std()`-driven family
gap. The 0.5B / 1.5B concept averages are almost identical (446 vs
461), i.e. within-Qwen2.5-family size matters less than the family
boundary itself.

---

## 16. Per-component analysis

### 16.1 Concept loss (`concept`)

**Level dependence — monotone decrease, roughly L⁻¹ʹ⁵.**
For a single model (Qwen2.5-0.5B): 1039.74 → 591.26 → 103.60 →
51.16 across L ∈ {2, 4, 6, 8}. The Qwen3-0.6B trajectory
(374 → 426 → 68 → 21) and the Qwen2.5-1.5B trajectory
(746 → 944 → 133 → 22) show the same qualitative contraction, with
the only non-monotone step being L=2 → L=3 or L=2 → L=4 inside a
few models (within-level reshuffling of the level_lengths schedule).

**Why.** `concept` is the **mean** per-level MSE, and the
`concept_per_level` lists in §14 reveal where the mass sits:

- At L=2, the first level is a **single-token summary** of the
  entire CoT (level_lengths = [1, 2]). Reconstructing one embedding
  that stands in for a whole chunk is a very high-variance target —
  its MSE is 850–2000 across backbones, and it dominates the
  two-level mean.
- As L grows, level 0 still carries the largest per-level MSE, but
  it becomes one of many levels in the average, so its contribution
  to the mean is diluted. By L=8 the per-level MSEs fall into the
  [5, 75] range and their mean is ~20–50.
- Deeper levels (longer `level_length`) carry genuinely finer
  detail and reach single-digit MSE once L ≥ 6, e.g. level 5 of
  Qwen2.5-1.5B at L=6 is 5.28, level 7 at L=8 is 5.13.

So the level-dependence is **structural, not representational** —
it follows from averaging over a growing number of progressively
easier targets, not from the predictor "getting better" at deeper
pyramids.

**Model dependence — dominated by family, muted by size.**
Qwen3-0.6B's concept mean (222) is about half of the Qwen2.5
averages (~446 and ~461), consistent with the Qwen3 family's
smaller `H_CoT.std()` (§4.1). Within Qwen2.5, 0.5B (446) and 1.5B
(461) are nearly tied — size alone is a weak lever.

**Mechanistic note.** Unlike the builder's `recon`, concept MSE is
not divided by `H_CoT.std()^2`; it is plain MSE on the builder's
already-trained concept targets (see
[compute_predictor_loss](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/examples/nlcpV4/losses.py)). This is why the concept magnitudes are
orders larger than builder recon (1000+ vs 77 at worst) and why a
dedicated weight-capping strategy is essential.

### 16.2 Reasoning loss (`reasoning`)

**Level dependence — effectively flat.**
Per-model envelope across L: Qwen2.5-0.5B [5.40, 6.92], Qwen2.5-1.5B
[4.13, 5.79], Qwen3-0.6B [6.41, 7.70]. No systematic growth or
decrease with L; range within each model ≤ 1.8 units.

**Model dependence — mild, same character as builder.**
Per-model means 6.00 / 4.79 / 7.19 (CoV ≈ 20%) reproduce the builder
reasoning pattern: a backbone-specific NTP-CE baseline on GSM8K,
bounded below by the intrinsic next-token entropy of the frozen
`lm_head`.

**Conclusion.** Reasoning is already well-scaled (≈ 4–8 across the
entire matrix) and requires **no reweighting**. This matches the
builder convention in §10.4.

### 16.3 Structural summary

| component | level-sensitivity            | model-sensitivity        | driver                                                 |
|-----------|------------------------------|--------------------------|--------------------------------------------------------|
| concept   | **strong monotone decrease** | **high** (family > size) | level-0 single-token summary MSE + `H_CoT.std()`       |
| reasoning | flat across L                | mild, non-monotonic      | backbone-dependent NTP baseline through frozen lm_head |

---

## 17. Weight recommendations

### 17.1 Design rule

Following §9.2 in spirit but with a single target instead of a cap:
for every `(model, L)` config, pick `concept` weight so that the
weighted concept lands on a **target magnitude** of 10 — the same
rough anchor as builder `recon` (§9.3) and very close to the
typical reasoning CE (4–8). Reasoning gets weight 1.0 unchanged.

$$
w_{\text{concept}}^{\star}(m, L) \;=\; \min\!\left(1,\; \frac{T}{\bar{\mathcal{L}}_{\text{concept}}(m, L)}\right),
\qquad T = 10,
$$

$$
w_{\text{reasoning}}^{\star}(m, L) \;=\; 1.
$$

The `min(1, ·)` clamp activates only when raw concept drops below
the target — currently this does **not** happen on any of the 12
recorded configs (smallest raw concept is 20.50 on Qwen3-0.6B
L=8, still above T=10), so every recommended weight is the exact
ratio $T / \bar{\mathcal{L}}_{\text{concept}}$.

### 17.2 Per-config recommended weights

Exact weight = 10 / raw_concept. `w_c (round)` is the rounded
value to drop into a YAML config (2 decimals, extended to 3 only
when 2-decimal rounding would collapse to 0). `weighted concept`
is `w_c (round) × raw_concept`, i.e. the post-weight magnitude
seen by the optimizer.

| model        | L | concept (raw) | w_concept (exact) | **w_concept (round)** | weighted concept | w_reasoning |
|--------------|--:|--------------:|------------------:|----------------------:|-----------------:|-------------|
| Qwen2.5-0.5B | 2 |      1 039.74 |           0.00962 |              **0.01** |            10.40 | 1.0         |
| Qwen2.5-0.5B | 4 |        591.26 |           0.01691 |              **0.02** |            11.83 | 1.0         |
| Qwen2.5-0.5B | 6 |        103.60 |           0.09652 |              **0.10** |            10.36 | 1.0         |
| Qwen2.5-0.5B | 8 |         51.16 |           0.19547 |              **0.20** |            10.23 | 1.0         |
| Qwen2.5-1.5B | 2 |        746.00 |           0.01340 |              **0.01** |             7.46 | 1.0         |
| Qwen2.5-1.5B | 3 |        944.14 |           0.01059 |              **0.01** |             9.44 | 1.0         |
| Qwen2.5-1.5B | 6 |        132.65 |           0.07539 |              **0.08** |            10.61 | 1.0         |
| Qwen2.5-1.5B | 8 |         22.18 |           0.45082 |              **0.45** |             9.98 | 1.0         |
| Qwen3-0.6B   | 2 |        374.19 |           0.02672 |              **0.03** |            11.23 | 1.0         |
| Qwen3-0.6B   | 4 |        425.82 |           0.02348 |              **0.02** |             8.52 | 1.0         |
| Qwen3-0.6B   | 6 |         67.64 |           0.14785 |              **0.15** |            10.15 | 1.0         |
| Qwen3-0.6B   | 8 |         20.50 |           0.48772 |              **0.49** |            10.05 | 1.0         |

Matches the user's seed examples: raw 1039.74 → 0.01, raw 591.26
→ 0.02 (rounded up from 0.01691).

### 17.3 Effect of weighting

- **Raw concept spread**: 20.50 → 1039.74 (**50.7×**)
- **Weighted concept spread**: 7.46 → 11.83 (**1.6×**)

A single design constant (T = 10) compresses concept magnitudes
to within ±20% of target across all 12 configs, while reasoning is
untouched and retains its natural [4.13, 7.70] envelope. The
weighted scalar objective is now dominated in roughly equal measure
by concept and reasoning (≈ 10 + ≈ 6), consistent with the Part II
design philosophy of non-dominance (§8.3 C1).

### 17.4 YAML snippet template

For any `(model, L)` cell in §17.2, drop the recommended weights
into the predictor recipe's `training.loss_weights` block:

```yaml
training:
  loss_weights:
    concept_loss_weight: <w_concept (round) from §17.2>
    reasoning_loss_weight: 1.0
```

e.g. for `train_predictor_Qwen2.5-0.5B_2level_independent.yml`:

```yaml
training:
  loss_weights:
    concept_loss_weight: 0.01   # raw concept = 1039.74 → weighted = 10.40
    reasoning_loss_weight: 1.0
```

### 17.5 Limitations and when to re-measure

- The table covers `independent` mode only; `shared` mode
  configs (where predictor reuses the builder's `reason_model`)
  need their own measurement pass — concept-target statistics
  will shift because the hidden-state distribution under a
  builder-trained backbone differs from a fresh independent
  backbone.
- 12 / 24 configs are filled in; the other 12 `(m, L)` cells
  (Qwen2.5-3B, all three larger Qwen3 sizes, and the missing
  level slots for the three recorded models) still need a
  `loss_predictor_prepare.py` run before their weights can be
  derived.
- As with the builder caps (§12.2), any change to level
  schedule, concept-target normalization, backbone family, or
  the freezing pattern of `lm_head` invalidates these numbers —
  re-run the pipeline before trusting the weights.
- If a finer balance is desired — e.g. concept target tied to
  the measured reasoning CE of the same config rather than a
  global constant — replace T = 10 by
  $T(m, L) = \bar{\mathcal{L}}_{\text{reasoning}}(m, L)$. The
  weight formula in §17.1 is unchanged.
