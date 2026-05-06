# Experimental Setup

> Target venue: NeurIPS / ICLR-style "Experimental Setup" section.
> This document is the full knob-level description of the NLCP V4
> pipeline as implemented in this repository. It is a *superset* —
> the paper version should pick the salient subset. Every number is
> traceable to a YAML under `configs/nlcpV4/{GSM8K,MATH}/AutoWeighted/`
> or to the design doc `examples/nlcpV4/nlcpV4-explain.md`.
>
> **Single source of truth for the method.**
> - Architecture: `examples/nlcpV4/nlcpV4-explain.md`
> - Loss formulas: `examples/nlcpV4/loss-desien-analysis.md`
> - All hyperparameters: `configs/nlcpV4/{GSM8K,MATH}/AutoWeighted/*.yml`

---

## 1. Datasets

We evaluate our method on mathematical reasoning benchmarks:
\texttt{GSM8K} \cite{GSM8K-arxiv21} and \texttt{MATH}
\cite{MATH-arxiv21}. Each benchmark provides a disjoint train / test
split; we train exclusively on the train split and report final
accuracy on the held-out test split. Additional out-of-distribution
robustness may be reported on \texttt{MATH-500} \cite{math500} and
\texttt{AIME2024} under a zero-shot transfer protocol using the
MATH-trained checkpoint (not fine-tuned on MATH-500 / AIME2024).

**Preprocessing.** Every training sample is tokenized with the
backbone tokenizer and truncated / right-padded to
`max_seq_len = 512` tokens. Batches are randomly shuffled each epoch
with `drop_last = true` to keep tensor shapes stable. All reported
evaluation numbers are computed on the official held-out test split
with no task-specific fine-tuning beyond the single training run on
the corresponding train split.

The two training datasets we instantiate in this codebase
(see `configs/nlcpV4/GSM8K/` and `configs/nlcpV4/MATH/`) are treated
interchangeably: identical architecture, identical training recipe,
only the data loader differs. Every result reported on \texttt{GSM8K}
uses a model trained on \texttt{GSM8K}-train; every result reported
on \texttt{MATH} uses a model trained on \texttt{MATH}-train.

---

## 2. Implementation Details

Our method (**NLCP V4**, the naming used in this repository) is a
**two-stage pipeline** built on top of a frozen causal language-model
backbone:

1. **Concept-Pyramid Builder** (Stage 1,
   `examples/nlcpV4/train_builder.py`). Given
   $(Q,\,\text{CoT},\,S)$, the Builder encodes the CoT into a
   hierarchical concept pyramid
   $C = [C_0, C_1, \ldots, C_{K-1}]$ where level $k$ contains
   $L_k = 2^{k}$ intra-level concepts (see §2.2). Every backbone
   weight is frozen; only pyramid-specific modules train.
2. **Concept Predictor** (Stage 2,
   `examples/nlcpV4/train_predictor.py`). The Predictor
   autoregressively produces the concept pyramid from $Q$ alone and,
   in the same unified teacher-forced forward, emits solution-token
   logits. The Builder is used in Stage 2 only to supply
   ground-truth concepts $C_{\text{gt}}$ (frozen, detached).

At inference time **only the Predictor is deployed** — the Builder
is never executed, so there is no Builder-induced inference overhead.
The three operating modes (`train_builder.py`,
`train_predictor.py`, `predictor._forward_inference`) match the
three boxes in §1.4.1 of `nlcpV4-explain.md`.

### 2.1 Backbone family

All experiments use the Qwen backbone family
\cite{qwen25, qwen3}. We sweep seven backbone sizes spanning roughly
two orders of magnitude in parameters:

| Backbone          | Hidden dim $D$ | Params |        Learning rate |
|-------------------|---------------:|-------:|---------------------:|
| Qwen/Qwen2.5-0.5B |            896 |  0.49B | $1\!\times\!10^{-4}$ |
| Qwen/Qwen2.5-1.5B |           1536 |  1.54B | $1\!\times\!10^{-4}$ |
| Qwen/Qwen2.5-3B   |           2048 |  3.09B | $5\!\times\!10^{-5}$ |
| Qwen/Qwen3-0.6B   |           1024 |  0.60B | $1\!\times\!10^{-4}$ |
| Qwen/Qwen3-1.7B   |           2048 |  1.72B | $1\!\times\!10^{-4}$ |
| Qwen/Qwen3-4B     |           2560 |  4.02B | $5\!\times\!10^{-5}$ |
| Qwen/Qwen3-8B     |           4096 |  8.19B | $5\!\times\!10^{-5}$ |

The backbone is loaded in `float32`. We found `bfloat16`/`float16`
LayerNorm, attention softmax, and the reconstruction / ordering MSE
terms numerically fragile — prone to NaN within a few hundred
training steps — so precision is intentionally not dropped.

### 2.2 Concept-Pyramid geometry

The pyramid has $K$ levels with a VAR-style doubling schedule
$L_k = 2^k$ for $k = 0, 1, \ldots, K-1$. Total pyramid length is
$N_K = \sum_k L_k = 2^K - 1$:

| $K$ | level\_lengths $(L_0, \ldots, L_{K-1})$ | Total concepts $N_K$ |
|----:|-----------------------------------------|---------------------:|
|   2 | $(1, 2)$                                |                    3 |
|   3 | $(1, 2, 4)$                             |                    7 |
|   4 | $(1, 2, 4, 8)$                          |                   15 |
|   5 | $(1, 2, 4, 8, 16)$                      |                   31 |
|   6 | $(1, 2, 4, 8, 16, 32)$                  |                   63 |
|   8 | $(1, 2, 4, 8, 16, 32, 64, 128)$         |                  255 |

Concept-embedding dimension equals the backbone hidden size (column
2 of §2.1) — this avoids any lossy projection between the backbone
and concept space and keeps the pyramid VAR-faithful. Intra-level
ordering uses 8 attention heads (`num_heads = 8`). Max encoded CoT
length is fixed at `max_seq_len = 512`.

Following the notation of `nlcpV4-explain.md`:
- $C_{k,j}$ — the $j$-th concept at level $k$.
- Inter-level ($k$) governs **granularity** (coarse-to-fine).
- Intra-level ($j$) governs **position** within the CoT at that
  granularity.

### 2.3 Builder architecture (Stage 1)

The Builder (`examples/nlcpV4/concept_builder.py`) wraps the frozen
backbone with:

- **Input projection**: $H_{\text{proj}} = \text{LayerNorm}(\text{Linear}(H_{\text{CoT}}))$.
- **Per-level residual decomposition** over $k = 0, \ldots, K-1$:
  \begin{align*}
  A_k &= \text{softmax}(Q_k \cdot H_{\text{rest},k}^\top / \sqrt{D}/\tau), \\
  C_k &= \text{level\_proj}_k(A_k \cdot H_{\text{rest},k}), \\
  R_k &= A_k^\top \cdot C_k, \quad
  H_{\text{hat},k+1} = H_{\text{hat},k} + R_k, \quad
  H_{\text{rest},k+1} = H_{\text{rest},k} - R_k.
  \end{align*}
- **Back projection**: $H_{\text{recon}} = \text{back\_proj}(H_{\text{hat},K})$,
  initialized with $\text{back\_proj}.W = \text{input\_proj}.W^\top$.
- **Positional-query initialization**
  (`use_positional_query_init: true`, $\alpha = 0.5$) — concept
  queries are seeded with DLCM-style positional priors.
- **Ordering head** — within each level, enforce
  $\text{exp\_pos}(C_{k,j}) < \text{exp\_pos}(C_{k,j+1})$ via a
  margin-based hinge loss (`ordering_loss_type: margin`, margin $=1.0$).
- **Residual regularizer** — penalizes $\lVert H_{\text{rest},K} \rVert_1$
  so levels do not overlap.

### 2.4 Predictor architecture (Stage 2)

The Predictor (`examples/nlcpV4/concept_predictor.py`) consumes the
teacher-forced tuple $(Q,\, C_{\text{gt}},\, S)$ packed into a
single contiguous sequence

$$
[\ Q_{1:n_Q},\ \underbrace{C_{0,0},\, C_{1,0},\, C_{1,1},\, \ldots,\, C_{K-1,L_{K-1}-1}}_{N_K \text{ slots}},\ S_{1:n_S}\ ]
$$

via `pack_qcs_sequences`, feeds it through one causal LM pass, then
gathers:
- concept-position hidden states $\rightarrow$ `concept_head`
  $\rightarrow \hat{C}_k$, and
- solution-position hidden states $\rightarrow$ `lm_head`
  $\rightarrow \text{logits}_S$.

Slot markers are `back_proj(C_{\text{gt}}) + \text{level\_emb} +
\text{position\_emb}`.

We report results for two Stage-2 variants (`use_shared_model`
switch, see `nlcpV4-explain.md` §6 and the
"ConceptPredictor use_shared_model Training Constraint" memo):

**Variant A — INDEPENDENT + LoRA (primary / headline).**
`use_shared_model: false`. The Predictor carries its *own* instance
of the same reason-model architecture (`predictor_model_name`
equals the paired Builder's `reason_model_name`). The Builder's
backbone and the Predictor's backbone are distinct module trees, so
LoRA can safely be attached to the Predictor without corrupting the
Builder.
- Predictor base backbone: **frozen** (`freeze: true`).
- LoRA on attention projections:
  `r = 16`, `lora_alpha = 32`,
  `target_modules = {q_proj, v_proj}`,
  `lora_dropout = 0.05`, `bias = none`.
- `back_proj`: fresh `nn.Linear(D, D_{\text{enc}})`, trained from
  scratch.
- Additional small trainables: `level_embeddings`,
  `position_embeddings`, `concept_head` (dropout $= 0$).

Total Predictor trainable surface:
$$
\text{LoRA}(q, v) \;\cup\; \text{back\_proj}
\;\cup\; \{\text{level\_emb},\, \text{pos\_emb},\, \text{concept\_head}\}.
$$

**Variant B — SHARED (control).**
`use_shared_model: true`. The Predictor's `reason_model` and
`back_proj` are aliases of the frozen Builder modules; LoRA is
forbidden here (it would corrupt the Builder). Only the three small
heads (`level_embeddings`, `position_embeddings`, `concept_head`)
train. Shared mode is maintained purely as an ablation control,
**not** the headline configuration.

### 2.5 Inference procedure

At inference, the Predictor operates autoregressively from $Q$ only
— the Builder, CoT, and solution $S$ are not used.

1. **Step 0 — prime the KV cache.**
   $h = \text{reason\_model}(\text{embed}(Q))$;
   $\hat{C}_0 = \text{concept\_head}(h[\text{last-real-}Q])$;
   store `past_kv`.
2. **Steps $t = 1 \ldots N_K - 1$** — one concept per step:
   \begin{align*}
   x &= \text{back\_proj}(\hat{C}_{t-1}) + \text{level\_emb}[t-1] + \text{position\_emb}[t-1],\\
   h &= \text{reason\_model}(x,\, \text{past\_kv} = \text{cache}),\\
   \hat{C}_t &= \text{concept\_head}(h[-1]);\quad \text{cache} \leftarrow \text{updated KV}.
   \end{align*}
3. **Output.** $[\hat{C}_0, \ldots, \hat{C}_{K-1}]$, then a standard
   autoregressive solution-token decode from the same model with
   the same cache.

---

## 4. Training

Training is a strict **two-stage pipeline**. Stage 1 trains the
Builder from a frozen backbone; Stage 2 loads the best-on-eval
Builder checkpoint with `strict_load: true` (fail-fast on any
weight mismatch) and trains the Predictor on top. Stages never
train jointly.

### 4.1 Shared hyperparameters (both stages, every config)

| Hyperparameter      | Value                           |
|---------------------|---------------------------------|
| Optimizer           | AdamW                           |
| Weight decay        | $0.01$                          |
| LR schedule         | linear warmup then linear decay |
| Warmup ratio        | $0.1$ of total steps            |
| Gradient clip (L2)  | $1.0$                           |
| Batch size          | $4$                             |
| Epochs              | $10$                            |
| Max sequence length | $512$ tokens                    |
| Precision           | `float32`                       |
| Dataloader workers  | $0$ (main-process)              |
| Random seed         | $42$                            |
| `drop_last`         | true                            |
| Shuffle             | true (per-epoch)                |

Learning rate is size-dependent: $1\!\times\!10^{-4}$ for backbones
$\leq 1.7B$ and $5\!\times\!10^{-5}$ for backbones $\geq 3B$
(see §2.1).

### 4.2 Stage 1 — Builder losses (from `loss-desien-analysis.md` §1–§5)

The Builder objective is a weighted sum of four terms
(`compute_builder_loss` in `examples/nlcpV4/losses.py`):

$$
\mathcal{L}_{\text{builder}}
 = w_{\text{rec}}\,\mathcal{L}_{\text{rec}}
 + w_{\text{ord}}\,\mathcal{L}_{\text{ord}}
 + w_{\text{res}}\,\mathcal{L}_{\text{res}}
 + w_{\text{reas}}\,\mathcal{L}_{\text{reas}}.
$$

- $\mathcal{L}_{\text{rec}} = \text{MSE}(\text{back\_proj}(H_{\text{hat},K}),\, H_{\text{CoT}})$ (masked).
- $\mathcal{L}_{\text{ord}}$: margin hinge within each level
  enforcing intra-level ordering (margin $= 1.0$).
- $\mathcal{L}_{\text{res}} = \lVert H_{\text{rest},K} \rVert_1$
  (residual regularizer).
- $\mathcal{L}_{\text{reas}}$: next-token CE on $S$ produced by the
  frozen backbone fed $[Q;\,C;\,S]$.

**Auto-calibrated weights (AutoWeighted).** For every
$(\text{backbone}, K)$ combination we measure raw mean losses
$\bar{\ell}_\star$ on 10 warm-up no-grad batches, then set
$$
w_{\text{rec}} = \min\!\Bigl(1,\, \tfrac{10}{\bar{\ell}_{\text{rec}}}\Bigr),\qquad
w_{\text{ord}} = \min\!\Bigl(1,\, \tfrac{6}{\bar{\ell}_{\text{ord}}}\Bigr),\qquad
w_{\text{res}} = w_{\text{reas}} = 1.
$$
This anchors the reconstruction scale to $\approx 10$ and the
ordering scale to $\approx 6$, matching the natural envelope of the
reasoning CE term ($\approx 4$–$8$), so no single objective
monopolizes the gradient. The raw measurements, derived weights,
and post-weighting scales are frozen into the YAML header of every
`configs/nlcpV4/*/AutoWeighted/train_builder_*.yml`.

### 4.3 Stage 2 — Predictor losses (from `loss-desien-analysis.md` §6)

The Predictor objective is
(`compute_predictor_loss` in `examples/nlcpV4/losses.py`):

$$
\mathcal{L}_{\text{predictor}}
 = w_c\,\mathcal{L}_{\text{concept}} + w_r\,\mathcal{L}_{\text{reas}},
$$

with
$\mathcal{L}_{\text{concept}} = \frac{1}{K}\sum_{k=0}^{K-1} \text{MSE}(\hat{C}_k, C_{k,\text{gt}})$
and $\mathcal{L}_{\text{reas}}$ the next-token CE on solution
tokens. Both terms are produced by a *single* unified teacher-forced
forward pass — not two independent passes.

Weights are again target-anchored:
$$
w_c = \min\!\Bigl(1,\, \tfrac{10}{\bar{\ell}_{\text{concept}}}\Bigr), \qquad
w_r = 1.
$$

A representative calibration for GSM8K / Qwen2.5-0.5B / $K = 2$ /
INDEPENDENT mode gives
$\bar{\ell}_{\text{concept}} \approx 1039.7$, so
$w_c \approx 9.6\!\times\!10^{-3}$ — the weighted concept term then
lands at $\approx 10.0$ and matches the reasoning CE envelope
($\approx 6.0$). Across the full $7 \times 6 = 42$-cell sweep
$w_c$ spans over an order of magnitude (from
$\approx 9.6\!\times\!10^{-3}$ at $K = 2$ up to $\approx 0.49$ at
$K = 8$), which is precisely why per-configuration
auto-calibration is required. All raw-measurement provenance is
preserved in the header of every
`configs/nlcpV4/*/AutoWeighted/train_predictor_*_independent.yml`.

### 4.4 Evaluation during training

Mid-training evaluation on the held-out test split runs every
`eval_step_interval = 500` training steps. A small fast probe
(`log_num_samples = 40`) runs every
`log_step_interval = 10` steps for trajectory monitoring, and a
full eval (`eval_num_samples = 1.0`, i.e. the entire test split)
runs at every eval step. The best-on-eval checkpoint is tracked
per run (`checkpoint_best_eval.pt`) and is the one consumed by
Stage 2.

### 4.5 Checkpointing

`checkpoint_clean = true`: only the epoch-start checkpoint of every
epoch and one final post-training checkpoint are retained, plus
the continuously-tracked best-on-train and best-on-eval checkpoints
(previous best files are removed when a new best is found). Stage 2
always loads `checkpoint_best_eval.pt` from the paired Stage 1 run
via `model.builder.checkpoint_path`.

### 4.6 Hardware and scheduling

Experiments are launched by a GPU-aware scheduler
(`examples/RunResults/run_predictor_experiments.py`) with
`--one-per-gpu`: exactly one experiment is pinned per idle GPU
(free VRAM $\geq 90\%$), the remaining experiments queue in FIFO,
and the queue drains as GPUs release memory. Every run is wrapped
in a tmux session for safe long-duration execution. A single fixed
seed (`seed = 42`) is used across the codebase; dataloader workers
are kept at $0$ to eliminate worker-ordering nondeterminism.

### 4.7 Configuration grid

| Axis               | Values                                                                                                        | Count |
|--------------------|---------------------------------------------------------------------------------------------------------------|------:|
| Backbone           | Qwen2.5-$\{0.5\text{B}, 1.5\text{B}, 3\text{B}\}$, Qwen3-$\{0.6\text{B}, 1.7\text{B}, 4\text{B}, 8\text{B}\}$ |     7 |
| Pyramid levels $K$ | $\{2, 3, 4, 5, 6, 8\}$                                                                                        |     6 |
| Stage-2 variant    | INDEPENDENT+LoRA (primary), SHARED (ablation)                                                                 |     2 |

Total primary cells: $7 \times 6 = 42$ per training dataset
(GSM8K, MATH), with matching SHARED ablations when reported.

---

## 5. Metrics

We report three complementary metrics — one for quality, one for
output length, one for wall-clock speed.

### 5.1 Accuracy

Accuracy is **exact-match on the final numerical answer** for both
\texttt{GSM8K} and \texttt{MATH} (and for any zero-shot transfer
onto \texttt{MATH-500} / \texttt{AIME2024}). Predictions are parsed
from the boxed / final-line convention of each benchmark and
compared modulo trivial formatting (leading/trailing whitespace,
`\frac{a}{b}` vs `a/b` normalization). No partial credit is given.
All accuracy numbers are measured on the held-out test split with
no task-specific fine-tuning beyond the single Stage-2 training on
the matching training split.

### 5.2 Token cost

Token cost reports the *inference-time verbosity*, defined as the
**mean number of tokens produced per test instance** averaged over
the evaluation split. For NLCP V4:

$$
\text{tokens}(\text{ours}) = N_K + T_{\text{solution}},
$$

where $N_K = \sum_{k=0}^{K-1} L_k = 2^K - 1$ is the fixed number of
concept slots (Table in §2.2) and $T_{\text{solution}}$ is the
length of the decoded solution in backbone tokens. For the
token-level autoregressive baselines,
$\text{tokens}(\text{baseline}) = T_{\text{solution}}$. Lower is
better.

### 5.3 Inference time

Inference time is the **mean wall-clock latency per test instance**,
in seconds, measured with `batch_size = 1`, `float32` precision,
and `torch.inference_mode()` enabled on a single GPU. We report two
numbers for every setting:
(a) total per-sample latency including any one-time setup cost
amortized over the split, and (b) generation-only latency
(forward passes only, excluding tokenization and post-processing).
Timings are averaged over three independent runs with the
evaluation split re-shuffled between runs, reported as mean $\pm$
one standard deviation. Lower is better.

The three metrics are read jointly: **Accuracy** for quality,
**Token cost** for output efficiency, **Inference time** for
wall-clock speed. A configuration is preferred if it
Pareto-dominates the baselines on at least two of the three axes
at matched backbone size.

---

## Reproducibility

Every configuration described above is a one-to-one mapping to a
YAML under `configs/nlcpV4/GSM8K/AutoWeighted/` or
`configs/nlcpV4/MATH/AutoWeighted/`. Each YAML includes a
self-documenting header recording the exact auto-calibration used
for its loss weights (measured raw values, weighting rule, and
post-weighting scales), so every run is bit-reproducible from the
config alone. The Stage-1 $\to$ Stage-2 pointers
(`model.builder.config_path` and `model.builder.checkpoint_path`)
are resolved strictly; any missing dependency triggers a fail-fast
error before the first optimizer step. Method-level references:
`examples/nlcpV4/nlcpV4-explain.md` (architecture),
`examples/nlcpV4/loss-desien-analysis.md` (loss formulas and code
line-numbers).
