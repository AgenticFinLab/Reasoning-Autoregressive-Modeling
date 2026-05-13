[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](https://swanlab.cn/@AfLab/ReasoningAR/overview)

# Reasoning Autoregressive Modeling (RAM)

A research codebase for **coarse-to-fine reasoning in LLMs**, inspired by
[VAR: Visual Autoregressive Modeling](https://arxiv.org/abs/2404.02905).
RAM treats a Chain-of-Thought (CoT) as a *multi-scale generative process*: a
compact, high-level plan is produced first, then progressively refined into
full natural-language reasoning. The core package `ram/` exposes reusable
building blocks (encoder / decoder / quantizer / VQ-VAE, losses, training
utilities, evaluation), and the experiments live under `examples/` with
configuration files under `configs/`.

---

## 1. Quick Start

### 1.1 Install

```bash
# Python 3.11+ required
pip install --upgrade setuptools
pip install -e .
```

After `pip install -e .` every module under `ram/` is importable as a regular
package (e.g. `from ram.models import build_text_vqvae`).

### 1.2 Smoke-test the import

```python
import ram
from ram.models import TextEncoder, TextDecoder, MultiScaleQuantizer, TextVQVAE
from ram.losses import VQAELoss
from ram.utils import load_config, set_seed
print(ram.__version__)
```

### 1.3 Run an experiment

Each experiment family lives in `examples/<variant>/`. The most actively
developed branch is `nlcpV4` (Concept Pyramid Builder + Predictor):

```bash
# Example: train an NLCP V4 builder on GSM8K with an AutoWeighted config
python3 examples/nlcpV4/train_builder.py -c configs/nlcpV4/GSM8K/AutoWeighted/train_builder_Qwen2.5-0.5B_4level.yml

# Post-training analysis / visualisation
python3 examples/nlcpV4/builder_concept_pyramid_analysis.py -c configs/nlcpV4/GSM8K/AutoWeighted/train_builder_Qwen2.5-0.5B_4level.yml --mode teacher_forced
```

See `examples/SlurmScripts/` for cluster launch templates and
`examples/RunResults/` for result-aggregation utilities.

---

## 2. Repository Layout

```text
.
├── ram/                    # Core package (installable via `pip install -e .`)
│   ├── models/             # Encoder, Decoder, Quantizer, Text-VQ-VAE, VAR-style ops
│   │   └── customized/     # basic_vae, basic_tar, tar, quant, sampling, regularization
│   ├── losses/             # Reconstruction, VQ, combined losses + registry
│   ├── evaluation/         # Text-reconstruction evaluator
│   ├── utils/              # Config loader, logging, storage, factory, serialization, tools
│   ├── data_load.py        # RamDataLoaderRegistry (wraps lmbase datasets)
│   ├── generic.py          # Dataclasses (TrainingConfig, ModelConfig, Samples, …)
│   └── __init__.py         # Public API surface
│
├── examples/               # Experiment implementations built on `ram`
│   ├── nlcpV4/             # Main track: Concept-Pyramid Builder / Predictor
│   ├── nlcpV3/             # Earlier NLCP iteration
│   ├── nlcpV2/             # Earliest NLCP prototype
│   ├── c3/                 # Context Cascade Compression
│   ├── ed/                 # Encoder–Decoder baseline
│   ├── eqd_token_level/    # Token-level equi-sized decoding
│   ├── eqd_scales/         # Multi-scale equi-sized decoding
│   ├── eqd_hierarchical/   # Hierarchical equi-sized decoding
│   ├── SlurmScripts/       # Cluster launch templates (Slurm)
│   ├── RunResults/         # Run-result aggregation / reporting
│   ├── GetResults/         # Metric extraction from logs / checkpoints
│   └── uTEST/              # Micro-tests for quick sanity checks
│
├── configs/                # YAML configs per experiment family
│   ├── nlcpV4/             # GSM8K/, MATH/, utest/
│   ├── nlcpV3/, nlcpV2/    # Earlier NLCP generations
│   ├── nlcp/, ModelLearn/  # Legacy / model-learn configs
│   ├── c3/, ed/, eqd_*/    # Per-variant configs + DeepSpeed JSON
│   └── */zero2.json        # DeepSpeed ZeRO-2 stage configs
│
├── docs/                   # Design notes, papers, specs, writing drafts
│   ├── references/         # PDFs of cited papers, organised by theme
│   ├── research-skills/    # Research-workflow playbooks
│   ├── VAR.md, dlcm.md     # Architecture / theory references
│   ├── concept-pyramid-V{1..3}.md, c3.md, losses.md, …
│   └── related-work.md, related-work.bib
│
├── third-part/             # Upstream source drops (read-only references)
│   ├── VAR-main/           # Visual Autoregressive reference implementation
│   ├── DeepSpeed-master/   # DeepSpeed source
│   ├── lmbase/             # Dataset / tokenizer base used by RAM
│   ├── large_concept_model-main/, C3-*/, ms-swift-main/
│
├── EXPERIMENT/             # Output root for runs, logs, reconstructions
├── logs/run_experiments/   # Slurm / launcher log captures
├── build/, ram.egg-info/   # Build artefacts (generated)
├── venv/                   # Local virtualenv (ignored)
├── requirements.txt        # Pinned runtime dependencies
├── pyproject.toml          # Build backend configuration
├── setup.py                # setuptools entry (name=`ram`, version from ram/__init__.py)
├── description.txt         # Long description fed to setuptools
└── LICENSE                 # Apache-2.0
```

---

## 3. Core Package: `ram/`

| Sub-module                           | Purpose                                                                                                                                |
|--------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| `ram.models.encoder` / `decoder`     | HuggingFace-backed `TextEncoder` and `TextDecoder` with factory builders.                                                              |
| `ram.models.quantizer`               | `MultiScaleQuantizer` — VAR-style multi-scale residual quantisation.                                                                   |
| `ram.models.text_vqvae`              | `TextVQVAE` assembling encoder + quantizer + decoder end-to-end.                                                                       |
| `ram.models.scale_ops`               | Scale / downsampling / upsampling primitives shared across variants.                                                                   |
| `ram.models.customized`              | Lower-level building blocks: `basic_vae`, `basic_tar`, `tar`, `quant`, `sampling`, `regularization`.                                   |
| `ram.losses`                         | `ReconstructionLoss`, `DualTokenizerReconstructionLoss`, `VQLoss`, `VQAELoss`, combined losses and a registry.                         |
| `ram.evaluation.text_reconstruction` | Evaluation loop for text-reconstruction quality.                                                                                       |
| `ram.utils`                          | `load_config`, `TrainingLogger`, `TrainingHistory`, `ReconstructionSampleStore`, `create_training_config`, JSON helpers.               |
| `ram.data_load`                      | `RamDataLoaderRegistry` — unified entry point over `lmbase` datasets (GSM8K, MATH, …).                                                 |
| `ram.generic`                        | Dataclasses: `TrainingConfig`, `ModelConfig`, `EncoderConfig`, `DecoderConfig`, `QuantizerConfig`, `TrainingStep`, `CheckpointData`, … |

Minimal usage pattern:

```python
from ram.utils import load_config
from ram.models import build_text_vqvae

config = load_config("configs/nlcpV4/utest/…yml")
model = build_text_vqvae(config["model"])
```

---

## 4. Experiment Families (`examples/`)

| Folder                                                   | One-line summary                                                                                                                                                                          |
|----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `nlcpV4/`                                                | **Active.** Concept-Pyramid Builder + (parallel) Predictor, with training / eval / analysis / loss-weight tooling and extensive design docs (`nlcpV4-explain.md`, `loss-*-analysis*.md`). |
| `nlcpV3/`                                                | Previous NLCP generation; kept for regression comparisons.                                                                                                                                |
| `nlcpV2/`                                                | Original NLCP prototype.                                                                                                                                                                  |
| `c3/`                                                    | Context Cascade Compression experiments.                                                                                                                                                  |
| `ed/`                                                    | Vanilla encoder-decoder baseline for ablation.                                                                                                                                            |
| `eqd_token_level/` / `eqd_scales/` / `eqd_hierarchical/` | "Equi-sized decoding" variants at different granularities.                                                                                                                                |
| `SlurmScripts/`                                          | Launch templates for cluster runs.                                                                                                                                                        |
| `RunResults/`                                            | Run-level result aggregation.                                                                                                                                                             |
| `GetResults/`                                            | Metric / artefact extraction utilities.                                                                                                                                                   |
| `uTEST/`                                                 | Micro-tests used during development.                                                                                                                                                      |

Each experiment folder typically ships `train_*.py`, `eval_*.py`,
`*_training_analysis.py`, and — for NLCP V4 — dedicated
`builder_concept_pyramid_analysis.py` and
`predictor_concept_pyramid_analysis.py` visualisers.

---

## 5. Configuration

Configurations are YAML files under `configs/<variant>/` and are consumed via
`ram.utils.load_config`. Most variants ship a paired DeepSpeed ZeRO-2
configuration (`zero2.json`). NLCP V4 configs are organised by dataset
(`GSM8K/`, `MATH/`) and typically include an `AutoWeighted/` subtree where loss
weights are generated from dataset statistics rather than hand-tuned.

---

## 6. References & Docs

- `docs/VAR.md` — VAR architecture notes that motivate RAM's design.
- `docs/concept-pyramid-V{1..3}.md`, `docs/concept-pyramid-critic.md` — NLCP
  design iterations.
- `docs/losses.md` — Loss-function design and normalisation.
- `docs/related-work.md`, `docs/related-work.bib` — Curated bibliography.
- `docs/references/` — Grouped PDFs (decoder-based, efficiency,
  inference-exploration, …).
- `examples/nlcpV4/nlcpV4-explain.md` — Canonical spec for NLCP V4.

---

## 7. License

Licensed under the **Apache License 2.0** — see [`LICENSE`](LICENSE).
