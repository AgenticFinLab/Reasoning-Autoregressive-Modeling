# DeepSpeed Complete Guide

> **Version**: Based on DeepSpeed master source code (`third-part/DeepSpeed-master`)
>
> **Purpose**: Comprehensive guide for designing, configuring, and training with DeepSpeed

---

## Table of Contents

1. [DeepSpeed Architecture Overview](#1-deepspeed-architecture-overview)
2. [Core Concepts](#2-core-concepts)
3. [Installation & Setup](#3-installation--setup)
4. [Configuration System](#4-configuration-system)
5. [ZeRO Optimization Deep Dive](#5-zero-optimization-deep-dive)
6. [Training APIs](#6-training-apis)
7. [Data Loading](#7-data-loading)
8. [Advanced Features](#8-advanced-features)
9. [Inference Optimization](#9-inference-optimization)
10. [Integration Patterns](#10-integration-patterns)
11. [Troubleshooting](#11-troubleshooting)
12. [Best Practices](#12-best-practices)
13. [Quick Reference](#13-quick-reference)

---

## 1. DeepSpeed Architecture Overview

### 1.1 What is DeepSpeed?

DeepSpeed is a deep learning optimization library that enables:
- **Extreme scale**: Train models with trillions of parameters
- **Memory efficiency**: ZeRO optimization reduces memory by up to 8x
- **Speed**: Mixed precision, gradient accumulation, efficient communication
- **Ease of use**: Minimal code changes required

### 1.2 Key Innovations

| Feature                 | Description                                                                    | Paper                                       |
|-------------------------|--------------------------------------------------------------------------------|---------------------------------------------|
| **ZeRO**                | Zero Redundancy Optimizer - partitions optimizer states, gradients, parameters | [SC'20](https://arxiv.org/abs/1910.02054)   |
| **ZeRO-Offload**        | Offload to CPU/NVMe for billion-scale models                                   | [ATC'21](https://arxiv.org/abs/2101.06840)  |
| **ZeRO-Infinity**       | Extreme scale with NVMe offloading                                             | [SC'21](https://arxiv.org/abs/2104.07857)   |
| **3D Parallelism**      | Data + Tensor + Pipeline parallelism                                           | -                                           |
| **DeepSpeed-MoE**       | Mixture of Experts training                                                    | [ICML'22](https://arxiv.org/abs/2201.05596) |
| **DeepSpeed-Inference** | Optimized inference with kernel injection                                      | [SC'22](https://arxiv.org/abs/2207.00032)   |

### 1.3 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Model (nn.Module)                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              DeepSpeed Engine (runtime/engine.py)            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   ZeRO      │  │  Optimizer  │  │  LR Scheduler       │  │
│  │  (zero/)    │  │  (AdamW,    │  │  (WarmupDecayLR)    │  │
│  │             │  │   LAMB, etc)│  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   BF16/FP16 │  │  Gradient   │  │  Checkpointing      │  │
│  │  (bf16/     │  │  Clipping   │  │  (checkpoint/)      │  │
│  │   fp16/)    │  │             │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Distributed Backend (NCCL/MPI)                  │
└─────────────────────────────────────────────────────────────┘
```

### 1.4 Parallelism Dimensions

DeepSpeed supports multiple parallelism strategies:

| Parallelism                | What It Splits              | Best For          | Source Code                  |
|----------------------------|-----------------------------|-------------------|------------------------------|
| **Data Parallel (DP)**     | Batch across GPUs           | Standard scaling  | `runtime/engine.py`          |
| **Tensor Parallel (TP)**   | Layer computation           | Large layers      | `runtime/tensor_parallel/`   |
| **Pipeline Parallel (PP)** | Model layers                | Very deep models  | `runtime/pipe/`              |
| **ZeRO**                   | Optimizer states, gradients | Memory efficiency | `runtime/zero/`              |
| **Sequence Parallel (SP)** | Sequence dimension          | Long sequences    | `runtime/sequence_parallel/` |

**3D Parallelism** = DP + TP + PP combined

---

## 2. Core Concepts

### 2.1 The DeepSpeed Engine

The `DeepSpeedEngine` is the core wrapper around your model:

```python
import deepspeed

# Initialize DeepSpeed - returns 4 items
model_engine, optimizer, dataloader, lr_scheduler = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    training_data=dataset,  # Optional
    config=ds_config        # DeepSpeed JSON config
)
```

**Return values**:
- `engine`: Wrapped model for distributed training
- `optimizer`: Wrapped optimizer (from config or passed)
- `training_dataloader`: DeepSpeed dataloader (if training_data provided)
- `lr_scheduler`: Wrapped scheduler (from config or passed)

**Key responsibilities**:
- Distributed training setup
- Mixed precision (BF16/FP16)
- ZeRO optimization
- Gradient accumulation
- Learning rate scheduling
- Checkpointing

**Source**: `deepspeed/__init__.py:80-142`

### 2.2 Training Loop

```python
# Standard PyTorch loop with DeepSpeed
for step, batch in enumerate(data_loader):
    loss = model_engine(batch)       # Forward
    model_engine.backward(loss)       # Backward
    model_engine.step()               # Optimizer step
```

**What DeepSpeed handles automatically**:
- ✅ Gradient averaging across GPUs
- ✅ Loss scaling (for FP16)
- ✅ Learning rate scheduling (at every step)
- ✅ ZeRO parameter partitioning
- ✅ Communication optimization
- ✅ Gradient accumulation boundaries

**Source**: `runtime/engine.py:forward/backward/step`

### 2.3 Configuration Philosophy

DeepSpeed uses a **two-file configuration system**:

| File             | Purpose                               | Example                       |
|------------------|---------------------------------------|-------------------------------|
| `config.yaml`    | Model, data, training hyperparameters | `learning_rate`, `batch_size` |
| `ds_config.json` | DeepSpeed-specific settings           | `zero_optimization`, `bf16`   |

**Why two files?**
- `config.yaml`: Framework-agnostic (can use with/without DeepSpeed)
- `ds_config.json`: DeepSpeed-specific optimizations

---

## 3. Installation & Setup

### 3.1 Installation

```bash
# Basic installation
pip install deepspeed

# With specific features
pip install deepspeed[cpu]      # CPU offloading support
pip install deepspeed[1bitadam] # 1-bit Adam optimizer
```

### 3.2 Launching DeepSpeed Training

```bash
# Method 1: Using deepspeed launcher (recommended)
deepspeed --num_gpus=4 train.py --deepspeed ds_config.json

# Method 2: Using torchrun
torchrun --nproc_per_node=4 train.py --deepspeed ds_config.json

# Multi-node
deepspeed --num_gpus=8 --num_nodes=2 --hostfile=hostfile.txt \
    train.py --deepspeed ds_config.json

# With CPU core binding for ZeRO-Offload
deepspeed --bind_cores_to_rank --num_gpus=4 train.py --deepspeed ds_config.json
```

### 3.3 Environment Variables

| Variable                  | Purpose               | Example           |
|---------------------------|-----------------------|-------------------|
| `CUDA_VISIBLE_DEVICES`    | GPU selection         | `0,1,2,3`         |
| `NCCL_DEBUG`              | NCCL debugging        | `INFO`            |
| `TORCH_DISTRIBUTED_DEBUG` | PyTorch debug         | `DETAIL`          |
| `LOCAL_RANK`              | Local GPU rank        | Set automatically |
| `RANK`                    | Global rank           | Set automatically |
| `WORLD_SIZE`              | Total processes       | Set automatically |
| `NCCL_P2P_DISABLE`        | Disable P2P if issues | `1`               |

---

## 4. Configuration System

### 4.1 Complete Configuration Reference

```json
{
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 2,
    
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0
        }
    },
    
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-5,
            "warmup_num_steps": 100,
            "total_num_steps": 1000
        }
    },
    
    "bf16": {
        "enabled": true,
        "immediate_grad_update": false
    },
    
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "none"
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "overlap_comm": true,
        "contiguous_gradients": true
    },
    
    "gradient_clipping": 1.0,
    "steps_per_print": 10,
    "wall_clock_breakdown": false
}
```

**Source**: `runtime/config.py`, `runtime/constants.py`

### 4.2 Batch Size Configuration

**The Golden Formula**:
```
train_batch_size = train_micro_batch_size_per_gpu × gradient_accumulation_steps × num_gpus
```

**Example**:
- 4 GPUs
- `train_micro_batch_size_per_gpu`: 4
- `gradient_accumulation_steps`: 2
- **Effective batch size**: 4 × 2 × 4 = 32

**Auto-calculation** (recommended):
```json
{
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 2
}
```
DeepSpeed automatically calculates `train_batch_size`.

**Source**: `runtime/constants.py:19-108`

### 4.3 Optimizer Configuration

**Supported Optimizers**:

| Optimizer  | Type String    | Use Case                    | Source                 |
|------------|----------------|-----------------------------|------------------------|
| AdamW      | `"AdamW"`      | Default, most common        | `runtime/config.py:72` |
| Adam       | `"Adam"`       | Classic Adam                | `runtime/config.py:71` |
| LAMB       | `"Lamb"`       | Large batch training        | `runtime/config.py:73` |
| LION       | `"Lion"`       | Memory efficient            | `runtime/config.py:80` |
| 1-bit Adam | `"OneBitAdam"` | Communication efficient     | `runtime/config.py:74` |
| Muon       | `"Muon"`       | Orthogonal gradient descent | `runtime/config.py:81` |

**AdamW Example**:
```json
{
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "torch_adam": false,
            "adam_w_mode": true
        }
    }
}
```

### 4.4 Learning Rate Schedulers

**Supported Schedulers**:

| Scheduler      | Type String        | Description           | Source                    |
|----------------|--------------------|-----------------------|---------------------------|
| WarmupDecayLR  | `"WarmupDecayLR"`  | Linear warmup + decay | `runtime/lr_schedules.py` |
| WarmupLR       | `"WarmupLR"`       | Warmup only           | `runtime/lr_schedules.py` |
| WarmupCosineLR | `"WarmupCosineLR"` | Warmup + cosine decay | `runtime/lr_schedules.py` |
| OneCycle       | `"OneCycle"`       | 1-cycle policy        | `runtime/lr_schedules.py` |

**WarmupDecayLR Example**:
```json
{
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-5,
            "warmup_num_steps": 100,
            "total_num_steps": 1000,
            "warmup_type": "linear"
        }
    }
}
```

**⚠️ Important**: `"auto"` is NOT supported for scheduler parameters. Use explicit integers.

### 4.5 Gradient Clipping

DeepSpeed provides gradient clipping via config:

```json
{
    "gradient_clipping": 1.0
}
```

**How it works**:
- Clips gradients by global norm
- Applied before optimizer step
- Value is the max norm threshold
- Set to `0` to disable (default)

**Alternative**: Use `max_grad_norm` in optimizer params:
```json
{
    "optimizer": {
        "type": "AdamW",
        "params": {
            "max_grad_norm": 1.0
        }
    }
}
```

**Source**: `runtime/constants.py:247-254`

### 4.6 Mixed Precision: BF16 vs FP16

**BF16** (recommended for Ampere+ GPUs: A100, H100):
```json
{
    "bf16": {
        "enabled": true,
        "immediate_grad_update": false,
        "check_overflow": false,
        "bf16_master_weights_and_grads": false
    }
}
```

**FP16** (for older GPUs: V100, etc.):
```json
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1,
        "initial_scale_power": 16
    }
}
```

**Comparison**:

| Feature       | BF16       | FP16         |
|---------------|------------|--------------|
| Exponent bits | 8          | 5            |
| Mantissa bits | 7          | 10           |
| Dynamic range | Wider      | Narrower     |
| Loss scaling  | Not needed | Required     |
| Hardware      | Ampere+    | All          |
| Stability     | Better     | Needs tuning |

**⚠️ Never enable both BF16 and FP16 simultaneously** - DeepSpeed will raise an error.

**Source**: `runtime/constants.py:116-150`, `runtime/bf16_optimizer.py`, `runtime/fp16/`

---

## 5. ZeRO Optimization Deep Dive

### 5.1 ZeRO Stages

ZeRO (Zero Redundancy Optimizer) partitions model states across GPUs:

| Stage | Partitions       | Memory Reduction | Use Case        |
|-------|------------------|------------------|-----------------|
| **0** | None (disabled)  | 1x               | Baseline        |
| **1** | Optimizer states | 4x               | Large models    |
| **2** | + Gradients      | 8x               | **Most common** |
| **3** | + Parameters     | 8x+              | Extreme scale   |

**Memory breakdown for Adam optimizer**:
- Parameters: 2 bytes (FP16/BF16)
- Gradients: 2 bytes (FP16/BF16)
- Optimizer states: 12 bytes (FP32 copy + momentum + variance)
- **Total**: 16 bytes per parameter

**Source**: `runtime/zero/config.py:81-88`, `runtime/zero/stage_1_and_2.py`, `runtime/zero/stage3.py`

### 5.2 ZeRO-2 Configuration

```json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "none"
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "round_robin_gradients": false
    }
}
```

**Key Parameters**:

| Parameter               | Description               | Default | Source               |
|-------------------------|---------------------------|---------|----------------------|
| `stage`                 | ZeRO stage (0-3)          | 0       | `zero/config.py:95`  |
| `allgather_bucket_size` | Bucket size for allgather | 5e8     | `zero/config.py:132` |
| `reduce_bucket_size`    | Bucket size for reduce    | 5e8     | `zero/config.py:113` |
| `overlap_comm`          | Overlap communication     | true    | `zero/config.py:138` |
| `contiguous_gradients`  | Reduce fragmentation      | true    | `zero/config.py:102` |
| `round_robin_gradients` | Better memory balance     | false   | `zero/config.py:302` |

### 5.3 ZeRO-3 Configuration

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    }
}
```

**ZeRO-3 Specific Parameters**:

| Parameter                                   | Description                     | Source               |
|---------------------------------------------|---------------------------------|----------------------|
| `stage3_max_live_parameters`                | Max params to keep in GPU       | `zero/config.py:238` |
| `stage3_max_reuse_distance`                 | Max distance for param reuse    | `zero/config.py:244` |
| `stage3_prefetch_bucket_size`               | Prefetch bucket size            | `zero/config.py:215` |
| `offload_param`                             | Offload parameters to CPU/NVMe  | `zero/config.py:157` |
| `stage3_gather_16bit_weights_on_model_save` | Gather weights on save          | `zero/config.py:250` |
| `stage3_allgather_sequential`               | Sequential allgather for memory | `zero/config.py:272` |

### 5.4 Offloading Options

**CPU Offload**:
```json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

**NVMe Offload** (ZeRO-Infinity):
```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme"
        },
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme"
        }
    }
}
```

**Source**: `runtime/zero/offload_config.py`, `docs/_tutorials/zero-offload.md`

### 5.5 ZeRO-3 Model Initialization

For ZeRO-3 with very large models, use special initialization:

```python
import deepspeed
from deepspeed.zero import Init

# Initialize model with ZeRO-3 context
with Init(data_parallel_group=None,
          remote_device="cpu",  # or "nvme"
          enabled=True):
    model = MyLargeModel()  # Model is partitioned immediately
```

**GatheredParameters context** for accessing full parameters:
```python
from deepspeed.zero import GatheredParameters

# Access full parameter for operations like initialization
with GatheredParameters(param, modifier_rank=0):
    # param is now full on rank 0
    param.data.normal_(mean=0.0, std=0.02)
```

**Source**: `runtime/zero/partition_parameters.py`

---

## 6. Training APIs

### 6.1 Basic Training Loop

```python
import deepspeed
import torch

# Initialize model
model = MyModel()

# Initialize DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json"
)

# Training loop
for step, batch in enumerate(data_loader):
    # Forward
    loss = model_engine(batch)
    
    # Backward
    model_engine.backward(loss)
    
    # Optimizer step
    model_engine.step()
```

### 6.2 Checkpointing

**Saving**:
```python
# Save checkpoint (all ranks must call)
client_sd = {'step': step, 'epoch': epoch}
model_engine.save_checkpoint(
    save_dir="checkpoints/",
    tag=f"step_{step}",
    client_state=client_sd
)
```

**Loading**:
```python
# Load checkpoint
_, client_sd = model_engine.load_checkpoint(
    load_dir="checkpoints/",
    tag="step_1000"
)
step = client_sd['step']
epoch = client_sd['epoch']
```

**⚠️ Important**: All processes must call `save_checkpoint`, not just rank 0.

**Source**: `runtime/engine.py:save_checkpoint/load_checkpoint`

### 6.3 Gradient Accumulation

DeepSpeed handles gradient accumulation automatically:

```python
# In config
{
    "gradient_accumulation_steps": 4
}
```

The training loop remains the same - DeepSpeed accumulates gradients internally.

**Checking accumulation boundary**:
```python
if model_engine.is_gradient_accumulation_boundary():
    # Do something only at actual optimizer step
    pass
```

### 6.4 Accessing Model Information

```python
# Get local rank
local_rank = model_engine.local_rank

# Get global rank
rank = model_engine.global_rank

# Get world size
world_size = model_engine.world_size

# Get current learning rate
lr = model_engine.get_lr()

# Set learning rate
model_engine.set_lr(new_lr)
```

---

## 7. Data Loading

### 7.1 DeepSpeed DataLoader

DeepSpeed provides an optimized DataLoader:

```python
from torch.utils.data import Dataset

# Create dataset
dataset = MyDataset()

# Initialize with DeepSpeed
model_engine, optimizer, dataloader, lr_scheduler = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    training_data=dataset,
    collate_fn=collate_fn,
    config=ds_config
)

# Use the returned dataloader
for batch in dataloader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

**Source**: `runtime/dataloader.py`

### 7.2 Manual DataLoader

If you need more control:

```python
from torch.utils.data import DataLoader, DistributedSampler

# Create sampler for distributed training
sampler = DistributedSampler(dataset, shuffle=True)

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=per_device_batch_size,
    sampler=sampler,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)

# Don't pass training_data to deepspeed.initialize
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)

# Training loop
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # Important for proper shuffling
    for batch in dataloader:
        loss = model_engine(batch)
        model_engine.backward(loss)
        model_engine.step()
```

---

## 8. Advanced Features

### 8.1 Activation Checkpointing

```python
from deepspeed.runtime.activation_checkpointing import checkpointing

# Enable in config
{
    "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": true,
        "contiguous_memory_optimization": false,
        "number_checkpoints": null,
        "synchronize_checkpoint_boundary": false,
        "profile": false
    }
}
```

**In model code**:
```python
from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint

# Wrap your layer with checkpointing
output = checkpoint(self.layer, input)
```

**Source**: `runtime/activation_checkpointing/`, `docs/_tutorials/progressive_layer_dropping.md`

### 8.2 Pipeline Parallelism

```python
from deepspeed.pipe import PipelineModule

# Define layers as a list
layers = [
    Layer1(),
    Layer2(),
    Layer3(),
    Layer4()
]

# Wrap model for pipeline parallelism
model = PipelineModule(
    layers=layers,
    num_stages=4,
    loss_fn=loss_function
)
```

**Source**: `runtime/pipe/`, `docs/_tutorials/pipeline.md`

### 8.3 Tensor Parallelism

```json
{
    "tensor_parallel": {
        "tp_size": 4,
        "mpu": null
    }
}
```

**Source**: `runtime/tensor_parallel/`, `docs/_tutorials/autotp-training.md`

### 8.4 Mixture of Experts (MoE)

```python
from deepspeed.moe.layer import MoE

# Replace FFN with MoE layer
moe_layer = MoE(
    hidden_size=hidden_size,
    expert=expert_module,
    num_experts=64,
    ep_size=8,  # Expert parallel size
    k=2,        # Top-k experts
    capacity_factor=1.0
)
```

**Source**: `moe/`, `docs/_tutorials/mixture-of-experts.md`

### 8.5 Communication Compression

```json
{
    "communication_data_type": "fp16",
    "round_robin_gradients": true
}
```

---

## 9. Inference Optimization

### 9.1 DeepSpeed Inference

DeepSpeed provides optimized inference with kernel injection:

```python
import deepspeed
import torch

# Load model
model = AutoModel.from_pretrained("model_name")

# Initialize inference engine
ds_engine = deepspeed.init_inference(
    model,
    tensor_parallel={"tp_size": 4},
    dtype=torch.half,
    replace_with_kernel_inject=True,
    max_tokens=1024
)

model = ds_engine.module
```

### 9.2 Inference Configuration

| Parameter                    | Description                                         |
|------------------------------|-----------------------------------------------------|
| `tensor_parallel.tp_size`    | Tensor parallel size                                |
| `dtype`                      | Data type (torch.half, torch.bfloat16, torch.float) |
| `replace_with_kernel_inject` | Inject optimized kernels                            |
| `injection_policy`           | Custom injection policy                             |
| `max_tokens`                 | Max tokens for allocation                           |

### 9.3 Quantization for Inference

```python
# INT8 quantization
ds_engine = deepspeed.init_inference(
    model,
    dtype=torch.int8,
    quantization={
        "enabled": True,
        "bits": 8
    }
)
```

**Source**: `inference/`, `docs/_tutorials/inference-tutorial.md`

---

## 10. Integration Patterns

### 10.1 With HuggingFace Transformers

```python
from transformers import AutoModel
import deepspeed

model = AutoModel.from_pretrained("bert-base-uncased")

# DeepSpeed config
ds_config = {
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 2,
    "optimizer": {"type": "AdamW", "params": {"lr": 5e-5}},
    "zero_optimization": {"stage": 2}
}

model_engine, _, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)
```

**Reference**: [HuggingFace DeepSpeed Integration](https://huggingface.co/docs/transformers/deepspeed)

### 10.2 With PyTorch Lightning

```python
import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy

trainer = pl.Trainer(
    strategy=DeepSpeedStrategy(
        stage=2,
        offload_optimizer=True
    ),
    devices=4,
    accelerator="gpu"
)
```

**Reference**: [PyTorch Lightning DeepSpeed](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.strategies.DeepSpeedStrategy.html)

### 10.3 Custom Model Integration

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        return self.decoder(self.encoder(x))

# DeepSpeed handles the rest
model = MyModel()
model_engine, _, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json"
)
```

---

## 11. Troubleshooting

### 11.1 Common Errors

**Error**: `TypeError: '>' not supported between instances of 'str' and 'int'`

**Cause**: Using `"auto"` for `train_batch_size` or scheduler parameters

**Solution**:
```json
// ❌ Wrong
{
    "train_batch_size": "auto"
}

// ✅ Correct
{
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 2
}
```

---

**Error**: `AssertionError: Check batch related parameters`

**Cause**: Batch size mismatch

**Solution**: Ensure:
```
train_batch_size = micro_batch × grad_acc × num_gpus
```

---

**Error**: `CUDA out of memory`

**Solutions**:
1. Reduce `train_micro_batch_size_per_gpu`
2. Enable gradient checkpointing
3. Use ZeRO-3 with offloading
4. Reduce model size
5. Enable CPU offloading

---

**Error**: `NCCL communication error`

**Solutions**:
1. Check `NCCL_DEBUG=INFO` for details
2. Ensure all GPUs are visible
3. Check network connectivity (multi-node)
4. Try `NCCL_P2P_DISABLE=1` if P2P issues

---

**Error**: `RuntimeError: Expected all tensors to be on the same device`

**Cause**: Model and data on different devices

**Solution**: DeepSpeed handles device placement - don't manually move model to GPU.

### 11.2 Debugging Tips

```bash
# Enable NCCL debugging
export NCCL_DEBUG=INFO

# Enable PyTorch distributed debugging
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Check GPU visibility
nvidia-smi

# Test DeepSpeed installation
deepspeed --version

# Disable P2P if having issues
export NCCL_P2P_DISABLE=1
```

---

## 12. Best Practices

### 12.1 Configuration Checklist

- [ ] `train_micro_batch_size_per_gpu` is set (integer)
- [ ] `gradient_accumulation_steps` is set (integer)
- [ ] `optimizer.type` is specified
- [ ] `optimizer.params.lr` is set
- [ ] ZeRO stage matches your use case
- [ ] Mixed precision (BF16/FP16) is enabled (not both)
- [ ] Gradient clipping is configured (if needed)
- [ ] Scheduler `total_num_steps` is set correctly

### 12.2 Performance Optimization

1. **Use BF16** on Ampere+ GPUs (better than FP16)
2. **Enable ZeRO-2** for most cases (good balance)
3. **Use gradient accumulation** for large effective batch sizes
4. **Enable communication overlap** (`overlap_comm: true`)
5. **Use contiguous gradients** to reduce fragmentation
6. **Pin memory** for CPU offloading (`pin_memory: true`)
7. **Use multiple data loader workers**

### 12.3 Memory Optimization

| Technique                | Memory Savings | When to Use                |
|--------------------------|----------------|----------------------------|
| ZeRO-1                   | 4x             | Large optimizer states     |
| ZeRO-2                   | 8x             | **Default recommendation** |
| ZeRO-3                   | 8x+            | Extreme model sizes        |
| CPU Offload              | 10x+           | Models > 10B params        |
| NVMe Offload             | 100x+          | Models > 100B params       |
| Activation Checkpointing | 2-3x           | Long sequences             |

### 12.4 Recommended Configs by Model Size

**Small Models (< 1B)**:
```json
{
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 4,
    "bf16": {"enabled": true},
    "zero_optimization": {"stage": 1}
}
```

**Medium Models (1B - 10B)**:
```json
{
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 8,
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": true,
        "contiguous_gradients": true
    }
}
```

**Large Models (10B - 100B)**:
```json
{
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 16,
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true
    }
}
```

**Extreme Models (> 100B)**:
```json
{
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 32,
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme"
        },
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme"
        }
    }
}
```

### 12.5 Training Workflow Checklist

**Before Training**:
- [ ] Test on single GPU first
- [ ] Verify data loading works correctly
- [ ] Check model forward/backward pass
- [ ] Validate configuration with `deepspeed --num_gpus=1`

**During Training**:
- [ ] Monitor GPU utilization (`nvidia-smi`)
- [ ] Check loss convergence
- [ ] Verify checkpoint saving/loading
- [ ] Monitor memory usage

**Multi-GPU Specific**:
- [ ] Verify all GPUs are detected
- [ ] Check gradient synchronization
- [ ] Test checkpoint consistency across ranks
- [ ] Validate batch size scaling

---

## 13. Quick Reference

### 13.1 Configuration Templates

**Template 1: Single GPU, Small Model**
```json
{
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 5e-5}
    },
    "bf16": {"enabled": true},
    "zero_optimization": {"stage": 0}
}
```

**Template 2: Multi-GPU, Medium Model (ZeRO-2)**
```json
{
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 8,
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 1e-5}
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-5,
            "warmup_num_steps": 100,
            "total_num_steps": 10000
        }
    },
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "overlap_comm": true,
        "contiguous_gradients": true
    },
    "gradient_clipping": 1.0
}
```

**Template 3: Large Model with CPU Offload (ZeRO-2)**
```json
{
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 16,
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 1e-5}
    },
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true
    }
}
```

**Template 4: Extreme Scale (ZeRO-3 + NVMe)**
```json
{
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 32,
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 5e-6}
    },
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme"
        },
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme"
        },
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_prefetch_bucket_size": 1e7,
        "stage3_param_persistence_threshold": 1e5,
        "stage3_gather_16bit_weights_on_model_save": true
    }
}
```

### 13.2 Source Code Reference

| Component          | File Path                       |
|--------------------|---------------------------------|
| **Initialization** | `deepspeed/__init__.py`         |
| **Engine**         | `runtime/engine.py`             |
| **Config**         | `runtime/config.py`             |
| **Constants**      | `runtime/constants.py`          |
| **ZeRO Config**    | `runtime/zero/config.py`        |
| **ZeRO Stage 1/2** | `runtime/zero/stage_1_and_2.py` |
| **ZeRO Stage 3**   | `runtime/zero/stage3.py`        |
| **BF16 Optimizer** | `runtime/bf16_optimizer.py`     |
| **FP16 Optimizer** | `runtime/fp16/`                 |
| **LR Schedules**   | `runtime/lr_schedules.py`       |
| **DataLoader**     | `runtime/dataloader.py`         |
| **Checkpointing**  | `runtime/model_checkpointing/`  |
| **Inference**      | `inference/`                    |
| **MoE**            | `moe/`                          |

### 13.3 Documentation Reference

| Topic                | Path                                          |
|----------------------|-----------------------------------------------|
| Getting Started      | `docs/_tutorials/getting-started.md`          |
| ZeRO Tutorial        | `docs/_tutorials/zero.md`                     |
| ZeRO-Offload         | `docs/_tutorials/zero-offload.md`             |
| Pipeline Parallelism | `docs/_tutorials/pipeline.md`                 |
| Tensor Parallelism   | `docs/_tutorials/autotp-training.md`          |
| MoE                  | `docs/_tutorials/mixture-of-experts.md`       |
| Inference            | `docs/_tutorials/inference-tutorial.md`       |
| Large Models         | `docs/_tutorials/large-models-w-deepspeed.md` |

### 13.4 Official Resources

- **Website**: https://www.deepspeed.ai/
- **GitHub**: https://github.com/deepspeedai/DeepSpeed
- **Documentation**: https://deepspeed.readthedocs.io/
- **Tutorials**: https://www.deepspeed.ai/tutorials/

---

## References

- **Source Code**: `third-part/DeepSpeed-master/`
- **Documentation**: `third-part/DeepSpeed-master/docs/`
- **Tutorials**: `third-part/DeepSpeed-master/docs/_tutorials/`
- **Engine**: `third-part/DeepSpeed-master/deepspeed/runtime/engine.py`
- **Config**: `third-part/DeepSpeed-master/deepspeed/runtime/config.py`
- **Constants**: `third-part/DeepSpeed-master/deepspeed/runtime/constants.py`
- **ZeRO Config**: `third-part/DeepSpeed-master/deepspeed/runtime/zero/config.py`
- **Official Docs**: https://www.deepspeed.ai/
