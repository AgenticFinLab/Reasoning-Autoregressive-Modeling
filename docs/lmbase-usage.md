# lmbase Utility Components Usage Guide

> **Repository**: https://github.com/AgenticFinLab/lmbase

This document describes reusable utility components in `lmbase`. **All components are available directly from the installed `lmbase` package** — import and use them without reimplementation.

---

## ⚠️ MANDATORY USAGE POLICY

**Before implementing any similar functionality, you MUST:**

1. Check if `lmbase` already provides the functionality
2. **Use `lmbase` components directly** if available
3. Only implement custom solutions if `lmbase` utilities are genuinely insufficient

**Rationale**: Centralized utilities ensure consistency, reduce maintenance burden, and benefit from collective improvements.

---

## Component Categories

| Category          | Components                                     | Mandatory |
|-------------------|------------------------------------------------|-----------|
| **Storage**       | `BlockBasedStoreManager`, `HistoryBuffer`      | ✅ YES     |
| **Environment**   | `device_map` configuration pattern             | ✅ YES     |
| **Dataset**       | `VisualTextBase`, `TextSample`, `data_factory` | Optional  |
| **Serialization** | `BaseContainer`                                | Optional  |

---

## 1. Storage Components (MANDATORY)

### 1.1 BlockBasedStoreManager

**Import**: `from lmbase.utils.tools import BlockBasedStoreManager`

**Purpose**: Block-based JSON storage with automatic file rotation and PyTorch tensor support.

**Use for**: Large-scale record storage, avoiding monolithic JSON files, tensor serialization.

**Strict Requirement**: Use this directly from `lmbase`. Do not reimplement.

---

### 1.2 HistoryBuffer

**Import**: `from lmbase.utils.history import HistoryBuffer`

**Purpose**: Memory-efficient history with hot (memory) + cold (disk) storage and automatic overflow persistence.

**Use for**: Long-running processes, time-series accumulation, memory-constrained environments.

**Strict Requirement**: Use this directly from `lmbase`. Do not reimplement.

---

## 2. Environment Configuration (MANDATORY)

### 2.1 Device Map Pattern

**Location**: `lmbase/template/` configuration files

**Purpose**: Centralized device assignment with priority-based GPU allocation.

**Pattern**:
```yaml
environment:
  device_map:
    Component_1:
      device: "auto"
      priority: 1
    Component_2:
      device: "auto"
      priority: 2
```

**Strict Requirement**: Use this pattern from `lmbase` templates. Do not scatter `device` settings elsewhere.

---

## 3. Dataset Components (Optional)

### 3.1 VisualTextBase

**Import**: `from lmbase.dataset.base import VisualTextBase, TextSample`

**Purpose**: Base class for LLM/VLM datasets with standardized sample containers.

**Use for**: New dataset implementations, consistent sample formatting.

---

### 3.2 Dataset Registry

**Import**: `from lmbase.dataset import registry`

**Purpose**: Centralized dataset loading by name.

**Available Datasets** (40+): GSM8K, MATH, MMMU, ScienceQA, AIME2024/2025, HumanEval, FinanceBench, HotpotQA, STaRK, etc.

---

## 4. Serialization Components (Optional)

### 4.1 BaseContainer

**Import**: `from lmbase.utils.tools import BaseContainer`

**Purpose**: JSON-serializable dataclass base with tensor support.

**Use for**: Output structures requiring JSON serialization with PyTorch tensors.

---

## Usage Rule

**All components are importable directly from the installed `lmbase` package.**

Do not copy source code. Do not reimplement. Import and use:

```python
from lmbase.utils.tools import BlockBasedStoreManager, BaseContainer
from lmbase.utils.history import HistoryBuffer
from lmbase.dataset.base import VisualTextBase, TextSample
from lmbase.dataset import registry
```

---

## Quick Reference

| Component                | Import Path               | Mandatory |
|--------------------------|---------------------------|-----------|
| `BlockBasedStoreManager` | `lmbase.utils.tools`      | ✅ YES     |
| `HistoryBuffer`          | `lmbase.utils.history`    | ✅ YES     |
| `device_map` pattern     | `lmbase/template/*`       | ✅ YES     |
| `VisualTextBase`         | `lmbase.dataset.base`     | Optional  |
| `data_factory`           | `lmbase.dataset.registry` | Optional  |
| `BaseContainer`          | `lmbase.utils.tools`      | Optional  |

---

## Repository Structure

```
lmbase/
├── lmbase/
│   ├── utils/
│   │   ├── tools.py          # BlockBasedStoreManager, BaseContainer
│   │   ├── history.py        # HistoryBuffer
│   │   └── ...
│   ├── template/             # Configuration templates
│   │   ├── MultiAgentRL/
│   │   └── ModelLearn/
│   ├── dataset/              # Dataset implementations
│   └── ...
└── docs/
    └── lmbase-usage.md       # This file
```

For API details, refer to source code docstrings in the installed package.
