# Research Plans

This directory contains the detailed phase-by-phase research methodology for the VSGR (Visual Spatial Graph Region) project.

## Overview

The research follows a systematic 9-phase methodology, progressing from research direction setting to final cross-experiment analysis. Each phase has clear inputs, outputs, and verification criteria.

```
Phase 0: Research Direction Setting (USER INPUT)
    ↓
Phase 1: Codebase Initialization (LLM)
    ↓
Phase 2: Modular Architecture Design (LLM)
    ↓
Phase 3: Literature Survey (LLM)
    ↓
Phase 4: Source Code Deep-Dive (LLM)
    ↓
Phase 5: Hypothesis Refinement (LLM) ←──────┐
    ↓                                        │
Phase 6: Experiment Design (LLM)             │
    ↓                                        │
Phase 7: Experiment Execution (LLM)          │
    ↓                                        │
Phase 8: Cross-Experiment Analysis (LLM)     │
    ↓                                        │
Conclusions ─────────────────────────────────┘
```

## Phase Documents

| Phase | Document                                                                   | Actor | Status             | Description                                                 |
|-------|----------------------------------------------------------------------------|-------|--------------------|-------------------------------------------------------------|
| 0     | [phase0-research-direction.md](phase0-research-direction.md)               | USER  | **Required Input** | Define research domain, hypothesis, verification approaches |
| 1     | [phase1-codebase-initialization.md](phase1-codebase-initialization.md)     | LLM   | Automated          | Transform template into installable package                 |
| 2     | [phase2-modular-architecture.md](phase2-modular-architecture.md)           | LLM   | Automated          | Design module hierarchy and base classes                    |
| 3     | [phase3-literature-survey.md](phase3-literature-survey.md)                 | LLM   | Automated          | Systematic paper search and analysis                        |
| 4     | [phase4-source-code-deepdive.md](phase4-source-code-deepdive.md)           | LLM   | Automated          | Download and analyze reference implementations              |
| 5     | [phase5-hypothesis-refinement.md](phase5-hypothesis-refinement.md)         | LLM   | Iterative          | Update hypothesis based on findings                         |
| 6     | [phase6-experiment-design.md](phase6-experiment-design.md)                 | LLM   | Automated          | Design progressive experiments                              |
| 7     | [phase7-experiment-execution.md](phase7-experiment-execution.md)           | LLM   | Automated          | Execute experiments and record results                      |
| 8     | [phase8-cross-experiment-analysis.md](phase8-cross-experiment-analysis.md) | LLM   | Automated          | Cross-experiment comparison and conclusions                 |

## VSGR Project Context

### Research Direction

**Domain:** Multi-agent reinforcement learning for graph-based visual reasoning in vision-language models

**Core Hypothesis:** Visual reasoning with region relationship graphs via multi-agent reinforcement learning outperforms sequential one-step-one-region approaches while maintaining full image context and reducing inference iterations.

**Package:** `vsgr` (Visual Spatial Graph Region)

### Key Components

- **gr_reason/** — Graph-based reasoning module with RegionGraph, multi-agent reasoning
- **models/** — VLM wrappers (LLaVA, Qwen2VL) with HuggingFace integration
- **evaluate/** — Evaluation framework for visual reasoning tasks
- **utils/** — Config loading and domain utilities

### Baseline Comparisons

| Method | Approach                     | Comparison Aspect         |
|--------|------------------------------|---------------------------|
| VLM-R³ | Sequential region reasoning  | Primary accuracy baseline |
| CoFFT  | Multi-step visual reasoning  | Efficiency baseline       |
| ParGo  | Grid-based region extraction | Region quality baseline   |

## How to Use

1. **For New Projects:** Copy these phase documents and adapt Phase 0 with your research direction
2. **For VSGR Development:** Reference the appropriate phase document for current development stage
3. **For Verification:** Each phase document includes verification checklists at the end

## Document Structure

Each phase document follows a consistent structure:

1. **Overview** — What this phase accomplishes
2. **Prerequisites** — What must be completed before starting
3. **Step-by-Step Execution** — Detailed instructions
4. **VSGR-Specific Examples** — Concrete examples for this project
5. **Output Verification** — Checklist to confirm completion
6. **Next Phase** — Pointer to subsequent phase

## Status Tracking

| Phase | Status         | Notes                                      |
|-------|----------------|--------------------------------------------|
| 0     | ✅ Completed    | VSGR research direction defined            |
| 1     | ✅ Completed    | Package structure established              |
| 2     | ✅ Completed    | Base classes and config templates designed |
| 3     | 🔄 In Progress | Literature survey ongoing                  |
| 4     | ⏳ Pending      | Awaiting Phase 3 completion                |
| 5     | ⏳ Pending      | Awaiting Phase 3-4 completion              |
| 6     | ⏳ Pending      | Awaiting Phase 5 completion                |
| 7     | ⏳ Pending      | Awaiting Phase 6 completion                |
| 8     | ⏳ Pending      | Awaiting Phase 7 completion                |

## Related Documents

- [Research-plan.md](../Research-plan.md) — Original combined research plan (superseded by this directory)
- [llm-coding-rules.md](../llm-coding-rules.md) — Coding standards for LLM development
- [Progress-record.md](../Progress-record.md) — Informal progress logging
- [reference.md](../reference.md) — Literature survey results

---

**Last Updated:** 2024-01-15  
**Maintainer:** VSGR Research Team
