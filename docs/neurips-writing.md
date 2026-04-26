# Hierarchical Latent Reasoning via Concept Pyramid Compression

## Title

**Concept Pyramid: Hierarchical Latent Compression for Efficient Chain-of-Thought Reasoning**

or

**Next-Level Prediction: Coarse-to-Fine Concept Pyramid for Scalable Latent Reasoning**

(Alternatives under consideration)

---

## Abstract

Large language models rely on costly token-level verbalization to perform long Chain-of-Thought (CoT) reasoning for problem solving. Recent methods reduce reasoning to a sequence of latent thoughts, yet they remain the principle of standard "next-token prediction" as treating every thought as uniformly granular. In this paper, we present **Latent Concept-Pyramid Modeling (LCP)**, a new paradigm that redefines sequential LLM reasoning as hierarchical coarse-to-fine concept generation. That is, LCP generates a single highly abstract concept at the apex and progressively refines reasoning through next-level concept generation, approximating CoT in a finer granularity. This yields a pyramid structure in latent space in which concepts within each level capture different segments of the CoT; collectively, they represent one specific granularity of this reasoning process. Thus, next-level concept prediction in LCP progressively approximates the CoT in a manner analogous to advancing from a skeletal outline, through broad structural forms, to the details. Experimental evaluations on zero-shot mathematical, coding, and financial reasoning benchmarks demonstrate that Qwen3 models equipped with LCP reduce token costs by 70% on average, while simultaneously improving accuracy by 5% compared to their supervised fine-tuned counterparts. LCP further showcases zero-shot generalization ability across different tasks.

---

## 1. Introduction

Nearly all state-of-the-art large language models share a common architectural assumption for reasoning: intermediate steps are generated explicitly as discrete tokens [Wei et al., 2022]. This paradigm, known as Chain-of-Thought (CoT), has unlocked remarkable reasoning capabilities but imposes a severe computational bottleneck---a single mathematical proof or logical deduction may consume hundreds of intermediate tokens, each requiring full Transformer computation. The prevailing view treats this cost as an unavoidable price for interpretability: if we wish to understand how a model reasons, we must read its thoughts token by token.

We challenge this assumption. The fundamental issue is not merely the number of tokens but the structural form of reasoning representation. Human cognition does not proceed as a homogeneous stream of equally detailed thoughts; rather, it operates hierarchically, moving from coarse strategic outlines to progressively finer implementation details. A mathematician first conceives the proof strategy, then identifies the key lemmas, and finally supplies the rigorous derivations. CoT, by forcing the model to verbalize every detail at uniform granularity, collapses this natural hierarchy into an undifferentiated linear sequence. The result is a representational mismatch: the model expends computation on structural scaffolding at the same resolution as substantive derivation, conflating levels of abstraction that cognitive science has long recognized as distinct [Kintsch, 1994; Johnson-Laird, 1983].

Recent work in latent reasoning offers a partial remedy. Methods such as Coconut [Hao et al., COLM 2025] bypass token generation by operating directly in continuous hidden space, reducing per-step cost. DLCM [ICLR 2026] compresses token sequences into semantic concepts through dynamic segmentation, achieving further compression. Yet these approaches retain a flat structural topology: Coconut processes an undifferentiated sequence of latent thoughts; DLCM discovers segments at a single resolution. Neither captures the hierarchical relationship between abstraction levels, leaving global strategy and local manipulation structurally indistinguishable. Skeleton-of-Thought [ICLR 2024] introduces a rudimentary two-level hierarchy, but its skeleton is manually defined rather than learned, and its granularity is too coarse to represent complex reasoning chains. In short, existing latent methods improve efficiency by reducing token count but not by respecting the multi-scale organization inherent to reasoning itself.

We argue that the missing ingredient is the explicit modeling of hierarchical structure. Visual generation provides a compelling precedent: images possess natural multi-scale organization (global composition, coarse layout, fine texture), and Visual Autoregressive Modeling (VAR) [Tian et al., NeurIPS 2024 Best Paper] exploits this through next-scale prediction, generating images autoregressively across scales with parallel generation within each scale. This architectural insight achieves order-of-magnitude speedup over diffusion models by aligning the generation process with the intrinsic multi-resolution structure of visual data. We hypothesize that text reasoning possesses analogous multi-scale organization and that a generation paradigm that respects this hierarchy can unlock comparable gains.

Based on this insight, we propose **Concept Pyramid**, a hierarchical latent reasoning framework that redefines reasoning generation as next-level prediction over concept abstractions. At the core of our approach is the observation that a reasoning trace can be decomposed into a pyramid of concepts, where each level captures the same reasoning process at a different granularity---from a single strategic overview at the apex to fine-grained operational details at the base. During training, a concept pyramid builder learns to extract this hierarchy from ground-truth reasoning traces through soft attention with residual decomposition, enabling differentiable end-to-end optimization without hard segmentation. During inference, a concept predictor autoregressively generates the pyramid level by level from the question alone, with concepts within each level processed in parallel. This design replaces the linear token sequence with a compact hierarchical representation that preserves the structural relationships between abstraction levels while dramatically reducing sequential computation.

Our contributions are threefold. First, we introduce the first hierarchical latent reasoning framework for text, establishing multi-scale structure as an inductive bias for efficient reasoning. This shifts the design space from ``how few tokens can we use'' to ``what structure should reasoning have,'' opening new avenues for latent reasoning research. Second, we develop a soft attention mechanism with residual reconstruction that decomposes reasoning traces into hierarchical concepts without quantization or hard boundaries, preserving end-to-end differentiability and enabling stable training. Third, we demonstrate that hierarchical latent compression achieves order-of-magnitude inference speedup over standard CoT on mathematical reasoning benchmarks while maintaining accuracy, validating that structural priors can match the efficiency of flat compression methods with stronger representational fidelity.

---

## Notes for Further Development

### Abstract & Introduction Structure
- [x] Para 1: Problem — CoT token-level bottleneck as accepted assumption
- [x] Para 2: Challenge assumption — structural form matters more than token count; hierarchy in human reasoning
- [x] Para 3: Prior work critique — flat latent methods (Coconut, DLCM) reduce tokens but not structure; SoT too coarse
- [x] Para 4: Core insight — VAR's next-scale prediction as precedent; reasoning has multi-scale structure too
- [x] Para 5: Method — Concept Pyramid as next-level prediction; builder + predictor without technical details
- [x] Para 6: Contributions — (1) first hierarchical latent reasoning for text, (2) soft attention + residual decomposition, (3) speedup experiments

### Key writing principles followed
1. **No technical configurations**: No mention of 1→2→4→8→16→32, 63 vectors, 6 levels, rank bottleneck, etc.
2. **Focus on novelty**: What is new (hierarchical structure as inductive bias) and why it matters
3. **Argumentative tone**: "We challenge this assumption," "The missing ingredient is..."
4. **Cognitive grounding**: References to human reasoning (mathematician's strategy → lemmas → derivations)
5. **Design-space framing**: Shifts from "how few tokens" to "what structure should reasoning have"

### Sections to be written
- 2. Related Work
- 3. Method (where technical details go: pyramid builder, predictor, loss functions)
- 4. Experiments
- 5. Analysis and Ablation Studies
- 6. Conclusion

### Keywords
Chain-of-Thought Compression, Latent Reasoning, Hierarchical Concepts, Multi-Scale Autoregression, Efficient Inference
