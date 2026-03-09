# Latent Chain-of-Thought Reasoning: Decoder-Based Models & Inference Exploration

## Survey Reference
**Reasoning Beyond Language: A Comprehensive Survey on Latent Chain-of-Thought Reasoning**
- arXiv: 2505.16782
- Authors: Xinghao Chen et al.
- GitHub: https://github.com/EIT-NLP/Awesome-Latent-CoT

---

## Part 1: Decoder-Based Models (Layer-wise Vertical Level)

This section covers **layer-wise vertical approaches** to latent reasoning - architectural modifications to decoder-only transformers that enable reasoning through depth recurrence and adaptive computation. These methods process reasoning vertically through model layers rather than horizontally through token sequences.

### 1.1 CoTFormer: Budget-Adaptive Chain-of-Thought Architecture

**Paper:** CoTFormer: A Chain-of-Thought Driven Architecture with Budget-Adaptive Computation Cost at Inference  
**Authors:** Amirkeivan Mohtashami, Matteo Pagliardini, Martin Jaggi (EPFL)  
**arXiv:** 2310.10845 | **Venue:** ICLR 2024

#### Core Innovation
CoTFormer observes that **Chain-of-Thought resembles employing a deeper transformer** by re-applying the model multiple times. It mimics CoT at the token level with architectural modifications.

#### Architecture Design
- **Token-level CoT Mimicry:** Each token goes through multiple "thought" iterations
- **Key Difference from Simple Recurrence:** Past tokens' attention remains accessible across iterations
- **Budget-Adaptive Computation:** Automatically allocates compute to tokens that need it most

#### Key Technical Contributions
1. **Weight Sharing with Memory:**
   - Unlike Universal Transformer's simple weight tying
   - Maintains access to previous intermediary token representations
   - Critical for capturing CoT-style iterative refinement

2. **Compute-Adaptive Inference:**
   - Variable depth per token based on complexity
   - Pushes perplexity-compute Pareto frontier forward
   - Reduces overall computation without accuracy loss

---

### 1.2 Huginn: Recurrent Depth for Test-Time Scaling

**Paper:** Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach  
**Authors:** Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, et al. (UMD, LLNL)  
**arXiv:** 2502.05171 | **Model:** 3.5B parameters, 800B tokens  
**Model:** https://huggingface.co/tomg-group-umd/huginn-3.5b

#### Core Innovation
Huginn demonstrates that **test-time computation can be scaled through depth recurrence** rather than generating more tokens. The architecture iterates a recurrent block to arbitrary depth at inference time.

#### Architecture Design
```
Input → [Pre-layers] → [Recurrent Block × N iterations] → [Post-layers] → Output
```

- **Recurrent Block:** A fixed set of transformer layers that can be repeatedly applied
- **Depth Unrolling:** At test time, the recurrent block is unrolled to varying depths based on task difficulty
- **No Specialized Training Data:** Unlike CoT, doesn't require reasoning traces in training

#### Key Technical Contributions
1. **Implicit Latent Reasoning:** Reasoning occurs within model's internal representations
2. **Adaptive Computation:** Zero-shot per-token adaptive compute, KV-cache sharing
3. **Test-Time Scaling Law:** 3.5B model achieves performance equivalent to 50B parameters

---

### 1.3 LTO: Latent Thinking Optimization

**Paper:** Latent Thinking Optimization: Your Latent Reasoning Language Model Secretly Encodes Reward Signals in Its Latent Thoughts  
**Authors:** Hanwen Du, Yuxin Dong, Xia Ning  
**arXiv:** 2509.26314

#### Core Innovation
LTO reveals that **latent thoughts can encode reward signals** that distinguish correct from incorrect reasoning. It introduces a Latent Reward Model (LRM) to optimize latent thinking processes.

#### Key Technical Contributions
1. **Latent Reward Discovery:**
   - Latent thoughts contain analyzable patterns
   - Can distinguish between correct and incorrect reasoning paths
   - Enables supervisory signals in latent space

2. **Latent Reward Model (LRM):**
   - Trained to identify patterns correlating with correct answers
   - Acts as supervisory tool for latent thinking optimization
   - Generalizes across various domains

3. **Reward-Based Optimization:**
   - KL-regularized reweighting and policy gradient methods
   - Refines latent thought trajectories
   - Enables more effective exploration of latent space

#### Experimental Results
- Significantly improves reasoning accuracy on Huginn-3.5B
- Effectively detects and corrects incorrect latent thinking patterns
- Demonstrates versatility across multiple reasoning domains

---

### 1.4 ITT: Inner Thinking Transformer

**Paper:** Inner Thinking Transformer: Leveraging Dynamic Depth Scaling to Foster Adaptive Internal Thinking  
**Authors:** Yilong Chen et al.  
**arXiv:** 2502.13842 | **Venue:** ACL 2025

#### Core Innovation
ITT **redefines layer computations as implicit thinking steps**, employing adaptive token routing and residual thinking connections to process critical tokens more deeply without increasing parameters.

#### Architecture Design
1. **Adaptive Token Routing:** Dynamic computation allocation based on token criticality
2. **Residual Thinking Connections:** Iterative refinement of representations across thinking steps
3. **Thinking Step Encoding:** Differentiates between reasoning phases

#### Key Technical Contributions
1. **Performance/Parameter Efficiency:**
   - 162M ITT achieves 96.5% of 466M Transformer performance
   - 43.2% reduction in training data requirements
   - 355M ITT matches 1B Transformer performance

2. **Gradient Spike Mitigation:**
   - Identifies critical tokens causing gradient spikes in standard Transformers
   - Allows deeper processing for these critical tokens
   - Architecture-aware optimization of implicit thinking

#### Implications for RAM Project
- **Efficiency:** Significant parameter savings with maintained performance
- **Adaptive depth:** Critical tokens get more reasoning compute
- **No output overhead:** Thinking is implicit, not verbalized

---

### 1.5 Pondering LM: Pretraining to Ponder in Continuous Space

**Paper:** Pretraining Language Models to Ponder in Continuous Space  
**Authors:** Boyi Zeng et al.  
**arXiv:** 2505.20674

#### Core Innovation
Pondering LM introduces a **pondering mechanism during pretraining** that allows models to engage in deeper cognitive processing by computing weighted sums of token embeddings rather than directly sampling.

#### Architecture Design
- **Pondering Step:** Additional computational step between hidden state and token prediction
- **Self-Supervised Learning:** Applied to GPT-2, Pythia, and LLaMA architectures
- **Continuous Space Processing:** Enables richer intermediate representations

#### Key Technical Contributions
1. **Horizontal Scaling Strategy:**
   - Improves generation quality during pretraining
   - Alternative to simply increasing model depth
   - Better parameter efficiency

2. **Performance Results:**
   - PonderingPythia-2.8B surpasses standard Pythia-6.9B
   - PonderingPythia-1B matches TinyLlama-1.1B (trained on 10x more data)
   - Addresses data scarcity and diminishing returns in scaling

---

### 1.6 Zhu et al.: Reasoning by Superposition

**Paper:** Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought  
**Authors:** Hanlin Zhu et al.  
**arXiv:** 2505.12514

#### Core Innovation
Provides **theoretical analysis** showing that continuous CoT can solve problems requiring **polynomial fewer steps** than discrete CoT by encoding multiple search frontiers simultaneously (superposition).

#### Key Theoretical Contributions
1. **Superposition States:**
   - Continuous CoT encodes multiple search frontiers simultaneously
   - Analogous to parallel breadth-first search
   - Two-layer transformer with continuous CoT solves graph reachability in fewer steps

2. **Discrete vs Continuous:**
   - Discrete CoT requires quadratic decoding steps (relative to graph vertices)
   - Continuous CoT achieves polynomial improvement
   - Sequential discrete approach leads to inefficiencies and local optima

3. **Empirical Validation:**
   - Encoding of multiple search paths emerges naturally during training
   - No explicit guidance required for superposition behavior

#### Implications for RAM Project
- **Theoretical foundation:** Justifies continuous reasoning over discrete
- **Parallel exploration:** Multiple reasoning paths explored simultaneously
- **Efficiency gains:** Polynomial improvement potential

---

### 1.7 MoR: Mixture-of-Recursions

**Paper:** Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation  
**Authors:** Sangmin Bae et al. (KAIST AI, Google DeepMind)  
**arXiv:** 2507.10524 | **Venue:** NeurIPS 2025

#### Core Innovation
MoR **integrates parameter sharing and adaptive computation** in a single Recursive Transformer, using lightweight routers to dynamically assign varying recursion depths to individual tokens.

#### Architecture Design
1. **Shared Layer Stack:** Reuses layers across recursion steps for parameter efficiency
2. **Lightweight Routers:** Dynamically assign recursion depth per token
3. **KV Sharing Variant:** Reuses KV pairs from initial recursion, reducing memory

#### Key Technical Contributions
1. **Token-Level Adaptive Depth:**
   - Each token can have different recursion depth
   - Focuses compute on tokens that need it most
   - Selective attention only on active tokens per depth

2. **Memory Efficiency:**
   - KV caching only for essential pairs
   - Reduced memory footprint vs standard recursive models
   - Maintains high throughput

3. **Pareto Frontier:**
   - Lower validation perplexity than baselines
   - Improved few-shot accuracy (135M to 1.7B parameters)
   - Higher throughput than traditional recursive models

#### Implications for RAM Project
- **Adaptive reasoning:** Different tokens get different reasoning depth
- **Efficiency:** Combined parameter sharing and adaptive computation
- **Scalability:** Works across model sizes from 135M to 1.7B

---

## Part 2: Inference Exploration (Token-wise Horizontal Level)

This section covers **token-wise horizontal approaches** to latent reasoning - methods that enhance reasoning through test-time scaling, parallel exploration, and iterative optimization in the latent space. These approaches operate horizontally across the token sequence.

### 2.1 Parallel Scaling Methods

#### 2.1.1 SoftCoT++: Test-Time Scaling with Soft Chain-of-Thought

**Paper:** SoftCoT++: Test-Time Scaling with Soft Chain-of-Thought Reasoning  
**Authors:** Yige Xu et al.  
**arXiv:** 2505.11484  
**Code:** https://github.com/xuyige/SoftCoT

##### Core Innovation
SoftCoT++ enables **diverse exploration of reasoning paths** through perturbation of latent thoughts using multiple specialized initial tokens, combined with contrastive learning.

##### Key Technical Contributions
1. **Test-Time Scaling in Continuous Space:**
   - Operates in continuous latent space rather than discrete token space
   - Reduces information loss from token generation
   - Preserves richer reasoning representations

2. **Diverse Path Exploration:**
   - Multiple specialized initial tokens prompt varied soft thoughts
   - Contrastive learning maximizes diversity among representations
   - Overcomes single reasoning path limitation of original SoftCoT

3. **Compatibility:**
   - Works with self-consistency and other scaling techniques
   - Generalizes across LLM architectures (LLaMA-3.1-8B, etc.)
   - Average 77.57% accuracy vs 76.88% baseline

---

#### 2.1.2 PCCoT: Parallel Continuous Chain-of-Thought with Jacobi Iteration

**Paper:** Parallel Continuous Chain-of-Thought with Jacobi Iteration  
**Authors:** Haoyi Wu, Zhihao Teng, Kewei Tu  
**arXiv:** 2506.18582 | **Venue:** EMNLP 2025

##### Core Innovation
PCCoT applies **Jacobi iteration to update latent thought tokens in parallel** rather than sequentially, achieving ~50% reduction in training and inference time.

##### Key Technical Contributions
1. **Parallel Jacobi Updates:**
   - Breaks sequential dependencies of latent thought tokens
   - All thought tokens updated simultaneously per iteration
   - Fixed-point convergence for stable reasoning

2. **Efficiency Gains:**
   - ~50% reduction in training time
   - ~50% reduction in inference time
   - Maintained or improved performance vs sequential methods

3. **Hyperparameter Flexibility:**
   - Number of latent tokens (c) adjustable
   - Number of iterations (T) tunable
   - Can replicate iCoT and Pause Tokens under specific settings

---

#### 2.1.3 Butt et al.: Soft Tokens, Hard Truths

**Paper:** Soft Tokens, Hard Truths  
**Authors:** Natasha Butt et al.  
**arXiv:** 2509.19170

##### Core Innovation
Introduces **scalable reinforcement learning** for continuous Chain-of-Thought without relying on pre-trained discrete models, using "soft" tokens (mixtures with noise for exploration).

##### Key Technical Contributions
1. **Continuous CoT via RL:**
   - Learns continuous CoTs from scratch with RL
   - Mixtures of tokens with added noise for exploration
   - No dependency on pre-trained discrete reasoning

2. **Training-Inference Decoupling:**
   - Train with continuous tokens for exploration
   - Switch to discrete tokens for inference deployment
   - Best of both worlds approach

3. **Diversity Benefits:**
   - Continuous CoT maintains higher hidden-state entropy
   - Promotes exploration in token space
   - Better preserves base model predictions on OOD tasks

##### Experimental Results
- Matches or surpasses discrete-token CoT on math reasoning
- Improved diversity in reasoning paths
- Better out-of-domain generalization

---

#### 2.1.4 KaVa: Latent Reasoning via Compressed KV-Cache Distillation

**Paper:** KaVa: Latent Reasoning via Compressed KV-Cache Distillation  
**Authors:** Kuzina et al.  
**arXiv:** 2510.02312

##### Core Innovation
KaVa introduces **self-distillation from compressed KV-cache** of a teacher model into a student model, using continuous latent tokens to align KV trajectories.

##### Architecture Design
1. **Teacher Mode:** Processes complete CoT trace, creates detailed KV-caches
2. **Student Mode:** Generates continuous latent thoughts, compact representation
3. **Compression:** KV-caches compressed without significant accuracy loss

##### Key Technical Contributions
1. **KV-Cache as Supervision:**
   - Abstract knowledge in KV-cache provides supervisory signal
   - Unstructured knowledge extraction from CoT processing
   - Enables latent reasoning without explicit CoT

2. **Efficiency:**
   - Eliminates verbose CoT output during inference
   - Maintains reasoning capability with compressed representation
   - Scales well with larger model architectures

3. **Generalization:**
   - Maintains performance when transitioning from equation-based to NL reasoning
   - Combines strengths of CoT-trained models with latent inference

---

#### 2.1.5 LTA-Thinker: Latent Thought-Augmented Training

**Paper:** LTA-thinker: Latent Thought-Augmented Training Framework for Large Language Models on Complex Reasoning  
**Authors:** Jiaqi Wang et al.  
**arXiv:** 2509.12875

##### Core Innovation
LTA-Thinker addresses the **"Overthinking" problem** by optimizing distributional variance of Latent Thoughts through a multi-objective co-training strategy.

##### Architecture Design
1. **Latent Thought Generation Architecture:** Increases variance in generated Latent Thought Vectors
2. **Distribution-Based Directional Optimization:** Constrains locality and scale of distribution

##### Key Technical Contributions
1. **Semantic Alignment Loss:**
   - Uses KL divergence to ensure relevance to input question
   - Latent thought semantically connected to problem

2. **Reasoning Focus Loss:**
   - Contrastive learning guides focus on critical reasoning steps
   - Prioritizes important reasoning over auxiliary steps

3. **Multi-Objective Co-Training:**
   - Combines standard SFT with innovative losses
   - Improves information efficiency
   - Reduces computational cost

##### Experimental Results
- State-of-the-art performance across various benchmarks
- Improved reasoning capabilities with better scalability
- Addresses overthinking inefficiency

---

#### 2.1.6 Pythia Arch (PonderLM-2): Pretraining with Latent Thoughts

**Paper:** PonderLM-2: Pretraining LLM with Latent Thoughts in Continuous Space  
**Authors:** Boyi Zeng et al.  
**arXiv:** 2509.23184

##### Core Innovation
PonderLM-2 introduces **latent thought generation during pretraining** - generating an intermediate latent thought (last hidden state) before predicting each token.

##### Key Technical Contributions
1. **Horizontal Scaling:**
   - Additional computational step during pretraining
   - Refines predictions in unconstrained continuous space
   - Alternative to simply deepening architecture

2. **Parameter Efficiency:**
   - PonderLM-2-Pythia-1.4B outperforms standard Pythia-2.8B
   - 1.4B model beats 2.8B model (2x parameters) on same data
   - Better performance with fewer resources

3. **Multiple Latent Thoughts:**
   - Multiple latent thoughts before each token (like CoT)
   - Consistently enhances model performance
   - Trained on 300B tokens from the Pile

---

### 2.2 Sequential Scaling Methods

#### 2.2.1 LatentSeek: Test-Time Policy Gradient in Latent Space

**Paper:** Seek in the Dark: Reasoning via Test-Time Instance-Level Policy Gradient in Latent Space  
**Authors:** Hengli Li et al. (BIGAI, UCLA)  
**arXiv:** 2505.13308  
**Project:** https://bigai-nlco.github.io/LatentSeek/

##### Core Innovation
LatentSeek proposes **Test-Time Instance-level Adaptation (TTIA)** in latent space, using policy gradient methods to iteratively refine representations without parameter updates.

##### Method Design
1. **Latent Space Optimization:**
   - Operates on model's hidden representations
   - Policy gradient iteratively updates latent representations
   - Self-generated reward signals guide optimization

2. **No Parameter Updates:**
   - All adaptation happens at inference time
   - Avoids catastrophic forgetting
   - No training data required

##### Experimental Results
| Benchmark | Improvement over Baseline |
|-----------|---------------------------|
| GSM8K     | +4.73% average            |
| MATH-500  | Consistent gains          |
| AIME2024  | Strong performance        |

- Outperforms CoT prompting and fine-tuning methods
- Efficient: converges within few iterations
- Lightweight and scalable

---

#### 2.2.2 System-1.5 Reasoning: Dynamic Shortcuts in Latent Space

**Paper:** System-1.5 Reasoning: Traversal in Language and Latent Spaces with Dynamic Shortcuts  
**Authors:** Xiaoqiang Wang et al.  
**arXiv:** 2505.18962 | **Venue:** NeurIPS 2025

##### Core Innovation
System-1.5 bridges **fast intuitive reasoning (System-1)** and **slow deliberate reasoning (System-2)** using dynamic shortcuts in latent space.

##### Architecture Design
1. **Model Depth Shortcut (DS):**
   - Non-critical tokens exit early via lightweight adapter branches
   - Critical tokens proceed through deeper layers
   - Adaptive depth allocation

2. **Step Shortcut (SS):**
   - Reuses hidden states across decoding steps
   - Skips trivial reasoning steps
   - Horizontal reasoning in latent space

##### Training Methodology
1. **Stage 1:** Distill natural language CoT into latent-space continuous thought
2. **Stage 2:** Distill full-path System-2 reasoning into adaptive shortcut paths

##### Experimental Results
| Metric          | Performance                   |
|-----------------|-------------------------------|
| GSM8K Accuracy  | Comparable to traditional CoT |
| Inference Speed | 20x faster                    |
| Token Reduction | 92.31%                        |

---

#### 2.2.3 LatentEvolve: Self-Evolving Test-Time Scaling

**Paper:** LatentEvolve: Self-Evolving Test-Time Scaling in Latent Space  
**Authors:** Guibin Zhang et al.  
**arXiv:** 2509.24771

##### Core Innovation
LatentEvolve mimics **human cognitive processes** through complementary learning system theory with daytime and nighttime scaling.

##### Architecture Design
1. **Daytime Scaling:** Quick retrieval of historical latent representations for immediate reasoning
2. **Nighttime Scaling:** Consolidates past optimizations (like memory consolidation during sleep)

##### Key Technical Contributions
1. **Dual Cognitive Systems:**
   - Fast retrieval for immediate reasoning needs
   - Slow consolidation for long-term learning
   - Alternates between fast and slow evolutionary processes

2. **Fully Unsupervised:**
   - No external supervision required
   - Self-evolving mechanism
   - Parameter-free adaptation

3. **Generalization:**
   - Strong performance across different domains
   - Works across model architectures
   - Up to 13.33% improvement over LatentSeek and TTRL

---

#### 2.2.4 FR-Ponder: Flexible Recurrent Pondering

**Paper:** Learning to Ponder: Adaptive Reasoning in Latent Space  
**Authors:** Yixin He, Lumingyuan Tang  
**arXiv:** 2509.24238

##### Core Innovation
FR-Ponder implements **instance-adaptive reasoning** through latent steering, using a sub-1M parameter controller to dynamically adjust reasoning depth.

##### Architecture Design
1. **Lightweight Controller:** <1M parameters, observes hidden states
2. **Latent Steering Vectors:** Pre-computed vectors guide reasoning depth
3. **Dynamic Halting:** Decides whether to halt or perform additional ponder steps

##### Key Technical Contributions
1. **Compute-Accuracy Balance:**
   - Group Relative Policy Optimization (GRPO) as reward signal
   - Balances task accuracy with computational efficiency
   - Avoids over-computation on simple tasks

2. **No Backbone Modification:**
   - Works without changing model weights
   - Flexible deployment
   - Interpretable steering directions

3. **Curriculum Learning:**
   - Learns optimal compute allocation over time
   - Correlation between compute and problem difficulty

##### Experimental Results
- Improved accuracy on GSM8K and MATH500
- Better compute-accuracy trade-off than early-exit baselines
- Reduced unnecessary computation on simple queries

---

## Comparative Analysis & Discussion

### Decoder-Based Models (Layer-wise Vertical) Comparison

| Method       | Mechanism                | Training          | Test-Time Scaling | Key Innovation                        |
|--------------|--------------------------|-------------------|-------------------|---------------------------------------|
| CoTFormer    | Token-level iterations   | Pretraining       | Yes (depth)       | Budget-adaptive compute               |
| Huginn       | Recurrent block          | Pretraining       | Yes (depth)       | Depth recurrence without extra tokens |
| LTO          | Latent reward model      | Post-training     | Yes               | Reward signals in latent thoughts     |
| ITT          | Adaptive token routing   | Pretraining       | Yes               | 96.5% of 3x model performance         |
| Pondering LM | Pondering step           | Pretraining       | No                | Weighted embedding processing         |
| Zhu et al.   | Continuous superposition | N/A (theoretical) | N/A               | Polynomial speedup proof              |
| MoR          | Mixture of recursions    | Pretraining       | Yes               | Token-level adaptive depth            |

### Inference Exploration (Token-wise Horizontal) Comparison

#### Parallel Scaling Methods
| Method      | Approach                    | Training     | Key Feature             |
|-------------|-----------------------------|--------------|-------------------------|
| SoftCoT++   | Multiple initial tokens     | Fine-tuning  | Contrastive diversity   |
| PCCoT       | Jacobi iteration            | Fine-tuning  | 50% time reduction      |
| Butt et al. | RL with soft tokens         | RL           | Continuous exploration  |
| KaVa        | KV-cache distillation       | Distillation | Compressed reasoning    |
| LTA-Thinker | Multi-objective training    | Fine-tuning  | Overthinking reduction  |
| Pythia Arch | Latent thoughts pretraining | Pretraining  | 1.4B > 2.8B performance |

#### Sequential Scaling Methods
| Method       | Approach               | Training     | Key Feature                      |
|--------------|------------------------|--------------|----------------------------------|
| LatentSeek   | Policy gradient        | None         | Test-time adaptation             |
| System-1.5   | Dynamic shortcuts      | Distillation | 20x speedup, 92% token reduction |
| LatentEvolve | Dual cognitive systems | None         | Self-evolving, 13% improvement   |
| FR-Ponder    | Latent steering        | RL           | Adaptive reasoning depth         |

### Key Insights for RAM Project

#### 1. Coarse-to-Fine via Latent Space
All approaches suggest reasoning at multiple abstraction levels:
- **Vertical (Layer-wise):** Early layers = coarse, deep layers = refined
- **Horizontal (Token-wise):** Multiple iterations/paths from coarse to fine
- **Superposition:** Parallel exploration of multiple hypotheses

#### 2. Training vs Inference Trade-offs
- **Pretraining methods (Huginn, MoR, Pythia):** Most powerful but require from-scratch training
- **Post-training methods (LTO, SoftCoT++):** Adapts existing models
- **Inference-only methods (LatentSeek, FR-Ponder):** Most flexible, no training

#### 3. Efficiency Considerations
- **Token overhead:** Latent methods avoid generating reasoning tokens
- **Compute allocation:** Adaptive methods focus resources on difficult parts
- **Speed vs Quality:** System-1.5 achieves 20x speedup with maintained accuracy

#### 4. Theoretical Foundations
- **Zhu et al.:** Proves polynomial advantage of continuous over discrete CoT
- **LTO:** Demonstrates reward signals exist in latent space
- **MoR:** Shows token-level adaptive depth is beneficial

---

## Part 3: Efficiency-Focused Decoding Strategies (From: A Survey on Parallel Reasoning)

### Survey Reference
**A Survey on Parallel Reasoning**
- arXiv: 2510.12164
- Authors: Ziqi Wang, Boye Niu, Zipeng Gao, et al.
- GitHub: https://github.com/PPPP-kaqiu/Awesome-Parallel-Reasoning

This section covers efficiency-focused decoding strategies that accelerate LLM inference through parallel token generation, speculative execution, and early exit mechanisms.

---

### 3.1 Speculative Decoding (Foundational Work)

**Paper:** Fast Inference from Transformers via Speculative Decoding  
**Authors:** Yaniv Leviathan, Matan Kalman, Yossi Matias (Google Research)  
**arXiv:** 2211.17192 | **Venue:** ICML 2023 Oral

#### Core Innovation
Speculative decoding is the foundational algorithm that enables **sampling from autoregressive models faster without changing outputs**. It computes several tokens in parallel by leveraging speculative execution.

#### Algorithm Design
```
1. Draft Model generates K candidate tokens speculatively
2. Target Model verifies all K tokens in parallel (single forward pass)
3. Accept tokens until first rejection, sample correction token
4. Repeat with accepted prefix
```

#### Key Technical Contributions
1. **Speculative Execution for LLMs:**
   - Hard language tasks contain easier subtasks approximated by efficient models
   - Draft model proposes, target model verifies in parallel
   - **Exact decoding:** Output distribution identical to target model

2. **No Retraining Required:**
   - Works with off-the-shelf models
   - No architecture changes needed
   - Plug-and-play acceleration

3. **Novel Sampling Method:**
   - Rejection sampling with modified acceptance criterion
   - Guarantees identical output distribution
   - Enables concurrent multi-token generation

#### Experimental Results
- **2X-3X speedup** on T5-XXL vs standard implementation
- Identical outputs (lossless acceleration)
- Memory-bandwidth bound becomes compute bound

#### Implications for RAM Project
- **Parallel verification:** Multiple reasoning steps verified simultaneously
- **Draft-verify paradigm:** Applicable to coarse-to-fine reasoning
- **Exact outputs:** No quality degradation from acceleration

---

### 3.2 Medusa: Multiple Decoding Heads

**Paper:** Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads  
**Authors:** Tianle Cai, Yuhong Li, Zhengyang Geng, et al. (Princeton, CMU, UIUC)  
**arXiv:** 2401.10774  
**Code:** https://github.com/FasterDecoding/Medusa

#### Core Innovation
Medusa augments LLMs with **extra decoding heads** that predict multiple subsequent tokens in parallel, eliminating the need for a separate draft model.

#### Architecture Design
```
Original LLM → [Backbone] → [LM Head] → Next Token
                    ↓
              [Medusa Head 1] → Token +1
              [Medusa Head 2] → Token +2
              [Medusa Head K] → Token +K
```

- **Multiple Heads:** K additional heads predict future tokens
- **Tree Attention:** Constructs multiple candidate continuations
- **Parallel Verification:** All candidates verified simultaneously

#### Key Technical Contributions
1. **No Draft Model Required:**
   - Single model with multiple heads
   - Eliminates draft model maintenance overhead
   - Tighter integration than speculative decoding

2. **Two Fine-tuning Options:**
   - **Medusa-1:** Heads fine-tuned on frozen backbone (lossless)
   - **Medusa-2:** Joint fine-tuning for better accuracy/speed trade-off

3. **Tree-Based Attention:**
   - Multiple candidate paths explored simultaneously
   - Efficient verification through tree structure
   - Typical acceptance scheme boosts acceptance rate

4. **Self-Distillation:**
   - Handles scenarios without training data
   - Model generates own training signal

#### Experimental Results
| Mode     | Speedup  | Quality       |
|----------|----------|---------------|
| Medusa-1 | 2.2x+    | Lossless      |
| Medusa-2 | 2.3-3.6x | Near-lossless |

#### Implications for RAM Project
- **Multi-head architecture:** Parallel prediction of reasoning steps
- **Tree exploration:** Natural fit for exploring multiple reasoning paths
- **Integrated design:** Simpler than separate draft model

---

### 3.3 EAGLE-3: Scaling Up Inference Acceleration

**Paper:** EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test  
**Authors:** Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang  
**arXiv:** 2503.01840

#### Core Innovation
EAGLE-3 abandons feature prediction in favor of **direct token prediction** and replaces reliance on top-layer features with **multi-layer feature fusion** via training-time test.

#### Evolution of EAGLE Series
```
EAGLE-1: Feature-level autoregression
    ↓
EAGLE-2: Dynamic draft trees
    ↓
EAGLE-3: Direct token prediction + multi-layer fusion
```

#### Key Technical Contributions
1. **Direct Token Prediction:**
   - EAGLE-1/2 predicted features, EAGLE-3 predicts tokens directly
   - Better scalability with more training data
   - Overcomes feature prediction constraints

2. **Multi-Layer Feature Fusion:**
   - Uses features from multiple layers, not just top layer
   - Training-time test technique for layer selection
   - Better representation for draft model

3. **Data Scaling Benefits:**
   - EAGLE-1/2 saw limited gains from more data
   - EAGLE-3 fully benefits from scaled training data
   - Better performance with increased compute

#### Experimental Results
| Model Type        | Speedup          | vs EAGLE-2   |
|-------------------|------------------|--------------|
| Chat              | Up to 6.5x       | ~1.4x faster |
| Reasoning         | Significant      | ~1.4x faster |
| SGLang (batch=64) | 1.38x throughput | -            |

#### Implications for RAM Project
- **Scalability:** Benefits from increased training data/compute
- **Multi-layer features:** Rich representations for reasoning
- **Direct prediction:** Simplifies draft model design

---

### 3.4 Lookahead Decoding: No Draft Model Required

**Paper:** Break the Sequential Dependency of LLM Inference Using Lookahead Decoding  
**Authors:** Yichao Fu, Peter Bailis, Ion Stoica, Hao Zhang (UC Berkeley, Hao AI Lab)  
**arXiv:** 2402.02057  
**Code:** https://github.com/hao-ai-lab/LookaheadDecoding

#### Core Innovation
Lookahead decoding accelerates LLM inference **without any auxiliary models** by using Jacobi iteration to generate multiple tokens in parallel.

#### Algorithm Design
```
Jacobi Iteration for LLMs:
1. Initialize future token positions with guesses
2. Run forward pass (all positions in parallel)
3. Update each position based on LLM output
4. Repeat until convergence (fixed point)
5. Accept converged n-grams
```

#### Key Technical Contributions
1. **No Draft Model:**
   - Single model generates all tokens
   - No need to train/maintain separate draft model
   - Generalizes across domains

2. **Jacobi Iteration Formulation:**
   - LLM decoding as fixed-point iteration
   - Multiple tokens computed per forward pass
   - N-grams verified through iteration convergence

3. **Hardware Efficiency:**
   - Trades compute for reduced decoding steps
   - Compatible with FlashAttention
   - Strong scaling on multiple GPUs

#### Experimental Results
- **1.8x speedup** on MT-bench
- **4x speedup** with strong scaling (multiple GPUs, code completion)
- No quality degradation

#### Implications for RAM Project
- **Self-contained:** No external models needed
- **Parallel iteration:** Natural for iterative reasoning refinement
- **Scalable:** Benefits from parallel hardware

---

### 3.5 Jacobi Forcing: AR to Parallel Decoder

**Paper:** Fast and Accurate Causal Parallel Decoding using Jacobi Forcing  
**Authors:** Lanxiang Hu, Siqi Kou, Yichao Fu, et al. (UCSD, DeepSeek, UC Berkeley)  
**arXiv:** 2512.14681  
**Code:** https://github.com/hao-ai-lab/Jacobi-Forcing

#### Core Innovation
Jacobi Forcing is a **progressive distillation paradigm** that trains AR models on their own parallel decoding trajectories, smoothly transitioning them into efficient parallel decoders.

#### Problem Addressed
- Diffusion LLMs (dLLMs) enable parallel decoding but suffer from:
  - Pretrain-to-posttrain distribution mismatch
  - Bidirectional attention conflicts with causal prior
  - Cannot reuse exact KV cache

#### Key Technical Contributions
1. **Jacobi Forcing Training:**
   - Train model on its own Jacobi decoding trajectories
   - Progressive shift from AR to parallel decoding
   - Preserves causal inference property

2. **Multi-Block Decoding:**
   - Decode multiple blocks simultaneously
   - Rejection recycling for higher acceptance
   - Up to 4.5x tokens per forward pass

3. **Causal Preservation:**
   - Maintains KV cache reuse
   - Compatible with existing AR infrastructure
   - No architectural changes to base model

#### Experimental Results
| Metric                   | Improvement        |
|--------------------------|--------------------|
| Wall-clock speedup       | 3.8x (coding/math) |
| Tokens per iteration     | 4.5x higher        |
| With rejection recycling | ~4.0x speedup      |

#### Implications for RAM Project
- **Smooth transition:** Converts existing AR models to parallel decoders
- **Trajectory training:** Model learns from its own reasoning trajectories
- **Efficiency:** Significant speedup with minimal quality loss

---

### 3.6 Online Speculative Decoding: Adaptive Draft Models

**Paper:** Online Speculative Decoding  
**Authors:** Xiaoxuan Liu, Lanxiang Hu, Peter Bailis, et al. (UC Berkeley)  
**arXiv:** 2310.07177

#### Core Innovation
Online speculative decoding **continuously updates the draft model** on observed user query data, adapting to the actual query distribution.

#### Problem Addressed
- Standard speculative decoding suffers when:
  - Draft model trained on different distribution than queries
  - Large capability gap between draft and target models
  - Diverse input types reduce acceptance rate

#### Key Technical Contributions
1. **Continuous Draft Model Update:**
   - Online learning from user queries
   - Knowledge distillation from target model corrections
   - Adapts to actual query distribution

2. **Distribution Alignment:**
   - Bridges training-query distribution gap
   - Improves draft model predictive accuracy
   - Maintains compact draft model size

3. **Knowledge Distillation:**
   - Draft model learns from target model's corrections
   - Efficient online update mechanism
   - No full retraining required

#### Experimental Results
| Metric                | Improvement    |
|-----------------------|----------------|
| Token acceptance rate | +0.1 to +0.65  |
| Latency reduction     | 1.42x to 2.17x |

#### Implications for RAM Project
- **Adaptive reasoning:** Draft models can adapt to specific reasoning patterns
- **Online learning:** Continuous improvement during deployment
- **Distribution matching:** Better alignment with actual use cases

---

### 3.7 Parallel-Probe: Efficient Parallel Thinking via 2D Probing

**Paper:** Parallel-Probe: Towards Efficient Parallel Thinking via 2D Probing  
**Authors:** Tong Zheng, Chengsong Huang, Runpeng Dai, et al. (UMD, UNC)  
**arXiv:** 2602.03845

#### Core Innovation
Parallel-Probe introduces **2D probing** to expose width-depth dynamics of parallel thinking, enabling efficient optimization of parallel reasoning through consensus-based early stopping and branch pruning.

#### Key Insights from 2D Probing
1. **Non-monotonic scaling** across width-depth allocations
2. **Heterogeneous reasoning branch lengths**
3. **Early stabilization of global consensus**

#### Key Technical Contributions
1. **2D Probing Interface:**
   - Periodically elicits intermediate answers from all branches
   - Exposes width-depth dynamics
   - Training-free approach

2. **Consensus-Based Early Stopping:**
   - Monitors global consensus across branches
   - Stops when consensus stabilizes
   - Reduces unnecessary computation

3. **Deviation-Based Branch Pruning:**
   - Dynamically adjusts parallel width
   - Prunes branches deviating from consensus
   - Focuses resources on promising paths

#### Experimental Results
| Metric            | Reduction                        |
|-------------------|----------------------------------|
| Sequential tokens | Up to 35.8%                      |
| Total token cost  | Over 25.8%                       |
| Accuracy          | Competitive with majority voting |

#### Implications for RAM Project
- **Width-depth trade-off:** Understanding optimal parallel reasoning allocation
- **Early stopping:** Avoid overthinking when consensus reached
- **Dynamic pruning:** Focus compute on promising reasoning paths

---

## Efficiency Methods Comparison

| Method               | Approach            | Draft Model | Speedup        | Key Feature     |
|----------------------|---------------------|-------------|----------------|-----------------|
| Speculative Decoding | Draft-verify        | Required    | 2-3x           | Exact outputs   |
| Medusa               | Multi-head          | No (heads)  | 2.2-3.6x       | Tree attention  |
| EAGLE-3              | Feature fusion      | Required    | Up to 6.5x     | Data scaling    |
| Lookahead            | Jacobi iteration    | No          | 1.8-4x         | Self-contained  |
| Jacobi Forcing       | Trajectory training | No          | 3.8-4x         | AR compatible   |
| Online Spec. Dec.    | Adaptive draft      | Required    | 1.4-2.2x       | Online learning |
| Parallel-Probe       | 2D probing          | No          | 25-35% savings | Early stopping  |

---

## References

### Decoder-Based Models (Layer-wise Vertical)
1. Mohtashami et al. (2023). CoTFormer: A Chain-of-Thought Driven Architecture. arXiv:2310.10845
2. Geiping et al. (2025). Scaling up Test-Time Compute with Latent Reasoning (Huginn). arXiv:2502.05171
3. Du et al. (2025). Latent Thinking Optimization (LTO). arXiv:2509.26314
4. Chen et al. (2025). Inner Thinking Transformer (ITT). arXiv:2502.13842
5. Zeng et al. (2025). Pretraining Language Models to Ponder in Continuous Space. arXiv:2505.20674
6. Zhu et al. (2025). Reasoning by Superposition: A Theoretical Perspective. arXiv:2505.12514
7. Bae et al. (2025). Mixture-of-Recursions (MoR). arXiv:2507.10524

### Inference Exploration - Parallel Scaling
8. Xu et al. (2025). SoftCoT++: Test-Time Scaling with Soft Chain-of-Thought. arXiv:2505.11484
9. Wu et al. (2025). PCCoT: Parallel Continuous Chain-of-Thought with Jacobi Iteration. arXiv:2506.18582
10. Butt et al. (2025). Soft Tokens, Hard Truths. arXiv:2509.19170
11. Kuzina et al. (2025). KaVa: Latent Reasoning via Compressed KV-Cache Distillation. arXiv:2510.02312
12. Wang et al. (2025). LTA-thinker: Latent Thought-Augmented Training Framework. arXiv:2509.12875
13. Zeng et al. (2025). PonderLM-2: Pretraining LLM with Latent Thoughts. arXiv:2509.23184

### Inference Exploration - Sequential Scaling
14. Li et al. (2025). LatentSeek: Test-Time Policy Gradient in Latent Space. arXiv:2505.13308
15. Wang et al. (2025). System-1.5 Reasoning: Dynamic Shortcuts. arXiv:2505.18962
16. Zhang et al. (2025). LatentEvolve: Self-Evolving Test-Time Scaling. arXiv:2509.24771
17. He & Tang (2025). FR-Ponder: Learning to Ponder Adaptive Reasoning. arXiv:2509.24238

### Survey
18. Chen et al. (2025). Reasoning Beyond Language: A Comprehensive Survey on Latent Chain-of-Thought Reasoning. arXiv:2505.16782
19. Wang et al. (2025). A Survey on Parallel Reasoning. arXiv:2510.12164

### Efficiency-Focused Decoding
20. Leviathan et al. (2022). Fast Inference from Transformers via Speculative Decoding. arXiv:2211.17192
21. Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework. arXiv:2401.10774
22. Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration. arXiv:2503.01840
23. Fu et al. (2024). Lookahead Decoding: Break Sequential Dependency. arXiv:2402.02057
24. Hu et al. (2025). Jacobi Forcing: Fast and Accurate Causal Parallel Decoding. arXiv:2512.14681
25. Liu et al. (2023). Online Speculative Decoding. arXiv:2310.07177
26. Zheng et al. (2026). Parallel-Probe: Efficient Parallel Thinking via 2D Probing. arXiv:2602.03845

---

## Downloaded PDFs Location

```
docs/references/
├── decoder-based-models/  (Original - keeping for reference)
│   ├── 2023-CoTFormer-Chain-of-Thought-Architecture.pdf
│   ├── 2024-Coconut-Training-LLMs-Reason-Continuous-Latent-Space.pdf
│   └── 2025-Huginn-Scaling-Test-Time-Compute-Latent-Reasoning.pdf
├── decoder-based-models-corrected/  (Corrected papers per survey)
│   ├── 2025-ITT-Inner-Thinking-Transformer.pdf
│   ├── 2025-LTO-Latent-Thinking-Optimization.pdf
│   ├── 2025-MoR-Mixture-of-Recursions.pdf
│   ├── 2025-Pondering-LM-Continuous-Space.pdf
│   └── 2025-Zhu-Reasoning-Superposition.pdf
├── inference-exploration/  (Original - keeping for reference)
│   ├── 2024-Quiet-STaR-Think-Before-Speaking.pdf
│   ├── 2025-AB-MCTS-Adaptive-Branching-Tree-Search.pdf
│   └── 2025-LatentSeek-Test-Time-Policy-Gradient-Latent-Space.pdf
├── inference-exploration-corrected/  (Corrected papers per survey)
│   ├── 2025-Butt-Soft-Tokens-Hard-Truths.pdf
│   ├── 2025-FR-Ponder-Adaptive-Latent.pdf
│   ├── 2025-KaVa-Latent-Reasoning-KV-Cache.pdf
│   ├── 2025-LatentEvolve-Self-Evolving.pdf
│   ├── 2025-LTA-Thinker-Latent-Thought-Training.pdf
│   ├── 2025-PCCoT-Parallel-Continuous-Jacobi.pdf
│   ├── 2025-PonderLM2-Pythia-Arch.pdf
│   ├── 2025-SoftCoT++-Test-Time-Scaling.pdf
│   └── 2025-System-1.5-Reasoning.pdf
└── efficiency/
    ├── 2022-Speculative-Decoding-Leviathan.pdf
    ├── 2023-Online-Speculative-Decoding.pdf
    ├── 2024-Lookahead-Decoding-Jacobi.pdf
    ├── 2024-Medusa-Multiple-Decoding-Heads.pdf
    ├── 2025-EAGLE3-Inference-Acceleration.pdf
    ├── 2025-Jacobi-Forcing-Parallel-Decoding.pdf
    └── 2026-Parallel-Probe-2D-Probing.pdf
```
