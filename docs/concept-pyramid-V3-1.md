# Concept Pyramid V3-1: Next-Segment Hierarchical Concept Generation

> **Paradigm**: From "Next-Level" to "Next-Segment" — Sequential over text position, parallel over concept granularity

---

## 1. Motivation & Core Insight

### 1.1 Limitation of V3's Next-Level Generation

V3's original design follows VAR's "Next-Scale" pattern:

```
V3 Next-Level Generation:
───────────────────────────────────────────────────────────────────────────
Step 0: Generate C_0 [1 concept]     ← Global understanding
Step 1: Generate C_1 [2 concepts]    ← Coarse reasoning
Step 2: Generate C_2 [4 concepts]    ← Medium reasoning
Step 3: Generate C_3 [8 concepts]    ← Fine reasoning
...

Problem: Concepts at different levels correspond to different text positions!
  C_0: "Overall problem structure"
  C_1: ["Given info", "What to find"]
  C_2: ["Distance=120", "Time=2h", "Formula", "Compute"]
  
  These concepts are NOT aligned by position!
```

**Key Issue**: Generating all Level 2 concepts after Level 1 breaks text coherence.

### 1.2 V3-1 Innovation: Next-Segment Generation

**Core Idea**: Generate concepts in text reading order, each segment contains multi-granularity concepts.

```
CoT Segmentation:
───────────────────────────────────────────────────────────────────────────
CoT: "Let me think. A train travels 120 km in 2 hours. Speed = Distance / Time.
      So speed = 120 / 2 = 60 km/h."

Segmentation:
  Seg_0: "Let me think."                          ← Initial thought
  Seg_1: "A train travels 120 km in 2 hours."      ← Given information
  Seg_2: "Speed = Distance / Time."                ← Formula identification
  Seg_3: "So speed = 120 / 2 = 60 km/h."           ← Computation & answer

V3-1 Generation Order:
───────────────────────────────────────────────────────────────────────────
Time 0: Generate concepts for Seg_0
  [C_0^0, C_1^0, C_2^0, C_3^0]  ← Multi-granularity for Seg_0
  
Time 1: Generate concepts for Seg_1
  [       C_1^1, C_2^1, C_3^1]  ← Multi-granularity for Seg_1
  
Time 2: Generate concepts for Seg_2
  [              C_2^2, C_3^2]  ← Multi-granularity for Seg_2
  
Time 3: Generate concepts for Seg_3
  [                     C_3^3]  ← Multi-granularity for Seg_3

Key Insight:
  - Sequential: Follows text reading order (Seg_0 → Seg_1 → Seg_2 → Seg_3)
  - Parallel: Within each segment, multiple granularity concepts generated together
```

### 1.3 Why Next-Segment?

```
═══════════════════════════════════════════════════════════════════════════
                    Next-Segment vs Next-Level
═══════════════════════════════════════════════════════════════════════════

Next-Level (V3 Original):
───────────────────────────────────────────────────────────────────────────
  Generate all coarse concepts first, then medium, then fine.
  
  Problem: Coherence break!
    After generating C_2 (fine for beginning), we jump to C_0 for end.
    
  Analogy: Writing an essay by writing all topic sentences first,
           then all supporting sentences, then all details.
           → Hard to maintain coherence!

Next-Segment (V3-1):
───────────────────────────────────────────────────────────────────────────
  Generate all concepts for Seg_0, then Seg_1, then Seg_2, etc.
  
  Advantage: Natural coherence!
    Each segment's concepts are generated together, maintaining local structure.
    
  Analogy: Writing an essay paragraph by paragraph.
           Each paragraph: topic + supporting + details together.
           → Natural flow!

═══════════════════════════════════════════════════════════════════════════
```

---

## 2. Architecture

### 2.1 Complete Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│              V3-1: Next-Segment Concept Generation Architecture          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  TRAINING PHASE (Q + CoT + Solution):                                   │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                          │
│   Step 1: Segment CoT                                                   │
│   ─────────────────────────────────────────────────────────────────    │
│   CoT → [Seg_0, Seg_1, ..., Seg_{T-1}]  (T segments)                   │
│                                                                          │
│   Step 2: Extract Segment-Level Concepts                                │
│   ─────────────────────────────────────────────────────────────────    │
│   For each segment t:                                                   │
│     H_t = Encoder(Q + Seg_≤t)  [B, L_t, D]                             │
│     C_t = AttentivePooling(H_t) = [C_0^t, C_1^t, ..., C_{K-t}^t]       │
│     (Concepts for segment t, granularity depends on position)          │
│                                                                          │
│   Step 3: Concatenate All Segment Concepts                              │
│   ─────────────────────────────────────────────────────────────────    │
│   AllConcepts = [C_0^0, C_1^0, C_2^0, C_3^0,  ← Seg_0 concepts        │
│                  C_1^1, C_2^1, C_3^1,          ← Seg_1 concepts        │
│                  C_2^2, C_3^2,                 ← Seg_2 concepts        │
│                  C_3^3]                        ← Seg_3 concepts        │
│                                                                          │
│   Step 4: Concept Transformer                                           │
│   ─────────────────────────────────────────────────────────────────    │
│   RefinedConcepts = ConceptTransformer(AllConcepts)                    │
│   (With segment-level causal mask)                                      │
│                                                                          │
│   Step 5: Decode to Solution                                            │
│   ─────────────────────────────────────────────────────────────────    │
│   Solution = TokenDecoder(RefinedConcepts)                             │
│                                                                          │
│  INFERENCE PHASE (Q only):                                              │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                          │
│   Step 1: Initialize                                                    │
│   ─────────────────────────────────────────────────────────────────    │
│   H = Encoder(Q)  [B, L, D]                                            │
│   GeneratedSegments = []                                                │
│                                                                          │
│   Step 2: Next-Segment Generation Loop                                  │
│   ─────────────────────────────────────────────────────────────────    │
│   for t = 0, 1, 2, ... until end:                                      │
│                                                                          │
│     # Generate concepts for segment t                                   │
│     C_t = SegmentConceptGenerator(H, GeneratedSegments)                │
│       = [C_0^t, C_1^t, ..., C_{K-t}^t]  (parallel generation)          │
│                                                                          │
│     # Parallel generation within segment                                │
│     C_0^t = QueryLevel0(H, GeneratedSegments)                          │
│     C_1^t = QueryLevel1(H, GeneratedSegments)  │ Parallel!             │
│     C_2^t = QueryLevel2(H, GeneratedSegments)  │ All concepts for      │
│     ...                                        │ this segment          │
│                                                                          │
│     GeneratedSegments.append(C_t)                                       │
│                                                                          │
│     if EndTokenPredicted(C_t):                                          │
│       break                                                             │
│                                                                          │
│   Step 3: Concept Transformer                                           │
│   ─────────────────────────────────────────────────────────────────    │
│   RefinedConcepts = ConceptTransformer(GeneratedSegments)              │
│                                                                          │
│   Step 4: Decode to Solution                                            │
│   ─────────────────────────────────────────────────────────────────    │
│   Solution = TokenDecoder(RefinedConcepts)                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Key Component: Segment Concept Generator

```
═══════════════════════════════════════════════════════════════════════════
                    Segment Concept Generator
═══════════════════════════════════════════════════════════════════════════

Purpose: Generate multi-granularity concepts for current segment

Input:
  - H: Encoder output for Q [B, L, D]
  - GeneratedSegments: List of previously generated segment concepts

Output:
  - C_t = [C_0^t, C_1^t, ..., C_{K-t}^t]  (concepts for segment t)

Architecture: Multi-Query Parallel Generation
───────────────────────────────────────────────────────────────────────────

# Shared context preparation
Context = Concat([
  H,                          # Question encoding
  Flatten(GeneratedSegments)  # Previous segment concepts
])  [B, L', D]

# Parallel generation for each granularity level
Queries:
  Q_0 = Linear(D → D)(Context)  # Query for level 0 (coarsest)
  Q_1 = Linear(D → D)(Context)  # Query for level 1
  Q_2 = Linear(D → D)(Context)  # Query for level 2
  ...
  Q_{K-t} = Linear(D → D)(Context)  # Query for finest level

# Each query attends to context to extract concept
C_0^t = Attention(Q_0, Context, Context)  [B, 1, D]
C_1^t = Attention(Q_1, Context, Context)  [B, 2, D]
C_2^t = Attention(Q_2, Context, Context)  [B, 4, D]
...
C_{K-t}^t = Attention(Q_{K-t}, Context, Context)  [B, 2^{K-t}, D]

# All C_*^t generated in parallel!

═══════════════════════════════════════════════════════════════════════════
```

### 2.3 Concept Structure Visualization

```
═══════════════════════════════════════════════════════════════════════════
                    Concept Pyramid Structure (V3-1)
═══════════════════════════════════════════════════════════════════════════

Generated Order (Sequential by Segment):
───────────────────────────────────────────────────────────────────────────
Time 0 (Seg_0): [C_0^0, C_1^0, C_2^0, C_3^0]
                  ↓     ↓      ↓      ↓
                [1]   [2]    [4]    [8]  concepts
                
Time 1 (Seg_1): [       C_1^1, C_2^1, C_3^1]
                         ↓      ↓      ↓
                       [2]    [4]    [8]  concepts
                       
Time 2 (Seg_2): [              C_2^2, C_3^2]
                                ↓      ↓
                              [4]    [8]  concepts
                              
Time 3 (Seg_3): [                     C_3^3]
                                       ↓
                                     [8]  concepts

Pyramid Visualization:
───────────────────────────────────────────────────────────────────────────
                    C_0^0
                   /       \
              C_1^0         C_1^1
             /     \       /     \
          C_2^0   C_2^1  C_2^2
         /   |   /   |  /   |
       C_3^0 ... (concepts for each segment)

Key Properties:
  1. Earlier segments have more granularity levels
  2. Later segments have fewer levels (less context needed)
  3. Total concepts: 1+2+4+8 + 2+4+8 + 4+8 + 8 = 54 (for T=4, K=4)

═══════════════════════════════════════════════════════════════════════════
```

---

## 3. Training

### 3.1 Training Overview

```
═══════════════════════════════════════════════════════════════════════════
                    V3-1 Training: Three-Stage Strategy
═══════════════════════════════════════════════════════════════════════════

Stage 1: CoT Segmentation & Concept Extraction
───────────────────────────────────────────────────────────────────────────
Goal: Learn to segment CoT and extract hierarchical concepts per segment

Input: Q + CoT + Solution

Step 1: Segment CoT
  CoT → [Seg_0, Seg_1, ..., Seg_{T-1}]
  
  Segmentation strategies:
    a) Fixed length: Each segment = N tokens
    b) Semantic: Split by sentence/clause boundaries
    c) Learned: Model learns optimal segmentation

Step 2: Extract Segment Concepts
  For t in [0, 1, ..., T-1]:
    H_t = Encoder(Q + Seg_≤t)
    C_t = AttentivePooling(H_t)  # Extract [C_0^t, ..., C_{K-t}^t]
    
Step 3: Store as training targets
  TargetConcepts = [C_0, C_1, ..., C_{T-1}]

Stage 2: Segment Concept Generator Training
───────────────────────────────────────────────────────────────────────────
Goal: Train generator to produce segment concepts without seeing future CoT

For each segment t:
  # Teacher (uses full CoT up to segment t)
  H_t^teacher = Encoder(Q + Seg_≤t)
  C_t^teacher = AttentivePooling(H_t^teacher)
  
  # Student (uses only Q and generated previous segments)
  H_t^student = Encoder(Q)
  C_t^student = SegmentConceptGenerator(
    H_t^student,
    [C_0^student, ..., C_{t-1}^student]
  )
  
  Loss: MSE(C_t^student, C_t^teacher)

Stage 3: End-to-End Solution Decoding
───────────────────────────────────────────────────────────────────────────
Goal: Learn to decode segment concepts to solution

Input: Q + CoT + Solution

Forward:
  Concepts = ExtractSegmentConcepts(Q + CoT)  # Stage 1 method
  RefinedConcepts = ConceptTransformer(Concepts)
  Solution_pred = TokenDecoder(RefinedConcepts)
  
Loss: CrossEntropy(Solution_pred, Solution_gt)

═══════════════════════════════════════════════════════════════════════════
```

### 3.2 Stage 1: CoT Segmentation

```
═══════════════════════════════════════════════════════════════════════════
                    Stage 1: CoT Segmentation Strategies
═══════════════════════════════════════════════════════════════════════════

Strategy A: Fixed-Length Segmentation
───────────────────────────────────────────────────────────────────────────
  Segment size = S tokens (e.g., S = 10)
  
  CoT: "Let me think. A train travels 120 km in 2 hours..."
  
  Seg_0: "Let me think. A train"  (10 tokens)
  Seg_1: "travels 120 km in 2"    (10 tokens)
  Seg_2: "hours. Speed = Distance" (10 tokens)
  ...
  
  Pros: Simple, predictable segment count
  Cons: May split semantic units

Strategy B: Semantic Segmentation (Recommended)
───────────────────────────────────────────────────────────────────────────
  Split by sentence boundaries or logical clauses
  
  CoT: "Let me think. A train travels 120 km in 2 hours. 
        Speed = Distance / Time. So speed = 120 / 2 = 60 km/h."
  
  Seg_0: "Let me think."
  Seg_1: "A train travels 120 km in 2 hours."
  Seg_2: "Speed = Distance / Time."
  Seg_3: "So speed = 120 / 2 = 60 km/h."
  
  Pros: Preserves semantic coherence
  Cons: Variable segment lengths

Strategy C: Learned Segmentation
───────────────────────────────────────────────────────────────────────────
  Model learns where to segment based on content
  
  Additional prediction head:
    P(end_of_segment | current_state)
    
  Training: Minimize segmentation loss + concept extraction loss
  
  Pros: Optimal for concept extraction
  Cons: More complex training

═══════════════════════════════════════════════════════════════════════════
```

### 3.3 Stage 2: Segment Concept Generator

```
═══════════════════════════════════════════════════════════════════════════
                    Stage 2: Segment Concept Generator Training
═══════════════════════════════════════════════════════════════════════════

Architecture Detail:
───────────────────────────────────────────────────────────────────────────

class SegmentConceptGenerator(nn.Module):
    """
    Generate multi-granularity concepts for current segment.
    Parallel generation within segment.
    """
    
    def __init__(self, num_levels, hidden_dim):
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        
        # Separate query projection for each level
        self.query_proj = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_levels)
        ])
        
        # Concept extraction attention
        self.concept_attn = MultiHeadAttention(hidden_dim)
        
    def forward(self, H, previous_segments):
        """
        Args:
            H: Encoder output [B, L, D]
            previous_segments: List of concept tensors
        
        Returns:
            concepts: List [C_0, C_1, ..., C_K] for this segment
        """
        # Prepare context
        context = torch.cat([H] + previous_segments, dim=1)
        
        # Generate concepts for each level in parallel
        concepts = []
        for level in range(self.num_levels):
            # Level-specific query
            Q = self.query_proj[level](context)
            
            # Extract concept via attention
            C_level = self.concept_attn(Q, context, context)
            
            # Downsample to appropriate granularity
            num_concepts = 2 ** level
            C_level = self.downsample(C_level, num_concepts)
            
            concepts.append(C_level)
        
        return concepts  # [C_0, C_1, ..., C_K]

Training Objective:
───────────────────────────────────────────────────────────────────────────
  For segment t:
    L_recon = Σ_{level} MSE(C_level^generated, C_level^target)
    L_end = BCE(P(end | concepts), end_label)  # Optional: predict segment end
    
    L_total = L_recon + λ * L_end

═══════════════════════════════════════════════════════════════════════════
```

### 3.4 Stage 3: Concept Transformer with Segment Causality

```
═══════════════════════════════════════════════════════════════════════════
                    Stage 3: Concept Transformer
═══════════════════════════════════════════════════════════════════════════

Key Design: Segment-Level Causal Mask
───────────────────────────────────────────────────────────────────────────

Concept Sequence:
  [C_0^0, C_1^0, C_2^0, C_3^0, C_1^1, C_2^1, C_3^1, C_2^2, C_3^2, C_3^3]
   └─Seg_0─┘  └─Seg_0─┘  └─Seg_0─┘  └─Seg_1─┘  └─Seg_1─┘  └─Seg_2─┘  └─Seg_3─┘

Causal Mask Rule:
  Concept at (segment t, level l) can attend to:
    1. All concepts from segments < t (full attention)
    2. Concepts from segment t with level ≤ l (within-segment causal)

Mask Matrix Visualization:
───────────────────────────────────────────────────────────────────────────

Position:  C_0^0  C_1^0  C_2^0  C_3^0  C_1^1  C_2^1  C_3^1  C_2^2  C_3^2  C_3^3
          ┌─────────────────────────────────────────────────────────────────┐
C_0^0     │  1     0      0      0      0      0      0      0      0      0 │
C_1^0     │  1     1      0      0      0      0      0      0      0      0 │
C_2^0     │  1     1      1      0      0      0      0      0      0      0 │
C_3^0     │  1     1      1      1      0      0      0      0      0      0 │
C_1^1     │  1     1      1      1      1      0      0      0      0      0 │  ← Can see all Seg_0
C_2^1     │  1     1      1      1      1      1      0      0      0      0 │
C_3^1     │  1     1      1      1      1      1      1      0      0      0 │
C_2^2     │  1     1      1      1      1      1      1      1      0      0 │  ← Can see Seg_0, Seg_1
C_3^2     │  1     1      1      1      1      1      1      1      1      0 │
C_3^3     │  1     1      1      1      1      1      1      1      1      1 │  ← Can see all previous
          └─────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════
```

---

## 4. Inference

### 4.1 Inference Algorithm

```
═══════════════════════════════════════════════════════════════════════════
                    V3-1 Inference: Next-Segment Generation
═══════════════════════════════════════════════════════════════════════════

Input: Question Q
Output: Solution

Algorithm:
───────────────────────────────────────────────────────────────────────────

# Step 1: Encode question
H = Encoder(Q)  [B, L, D]

# Step 2: Initialize
generated_segments = []
segment_cache = {}  # KV cache for efficiency

# Step 3: Next-Segment Generation Loop
for t = 0, 1, 2, ...:
    
    # Generate concepts for segment t (PARALLEL within segment)
    C_t = segment_generator.generate(
        H=H,
        previous_segments=generated_segments,
        max_levels=K-t  # Fewer levels for later segments
    )
    
    # C_t = [C_0^t, C_1^t, ..., C_{K-t}^t]
    # All generated in parallel via multi-query attention
    
    generated_segments.append(C_t)
    
    # Check if generation should end
    end_prob = segment_generator.predict_end(C_t)
    if end_prob > threshold:
        break

# Step 4: Refine concepts
all_concepts = flatten(generated_segments)
refined_concepts = concept_transformer(all_concepts)

# Step 5: Decode to solution
solution = token_decoder(refined_concepts)

return solution

═══════════════════════════════════════════════════════════════════════════
```

### 4.2 Parallel Generation Within Segment

```
═══════════════════════════════════════════════════════════════════════════
                    Parallel Generation: Detailed View
═══════════════════════════════════════════════════════════════════════════

For segment t, generating [C_0^t, C_1^t, C_2^t, C_3^t]:

Input:
  H: [B, L, D]  (question encoding)
  Prev: [B, L_prev, D]  (flattened previous segment concepts)
  Context = Concat(H, Prev): [B, L + L_prev, D]

Parallel Generation:
───────────────────────────────────────────────────────────────────────────

# Level 0 (1 concept)
Q_0 = W_q^0 @ Context  [B, L+L_prev, D]
C_0^t = Attention(Q_0, Context, Context)  [B, 1, D]
  ↓ Pool to 1 concept

# Level 1 (2 concepts)  ──┐
Q_1 = W_q^1 @ Context     │ Parallel!
C_1^t = Attention(Q_1, Context, Context)  [B, 2, D]
  ↓ Pool to 2 concepts    │
                          │
# Level 2 (4 concepts)  ──┤
Q_2 = W_q^2 @ Context     │ Parallel!
C_2^t = Attention(Q_2, Context, Context)  [B, 4, D]
  ↓ Pool to 4 concepts    │
                          │
# Level 3 (8 concepts)  ──┘
Q_3 = W_q^3 @ Context
C_3^t = Attention(Q_3, Context, Context)  [B, 8, D]
  ↓ Pool to 8 concepts

All four levels computed in parallel!
Only one forward pass through context.

═══════════════════════════════════════════════════════════════════════════
```

### 4.3 Inference Example

```
═══════════════════════════════════════════════════════════════════════════
                    Inference Example: Step by Step
═══════════════════════════════════════════════════════════════════════════

Question: "A train travels 120 km in 2 hours. What's its speed?"

Step 1: Encode
───────────────────────────────────────────────────────────────────────────
  H = Encoder("A train travels 120 km in 2 hours. What's its speed?")

Step 2: Generate Segments
───────────────────────────────────────────────────────────────────────────

Segment 0 (Initial Understanding):
───────────────────────────────────────────────────────────────────────────
  Parallel Generation:
    C_0^0 = ["Physics problem: find speed"]  [1 concept]
    C_1^0 = ["Given: motion info", "Goal: calculate"]  [2 concepts]
    C_2^0 = ["Object: train", "Quantity: 120km", "Quantity: 2h", "Target: speed"]
    C_3^0 = ["Read carefully", "Identify knowns", "Identify unknowns", ...]
  
  Check end: No (problem not solved)

Segment 1 (Information Extraction):
───────────────────────────────────────────────────────────────────────────
  Parallel Generation:
    C_1^1 = ["Distance=120km", "Time=2h"]  [2 concepts]
    C_2^1 = ["120 km distance", "2 hours time", "speed unknown", "formula needed"]
    C_3^1 = ["Numerical: 120", "Unit: km", "Numerical: 2", "Unit: hours", ...]
  
  (No C_0^1: global concept already captured)
  
  Check end: No

Segment 2 (Strategy & Formula):
───────────────────────────────────────────────────────────────────────────
  Parallel Generation:
    C_2^2 = ["Speed formula", "v = d / t", "Apply to problem"]
    C_3^2 = ["Recall definition", "Identify variables", "Substitute values", ...]
  
  Check end: No

Segment 3 (Computation & Answer):
───────────────────────────────────────────────────────────────────────────
  Parallel Generation:
    C_3^3 = ["Calculate 120/2", "Result: 60", "Unit: km/h", "Final answer"]
  
  Check end: Yes! (end token predicted)

Step 3: Refine & Decode
───────────────────────────────────────────────────────────────────────────
  AllConcepts = [C_0^0, C_1^0, C_2^0, C_3^0, C_1^1, C_2^1, C_3^1, C_2^2, C_3^2, C_3^3]
  Refined = ConceptTransformer(AllConcepts)
  Solution = TokenDecoder(Refined) = "60 km/h"

Output: "60 km/h"

═══════════════════════════════════════════════════════════════════════════
```

---

## 5. Comparison: V3 vs V3-1

```
═══════════════════════════════════════════════════════════════════════════
                    V3 (Next-Level) vs V3-1 (Next-Segment)
═══════════════════════════════════════════════════════════════════════════

Generation Order:
───────────────────────────────────────────────────────────────────────────
V3 (Next-Level):
  Step 0: C_0 = [C_0^0]  (1 concept)
  Step 1: C_1 = [C_1^0, C_1^1]  (2 concepts)
  Step 2: C_2 = [C_2^0, C_2^1, C_2^2, C_2^3]  (4 concepts)
  Step 3: C_3 = [8 concepts]
  
  Problem: Concepts not aligned with text position!

V3-1 (Next-Segment):
  Step 0 (Seg_0): [C_0^0, C_1^0, C_2^0, C_3^0]  (all for first text segment)
  Step 1 (Seg_1): [C_1^1, C_2^1, C_3^1]  (all for second text segment)
  Step 2 (Seg_2): [C_2^2, C_3^2]  (all for third text segment)
  Step 3 (Seg_3): [C_3^3]  (all for fourth text segment)
  
  Advantage: Concepts aligned with text reading order!

Parallelism:
───────────────────────────────────────────────────────────────────────────
V3:
  - Within-level: Concepts at same level can be parallel
  - Across-level: Sequential (must finish level k before k+1)
  
V3-1:
  - Within-segment: All granularity levels parallel
  - Across-segment: Sequential (must finish segment t before t+1)

Text Coherence:
───────────────────────────────────────────────────────────────────────────
V3:
  - May generate fine concepts for text end before coarse concepts for text start
  - Potential coherence issues
  
V3-1:
  - Follows natural text reading order
  - Each segment's concepts generated together
  - Better coherence

═══════════════════════════════════════════════════════════════════════════
```

---

## 6. Implementation Notes

### 6.1 File Structure

```
nlcpV3_1/
├── __init__.py
├── config.py                    # V3-1 configuration
├── encoder.py                   # Same as V2/V3
├── cot_segmenter.py            # NEW: Segment CoT into pieces
│   ├── FixedLengthSegmenter
│   ├── SemanticSegmenter
│   └── LearnedSegmenter
├── segment_concept_generator.py # NEW: Generate multi-granularity concepts per segment
├── concept_transformer.py       # Modified: Segment-level causal mask
├── token_decoder.py             # Same as V3 (decode to solution)
└── model.py                     # V3-1 model integrating all components
```

### 6.2 Key Implementation Details

```
═══════════════════════════════════════════════════════════════════════════
                    Key Implementation: Segment Concept Generator
═══════════════════════════════════════════════════════════════════════════

class SegmentConceptGenerator(nn.Module):
    """
    Generate multi-granularity concepts for a single segment.
    Parallel generation within segment via multi-query attention.
    """
    
    def __init__(self, config):
        self.max_levels = config.max_levels
        self.hidden_dim = config.hidden_dim
        
        # Level-specific query projections
        self.level_queries = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim)
            for _ in range(self.max_levels)
        ])
        
        # Shared attention mechanism
        self.attention = MultiHeadAttention(
            dim=self.hidden_dim,
            num_heads=config.num_heads
        )
        
        # Level-specific output projections
        self.level_outputs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim)
            )
            for _ in range(self.max_levels)
        ])
        
        # End-of-segment predictor
        self.end_predictor = nn.Linear(self.hidden_dim, 1)
    
    def forward(self, H, previous_segments, current_level_max):
        """
        Generate concepts for current segment.
        
        Args:
            H: [B, L, D] question encoding
            previous_segments: List of [B, *, D] tensors
            current_level_max: Maximum levels for this segment (decreases over time)
        
        Returns:
            concepts: List of [B, 2^level, D] tensors
            end_prob: [B, 1] probability of segment end
        """
        # Prepare context
        context_parts = [H] + list(previous_segments)
        context = torch.cat(context_parts, dim=1)  # [B, L_total, D]
        
        # Generate concepts for each level (PARALLEL)
        concepts = []
        for level in range(current_level_max):
            # Level-specific query
            Q = self.level_queries[level](context)  # [B, L_total, D]
            
            # Attention to context
            attn_out = self.attention(Q, context, context)  # [B, L_total, D]
            
            # Pool to appropriate number of concepts
            num_concepts = 2 ** level
            C_level = self._pool_to_concepts(attn_out, num_concepts)
            
            # Output projection
            C_level = self.level_outputs[level](C_level)
            
            concepts.append(C_level)
        
        # Predict end of segment
        end_feat = concepts[-1].mean(dim=1)  # [B, D]
        end_prob = torch.sigmoid(self.end_predictor(end_feat))  # [B, 1]
        
        return concepts, end_prob
    
    def _pool_to_concepts(self, x, num_concepts):
        """Pool sequence to specified number of concept tokens."""
        B, L, D = x.shape
        
        if num_concepts >= L:
            # Pad if needed
            padding = num_concepts - L
            x = F.pad(x, (0, 0, 0, padding))
            return x[:, :num_concepts, :]
        else:
            # Adaptive pooling
            x = x.permute(0, 2, 1)  # [B, D, L]
            x = F.adaptive_avg_pool1d(x, num_concepts)
            x = x.permute(0, 2, 1)  # [B, num_concepts, D]
            return x

═══════════════════════════════════════════════════════════════════════════
```

---

## 7. Summary

### 7.1 V3-1 Core Design

```
═══════════════════════════════════════════════════════════════════════════
                    V3-1: Next-Segment Generation
═══════════════════════════════════════════════════════════════════════════

Key Innovation:
  - Sequential over TEXT POSITION (segment by segment)
  - Parallel over CONCEPT GRANULARITY (multi-level within segment)

Generation Pattern:
  Time 0: [C_0^0, C_1^0, C_2^0, C_3^0]  ← Seg_0 (all granularities)
  Time 1: [C_1^1, C_2^1, C_3^1]          ← Seg_1 (all granularities)
  Time 2: [C_2^2, C_3^2]                 ← Seg_2 (all granularities)
  Time 3: [C_3^3]                        ← Seg_3 (all granularities)

Advantages:
  1. Natural text coherence (follows reading order)
  2. Parallel within segment (efficient)
  3. Multi-granularity per segment (rich representation)

═══════════════════════════════════════════════════════════════════════════
```

### 7.2 Comparison with V3

| Aspect             | V3 (Next-Level)           | V3-1 (Next-Segment)          |
|:-------------------|:--------------------------|:-----------------------------|
| **Sequential dim** | Concept level             | Text position (segment)      |
| **Parallel dim**   | Within-level              | Within-segment (multi-level) |
| **Coherence**      | May break                 | Natural text order           |
| **Training**       | Level-by-level            | Segment-by-segment           |
| **Inference**      | Generate all coarse first | Generate segment-by-segment  |

### 7.3 Open Questions

1. **Optimal segmentation**: Fixed vs semantic vs learned?
2. **Level scheduling**: How many levels per segment?
3. **End prediction**: When to stop generating segments?
4. **Comparison**: Which performs better empirically?
