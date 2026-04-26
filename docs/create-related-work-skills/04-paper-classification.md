# Stage 04: Paper Classification and Taxonomy

> **Purpose**: Organize all read papers into a two-level hierarchy
> (Category → Sub-Category) derived from `00-user-define.md`.
> This taxonomy determines the section structure of `related-work.md`.

---

## 1. Input: Read Stage 00 and Stage 03

| Source                | What to Extract                                                                                      |
|-----------------------|------------------------------------------------------------------------------------------------------|
| `00-user-define.md`   | Research Area, Target, Direction, Motivation, Idea, Key Methods, Comparison Dimensions, Search Scope |
| `03-paper-reading.md` | Paper Database with CAT/REL tags                                                                     |

---

## 2. Taxonomy Derivation Protocol

### 2.1 How Categories Are Born

Categories come DIRECTLY from `00-user-define.md`. Every category must
trace back to at least one field. No free-floating categories allowed.

```
Derivation Rules:

Rule 1: Each Key Method from 00 → one Category
  Example: Key Methods = "Concept pyramid, Residual decomposition,
           Cross-attention, Scale-level AR"
  → Category 1: Multi-Scale Representation
  → Category 2: Residual and Decomposition Methods
  → Category 3: Cross-Attention Mechanisms
  → Category 4: Autoregressive Generation

Rule 2: Main Motivation → one Category for problem-oriented papers
  Example: Motivation = "CoT generates excessive tokens"
  → Category: Chain-of-Thought Compression

Rule 3: Main Idea → one Category for the core approach
  Example: Idea = "Hierarchical concept pyramid"
  → Category: Hierarchical / Pyramidal Architectures

Rule 4: Search Scope → one Category for each distinct research thread
  Example: Scope = "latent reasoning, CoT compression,
            multi-scale generation, VAR, concept decomposition"
  → Each term maps to or merges into a Category

Rule 5: Comparison Dimensions → validation axis for each Category
  These become the TABLE COLUMNS in synthesis sections,
  not categories themselves.
```

### 2.2 Minimum and Maximum

| Level                       | Minimum | Typical | Maximum |
|-----------------------------|---------|---------|---------|
| Categories                  | 3       | 5-7     | 10      |
| Sub-categories per Category | 2       | 2-4     | 6       |
| Total Sub-categories        | 6       | 10-20   | 40      |

---

## 3. Category Construction

### 3.1 Step-by-Step Process

```
Step 1: List all Key Methods from 00-user-define.md
  M1: [method 1]
  M2: [method 2]
  M3: [method 3]
  ...

Step 2: List all Search Scope threads from 00-user-define.md
  S1: [scope thread 1]
  S2: [scope thread 2]
  S3: [scope thread 3]
  ...

Step 3: Extract the Main Motivation as a problem-oriented thread
  G1: [motivation / gap]

Step 4: Extract the Main Idea as a core approach thread
  I1: [core idea]

Step 5: Group related threads into Categories
  Merge threads that address the same problem or use the same technique.
  Split threads that cover distinct aspects.

Step 6: For each Category, define Sub-Categories
  Based on: different approaches, different problem formulations,
  different architectural choices, different training paradigms.
```

### 3.2 Category Naming Rules

```
Rule 1: Name = "[Technical Descriptor] [Approach/Problem]"
  Good:   "Latent Reasoning Methods"
  Good:   "Multi-Scale Generative Models"
  Bad:    "Other Methods"          (vague)
  Bad:    "Background"             (non-technical)

Rule 2: Each Category name must clearly differentiate from others
  Good:   "Token-Level CoT Compression" vs "Latent-Space Reasoning"
  Bad:    "CoT Methods 1" vs "CoT Methods 2"

Rule 3: Category names should map to related-work.md section headers
  Section 1 → Category 1 name
  Section 2 → Category 2 name
  ...
```

---

## 4. Sub-Category Construction

### 4.1 Sub-Category Derivation Methods

Within each Category, split into Sub-Categories using ONE of:

| Split Method           | When to Use                           | Example                                    |
|------------------------|---------------------------------------|--------------------------------------------|
| By architecture        | Same problem, different designs       | "Parallel vs Sequential"                   |
| By representation      | Same goal, different spaces           | "Continuous Latent vs Discrete Token"      |
| By training paradigm   | Same architecture, different training | "RL-based vs SFT-based vs Self-supervised" |
| By problem formulation | Same area, different angles           | "Efficiency-focused vs Quality-focused"    |
| By modality            | Different input/output types          | "Text-only vs Multimodal"                  |
| By scale               | Different granularity                 | "Single-scale vs Multi-scale"              |

### 4.2 Sub-Category Naming Rules

```
Rule 1: Name must indicate the SPLIT CRITERION
  Good:   "Continuous Latent Representations"
  Good:   "Discrete Token Compression"
  Bad:    "Sub-method A"

Rule 2: Sub-categories within a Category must be MUTUALLY EXCLUSIVE
  A paper should fit in exactly ONE sub-category per Category.

Rule 3: Each sub-category must have at least 2 papers
  If only 1 paper → merge into the closest sub-category.
```

---

## 5. Example: Full Taxonomy for {{RESEARCH_TARGET}}

Using the NLCP example from `00-user-define.md`:

```
Category 1: Chain-of-Thought Compression
  1.1 Token-Level CoT Compression    (early exit, sparse CoT, CoT trimming)
  1.2 Latent-Space Reasoning         (pause tokens, thinking tokens, latent CoT)
  1.3 Implicit Reasoning             (without explicit CoT at all)

Category 2: Multi-Scale Representation
  2.1 Feature Pyramid Networks       (FPN, U-Net style top-down/bottom-up)
  2.2 Hierarchical VQ-VAE            (VQ-VAE-2, VQGAN, multi-scale codebooks)
  2.3 Laplacian / Residual Pyramids  (Laplacian pyramid GAN, residual flow)
  2.4 Concept Decomposition          (sparse coding, dictionary learning)

Category 3: Autoregressive Generation Beyond Tokens
  3.1 Image/Visual Autoregression    (VAR, VQGAN, raster-scan AR)
  3.2 Multi-Scale AR Models          (VAR-style coarse-to-fine)
  3.3 Non-Sequential Generation      (diffusion, flow matching)

Category 4: Cross-Attention and Refinement Mechanisms
  4.1 Cross-Attention for Fusion     (encoder-decoder, multi-source fusion)
  4.2 Iterative Refinement           (decode-refine loops, error correction)
  4.3 Adaptive Computation           (dynamic depth, early exit)

Category 5: Reasoning with Structured Representations
  5.1 Graph-Based Reasoning          (GNN + LLM, knowledge graphs)
  5.2 Program-Based Reasoning        (code as reasoning, PAL, PoT)
  5.3 Decomposed Reasoning           (least-to-most, plan-and-solve)
```

---

## 6. Paper Assignment Protocol

### 6.1 Assign Each Paper to Category + Sub-Category

```
For each paper in the Paper Database (from Stage 03):

Step 1: Read its PROBLEM + INSIGHT + METHOD fields
Step 2: Match against Category definitions
  - Primary Category: where the paper's CORE contribution fits
  - A paper goes in exactly ONE Category
Step 3: Match against Sub-Category definitions within that Category
  - A paper goes in exactly ONE Sub-Category
Step 4: If a paper fits multiple Categories:
  - Place in the Category closest to its CORE contribution
  - Add a cross-reference note: "Also relevant to Category X.Y"
Step 5: If a paper fits NO Category:
  - Either create a new Sub-Category under the closest Category
  - Or create a new Category if the gap is significant
```

### 6.2 Assignment Table Template

```
## Paper Assignments for {{RESEARCH_TARGET}}

### Category 1: [Name]
| Sub-Category | Papers                    | Count |
|--------------|---------------------------|-------|
| 1.1 [Name]   | Paper A, Paper B, Paper C | 3     |
| 1.2 [Name]   | Paper D, Paper E          | 2     |
| 1.3 [Name]   | Paper F                   | 1     |

### Category 2: [Name]
| Sub-Category | Papers                    | Count |
|--------------|---------------------------|-------|
| 2.1 [Name]   | Paper G, Paper H          | 2     |
| 2.2 [Name]   | Paper I, Paper J, Paper K | 3     |
...

### Unassigned Papers
| Paper   | Reason                   | Action                      |
|---------|--------------------------|-----------------------------|
| Paper L | Doesn't fit any category | Create new sub-category 3.4 |
```

---

## 7. Taxonomy Validation

### 7.1 Completeness Checks

| Check              | Condition                                  | Fix                                    |
|--------------------|--------------------------------------------|----------------------------------------|
| Empty Category     | Category has 0 papers                      | Remove or merge with adjacent category |
| Empty Sub-Category | Sub-Category has 0 papers                  | Remove or merge                        |
| Unassigned papers  | Papers not in any Category                 | Create new Sub-Category or reassign    |
| Oversized Category | One Category has >40% of papers            | Split into two Categories              |
| Overlap            | Same paper assigned to multiple Categories | Choose primary, add cross-reference    |

### 7.2 Alignment with 00-user-define.md

| Check                  | Condition                                     | Fix                           |
|------------------------|-----------------------------------------------|-------------------------------|
| Orphan Category        | No field in 00 maps to this Category          | Justify or remove             |
| Missing Method         | A Key Method from 00 has no Category          | Create Category for it        |
| Missing Scope          | A Search Scope thread from 00 has no Category | Create Category or merge      |
| No motivation Category | Main Motivation not reflected                 | Add problem-oriented Category |

### 7.3 Structural Checks

| Check             | Condition                                    | Fix                                        |
|-------------------|----------------------------------------------|--------------------------------------------|
| Flat Category     | Category has only 1 Sub-Category             | Add more Sub-Categories or merge up        |
| Deep Sub-Category | Sub-Category has >8 papers                   | Split into finer Sub-Categories            |
| Imbalance         | Some Categories have many papers, others few | May indicate search bias; revisit Stage 02 |

---

## 8. Mapping to related-work.md Structure

The taxonomy directly defines the document structure:

```
# Related Work

## 1. [Category 1 Name]                    ← Section header
### 1.1 [Sub-Category 1.1 Name]            ← Sub-section
#### 1.1.1 [Paper Title]                   ← Paper entry (Stage 05 template)
#### 1.1.2 [Paper Title]
### 1.2 [Sub-Category 1.2 Name]
#### 1.2.1 [Paper Title]
#### 1.2.2 [Paper Title]
### 1.N Synthesis: [Category 1 Summary]    ← Comparison table

## 2. [Category 2 Name]
### 2.1 [Sub-Category 2.1 Name]
...
### 2.N Synthesis: [Category 2 Summary]

...

## N. Overall Positioning                   ← Final positioning section
  How our work relates to ALL categories
  Key differentiator from each category
```

---

## Validation Checklist for Stage 04

- [ ] Every Category traces back to a field in `00-user-define.md`
- [ ] Every Category has at least 2 Sub-Categories
- [ ] Every Sub-Category has at least 2 papers
- [ ] No paper is assigned to more than one Category (cross-references OK)
- [ ] No papers remain unassigned
- [ ] No Category has >40% of all papers (no dominance)
- [ ] Each Key Method from 00 has a corresponding Category
- [ ] Main Motivation is reflected in at least one Category
- [ ] Category names are technical and specific (no "Other" or "Background")
- [ ] Sub-Category names indicate the split criterion
- [ ] Assignment Table is complete
