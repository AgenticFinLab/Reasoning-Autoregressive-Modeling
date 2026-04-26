# Stage 05: Paper Entry Writing

> **Purpose**: Write complete paper entries in `docs/related-work.md`
> using the unified 9-component template. Every entry must be
> thorough, accurate, and consistently formatted.

---

## 1. Input: Read Paper Database and Taxonomy

Read the Paper Database from `03-paper-reading.md`, the Taxonomy
from `04-paper-classification.md`, and `00-user-define.md`.

**Writing priority**: Critical -> High -> Medium -> Low

---

## 2. Unified Paper Entry Template (9 Components)

### Component 1: Categorization Tags

```
**[CAT: X] [REL: Y]**
```
- CAT: Core | Efficiency | Training | Analysis | Theory
- REL: Critical | High | Medium | Low

### Component 2: Paper Metadata

```
**Paper**: "Full Paper Title"
**Authors**: Author Names (if notable)
**Venue**: Conference/Journal Name Year
**Link**: https://arxiv.org/abs/XXXX.XXXXX
**Code**: https://github.com/... (or **Code**: Null)
```

Rules: Title must match PDF exactly. Link must resolve.

### Component 3: Summary

- Length: 3-6 sentences
- Content: motivation + contribution + key result
- Style: Explain what the paper DOES, not just what it CLAIMS

**Bad**: "This paper proposes a new method for latent reasoning."
**Good**: "This paper proposes Pause Tokens that insert computation steps without generating intermediate tokens, improving GSM8K from 79.1% to 84.2% while reducing output tokens by 40%."

### Component 4: Core Motivation

- Question: Why did the authors write this? What gap?
- Format: Start with the PROBLEM, not the solution
- Structure: [Prior approaches] have limitation [X]. This matters because [consequence]. The authors seek to [address this gap].

### Component 5: Core Idea

- Question: What is the single most important insight?
- Format: Transformation, formula, or before/after comparison
- Use ASCII code blocks or equations

Example:
```
Before: Q -> [LLM generates full CoT tokens] -> A
               (100+ tokens, slow, verbose)
After:  Q -> [LLM generates latent vectors] -> A
               (1 vector, fast, compact)
```

### Component 6: Core Method

- Question: How do they implement the core idea?
- Depth: Step-by-step with architecture details
- Format: ASCII diagrams + pseudocode
- Include: Input -> Process -> Output pipeline

Structure:
```
1. [Input description with dimensions]
2. [Step 1 with architecture component]
3. [Step 2 with training objective]
4. [Step 3 with output format]
5. [Key design choices and rationale]
```

ASCII diagram example:
```
Input: Question Q
    |
[Embedding Layer]
    |
+----------------------------------+
| Step 1: [Component Name]         |
| h = Encoder(Q)                    |
|         |                         |
| Step 2: [Component Name]         |
| z = Project(h)                    |
|         |                         |
| Step 3: [Component Name]         |
| A = Decode(z)                     |
+----------------------------------+
    |
Output: Answer A
```

### Component 7: Example

- Requirement: Concrete, simple, self-contained
- Style: Show BEFORE (baseline) and AFTER (their method) side by side
- Goal: Reader understands contribution from this alone

Structure:
```
Problem: [Concrete toy problem]

BEFORE (baseline):
  [Step-by-step trace]
  Result: [output, token count, latency]

AFTER (this paper):
  [Step-by-step trace]
  Result: [output, token count, latency]

Key difference: [1 sentence]
```

### Component 8: Key Results (Optional but Recommended)

- Bullet points with quantitative results
- Always cite specific tables/sections from the paper

Format:
```
- Accuracy: [X]% on [dataset] (Table [N])
- Speedup: [X]x faster than [baseline] (Section [N])
- Token reduction: [X]% fewer tokens (Figure [N])
```

### Component 9: Relationship to Our Work

Brief text: 1-2 sentences summarizing the relationship.

Comparison table (MUST include at least 3 dimensions):
```
| Aspect  | Their Work       | Our Work ({{RESEARCH_TARGET}}) |
|---------|------------------|--------------------------------|
| [Dim 1] | [Their approach] | [Our approach]                 |
| [Dim 2] | [Their approach] | [Our approach]                 |
| [Dim 3] | [Their approach] | [Our approach]                 |
```

Use dimensions from `00-user-define.md` Comparison Dimensions.

---

## 3. Writing Order

Write components in this specific order:

```
1. Core Motivation  -- grounds everything in the problem
2. Core Idea         -- the central insight
3. Core Method       -- technical implementation
4. Example           -- concretize with a toy problem
5. Summary           -- synthesize into concise overview
6. Relationship      -- compare systematically
7. Tags and Metadata -- categorize and link
8. Key Results       -- add quantitative support
```

---

## 4. Writing Style Guidelines

| Guideline              | Rule                                            |
|------------------------|-------------------------------------------------|
| Accuracy over hype     | "5% improvement" not "significant improvement"  |
| Specific over vague    | "84.2% on GSM8K" not "high accuracy"            |
| Active voice           | "The authors propose" not "It is proposed that" |
| Consistent terminology | Use terms from `00-user-define.md`              |
| No inline comments     | All explanations as full sentences              |
| Dimension annotations  | Mark all tensor shapes: [B, L, D]               |

---

## 5. Section Organization in related-work.md

### Header Hierarchy

```
### 1.   [Section Name]        <- ORGANIZATION header (no template)
#### 1.1 [Paper Title]         <- PAPER entry (needs full template)
#### 1.2 [Paper Title]         <- PAPER entry
### 2.   [Section Name]        <- ORGANIZATION header
```

### Distinguishing Papers from Organization Headers

| Feature                     | Paper Entry | Organization Header |
|-----------------------------|-------------|---------------------|
| Has [CAT:X] [REL:Y]?        | Yes         | No                  |
| Has **Paper**: line?        | Yes         | No                  |
| Has **Link**: line?         | Yes         | No                  |
| Needs 9-component template? | Yes         | No                  |

---

## 6. Synthesis Sections

At the end of each thematic group, add a synthesis:

```
### X.N Synthesis: [Theme Name]

| Method   | [Dim 1] | [Dim 2] | [Dim 3] | [Dim 4] |
|----------|---------|---------|---------|---------|
| Paper A  | ...     | ...     | ...     | ...     |
| Paper B  | ...     | ...     | ...     | ...     |
| **Ours** | ...     | ...     | ...     | ...     |

**Gap identified**: [What is missing from the literature?]
**Our position**: [Where does our work fit?]
```

---

## 7. Common Pitfalls

| Pitfall             | Bad                                   | Good                                                                                                           |
|---------------------|---------------------------------------|----------------------------------------------------------------------------------------------------------------|
| Shallow summary     | "Proposes a new method"               | "Proposes pause tokens that insert computation steps without token generation, improving GSM8K 79.1% to 84.2%" |
| Motivation = method | "They use RL to train"                | "Prior methods require curriculum learning with catastrophic forgetting. They seek single-stage training."     |
| Missing example     | "Method compresses traces"            | "Standard CoT: 45 tokens -> Theirs: 1 latent vector (2048-dim) encoding same reasoning"                        |
| Vague relationship  | "Related to our work"                 | "Both use multi-scale features, but theirs is sequential while ours uses parallel cross-scale attention"       |
| Inconsistent terms  | "latent vectors" then "hidden states" | Pick one term, note when papers differ                                                                         |
| Unchecked claims    | "Achieves 10x speedup"                | "Achieves 1.6-2.0x speedup (Section 4.2, Table 3)"                                                             |

---

## Validation Checklist for Stage 04

- [ ] Every paper entry has all 9 components
- [ ] [CAT:X] [REL:Y] tags present and accurate
- [ ] Paper title matches PDF exactly
- [ ] Link is clickable and resolves
- [ ] Summary explains what the paper DOES
- [ ] Core Motivation starts with the problem
- [ ] Core Idea expressed as a single clear insight
- [ ] Core Method includes step-by-step technical details
- [ ] Example is concrete, simple, self-contained
- [ ] Relationship table has at least 3 dimensions
- [ ] Terminology consistent with Stage 0
- [ ] No placeholder text (TBD/TODO/...)
- [ ] Organization headers distinct from paper entries
