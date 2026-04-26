# Stage 03: Paper Reading and Information Extraction

> **Purpose**: Read each candidate paper from Stage 02 and extract
> structured information. Bridge between discovery and writing.
>
> **Rule**: NEVER write a paper entry without reading the paper first.

---

## 1. Input: Read Candidate Paper List

Read the finalized Candidate Paper List from `02-search-execute.md`.

**Reading priority**: HIGH -> MEDIUM -> LOW

---

## 2. Multi-Source Reading Protocol

### 2.1 Reading Order Per Paper

```
Step 1: Primary source (arXiv PDF or official paper)
  Read: Abstract -> Introduction -> Method -> Experiments -> Conclusion

Step 2: Supplementary sources (HIGH/MEDIUM only)
  - GitHub README: clearer explanations
  - OpenReview: author responses and reviewer discussions
  - Author blog posts: accessible summaries

Step 3: Cross-validation (HIGH only)
  - Citing papers' descriptions of this work
  - Community discussions
```

### 2.2 Reading Depth by Relevance

| Relevance | What to Read | Depth | Time |
|-----------|-------------|-------|------|
| HIGH | Full paper + supplementary | Complete | 30-60 min |
| MEDIUM | Abstract + Intro + Key Figures | Core insight | 15-20 min |
| LOW | Abstract + Introduction only | Orientation | 5-10 min |

---

## 3. Structured Information Extraction

### 3.1 Extraction Template

For EVERY paper, fill in this template:

```
Paper: [Full Title]
Venue: [Conference/Journal Year]
Link: [arXiv or conference URL]

1. PROBLEM: [1 sentence — what problem are they solving?]
2. INSIGHT: [1 sentence — what is the key insight/idea?]
3. METHOD: [3-5 sentences — how do they implement the insight?]
   - Key architecture component:
   - Key training objective:
   - Key data requirement:
4. RESULT: [1-2 sentences with specific numbers from the paper]
5. LIMITATIONS: [1 sentence — what doesn't it do?]
6. CONNECTION TO {{RESEARCH_TARGET}}: [2-3 sentences]
```

### 3.2 Extraction Rules

| Rule | Description |
|------|------------|
| Specificity | "84.2% on GSM8K" not "high accuracy" |
| Accuracy | All numbers must come from the paper |
| Completeness | Every field must be filled — no TBD/TODO |
| Conciseness | PROBLEM and INSIGHT are 1 sentence each |
| Terminology | Use terminology from `00-user-define.md` |

### 3.3 Categorization Tags

```
**[CAT: X] [REL: Y]**

CAT: Core | Efficiency | Training | Analysis | Theory
REL: Critical | High | Medium | Low
```

- Critical: same problem, nearly identical method
- High: strong conceptual overlap or complementary technique
- Medium: related area with useful insights
- Low: peripheral relevance

---

## 4. Batch Reading Strategy

### 4.1 Grouping Papers

```
Group 1: Same-method papers -> compare approaches efficiently
Group 2: Same-venue papers -> identify temporal trends
Group 3: Same-author papers -> understand research trajectory
Group 4: Foundational -> Derived -> read older papers first
```

### 4.2 Agent Template — Paper Reading

```
You are a research paper reading agent. Read the following paper
and extract structured information:

Paper: [TITLE]
Link: [URL]

Read: Abstract -> Introduction -> Method -> Experiments -> Conclusion

Return:
1. PROBLEM: [1 sentence]
2. INSIGHT: [1 sentence]
3. METHOD: [3-5 sentences with key architecture, training, data]
4. RESULT: [1-2 sentences with specific numbers]
5. LIMITATIONS: [1 sentence]
6. CONNECTION: [2-3 sentences relating to {{RESEARCH_TARGET}}]

Be accurate. All numbers must come from the paper.
Use terminology: {{TERMINOLOGY_LIST}}
```

### 4.3 Parallel Agent Deployment

```
Run 4-6 agents in parallel, each reading 1-2 papers:
  Agent 1: Read Paper 1 (HIGH relevance)
  Agent 2: Read Paper 2 (HIGH relevance)
  Agent 3: Read Paper 3 (HIGH relevance)
  Agent 4: Read Papers 4-5 (MEDIUM relevance)
  Agent 5: Read Papers 6-7 (MEDIUM relevance)
```

---

## 5. Paper Database Output

After reading all papers, compile into a structured database:

```
## Paper Database for {{RESEARCH_TARGET}}

### Critical Relevance (must include)
| # | Title | Venue | Problem | Insight | CAT | REL |
|---|-------|-------|---------|---------|-----|-----|

### High Relevance (must include)
| # | Title | Venue | Problem | Insight | CAT | REL |
|---|-------|-------|---------|---------|-----|-----|

### Medium Relevance (selective include)
| # | Title | Venue | Problem | Insight | CAT | REL |
|---|-------|-------|---------|---------|-----|-----|

### Low Relevance (mention only if needed)
| # | Title | Venue | Problem | Insight | CAT | REL |
|---|-------|-------|---------|---------|-----|-----|
```

---

## 6. Cross-Reference During Reading

| What to Look For | Where to Check | Action |
|-----------------|---------------|--------|
| Papers NOT in Candidate List | Reference section | Add to Candidate List |
| New terms not in keyword matrix | Throughout paper | Add to keyword matrix |
| Contradicting/extend findings | Related Work section | Note for writing |
| Repeated authors | Author lists | Track their other work |

---

## Validation Checklist for Stage 03

- [ ] All HIGH-relevance papers read with full extraction
- [ ] All MEDIUM-relevance papers read with core extraction
- [ ] Extraction template filled for every paper (no TBD/TODO)
- [ ] All numbers verified against original paper
- [ ] CAT/REL tags assigned for every paper
- [ ] Paper Database compiled and organized by relevance
- [ ] New terms added to keyword matrix
- [ ] New papers added to Candidate List
- [ ] Candidate Paper List Status updated: FOUND -> READ
