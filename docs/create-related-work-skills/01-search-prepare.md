# Stage 01: Search Preparation

> **Purpose**: Build a systematic, comprehensive search system from the
> user definition in `00-user-define.md`. Derive venues, keywords,
> search queries, and execution plans — ALL from Stage 0 fields.
>
> **Venue Policy**: ONLY CCF-A conferences or recognized top-tier
> non-ranked venues (e.g., COLM, TMLR). No CCF-B/C or below.

---

## 1. Input: Read Stage 0

Read `00-user-define.md` and extract:

| Source Field               | Extract For                                  |
|----------------------------|----------------------------------------------|
| Research Area              | Domain classification → venue selection      |
| Research Target            | Problem keywords → relevance criteria        |
| Main Motivation            | Gap/limitation keywords → motivation queries |
| Main Idea                  | Method keywords → technique identification   |
| Key Methods                | Named techniques → method keyword matrix     |
| Comparison Dimensions      | Comparison table columns                     |
| Search Scope Summary       | Broad search direction                       |
| Primary Research Community | Which CCF-A venues to target                 |
| Year Range                 | Time window for all searches                 |

---

## 2. Keyword Extraction Protocol

### Step 1: Extract Domain Keywords (D1–Dn)

From **Research Area** + **Search Scope Summary**:

```
D1: [primary domain term, synonym1, synonym2, abbreviation]
D2: [secondary domain term, synonym1, synonym2, abbreviation]
D3: [tertiary domain term, synonym1, synonym2, abbreviation]
...
```

**Rules**:
- Include hyphen variants: "multi-agent" OR "multiagent"
- Include abbreviation expansions: "LLM" OR "large language model"
- Include variant spellings: "modeling" OR "modelling"
- Minimum 3 domain keywords, maximum 8

### Step 2: Extract Method Keywords (M1–Mn)

From **Main Idea** + **Key Methods**:

```
M1: [primary method term, synonym1, synonym2, abbreviation]
M2: [secondary method term, synonym1, synonym2, abbreviation]
M3: [tertiary method term, synonym1, synonym2, abbreviation]
...
```

**Rules**:
- Include full names AND abbreviations: "CoT" OR "chain-of-thought"
- Include related/sibling techniques that solve the same problem
- Minimum 3 method keywords, maximum 8

### Step 3: Extract Motivation Keywords

From **Main Motivation**:

```
G1: [gap/limitation term 1, related phrase]
G2: [gap/limitation term 2, related phrase]
...
```

These catch papers about the same PROBLEM even if they use different METHODS.

### Output Template

```
## Extracted Keywords for {{RESEARCH_TARGET}}

### Domain Keywords
D1: [term, syn1, syn2]
D2: [term, syn1, syn2]
D3: [term, syn1, syn2]

### Method Keywords
M1: [term, syn1, syn2]
M2: [term, syn1, syn2]
M3: [term, syn1, syn2]

### Motivation Keywords
G1: [gap term, related phrase]
G2: [gap term, related phrase]
```

---

## 3. Venue Selection — CCF-A + Recognized Top Venues ONLY

### 3.1 CCF-A Conference Registry

Select venues from the table below based on **Primary Research Community** from Stage 0.

| Community             | CCF-A Conferences                    | Non-Ranked Top Venues |
|-----------------------|--------------------------------------|-----------------------|
| AI/ML (general)       | NeurIPS, ICML, ICLR, AAAI, IJCAI     | COLM, TMLR            |
| NLP / Language        | ACL, EMNLP                           | COLM, TMLR            |
| Computer Vision       | CVPR, ICCV, ECCV                     | —                     |
| Multi-Agent Systems   | AAMAS                                | —                     |
| Robotics              | ICRA, IROS                           | CoRL                  |
| Systems / DB          | OSDI, SOSP, SIGMOD, VLDB             | —                     |
| Security / Privacy    | IEEE S&P, CCS, USENIX Security, NDSS | —                     |
| HCI                   | CHI                                  | —                     |
| Theoretical CS        | STOC, FOCS, SODA                     | —                     |
| Information Retrieval | SIGIR, WWW                           | —                     |
| Software Engineering  | ICSE, FSE                            | —                     |
| Data Mining           | KDD                                  | —                     |
| Multimedia            | ACM Multimedia                       | —                     |

### 3.2 Venue Selection Rules

```
Given: Primary Research Community from Stage 0

Rule 1: Select ALL CCF-A conferences in the primary community
Rule 2: Add non-ranked top venues relevant to the community
Rule 3: If cross-domain (e.g., AI + Finance), include CCF-A from BOTH communities
Rule 4: If Research Area involves language modeling, ADD COLM and TMLR
Rule 5: Year Range from Stage 0 applies to ALL selected venues
```

### 3.3 Output: Selected Venues Table

```
## Selected Venues for {{RESEARCH_TARGET}}

| #   | Venue   | Tier  | Community | Years     | Rationale                   |
|-----|---------|-------|-----------|-----------|-----------------------------|
| 1   | NeurIPS | CCF-A | AI/ML     | 2022-2026 | Primary ML venue            |
| 2   | ICML    | CCF-A | AI/ML     | 2022-2026 | Primary ML venue            |
| 3   | ICLR    | CCF-A | AI/ML     | 2022-2026 | Primary ML venue            |
| 4   | ACL     | CCF-A | NLP       | 2022-2026 | Primary NLP venue           |
| 5   | EMNLP   | CCF-A | NLP       | 2022-2026 | Primary NLP venue           |
| 6   | COLM    | Top   | LM        | 2024-2026 | Top language modeling venue |
| 7   | TMLR    | Top   | ML        | 2022-2026 | Top ML journal              |
| ... | ...     | ...   | ...       | ...       | ...                         |
```

**Minimum**: 4 venues. **Typical**: 6–8 venues. **Maximum**: 12 venues.

---

## 4. Keyword Matrix Construction

### 4.1 Cross-Product Generation

Generate ALL D_i × M_j combinations. **No combination may be skipped.**

```
| Combo | Query String               |
|-------|----------------------------|
| D1×M1 | "D1 M1" OR "D1_syn M1_syn" |
| D1×M2 | "D1 M2" OR "D1_syn M2_syn" |
| D1×M3 | "D1 M3" OR "D1_syn M3_syn" |
| D2×M1 | "D2 M1" OR "D2_syn M1_syn" |
| D2×M2 | "D2 M2" OR "D2_syn M2_syn" |
| D2×M3 | "D2 M3" OR "D2_syn M2_syn" |
| D3×M1 | "D3 M1" OR "D3_syn M1_syn" |
| D3×M2 | "D3 M2" OR "D3_syn M2_syn" |
| D3×M3 | "D3 M3" OR "D3_syn M3_syn" |
| ...   | ...                        |
```

### 4.2 Motivation-Based Queries

```
| Combo | Query String                                   |
|-------|------------------------------------------------|
| G1×M1 | "G1 M1" — papers addressing gap1 using method1 |
| G1×M2 | "G1 M2" — papers addressing gap1 using method2 |
| G2×M1 | "G2 M1" — papers addressing gap2 using method1 |
| ...   | ...                                            |
```

### 4.3 Total Query Count

```
Total = |D| × |M| + |G| × |M|
Minimum: 3×3 + 2×3 = 15 queries
Typical: 5×5 + 3×5 = 40 queries
Maximum: 8×8 + 4×8 = 96 queries
```

---

## 5. Search URL Registry

### 5.1 Academic Search Platforms

| Platform             | URL Template                                                                                                                                                     | Coverage                   |
|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------|
| Semantic Scholar API | `https://api.semanticscholar.org/graph/v1/paper/search?query=KEYWORDS&venue=VENUE&year=START-END&limit=100&fields=title,authors,abstract,venue,year,externalIds` | All CS, includes abstracts |
| Semantic Scholar Web | `https://www.semanticscholar.org/search?q=KEYWORDS&year%5B0%5D=START&year%5B1%5D=END`                                                                            | Web search interface       |
| DBLP API             | `https://dblp.org/search/publ/api?q=KEYWORDS&format=json&h=1000`                                                                                                 | All CS conferences         |
| DBLP Browse          | `https://dblp.org/db/conf/VENUE/index.html`                                                                                                                      | Browse proceedings         |
| Google Scholar       | `https://scholar.google.com/scholar?q=KEYWORDS&as_ylo=START&as_yhi=END`                                                                                          | Broadest coverage          |
| arXiv Search         | `https://arxiv.org/search/?query=KEYWORDS&searchtype=all`                                                                                                        | Preprints                  |
| arXiv API            | `http://export.arxiv.org/api/query?search_query=all:KEYWORDS&start=0&max_results=100`                                                                            | Programmatic               |

### 5.2 Venue-Specific Proceedings Pages

| Venue          | URL                                                       | Format               |
|----------------|-----------------------------------------------------------|----------------------|
| NeurIPS        | `https://papers.nips.cc/`                                 | Official proceedings |
| ICML           | `https://proceedings.mlr.press/`                          | Official proceedings |
| ICLR           | `https://openreview.net/group?id=ICLR.cc/YEAR/Conference` | OpenReview           |
| AAAI           | `https://ojs.aaai.org/index.php/AAAI/issue/archive`       | Official archive     |
| ACL/EMNLP      | `https://aclanthology.org/`                               | ACL Anthology        |
| CVPR/ICCV/ECCV | `https://dblp.org/db/conf/cvpr/index.html`                | DBLP                 |
| COLM           | `https://openreview.net/group?id=COLM.cc/YEAR/Conference` | OpenReview           |

### 5.3 Cross-Reference Tools

| Tool                        | URL                                                                                          | Use For            |
|-----------------------------|----------------------------------------------------------------------------------------------|--------------------|
| Google Scholar Cited By     | `https://scholar.google.com/scholar?cites=PAPER_ID`                                          | Find citing papers |
| Semantic Scholar Citations  | `https://api.semanticscholar.org/graph/v1/paper/PAPER_ID/citations?fields=title,venue,year`  | Citation graph     |
| Semantic Scholar References | `https://api.semanticscholar.org/graph/v1/paper/PAPER_ID/references?fields=title,venue,year` | Reference graph    |

### 5.4 Venue Verification Tools

Use these to confirm whether an arXiv preprint was accepted at a target venue.

| Tool             | URL                                                                    | What It Shows                                               |
|------------------|------------------------------------------------------------------------|-------------------------------------------------------------|
| arXiv Comments   | `https://arxiv.org/abs/ID` — check the `Comments` field                | "Accepted to NeurIPS 2025", "Published at ICML 2025", etc.  |
| arXiv API        | `http://export.arxiv.org/api/query?search_query=id:ID`                 | `<arxiv:comment>` field with acceptance info                |
| OpenReview       | `https://openreview.net/forum?id=PAPER_ID`                             | Accepted/Rejected badge for ICLR, COLM                      |
| Semantic Scholar | `https://api.semanticscholar.org/graph/v1/paper/arxiv:ID?fields=venue` | `venue: "NeurIPS"` or `venue: "ArXiv"`                      |
| DBLP             | `https://dblp.org/search/publ/api?q=TITLE&format=json`                 | If found → confirmed published; not found → likely preprint |

---

## 6. arXiv Category Selection

Based on **Primary Research Community** from Stage 0:

| Community       | Primary arXiv Categories | Secondary Categories |
|-----------------|--------------------------|----------------------|
| AI/ML           | cs.AI, cs.LG             | stat.ML              |
| NLP             | cs.CL                    | cs.AI                |
| Computer Vision | cs.CV                    | cs.AI, cs.LG         |
| Multi-Agent     | cs.MA                    | cs.AI                |
| Robotics        | cs.RO                    | cs.AI                |
| Systems         | cs.DC, cs.DB             | cs.SE                |
| Security        | cs.CR                    | cs.AI                |
| Theory          | cs.CC, cs.DS             | math.CO              |

**Cross-domain research**: Combine categories from ALL relevant communities.

---

## 7. Search Execution Plan

### 7.1 Phase Structure

```
Phase A: Proceedings Scan (per venue × year)
  For each (Venue, Year) in Selected Venues:
    1. Access proceedings via URL from Section 5.2
    2. Filter ALL paper titles by keyword matrix
    3. Read abstracts of matches
    4. Classify: RELEVANT / POSSIBLY_RELEVANT / NOT_RELEVANT
    5. Add RELEVANT papers to candidate list

Phase B: Keyword Combination Scan (per D×M query)
  For each query in Keyword Matrix:
    1. Search on Semantic Scholar API + Google Scholar + arXiv
    2. Filter results by venue = CCF-A or recognized top venue
    3. Read abstracts of matches
    4. Add new RELEVANT papers to candidate list

Phase C: Cross-Reference Expansion (per found paper)
  For each RELEVANT paper found in Phases A–B:
    1. Check its references → discover foundational papers
    2. Check who cites it → discover newer papers
    3. Check authors' other publications → discover related work
    4. Filter by venue policy (CCF-A or recognized top only)
    5. Add new papers to candidate list

Phase D: Keyword Expansion Loop
  While new terms are discovered during reading:
    1. Add new term to keyword matrix
    2. Re-search all venue×year with the new term
    3. Add newly discovered papers
```

### 7.2 Search Depth by Venue Tier

| Venue Type                  | Papers to Scan        | Depth                        | Time Budget            |
|-----------------------------|-----------------------|------------------------------|------------------------|
| CCF-A (primary community)   | ALL papers            | Title + Abstract             | 1–2 hrs per venue×year |
| CCF-A (secondary community) | Keyword-filtered only | Title + Abstract for matches | 30 min per venue×year  |
| Recognized Top (COLM/TMLR)  | ALL papers            | Title + Abstract             | 1 hr per venue×year    |

### 7.3 Estimated Scope

```
Venues: 6–8
Years: 3–4
Venue×Year combinations: 18–32
Keyword combinations: 15–96
Total search time: 15–40 hours (spread across sessions)
```

---

## 8. Coverage Tracker Template

Initialize this tracker BEFORE starting any search. Update after every session.

```
## Coverage Tracker for {{RESEARCH_TARGET}}

### Proceedings Coverage
| Venue   | Year | Total Papers | Scanned | Relevant | Status      |
|---------|------|--------------|---------|----------|-------------|
| NeurIPS | 2026 | ~4000        | 0       | 0        | NOT_STARTED |
| NeurIPS | 2025 | ~4000        | 0       | 0        | NOT_STARTED |
| NeurIPS | 2024 | ~4000        | 0       | 0        | NOT_STARTED |
| ICML    | 2026 | ~3000        | 0       | 0        | NOT_STARTED |
| ...     | ...  | ...          | ...     | ...      | ...         |

### Keyword Matrix Coverage
| Combination | Searched | Papers Found | Status      |
|-------------|----------|--------------|-------------|
| D1×M1       | NO       | 0            | NOT_STARTED |
| D1×M2       | NO       | 0            | NOT_STARTED |
| ...         | ...      | ...          | ...         |

### Cross-Reference Coverage
| Seed Paper | References Checked | Citations Checked | Status |
|------------|--------------------|-------------------|--------|
| [none yet] | —                  | —                 | —      |
```

**Status values**: NOT_STARTED → IN_PROGRESS → COMPLETE

---

## 9. Candidate Paper List Template

All papers discovered during search go here BEFORE reading (Stage 03).

```
## Candidate Papers for {{RESEARCH_TARGET}}

| #   | Title   | Venue   | Year   | Link  | Relevance    | Source      | Status             |
|-----|---------|---------|--------|-------|--------------|-------------|--------------------|
| 1   | [title] | [venue] | [year] | [url] | HIGH/MED/LOW | Phase A/B/C | FOUND/READ/WRITTEN |
| ... |         |         |        |       |              |             |                    |
```

**Status flow**: FOUND → READ (Stage 03) → WRITTEN (Stage 04)

---

## Validation Checklist for Stage 01

Before proceeding to Stage 02 (Search Execution), verify:

- [ ] All fields from Stage 0 have been read and used
- [ ] At least 3 Domain Keywords and 3 Method Keywords extracted
- [ ] At least 2 Motivation Keywords extracted
- [ ] Venue selection follows CCF-A-only policy (with recognized exceptions)
- [ ] All D×M and G×M combinations enumerated (no gaps)
- [ ] Coverage tracker initialized with all venue×year combinations
- [ ] Candidate Paper List template ready
- [ ] Search URLs verified for each selected venue
