# Stage 02: Search Execution

> **Purpose**: Execute the search plan from `01-search-prepare.md`.
> Scan venue proceedings, run keyword queries, expand via cross-references,
> and produce a complete candidate paper list.
>
> **Venue Policy**: CCF-A or recognized top venues ONLY.
> Any paper not from these venues must be explicitly justified.

---

## 1. Input: Read Stage 01

Read the following from `01-search-prepare.md`:

| Item                  | Use For                 |
|-----------------------|-------------------------|
| Selected Venues Table | Which venues to scan    |
| Keyword Matrix        | Which queries to run    |
| Search URLs           | Where to search         |
| Coverage Tracker      | What has been searched  |
| Candidate Paper List  | Where to record results |

---

## 2. Phase A: Proceedings Scan

### 2.1 Protocol — Per Venue x Year

```
Step 1: Access proceedings
  - Open the venue-specific URL from 01-search-prepare.md Section 5.2
  - If URL fails, use DBLP: https://dblp.org/db/conf/VENUE/index.html

Step 2: Scan ALL paper titles
  - For CCF-A primary community venues: scan every paper title
  - For CCF-A secondary community venues: filter by keyword matrix first

Step 3: Filter by keyword matrix
  - A paper matches if its title contains at least 1 Domain keyword
    AND at least 1 Method keyword
  - OR if its title contains any Motivation keyword

Step 4: Read abstracts of matches
  - RELEVANT: clearly addresses the same problem or uses closely related methods
  - POSSIBLY_RELEVANT: tangential overlap, needs deeper look
  - NOT_RELEVANT: no real connection despite keyword match

Step 5: Record in Candidate Paper List
  - Add RELEVANT and POSSIBLY_RELEVANT papers
  - Set Source = "Phase A: [Venue] [Year]"
  - Set Status = FOUND

Step 6: Update Coverage Tracker
  - Increment Scanned count
  - Increment Relevant count
  - Set Status = COMPLETE for this venue x year
```

### 2.2 Agent Template — Proceedings Scan

```
You are a literature search agent. Scan [VENUE] [YEAR] proceedings
for papers related to: {{RESEARCH_AREA}}.

Search keywords: {{KEYWORD_LIST}}

Step 1: Go to [VENUE_URL] and find the full list of accepted papers.
Step 2: For each paper, check if the title or abstract contains any
        of the search keywords.
Step 3: For matches, read their abstracts.
Step 4: Return a structured list of ALL relevant papers with:
  - Exact title
  - Authors
  - Link (arxiv or conference page)
  - 2-sentence relevance summary
  - Relevance: HIGH / MEDIUM / LOW

IMPORTANT: Only include papers from CCF-A or recognized top venues.
Be exhaustive. Return ALL matches, not just top ones.
```

### 2.3 Parallel Agent Deployment

```
Phase A (run 4-6 agents in parallel):
  Agent 1: Scan [Primary CCF-A Venue 1] [Current Year]
  Agent 2: Scan [Primary CCF-A Venue 1] [Current Year - 1]
  Agent 3: Scan [Primary CCF-A Venue 2] [Current Year]
  Agent 4: Scan [Primary CCF-A Venue 3] [Current Year]
  Agent 5: Scan [Secondary CCF-A Venue 1] [Current Year]
  Agent 6: Scan [COLM] [Current Year]   (if applicable)
```

After agents complete: merge results, deduplicate, update Coverage Tracker.

### 2.4 Venue Verification — Determine Acceptance Status

When a paper is found on arXiv without a clear venue, you MUST verify
whether it was accepted at a CCF-A or recognized top venue.

**Method 1: arXiv Comments Field**

arXiv papers often include acceptance info in the `Comments` metadata.

```
Example arXiv pages:
  https://arxiv.org/abs/2401.12345

Comments field patterns to look for:
  "Accepted to NeurIPS 2025"
  "Published at ICML 2025"
  "To appear at ICLR 2025"
  "Accepted as spotlight at NeurIPS 2024"
  "Camera-ready for ACL 2025"
  "Oral presentation at EMNLP 2024"

How to extract programmatically:
  arXiv API: http://export.arxiv.org/api/query?search_query=id:2401.12345
  Response contains <arxiv:comment> field with acceptance info
```

**Method 2: OpenReview**

ICLR, COLM, and some NeurIPS workshops use OpenReview.
Acceptance status is directly visible on the forum page.

```
URL pattern: https://openreview.net/forum?id=PAPER_ID

What to check:
  - Look for "Accepted" / "Rejected" badge at the top
  - Check the decision metadata under "Official Review"
  - ICLR: search https://openreview.net/group?id=ICLR.cc/YEAR/Conference
  - COLM:  search https://openreview.net/group?id=COLM.cc/YEAR/Conference

API endpoint:
  https://api2.openreview.net/notes?content.venueid=ICLR.cc/2025/Conference&content.title={TITLE}
```

**Method 3: Semantic Scholar Venue Field**

```
API: https://api.semanticscholar.org/graph/v1/paper/arxiv:2401.12345
     ?fields=venue,publicationVenue

Returns:
  "venue": "NeurIPS"    (if published at a known venue)
  "venue": "ArXiv"      (if still a preprint only)
  "venue": ""            (no venue info)
```

**Method 4: DBLP Lookup**

```
API: https://dblp.org/search/publ/api?q=TITLE&format=json

DBLP only indexes papers that appeared in conference proceedings
or journals. If a paper is found on DBLP, it is confirmed published.

If NOT found on DBLP but on arXiv: likely still a preprint.
```

**Decision Matrix for arXiv Preprints**

| arXiv Comment             | Semantic Scholar Venue | DBLP      | Decision                              |
|---------------------------|------------------------|-----------|---------------------------------------|
| "Accepted to [CCF-A]"     | Any                    | Any       | **KEEP** — confirmed CCF-A            |
| "Under review at [CCF-A]" | Any                    | Any       | **DISCARD** — not yet accepted        |
| "Accepted at [CCF-B/C]"   | Any                    | Any       | **DISCARD** — wrong tier              |
| No comment                | CCF-A venue            | Found     | **KEEP** — confirmed via DBLP/SS      |
| No comment                | ArXiv / empty          | Not found | **KEEP as TENTATIVE** — revisit later |
| No comment                | CCF-B/C venue          | Found     | **DISCARD** — wrong tier              |

**Tentative papers**: Mark as `TENTATIVE` in Candidate Paper List.
Re-check acceptance status monthly or before writing related-work.md.

---

## 3. Phase B: Keyword Combination Scan

### 3.1 Protocol — Per Keyword Combination

```
For each (D_i x M_j) combination in the Keyword Matrix:

Step 1: Search Semantic Scholar API
  URL: https://api.semanticscholar.org/graph/v1/paper/search
       ?query="D_i+M_j"
       &year={{START}}-{{END}}
       &limit=100
       &fields=title,authors,abstract,venue,year,externalIds

Step 2: Search Google Scholar
  URL: https://scholar.google.com/scholar?q="D_i+M_j"
       &as_ylo={{START}}&as_yhi={{END}}

Step 3: Search arXiv
  URL: https://arxiv.org/search/?query="D_i+M_j"&searchtype=all

Step 4: Filter results by venue policy
  - KEEP: papers from CCF-A venues or recognized top venues
  - KEEP: arXiv preprints that PASS venue verification (see Section 2.4)
  - DISCARD: papers from CCF-B/C or unknown venues

Step 5: Verify venue acceptance for arXiv preprints (see Section 2.4)

Step 6: Add new papers to Candidate Paper List
  - Set Source = "Phase B: D_i x M_j"
  - Set Status = FOUND

Step 7: Update Keyword Matrix Coverage in tracker
```

### 3.2 Agent Template — Keyword Scan

```
You are a literature search agent. Search for papers matching:
"D_i M_j"

Search on ALL of these sources:
1. Semantic Scholar: https://www.semanticscholar.org/search?q=D_i+M_j
2. Google Scholar: https://scholar.google.com/scholar?q=D_i+M_j
3. arXiv: https://arxiv.org/search/?query=D_i+M_j&searchtype=all

For each paper found:
  - Exact title
  - Authors
  - Link
  - Venue and year
  - 1-sentence relevance summary
  - Relevance: HIGH / MEDIUM / LOW

IMPORTANT: Only include papers from: {{CCF_A_VENUE_LIST}}
Return ALL unique papers. Deduplicate across sources.
```

### 3.3 Parallel Deployment

```
Phase B (run 4-6 agents in parallel):
  Agent 1: Search D1 x M1 through D1 x Mn on Semantic Scholar
  Agent 2: Search D2 x M1 through D2 x Mn on Semantic Scholar
  Agent 3: Search D3 x M1 through D4 x Mn on Google Scholar
  Agent 4: Search D5 x M1 through Dn x Mn on arXiv
  Agent 5: Search G1 x M1 through Gn x Mn on Semantic Scholar
  Agent 6: Search G1 x M1 through Gn x Mn on Google Scholar
```

---

## 4. Phase C: Cross-Reference Expansion

### 4.1 Protocol — Per Relevant Paper

```
For each paper marked RELEVANT in the Candidate Paper List:

Step 1: Check its references
  - Semantic Scholar API:
    https://api.semanticscholar.org/graph/v1/paper/PAPER_ID/references
    ?fields=title,venue,year
  - Filter by venue policy (CCF-A or recognized top only)
  - Add new papers to Candidate Paper List

Step 2: Check who cites it
  - Google Scholar: https://scholar.google.com/scholar?cites=PAPER_ID
  - Or Semantic Scholar API:
    https://api.semanticscholar.org/graph/v1/paper/PAPER_ID/citations
    ?fields=title,venue,year
  - Filter by venue policy

Step 3: Check authors' other publications
  - Search: "[Author Name]" + domain keywords on Semantic Scholar
  - Filter by venue policy

Step 4: Read its "Related Work" section
  - Cited foundational papers may have been missed
  - Add any relevant citations to Candidate Paper List
```

### 4.2 Stopping Criteria

Stop cross-reference expansion when:
- 3 consecutive papers yield no new candidates
- All HIGH-relevance papers have been cross-referenced
- Candidate list exceeds 50 papers (prioritize reading over more discovery)

---

## 5. Phase D: Keyword Expansion Loop

### 5.1 Protocol

```
When a new technical term is discovered during reading:
  1. Add it to the keyword matrix as D_new or M_new
  2. Generate new cross-product combinations
  3. Re-search all venue x year with the new combinations
  4. Add newly discovered papers to Candidate Paper List
  5. Update Coverage Tracker

Example:
  Reading Paper X -> discover term "residual quantization"
  Add M_new: "residual quantization, RQ, residual VQ"
  Generate: D1 x M_new, D2 x M_new, ...
  Re-search all venues with new queries
```

### 5.2 When to Re-Run

| Trigger                               | Action                         |
|---------------------------------------|--------------------------------|
| Discover >= 2 new technical terms     | Re-search with expanded matrix |
| Finish reading a HIGH-relevance paper | Check its references           |
| Weekly routine                        | Check arXiv new submissions    |
| Before writing related-work section   | Final verification pass        |

---

## 6. Output: Finalized Candidate Paper List

```
| # | Title | Venue   | Year | Link        | Relevance | Source  | Status |
|---|-------|---------|------|-------------|-----------|---------|--------|
| 1 | ...   | NeurIPS | 2025 | https://... | HIGH      | Phase A | FOUND  |
| 2 | ...   | ICML    | 2025 | https://... | HIGH      | Phase B | FOUND  |
| 3 | ...   | ICLR    | 2024 | https://... | MEDIUM    | Phase C | FOUND  |
```

**Typical output**: 20-60 candidate papers for a well-defined research area.

---

## 7. Gap Detection

After all phases, run these checks:

| Check               | Condition                              | Action                        |
|---------------------|----------------------------------------|-------------------------------|
| Zero-paper venue    | Venue has 0 candidates                 | Re-search that venue          |
| Year gap            | Papers from some years but not others  | Fill the gap                  |
| Venue concentration | >60% from 1-2 venues                   | Expand search                 |
| Missing keywords    | Some D x M combinations have 0 results | Verify they were searched     |
| Temporal skew       | All papers from same year              | Widen year range              |
| No cross-refs       | 0 papers from Phase C                  | Run cross-reference expansion |

---

## Validation Checklist for Stage 02

- [ ] All venue x year combinations in Coverage Tracker are COMPLETE
- [ ] All D x M and G x M keyword combinations have been searched
- [ ] Cross-reference expansion run for all HIGH-relevance papers
- [ ] Keyword expansion loop executed (no new terms pending re-search)
- [ ] Gap detection passed with no issues
- [ ] Candidate Paper List has at least 15 papers
- [ ] All papers from CCF-A or recognized top venues
- [ ] Coverage Tracker is fully updated
