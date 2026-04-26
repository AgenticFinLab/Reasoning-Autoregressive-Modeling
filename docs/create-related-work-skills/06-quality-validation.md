# Stage 06: Quality Validation and Iteration

> **Purpose**: Validate completeness, accuracy, and consistency of
> `docs/related-work.md`. Run automated checks, fix issues iteratively,
> and ensure no gaps remain before final submission.

---

## 1. Input: Read related-work.md

Read `docs/related-work.md` and the Paper Database from `03-paper-reading.md`.

---

## 2. Automated Validation

### 2.1 Validation Script

```python
import re

with open('docs/related-work.md') as f:
    content = f.read()

papers = re.findall(r'^(#### \d+\.\d+ .*?)\n', content, re.MULTILINE)
positions = [m.start() for m in re.finditer(r'^#### \d+\.\d+ ', content, re.MULTILINE)]

required_sections = [
    'Summary', 'Core Motivation', 'Core Idea',
    'Core Method', 'Example', 'Relationship to Our Work',
]

incomplete = []
for i, pos in enumerate(positions):
    end_pos = positions[i + 1] if i + 1 < len(positions) else len(content)
    block = content[pos:end_pos]
    has_tags = '[CAT:' in block and '[REL:' in block
    missing = [s for s in required_sections if not re.search(r'#####?\s+' + re.escape(s), block)]
    if not has_tags or missing:
        incomplete.append((papers[i], missing, not has_tags))

print(f"Total paper entries: {len(papers)}")
print(f"Complete: {len(papers) - len(incomplete)}")
print(f"Incomplete: {len(incomplete)}")
for title, missing, no_tags in incomplete:
    issues = missing + (['missing CAT/REL tags'] if no_tags else [])
    print(f"  {title}: {', '.join(issues)}")
```

### 2.2 Validation Checks

| Check                   | What to Verify                      | Pass Criteria                  |
|-------------------------|-------------------------------------|--------------------------------|
| Structural completeness | All 9 components present            | 0 missing sections             |
| CAT/REL tags            | Every paper has tags                | 0 papers without tags          |
| Links valid             | Every **Link** resolves             | All links return 200           |
| No placeholders         | No TBD/TODO/...                     | 0 placeholder occurrences      |
| Venue policy            | All papers from CCF-A or top venues | 0 non-compliant                |
| Terminology consistency | Same terms across entries           | No contradictory usage         |
| Comparison dimensions   | Tables use consistent columns       | Same dimensions across entries |

### 2.3 Quick Grep Checks

```bash
grep -n "TBD\|TODO\|\.\.\." docs/related-work.md
grep -n "^#### " docs/related-work.md | grep -v "\[CAT:"
grep -n "\*\*Venue\*\*" docs/related-work.md
```

---

## 3. Gap Detection

### 3.1 Coverage Gap Checks

| Check               | Condition                                 | Action                 |
|---------------------|-------------------------------------------|------------------------|
| Zero-paper venue    | Venue has 0 papers in related-work.md     | Re-search that venue   |
| Year gap            | Papers from some years but not others     | Fill the gap           |
| Venue concentration | >60% from 1-2 venues                      | Expand search          |
| Missing methods     | No paper covers a Key Method from Stage 0 | Search for that method |
| Temporal skew       | All papers from same year                 | Widen year range       |

### 3.2 Content Gap Checks

| Check               | Condition                      | Action                  |
|---------------------|--------------------------------|-------------------------|
| No comparison table | Entry lacks relationship table | Add table with >=3 dims |
| No example          | Entry lacks concrete example   | Write toy problem       |
| Shallow method      | Core Method < 5 sentences      | Expand with details     |
| Vague motivation    | Describes method, not problem  | Rewrite from the gap    |

---

## 4. Accuracy Verification (HIGH/Critical only)

```
For each HIGH/Critical paper:

Step 1: Re-read written Summary against paper abstract
  -> claims match, numbers match, no exaggeration

Step 2: Verify Core Idea matches paper's central claim
  -> the insight is actually what the paper proposes

Step 3: Check Example accurately illustrates the method
  -> the toy problem correctly shows before/after

Step 4: Confirm comparison table is fair and accurate
  -> their approach is not misrepresented

Step 5: Verify all numbers come from the paper
  -> every quantitative claim has a source (Table N, Section N)
```

### Common Accuracy Issues

| Issue                 | Example                          | Fix                           |
|-----------------------|----------------------------------|-------------------------------|
| Exaggerated claims    | "significant improvement" for 2% | Use exact: "2.1% improvement" |
| Wrong attribution     | Method X does Y when it does Z   | Re-read method section        |
| Outdated numbers      | arXiv v1 vs camera-ready         | Check final version           |
| Misleading comparison | Different settings/models        | Note comparison conditions    |

---

## 5. Consistency Check

| What to Check                   | How                                       |
|---------------------------------|-------------------------------------------|
| Same paper in multiple sections | Search title -> facts must match          |
| Terminology consistency         | Grep for term variants -> unify           |
| Comparison table columns        | All tables use same dimension names       |
| Reference format                | All entries use same heading level (####) |

### Duplicate Detection

```bash
grep -o "^#### [0-9.]* .*" docs/related-work.md | sort -t' ' -k3 | uniq -d -f2
```

If a paper appears in multiple sections:
- Keep consistent core facts (method, key result)
- Section-appropriate emphasis
- Add cross-reference: "See Section X.Y for detailed discussion"

---

## 6. Iterative Refinement Loop

```
Run Validation Script
    |
Identify Incomplete Papers
    |
Fix Batch (write missing sections)
    |
Re-validate
    |
Still incomplete? -- YES -> loop back
    | NO
Run Gap Detection
    |
Gaps found? -- YES -> return to Stage 02 (search)
    | NO
Run Accuracy Verification
    |
Issues found? -- YES -> fix and re-verify
    | NO
DONE -- related-work.md is complete
```

### Refinement Schedule

| Frequency                   | Action                     | Rationale            |
|-----------------------------|----------------------------|----------------------|
| After every 5-10 entries    | Run validation script      | Catch issues early   |
| After each thematic section | Run gap detection          | Section completeness |
| After all entries written   | Full accuracy verification | Final quality pass   |
| Before paper submission     | Complete re-validation     | Nothing missed       |

---

## 7. Batch Processing for Large Datasets

When upgrading 20+ papers:

**Phase 1: Structural Completion**
- Run validation -> identify all missing sections
- Use bulk templates for MEDIUM/LOW papers
- Focus effort on HIGH/Critical papers

**Phase 2: Manual Upgrade of Critical Papers**
- Custom Core Method with ASCII diagrams
- Concrete, paper-specific Examples
- Accurate comparison tables

**Phase 3: Incremental Refinement**
- Re-run validation
- Fix remaining issues
- Spot-check bulk-generated content

**Phase 4: Final Validation**
- Complete re-run of all checks
- Manual read-through of HIGH/Critical entries
- Confirm 0 incomplete entries

---

## 8. Stub-to-Full Entry Pipeline

```
Step 1: Research the paper (Stage 03 extraction template)
Step 2: Write Tags + Metadata (Components 1-2)
Step 3: Write Core Motivation (Component 4)
Step 4: Write Core Idea (Component 5)
Step 5: Write Core Method (Component 6)
Step 6: Write Example (Component 7)
Step 7: Write Summary (Component 3) -- LAST because it synthesizes
Step 8: Write Relationship table (Component 9)
Step 9: Validate with this stage's checks
```

---

## Final Validation Checklist

- [ ] Validation script passes: 0 incomplete entries
- [ ] All papers from CCF-A or recognized top venues
- [ ] No placeholder text (TBD/TODO/...)
- [ ] All links resolve correctly
- [ ] All quantitative claims verified against original papers
- [ ] Terminology consistent with Stage 0 Terminology Table
- [ ] Comparison tables use consistent dimensions
- [ ] No duplicate entries with contradictory facts
- [ ] Coverage tracker shows all venue x year COMPLETE
- [ ] Gap detection shows 0 issues
- [ ] Manual read-through of HIGH/Critical entries confirms accuracy
