# Skill: Generate Production-Quality BibTeX from Markdown Literature Reviews

## Overview

This skill describes the complete workflow for converting a Markdown-based literature review (where each paper has title, link, summary, etc.) into a production-quality `related-work.bib` file. The output BibTeX must have:

- **Verified article details**: correct title, complete author list, accurate year.
- **Identified venue**: actual conference or journal; arXiv is used only as a last resort.
- **Short citation keys**: `MethodAbbreviation-VenueYear` format (e.g., `Coconut-COLM25`, `VAR-NeurIPS24`).
- **Correct entry types**: `@inproceedings` for conferences/workshops, `@article` for journals and arXiv-only papers.
- **Proper formatting**: `Proc.~` prefix for all conference `booktitle` fields.
- **Accurate categorization**: topical section comments in the BibTeX file.

---

## Quick Start

For immediate use, follow these three steps:

### Step 1: Ensure your `related-work.md` follows the standard format

Each paper should have at minimum:
```markdown
### X.Y Paper Title (Venue Year)

**Paper**: "Full Paper Title"
**Link**: https://arxiv.org/abs/XXXX.XXXXX
**Code**: Null

#### Summary
...
```

### Step 2: Copy the complete script from Phase 6

Save it as `generate_bib.py` in your project root.

### Step 3: Run the script

```bash
python3 generate_bib.py docs/related-work.md docs/related-work.bib
```

The first run will fetch metadata from arXiv and cache it in `.bib_cache.json`. Subsequent runs are nearly instantaneous.

### Expected Markdown Structure

The script recognizes these patterns automatically:

| Element           | Pattern                        | Required?                        |
|-------------------|--------------------------------|----------------------------------|
| Main paper header | `### 1.1 Title`                | Yes                              |
| Sub-paper header  | `#### 1.1.1 Title`             | Yes                              |
| Paper title       | `**Paper**: "Title"`           | No (falls back to header)        |
| Link              | `**Link**: URL`                | No (falls back to search)        |
| Authors           | `**Authors**: Name1 and Name2` | No (fetched from arXiv)          |
| Venue             | `**Venue**: ICLR 2026`         | No (inferred from arXiv comment) |
| Code              | `**Code**: URL or Null`        | No                               |
| Section topic     | `## Section Name`              | No (groups by section)           |

---

## Phase 1: Extract and Inventory Papers from Markdown

### 1.1 Parse the Source Markdown

The source `related-work.md` follows a specific structure with two types of papers:

**Main papers** (`### X.Y Title`):
```markdown
### 10.1 Paper Title (Venue Year)

**[CAT: Core] [REL: High]**

**Paper**: "Full Paper Title"
**Link**: https://arxiv.org/abs/XXXX.XXXXX
**Code**: Null

#### Summary
...
```

**Sub-papers** (`#### X.Y.Z Title`):
```markdown
#### 17.1.1 Paper Title (Venue Year)

**[CAT: Training] [REL: Medium]**

**Paper**: "Full Paper Title"
**Link**: https://openreview.net/forum?id=XXXX
```

### 1.2 Complete Extraction Script

Use this comprehensive regex-based parser:

```python
import re
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Paper:
    raw_title: str        # Title from markdown header
    paper_title: str      # Title from **Paper**: line
    link: str
    arxiv_id: Optional[str]
    openreview_id: Optional[str]
    authors: str          # From **Authors**: line (if present)
    venue: str            # From **Venue**: line (if present)
    topic: str            # Section header this paper belongs to
    is_main: bool         # True if ### header, False if #### header

def extract_papers_from_markdown(filepath: str) -> List[Paper]:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    papers = []
    current_topic = "Uncategorized"
    
    # Find all section headers and their positions
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        # Track section topics (## headers)
        if line.startswith('## ') and not line.startswith('### '):
            current_topic = line.strip('# ').strip()
        
        # Main papers: ### X.Y Title
        if re.match(r'^### \d+\.\d+ ', line):
            paper = _parse_paper_block(lines, i, current_topic, is_main=True)
            if paper:
                papers.append(paper)
        
        # Sub-papers: #### X.Y.Z Title
        elif re.match(r'^#### \d+\.\d+\.\d+ ', line):
            paper = _parse_paper_block(lines, i, current_topic, is_main=False)
            if paper:
                papers.append(paper)
    
    return papers

def _parse_paper_block(lines, start_idx, topic, is_main):
    """Parse a paper block starting at the given line index."""
    header = lines[start_idx].strip()
    
    # Extract raw title from header
    raw_title = re.sub(r'^#+\s+\d+(\.\d+)+\s*', '', header)
    raw_title = raw_title.strip()
    
    # Find block end (next paper or section)
    end_idx = start_idx + 1
    while end_idx < len(lines):
        if re.match(r'^#{3,4} \d+', lines[end_idx]):
            break
        if lines[end_idx].startswith('---') and end_idx > start_idx + 1:
            end_idx += 1
            break
        end_idx += 1
    
    block = '\n'.join(lines[start_idx:end_idx])
    
    # Extract Paper title
    paper_match = re.search(r'\*\*Paper\*\*:\s*"([^"]+)"', block)
    paper_title = paper_match.group(1) if paper_match else raw_title
    
    # Extract Link
    link_match = re.search(r'\*\*Link\*\*:\s*(\S+)', block)
    link = link_match.group(1) if link_match else ''
    
    # Extract Authors
    authors_match = re.search(r'\*\*Authors\*\*:\s*([^
]+)', block)
    authors = authors_match.group(1).strip() if authors_match else ''
    
    # Extract Venue
    venue_match = re.search(r'\*\*Venue\*\*:\s*([^
]+)', block)
    venue = venue_match.group(1).strip() if venue_match else ''
    
    # Extract arXiv ID
    arxiv_id = None
    if 'arxiv.org' in link:
        m = re.search(r'arxiv\.org/(?:abs|pdf)/(\d{4,5}\.\d{4,5})', link)
        if m:
            arxiv_id = m.group(1)
    
    # Extract OpenReview ID
    openreview_id = None
    if 'openreview.net' in link:
        m = re.search(r'openreview\.net/forum\?id=([\w-]+)', link)
        if m:
            openreview_id = m.group(1)
    
    return Paper(
        raw_title=raw_title,
        paper_title=paper_title,
        link=link,
        arxiv_id=arxiv_id,
        openreview_id=openreview_id,
        authors=authors,
        venue=venue,
        topic=topic,
        is_main=is_main
    )

# Usage
papers = extract_papers_from_markdown('docs/related-work.md')
print(f"Found {len(papers)} papers")
for p in papers[:3]:
    print(f"  {p.paper_title} | arxiv={p.arxiv_id} | or={p.openreview_id}")
```

**Important regex details**:
- `^### \d+\.\d+ ` matches main papers like `### 10.1 Title`
- `^#### \d+\.\d+\.\d+ ` matches sub-papers like `#### 17.1.1 Title`
- `\*\*Paper\*\*:\s*"([^"]+)"` extracts the quoted paper title
- `\*\*Link\*\*:\s*(\S+)` extracts the URL
- `\*\*Venue\*\*:\s*([^
]+)` extracts the venue line (if present)

### 1.3 Extract Link Types

After extraction, categorize each paper by its link type. The parser must handle all common academic link patterns:

| Link Pattern                   | Type                | Example                                                  | Metadata Source |
|--------------------------------|---------------------|----------------------------------------------------------|-----------------|
| `arxiv.org/abs/XXXX.XXXXX`     | arXiv               | `https://arxiv.org/abs/2412.06769`                       | arXiv API       |
| `arxiv.org/pdf/XXXX.XXXXX.pdf` | arXiv (PDF)         | `https://arxiv.org/pdf/2412.06769.pdf`                   | arXiv API       |
| `openreview.net/forum?id=XXXX` | OpenReview          | `https://openreview.net/forum?id=CbK7lYbmv8`             | OpenReview API  |
| `aclanthology.org/XXXX`        | ACL Anthology       | `https://aclanthology.org/2025.acl-long.1369`            | Anthology URL   |
| `ojs.aaai.org/...`             | AAAI Proceedings    | `https://ojs.aaai.org/index.php/AAAI/article/view/40513` | AAAI OJS        |
| `proceedings.neurips.cc/...`   | NeurIPS Proceedings | `https://proceedings.neurips.cc/...`                     | Conference page |
| `proceedings.mlr.press/...`    | PMLR Proceedings    | `http://proceedings.mlr.press/v235/...`                  | PMLR page       |
| `ieeexplore.ieee.org/...`      | IEEE Xplore         | `https://ieeexplore.ieee.org/document/...`               | IEEE page       |
| `dl.acm.org/doi/...`           | ACM Digital Library | `https://dl.acm.org/doi/10.1145/...`                     | ACM DL          |
| `link.springer.com/...`        | Springer            | `https://link.springer.com/article/...`                  | Springer page   |
| `www.nature.com/...`           | Nature              | `https://www.nature.com/articles/...`                    | Nature page     |
| `www.science.org/...`          | Science             | `https://www.science.org/doi/...`                        | Science page    |
| `hal.science/...`              | HAL                 | `https://hal.science/hal-XXXXXX`                         | HAL API         |
| `proceedings.ijcai.org/...`    | IJCAI               | `https://www.ijcai.org/proceedings/...`                  | IJCAI page      |
| `github.com/...`               | Code only           | `https://github.com/org/repo`                            | Manual lookup   |
| Direct PDF (`.pdf`)            | PDF only            | `https://site.com/paper.pdf`                             | Manual lookup   |
| `Null` / missing               | No link             | `Null` or absent                                         | Search by title |

### 1.4 Robust arXiv ID Extraction

arXiv IDs have evolved over time. The parser must handle all formats:

```python
def extract_arxiv_id(url: str) -> Optional[str]:
    """Extract arXiv ID from various URL formats."""
    if not url or 'arxiv.org' not in url:
        return None
    
    # Modern format: 4-5 digits . 4-5 digits (with optional v1, v2 suffix)
    m = re.search(r'arxiv\.org/(?:abs|pdf)/(\d{4,5}\.\d{4,5})(?:v\d+)?', url)
    if m:
        return m.group(1)
    
    # Older format: archive/XXXX.XXXX (e.g., cs/0101001)
    m = re.search(r'arxiv\.org/(?:abs|pdf)/([a-z-]+/\d{7})', url)
    if m:
        return m.group(1)
    
    return None
```

### 1.5 Handling Missing or Partial Metadata

When `related-work.md` lacks structured fields, infer from context:

| Scenario               | Fallback Strategy                                            |
|------------------------|--------------------------------------------------------------|
| No `**Paper**:` line   | Use the markdown header title (after section number)         |
| No `**Link**:` line    | Mark as `Null`; search by title during Phase 3               |
| No `**Authors**:` line | Fetch from arXiv/OpenReview API; if unavailable, use `"TBD"` |
| No `**Venue**:` line   | Infer from arXiv comment, URL pattern, or header text        |
| `**Code**: Null`       | Omit `note` field about code availability                    |
| Link is GitHub-only    | Skip auto-metadata; manual entry required                    |

**Header title parsing**: The markdown header `### 10.1 Method Name (Venue Year)` often contains venue hints. Parse the parenthetical:

```python
def parse_header_hints(header: str) -> dict:
    """Extract venue and year hints from markdown headers."""
    hints = {}
    # Pattern: "Title (Venue Year)" or "Title (Year)"
    m = re.search(r'\(([^)]+)\)\s*$', header)
    if m:
        content = m.group(1)
        year_match = re.search(r'20(\d{2})', content)
        if year_match:
            hints['year'] = '20' + year_match.group(1)
        # Common venue abbreviations in headers
        venue_map = {
            'iclr': 'ICLR', 'icml': 'ICML', 'neurips': 'NeurIPS', 'nips': 'NeurIPS',
            'acl': 'ACL', 'emnlp': 'EMNLP', 'naacl': 'NAACL', 'eacl': 'EACL',
            'cvpr': 'CVPR', 'iccv': 'ICCV', 'eccv': 'ECCV',
            'aaai': 'AAAI', 'ijcai': 'IJCAI', 'coling': 'COLING',
            'colm': 'COLM', 'tmlr': 'TMLR', 'jmlr': 'JMLR',
        }
        content_lower = content.lower()
        for key, abbrev in venue_map.items():
            if key in content_lower:
                hints['venue'] = abbrev
                break
    return hints
```

---

## Phase 2: Batch Metadata Retrieval via arXiv API

### 2.1 Query the arXiv API

Use the arXiv Atom API to fetch metadata in bulk. Batch requests by 10 IDs at a time with a 3-second delay between batches to respect rate limits.

```python
import urllib.request
import xml.etree.ElementTree as ET
import time

def fetch_arxiv_metadata(arxiv_ids):
    """Fetch metadata for a list of arXiv IDs.
    Returns dict: id -> {'title': str, 'authors': list, 'comment': str, 'year': str}
    """
    results = {}
    for i in range(0, len(arxiv_ids), 10):
        batch = arxiv_ids[i:i+10]
        url = f'http://export.arxiv.org/api/query?id_list={",".join(batch)}'
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read().decode()
            root = ET.fromstring(data)
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            for entry in root.findall('atom:entry', ns):
                id_elem = entry.find('atom:id', ns)
                if id_elem is None:
                    continue
                aid = id_elem.text.split('/')[-1].replace('v1', '').replace('v2', '')
                title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                # Authors
                authors = []
                for author in entry.findall('atom:author', ns):
                    name = author.find('atom:name', ns)
                    if name is not None:
                        authors.append(name.text.strip())
                # Comment (often contains acceptance info)
                comment = entry.find('arxiv:comment', ns)
                comment_text = comment.text.strip() if comment is not None else ''
                # Published date -> year
                published = entry.find('atom:published', ns)
                year = published.text[:4] if published is not None else ''
                results[aid] = {
                    'title': title,
                    'authors': authors,
                    'comment': comment_text,
                    'year': year
                }
        except Exception as e:
            print(f'Error for batch {batch}: {e}')
        time.sleep(3)
    return results
```

### 2.2 Interpret arXiv Comments

arXiv comments are a **primary source** for venue information. Common patterns:

| Comment Pattern                                           | Meaning                                   |
|-----------------------------------------------------------|-------------------------------------------|
| `Accepted to ICML 2025`                                   | Accepted to ICML 2025 main conference     |
| `Accepted to NeurIPS 2025 (Spotlight)`                    | NeurIPS 2025 with spotlight presentation  |
| `Published in NeurIPS 2025 (Spotlight)`                   | Confirmed publication                     |
| `Accepted by NeurIPS 2024 D&B Track`                      | NeurIPS Datasets and Benchmarks track     |
| `Accepted to ICLR 2026`                                   | ICLR 2026 main conference                 |
| `LIT Workshop @ ICLR 2026`                                | Accepted to ICLR 2026 workshop            |
| `Long paper accepted to the main conference of AACL 2025` | AACL 2025 main conference                 |
| `Accepted as main paper at EACL 2026`                     | EACL 2026 main conference                 |
| `Accepted by CHI2025`                                     | CHI 2025                                  |
| `Published at COLM 2025`                                  | COLM 2025                                 |
| `Accepted by TMLR`                                        | Transactions on Machine Learning Research |
| `Under review as a conference paper at ICLR 2026`         | Still under review; no venue yet          |
| `Submitted to ICLR 2026`                                  | Submitted but decision unknown            |
| `Withdrawn`                                               | Withdrawn from submission                 |
| `Desk Rejected`                                           | Desk rejected                             |

**Action**: Parse comments automatically with regex, but always verify suspicious or high-stakes claims via web search.

### 2.3 Metadata Caching

To avoid re-fetching metadata every time (especially during iterative development), implement a simple JSON cache:

```python
import json
import os

def load_metadata_cache(cache_path='.bib_cache.json'):
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)
    return {}

def save_metadata_cache(cache, cache_path='.bib_cache.json'):
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)

def fetch_arxiv_metadata_with_cache(arxiv_ids, cache_path='.bib_cache.json'):
    cache = load_metadata_cache(cache_path)
    
    # Find IDs not in cache
    missing_ids = [aid for aid in arxiv_ids if aid not in cache]
    
    if missing_ids:
        print(f"Fetching {len(missing_ids)} new papers from arXiv API...")
        new_data = fetch_arxiv_metadata(missing_ids)  # from Section 2.1
        cache.update(new_data)
        save_metadata_cache(cache, cache_path)
    else:
        print("All metadata found in cache.")
    
    # Return only requested IDs
    return {aid: cache[aid] for aid in arxiv_ids if aid in cache}
```

**Benefits**:
- Re-running the script takes seconds instead of minutes
- Avoids API rate limit issues during development
- Cache can be manually edited to fix incorrect metadata
- `.bib_cache.json` should be in `.gitignore`

---

## Phase 2.5: Non-arXiv Paper Handling

Not all papers in `related-work.md` have arXiv links. For papers with OpenReview, ACL Anthology, or direct conference links, use alternative metadata sources.

### OpenReview Papers

For `openreview.net/forum?id=XXXX` links:

```python
import urllib.request
import json

def fetch_openreview_metadata(forum_id):
    """Fetch metadata from OpenReview API."""
    url = f'https://api.openreview.net/notes?id={forum_id}'
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        
        note = data['notes'][0]
        content = note['content']
        
        return {
            'title': content.get('title', {}).get('value', ''),
            'authors': content.get('authors', {}).get('value', []),
            'year': str(note.get('tcdate', ''))[:4] if note.get('tcdate') else '',
            'venue': _infer_venue_from_invitation(note.get('invitation', '')),
            'abstract': content.get('abstract', {}).get('value', '')
        }
    except Exception as e:
        print(f'Error fetching OpenReview {forum_id}: {e}')
        return None

def _infer_venue_from_invitation(invitation):
    """Infer venue from OpenReview invitation string."""
    if 'iclr' in invitation.lower():
        year_match = re.search(r'20\d{2}', invitation)
        year = year_match.group(0) if year_match else ''
        if 'workshop' in invitation.lower():
            return f'ICLR Workshop {year}'
        return f'ICLR {year}'
    elif ' neurips' in invitation.lower() or 'nips' in invitation.lower():
        year_match = re.search(r'20\d{2}', invitation)
        return f'NeurIPS {year_match.group(0)}' if year_match else 'NeurIPS'
    elif 'icml' in invitation.lower():
        year_match = re.search(r'20\d{2}', invitation)
        return f'ICML {year_match.group(0)}' if year_match else 'ICML'
    return 'Unknown'
```

### ACL Anthology Papers

For `aclanthology.org/XXXX` links, the URL itself contains venue information:

```python
def parse_acl_anthology_url(url):
    """Extract venue and year from ACL Anthology URL."""
    # Pattern: https://aclanthology.org/2025.acl-long.1369/
    m = re.search(r'aclanthology\.org/(\d{4})\.([\w-]+)\.([\w-]+)', url)
    if m:
        year = m.group(1)
        venue_code = m.group(2)
        
        # Map venue codes to full names
        venue_map = {
            'acl': 'ACL',
            'emnlp': 'EMNLP',
            'naacl': 'NAACL',
            'eacl': 'EACL',
            'aacl': 'AACL',
            'coling': 'COLING',
            'findings-acl': 'ACL Findings',
            'findings-emnlp': 'EMNLP Findings',
            'findings-naacl': 'NAACL Findings',
        }
        
        venue_name = venue_map.get(venue_code, venue_code.upper())
        return {'venue': f'{venue_name} {year}', 'year': year}
    return None
```

### Direct Conference / Journal Links

For papers with links to conference proceedings or journal pages:

```python
def infer_venue_from_url(url):
    """Infer venue from URL patterns when arXiv/OpenReview/Anthology are not available."""
    url_lower = url.lower()
    
    patterns = [
        (r'proceedings\.neurips\.cc', 'NeurIPS', r'/(\d{4})/'),
        (r'icml\.cc', 'ICML', r'/(\d{4})/'),
        (r'iclr\.cc', 'ICLR', r'/(\d{4})/'),
        (r'cvpr\.thecvf\.com', 'CVPR', r'/(\d{4})/'),
        (r'aaai\.org', 'AAAI', r'/(\d{4})/'),
        (r'ojs\.aaai\.org', 'AAAI', r'/(\d{4})/'),
        (r'tmlr\.org', 'TMLR', None),
    ]
    
    for pattern, venue, year_pattern in patterns:
        if re.search(pattern, url_lower):
            year = ''
            if year_pattern:
                year_match = re.search(year_pattern, url)
                if year_match:
                    year = year_match.group(1)
            return {'venue': f'{venue} {year}'.strip(), 'year': year}
    
    return None
```

---

## Phase 3: Deep Venue Verification (Multi-Source Strategy)

**Never rely solely on arXiv comments.** For every paper without a clear, trusted venue, perform deep verification using the following sources in priority order.

### 3.1 Source Hierarchy

| Priority | Source                             | URL Pattern                                                     | Best For                                           |
|----------|------------------------------------|-----------------------------------------------------------------|----------------------------------------------------|
| 1        | **OpenReview**                     | `https://openreview.net/forum?id=XXXX`                          | ICLR, NeurIPS, ICML, COLM, ACL workshops           |
| 2        | **ACL Anthology**                  | `https://aclanthology.org/`                                     | ACL, EMNLP, NAACL, EACL, COLING, Findings          |
| 3        | **PMLR Proceedings**               | `http://proceedings.mlr.press/`                                 | ICML papers                                        |
| 4        | **NeurIPS Proceedings**            | `https://neurips.cc/virtual/YYYY/poster/XXX`                    | NeurIPS papers                                     |
| 5        | **DBLP**                           | `https://dblp.org/`                                             | Computer science bibliographic data                |
| 6        | **Google Scholar**                 | Search by title                                                 | Cross-checking venues, finding proceedings entries |
| 7        | **Author Personal Websites**       | Search "[First Author] homepage"                                | CVPR, ICCV, ECCV, personal confirmations           |
| 8        | **HuggingFace Papers / CatalyzeX** | `https://huggingface.co/papers` or `https://www.catalyzex.com/` | ML community-sourced venue labels                  |
| 9        | **arXiv Listing Pages**            | `https://arxiv.org/list/cs.CL/YYYY?skip=N`                      | Comments not shown on abstract page                |
| 10       | **LinkedIn / X (Twitter)**         | Search paper title + "accepted"                                 | Social confirmations from authors                  |

### 3.2 OpenReview Verification Protocol

OpenReview is the most reliable source for ML conferences. For each paper:

1. Search OpenReview with the exact title: `https://openreview.net/search?term=[TITLE]&group=all&content=all&source=all`
2. If found, check the **forum page** for:
   - `Accepted` status and exact venue (e.g., `ICLR 2026 Poster`)
   - Workshop acceptance (e.g., `LIT Workshop @ ICLR 2026`)
   - `Submitted` / `Under Review` / `Withdrawn` / `Desk Rejected`
3. For accepted papers, the PDF may contain a footer like `Published as a conference paper at ICLR 2026`.

**Important**: Workshop acceptances at ICLR/NeurIPS/ICML are listed as `[WorkshopName] @ [Conference] [Year]`. These are `@inproceedings` but the `booktitle` should be `Proc.~[Conference] Workshop on [WorkshopName]`.

### 3.3 ACL Anthology Verification Protocol

For NLP papers, ACL Anthology is the canonical source:

1. Search: `https://aclanthology.org/search/?q=[title]`
2. Look for exact title match in the results.
3. The anthology URL reveals the venue: e.g., `2025.acl-long.1369` = ACL 2025 long paper.
4. For Findings: `2025.findings-emnlp.XXX` = EMNLP Findings 2025.

### 3.4 Conference Virtual Pages

For recent conferences (2024–2026), check the virtual conference pages:

- **ICLR**: `https://iclr.cc/virtual/2026/poster/XXX` or `https://iclr.cc/virtual/2026/events/oral`
- **NeurIPS**: `https://neurips.cc/virtual/2025/poster/XXX`
- **CVPR**: `https://cvpr.thecvf.com/virtual/2026/papers.html`
- **ICML**: `https://icml.cc/virtual/2025/poster/XXX`

Search these pages for the paper title.

### 3.5 Author Website Verification

When other sources conflict or are missing, check the first author's personal website:

1. Search: `"[FirstName] [LastName]" homepage publications`
2. Look for the paper in their publication list; venues are often explicitly stated.
3. Example: Jibin Wu's website explicitly lists `ReLaX: ... CVPR 2026`.

### 3.6 Resolving Conflicts

When sources conflict (e.g., arXiv comment says "NAACL 2025" but date is March 2026):

1. Prefer **official proceedings / anthology** over arXiv comments.
2. Prefer **author website** over social media.
3. Prefer **OpenReview official status** over blog posts.
4. If a paper's arXiv date is **after** the claimed conference date, the claim is likely wrong.
5. When in doubt, **keep as arXiv-only** rather than guessing.

---

## Phase 4: BibTeX Formatting Standards

### 4.1 Entry Type Selection

| Type             | When to Use                          | Required Fields                        | Optional Fields                                  |
|------------------|--------------------------------------|----------------------------------------|--------------------------------------------------|
| `@inproceedings` | Conference papers, workshop papers   | `title`, `author`, `booktitle`, `year` | `pages`, `volume`, `publisher`, `address`, `url` |
| `@article`       | Journal papers, arXiv-only preprints | `title`, `author`, `journal`, `year`   | `volume`, `number`, `pages`, `publisher`, `url`  |
| `@book`          | Books, edited volumes                | `title`, `author`, `publisher`, `year` | `volume`, `series`, `address`                    |

**Never** use `@misc` or `@conference`.

### 4.2 Citation Key Naming Convention

Format: `MethodAbbreviation-VenueYear`

Rules:
- `MethodAbbreviation`: A short, memorable abbreviation of the paper's method name.
  - Extract capitalized words from the title: `Chain of Thought` → `CoT`
  - Use well-known abbreviations: `LLaMA`, `GPT`, `BERT`, `T5`
  - For single-word methods, use the full word: `Coconut`, `Huginn`
  - For numbered methods, preserve the number: `System15`, `GPT4`
  - Avoid generic words: `Learning`, `Neural`, `Deep`, `Model`, `Network`
- `VenueYear`: The venue abbreviation + last two digits of year.
  - NeurIPS 2025 → `NeurIPS25`
  - ICLR 2026 → `ICLR26`
  - EMNLP Findings 2025 → `EMNLPFindings25`
  - ICLR Workshop 2026 → `ICLRW26`
  - arXiv-only 2025 → `arXiv25`
  - TMLR (journal) → `TMLR25`
  - IEEE TPAMI (journal) → `TPAMI25`

**Conflict resolution**: If two papers would have the same key, append a distinguishing letter:
```
LLaDA-NeurIPS25
LLaDA-ICLR26      # If a second LLaDA paper exists
LLaDA-ICLR26b     # If a third exists
```

Examples:
```
Coconut-COLM25
VAR-NeurIPS24
DiffGuidedLM-ACLFindings24
DLCM-ICLRW26
GDM-Science25
```

### 4.3 Conference `booktitle` Format

All conference `booktitle` fields **must** use the `Proc.~` prefix followed by the full conference name:

```bibtex
booktitle = {Proc.~Conference on Neural Information Processing Systems},
booktitle = {Proc.~International Conference on Learning Representations},
booktitle = {Proc.~International Conference on Machine Learning},
booktitle = {Proc.~Conference on Empirical Methods in Natural Language Processing},
booktitle = {Proc.~Annual Meeting of the Association for Computational Linguistics},
booktitle = {Proc.~Annual Meeting of the Association for Computational Linguistics Findings},
booktitle = {Proc.~Conference on Computer Vision and Pattern Recognition},
booktitle = {Proc.~Conference on Language Modeling},
booktitle = {Proc.~AAAI Conference on Artificial Intelligence},
booktitle = {Proc.~ICLR Workshop on Latent and Implicit Thinking},
booktitle = {Proc.~Asia-Pacific Chapter of the Association for Computational Linguistics},
```

### 4.4 Journal / arXiv `journal` Format

For `@article` entries:

```bibtex
journal = {arXiv preprint arXiv:XXXX.XXXXX},   % arXiv-only
journal = {Transactions on Machine Learning Research},  % TMLR
journal = {Journal of Machine Learning Research},  % JMLR
journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},  % TPAMI
journal = {International Journal of Computer Vision},  % IJCV
journal = {Nature},  % Nature
journal = {Science},  % Science
```

### 4.5 arXiv-Specific Fields

For arXiv-only papers, include `eprint` and `archivePrefix` for compatibility with BibLaTeX and modern LaTeX workflows:

```bibtex
@article{Coconut-arXiv24,
  title = {Chain of Continuous Thought},
  author = {Ding, Nan and others},
  journal = {arXiv preprint arXiv:2412.01831},
  year = {2024},
  eprint = {2412.01831},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2412.01831},
}
```

**Benefits**:
- `eprint` + `archivePrefix` enables BibLaTeX to render `arXiv:2412.01831 [cs.CL]` automatically
- `url` provides a clickable link in both BibTeX and BibLaTeX
- This format is the de-facto standard for arXiv entries in ML papers

### 4.6 Author Formatting

- Use `and` as separator: `author = {Alice Smith and Bob Jones},`
- Include full first names when available (not just initials).
- Preserve special characters: `Loïc Barrault`, `José`, `Müller`.
- For large consortium papers, include the named team if present: `LCM team and Loïc Barrault and ...`

### 4.7 Title Formatting

- Preserve original capitalization (do not force sentence case).
- Preserve special LaTeX symbols: `\'e`, `$\nabla$`, etc.
- Remove line breaks from arXiv API responses.

### 4.8 File Section Comments

Group papers by topic with clear section headers:

```bibtex
% ============================================================
% Continuous Latent Space Reasoning
% ============================================================

% Paper_Name_Short_Description (arXiv:XXXX.XXXXX)
@inproceedings{Key-VenueYear,
  ...
}
```

The arXiv ID comment above each entry serves as a quick reference for manual verification.

---

## Phase 5: Automated Validation

After generating or updating the `.bib` file, run a validation script to catch errors.

### 5.1 Validation Script

```python
import re

def validate_bib(filepath):
    with open(filepath) as f:
        content = f.read()

    entries = list(re.finditer(r'@(article|inproceedings)\{([^,]+),', content))
    print(f'Total entries: {len(entries)}')

    article_count = sum(1 for e in entries if e.group(1) == 'article')
    inproc_count = sum(1 for e in entries if e.group(1) == 'inproceedings')
    print(f'@article: {article_count}, @inproceedings: {inproc_count}')

    errors = []

    # Check missing authors
    for m in re.finditer(r'@(article|inproceedings)\{([^,]+),', content):
        block_end = content.find('@', m.end())
        if block_end == -1:
            block_end = len(content)
        block = content[m.start():block_end]
        if 'author' not in block:
            errors.append(f'MISSING AUTHOR: {m.group(2)}')

    # Check year mismatches (year in key vs year field)
    for m in re.finditer(r'@(article|inproceedings)\{([^,]+),', content):
        key = m.group(2)
        block_end = content.find('@', m.end())
        if block_end == -1:
            block_end = len(content)
        block = content[m.start():block_end]
        year_match = re.search(r'year\s*=\s*\{(\d{4})\}', block)
        if year_match:
            year = year_match.group(1)
            # Extract year from last hyphen-separated segment
            key_year = None
            if '-' in key:
                last_part = key.split('-')[-1]
                m2 = re.search(r'(\d{2})', last_part)
                if m2:
                    key_year = '20' + m2.group(1)
            if key_year and key_year != year:
                errors.append(f'YEAR MISMATCH: {key} -> key year {key_year}, field year {year}')

    # Check Proc.~ prefix for inproceedings
    for m in re.finditer(r'@inproceedings\{([^,]+),', content):
        key = m.group(1)
        block_end = content.find('@', m.end())
        if block_end == -1:
            block_end = len(content)
        block = content[m.start():block_end]
        bt_match = re.search(r'booktitle\s*=\s*\{([^}]+)\}', block)
        if bt_match:
            bt = bt_match.group(1)
            if not bt.startswith('Proc.~'):
                errors.append(f'MISSING Proc.~ PREFIX: {key} -> {bt}')

    if errors:
        print('\n'.join(errors))
    else:
        print('All checks passed!')
    return len(errors) == 0

# Run it
validate_bib('docs/related-work.bib')
```

### 5.2 Manual Spot Checks

For high-profile or suspicious papers, always do a manual spot check:

1. Open the paper's arXiv page and verify the title matches exactly.
2. Check the author list against the PDF.
3. If a venue is claimed, open the conference proceedings/anthology and search for the title.

---

## Phase 5.5: Duplicate Detection

Papers may appear in multiple sections of `related-work.md` (e.g., both Section 10 and Section 14). Detect and deduplicate before generating BibTeX:

```python
def detect_duplicates(papers):
    """Detect duplicate papers by arXiv ID, OpenReview ID, or title similarity."""
    seen = {}
    duplicates = []
    unique_papers = []
    
    for p in papers:
        # Primary key: arXiv ID
        if p.arxiv_id and p.arxiv_id in seen:
            duplicates.append((p, seen[p.arxiv_id]))
            continue
        # Secondary key: OpenReview ID
        if p.openreview_id and p.openreview_id in seen:
            duplicates.append((p, seen[p.openreview_id]))
            continue
        # Tertiary key: normalized title
        norm_title = re.sub(r'[^\w]', '', p.paper_title.lower())
        if norm_title in seen:
            duplicates.append((p, seen[norm_title]))
            continue
        
        # Record seen
        if p.arxiv_id:
            seen[p.arxiv_id] = p
        if p.openreview_id:
            seen[p.openreview_id] = p
        seen[norm_title] = p
        unique_papers.append(p)
    
    if duplicates:
        print(f"WARNING: Found {len(duplicates)} duplicate papers:")
        for dup, orig in duplicates:
            print(f"  DUPLICATE: {dup.paper_title}")
            print(f"    Original: Section '{orig.topic}'")
            print(f"    Duplicate: Section '{dup.topic}'")
    
    return unique_papers
```

**Handling strategy**: Keep the first occurrence (usually the more detailed entry) and skip duplicates.

---

## Phase 5.6: Incremental Update Strategy

When `related-work.md` is updated (new papers added), you don't want to regenerate the entire `.bib` file from scratch — you might lose manual fixes.

```python
def merge_bib_files(existing_path, new_entries, output_path):
    """Merge new BibTeX entries with existing file, preserving manual edits."""
    import re
    
    # Parse existing entries
    existing_entries = {}
    if os.path.exists(existing_path):
        with open(existing_path, 'r') as f:
            content = f.read()
        
        for m in re.finditer(r'@(article|inproceedings)\{([^,]+),', content):
            key = m.group(2)
            block_end = content.find('@', m.end())
            if block_end == -1:
                block_end = len(content)
            existing_entries[key] = content[m.start():block_end]
    
    # Merge: new entries override existing ones with same key
    merged = dict(existing_entries)
    for entry in new_entries:
        key_match = re.search(r'@(article|inproceedings)\{([^,]+),', entry)
        if key_match:
            key = key_match.group(2)
            merged[key] = entry
    
    # Write merged file
    with open(output_path, 'w') as f:
        f.write('% Auto-generated from docs/related-work.md\n')
        f.write('% Manual edits may be preserved during incremental updates\n\n')
        for entry in merged.values():
            f.write(entry.strip() + '\n\n')
    
    print(f"Wrote {len(merged)} entries to {output_path}")
    print(f"  New: {len(new_entries)}")
    print(f"  Preserved from existing: {len(existing_entries) - len(set(existing_entries) & set(merged))}")
```

**Best practice**: Run the full pipeline periodically (e.g., weekly), but use incremental merge for daily updates.

---

## Phase 5.7: Error Handling and Recovery

### Common Errors and Solutions

| Error                                     | Cause                                      | Solution                                                |
|-------------------------------------------|--------------------------------------------|---------------------------------------------------------|
| `HTTP Error 503` from arXiv API           | Rate limit exceeded                        | Increase `time.sleep()` to 5-10 seconds between batches |
| `XML Parse Error`                         | arXiv API returned malformed XML           | Retry the batch; if persistent, fetch IDs individually  |
| `Connection Timeout`                      | Network instability                        | Add retry logic with exponential backoff                |
| Empty metadata for valid arXiv ID         | Paper withdrawn or ID incorrect            | Skip the paper and log a warning                        |
| OpenReview API returns 404                | Forum ID incorrect or paper private        | Fallback to arXiv metadata or manual lookup             |
| Title mismatch between arXiv and markdown | arXiv title updated after markdown written | Use arXiv title as authoritative source                 |

### Robust Fetching with Retries

```python
import time
import urllib.error

def fetch_with_retries(url, max_retries=3, timeout=30):
    """Fetch URL with exponential backoff on failure."""
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read().decode()
        except urllib.error.HTTPError as e:
            if e.code == 503 and attempt < max_retries - 1:
                wait = 2 ** attempt  # 1, 2, 4 seconds
                print(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            raise
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  Error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise
    return None
```

---

## Phase 6: Complete End-to-End Script

Here is a complete, production-ready Python script that processes any `related-work.md` and produces a LaTeX-ready `related-work.bib`:

```python
#!/usr/bin/env python3
"""
Generate related-work.bib from related-work.md
Usage: python generate_bib.py docs/related-work.md docs/related-work.bib

This script is GENERIC — it works with any related-work.md that follows
standard markdown conventions (### headers for papers, **Paper**: lines, etc.)
"""

import re
import os
import sys
import json
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple

# ============================================================
# CONFIGURATION
# ============================================================
CACHE_PATH = '.bib_cache.json'
ARXIV_BATCH_SIZE = 10
ARXIV_DELAY = 3  # seconds between batches
MAX_RETRIES = 3

# Full venue name mapping for booktitle generation
VENUE_FULL_NAMES = {
    'NeurIPS': 'Conference on Neural Information Processing Systems',
    'ICML': 'International Conference on Machine Learning',
    'ICLR': 'International Conference on Learning Representations',
    'AAAI': 'AAAI Conference on Artificial Intelligence',
    'IJCAI': 'International Joint Conference on Artificial Intelligence',
    'ACL': 'Annual Meeting of the Association for Computational Linguistics',
    'ACLFindings': 'Annual Meeting of the Association for Computational Linguistics Findings',
    'EMNLP': 'Conference on Empirical Methods in Natural Language Processing',
    'EMNLPFindings': 'Conference on Empirical Methods in Natural Language Processing Findings',
    'NAACL': 'Conference of the North American Chapter of the Association for Computational Linguistics',
    'EACL': 'Conference of the European Chapter of the Association for Computational Linguistics',
    'AACL': 'Asia-Pacific Chapter of the Association for Computational Linguistics',
    'COLING': 'International Conference on Computational Linguistics',
    'COLM': 'Conference on Language Modeling',
    'CVPR': 'Conference on Computer Vision and Pattern Recognition',
    'ICCV': 'IEEE International Conference on Computer Vision',
    'ECCV': 'European Conference on Computer Vision',
    'WACV': 'Winter Conference on Applications of Computer Vision',
    'BMVC': 'British Machine Vision Conference',
    'TMLR': 'Transactions on Machine Learning Research',
    'JMLR': 'Journal of Machine Learning Research',
    'TPAMI': 'IEEE Transactions on Pattern Analysis and Machine Intelligence',
    'IJCV': 'International Journal of Computer Vision',
    'TACL': 'Transactions of the Association for Computational Linguistics',
}

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Paper:
    raw_title: str
    paper_title: str
    link: str
    arxiv_id: Optional[str]
    openreview_id: Optional[str]
    authors: str
    venue_hint: str
    code_link: str
    topic: str
    is_main: bool
    header_hints: Dict = field(default_factory=dict)

@dataclass
class BibEntry:
    key: str
    entry_type: str
    title: str
    authors: str
    year: str
    venue: str
    booktitle: Optional[str]
    journal: Optional[str]
    url: str
    eprint: Optional[str]
    archive_prefix: Optional[str]
    note: Optional[str]
    topic: str

# ============================================================
# PHASE 1: ROBUST EXTRACTION
# ============================================================

def extract_papers_from_markdown(filepath: str) -> List[Paper]:
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    papers = []
    current_topic = "Uncategorized"
    for i, line in enumerate(lines):
        line = line.rstrip('\n')
        if line.startswith('## ') and not line.startswith('### '):
            current_topic = line.strip('# ').strip()
        if re.match(r'^### \d+\.\d+ ', line):
            paper = _parse_paper_block(lines, i, current_topic, is_main=True)
            if paper:
                papers.append(paper)
        elif re.match(r'^#### \d+\.\d+\.\d+ ', line):
            paper = _parse_paper_block(lines, i, current_topic, is_main=False)
            if paper:
                papers.append(paper)
    return papers

def _parse_paper_block(lines, start_idx, topic, is_main):
    header = lines[start_idx].strip()
    raw_title = re.sub(r'^#+\s+\d+(\.\d+)+\s*', '', header).strip()
    header_hints = parse_header_hints(raw_title)
    end_idx = start_idx + 1
    while end_idx < len(lines):
        if re.match(r'^#{3,4} \d+', lines[end_idx]):
            break
        end_idx += 1
    block = ''.join(lines[start_idx:end_idx])
    paper_match = re.search(r'\*\*Paper\*\*:\s*"([^"]+)"', block)
    paper_title = paper_match.group(1) if paper_match else raw_title
    link_match = re.search(r'\*\*Link\*\*:\s*(\S+)', block)
    link = link_match.group(1) if link_match else ''
    authors_match = re.search(r'\*\*Authors\*\*:\s*([^\n]+)', block)
    authors = authors_match.group(1).strip() if authors_match else ''
    venue_match = re.search(r'\*\*Venue\*\*:\s*([^\n]+)', block)
    venue_hint = venue_match.group(1).strip() if venue_match else ''
    code_match = re.search(r'\*\*Code\*\*:\s*(\S+)', block)
    code_link = code_match.group(1) if code_match else ''
    arxiv_id = extract_arxiv_id(link)
    openreview_id = None
    if 'openreview.net' in link:
        m = re.search(r'openreview\.net/forum\?id=([\w-]+)', link)
        openreview_id = m.group(1) if m else None
    return Paper(
        raw_title=raw_title, paper_title=paper_title, link=link,
        arxiv_id=arxiv_id, openreview_id=openreview_id,
        authors=authors, venue_hint=venue_hint, code_link=code_link,
        topic=topic, is_main=is_main, header_hints=header_hints
    )

def parse_header_hints(header: str) -> Dict:
    hints = {}
    m = re.search(r'\(([^)]+)\)\s*$', header)
    if m:
        content = m.group(1)
        year_match = re.search(r'20(\d{2})', content)
        if year_match:
            hints['year'] = '20' + year_match.group(1)
        venue_map = {
            'iclr': 'ICLR', 'icml': 'ICML', 'neurips': 'NeurIPS', 'nips': 'NeurIPS',
            'acl': 'ACL', 'emnlp': 'EMNLP', 'naacl': 'NAACL', 'eacl': 'EACL',
            'cvpr': 'CVPR', 'iccv': 'ICCV', 'eccv': 'ECCV',
            'aaai': 'AAAI', 'ijcai': 'IJCAI', 'coling': 'COLING',
            'colm': 'COLM', 'tmlr': 'TMLR', 'jmlr': 'JMLR',
        }
        content_lower = content.lower()
        for key, abbrev in venue_map.items():
            if key in content_lower:
                hints['venue'] = abbrev
                break
    return hints

def extract_arxiv_id(url: str) -> Optional[str]:
    if not url or 'arxiv.org' not in url:
        return None
    m = re.search(r'arxiv\.org/(?:abs|pdf)/(\d{4,5}\.\d{4,5})(?:v\d+)?', url)
    if m:
        return m.group(1)
    m = re.search(r'arxiv\.org/(?:abs|pdf)/([a-z-]+/\d{7})', url)
    if m:
        return m.group(1)
    return None

# ============================================================
# PHASE 2: METADATA FETCHING WITH RETRIES
# ============================================================

def load_cache(path=CACHE_PATH) -> Dict:
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_cache(cache, path=CACHE_PATH):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)

def fetch_with_retries(url: str, max_retries=MAX_RETRIES, timeout=30) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read().decode('utf-8')
        except urllib.error.HTTPError as e:
            if e.code == 503 and attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"    Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"    HTTP {e.code} for {url}")
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"    Error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
                continue
            print(f"    Failed after {max_retries} attempts: {url}")
            return None
    return None

def fetch_arxiv_metadata(arxiv_ids: List[str]) -> Dict:
    cache = load_cache()
    missing = [aid for aid in arxiv_ids if aid not in cache]
    if missing:
        print(f"  Fetching {len(missing)} papers from arXiv API...")
    for i in range(0, len(missing), ARXIV_BATCH_SIZE):
        batch = missing[i:i + ARXIV_BATCH_SIZE]
        url = f'http://export.arxiv.org/api/query?id_list={",".join(batch)}'
        data = fetch_with_retries(url)
        if data is None:
            continue
        try:
            root = ET.fromstring(data)
        except ET.ParseError as e:
            print(f"    XML parse error: {e}")
            continue
        ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
        for entry in root.findall('atom:entry', ns):
            id_elem = entry.find('atom:id', ns)
            if id_elem is None:
                continue
            aid = id_elem.text.split('/')[-1]
            aid = re.sub(r'v\d+$', '', aid)
            title_elem = entry.find('atom:title', ns)
            title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else ''
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns)
                if name is not None and name.text:
                    authors.append(name.text.strip())
            comment = entry.find('arxiv:comment', ns)
            comment_text = comment.text.strip() if comment is not None else ''
            published = entry.find('atom:published', ns)
            year = published.text[:4] if published is not None else ''
            cache[aid] = {'title': title, 'authors': authors, 'comment': comment_text, 'year': year}
        save_cache(cache)
        if i + ARXIV_BATCH_SIZE < len(missing):
            time.sleep(ARXIV_DELAY)
    return {aid: cache[aid] for aid in arxiv_ids if aid in cache}

# ============================================================
# PHASE 3: VENUE INFERENCE
# ============================================================

def infer_venue_from_comment(comment: str) -> Tuple[Optional[str], bool]:
    if not comment:
        return None, False
    comment_lower = comment.lower()
    if any(k in comment_lower for k in ['withdrawn', 'desk rejected']):
        return 'arXiv', False
    if any(k in comment_lower for k in ['under review', 'submitted to', 'submitted at']):
        return None, False
    venue_patterns = [
        (r'(?:accepted|published)\s+(?:to|at|by|in)\s+([\w\s&]+?)(?:\s+\d{4})?', False),
        (r'([\w\s]+?)\s+workshop\s*@?\s*([\w\s]+?)(?:\s+\d{4})?', True),
        (r'([\w\s]+?)\s+@\s*([\w\s]+?)(?:\s+\d{4})?', True),
    ]
    for pattern, is_workshop in venue_patterns:
        m = re.search(pattern, comment_lower, re.IGNORECASE)
        if m:
            venue = ' '.join(g.strip().title() for g in m.groups() if g and g.strip())
            abbrev = map_venue_to_abbrev(venue)
            return abbrev, is_workshop
    return None, False

def map_venue_to_abbrev(venue_str: str) -> str:
    venue_lower = venue_str.lower()
    mapping = {
        'neural information processing systems': 'NeurIPS',
        'international conference on learning representations': 'ICLR',
        'international conference on machine learning': 'ICML',
        'empirical methods in natural language processing': 'EMNLP',
        'association for computational linguistics': 'ACL',
        'conference on language modeling': 'COLM',
        'computer vision and pattern recognition': 'CVPR',
        'aaai conference on artificial intelligence': 'AAAI',
        'international joint conference on artificial intelligence': 'IJCAI',
        'north american chapter': 'NAACL',
        'european chapter': 'EACL',
        'asia-pacific chapter': 'AACL',
        'computational linguistics': 'COLING',
        'machine learning research': 'TMLR',
    }
    for key, abbrev in mapping.items():
        if key in venue_lower:
            return abbrev
    return venue_str.strip()

def infer_venue_from_url(url: str) -> Optional[str]:
    if not url or url.lower() == 'null':
        return None
    url_lower = url.lower()
    patterns = [
        (r'aclanthology\.org/(\d{4})\.([\w-]+)', 'anthology'),
        (r'openreview\.net', 'OpenReview'),
        (r'ojs\.aaai\.org', 'AAAI'),
        (r'proceedings\.neurips\.cc', 'NeurIPS'),
        (r'proceedings\.mlr\.press', 'ICML'),
        (r'iclr\.cc', 'ICLR'),
        (r'cvpr\.thecvf\.com', 'CVPR'),
        (r'tmlr\.org', 'TMLR'),
        (r'ieeexplore\.ieee\.org', 'IEEE'),
        (r'dl\.acm\.org', 'ACM'),
    ]
    for pattern, source in patterns:
        if re.search(pattern, url_lower):
            return source
    return None

# ============================================================
# PHASE 4: CITATION KEY & BIBTEX GENERATION
# ============================================================

_used_keys: Set[str] = set()

def generate_citation_key(paper: Paper, venue: str, year: str) -> str:
    global _used_keys
    method = extract_method_abbreviation(paper.paper_title)
    venue_part = venue if venue else 'arXiv'
    year_short = year[-2:] if year and len(year) >= 2 else 'XX'
    base_key = f"{method}-{venue_part}{year_short}"
    key = base_key
    suffix = 'a'
    while key in _used_keys:
        key = f"{base_key}{suffix}"
        suffix = chr(ord(suffix) + 1)
    _used_keys.add(key)
    return key

def extract_method_abbreviation(title: str) -> str:
    title = re.split(r'[:\(\-]', title)[0].strip()
    words = re.findall(r'[A-Z][a-z]+|[A-Z]{2,}', title)
    if words:
        generic = {'A', 'An', 'The', 'On', 'In', 'For', 'Of', 'And', 'Or', 'With',
                   'To', 'From', 'By', 'Using', 'Via', 'Towards', 'Beyond',
                   'Learning', 'Neural', 'Deep', 'Model', 'Network', 'Models',
                   'Networks', 'Based', 'Approach', 'Framework', 'Method'}
        filtered = [w for w in words if w not in generic]
        if filtered:
            abbrev = ''.join(w[:3] if len(w) > 3 else w for w in filtered[:3])
            if len(abbrev) >= 3:
                return abbrev
    acronyms = re.findall(r'\b[A-Z]{2,6}\b', title)
    if acronyms:
        return acronyms[0]
    all_words = re.findall(r'[A-Za-z]+', title)
    meaningful = [w for w in all_words if w.lower() not in
                  {'a', 'an', 'the', 'on', 'in', 'for', 'of', 'and', 'or', 'with'}]
    if meaningful:
        return meaningful[0][:8]
    return 'Paper'

def escape_bibtex(text: str) -> str:
    if not text:
        return text
    replacements = [
        ('\\', '\\textbackslash{}'),
        ('&', '\\&'),
        ('%', '\\%'),
        ('$', '\\$'),
        ('#', '\\#'),
        ('_', '\\_'),
        ('{', '\\{'),
        ('}', '\\}'),
        ('~', '\\textasciitilde{}'),
        ('^', '\\textasciicircum{}'),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text

def format_authors(authors_input) -> str:
    if not authors_input:
        return 'TBD'
    if isinstance(authors_input, list):
        authors = authors_input
    else:
        if ' and ' in authors_input:
            authors = [a.strip() for a in authors_input.split(' and ')]
        else:
            authors = [a.strip() for a in authors_input.split(',')]
    formatted = []
    for author in authors:
        author = author.strip()
        if not author:
            continue
        if ',' in author:
            formatted.append(author)
            continue
        parts = author.split()
        if len(parts) >= 2:
            first_names = ' '.join(parts[:-1])
            last_name = parts[-1]
            formatted.append(f"{last_name}, {first_names}")
        else:
            formatted.append(author)
    return ' and '.join(formatted) if formatted else 'TBD'

def build_booktitle(venue: str) -> Optional[str]:
    if not venue or venue == 'arXiv':
        return None
    m = re.match(r'([A-Z]+)W(\d{2})', venue)
    if m:
        base = m.group(1)
        return f"Proc.~{base} Workshop"
    if 'Findings' in venue:
        base = venue.replace('Findings', '')
        full = VENUE_FULL_NAMES.get(venue, VENUE_FULL_NAMES.get(base, base))
        return f"Proc.~{full}"
    full = VENUE_FULL_NAMES.get(venue, venue)
    return f"Proc.~{full}"

def generate_bibtex(entry: BibEntry) -> str:
    lines = ['@' + entry.entry_type + '{' + entry.key + ',']
    lines.append(f"  title = {{{escape_bibtex(entry.title)}}},")
    lines.append(f"  author = {{{entry.authors}}},")
    lines.append(f"  year = {{{entry.year}}},")
    if entry.entry_type == 'inproceedings' and entry.booktitle:
        lines.append(f"  booktitle = {{{entry.booktitle}}},")
    elif entry.entry_type == 'article' and entry.journal:
        lines.append(f"  journal = {{{entry.journal}}},")
    if entry.eprint:
        lines.append(f"  eprint = {{{entry.eprint}}},")
    if entry.archive_prefix:
        lines.append(f"  archivePrefix = {{{entry.archive_prefix}}},")
    if entry.url and entry.url.lower() != 'null':
        lines.append(f"  url = {{{entry.url}}},")
    if entry.note:
        lines.append(f"  note = {{{entry.note}}},")
    lines.append("}")
    return '\n'.join(lines)

# ============================================================
# MAIN PIPELINE
# ============================================================

def main(md_path, bib_path):
    global _used_keys
    _used_keys = set()
    print(f"=== Reading {md_path} ===")
    papers = extract_papers_from_markdown(md_path)
    print(f"Found {len(papers)} papers")
    if not papers:
        print("No papers found! Check your markdown format.")
        sys.exit(1)
    unique_papers = []
    seen: Dict[str, Paper] = {}
    for p in papers:
        key = p.arxiv_id or p.openreview_id or re.sub(r'[^\w]', '', p.paper_title.lower())
        if key not in seen:
            seen[key] = p
            unique_papers.append(p)
        else:
            print(f"  Duplicate skipped: {p.paper_title[:60]}...")
    print(f"After dedup: {len(unique_papers)} papers")
    arxiv_ids = [p.arxiv_id for p in unique_papers if p.arxiv_id]
    metadata = {}
    if arxiv_ids:
        print(f"Fetching metadata for {len(arxiv_ids)} arXiv papers...")
        metadata = fetch_arxiv_metadata(arxiv_ids)
        print(f"  Retrieved metadata for {len(metadata)} papers")
    entries = []
    missing_metadata = []
    for paper in unique_papers:
        meta = metadata.get(paper.arxiv_id, {}) if paper.arxiv_id else {}
        title = meta.get('title', paper.paper_title)
        if not title:
            title = paper.raw_title
        authors = meta.get('authors', [])
        if not authors and paper.authors:
            authors = paper.authors
        year = meta.get('year', '')
        if not year:
            year = paper.header_hints.get('year', '')
        comment = meta.get('comment', '')
        venue, is_workshop = infer_venue_from_comment(comment)
        if not venue and paper.venue_hint:
            venue = paper.venue_hint
        if not venue:
            venue = paper.header_hints.get('venue', '')
        if not venue and paper.link:
            url_venue = infer_venue_from_url(paper.link)
            if url_venue and url_venue != 'OpenReview':
                venue = url_venue
        if not venue:
            venue = 'arXiv'
        if venue == 'arXiv' or 'arXiv' in venue:
            entry_type = 'article'
            journal = f"arXiv preprint arXiv:{paper.arxiv_id}" if paper.arxiv_id else 'arXiv preprint'
            booktitle = None
            eprint = paper.arxiv_id
            archive_prefix = 'arXiv'
        else:
            entry_type = 'inproceedings'
            journal = None
            booktitle = build_booktitle(venue)
            eprint = None
            archive_prefix = None
        key = generate_citation_key(paper, venue, year)
        note = None
        if paper.code_link and paper.code_link.lower() != 'null':
            note = f"Code: {paper.code_link}"
        entry = BibEntry(
            key=key, entry_type=entry_type, title=title,
            authors=format_authors(authors), year=year, venue=venue,
            booktitle=booktitle, journal=journal, url=paper.link,
            eprint=eprint, archive_prefix=archive_prefix, note=note,
            topic=paper.topic
        )
        entries.append(entry)
        if not year or not authors or authors == 'TBD':
            missing_metadata.append(paper.paper_title)
    print(f"\n=== Writing {bib_path} ===")
    with open(bib_path, 'w', encoding='utf-8') as f:
        f.write('% Auto-generated from related-work.md\n')
        f.write('% Manual edits may be preserved during incremental updates\n')
        f.write('% Run: python generate_bib.py related-work.md related-work.bib\n\n')
        current_topic = None
        for entry in entries:
            if entry.topic != current_topic:
                current_topic = entry.topic
                f.write(f'% ============================================================\n')
                f.write(f'% {current_topic}\n')
                f.write(f'% ============================================================\n\n')
            f.write(generate_bibtex(entry))
            f.write('\n\n')
    print(f"Wrote {len(entries)} entries to {bib_path}")
    if missing_metadata:
        print(f"\nWARNING: {len(missing_metadata)} papers missing year or authors:")
        for t in missing_metadata[:10]:
            print(f"  - {t[:80]}")
    print("Done!")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python generate_bib.py <input.md> <output.bib>")
        print("Example: python generate_bib.py docs/related-work.md docs/related-work.bib")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
```

**Dependencies**: Standard library only (`re`, `os`, `sys`, `json`, `time`, `urllib`, `xml`). No external packages required.

**What the improved script handles**:
- **Robust arXiv ID extraction**: modern format (2401.12345), old format (cs/0101001), version suffixes
- **Header hint parsing**: extracts `(Venue Year)` from markdown headers as fallback
- **Multi-source venue inference**: arXiv comment → markdown hint → header hint → URL pattern
- **Citation key deduplication**: automatically appends `a`, `b`, `c` for colliding keys
- **`eprint` + `archivePrefix`**: proper arXiv BibTeX fields for BibLaTeX compatibility
- **Code availability notes**: adds `note = {Code: URL}` when `**Code**:` is present
- **Graceful degradation**: works even when arXiv API is unavailable; uses cached/header data
- **Author format robustness**: handles list input, string input, comma-separated, "and"-separated
- **Progress reporting**: shows fetch progress, duplicate warnings, missing metadata alerts
---

## Phase 7: Common Pitfalls and How to Avoid Them

| Pitfall                                   | Why It Happens                                  | Solution                                                                                  |
|-------------------------------------------|-------------------------------------------------|-------------------------------------------------------------------------------------------|
| **Missing venue for famous papers**       | Assuming high-profile papers are "just arXiv"   | Always search OpenReview and author websites for Meta/Google/Stanford papers              |
| **Workshop vs Main Conference confusion** | OpenReview shows both on same page              | Read the exact venue label: `LIT Workshop @ ICLR 2026` is a workshop, not ICLR main       |
| **Year mismatch in key**                  | Method name contains numbers (e.g., `System15`) | Validate using the year segment after the last hyphen, not the first number in the key    |
| **arXiv comment outdated**                | Authors update the paper but not the comment    | Cross-check with the actual conference virtual page                                       |
| **Date mismatch with claimed venue**      | Paper posted to arXiv after conference date     | If arXiv date > conference date, reject the claim unless strong evidence exists           |
| **Missing `Proc.~` prefix**               | Copy-pasting from other BibTeX sources          | Enforce via validation script                                                             |
| **Incomplete author list**                | arXiv API truncates or uses `et al.`            | Compare with PDF author list; use OpenReview for complete lists                           |
| **False positives from social media**     | LinkedIn posts misremember or are aspirational  | Only trust author-owned websites or official proceedings                                  |
| **D&B Track vs Main Conference**          | NeurIPS has multiple tracks                     | Specify `Datasets and Benchmarks Track` in notes if relevant, but key remains `NeurIPSYY` |
| **Withdrawn / Desk Rejected papers**      | Paper was on OpenReview but rejected            | Keep as `@article` with `journal = {arXiv preprint ...}`; do not invent a venue           |

---

## Appendix A: Venue Name Reference Table

| Abbreviation           | Full `booktitle` or `journal`                                                                                 |
|------------------------|---------------------------------------------------------------------------------------------------------------|
| **ML/AI**              |                                                                                                               |
| NeurIPS                | `Proc.~Conference on Neural Information Processing Systems`                                                   |
| ICML                   | `Proc.~International Conference on Machine Learning`                                                          |
| ICLR                   | `Proc.~International Conference on Learning Representations`                                                  |
| ICLRW                  | `Proc.~ICLR Workshop on [WorkshopName]`                                                                       |
| AAAI                   | `Proc.~AAAI Conference on Artificial Intelligence`                                                            |
| IJCAI                  | `Proc.~International Joint Conference on Artificial Intelligence`                                             |
| UAI                    | `Proc.~Conference on Uncertainty in Artificial Intelligence`                                                  |
| AISTATS                | `Proc.~International Conference on Artificial Intelligence and Statistics`                                    |
| COLT                   | `Proc.~Annual Conference on Learning Theory`                                                                  |
| KDD                    | `Proc.~ACM SIGKDD Conference on Knowledge Discovery and Data Mining`                                          |
| WSDM                   | `Proc.~ACM International Conference on Web Search and Data Mining`                                            |
| CIKM                   | `Proc.~ACM International Conference on Information and Knowledge Management`                                  |
| SDM                    | `Proc.~SIAM International Conference on Data Mining`                                                          |
| ICDM                   | `Proc.~IEEE International Conference on Data Mining`                                                          |
| PAKDD                  | `Proc.~Pacific-Asia Conference on Knowledge Discovery and Data Mining`                                        |
| ECML-PKDD              | `Proc.~European Conference on Machine Learning and Principles and Practice of Knowledge Discovery`            |
| **NLP**                |                                                                                                               |
| ACL                    | `Proc.~Annual Meeting of the Association for Computational Linguistics`                                       |
| ACLFindings            | `Proc.~Annual Meeting of the Association for Computational Linguistics Findings`                              |
| EMNLP                  | `Proc.~Conference on Empirical Methods in Natural Language Processing`                                        |
| EMNLPFindings          | `Proc.~Conference on Empirical Methods in Natural Language Processing Findings`                               |
| NAACL                  | `Proc.~Conference of the North American Chapter of the Association for Computational Linguistics`             |
| EACL                   | `Proc.~Conference of the European Chapter of the Association for Computational Linguistics`                   |
| AACL                   | `Proc.~Asia-Pacific Chapter of the Association for Computational Linguistics`                                 |
| COLING                 | `Proc.~International Conference on Computational Linguistics`                                                 |
| LREC                   | `Proc.~International Conference on Language Resources and Evaluation`                                         |
| CoNLL                  | `Proc.~Conference on Computational Natural Language Learning`                                                 |
| COLM                   | `Proc.~Conference on Language Modeling`                                                                       |
| COLMW                  | `Proc.~COLM Workshop on [WorkshopName]`                                                                       |
| **Vision**             |                                                                                                               |
| CVPR                   | `Proc.~Conference on Computer Vision and Pattern Recognition`                                                 |
| ICCV                   | `Proc.~IEEE International Conference on Computer Vision`                                                      |
| ECCV                   | `Proc.~European Conference on Computer Vision`                                                                |
| WACV                   | `Proc.~Winter Conference on Applications of Computer Vision`                                                  |
| BMVC                   | `Proc.~British Machine Vision Conference`                                                                     |
| ACCV                   | `Proc.~Asian Conference on Computer Vision`                                                                   |
| ICIP                   | `Proc.~IEEE International Conference on Image Processing`                                                     |
| **Speech/Audio**       |                                                                                                               |
| ICASSP                 | `Proc.~IEEE International Conference on Acoustics, Speech and Signal Processing`                              |
| INTERSPEECH            | `Proc.~Annual Conference of the International Speech Communication Association`                               |
| ASRU                   | `Proc.~IEEE Automatic Speech Recognition and Understanding Workshop`                                          |
| SLT                    | `Proc.~IEEE Spoken Language Technology Workshop`                                                              |
| **IR/Web**             |                                                                                                               |
| SIGIR                  | `Proc.~Annual International ACM SIGIR Conference on Research and Development in Information Retrieval`        |
| WWW                    | `Proc.~The Web Conference`                                                                                    |
| WSDM                   | `Proc.~ACM International Conference on Web Search and Data Mining`                                            |
| ECIR                   | `Proc.~European Conference on Information Retrieval`                                                          |
| **Robotics**           |                                                                                                               |
| ICRA                   | `Proc.~IEEE International Conference on Robotics and Automation`                                              |
| IROS                   | `Proc.~IEEE/RSJ International Conference on Intelligent Robots and Systems`                                   |
| RSS                    | `Proc.~Robotics: Science and Systems`                                                                         |
| CoRL                   | `Proc.~Conference on Robot Learning`                                                                          |
| **Systems/HCI**        |                                                                                                               |
| OSDI                   | `Proc.~USENIX Symposium on Operating Systems Design and Implementation`                                       |
| SOSP                   | `Proc.~ACM Symposium on Operating Systems Principles`                                                         |
| NSDI                   | `Proc.~USENIX Symposium on Networked Systems Design and Implementation`                                       |
| EuroSys                | `Proc.~ACM European Conference on Computer Systems`                                                           |
| ASPLOS                 | `Proc.~ACM International Conference on Architectural Support for Programming Languages and Operating Systems` |
| CHI                    | `Proc.~ACM Conference on Human Factors in Computing Systems`                                                  |
| UIST                   | `Proc.~ACM Symposium on User Interface Software and Technology`                                               |
| **Journals**           |                                                                                                               |
| TMLR                   | `Transactions on Machine Learning Research`                                                                   |
| JMLR                   | `Journal of Machine Learning Research`                                                                        |
| TPAMI                  | `IEEE Transactions on Pattern Analysis and Machine Intelligence`                                              |
| IJCV                   | `International Journal of Computer Vision`                                                                    |
| TACL                   | `Transactions of the Association for Computational Linguistics`                                               |
| TIST                   | `ACM Transactions on Intelligent Systems and Technology`                                                      |
| TKDD                   | `ACM Transactions on Knowledge Discovery from Data`                                                           |
| TNNLS                  | `IEEE Transactions on Neural Networks and Learning Systems`                                                   |
| TIP                    | `IEEE Transactions on Image Processing`                                                                       |
| TOG                    | `ACM Transactions on Graphics`                                                                                |
| CACM                   | `Communications of the ACM`                                                                                   |
| Nature                 | `Nature`                                                                                                      |
| NatureMI               | `Nature Machine Intelligence`                                                                                 |
| Science                | `Science`                                                                                                     |
| PNAS                   | `Proceedings of the National Academy of Sciences`                                                             |
| arXiv                  | `arXiv preprint arXiv:XXXX.XXXXX`                                                                             |
| **Workshops (select)** |                                                                                                               |
| NeurIPSW               | `Proc.~NeurIPS Workshop on [WorkshopName]`                                                                    |
| ICMLW                  | `Proc.~ICML Workshop on [WorkshopName]`                                                                       |
| ACLW                   | `Proc.~ACL Workshop on [WorkshopName]`                                                                        |
| EMNLPW                 | `Proc.~EMNLP Workshop on [WorkshopName]`                                                                      |
| CVPRW                  | `Proc.~CVPR Workshop on [WorkshopName]`                                                                       |
| ICCVW                  | `Proc.~ICCV Workshop on [WorkshopName]`                                                                       |
| **Interdisciplinary**  |                                                                                                               |
| ICSE                   | `Proc.~IEEE/ACM International Conference on Software Engineering`                                             |
| FSE                    | `Proc.~ACM SIGSOFT International Symposium on Foundations of Software Engineering`                            |
| PLDI                   | `Proc.~ACM SIGPLAN Conference on Programming Language Design and Implementation`                              |
| POPL                   | `Proc.~ACM SIGPLAN Symposium on Principles of Programming Languages`                                          |
| AAAIW                  | `Proc.~AAAI Workshop on [WorkshopName]`                                                                       |
| IJCAIW                 | `Proc.~IJCAI Workshop on [WorkshopName]`                                                                      |

---

## Appendix B: Author Name Handling

### B.1 Author Format Conversion

BibTeX expects `"LastName, FirstName and LastName2, FirstName2"` format. Convert from various input formats:

```python
def normalize_authors(author_input):
    """
    Convert various author formats to BibTeX format.
    
    Input formats supported:
    - List: ["Alice Smith", "Bob Jones"]
    - String with 'and': "Alice Smith and Bob Jones"
    - String with commas: "Smith, Alice and Jones, Bob"
    - Single author: "Alice Smith"
    """
    if not author_input:
        return ''
    
    # If already a list
    if isinstance(author_input, list):
        authors = author_input
    else:
        # Split by ' and ' (BibTeX separator)
        authors = [a.strip() for a in author_input.split(' and ')]
    
    formatted = []
    for author in authors:
        author = author.strip()
        if not author:
            continue
        
        # Already in "Last, First" format?
        if ',' in author:
            formatted.append(author)
            continue
        
        # "First Last" or "First Middle Last" -> "Last, First Middle"
        parts = author.split()
        if len(parts) >= 2:
            first_names = ' '.join(parts[:-1])
            last_name = parts[-1]
            formatted.append(f"{last_name}, {first_names}")
        else:
            formatted.append(author)
    
    return ' and '.join(formatted)
```

### B.2 Special Character Handling

Many papers have special characters in titles and author names. Handle them correctly:

```python
def escape_bibtex(text):
    """Escape special characters for BibTeX compatibility."""
    if not text:
        return text
    
    # Order matters: escape backslash first
    replacements = [
        ('\\', '\\textbackslash{}'),
        ('&', '\\&'),
        ('%', '\\%'),
        ('$', '\\$'),
        ('#', '\\#'),
        ('_', '\\_'),
        ('{', '\\{'),
        ('}', '\\}'),
        ('~', '\\textasciitilde{}'),
        ('^', '\\textasciicircum{}'),
    ]
    
    for old, new in replacements:
        text = text.replace(old, new)
    
    return text

def preserve_math_symbols(text):
    """Preserve math symbols by wrapping in $...$."""
    # Common math patterns
    math_patterns = [
        (r'∇', r'$\\nabla$'),
        (r'α', r'$\\alpha$'),
        (r'β', r'$\\beta$'),
        (r'γ', r'$\\gamma$'),
        (r'δ', r'$\\delta$'),
        (r'θ', r'$\\theta$'),
        (r'λ', r'$\\lambda$'),
        (r'μ', r'$\\mu$'),
        (r'σ', r'$\\sigma$'),
        (r'τ', r'$\\tau$'),
        (r'ω', r'$\\omega$'),
        (r'ε', r'$\\epsilon$'),
        (r'×', r'$\\times$'),
        (r'→', r'$\\rightarrow$'),
        (r'←', r'$\\leftarrow$'),
    ]
    
    for old, new in math_patterns:
        text = text.replace(old, new)
    
    return text
```

### B.3 Unicode Author Names

Preserve accented characters using LaTeX macros:

| Character | LaTeX    | Character | LaTeX    |
|-----------|----------|-----------|----------|
| á         | `\\'a`   | Á         | `\\'A`   |
| é         | `\\'e`   | É         | `\\'E`   |
| í         | `\\'i`   | Í         | `\\'I`   |
| ó         | `\\'o`   | Ó         | `\\'O`   |
| ú         | `\\'u`   | Ú         | `\\'U`   |
| ñ         | `\\~n`   | Ñ         | `\\~N`   |
| ç         | `\\c{c}` | Ç         | `\\c{C}` |
| ü         | `\\"u`   | Ü         | `\\"U`   |
| ö         | `\\"o`   | Ö         | `\\"O`   |
| ä         | `\\"a`   | Ä         | `\\"A`   |
| ß         | `\\ss`   | ø         | `\\o`    |
| å         | `\\aa`   | Å         | `\\AA`   |

**Tip**: For authors with many special characters, consider using `\usepackage[utf8]{inputenc}` in your LaTeX preamble and keeping UTF-8 characters as-is.

---

## Appendix C: Quick-Check Decision Tree

```
Does the paper have an arXiv ID?
├── Yes -> Query arXiv API for comment
│   ├── Comment contains "Accepted to [Venue] [Year]"?
│   │   ├── Yes -> Verify with conference proceedings / anthology
│   │   └── No -> Proceed to deep search
│   └── Comment says "Under review" / "Submitted"?
│       ├── Yes -> Check OpenReview for current status
│       └── No -> Keep as arXiv-only
└── No -> Search by exact title
    ├── Found on OpenReview with acceptance?
    │   └── Use OpenReview venue
    ├── Found on ACL Anthology?
    │   └── Use Anthology venue
    ├── Found on DBLP / Google Scholar?
    │   └── Cross-check with 2+ sources
    └── Not found anywhere?
        └── Keep as arXiv-only (or omit if not on arXiv)
```

---

## Summary Checklist

Before declaring a `related-work.bib` file complete:

- [ ] All papers from the source markdown are included.
- [ ] Every entry has a complete `author` field.
- [ ] Every entry has a `title` field matching the official paper title.
- [ ] Every entry has a `year` field matching the publication year.
- [ ] Conference papers use `@inproceedings` with `Proc.~` prefix in `booktitle`.
- [ ] Journal/arXiv papers use `@article` with proper `journal` field.
- [ ] Citation keys follow `MethodAbbreviation-VenueYear` format.
- [ ] arXiv is used only as a fallback when no conference/journal venue is found.
- [ ] The validation script reports 0 errors.
- [ ] Papers are grouped by topic with clear section comments.
