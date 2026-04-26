import urllib.request
import xml.etree.ElementTree as ET
import time

arxiv_ids = [
    "2512.24617",
    "2510.07358",
    "2602.08984",
    "2504.19095",
    "2601.21598",
    "2602.04246",
    "2512.21711",
    "2602.10229",
    "2511.16885",
    "2602.08332",
    "2602.09670",
    "2604.02029",
    "2602.08220",
    "2502.12949",
    "2510.12164",
    "2512.01278",
    "2506.21734",
    "2412.08821",
    "2408.00655",
    "2511.15244",
    "2502.06171",
    "2512.07558",
    "2503.18866",
    "2501.19201",
    "2603.06222",
    "2510.25741",
    "2508.12587",
    "2505.13308",
]

results = {}
for i in range(0, len(arxiv_ids), 10):
    batch = arxiv_ids[i : i + 10]
    url = f'http://export.arxiv.org/api/query?id_list={",".join(batch)}'
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read().decode()
        root = ET.fromstring(data)
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        for entry in root.findall("atom:entry", ns):
            id_elem = entry.find("atom:id", ns)
            if id_elem is None:
                continue
            aid = id_elem.text.split("/")[-1]
            title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
            comment = entry.find("arxiv:comment", ns)
            comment_text = (
                comment.text.strip().replace("\n", " ") if comment is not None else ""
            )
            results[aid] = {"title": title, "comment": comment_text}
    except Exception as e:
        print(f"Error for batch {batch}: {e}")
    time.sleep(3)

for aid in arxiv_ids:
    r = results.get(aid, {})
    print(f'{aid}\t{r.get("title", "?")}\t{r.get("comment", "")}')
