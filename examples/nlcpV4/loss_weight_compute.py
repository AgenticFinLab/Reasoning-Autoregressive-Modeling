"""Compute per-(model, level) Builder loss weights from ``<dataset>_Loss_prepare.json``.

Usage:
    python3 examples/nlcpV4/loss_weight_compute.py -f EXPERIMENT/nlcpV4/builder/{dataset}_Loss_prepare.json

The CSV sidecar is named from the input stem, so the dataset prefix
is preserved automatically:

    -f .../GSM8K_Loss_prepare.json  →  .../GSM8K_Loss_prepare_weights.csv
    -f .../MATH_Loss_prepare.json   →  .../MATH_Loss_prepare_weights.csv

Weighting rules (applied to the raw per-component mean of each config):

    residual_loss_weight  = 1.0                       always
    reasoning_loss_weight = 1.0                       always  (keep raw unchanged)
    ordering_loss_weight  = 1.0                       if raw_ordering <  6
                          = 6.0  / raw_ordering       otherwise  (cap weighted at ~6)
    recon_loss_weight     = 1.0                       if raw_recon    < 10
                          = 10.0 / raw_recon          otherwise  (cap weighted at ~10)

Priority hierarchy (weighted contribution, largest first):
    recon    (capped at ~10)
    reasoning (raw, typically 4 - 8)
    ordering  (capped at ~6)
    residual  (raw, typically 0.8 for L<8, ~1.2 for L=8)

Output:
  1. A table on stdout listing raw losses, computed weights, and the
     resulting weighted losses for every config in the JSON, sorted
     by model size and pyramid level, followed by copy-ready YAML
     weight blocks.
  2. A ``<stem>_weights.csv`` file written next to the input JSON
     (e.g. ``GSM8K_Loss_prepare.json`` → ``GSM8K_Loss_prepare_weights.csv``)
     containing one flat row per config with raw / weight / weighted
     values for direct downstream analysis.
  3. A dataset-aware analytical summary block, dispatched from the
     ``dataset`` field encoded in each JSON key (``{dataset}/train_...``)
     so the report matches the per-dataset design documents:
       - GSM8K → `loss-weights-analysis-gsm8k.md` §9.6 (per-level
         ordering closed form) + §9.7 (per-model recon closed form;
         valid because within-model level spread < 20%).
       - MATH  → `loss-weights-analysis-math.md` §9.6 (per-level
         ordering closed form) + §9.7 (full 6×6 per-(model, level)
         recon weight grid; the per-model closed form breaks down
         on MATH due to the L=8 recon spike, up to 4.9× for
         Qwen2.5-3B).

Access is strict fail-fast: missing keys raise KeyError at parse
time rather than being silently defaulted.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# ── Threshold constants ──────────────────────────────────────────────
RECON_CAP = 10.0
ORDERING_CAP = 6.0


# ── Canonical ordering for the reports ───────────────────────────────
MODEL_SIZE_ORDER: dict[str, int] = {
    "Qwen2.5-0.5B": 0,
    "Qwen3-0.6B": 1,
    "Qwen2.5-1.5B": 2,
    "Qwen3-1.7B": 3,
    "Qwen2.5-3B": 4,
    "Qwen3-4B": 5,
    "Qwen3-8B": 6,
}


# Config key layout in Loss_prepare.json:
#   "{dataset}/train_{module}_{model}_{level}level"
# e.g. "GSM8K/train_builder_Qwen3-8B_2level"
_KEY_HEAD = re.compile(r"^(?P<dataset>[^/]+)/train_(?P<module>[^_]+)_(?P<rest>.+)$")
_KEY_TAIL = re.compile(r"^(?P<model>.+)_(?P<level>\d+)level$")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the loss-weight computation script."""
    p = argparse.ArgumentParser(
        description=(
            "Compute per-config Builder loss weights from a "
            "Loss_prepare.json results file."
        ),
    )
    p.add_argument(
        "-f",
        "--file",
        required=True,
        type=Path,
        help=(
            "Path to a Loss_prepare.json file produced by "
            "examples/RunResults/loss_prepare.py."
        ),
    )
    return p.parse_args()


def cap_weight(raw: float, cap: float) -> float:
    """Return the weight such that ``raw * weight`` does not exceed ``cap``.

    When ``raw < cap`` the component is already below the cap and the
    weight stays at ``1.0`` — the loss is kept at its natural scale.
    When ``raw >= cap`` the weight is ``cap / raw`` so the weighted
    value sits at ``cap``. ``raw <= 0`` is degenerate and returns
    ``1.0`` (there is nothing to cap).
    """
    if raw <= 0.0 or raw < cap:
        return 1.0
    return cap / raw


def parse_entry_key(key: str) -> dict | None:
    """Split a Loss_prepare JSON key into its structured components."""
    m_head = _KEY_HEAD.match(key)
    if m_head is None:
        return None
    m_tail = _KEY_TAIL.match(m_head.group("rest"))
    if m_tail is None:
        return None
    return {
        "dataset": m_head.group("dataset"),
        "module": m_head.group("module"),
        "model": m_tail.group("model"),
        "level": int(m_tail.group("level")),
    }


def compute_row(key: str, entry: dict) -> dict | None:
    """Build a single report row from a Loss_prepare JSON entry.

    Strict fail-fast access: ``entry["stats"]["raw"][component]["mean"]``
    must exist for every component (recon / ordering / residual /
    reasoning). Missing keys raise ``KeyError`` at the caller.
    """
    meta = parse_entry_key(key)
    if meta is None:
        return None

    raw = entry["stats"]["raw"]
    r_recon = float(raw["recon"]["mean"])
    r_ord = float(raw["ordering"]["mean"])
    r_res = float(raw["residual"]["mean"])
    r_rea = float(raw["reasoning"]["mean"])

    w_recon = cap_weight(r_recon, RECON_CAP)
    w_ord = cap_weight(r_ord, ORDERING_CAP)
    w_res = 1.0
    w_rea = 1.0

    return {
        **meta,
        "key": key,
        # raw per-component means
        "r_recon": r_recon,
        "r_ord": r_ord,
        "r_res": r_res,
        "r_rea": r_rea,
        # computed per-config weights
        "w_recon": w_recon,
        "w_ord": w_ord,
        "w_res": w_res,
        "w_rea": w_rea,
        # resulting weighted per-component values
        "wt_recon": r_recon * w_recon,
        "wt_ord": r_ord * w_ord,
        "wt_res": r_res * w_res,
        "wt_rea": r_rea * w_rea,
    }


def print_header(file_path: Path, n_entries: int) -> None:
    """Print a short banner describing the source JSON and entry count."""
    print(f"Source : {file_path}")
    print(f"Entries: {n_entries}")
    print(
        f"Rules  : recon cap={RECON_CAP:g}, ordering cap={ORDERING_CAP:g}, "
        f"residual=1.0, reasoning=1.0"
    )
    print()


def print_table(rows: list[dict]) -> None:
    """Print a wide table of raw / weight / weighted values per config."""
    cols = [
        ("model", "<", 14),
        ("L", ">", 3),
        ("raw_recon", ">", 10),
        ("raw_ord", ">", 9),
        ("raw_res", ">", 8),
        ("raw_rea", ">", 9),
        ("w_recon", ">", 9),
        ("w_ord", ">", 9),
        ("w_res", ">", 7),
        ("w_rea", ">", 7),
        ("wt_recon", ">", 9),
        ("wt_ord", ">", 8),
        ("wt_res", ">", 8),
        ("wt_rea", ">", 8),
        ("wt_total", ">", 9),
    ]

    def fmt_header() -> str:
        return " ".join(f"{name:{align}{w}}" for name, align, w in cols)

    print(fmt_header())
    print(" ".join("-" * w for _, _, w in cols))

    for r in rows:
        wt_total = r["wt_recon"] + r["wt_ord"] + r["wt_res"] + r["wt_rea"]
        vals = [
            f"{r['model']:<14}",
            f"{r['level']:>3}",
            f"{r['r_recon']:>10.3f}",
            f"{r['r_ord']:>9.3f}",
            f"{r['r_res']:>8.3f}",
            f"{r['r_rea']:>9.3f}",
            f"{r['w_recon']:>9.4f}",
            f"{r['w_ord']:>9.4f}",
            f"{r['w_res']:>7.2f}",
            f"{r['w_rea']:>7.2f}",
            f"{r['wt_recon']:>9.3f}",
            f"{r['wt_ord']:>8.3f}",
            f"{r['wt_res']:>8.3f}",
            f"{r['wt_rea']:>8.3f}",
            f"{wt_total:>9.3f}",
        ]
        print(" ".join(vals))


# Column order used in both the on-screen table and the CSV export.
# Keep these in sync so the CSV can be diff-ed against the printed table.
CSV_COLUMNS: list[str] = [
    "dataset",
    "module",
    "model",
    "level",
    "key",
    "raw_recon",
    "raw_ordering",
    "raw_residual",
    "raw_reasoning",
    "w_recon",
    "w_ordering",
    "w_residual",
    "w_reasoning",
    "wt_recon",
    "wt_ordering",
    "wt_residual",
    "wt_reasoning",
    "wt_total",
]


def write_csv(rows: list[dict], out_path: Path) -> None:
    """Write per-config raw / weight / weighted values to ``out_path``.

    The CSV is intentionally flat (one row per config) and uses the
    same column order as :data:`CSV_COLUMNS` so it can be compared
    directly with the printed table.
    """
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "dataset": r["dataset"],
                    "module": r["module"],
                    "model": r["model"],
                    "level": r["level"],
                    "key": r["key"],
                    "raw_recon": f"{r['r_recon']:.6f}",
                    "raw_ordering": f"{r['r_ord']:.6f}",
                    "raw_residual": f"{r['r_res']:.6f}",
                    "raw_reasoning": f"{r['r_rea']:.6f}",
                    "w_recon": f"{r['w_recon']:.6f}",
                    "w_ordering": f"{r['w_ord']:.6f}",
                    "w_residual": f"{r['w_res']:.6f}",
                    "w_reasoning": f"{r['w_rea']:.6f}",
                    "wt_recon": f"{r['wt_recon']:.6f}",
                    "wt_ordering": f"{r['wt_ord']:.6f}",
                    "wt_residual": f"{r['wt_res']:.6f}",
                    "wt_reasoning": f"{r['wt_rea']:.6f}",
                    "wt_total": f"{r['wt_recon'] + r['wt_ord'] + r['wt_res'] + r['wt_rea']:.6f}",
                }
            )


def print_weight_blocks(rows: list[dict]) -> None:
    """Print compact YAML-style weight blocks, one per config key.

    The blocks are meant to be read / copied by hand into the
    ``training.loss_weights`` section of the corresponding YAML.
    """
    print()
    print("=" * 70)
    print("Per-config weights (copy into training.loss_weights):")
    print("=" * 70)
    for r in rows:
        print(
            f"# {r['key']}  "
            f"(raw: recon={r['r_recon']:.2f}, ord={r['r_ord']:.2f}, "
            f"res={r['r_res']:.2f}, rea={r['r_rea']:.2f})"
        )
        print(
            f"  recon_loss_weight:     {r['w_recon']:.4f}"
            f"    # weighted \u2248 {r['wt_recon']:.3f}"
        )
        print(
            f"  ordering_loss_weight:  {r['w_ord']:.4f}"
            f"    # weighted \u2248 {r['wt_ord']:.3f}"
        )
        print(
            f"  residual_loss_weight:  {r['w_res']:.4f}"
            f"    # weighted \u2248 {r['wt_res']:.3f}"
        )
        print(
            f"  reasoning_loss_weight: {r['w_rea']:.4f}"
            f"    # weighted \u2248 {r['wt_rea']:.3f}"
        )
        print()


def _per_level_ordering_summary(rows: list[dict]) -> list[tuple[int, float, float]]:
    """Return ``(L, mu_ord(L), w_ord*(L))`` triples, sorted by L.

    This implements the §9.6 closed form used in both
    ``loss-weights-analysis-gsm8k.md`` and ``loss-weights-analysis-math.md``:
    ordering is nearly model-invariant, so the per-level raw mean
    across models is averaged and passed through the cap rule once.
    """
    levels = sorted({r["level"] for r in rows})
    out: list[tuple[int, float, float]] = []
    for L in levels:
        cell = [r["r_ord"] for r in rows if r["level"] == L]
        if not cell:
            continue
        mu = sum(cell) / len(cell)
        out.append((L, mu, cap_weight(mu, ORDERING_CAP)))
    return out


def _per_model_recon_summary(rows: list[dict]) -> list[tuple[str, float, float]]:
    """Return ``(model, mu_recon(m), w_recon*(m))`` triples, sorted by size.

    This implements the GSM8K §9.7 closed form: recon is nearly
    level-invariant within a model (within-model spread < 20% on
    GSM8K), so the per-model raw mean across levels is averaged and
    passed through the cap rule once.
    """
    models = sorted(
        {r["model"] for r in rows},
        key=lambda m: MODEL_SIZE_ORDER.get(m, 99),
    )
    out: list[tuple[str, float, float]] = []
    for m in models:
        cell = [r["r_recon"] for r in rows if r["model"] == m]
        if not cell:
            continue
        mu = sum(cell) / len(cell)
        out.append((m, mu, cap_weight(mu, RECON_CAP)))
    return out


def report_gsm8k_summary(rows: list[dict]) -> None:
    """Emit the GSM8K closed-form summary (loss-weights-analysis-gsm8k.md).

    Prints:
      1. §9.6 per-level ordering weights (model-invariant closed form).
      2. §9.7 per-model recon weights    (level-invariant closed form).

    Both reductions are valid for GSM8K because ordering CoV across
    models is ~1.2% and recon within-model spread is <20%. On MATH the
    recon reduction fails; see :func:`report_math_summary`.
    """
    print()
    print("=" * 70)
    print("GSM8K §9.6 — per-level ordering weights (model-invariant)")
    print("=" * 70)
    print(f"  {'L':>2}  {'mu_ord(L)':>10}  {'w_ord*(L)':>10}")
    print(f"  {'-' * 2:>2}  {'-' * 10:>10}  {'-' * 10:>10}")
    for L, mu, w in _per_level_ordering_summary(rows):
        print(f"  {L:>2}  {mu:>10.4f}  {w:>10.4f}")

    print()
    print("=" * 70)
    print("GSM8K §9.7 — per-model recon weights (level-invariant)")
    print("=" * 70)
    print(f"  {'model':<14}  {'mu_recon(m)':>12}  {'w_recon*(m)':>12}")
    print(f"  {'-' * 14:<14}  {'-' * 12:>12}  {'-' * 12:>12}")
    for m, mu, w in _per_model_recon_summary(rows):
        print(f"  {m:<14}  {mu:>12.4f}  {w:>12.4f}")


def report_math_summary(rows: list[dict]) -> None:
    """Emit the MATH summary (loss-weights-analysis-math.md).

    Prints:
      1. §9.6 per-level ordering weights (model-invariant closed form;
         numerically identical to GSM8K within ±0.01).
      2. §9.7 per-(model, level) recon weight grid (no closed form).

    The GSM8K per-model reduction does NOT hold on MATH because the
    L=8 recon spike blows up the within-model spread to 4.9× for
    Qwen2.5-3B; every cell therefore carries its own weight.
    """
    print()
    print("=" * 70)
    print("MATH §9.6 — per-level ordering weights (model-invariant)")
    print("=" * 70)
    print(f"  {'L':>2}  {'mu_ord(L)':>10}  {'w_ord*(L)':>10}")
    print(f"  {'-' * 2:>2}  {'-' * 10:>10}  {'-' * 10:>10}")
    for L, mu, w in _per_level_ordering_summary(rows):
        print(f"  {L:>2}  {mu:>10.4f}  {w:>10.4f}")

    print()
    print("=" * 70)
    print("MATH §9.7 — per-(model, level) recon weights (full grid)")
    print("=" * 70)
    levels = sorted({r["level"] for r in rows})
    models = sorted(
        {r["model"] for r in rows},
        key=lambda m: MODEL_SIZE_ORDER.get(m, 99),
    )
    lookup = {(r["model"], r["level"]): r["w_recon"] for r in rows}
    header = f"  {'model':<14}" + "".join(f"  {'L=' + str(L):>8}" for L in levels)
    print(header)
    print(f"  {'-' * 14:<14}" + "".join(f"  {'-' * 8:>8}" for _ in levels))
    for m in models:
        parts = [f"  {m:<14}"]
        for L in levels:
            w = lookup.get((m, L))
            parts.append(f"  {'—':>8}" if w is None else f"  {w:>8.4f}")
        print("".join(parts))


# Dataset-aware dispatch table: maps the ``dataset`` field parsed from
# each Loss_prepare JSON key to the summary-emission function that
# matches the per-dataset design document.
DATASET_REPORTERS: dict[str, callable] = {
    "GSM8K": report_gsm8k_summary,
    "MATH": report_math_summary,
}


def dispatch_dataset_summary(rows: list[dict]) -> None:
    """Group rows by dataset and invoke the matching reporter for each.

    In practice a single ``<dataset>_Loss_prepare.json`` contains one
    dataset, so this produces exactly one summary block. The grouping
    makes the code robust to future mixed files without special-casing.
    """
    by_ds: dict[str, list[dict]] = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    for ds, ds_rows in by_ds.items():
        reporter = DATASET_REPORTERS.get(ds)
        if reporter is None:
            print()
            print(
                f"[INFO] No dataset-specific summary defined for '{ds}'. "
                f"Supported: {sorted(DATASET_REPORTERS)}."
            )
            continue
        reporter(ds_rows)


def main() -> int:
    """CLI entry point: recompute loss weights from Loss_prepare.json."""
    args = parse_args()
    if not args.file.is_file():
        print(f"[ERROR] File not found: {args.file}", file=sys.stderr)
        return 1

    data = json.loads(args.file.read_text(encoding="utf-8"))

    rows: list[dict] = []
    for key, entry in data.items():
        row = compute_row(key, entry)
        if row is not None:
            rows.append(row)

    if not rows:
        print(
            "[ERROR] No parsable entries found. Expected keys like "
            "'DATASET/train_MODULE_MODEL_Nlevel' (e.g. "
            "'GSM8K/train_builder_Qwen3-8B_2level').",
            file=sys.stderr,
        )
        return 1

    rows.sort(key=lambda r: (MODEL_SIZE_ORDER.get(r["model"], 99), r["level"]))

    print_header(args.file, len(rows))
    print_table(rows)
    print_weight_blocks(rows)
    # Dataset-aware analytical summary (GSM8K §9.6/§9.7 closed forms,
    # or MATH §9.6 closed form + §9.7 per-(model, level) grid).
    dispatch_dataset_summary(rows)

    # Emit a CSV sidecar next to the input JSON. Name: <stem>_weights.csv
    csv_path = args.file.with_name(args.file.stem + "_weights.csv")
    write_csv(rows, csv_path)
    print(f"[CSV] Wrote {len(rows)} rows to {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
