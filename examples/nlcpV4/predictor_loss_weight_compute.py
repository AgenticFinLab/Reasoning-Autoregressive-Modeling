"""Compute per-(model, level, mode) Predictor loss weights from ``<dataset>_Loss_prepare_<mode>.json``.

Usage:
    python3 examples/nlcpV4/predictor_loss_weight_compute.py -f EXPERIMENT/nlcpV4/predictor/GSM8K_Loss_prepare_independent.json
    python3 examples/nlcpV4/predictor_loss_weight_compute.py -f EXPERIMENT/nlcpV4/predictor/GSM8K_Loss_prepare_shared.json
    python3 examples/nlcpV4/predictor_loss_weight_compute.py -f ...Loss_prepare_independent.json -t 10 -M independent

The CSV sidecar is named from the input stem, so the dataset + mode
prefix is preserved automatically:

    -f .../GSM8K_Loss_prepare_independent.json  →  .../GSM8K_Loss_prepare_independent_weights.csv
    -f .../GSM8K_Loss_prepare_shared.json       →  .../GSM8K_Loss_prepare_shared_weights.csv

Weighting rules (applied to the raw per-component mean of each config):

    concept_loss_weight   = 1.0                      if raw_concept < T
                          = T / raw_concept          otherwise     (anchor weighted at ~T)
    reasoning_loss_weight = 1.0                      always        (keep raw unchanged)

Default target ``T = 10`` matches
``loss-weights-analysis-gsm8k.md`` §17.1 ("keep weighted concept ≈ 10,
leave reasoning unchanged"). Override via ``-t / --target``.

Priority hierarchy (weighted contribution, largest first):
    concept    (anchored at ~T)
    reasoning  (raw, typically 4 - 8)

Output:
  1. A table on stdout listing raw losses, computed weights, and the
     resulting weighted losses for every config in the JSON, sorted
     by model size and pyramid level, followed by copy-ready YAML
     weight blocks.
  2. A ``<stem>_weights.csv`` file written next to the input JSON
     (e.g. ``GSM8K_Loss_prepare_independent.json`` →
     ``GSM8K_Loss_prepare_independent_weights.csv``) containing one
     flat row per config with raw / weight / weighted values for
     direct downstream analysis.
  3. A dataset-aware analytical summary block, dispatched from the
     ``dataset`` field encoded in each JSON key
     (``{dataset}/train_predictor_...``):
       - GSM8K → `loss-weights-analysis-gsm8k.md` §17 (per-config
         concept weights; reasoning stays at 1.0).
       - MATH  → fallback generic summary (same layout; update when
         a MATH-specific analysis document is written).

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
# Anchor for the concept MSE; weighted concept will not exceed this
# unless the raw value is already below it (in which case the weight
# stays at 1.0). Default matches loss-weights-analysis-gsm8k.md §17.
CONCEPT_TARGET_DEFAULT = 10.0


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


VALID_MODES: tuple[str, ...] = ("shared", "independent")


# Config key layout in <dataset>_Loss_prepare_<mode>.json:
#   "{dataset}/train_{module}_{model}_{level}level[_{mode}]"
# e.g. "GSM8K/train_predictor_Qwen2.5-0.5B_2level_independent"
#      "GSM8K/train_predictor_Qwen3-0.6B_4level_shared"
# The trailing ``_{mode}`` suffix is optional for robustness against
# older JSONs; when absent the mode falls back to CLI ``-M`` or the
# mode inferred from the filename stem.
_KEY_HEAD = re.compile(r"^(?P<dataset>[^/]+)/train_(?P<module>[^_]+)_(?P<rest>.+)$")
_KEY_TAIL = re.compile(
    r"^(?P<model>.+)_(?P<level>\d+)level" r"(?:_(?P<mode>shared|independent))?$"
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the predictor loss-weight computation script."""
    p = argparse.ArgumentParser(
        description=(
            "Compute per-config Predictor loss weights from a "
            "<dataset>_Loss_prepare_<mode>.json results file."
        ),
    )
    p.add_argument(
        "-f",
        "--file",
        required=True,
        type=Path,
        help=(
            "Path to a Loss_prepare_<mode>.json file produced by "
            "examples/RunResults/loss_predictor_prepare.py."
        ),
    )
    p.add_argument(
        "-t",
        "--target",
        type=float,
        default=CONCEPT_TARGET_DEFAULT,
        help=(
            "Concept-loss anchor target. "
            f"Default {CONCEPT_TARGET_DEFAULT:g} (weighted concept ≈ {CONCEPT_TARGET_DEFAULT:g})."
        ),
    )
    p.add_argument(
        "-M",
        "--mode",
        choices=VALID_MODES,
        default=None,
        help=(
            "Filter / default mode. If given, only entries whose "
            "mode matches (from key suffix, 'mode' field, or inferred "
            "from filename) are kept. If omitted, all recorded modes "
            "are processed."
        ),
    )
    return p.parse_args()


def cap_weight(raw: float, cap: float) -> float:
    """Return the weight such that ``raw * weight`` does not exceed ``cap``.

    When ``raw < cap`` the component is already below the anchor and
    the weight stays at ``1.0`` — the loss is kept at its natural
    scale. When ``raw >= cap`` the weight is ``cap / raw`` so the
    weighted value sits at ``cap``. ``raw <= 0`` is degenerate and
    returns ``1.0`` (there is nothing to cap).
    """
    if raw <= 0.0 or raw < cap:
        return 1.0
    return cap / raw


def _infer_mode_from_filename(path: Path) -> str | None:
    """Return ``'shared'`` or ``'independent'`` if the stem ends in either."""
    stem = path.stem.lower()
    for m in VALID_MODES:
        if stem.endswith(f"_{m}"):
            return m
    return None


def parse_entry_key(key: str, default_mode: str | None) -> dict | None:
    """Split a Loss_prepare JSON key into its structured components.

    Mode resolution (first non-None wins):
      1. ``_{mode}`` suffix embedded in the key itself.
      2. ``default_mode`` — typically inferred from filename or from
         the per-entry ``entry['mode']`` by the caller.
    """
    m_head = _KEY_HEAD.match(key)
    if m_head is None:
        return None
    m_tail = _KEY_TAIL.match(m_head.group("rest"))
    if m_tail is None:
        return None
    mode = m_tail.group("mode") or default_mode
    return {
        "dataset": m_head.group("dataset"),
        "module": m_head.group("module"),
        "model": m_tail.group("model"),
        "level": int(m_tail.group("level")),
        "mode": mode,
    }


def compute_row(
    key: str,
    entry: dict,
    target: float,
    fallback_mode: str | None,
) -> dict | None:
    """Build a single report row from a Loss_prepare JSON entry.

    Strict fail-fast access: ``entry["stats"]["raw"][component]["mean"]``
    must exist for concept and reasoning. Missing keys raise
    ``KeyError`` at the caller.
    """
    # Prefer per-entry 'mode' field when present; fall back to the
    # filename-inferred or CLI-supplied default.
    entry_mode = entry.get("mode")
    default_mode = entry_mode if entry_mode in VALID_MODES else fallback_mode

    meta = parse_entry_key(key, default_mode)
    if meta is None:
        return None

    raw = entry["stats"]["raw"]
    r_concept = float(raw["concept"]["mean"])
    r_rea = float(raw["reasoning"]["mean"])

    w_concept = cap_weight(r_concept, target)
    w_rea = 1.0

    return {
        **meta,
        "key": key,
        # raw per-component means
        "r_concept": r_concept,
        "r_rea": r_rea,
        # computed per-config weights
        "w_concept": w_concept,
        "w_rea": w_rea,
        # resulting weighted per-component values
        "wt_concept": r_concept * w_concept,
        "wt_rea": r_rea * w_rea,
    }


def print_header(
    file_path: Path, n_entries: int, target: float, mode_filter: str | None
) -> None:
    """Print a short banner describing the source JSON and entry count."""
    print(f"Source : {file_path}")
    print(f"Entries: {n_entries}")
    mode_str = mode_filter if mode_filter else "all"
    print(
        f"Rules  : concept target={target:g}, reasoning=1.0, " f"mode filter={mode_str}"
    )
    print()


def print_table(rows: list[dict]) -> None:
    """Print a wide table of raw / weight / weighted values per config."""
    cols = [
        ("model", "<", 14),
        ("L", ">", 3),
        ("mode", "<", 12),
        ("raw_concept", ">", 12),
        ("raw_rea", ">", 9),
        ("w_concept", ">", 10),
        ("w_rea", ">", 7),
        ("wt_concept", ">", 11),
        ("wt_rea", ">", 9),
        ("wt_total", ">", 9),
    ]

    def fmt_header() -> str:
        return " ".join(f"{name:{align}{w}}" for name, align, w in cols)

    print(fmt_header())
    print(" ".join("-" * w for _, _, w in cols))

    for r in rows:
        wt_total = r["wt_concept"] + r["wt_rea"]
        mode = r["mode"] if r["mode"] is not None else "?"
        vals = [
            f"{r['model']:<14}",
            f"{r['level']:>3}",
            f"{mode:<12}",
            f"{r['r_concept']:>12.3f}",
            f"{r['r_rea']:>9.3f}",
            f"{r['w_concept']:>10.5f}",
            f"{r['w_rea']:>7.2f}",
            f"{r['wt_concept']:>11.3f}",
            f"{r['wt_rea']:>9.3f}",
            f"{wt_total:>9.3f}",
        ]
        print(" ".join(vals))


# Column order used in both the on-screen table and the CSV export.
# Keep these in sync so the CSV can be diff-ed against the printed table.
CSV_COLUMNS: list[str] = [
    "dataset",
    "module",
    "mode",
    "model",
    "level",
    "key",
    "raw_concept",
    "raw_reasoning",
    "w_concept",
    "w_reasoning",
    "wt_concept",
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
                    "mode": r["mode"] if r["mode"] is not None else "",
                    "model": r["model"],
                    "level": r["level"],
                    "key": r["key"],
                    "raw_concept": f"{r['r_concept']:.6f}",
                    "raw_reasoning": f"{r['r_rea']:.6f}",
                    "w_concept": f"{r['w_concept']:.6f}",
                    "w_reasoning": f"{r['w_rea']:.6f}",
                    "wt_concept": f"{r['wt_concept']:.6f}",
                    "wt_reasoning": f"{r['wt_rea']:.6f}",
                    "wt_total": f"{r['wt_concept'] + r['wt_rea']:.6f}",
                }
            )


def print_weight_blocks(rows: list[dict]) -> None:
    """Print compact YAML-style weight blocks, one per config key.

    The blocks are meant to be read / copied by hand into the
    ``training.loss_weights`` section of the corresponding YAML.
    Concept weight is emitted with an extra decimal so very small
    values (e.g. 0.00962) don't round to zero when trimmed to 4
    decimal places.
    """
    print()
    print("=" * 70)
    print("Per-config weights (copy into training.loss_weights):")
    print("=" * 70)
    for r in rows:
        mode = r["mode"] if r["mode"] is not None else "?"
        print(
            f"# {r['key']}  "
            f"(mode={mode}, raw: concept={r['r_concept']:.2f}, "
            f"rea={r['r_rea']:.2f})"
        )
        print(
            f"  concept_loss_weight:   {r['w_concept']:.5f}"
            f"    # weighted \u2248 {r['wt_concept']:.3f}"
        )
        print(
            f"  reasoning_loss_weight: {r['w_rea']:.4f}"
            f"    # weighted \u2248 {r['wt_rea']:.3f}"
        )
        print()


def _per_level_concept_summary(
    rows: list[dict],
    target: float,
) -> list[tuple[int, float, float]]:
    """Return ``(L, mu_concept(L), w_concept*(L))`` triples, sorted by L.

    Per-level reduction across models within the current rowset. Useful
    for eyeballing L-dependence of the concept magnitude (which is
    strongly monotone in L — see gsm8k §16.1).
    """
    levels = sorted({r["level"] for r in rows})
    out: list[tuple[int, float, float]] = []
    for L in levels:
        cell = [r["r_concept"] for r in rows if r["level"] == L]
        if not cell:
            continue
        mu = sum(cell) / len(cell)
        out.append((L, mu, cap_weight(mu, target)))
    return out


def _per_model_concept_summary(
    rows: list[dict],
    target: float,
) -> list[tuple[str, float, float]]:
    """Return ``(model, mu_concept(m), w_concept*(m))`` triples, sorted by size.

    Per-model reduction across levels within the current rowset. Use
    with caution: concept magnitude is **strongly L-dependent**
    (see gsm8k §15), so the per-model closed form is informational
    only — per-config weights are the correct thing to deploy.
    """
    models = sorted(
        {r["model"] for r in rows},
        key=lambda m: MODEL_SIZE_ORDER.get(m, 99),
    )
    out: list[tuple[str, float, float]] = []
    for m in models:
        cell = [r["r_concept"] for r in rows if r["model"] == m]
        if not cell:
            continue
        mu = sum(cell) / len(cell)
        out.append((m, mu, cap_weight(mu, target)))
    return out


def _report_generic_summary(rows: list[dict], target: float, dataset: str) -> None:
    """Emit per-level and per-model concept summaries for ``dataset``."""
    mode = rows[0]["mode"] if rows and rows[0]["mode"] else "?"

    print()
    print("=" * 70)
    print(
        f"{dataset} (mode={mode}) — per-level concept weights "
        f"(averaged across models)"
    )
    print("=" * 70)
    print(f"  {'L':>2}  {'mu_concept(L)':>14}  {'w_concept*(L)':>14}")
    print(f"  {'-' * 2:>2}  {'-' * 14:>14}  {'-' * 14:>14}")
    for L, mu, w in _per_level_concept_summary(rows, target):
        print(f"  {L:>2}  {mu:>14.4f}  {w:>14.5f}")

    print()
    print("=" * 70)
    print(
        f"{dataset} (mode={mode}) — per-model concept summary "
        f"(averaged across levels; informational only — concept is "
        f"L-dependent)"
    )
    print("=" * 70)
    print(f"  {'model':<14}  {'mu_concept(m)':>14}  {'w_concept*(m)':>14}")
    print(f"  {'-' * 14:<14}  {'-' * 14:>14}  {'-' * 14:>14}")
    for m, mu, w in _per_model_concept_summary(rows, target):
        print(f"  {m:<14}  {mu:>14.4f}  {w:>14.5f}")


def report_gsm8k_summary(rows: list[dict], target: float) -> None:
    """Emit the GSM8K Part III summary (loss-weights-analysis-gsm8k.md §17).

    The predictor concept weight is **per-config** (since concept is
    strongly L-dependent). We still print per-level and per-model
    reductions so the reader can see the dominant axes at a glance.
    """
    _report_generic_summary(rows, target, "GSM8K")


def report_math_summary(rows: list[dict], target: float) -> None:
    """Emit a generic MATH summary pending a dedicated design document."""
    _report_generic_summary(rows, target, "MATH")


# Dataset-aware dispatch table: maps the ``dataset`` field parsed from
# each Loss_prepare JSON key to the summary-emission function that
# matches the per-dataset design document.
DATASET_REPORTERS: dict[str, callable] = {
    "GSM8K": report_gsm8k_summary,
    "MATH": report_math_summary,
}


def dispatch_dataset_summary(rows: list[dict], target: float) -> None:
    """Group rows by dataset and invoke the matching reporter for each.

    In practice a single ``<dataset>_Loss_prepare_<mode>.json`` contains
    one dataset, so this produces exactly one summary block. The
    grouping makes the code robust to future mixed files without
    special-casing.
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
        reporter(ds_rows, target)


def main() -> int:
    """CLI entry point: recompute predictor loss weights from Loss_prepare JSON."""
    args = parse_args()
    if not args.file.is_file():
        print(f"[ERROR] File not found: {args.file}", file=sys.stderr)
        return 1

    if args.target <= 0.0:
        print(
            f"[ERROR] --target must be positive; got {args.target!r}.",
            file=sys.stderr,
        )
        return 1

    data = json.loads(args.file.read_text(encoding="utf-8"))

    # Fallback mode resolution order:
    #   (1) explicit --mode CLI arg
    #   (2) mode suffix inferred from filename stem
    #   (3) per-entry 'mode' field (handled inside compute_row)
    fallback_mode = args.mode or _infer_mode_from_filename(args.file)

    rows: list[dict] = []
    for key, entry in data.items():
        row = compute_row(key, entry, args.target, fallback_mode)
        if row is not None:
            rows.append(row)

    if not rows:
        print(
            "[ERROR] No parsable entries found. Expected keys like "
            "'DATASET/train_predictor_MODEL_Nlevel[_MODE]' (e.g. "
            "'GSM8K/train_predictor_Qwen2.5-0.5B_2level_independent').",
            file=sys.stderr,
        )
        return 1

    # Apply mode filter defensively: if the user asked for a specific
    # mode, drop any entries that don't match (protects against
    # hand-merged JSONs and keeps the summary coherent).
    if args.mode is not None:
        before = len(rows)
        rows = [r for r in rows if r["mode"] == args.mode]
        dropped = before - len(rows)
        if dropped > 0:
            print(
                f"[INFO] --mode={args.mode} filtered out {dropped} entries "
                f"of other modes.",
                file=sys.stderr,
            )
        if not rows:
            print(
                f"[ERROR] No entries with mode={args.mode!r} remain.",
                file=sys.stderr,
            )
            return 1

    rows.sort(
        key=lambda r: (
            MODEL_SIZE_ORDER.get(r["model"], 99),
            r["level"],
            r["mode"] or "",
        )
    )

    print_header(args.file, len(rows), args.target, args.mode)
    print_table(rows)
    print_weight_blocks(rows)
    # Dataset-aware analytical summary (GSM8K §17 layout; generic
    # fallback otherwise).
    dispatch_dataset_summary(rows, args.target)

    # Emit a CSV sidecar next to the input JSON. Name: <stem>_weights.csv
    csv_path = args.file.with_name(args.file.stem + "_weights.csv")
    write_csv(rows, csv_path)
    print(f"[CSV] Wrote {len(rows)} rows to {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
