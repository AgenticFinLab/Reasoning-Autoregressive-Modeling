"""Download experiment results (checkpoints + logs) from remote server via SCP.

Usage:
    python3 examples/GetResults/run_scp.py -m <module> -e <experiment>
    python3 examples/GetResults/run_scp.py -m <module> -e all
    python3 examples/GetResults/run_scp.py -m <module> -e all -i checkpoints
    python3 examples/GetResults/run_scp.py -m <module> -e all -i 'checkpoints/*best.pt'

Arguments:
    -m / --module      Module name: "builder" or "predictor"
    -e / --experiment  Experiment name (directory under EXPERIMENT/nlcpV4/<module>/)
                       Pass "all" to iterate over every experiment discovered
                       recursively from configs/nlcpV4/**/train_<module>_*.yml
                       (experiment names are read from each config's
                       log.save_folder, so nested layouts like
                       configs/nlcpV4/GSM8K/AutoWeighted/ are supported).
    -i / --ignore      Glob pattern (relative to the experiment dir) of
                       artifacts to skip entirely. May be given multiple
                       times. Examples:
                         -i checkpoints               -> skip whole dir
                         -i checkpoints/*best.pt      -> skip by glob
                         -i logs                      -> skip the logs dir

Behavior:
    1. Build full remote & local paths from the two args.
    2. Drop any artifact whose relative path matches a `-i` pattern.
    3. For each remaining artifact (each checkpoint file + logs/), check
       the local copy independently. Present items are skipped with [SKIP].
    4. If everything is already local or ignored -> exit 0 with [DONE].
    5. Otherwise -> SCP the missing items from remote. Missing remote files
       are treated as warnings (scp non-zero exit); the loop keeps going.

Example:
    python3 examples/GetResults/run_scp.py -m builder \\
        -e GSM8K_Qwen2.5-0.5B_6level
    python3 examples/GetResults/run_scp.py -m builder -e all
    python3 examples/GetResults/run_scp.py -m builder -e all -i checkpoints
"""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path

import yaml

# --- Default remote / local base paths ---------------------------------
REMOTE_HOST = "sjia@10.123.4.30"
REMOTE_BASE = "/home/sjia/projects/Reasoning-Autoregressive-Modeling/EXPERIMENT/nlcpV4"
LOCAL_BASE = Path("./EXPERIMENT/nlcpV4")
CONFIGS_ROOT = Path("./configs/nlcpV4")

# --- Files to fetch (relative to <module>/<experiment>/) ----------------
CHECKPOINT_FILES = [
    "checkpoints/checkpoint_best_eval.pt",
    "checkpoints/checkpoint_best.pt",
]
LOGS_DIR = "logs"

VALID_MODULES = {"builder", "predictor"}
ALL_KEYWORD = "all"


def _has_content(directory: Path) -> bool:
    """True iff directory exists and contains at least one file/subdir."""
    return directory.is_dir() and any(directory.iterdir())


def _cleanup_empty_dir(directory: Path) -> bool:
    """Remove `directory` if it exists and is empty. Returns True if removed.

    Used to roll back empty directories left behind when scp failed (remote
    file missing). Leaving empty local dirs is misleading, so we prune them.
    """
    if directory.is_dir() and not any(directory.iterdir()):
        try:
            directory.rmdir()
            return True
        except OSError:
            return False
    return False


def _run_scp(remote_src: str, local_dst: Path, recursive: bool = False) -> int:
    """Run scp; return its exit code. Prints the command for transparency."""
    local_dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["scp"]
    if recursive:
        cmd.append("-r")
    cmd.extend([remote_src, str(local_dst)])
    print(f"  $ {' '.join(cmd)}")
    return subprocess.call(cmd)


def _matches_ignore(rel_path: str, patterns: list[str]) -> str | None:
    """Return the first pattern that matches ``rel_path``, or None.

    A pattern matches when:
      * ``rel_path`` equals the pattern exactly, OR
      * the pattern is a parent directory of ``rel_path`` (e.g.
        pattern="checkpoints" matches "checkpoints/foo.pt"), OR
      * ``fnmatch.fnmatchcase`` matches the pattern against ``rel_path``
        (supports globs like ``checkpoints/*best.pt``).
    """
    rp = rel_path.strip("/")
    for p in patterns:
        pat = p.strip("/")
        if not pat:
            continue
        if rp == pat or rp.startswith(pat + "/"):
            return p
        if fnmatch.fnmatchcase(rp, pat):
            return p
    return None


def discover_experiments(module: str) -> list[str]:
    """Scan configs/nlcpV4/**/train_<module>_*.yml -> experiment names.

    The experiment name is taken from each config's ``log.save_folder``
    basename, which is the exact directory the trainer creates on disk.
    Reading the source of truth (instead of deriving from file paths)
    makes discovery robust to nested config layouts such as
    ``configs/nlcpV4/GSM8K/AutoWeighted/train_builder_*.yml``, which
    map to experiments named ``GSM8K_<model>_<level>level_AutoWeighted``.

    Duplicate ``save_folder`` values across configs are collapsed so each
    experiment is processed at most once.
    """
    if not CONFIGS_ROOT.is_dir():
        print(f"[WARN] Configs root not found: {CONFIGS_ROOT}")
        return []

    prefix = f"train_{module}_"
    seen: set[str] = set()
    experiments: list[str] = []
    for yml in sorted(CONFIGS_ROOT.rglob(f"{prefix}*.yml")):
        with yml.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        save_folder = cfg["log"]["save_folder"]
        name = Path(save_folder).name
        if name in seen:
            print(f"[WARN] Duplicate experiment '{name}' from {yml}; skipping.")
            continue
        seen.add(name)
        experiments.append(name)
    return experiments


def process_experiment(
    module: str, experiment: str, ignore_patterns: list[str] | None = None
) -> int:
    """Process a single experiment. Returns 0 on success (incl. all-skipped),
    non-zero when at least one artifact failed to transfer."""
    ignore_patterns = ignore_patterns or []
    remote_path = f"{REMOTE_BASE}/{module}/{experiment}"
    local_path = LOCAL_BASE / module / experiment

    print("=" * 70)
    print(f"Module     : {module}")
    print(f"Experiment : {experiment}")
    print(f"Remote     : {REMOTE_HOST}:{remote_path}")
    print(f"Local      : {local_path}")
    if ignore_patterns:
        print(f"Ignore     : {ignore_patterns}")
    print("=" * 70)

    # --- Per-item idempotent check -------------------------------------
    # Each artifact is checked independently so we never overwrite or
    # nest an existing directory (scp -r onto an existing directory
    # would create `.../logs/logs/` instead of merging).
    local_logs_dir = local_path / "logs"

    ckpt_to_fetch: list[str] = []
    for rel_path in CHECKPOINT_FILES:
        hit = _matches_ignore(rel_path, ignore_patterns)
        if hit is not None:
            print(f"[IGNORE] {rel_path} (matched -i {hit!r})")
            continue
        local_file = local_path / rel_path
        if local_file.is_file() and local_file.stat().st_size > 0:
            print(f"[SKIP] {rel_path} already exists at {local_file}")
        else:
            ckpt_to_fetch.append(rel_path)

    logs_ignored = _matches_ignore(LOGS_DIR, ignore_patterns)
    if logs_ignored is not None:
        print(f"[IGNORE] {LOGS_DIR}/ (matched -i {logs_ignored!r})")
        fetch_logs = False
    else:
        fetch_logs = not _has_content(local_logs_dir)
        if not fetch_logs:
            print(
                f"[SKIP] {LOGS_DIR}/ already exists and is non-empty at "
                f"{local_logs_dir}"
            )

    if not ckpt_to_fetch and not fetch_logs:
        print(
            "\n[DONE] All target artifacts already present locally or "
            "ignored. Remove the specific file/directory to force a "
            "re-download."
        )
        return 0

    # --- Copy the missing items ----------------------------------------
    print("[FETCH] Downloading missing items from remote...")

    failures: list[str] = []

    for rel_path in ckpt_to_fetch:
        remote_src = f"{REMOTE_HOST}:{remote_path}/{rel_path}"
        local_dst = local_path / rel_path
        print(f"\n-> {rel_path}")
        rc = _run_scp(remote_src, local_dst, recursive=False)
        if rc != 0:
            print(
                f"   [MISS] remote {rel_path} unavailable or scp failed "
                f"(exit={rc}); skipping this item."
            )
            failures.append(rel_path)

    # Fetch the entire logs/ directory if missing/empty.
    # IMPORTANT: `scp -r src dst/` where dst is an EXISTING directory nests
    # the remote source inside it (creating `.../logs/logs/`). We therefore
    # (a) remove an empty stub dir if it exists, and (b) copy the remote
    # `logs` directory INTO `local_path` (the parent), so scp recreates
    # `logs/` at the correct level.
    if fetch_logs:
        remote_logs_src = f"{REMOTE_HOST}:{remote_path}/{LOGS_DIR}"
        print(f"\n-> {LOGS_DIR}/ (recursive)")
        if local_logs_dir.exists() and not any(local_logs_dir.iterdir()):
            local_logs_dir.rmdir()
        local_path.mkdir(parents=True, exist_ok=True)
        rc = _run_scp(remote_logs_src, local_path, recursive=True)
        if rc != 0:
            print(
                f"   [MISS] remote {LOGS_DIR}/ unavailable or scp failed "
                f"(exit={rc}); skipping this item."
            )
            failures.append(f"{LOGS_DIR}/")

    print("\n" + "=" * 70)

    # --- Prune empty directories left behind by failed scp -------------
    # When a remote artifact doesn't exist, scp fails after we already
    # created parent directories (scp won't create them itself for file
    # targets). Remove any freshly-created empty folders so the local
    # layout truthfully reflects what was actually fetched.
    for sub in ("checkpoints", LOGS_DIR):
        removed = _cleanup_empty_dir(local_path / sub)
        if removed:
            print(f"[CLEAN] Removed empty {sub}/ (no remote data)")
    if _cleanup_empty_dir(local_path):
        print(f"[CLEAN] Removed empty experiment dir {local_path} (no remote data)")

    if failures:
        print(f"[PARTIAL] {experiment}: missing/failed items: {failures}")
        return 1
    print(f"[DONE] {experiment}: all missing files downloaded to {local_path}")
    return 0


def main() -> int:
    """CLI entry point: parse args and dispatch to process_experiment.

    Returns a shell exit code: 0 on full success (all artifacts fetched
    or already present), 1 when at least one artifact failed to transfer
    in at least one experiment.
    """
    parser = argparse.ArgumentParser(
        description="SCP experiment results from remote server to local machine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--module",
        required=True,
        choices=sorted(VALID_MODULES),
        help="Module name: 'builder' or 'predictor'.",
    )
    parser.add_argument(
        "-e",
        "--experiment",
        required=True,
        help=(
            "Experiment directory name under EXPERIMENT/nlcpV4/<module>/. "
            "Use 'all' to discover experiments recursively from "
            "configs/nlcpV4/**/train_<module>_*.yml (experiment names are "
            "read from each config's log.save_folder)."
        ),
    )
    parser.add_argument(
        "-i",
        "--ignore",
        action="append",
        default=[],
        metavar="PATTERN",
        help=(
            "Artifact pattern to SKIP, relative to the experiment dir. "
            "May be given multiple times. Matching rules: exact path, "
            "parent directory prefix, or fnmatch glob. Examples: "
            "'-i checkpoints' skips the whole checkpoints dir; "
            "'-i checkpoints/*best.pt' skips files matching the glob; "
            "'-i logs' skips the logs directory."
        ),
    )
    args = parser.parse_args()

    module: str = args.module
    experiment: str = args.experiment
    ignore_patterns: list[str] = list(args.ignore or [])

    if experiment == ALL_KEYWORD:
        experiments = discover_experiments(module)
        if not experiments:
            print(
                f"[ERROR] No experiments discovered for module={module} "
                f"under {CONFIGS_ROOT}."
            )
            return 1
        print(
            f"[ALL] Discovered {len(experiments)} experiment(s) for "
            f"module={module}:"
        )
        for e in experiments:
            print(f"  - {e}")
        print()

        partial: list[str] = []
        for e in experiments:
            rc = process_experiment(module, e, ignore_patterns)
            if rc != 0:
                partial.append(e)
            print()

        print("=" * 70)
        print(
            f"[SUMMARY] module={module}: "
            f"total={len(experiments)} partial={len(partial)}"
        )
        if partial:
            print("Experiments with missing/failed items:")
            for e in partial:
                print(f"  - {e}")
        return 0

    # Single-experiment mode.
    return process_experiment(module, experiment, ignore_patterns)


if __name__ == "__main__":
    sys.exit(main())
