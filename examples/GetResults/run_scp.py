"""Download experiment results (checkpoints + logs) from remote server via SCP.

Usage:
    python3 examples/GetResults/run_scp.py -m <module> -e <experiment>
    python3 examples/GetResults/run_scp.py -m <module> -e all

Arguments:
    -m / --module      Module name: "builder" or "predictor"
    -e / --experiment  Experiment name (directory under EXPERIMENT/nlcpV4/<module>/)
                       Pass "all" to iterate over every experiment discovered
                       from configs/nlcpV4/*/train_<module>_*.yml.

Behavior:
    1. Build full remote & local paths from the two args.
    2. For each artifact (each checkpoint file + logs/), check the local copy
       independently. Present items are skipped with [SKIP].
    3. If everything is already local -> exit 0 with [DONE].
    4. Otherwise -> SCP the missing items from remote. Missing remote files
       are treated as warnings (scp non-zero exit); the loop keeps going.

Example:
    python3 examples/GetResults/run_scp.py -m builder \\
        -e GSM8K_Qwen2.5-0.5B_6level
    python3 examples/GetResults/run_scp.py -m builder -e all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

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


def discover_experiments(module: str) -> list[str]:
    """Scan configs/nlcpV4/<dataset>/train_<module>_<rest>.yml -> experiment names.

    Experiment naming convention (must match log.save_folder in configs):
        {dataset}_{rest}
    e.g. configs/nlcpV4/GSM8K/train_builder_Qwen2.5-0.5B_2level.yml
         -> experiment 'GSM8K_Qwen2.5-0.5B_2level'
    """
    if not CONFIGS_ROOT.is_dir():
        print(f"[WARN] Configs root not found: {CONFIGS_ROOT}")
        return []

    prefix = f"train_{module}_"
    experiments: list[str] = []
    for dataset_dir in sorted(CONFIGS_ROOT.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        for yml in sorted(dataset_dir.glob(f"{prefix}*.yml")):
            rest = yml.stem[len(prefix) :]
            experiments.append(f"{dataset}_{rest}")
    return experiments


def process_experiment(module: str, experiment: str) -> int:
    """Process a single experiment. Returns 0 on success (incl. all-skipped),
    non-zero when at least one artifact failed to transfer."""
    remote_path = f"{REMOTE_BASE}/{module}/{experiment}"
    local_path = LOCAL_BASE / module / experiment

    print("=" * 70)
    print(f"Module     : {module}")
    print(f"Experiment : {experiment}")
    print(f"Remote     : {REMOTE_HOST}:{remote_path}")
    print(f"Local      : {local_path}")
    print("=" * 70)

    # --- Per-item idempotent check -------------------------------------
    # Each artifact is checked independently so we never overwrite or
    # nest an existing directory (scp -r onto an existing directory
    # would create `.../logs/logs/` instead of merging).
    local_logs_dir = local_path / "logs"

    ckpt_to_fetch: list[str] = []
    for rel_path in CHECKPOINT_FILES:
        local_file = local_path / rel_path
        if local_file.is_file() and local_file.stat().st_size > 0:
            print(f"[SKIP] {rel_path} already exists at {local_file}")
        else:
            ckpt_to_fetch.append(rel_path)

    fetch_logs = not _has_content(local_logs_dir)
    if not fetch_logs:
        print(
            f"[SKIP] {LOGS_DIR}/ already exists and is non-empty at "
            f"{local_logs_dir}"
        )

    if not ckpt_to_fetch and not fetch_logs:
        print(
            "\n[DONE] All target artifacts already present locally. "
            "Remove the specific file/directory to force a re-download."
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
            "Use 'all' to discover experiments from "
            "configs/nlcpV4/*/train_<module>_*.yml."
        ),
    )
    args = parser.parse_args()

    module: str = args.module
    experiment: str = args.experiment

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
            rc = process_experiment(module, e)
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
    return process_experiment(module, experiment)


if __name__ == "__main__":
    sys.exit(main())
