"""Download experiment results (checkpoints + logs) from remote server via SCP.

Usage:
    python3 examples/GetResults/run_scp.py -m <module> -e <experiment>

Arguments:
    -m / --module      Module name: "builder" or "predictor"
    -e / --experiment  Experiment name (directory under EXPERIMENT/nlcpV4/<module>/)

Behavior:
    1. Build full remote & local paths from the two args.
    2. Check whether the local target already has non-empty `checkpoints/`
       and `logs/` directories.
    3. If both exist and contain files -> print "already exists" and exit.
    4. Otherwise -> SCP the following from remote:
         - checkpoints/checkpoint_best_eval.pt
         - checkpoints/checkpoint_best.pt
         - logs/  (entire directory, recursive)

Example:
    python3 examples/GetResults/run_scp.py -m builder \\
        -e GSM8K_Qwen2.5-0.5B_6level
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# ─── Default remote / local base paths ───────────────────────────────────
REMOTE_HOST = "sjia@10.123.4.30"
REMOTE_BASE = "/home/sjia/projects/Reasoning-Autoregressive-Modeling/EXPERIMENT/nlcpV4"
LOCAL_BASE = Path("./EXPERIMENT/nlcpV4")

# ─── Files to fetch (relative to <module>/<experiment>/) ─────────────────
CHECKPOINT_FILES = [
    "checkpoints/checkpoint_best_eval.pt",
    "checkpoints/checkpoint_best.pt",
]
LOGS_DIR = "logs"

VALID_MODULES = {"builder", "predictor"}


def _has_content(directory: Path) -> bool:
    """True iff directory exists and contains at least one file/subdir."""
    return directory.is_dir() and any(directory.iterdir())


def _run_scp(remote_src: str, local_dst: Path, recursive: bool = False) -> int:
    """Run scp; return its exit code. Prints the command for transparency."""
    local_dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["scp"]
    if recursive:
        cmd.append("-r")
    cmd.extend([remote_src, str(local_dst)])
    print(f"  $ {' '.join(cmd)}")
    return subprocess.call(cmd)


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
        help="Experiment directory name under EXPERIMENT/nlcpV4/<module>/.",
    )
    args = parser.parse_args()

    module: str = args.module
    experiment: str = args.experiment

    # ── Step 1: Build full remote & local paths ──────────────────────
    remote_path = f"{REMOTE_BASE}/{module}/{experiment}"
    local_path = LOCAL_BASE / module / experiment

    print("=" * 70)
    print(f"Module     : {module}")
    print(f"Experiment : {experiment}")
    print(f"Remote     : {REMOTE_HOST}:{remote_path}")
    print(f"Local      : {local_path}")
    print("=" * 70)

    # ── Step 2: Per-item idempotent check ────────────────────────────
    # Each artifact is checked independently so we never overwrite or
    # nest an existing directory (scp -r onto an existing directory
    # would create `.../logs/logs/` instead of merging).
    local_logs_dir = local_path / "logs"

    # Classify each checkpoint file as present / missing.
    ckpt_to_fetch: list[str] = []
    for rel_path in CHECKPOINT_FILES:
        local_file = local_path / rel_path
        if local_file.is_file() and local_file.stat().st_size > 0:
            print(f"[SKIP] {rel_path} already exists at {local_file}")
        else:
            ckpt_to_fetch.append(rel_path)

    # Logs: only fetch if the directory is missing or empty.
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

    # ── Step 3: Copy the missing items only ──────────────────────────
    print("[FETCH] Downloading missing items from remote...")

    failures: list[str] = []

    # Fetch individual checkpoint files that are missing.
    for rel_path in ckpt_to_fetch:
        remote_src = f"{REMOTE_HOST}:{remote_path}/{rel_path}"
        local_dst = local_path / rel_path
        print(f"\n-> {rel_path}")
        rc = _run_scp(remote_src, local_dst, recursive=False)
        if rc != 0:
            print(f"   [WARN] scp returned exit code {rc} for {rel_path}")
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
            print(f"   [WARN] scp returned exit code {rc} for {LOGS_DIR}/")
            failures.append(f"{LOGS_DIR}/")

    print("\n" + "=" * 70)
    if failures:
        print(f"[DONE WITH WARNINGS] Failed items: {failures}")
        return 1
    print(f"[DONE] All files downloaded successfully to: {local_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
