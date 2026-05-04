"""Download experiment results (checkpoints + logs) and/or Loss_prepare.json
files from the remote server via SCP.

Usage:
    # ==================================================================
    # A. Single-experiment fetches (the simplest case)
    # ==================================================================

    # A1. Fetch one BUILDER experiment's best checkpoints + logs/.
    #     The experiment dir name is exactly what log.save_folder creates
    #     on the remote (e.g. ``GSM8K_Qwen2.5-0.5B_6level``). Re-running
    #     is idempotent: already-present local files are [SKIP]-ed.
    python3 examples/GetResults/run_scp.py -m builder -e GSM8K_Qwen2.5-0.5B_6level

    # A2. Same, but for a PREDICTOR experiment.
    python3 examples/GetResults/run_scp.py -m predictor -e GSM8K_Qwen2.5-0.5B_6level

    # A3. Fetch a nested-layout experiment (e.g. AutoWeighted variants).
    #     Pass the resolved experiment directory name, NOT the config
    #     path; ``-e all`` below shows how to avoid typing these by hand.
    python3 examples/GetResults/run_scp.py -m builder -e GSM8K_Qwen3-4B_8level_AutoWeighted

    # ==================================================================
    # B. Batch experiment fetches (``-e all``)
    # ==================================================================

    # B1. Fetch EVERY experiment discovered from
    #     configs/nlcpV4/**/train_<module>_*.yml. Experiment names are
    #     read from each config's log.save_folder, so nested layouts
    #     (e.g. GSM8K/AutoWeighted/) are handled transparently.
    python3 examples/GetResults/run_scp.py -m builder -e all

    # B2. Same for predictor. Useful after a sweep of predictor trainings.
    python3 examples/GetResults/run_scp.py -m predictor -e all

    # ==================================================================
    # C. Ignore patterns (``-i``, repeatable)
    # ==================================================================
    # Each -i PATTERN is matched against the artifact's path RELATIVE to
    # <module>/<experiment>/. Three matching rules are tried in order:
    # exact match, parent-directory prefix, or fnmatch glob.

    # C1. Drop the whole checkpoints/ dir (download ONLY logs/).
    #     Handy when you just want training curves / SwanLab metadata.
    python3 examples/GetResults/run_scp.py -m builder -e all -i checkpoints

    # C2. Drop only the ``*best.pt`` files but keep ``*best_eval.pt``.
    #     Pattern uses fnmatch glob semantics.
    python3 examples/GetResults/run_scp.py -m builder -e all -i 'checkpoints/*best.pt'

    # C3. Drop BOTH checkpoint dir and logs dir (metadata-only dry run).
    #     The script will still print [STORAGE] + per-experiment banners,
    #     making this a safe way to preview what WOULD be fetched.
    python3 examples/GetResults/run_scp.py -m builder -e all -i checkpoints -i logs

    # ==================================================================
    # D. Loss_prepare.json fetches (``-d``, repeatable)
    # ==================================================================
    # Remote path layout (produced by loss_prepare.py):
    #   <storage_root>/EXPERIMENT/nlcpV4/<module>/<dataset>_Loss_prepare.json

    # D1. Fetch ONLY GSM8K's Loss_prepare.json for the builder module
    #     (no checkpoints, no logs). Skips silently if already local.
    python3 examples/GetResults/run_scp.py -m builder -d GSM8K

    # D2. Fetch multiple datasets' Loss_prepare.json in one run.
    python3 examples/GetResults/run_scp.py -m builder -d GSM8K -d MATH

    # D3. Auto-discover datasets from
    #     configs/nlcpV4/<dataset>/train_<module>_*.yml. Equivalent to
    #     passing every discovered -d, but zero maintenance.
    python3 examples/GetResults/run_scp.py -m builder -d all

    # D4. Same, but for the predictor module.
    python3 examples/GetResults/run_scp.py -m predictor -d all

    # ==================================================================
    # E. Combined fetches (experiments + Loss_prepare.json in one pass)
    # ==================================================================

    # E1. Fetch every experiment AND every dataset's Loss_prepare.json.
    #     The Loss_prepare pass runs first (fast, small files), then
    #     the experiment pass (slow, large checkpoints).
    python3 examples/GetResults/run_scp.py -m builder -e all -d all

    # E2. Fetch a single experiment + that dataset's Loss_prepare.json.
    python3 examples/GetResults/run_scp.py -m builder -e GSM8K_Qwen2.5-0.5B_6level -d GSM8K

    # E3. Fetch everything EXCEPT checkpoints (logs + Loss_prepare only).
    python3 examples/GetResults/run_scp.py -m builder -e all -d all -i checkpoints

    # ==================================================================
    # F. Non-default storage roots (``-s``) and local redirection
    # ==================================================================
    # The REMOTE base is ALWAYS ``<storage_root>/EXPERIMENT/nlcpV4``.
    # ``-s`` must match the ``-s`` used when training was LAUNCHED on
    # the remote side (same convention as the rest of the project).

    # F1. Remote artifacts live under /Data/ReasoningNLCP on the server.
    #     LOCAL sink is unchanged (``./EXPERIMENT/nlcpV4``).
    python3 examples/GetResults/run_scp.py -m builder -e all -s /Data/ReasoningNLCP

    # F2. Pull to an external disk (e.g. Time Machine / backup drive).
    #     ``--local-base`` overrides ONLY the local side.
    python3 examples/GetResults/run_scp.py -m builder -e all -s /Data/ReasoningNLCP --local-base /Volumes/Backup/ReasoningAR/EXPERIMENT/nlcpV4

    # F3. Pull Loss_prepare.json files from a non-default storage root.
    python3 examples/GetResults/run_scp.py -m builder -d all -s /Data/ReasoningNLCP

Arguments:
    -s / --storage-root   REMOTE storage root (on the SSH target). Listed
                          FIRST because it controls the remote base this
                          script reads from. The remote base becomes
                          ``<storage_root>/EXPERIMENT/nlcpV4``. Default
                          is ``./`` — resolved against the remote
                          user's $HOME. NEVER a hardcoded user-path
                          fallback. Must match the ``-s`` that training
                          on the remote was launched with. Accepts
                          absolute or relative paths.
    -m / --module         Module name: "builder" or "predictor".
    -e / --experiment     Experiment name (directory under
                          EXPERIMENT/nlcpV4/<module>/). Optional when
                          ``-d/--dataset`` is provided.
                          Pass "all" to iterate over every experiment
                          discovered recursively from
                          configs/nlcpV4/**/train_<module>_*.yml
                          (experiment names are read from each config's
                          log.save_folder, so nested layouts like
                          configs/nlcpV4/GSM8K/AutoWeighted/ are supported).
    -d / --dataset        Dataset name. Fetches
                          ``<module>/<dataset>_Loss_prepare.json`` (the
                          output layout of ``loss_prepare.py``). May be
                          given multiple times, or pass ``-d all`` to
                          auto-discover datasets from the immediate
                          child directories of configs/nlcpV4/ that
                          contain at least one ``train_<module>_*.yml``.
                          At least one of ``-e`` / ``-d`` is required.
    -i / --ignore         Glob pattern (relative to the experiment dir)
                          of artifacts to skip entirely. May be given
                          multiple times. Examples:
                            -i checkpoints               -> skip whole dir
                            -i checkpoints/*best.pt      -> skip by glob
                            -i logs                      -> skip the logs dir
    --local-base          Override the LOCAL destination base dir.
                          Default is ``./EXPERIMENT/nlcpV4`` (resolved
                          against current working directory).

Behavior:
    1. Build full remote & local paths from the two args.
    2. Drop any artifact whose relative path matches a `-i` pattern.
    3. For each remaining artifact (each checkpoint file + logs/), check
       the local copy independently. Present items are skipped with [SKIP].
    4. For each -d dataset, check the local Loss_prepare.json copy
       independently; present files are skipped with [SKIP].
    5. If everything is already local or ignored -> exit 0 with [DONE].
    6. Otherwise -> SCP the missing items from remote. Missing remote files
       are treated as warnings (scp non-zero exit); the loop keeps going.
"""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path

import yaml

# --- Default remote / local base paths ---------------------------------
# The remote host is still hardcoded (``REMOTE_HOST``) because this
# tool is SSH-based and pointing at a specific lab machine. The remote
# and local directory bases, however, are now ALWAYS derived from the
# ``-s`` flag (default ``./``) — there is NO silent fallback to
# some hardcoded user-home path on the remote. The resolved bases
# are printed as a ``[STORAGE]`` block at startup so every run is
# self-documenting.
REMOTE_HOST = "sjia@10.123.4.30"
CONFIGS_ROOT = Path("./configs/nlcpV4")
EXPERIMENT_SUBPATH = "EXPERIMENT/nlcpV4"

# --- Files to fetch (relative to <module>/<experiment>/) ----------------
CHECKPOINT_FILES = [
    "checkpoints/checkpoint_best_eval.pt",
    "checkpoints/checkpoint_best.pt",
]
LOGS_DIR = "logs"

VALID_MODULES = {"builder", "predictor"}
ALL_KEYWORD = "all"
LOSS_PREPARE_SUFFIX = "_Loss_prepare.json"


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


def discover_datasets(module: str) -> list[str]:
    """Scan configs/nlcpV4/<dataset>/**/train_<module>_*.yml -> dataset names.

    The dataset is taken from the FIRST directory component under
    ``CONFIGS_ROOT``, mirroring ``loss_prepare.py``'s own discovery
    convention (``configs/nlcpV4/{dataset}/train_{module}_*.yml``).
    Nested layouts like ``configs/nlcpV4/GSM8K/AutoWeighted/*.yml``
    still resolve to dataset = ``GSM8K``.

    Duplicate dataset names are collapsed so each dataset is processed
    at most once.
    """
    if not CONFIGS_ROOT.is_dir():
        print(f"[WARN] Configs root not found: {CONFIGS_ROOT}")
        return []

    prefix = f"train_{module}_"
    seen: set[str] = set()
    datasets: list[str] = []
    for yml in sorted(CONFIGS_ROOT.rglob(f"{prefix}*.yml")):
        rel = yml.relative_to(CONFIGS_ROOT)
        if len(rel.parts) < 2:
            continue
        ds = rel.parts[0]
        if ds in seen:
            continue
        seen.add(ds)
        datasets.append(ds)
    return datasets


def process_loss_prepare(
    module: str,
    dataset: str,
    *,
    remote_base: str,
    local_base: Path,
) -> int:
    """Fetch ``<module>/<dataset>_Loss_prepare.json`` if not already local.

    Mirrors ``process_experiment`` in style: prints a banner, performs an
    idempotent local existence check (non-empty file -> [SKIP]), invokes
    ``scp`` once for the missing file, and returns 0/1 accordingly.

    ``remote_base`` and ``local_base`` are REQUIRED (no default). The
    caller resolves them from ``-s`` / ``--local-base`` and prints the
    ``[STORAGE]`` block so the user always sees which directories are
    being read from / written to.
    """
    filename = f"{dataset}{LOSS_PREPARE_SUFFIX}"
    remote_path = f"{remote_base}/{module}/{filename}"
    local_path = local_base / module / filename

    print("=" * 70)
    print(f"Module      : {module}")
    print(f"Dataset     : {dataset}")
    print(f"LossPrepare : {filename}")
    print(f"Remote      : {REMOTE_HOST}:{remote_path}")
    print(f"Local       : {local_path}")
    print("=" * 70)

    if local_path.is_file() and local_path.stat().st_size > 0:
        print(f"[SKIP] {filename} already exists at {local_path}")
        return 0

    remote_src = f"{REMOTE_HOST}:{remote_path}"
    print(f"[FETCH] -> {filename}")
    rc = _run_scp(remote_src, local_path, recursive=False)
    if rc != 0:
        print(
            f"   [MISS] remote {filename} unavailable or scp failed "
            f"(exit={rc}); skipping this item."
        )
        # scp may have created a zero-byte stub; remove it so a re-run
        # retries instead of [SKIP]-ing an empty placeholder.
        if local_path.exists() and local_path.stat().st_size == 0:
            try:
                local_path.unlink()
            except OSError:
                pass
        return 1
    print(f"[DONE] {filename} downloaded to {local_path}")
    return 0


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
    module: str,
    experiment: str,
    ignore_patterns: list[str],
    *,
    remote_base: str,
    local_base: Path,
) -> int:
    """Process a single experiment. Returns 0 on success (incl. all-skipped),
    non-zero when at least one artifact failed to transfer.

    ``remote_base`` and ``local_base`` are REQUIRED (no default). The
    caller must resolve them from ``-s`` / ``--local-base`` and print
    the ``[STORAGE]`` block so the user always sees which directories
    are being read from / written to."""
    remote_path = f"{remote_base}/{module}/{experiment}"
    local_path = local_base / module / experiment

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
        "-s",
        "--storage-root",
        type=str,
        default="./",
        help=(
            "REMOTE storage root (on the SSH target). The remote base "
            f"becomes ``<storage_root>/{EXPERIMENT_SUBPATH}``. Default "
            "is ``./`` — i.e. the remote user's $HOME on the SSH host "
            "(NO hardcoded user-path fallback). Pass ``-s /Data/<proj>`` "
            "when training on the remote was launched with a matching "
            "``-s``. The resolved remote + local bases are printed as "
            "a ``[STORAGE]`` block at startup so mismatches are caught "
            "before any SSH traffic."
        ),
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
        required=False,
        default=None,
        help=(
            "Experiment directory name under EXPERIMENT/nlcpV4/<module>/. "
            "Use 'all' to discover experiments recursively from "
            "configs/nlcpV4/**/train_<module>_*.yml (experiment names are "
            "read from each config's log.save_folder). Optional when "
            "-d/--dataset is provided; at least one of -e / -d is required."
        ),
    )
    parser.add_argument(
        "-d",
        "--dataset",
        action="append",
        default=[],
        metavar="DATASET",
        help=(
            "Dataset name whose <module>/<dataset>_Loss_prepare.json "
            "should be fetched (the output layout of loss_prepare.py). "
            "May be given multiple times. Pass '-d all' to auto-discover "
            "datasets from configs/nlcpV4/<dataset>/train_<module>_*.yml. "
            "At least one of -e / -d is required."
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
    parser.add_argument(
        "--local-base",
        type=str,
        default="",
        help=(
            "Override the LOCAL destination base directory. Default "
            f"is ``./{EXPERIMENT_SUBPATH}`` (resolved against current "
            "working directory). Use only when downloaded files "
            "should land somewhere other than the project-local "
            "EXPERIMENT/ tree (e.g. an external disk mount)."
        ),
    )
    args = parser.parse_args()

    module: str = args.module
    experiment: str | None = args.experiment
    datasets_arg: list[str] = list(args.dataset)
    ignore_patterns: list[str] = list(args.ignore)

    # At least one of -e / -d must be provided. Enforced here rather
    # than via argparse because BOTH flags are optional individually
    # but one of them is mandatory collectively.
    if experiment is None and not datasets_arg:
        print("[ERROR] At least one of -e/--experiment or -d/--dataset " "is required.")
        return 2

    # Resolve storage bases — both sides are ALWAYS derived from CLI
    # flags (no hardcoded fallback). The remote storage root may be
    # any directory string understood by the remote shell (absolute or
    # relative to remote $HOME). The local side defaults to
    # ``./EXPERIMENT/nlcpV4`` (resolved against current working dir).
    storage_root: str = args.storage_root
    remote_base = f"{storage_root.rstrip('/')}/{EXPERIMENT_SUBPATH}"
    local_base = (
        Path(args.local_base) if args.local_base else Path("./") / EXPERIMENT_SUBPATH
    )

    # Surface the resolved paths up front. Zero ambiguity about where
    # files are pulled from (remote) or written to (local).
    cwd = Path.cwd().resolve()
    _local_abs = local_base.expanduser()
    if not _local_abs.is_absolute():
        _local_abs = (cwd / _local_abs).resolve()
    print(f"[STORAGE] storage_root = {storage_root!r} (cwd={cwd})")
    print(f"[STORAGE]   remote base = {REMOTE_HOST}:{remote_base}")
    print(f"[STORAGE]   local  base = {local_base}")
    print(f"[STORAGE]                 (absolute: {_local_abs})")

    # --- Resolve datasets (expand 'all', dedupe, preserve order) -------
    # Done before experiment processing so the [ALL] listing shows both
    # experiments AND datasets up front.
    resolved_datasets: list[str] = []
    if datasets_arg:
        if ALL_KEYWORD in datasets_arg:
            resolved_datasets = discover_datasets(module)
            if not resolved_datasets:
                print(
                    f"[ERROR] No datasets discovered for module={module} "
                    f"under {CONFIGS_ROOT}."
                )
                return 1
        else:
            seen: set[str] = set()
            for ds in datasets_arg:
                if ds not in seen:
                    seen.add(ds)
                    resolved_datasets.append(ds)
        print(
            f"[DATASETS] {len(resolved_datasets)} dataset(s) for "
            f"module={module}: {resolved_datasets}"
        )
        print()

    # --- Loss_prepare.json fetch pass (runs BEFORE experiments) --------
    # Fast / lightweight so the user sees the JSON land first even when
    # the experiment pass is slow.
    lp_partial: list[str] = []
    for ds in resolved_datasets:
        rc = process_loss_prepare(
            module,
            ds,
            remote_base=remote_base,
            local_base=local_base,
        )
        if rc != 0:
            lp_partial.append(f"{ds}{LOSS_PREPARE_SUFFIX}")
        print()

    # --- Experiment fetch pass (only when -e was provided) -------------
    if experiment is None:
        print("=" * 70)
        print(
            f"[SUMMARY] module={module}: "
            f"loss_prepare_total={len(resolved_datasets)} "
            f"loss_prepare_partial={len(lp_partial)}"
        )
        if lp_partial:
            print("Missing/failed Loss_prepare.json files:")
            for f in lp_partial:
                print(f"  - {f}")
        return 0

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
            rc = process_experiment(
                module,
                e,
                ignore_patterns,
                remote_base=remote_base,
                local_base=local_base,
            )
            if rc != 0:
                partial.append(e)
            print()

        print("=" * 70)
        print(
            f"[SUMMARY] module={module}: "
            f"total={len(experiments)} partial={len(partial)} "
            f"loss_prepare_total={len(resolved_datasets)} "
            f"loss_prepare_partial={len(lp_partial)}"
        )
        if partial:
            print("Experiments with missing/failed items:")
            for e in partial:
                print(f"  - {e}")
        if lp_partial:
            print("Missing/failed Loss_prepare.json files:")
            for f in lp_partial:
                print(f"  - {f}")
        return 0

    # Single-experiment mode.
    return process_experiment(
        module,
        experiment,
        ignore_patterns,
        remote_base=remote_base,
        local_base=local_base,
    )


if __name__ == "__main__":
    sys.exit(main())
