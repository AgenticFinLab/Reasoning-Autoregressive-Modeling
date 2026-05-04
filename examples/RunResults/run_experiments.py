"""Launch multiple training experiments in parallel via tmux sessions.

Purpose:
    Fan out a batch of YAML-configured training runs across independent
    tmux sessions so they execute in parallel without blocking each other
    (and without requiring a job scheduler). One session per experiment.

Convention (matches examples/RunResults/loss_prepare.py):
    * Package is fixed to ``nlcpV4``.
    * ``-m/--module {builder,predictor}`` selects which training script to
      run: ``examples/nlcpV4/train_{module}.py`` — see the ``Usage:``
      comment at the top of every config file for the reference invocation.
    * ``-d/--dataset`` resolves to ``configs/nlcpV4/{dataset}/``.
    * ``-e/--experiments`` is one or more YAML files under that directory.
      Each entry may be:
        - a bare filename                    -> configs/nlcpV4/{d}/foo.yml
        - a relative path (nested subdir)    -> configs/nlcpV4/{d}/AutoWeighted/foo.yml
        - an absolute path                   -> used verbatim
      So AutoWeighted/ variants live in exactly the same CLI as baseline.

Tmux session naming:
    ``{module}-{dataset}-{experiment_label}`` where ``experiment_label``
    is the config path relative to ``configs/nlcpV4/{dataset}/`` with
    directory separators replaced by ``-`` and the ``.yml`` stripped.

GPU scheduling (new):
    Before launching, the runner tries to:
      1. Enumerate free GPUs via ``nvidia-smi`` (skipped on machines
         without NVIDIA drivers, e.g. macOS; in that case scheduling
         is disabled with a warning and all experiments are launched
         without ``CUDA_VISIBLE_DEVICES`` pinning).
      2. Estimate each experiment's peak memory from its YAML:
            backbone size (parsed from ``reason_model_name``)
          × bytes-per-elem (4 for fp32, 2 for bf16/fp16)
          + Adam state (×3 of weights for full FT; ~0.2× for LoRA/frozen)
          + activation memory ≈ B × L × D × bytes × 30  (conservative)
          × ``--mem-safety-factor`` (default 1.5)
         This is a rule-of-thumb upper bound — override with
         ``--mem-per-exp-mb`` when you know the real number.
      3. Best-fit-decreasing pack: sort experiments by estimated memory
         DESC, then assign each to the GPU with the smallest remaining
         free budget that still fits. This keeps big GPUs open for big
         jobs. Each assigned session is launched with
         ``CUDA_VISIBLE_DEVICES={gpu_index}`` inside its tmux command.
      4. Experiments that don't fit go to a queue. With ``--wait-for-gpu``
         the launcher polls ``nvidia-smi`` every ``--poll-interval``
         seconds and places queued jobs as memory frees up. Without it,
         overflow jobs are skipped with a clear report.
      5. Reservation tracking survives across passes. Every assignment
         records ``(gpu_idx, mem_mb, committed_at)`` in a run-wide
         ``_Reservation`` list. Each new scheduling pass recomputes
         ``gpus[].reserved_mb`` from those records so the freshly
         polled ``free_mb`` from nvidia-smi is NEVER interpreted as
         "this GPU is empty" during the 30–120 s window where a
         launched session has not yet allocated its memory. After
         ``--warmup-seconds`` a reservation is dropped and nvidia-smi
         is trusted instead — this prevents reservations from
         accumulating forever as long-running jobs continue. This is
         the fix for the classic "tmux launched but process not yet
         allocated" OOM race.
      6. Pre-flight hard-bail: if any experiment's estimated memory
         exceeds the largest visible GPU's effective budget, the
         launcher refuses to start. Otherwise ``--wait-for-gpu``
         would spin forever on an impossible job.

    Opt-out: pass ``--no-gpu-schedule`` to disable the whole mechanism
    and launch everything in parallel (the pre-scheduler behaviour).

Simple mode (``--one-per-gpu``):
    If you don't want memory packing and just want "N GPUs = N concurrent
    experiments, queue the rest", pass ``--one-per-gpu``. Semantics:
      * Memory estimation is skipped entirely.
      * Each experiment is pinned to exactly one idle GPU.
      * A GPU counts as idle iff ``free_mb / total_mb >=
        --gpu-idle-mem-fraction`` (default 0.9 — i.e., nothing else is
        using it) AND we have no active reservation on it.
      * Experiments past the idle-GPU count wait in FIFO. The launcher
        polls every ``--poll-interval`` seconds; as soon as a GPU goes
        idle again (because a previous session finished and released
        its memory), the next queued experiment is launched on it.
      * ``--wait-for-gpu`` is implicit in this mode — the launcher
        stays alive until the queue drains.
    This is the recommended mode when each experiment saturates a GPU
    on its own (7B+ models, full FT). It sidesteps OOM by design.

Usage (one-per-gpu):
    # 4 GPUs, 8 experiments: 4 run immediately, 4 wait; as each
    # finishes, the next queued starts on the freed GPU.
    python3 examples/RunResults/run_experiments.py -m builder -d GSM8K --one-per-gpu -e exp1.yml exp2.yml exp3.yml exp4.yml exp5.yml exp6.yml exp7.yml exp8.yml

    # Launch two baseline experiments in parallel:
    python3 examples/RunResults/run_experiments.py -m builder -d GSM8K -e train_builder_Qwen2.5-0.5B_2level.yml train_builder_Qwen2.5-0.5B_4level.yml

    # Mix baseline + AutoWeighted variants in one launch:
    python3 examples/RunResults/run_experiments.py -m builder -d GSM8K -e train_builder_Qwen2.5-0.5B_2level.yml AutoWeighted/train_builder_Qwen2.5-0.5B_2level.yml

    # Keep the launcher alive and drain a large queue as GPUs free up:
    python3 examples/RunResults/run_experiments.py -m builder -d GSM8K -e AutoWeighted/*.yml --wait-for-gpu

    # Preview the plan (incl. GPU assignments) without touching tmux:
    python3 examples/RunResults/run_experiments.py -m builder -d GSM8K -e AutoWeighted/*.yml --dry-run

    # Restrict scheduling to GPUs 0 and 2:
    python3 examples/RunResults/run_experiments.py -m builder -d GSM8K -e AutoWeighted/*.yml --gpus 0,2

    # Redirect every child trainer's relative log paths under a storage
    # root. The launcher forwards ``-s /Data/<proj>`` to each spawned
    # train_{module}.py inside its tmux session, so save_folder /
    # checkpoint_path / log_path all land under /Data/<proj>/EXPERIMENT/...
    # Use this on servers where the project-local EXPERIMENT/ tree is
    # not writable or intentionally kept small.
    python3 examples/RunResults/run_experiments.py -m builder -d GSM8K -e AutoWeighted/*.yml -s /Data/<proj> --one-per-gpu

    # Resume a batch (all -e experiments resume from their latest
    # on-disk checkpoint). ``--resume`` is a boolean flag applied to
    # EVERY child in the batch — the child trainer hard-errors if a
    # per-experiment checkpoint directory is empty. Pair with
    # ``--swanlab-ids`` to pin specific SwanLab runs; use ``-`` as a
    # placeholder slot to defer to the child's logs/<exp>/swanlab.json.
    # ``-s`` is orthogonal and combines freely.
    python3 examples/RunResults/run_experiments.py -m builder -d GSM8K -e expA.yml expB.yml expC.yml -s /Data/<proj> --resume --swanlab-ids abc123 - xyz789

Resume (SwanLab):
    Resume is a pure CLI concern at BOTH layers: the launcher's
    ``--resume`` flag is forwarded verbatim to every spawned
    ``train_{module}.py --resume``. YAML configs no longer carry a
    ``training.resume`` field — ``--resume`` at the command line
    is the SINGLE source of truth.

    Each child trainer also accepts its own ``--swanlab-id`` (see
    ``train_{module}.py``). When the launcher is passed ``--swanlab-ids
    id1 id2 ... idN`` it fans them out 1-to-1 onto the N ``-e``
    experiments in the SAME order. Rules:
      * Length MUST equal ``len(-e)`` or the launcher aborts with
        exit code 2 BEFORE any tmux session is started.
      * A literal ``-`` placeholder keeps the slot but DEFERS id
        recovery to the child's ``logs/<exp>/swanlab.json`` (written
        on every fresh run, updated on every resume).
      * ``--swanlab-ids`` is only meaningful alongside ``--resume``.
        Without ``--resume``, children ignore any forwarded id and
        let SwanLab allocate a fresh one.
      * If both the CLI slot AND ``logs/<exp>/swanlab.json`` are
        missing while ``--resume`` is set, the child hard-errors
        (``SwanLabIdMissingError``) rather than start a
        disconnected SwanLab run that loses history continuity.

    Usage:
        python3 examples/RunResults/run_experiments.py -m builder -d GSM8K -e AutoWeighted/train_builder_Qwen3-0.6B_2level.yml AutoWeighted/train_builder_Qwen2.5-3B_6level.yml AutoWeighted/train_builder_Qwen2.5-3B_4level.yml AutoWeighted/train_builder_Qwen2.5-3B_2level.yml -s /Data/ReasoningNLCP --resume --swanlab-ids sd0j2fdldk1t6hlyoj84x fznay39wc04usqdom983t otcg3faik7e0v10z5nc4x wv1p7lmtnnme42ugj4uri --one-per-gpu


Storage-root behaviour (``-s``):
    * Default: ``./`` (current working directory). NEVER an implicit
      project root — this avoids silent writes to whichever folder
      the script happens to resolve to. Every spawned trainer
      receives the launcher's ``-s`` value via its own ``-s`` flag.
    * Custom root (``-s /Data/<proj>``): the launcher passes the SAME
      value to every child ``train_{module}.py``; each trainer then
      calls ``apply_storage_root(config, storage_root)`` after
      loading its YAML, so RELATIVE paths under ``config.log`` get
      prepended with the root. Absolute paths in YAML are preserved.
    * Every tool prints a ``[STORAGE]`` block at startup — launcher
      prints the forwarded value here, each child trainer prints
      its per-config resolved save_folder/checkpoint_path/log_path.
      Match this with analysis / SCP tools (``-s`` on
      ``builder_training_analysis.py``, ``run_scp.py``,
      ``loss_prepare.py``) so every tool looks at the same place.

Attaching:
    Sessions are started detached. To attach/detach:
        tmux ls                 # list sessions
        tmux attach -t <name>   # attach (Ctrl-b then d to detach)
        tmux kill-session -t <name>

Post-mortem logs:
    Every session's stdout+stderr is tee'd to
    ``logs/run_experiments/{session_name}.log`` under the project root.
    So even if the command crashes in the first second and the tmux
    session closes before you can attach, the full traceback is still
    on disk. ``--keep-alive`` additionally holds the session open after
    the command exits; without it, sessions only stay open on FAILURE
    (non-zero exit code) so you can still inspect the pane.

Conda environment:
    Before launching any session the launcher runs a pre-flight check:
      1. Verify ``conda`` is on PATH.
      2. ``conda env list`` — if the target env (default ``LMSim``,
         overridable with ``--conda-env``) does NOT exist, auto-create
         it with ``conda create -n {env} python={version} -y`` where
         ``version`` comes from ``--python-version`` (default 3.11).
      3. Each tmux session then sources ``conda.sh``, activates the
         env, and runs ``python`` (the env's interpreter) on the
         training script.
    Creating the env happens once in the LAUNCHER process so parallel
    sessions never race each other to create the same env.
"""

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

# ── Project paths ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "nlcpV4"
VALID_MODULES = ("builder", "predictor")


# =====================================================================
# CLI
# =====================================================================


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the experiment launcher."""
    parser = argparse.ArgumentParser(
        description=(
            "Launch multiple train_{module}.py experiments in parallel tmux "
            "sessions, with automatic GPU-memory-aware scheduling."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── Storage root (MUST come first: it controls every output path) ──
    parser.add_argument(
        "-s",
        "--storage-root",
        type=str,
        default="./",
        help=(
            "Prefix forwarded to each child train_{module}.py as -s. "
            "Relative output paths in every experiment's config.log "
            "(save_folder / checkpoint_path / log_path) will land under "
            "this directory. Absolute paths in YAML are preserved. "
            "Default is './' (current working directory) — NEVER "
            "silently derived from a project root. The resolved value "
            "is printed as a ``[STORAGE]`` block at launcher startup "
            "AND every spawned trainer prints its own per-config "
            "``[STORAGE]`` block, so you can see exactly where each "
            "run writes its checkpoints / logs."
        ),
    )
    # ── Core selection ──────────────────────────────────────────────
    parser.add_argument(
        "-m",
        "--module",
        type=str,
        required=True,
        choices=sorted(VALID_MODULES),
        help=(
            "Module name: 'builder' or 'predictor'. Selects which training "
            "script to invoke: examples/nlcpV4/train_{module}.py."
        ),
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., GSM8K). Resolves configs/nlcpV4/{dataset}/.",
    )
    parser.add_argument(
        "-e",
        "--experiments",
        type=str,
        nargs="+",
        required=True,
        help=(
            "One or more YAML config files. Each may be a bare filename, a "
            "path relative to configs/nlcpV4/{dataset}/ (e.g., "
            "'AutoWeighted/foo.yml'), or an absolute path."
        ),
    )
    # ── Runtime flags ───────────────────────────────────────────────
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan (incl. GPU assignments) without touching tmux.",
    )
    parser.add_argument(
        "--kill-existing",
        action="store_true",
        help=(
            "If any target tmux session name already exists, kill it and "
            "relaunch. WITHOUT this flag the launcher REFUSES to start the "
            "whole run as soon as even one colliding session is detected "
            "(fail-fast): a stale session from a previous crashed run "
            "still holds its GPU and would cause the new launch to "
            "double-schedule onto the same device. Pass this flag only "
            "when you are sure every colliding session should be replaced."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "Forward --resume to every spawned train_{module}.py. When "
            "set, each child auto-discovers the latest checkpoint under "
            "its own log.checkpoint_path and resumes; hard-errors if "
            "that directory is missing or empty. Resume is a pure CLI "
            "concern \u2014 YAML configs carry no training.resume field. "
            "Pair with --swanlab-ids to pin SwanLab run ids per slot."
        ),
    )
    parser.add_argument(
        "--swanlab-ids",
        type=str,
        nargs="+",
        default=None,
        help=(
            "SwanLab run IDs to resume, ONE per -e experiment, in the "
            "SAME order. Use '-' as a placeholder to defer to that "
            "experiment's on-disk logs/<exp>/swanlab.json. Only "
            "meaningful alongside --resume; without --resume the child "
            "ignores forwarded ids and lets SwanLab allocate fresh "
            "ones. If both the CLI slot and the on-disk swanlab.json "
            "are missing under --resume, the child hard-errors rather "
            "than start a disconnected SwanLab run. Length MUST equal "
            "the number of -e arguments or the launcher aborts."
        ),
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help=(
            "Always keep the tmux session open after the training command "
            "exits (success or failure). Default: session stays open only "
            "on FAILURE (non-zero exit code) so crashes are visible; on "
            "success the session closes automatically."
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/run_experiments",
        help=(
            "Directory (relative to project root, or absolute) where each "
            "session's combined stdout+stderr is tee'd as "
            "'{session_name}.log'. Default: logs/run_experiments/"
        ),
    )
    # ── Conda ───────────────────────────────────────────────────────
    parser.add_argument(
        "--conda-env",
        type=str,
        default="LMSim",
        help=(
            "Conda env name to activate inside each tmux session. If it "
            "does not exist, the launcher creates it before fan-out. "
            "Default: LMSim."
        ),
    )
    parser.add_argument(
        "--python-version",
        type=str,
        default="3.11",
        help=(
            "Python version used when auto-creating the conda env. Only "
            "consulted if the env does not already exist. Default: 3.11."
        ),
    )
    # ── GPU scheduling ──────────────────────────────────────────────
    parser.add_argument(
        "--no-gpu-schedule",
        action="store_true",
        help=(
            "Disable GPU-aware scheduling entirely. Launch all experiments "
            "in parallel without CUDA_VISIBLE_DEVICES pinning or memory "
            "checks. Equivalent to pre-scheduler behaviour."
        ),
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help=(
            "Comma-separated GPU indices to restrict scheduling to "
            "(e.g., '0,2,3'). Default: all GPUs reported by nvidia-smi."
        ),
    )
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=0.90,
        help=(
            "Fraction of each GPU's current FREE memory the scheduler is "
            "allowed to hand out (default 0.90 = leave 10%% headroom for "
            "kernel/framework overhead and fragmentation)."
        ),
    )
    parser.add_argument(
        "--mem-safety-factor",
        type=float,
        default=1.5,
        help=(
            "Multiplier applied to the raw memory estimate (weights + "
            "optimizer + activations). Default 1.5 — conservative; lower "
            "if you know your workload fits tighter."
        ),
    )
    parser.add_argument(
        "--mem-per-exp-mb",
        type=int,
        default=None,
        help=(
            "Override the per-experiment memory estimate with a fixed "
            "value in MiB. Useful when the heuristic is wrong for your "
            "specific workload. Applied uniformly to all experiments."
        ),
    )
    parser.add_argument(
        "--wait-for-gpu",
        action="store_true",
        help=(
            "If some experiments cannot be placed immediately, keep the "
            "launcher alive and poll nvidia-smi every --poll-interval "
            "seconds, launching queued experiments as memory frees up. "
            "Without this flag, queued experiments are reported and skipped."
        ),
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help=("Seconds between GPU polls when --wait-for-gpu is set. " "Default 30."),
    )
    parser.add_argument(
        "--warmup-seconds",
        type=int,
        default=90,
        help=(
            "How long a just-launched experiment's memory reservation is "
            "kept in the scheduler's books BEFORE we trust nvidia-smi to "
            "reflect its actual allocation. A freshly launched session "
            "typically takes 30\u2013120 s (conda activate + python + torch "
            "+ HF model load) before its memory appears in nvidia-smi. "
            "During this window, the scheduler must still count the "
            "reservation or it will double-schedule. Default 90s."
        ),
    )
    parser.add_argument(
        "--launch-stagger",
        type=float,
        default=0.0,
        help=(
            "Seconds to sleep between consecutive tmux launches. "
            "0 (default) launches all assigned sessions back-to-back. "
            "Set to e.g. 5.0 to smear out CUDA init and HF weight-load "
            "pressure when packing multiple jobs on the same GPU."
        ),
    )
    # ── Simple "one experiment per GPU" mode ──────────────────────────
    parser.add_argument(
        "--one-per-gpu",
        action="store_true",
        help=(
            "Simple scheduling mode: pin EXACTLY ONE experiment per GPU. "
            "Skips memory estimation entirely and instead limits "
            "concurrency to the number of currently-idle GPUs (as "
            "reported by nvidia-smi). When more experiments than idle "
            "GPUs are supplied, the excess ones wait in a FIFO queue and "
            "the launcher polls every --poll-interval seconds; as soon "
            "as a GPU goes idle again, the next queued experiment is "
            "launched on it. Use this when each experiment is big enough "
            "to saturate a GPU on its own (the common case for 7B+ "
            "backbones or full fine-tuning) — it sidesteps OOM entirely "
            "by never co-locating two jobs on the same device. Implicitly "
            "enables --wait-for-gpu behaviour so the queue drains to "
            "completion."
        ),
    )
    parser.add_argument(
        "--gpu-idle-mem-fraction",
        type=float,
        default=0.9,
        help=(
            "Only used with --one-per-gpu. A GPU counts as 'idle' (and "
            "thus eligible to accept a new experiment) only if its "
            "currently free memory is at least this fraction of total "
            "memory. Default 0.9 (≥ 90%% free). Raise toward 1.0 to be "
            "stricter (e.g., refuse a GPU with ANY other process on it); "
            "lower if you want to tolerate a small amount of shared use."
        ),
    )
    return parser.parse_args()


# =====================================================================
# Conda helpers
# =====================================================================


def _conda_base() -> Path:
    """Return the conda base directory (``CONDA_PREFIX`` of the base env)."""
    result = subprocess.run(
        ["conda", "info", "--base"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def _conda_env_exists(env_name: str) -> bool:
    """True iff ``conda env list`` contains an env named exactly ``env_name``."""
    result = subprocess.run(
        ["conda", "env", "list", "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    for env_path in data.get("envs", []):
        if Path(env_path).name == env_name:
            return True
    return False


def ensure_conda_env(env_name: str, python_version: str) -> Path:
    """Guarantee ``env_name`` exists; create it if missing. Return conda base."""
    if shutil.which("conda") is None:
        raise RuntimeError(
            "conda is not installed or not on PATH. Install Miniconda/"
            "Anaconda first, or re-run inside a shell where `conda` is "
            "available."
        )
    base = _conda_base()
    if _conda_env_exists(env_name):
        print(f"[conda] env {env_name!r} already exists — reusing.")
        return base
    print(
        f"[conda] env {env_name!r} not found — creating with "
        f"python={python_version} (this may take a minute) ..."
    )
    subprocess.run(
        [
            "conda",
            "create",
            "-n",
            env_name,
            f"python={python_version}",
            "-y",
        ],
        check=True,
    )
    print(f"[conda] env {env_name!r} created.")
    return base


# =====================================================================
# GPU detection (nvidia-smi)
# =====================================================================


@dataclass
class GPUInfo:
    """Snapshot of one GPU's memory status.

    ``reserved_mb`` tracks memory the scheduler has *already* handed out
    in this launcher run (not memory that nvidia-smi is reporting as
    used). Callers should compute availability as
    ``int(free_mb * fraction) - reserved_mb``.
    """

    index: int
    name: str
    free_mb: int
    total_mb: int
    reserved_mb: int = 0

    def available_mb(self, fraction: float) -> int:
        return max(0, int(self.free_mb * fraction) - self.reserved_mb)


def detect_gpus(restrict_indices: list[int] | None = None) -> list[GPUInfo]:
    """Query ``nvidia-smi`` for per-GPU (index, name, free_mb, total_mb).

    Returns ``[]`` if:
      * nvidia-smi is not on PATH (e.g., macOS dev machine)
      * the call fails / times out
      * restrict_indices filters out everything

    An empty list signals the caller to either proceed without
    scheduling (``--no-gpu-schedule`` implicit on non-NVIDIA hosts)
    or to abort, depending on context.
    """
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"[WARN] nvidia-smi failed: {e}", file=sys.stderr)
        return []

    gpus: list[GPUInfo] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            idx = int(parts[0])
            name = parts[1]
            free = int(parts[2])
            total = int(parts[3])
        except ValueError:
            continue
        if restrict_indices is not None and idx not in restrict_indices:
            continue
        gpus.append(GPUInfo(index=idx, name=name, free_mb=free, total_mb=total))
    return gpus


def parse_gpu_restriction(spec: str | None) -> list[int] | None:
    """Parse ``--gpus "0,2,3"`` into a list of ints, or None if unset."""
    if spec is None:
        return None
    out: list[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out if out else None


# =====================================================================
# Run-wide reservation tracking (survives across scheduling passes)
# =====================================================================


@dataclass
class _Reservation:
    """One memory reservation made by the scheduler in this run.

    Each successful ``schedule_assignments`` placement produces exactly
    one of these. They are kept in a module-level list so subsequent
    scheduling passes (wait-for-gpu polling) can reconstruct the true
    available budget per GPU — crucially, DURING the window where a
    tmux-launched session has not yet allocated its GPU memory and
    ``nvidia-smi`` still reports that memory as free.
    """

    gpu_index: int
    mem_mb: int
    # Wall-clock ``time.time()`` at the moment the reservation was made;
    # used together with ``warmup_seconds`` to decide when the reservation
    # has "settled" and can be dropped in favour of live nvidia-smi data.
    committed_at: float


def _apply_reservations(
    gpus: list[GPUInfo],
    reservations: list[_Reservation],
    now: float,
    warmup_seconds: float,
) -> list[_Reservation]:
    """Set ``g.reserved_mb`` on each gpu to reflect UN-SETTLED reservations.

    Semantics:
      * A reservation is "un-settled" if ``now - committed_at <
        warmup_seconds``. We believe the process has not yet allocated
        its memory and nvidia-smi's ``free_mb`` over-reports reality.
      * A reservation is "settled" if it has aged past warmup. At that
        point nvidia-smi should be reflecting the actual usage, so
        continuing to subtract our reservation would DOUBLE-count. We
        drop it.

    Returns the new list of still-active reservations (with settled
    ones pruned) so callers can replace their state.
    """
    active: list[_Reservation] = []
    by_idx: dict[int, int] = {}
    for r in reservations:
        if (now - r.committed_at) < warmup_seconds:
            active.append(r)
            by_idx[r.gpu_index] = by_idx.get(r.gpu_index, 0) + r.mem_mb
    for g in gpus:
        g.reserved_mb = by_idx.get(g.index, 0)
    return active


# =====================================================================
# Memory estimation
# =====================================================================


# Known backbone sizes (HF model name tail -> parameter count in billions).
# Used for exact lookup; falls back to regex parsing for anything else.
_KNOWN_MODEL_SIZES_B: dict[str, float] = {
    "Qwen2.5-0.5B": 0.5,
    "Qwen2.5-1.5B": 1.5,
    "Qwen2.5-3B": 3.0,
    "Qwen2.5-7B": 7.0,
    "Qwen3-0.6B": 0.6,
    "Qwen3-1.7B": 1.7,
    "Qwen3-4B": 4.0,
    "Qwen3-8B": 8.0,
}
# Match "...0.5B..." or "...8B..." etc.
_SIZE_REGEX = re.compile(r"(\d+(?:\.\d+)?)\s*[Bb](?:[^\d]|$)")


def parse_model_params_b(model_name: str) -> float:
    """Extract parameter count (in billions) from an HF model id.

    1. Strip any org prefix (``Qwen/Qwen2.5-0.5B`` -> ``Qwen2.5-0.5B``).
    2. Try the known-sizes table (most reliable for our Qwen matrix).
    3. Fallback regex matching ``X.YB`` / ``XB`` tokens.
    4. Fail-fast if nothing matches — we'd rather refuse to schedule than
       silently assume the model is tiny.
    """
    tail = model_name.rsplit("/", 1)[-1]
    if tail in _KNOWN_MODEL_SIZES_B:
        return _KNOWN_MODEL_SIZES_B[tail]
    m = _SIZE_REGEX.search(tail)
    if m:
        return float(m.group(1))
    raise ValueError(
        f"Could not parse parameter count from model name: {model_name!r}. "
        f"Add it to _KNOWN_MODEL_SIZES_B or pass --mem-per-exp-mb to override."
    )


def estimate_experiment_memory_mb(
    config_path: Path,
    safety_factor: float,
) -> tuple[int, dict]:
    """Estimate GPU memory (MiB) needed to train this experiment.

    Rule-of-thumb, intentionally conservative:

        bytes_per_elem = 4 (fp32) or 2 (bf16/fp16)
        weights_mb      = params * bytes / 1e6
        optimizer_mb    = weights_mb * 3     (Adam full FT: grads + m + v)
                        = weights_mb * 0.2   (LoRA / frozen backbone)
        activation_mb   = B * L * D * bytes * 30 / 1e6
                          (×30 ≈ attn + mlp intermediates kept for backward)
        total_mb        = (weights + optimizer + activation) * safety_factor

    The inexactness is unavoidable — attention kernel choice, gradient
    checkpointing, framework overhead, and tokenizer-side padding all
    shift the real peak. Use ``--mem-per-exp-mb`` to override when you
    have a measured number.

    Returns (total_mb_int, breakdown_dict) so callers can print details.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    reason_cfg = model_cfg["reason_model"]
    pyramid_cfg = model_cfg["pyramid"]
    training_cfg = cfg["training"]

    params_b = parse_model_params_b(reason_cfg["reason_model_name"])
    params = params_b * 1e9

    dtype = reason_cfg["torch_dtype"]
    bytes_per_elem = 2 if dtype in ("bfloat16", "float16") else 4

    # LoRA / frozen backbone means no full-model grads / Adam state.
    train_rm = training_cfg["reason_model"]
    lora_cfg = train_rm["lora"]
    freeze = bool(train_rm["freeze"])
    lora_or_frozen = (lora_cfg is not None) or freeze

    weights_mb = params * bytes_per_elem / 1e6
    optimizer_mb = weights_mb * (0.2 if lora_or_frozen else 3.0)

    # Activation pressure from the autoregressive reasoning forward:
    # [Q | concepts | S] sequence through the full reason_model.
    batch_size = int(training_cfg["batch_size"])
    max_seq_len = int(pyramid_cfg["max_seq_len"])
    hidden_dim = int(pyramid_cfg["hidden_dim"])
    activation_mb = batch_size * max_seq_len * hidden_dim * bytes_per_elem * 30 / 1e6

    base_mb = weights_mb + optimizer_mb + activation_mb
    total_mb = int(base_mb * safety_factor)

    breakdown = {
        "model": reason_cfg["reason_model_name"],
        "params_B": params_b,
        "dtype": dtype,
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
        "hidden_dim": hidden_dim,
        "lora_or_frozen": lora_or_frozen,
        "weights_mb": int(weights_mb),
        "optimizer_mb": int(optimizer_mb),
        "activation_mb": int(activation_mb),
        "safety_factor": safety_factor,
        "total_mb": total_mb,
    }
    return total_mb, breakdown


# =====================================================================
# Scheduler
# =====================================================================


@dataclass
class Experiment:
    raw: str
    cfg_path: Path
    session_name: str
    mem_mb: int
    breakdown: dict
    # Assigned at scheduling time; None means queued.
    gpu_index: int | None = None
    # Optional per-experiment SwanLab run id forwarded from the
    # launcher's ``--swanlab-ids`` flag. ``None`` means "defer to the
    # child's on-disk swanlab.json" (or hard-error if neither source
    # has it).
    swanlab_id: str | None = None


def schedule_assignments(
    experiments: list[Experiment],
    gpus: list[GPUInfo],
    gpu_memory_fraction: float,
) -> tuple[list[Experiment], list[Experiment]]:
    """Best-fit-decreasing pack over the current GPU snapshot.

    Strategy:
      1. Sort experiments by ``mem_mb`` DESC so heavy jobs get first pick
         (classic first-fit-decreasing / best-fit-decreasing bin pack).
      2. For each experiment, pick the GPU with the SMALLEST remaining
         available budget that still fits (best-fit), so bigger GPUs
         stay open for bigger future jobs.
      3. Subtract the assigned memory from the chosen GPU's
         ``reserved_mb`` so subsequent experiments see an updated budget.

    Experiments that don't fit any GPU are returned in a separate
    ``queued`` list (``gpu_index`` left as ``None``). They can either be
    skipped (default) or fed to a polling loop via ``--wait-for-gpu``.

    NOTE: this function MUTATES ``gpus`` (updating ``reserved_mb``) so
    callers can pass the same list to subsequent scheduling passes and
    reservations accumulate. Pass a fresh snapshot from ``detect_gpus``
    if you want to start from the current live free-memory figures.
    """
    sorted_exps = sorted(experiments, key=lambda e: e.mem_mb, reverse=True)
    gpu_by_idx = {g.index: g for g in gpus}

    assigned: list[Experiment] = []
    queued: list[Experiment] = []

    for exp in sorted_exps:
        candidates = [
            g
            for g in gpu_by_idx.values()
            if g.available_mb(gpu_memory_fraction) >= exp.mem_mb
        ]
        if not candidates:
            queued.append(exp)
            continue
        # Best-fit: smallest remaining availability that still fits.
        candidates.sort(key=lambda g: g.available_mb(gpu_memory_fraction))
        chosen = candidates[0]
        chosen.reserved_mb += exp.mem_mb
        exp.gpu_index = chosen.index
        assigned.append(exp)

    return assigned, queued


def schedule_one_per_gpu(
    experiments: list[Experiment],
    gpus: list[GPUInfo],
    idle_mem_fraction: float,
) -> tuple[list[Experiment], list[Experiment]]:
    """Pin at most ONE experiment per GPU, FIFO order.

    A GPU is eligible iff BOTH conditions hold:
      * No active reservation of ours on it (``reserved_mb == 0``).
        The caller is expected to have replayed the reservation ledger
        via ``_apply_reservations`` so that this reflects sessions we
        launched within the warmup window. Any reservation, regardless
        of size, means "this slot is taken by us".
      * Currently idle: ``free_mb / total_mb >= idle_mem_fraction``.
        This protects against stealing a GPU that another user is
        already using, and (after warmup) confirms that a previously
        launched session has released its memory.

    Experiments are assigned in FIFO order (list order), one per
    eligible GPU. Anything past the idle-GPU count stays queued.

    MUTATES ``gpus``: bumps ``reserved_mb`` on each chosen GPU by 1 so
    that within the SAME pass a second experiment cannot claim the
    same slot. The actual memory figure doesn't matter in this mode —
    only the "slot taken" flag does.
    """
    assigned: list[Experiment] = []
    queued: list[Experiment] = []

    # Preserve caller order; do NOT sort by memory (it's irrelevant here).
    it = iter(experiments)
    for exp in it:
        chosen: GPUInfo | None = None
        for g in gpus:
            if g.reserved_mb > 0:
                # Our own active reservation — slot taken.
                continue
            if g.total_mb <= 0:
                continue
            if (g.free_mb / g.total_mb) < idle_mem_fraction:
                # External process is using this GPU.
                continue
            chosen = g
            break
        if chosen is None:
            queued.append(exp)
            # All remaining experiments must also queue — no GPU left.
            for rest in it:
                queued.append(rest)
            break
        # Mark the slot taken for this pass so a second experiment in
        # the same scheduling round cannot claim the same GPU.
        chosen.reserved_mb += 1
        exp.gpu_index = chosen.index
        assigned.append(exp)

    return assigned, queued


# =====================================================================
# Path + session helpers
# =====================================================================


def resolve_experiment(raw: str, dataset_dir: Path) -> Path:
    """Resolve a user-supplied experiment path to an absolute config path."""
    p = Path(raw)
    resolved = p if p.is_absolute() else dataset_dir / p
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Config not found: {resolved} (from -e argument {raw!r})"
        )
    if resolved.suffix != ".yml":
        raise ValueError(
            f"Expected a .yml file, got: {resolved} (from -e argument {raw!r})"
        )
    return resolved


def experiment_label(config_path: Path, dataset_dir: Path) -> str:
    """Derive the ``{experiment_label}`` part of the tmux session name."""
    try:
        rel = config_path.relative_to(dataset_dir)
    except ValueError:
        return config_path.stem
    parts = [*rel.parent.parts, rel.stem]
    parts = [p for p in parts if p and p != "."]
    return "-".join(parts)


def sanitize_session_name(name: str) -> str:
    """tmux disallows ``.`` and ``:`` in session names — replace with ``_``."""
    return name.replace(".", "_").replace(":", "_")


def tmux_session_exists(session_name: str) -> bool:
    """True iff a tmux session with this exact name is currently running."""
    result = subprocess.run(
        ["tmux", "has-session", "-t", f"={session_name}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def list_tmux_sessions() -> set[str]:
    """Return the set of ALL currently-live tmux session names.

    Uses a single ``tmux list-sessions`` call so the pre-flight
    collision check can run in O(1) subprocess calls regardless of
    how many experiments we plan to launch. Returns an empty set if
    the tmux server has no sessions (``tmux ls`` exits non-zero in
    that case) or if tmux is unavailable.
    """
    if shutil.which("tmux") is None:
        return set()
    result = subprocess.run(
        ["tmux", "list-sessions", "-F", "#{session_name}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        # tmux returns non-zero when no sessions exist ("no server running"
        # or "no sessions"). Both cases mean "nothing to collide with".
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def kill_tmux_session(session_name: str) -> None:
    """Terminate the named tmux session; no-op if it does not exist."""
    subprocess.run(
        ["tmux", "kill-session", "-t", session_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


# =====================================================================
# Inner-command construction & launch
# =====================================================================


def build_inner_command(
    exp: Experiment,
    train_script: Path,
    log_dir: Path,
    conda_base: Path | None,
    conda_env: str,
    keep_alive: bool,
    storage_root: str,
    resume: bool = False,
) -> tuple[str, Path]:
    """Build the full shell command that tmux executes for one experiment.

    Returns (inner_command_string, log_path).

    The sequence inside a single ``bash -o pipefail -c '...'`` is:
      1. cd into PROJECT_ROOT
      2. source conda.sh
      3. conda activate {env}
      4. CUDA_VISIBLE_DEVICES={gpu} PYTHONUNBUFFERED=1 python {script} -c {cfg} | tee {log}

    ``bash -o pipefail`` ensures Python's exit code survives the tee pipe;
    ``PYTHONUNBUFFERED=1`` ensures tracebacks reach the log before the
    process tears down. The post-command tail keeps the session alive
    only on failure (or always, with --keep-alive) so crash panes are
    still inspectable.
    """
    log_path = log_dir / f"{exp.session_name}.log"
    conda_sh = (
        (conda_base / "etc" / "profile.d" / "conda.sh")
        if conda_base is not None
        else Path("$(conda info --base)/etc/profile.d/conda.sh")
    )
    gpu_prefix = (
        f"CUDA_VISIBLE_DEVICES={exp.gpu_index} " if exp.gpu_index is not None else ""
    )
    storage_arg = f"-s {shlex.quote(storage_root)} "
    # Resume is a pure launcher-wide boolean — when on, every child
    # gets --resume appended; when off, the flag is absent entirely so
    # the child's default (fresh-start) applies. This mirrors the
    # train_builder.py contract where --resume is the SINGLE source of
    # truth (no YAML training.resume field).
    resume_arg = "--resume " if resume else ""
    # Forward a per-experiment SwanLab id only when the launcher was
    # explicitly given one for this slot. An empty slot (``None``)
    # means "let the child resolve it from logs/<exp>/swanlab.json",
    # which is the normal case for resumes originating from a prior
    # crash on this same host.
    swanlab_arg = (
        f"--swanlab-id {shlex.quote(exp.swanlab_id)} " if exp.swanlab_id else ""
    )
    inner_cmd = (
        f"cd {shlex.quote(str(PROJECT_ROOT))} && "
        f"source {shlex.quote(str(conda_sh))} && "
        f"conda activate {shlex.quote(conda_env)} && "
        f"{gpu_prefix}PYTHONUNBUFFERED=1 python "
        f"{shlex.quote(str(train_script))} "
        f"-c {shlex.quote(str(exp.cfg_path))} "
        f"{storage_arg}"
        f"{resume_arg}"
        f"{swanlab_arg}"
        f"2>&1 | tee {shlex.quote(str(log_path))}"
    )
    piped = f"bash -o pipefail -c {shlex.quote(inner_cmd)}"
    if keep_alive:
        tail = (
            f"ec=$?; echo; "
            f"echo '[run_experiments] exited with code '\"$ec\"' — log: {log_path}'; "
            f"exec $SHELL"
        )
    else:
        tail = (
            f"ec=$?; "
            f'if [ "$ec" != "0" ]; then '
            f"echo; "
            f"echo '[run_experiments] FAILED with code '\"$ec\"' — log: {log_path}'; "
            f"exec $SHELL; "
            f"fi"
        )
    return f"{piped}; {tail}", log_path


def launch_one(
    exp: Experiment,
    inner: str,
    log_path: Path,
    args: argparse.Namespace,
) -> str:
    """Print the plan for one experiment and (unless --dry-run) launch it.

    Returns one of: "launched", "skipped", "error", "dry-run".
    """
    rel_cfg = exp.cfg_path.relative_to(PROJECT_ROOT)
    try:
        rel_log = log_path.relative_to(PROJECT_ROOT)
    except ValueError:
        rel_log = log_path
    gpu_str = f"GPU{exp.gpu_index}" if exp.gpu_index is not None else "GPU?"

    print(f"[{exp.session_name}]")
    print(f"  config : {rel_cfg}")
    print(f"  log    : {rel_log}")
    print(f"  mem_est: {exp.mem_mb} MiB  ({gpu_str})")
    print(f"  command: {inner}")

    if args.dry_run:
        print("  (dry-run: not launched)\n")
        return "dry-run"

    if tmux_session_exists(exp.session_name):
        if args.kill_existing:
            print(f"  [info] killing existing session {exp.session_name!r}")
            kill_tmux_session(exp.session_name)
        else:
            print("  [SKIP] session already exists. Use --kill-existing.\n")
            return "skipped"

    tmux_cmd = ["tmux", "new-session", "-d", "-s", exp.session_name, inner]
    result = subprocess.run(tmux_cmd)
    if result.returncode != 0:
        print(
            f"  [ERROR] tmux new-session failed (exit {result.returncode})\n",
            file=sys.stderr,
        )
        return "error"
    print(f"  [OK] launched detached session {exp.session_name!r}\n")
    return "launched"


# =====================================================================
# main
# =====================================================================


def main() -> int:
    """CLI entry point: parse args, schedule experiments, launch tmux sessions."""
    args = parse_args()

    # ── Surface the storage-root contract up front ──────────────────
    # Every spawned trainer will receive ``-s {storage_root}`` (default
    # ``./``) and emit its own ``[STORAGE]`` block per config. Showing
    # the launcher-level value here makes the cross-tool contract
    # visible in a single log line, so misaligned ``-s`` values between
    # launcher and downstream analysis/scp tools fail loudly.
    print(
        f"[STORAGE] launcher storage_root = {args.storage_root!r} "
        f"(forwarded as -s to every child train_{args.module}.py); "
        f"cwd={Path.cwd().resolve()}"
    )

    # ── Surface the --swanlab-ids fan-out up front ─────────────────────
    # Print a single banner line so misaligned --swanlab-ids lists are
    # easy to spot in the launch log. Actual length validation happens
    # after we resolve -e, because the validation message references the
    # resolved experiment count.
    if args.resume:
        print(
            f"[RESUME] --resume is ON; every child train_{args.module}.py "
            f"will auto-discover its latest checkpoint under "
            f"log.checkpoint_path. Missing/empty dirs → hard error."
        )
        if args.swanlab_ids is None:
            print(
                "[RESUME] No --swanlab-ids provided; every child will "
                "fall back to logs/<exp>/swanlab.json (hard error if "
                "that file is missing)."
            )
    elif args.swanlab_ids is not None:
        print(
            "[WARN] --swanlab-ids given without --resume; child trainers "
            "will IGNORE forwarded ids on a fresh run. Did you mean to "
            "pass --resume?"
        )
    if args.swanlab_ids is not None:
        n_supplied = sum(1 for s in args.swanlab_ids if s != "-")
        n_deferred = sum(1 for s in args.swanlab_ids if s == "-")
        print(
            f"[SWANLAB] --swanlab-ids received {len(args.swanlab_ids)} slot(s): "
            f"{n_supplied} explicit id(s), {n_deferred} deferred ('-' → "
            f"logs/<exp>/swanlab.json). Child aborts if both sources miss."
        )

    # ── Validate tmux availability ─────────────────────────────────────
    if not args.dry_run and shutil.which("tmux") is None:
        print(
            "[ERROR] tmux is not installed or not on PATH. "
            "Install it (e.g., `brew install tmux`) or pass --dry-run.",
            file=sys.stderr,
        )
        return 2

    # ── Resolve paths ──────────────────────────────────────────────────
    dataset_dir = (PROJECT_ROOT / "configs" / PACKAGE / args.dataset).resolve()
    if not dataset_dir.is_dir():
        print(f"[ERROR] Dataset config dir not found: {dataset_dir}", file=sys.stderr)
        return 2

    train_script = (
        PROJECT_ROOT / "examples" / PACKAGE / f"train_{args.module}.py"
    ).resolve()
    if not train_script.is_file():
        print(f"[ERROR] Training script not found: {train_script}", file=sys.stderr)
        return 2

    # ── Conda pre-flight (before anything GPU-related so a bad env
    # fails fast and we don't waste time probing nvidia-smi). ──────────
    conda_base: Path | None = None
    if not args.dry_run:
        try:
            conda_base = ensure_conda_env(args.conda_env, args.python_version)
        except (RuntimeError, subprocess.CalledProcessError) as e:
            print(f"[ERROR] conda setup failed: {e}", file=sys.stderr)
            return 2

    # Resolve (and create) the log directory — tmux sessions close as
    # soon as their command exits, so without an on-disk log a quick
    # crash would leave no trace.
    log_dir_arg = Path(args.log_dir)
    log_dir = (
        log_dir_arg if log_dir_arg.is_absolute() else (PROJECT_ROOT / log_dir_arg)
    ).resolve()
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)

    # Resolve all experiments up-front so we abort before launching any
    # session if one path is invalid — avoids partial fan-out.
    try:
        resolved_experiments = [
            (raw, resolve_experiment(raw, dataset_dir)) for raw in args.experiments
        ]
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    # ── Align --swanlab-ids with -e experiments ────────────────────────
    # Strict length check: positional pairing only makes sense when the
    # two lists have the same arity. A '-' placeholder keeps the slot
    # but defers id recovery to the child's logs/<exp>/swanlab.json.
    swanlab_ids_aligned: list[str | None]
    if args.swanlab_ids is None:
        swanlab_ids_aligned = [None] * len(resolved_experiments)
    else:
        if len(args.swanlab_ids) != len(resolved_experiments):
            print(
                f"[ERROR] --swanlab-ids length ({len(args.swanlab_ids)}) "
                f"does not match number of -e experiments "
                f"({len(resolved_experiments)}). Use '-' as a placeholder "
                f"for any slot that should fall back to the child's "
                f"logs/<exp>/swanlab.json.",
                file=sys.stderr,
            )
            return 2
        swanlab_ids_aligned = [None if s == "-" else s for s in args.swanlab_ids]

    # ── Build Experiment objects + estimate memory ─────────────────────
    experiments: list[Experiment] = []
    seen_names: set[str] = set()
    for (raw, cfg_path), swanlab_id in zip(resolved_experiments, swanlab_ids_aligned):
        label = experiment_label(cfg_path, dataset_dir)
        session_name = sanitize_session_name(f"{args.module}-{args.dataset}-{label}")
        if session_name in seen_names:
            print(
                f"[ERROR] Duplicate session name within this launch: "
                f"{session_name!r} (from {raw!r}).",
                file=sys.stderr,
            )
            return 2
        seen_names.add(session_name)

        if args.one_per_gpu:
            # In "one experiment per GPU" mode the memory estimate is
            # irrelevant — we allocate entire GPUs, not fractions of
            # them. Skip estimation so we never fail on a YAML that
            # the heuristic can't parse. Use a sentinel value of 1
            # (not 0) so the reservation ledger's sum-based "slot
            # taken" check still fires during warmup.
            mem_mb = 1
            breakdown = {"one_per_gpu": True}
        elif args.mem_per_exp_mb is not None:
            mem_mb = args.mem_per_exp_mb
            breakdown = {"override_mb": mem_mb}
        else:
            try:
                mem_mb, breakdown = estimate_experiment_memory_mb(
                    cfg_path, args.mem_safety_factor
                )
            except (KeyError, ValueError) as e:
                print(
                    f"[ERROR] Could not estimate memory for {cfg_path}: {e}\n"
                    f"        Pass --mem-per-exp-mb to bypass the heuristic.",
                    file=sys.stderr,
                )
                return 2
        experiments.append(
            Experiment(
                raw=raw,
                cfg_path=cfg_path,
                session_name=session_name,
                mem_mb=mem_mb,
                breakdown=breakdown,
                swanlab_id=swanlab_id,
            )
        )

    # ── Pre-flight: refuse to run if any target tmux session is alive ──
    # Snapshot ALL currently-live tmux sessions in ONE ``tmux ls`` call,
    # then intersect with the planned session-name set. This is both
    # cheaper than a per-experiment ``tmux has-session`` probe and more
    # atomic (no race between N separate subprocess calls).
    #
    # A leftover session from an earlier (possibly crashed) launch holds
    # its GPU and makes this new run silently double-schedule onto the
    # same physical device. The old behaviour of skipping each colliding
    # experiment with a warning produced mysterious "OOM on an idle GPU"
    # bugs that were very hard to diagnose. Bail LOUDLY instead so the
    # operator deals with the leftovers first.
    # Opt-out: ``--kill-existing`` explicitly replaces colliding sessions.
    if not args.dry_run and not args.kill_existing:
        live_sessions = list_tmux_sessions()
        planned_names = {e.session_name for e in experiments}
        colliding = sorted(planned_names & live_sessions)
        if colliding:
            print(
                "\n[ERROR] Refusing to launch: the following tmux session(s) "
                "from an earlier run are still alive. A leftover session "
                "holds GPU memory and would cause this launch to "
                "double-schedule onto the same device.",
                file=sys.stderr,
            )
            for name in colliding:
                print(f"  - {name}", file=sys.stderr)
            kill_all = " ; ".join(
                f"tmux kill-session -t {shlex.quote(n)}" for n in colliding
            )
            print(
                "\n  Fix options:\n"
                "    1. Inspect each:   tmux attach -t <name>   "
                "(Ctrl-b then d to detach)\n"
                "    2. Kill one:       tmux kill-session -t <name>\n"
                "    3. Kill all listed above in one go:\n"
                f"         {kill_all}\n"
                "    4. Or re-run this exact command with --kill-existing "
                "to auto-replace every colliding session.",
                file=sys.stderr,
            )
            return 2

    # ── GPU scheduling ─────────────────────────────────────────────────
    restrict = parse_gpu_restriction(args.gpus)
    gpus: list[GPUInfo] = []
    scheduling_enabled = not args.no_gpu_schedule
    if scheduling_enabled:
        gpus = detect_gpus(restrict_indices=restrict)
        if not gpus:
            # --one-per-gpu is a HARD promise of bounded concurrency.
            # Silently falling back to "launch everything in parallel"
            # would violate the user's intent, so we refuse instead.
            if args.one_per_gpu and not args.dry_run:
                print(
                    "[ERROR] --one-per-gpu requires at least one GPU visible "
                    "to nvidia-smi, but none were detected. Install NVIDIA "
                    "drivers or drop --one-per-gpu. (On macOS/CPU-only hosts, "
                    "use --dry-run to preview.)",
                    file=sys.stderr,
                )
                return 2
            print(
                "[WARN] No GPUs detected (nvidia-smi missing or returned empty). "
                "Scheduling disabled — all experiments will launch in parallel "
                "without CUDA_VISIBLE_DEVICES pinning.",
                file=sys.stderr,
            )
            scheduling_enabled = False

    # ── Preview header ─────────────────────────────────────────────────
    print(f"Project root   : {PROJECT_ROOT}")
    print(f"Training script: {train_script.relative_to(PROJECT_ROOT)}")
    print(f"Dataset dir    : {dataset_dir.relative_to(PROJECT_ROOT)}")
    print(f"Conda env      : {args.conda_env}")
    print(f"Experiments    : {len(experiments)}")
    if scheduling_enabled:
        if args.one_per_gpu:
            print(
                f"GPU scheduling : ENABLED [one-per-gpu] "
                f"(idle_threshold={args.gpu_idle_mem_fraction})"
            )
        else:
            print(
                f"GPU scheduling : ENABLED (fraction={args.gpu_memory_fraction}, "
                f"safety={args.mem_safety_factor})"
            )
        print(f"Detected GPUs  : {len(gpus)}")
        for g in gpus:
            mark = ""
            if args.one_per_gpu and g.total_mb > 0:
                idle = (g.free_mb / g.total_mb) >= args.gpu_idle_mem_fraction
                mark = "  [idle]" if idle else "  [busy]"
            print(
                f"  GPU{g.index} ({g.name}): "
                f"{g.free_mb}/{g.total_mb} MiB free{mark}"
            )
    else:
        print("GPU scheduling : DISABLED")
    print()

    # ── Pre-flight hard-bail ───────────────────────────────────────────
    # If ANY experiment is too big for the largest visible GPU, refuse
    # to start — otherwise --wait-for-gpu would spin forever on an
    # impossible job, and without it we'd silently fan out only a
    # subset while the oversized ones hit OOM at model-load time.
    if scheduling_enabled and experiments and not args.one_per_gpu:
        max_budget = max(int(g.total_mb * args.gpu_memory_fraction) for g in gpus)
        oversized = [e for e in experiments if e.mem_mb > max_budget]
        if oversized:
            print(
                f"[ERROR] {len(oversized)} experiment(s) exceed the largest "
                f"visible GPU's effective budget ({max_budget} MiB at "
                f"fraction={args.gpu_memory_fraction}).",
                file=sys.stderr,
            )
            for e in oversized:
                print(
                    f"  - {e.session_name}: needs {e.mem_mb} MiB",
                    file=sys.stderr,
                )
            print(
                "        Reduce model/batch/seq, raise --gpu-memory-fraction, "
                "lower --mem-safety-factor, or override with --mem-per-exp-mb.",
                file=sys.stderr,
            )
            return 2

    # ── Run the scheduler (or skip it) ─────────────────────────────────
    # Persistent reservation ledger — survives across passes so that
    # newly-polled ``nvidia-smi`` snapshots during the warmup window
    # (tmux launched but process not yet allocated) don't mis-report
    # a GPU as empty and cause double-scheduling → OOM.
    run_reservations: list[_Reservation] = []
    if scheduling_enabled:
        # First pass: starting state has no prior reservations, so the
        # apply step is a no-op but kept for symmetry with the wait loop.
        run_reservations = _apply_reservations(
            gpus, run_reservations, time.time(), args.warmup_seconds
        )
        if args.one_per_gpu:
            assigned, queued = schedule_one_per_gpu(
                experiments, gpus, args.gpu_idle_mem_fraction
            )
        else:
            assigned, queued = schedule_assignments(
                experiments, gpus, args.gpu_memory_fraction
            )
    else:
        assigned, queued = experiments, []

    # In --one-per-gpu mode, implicitly keep the launcher alive so the
    # queue drains to completion — the whole point of this mode is
    # "wait for a GPU to free up before launching the next one".
    wait_for_gpu = args.wait_for_gpu or (
        scheduling_enabled and args.one_per_gpu and not args.dry_run
    )

    # ── Launch loop: assigned experiments ──────────────────────────────
    stats = {"launched": 0, "skipped": 0, "error": 0, "dry-run": 0}
    for i, exp in enumerate(assigned):
        inner, log_path = build_inner_command(
            exp,
            train_script=train_script,
            log_dir=log_dir,
            conda_base=conda_base,
            conda_env=args.conda_env,
            keep_alive=args.keep_alive,
            storage_root=args.storage_root,
            resume=args.resume,
        )
        status = launch_one(exp, inner, log_path, args)
        stats[status] += 1
        # Commit a reservation for every successfully launched session.
        # We use time.time() now so warmup is counted from the moment
        # tmux returned, which matches when the child process begins
        # its conda/python import chain.
        if status == "launched" and exp.gpu_index is not None:
            run_reservations.append(
                _Reservation(
                    gpu_index=exp.gpu_index,
                    mem_mb=exp.mem_mb,
                    committed_at=time.time(),
                )
            )
            # Optional stagger: smear concurrent CUDA init cost so two
            # large jobs don't both hit ``torch.cuda`` allocation at
            # the exact same moment on the same device (driver-level
            # contention has been observed on 7B+ models).
            if args.launch_stagger > 0 and i < len(assigned) - 1 and not args.dry_run:
                time.sleep(args.launch_stagger)

    # ── Overflow handling: --wait-for-gpu drains the queue ─────────────
    if queued and scheduling_enabled and wait_for_gpu and not args.dry_run:
        mode_tag = "one-per-gpu" if args.one_per_gpu else "memory-pack"
        print(
            f"\n[wait-for-gpu:{mode_tag}] {len(queued)} experiment(s) queued — "
            f"polling nvidia-smi every {args.poll_interval}s "
            f"(warmup={args.warmup_seconds}s, "
            f"stagger={args.launch_stagger}s).\n"
        )
        while queued:
            time.sleep(args.poll_interval)
            fresh = detect_gpus(restrict_indices=restrict)
            if not fresh:
                print(
                    "[wait-for-gpu] nvidia-smi returned empty — aborting wait "
                    "loop; remaining experiments left unscheduled.",
                    file=sys.stderr,
                )
                break
            # CRITICAL: replay the reservation ledger onto the freshly
            # polled GPU list BEFORE scheduling. This subtracts memory
            # for any session we launched within the last
            # --warmup-seconds whose allocation may not yet be visible
            # in nvidia-smi, and drops stale reservations whose memory
            # IS already reflected in the live numbers.
            now = time.time()
            run_reservations = _apply_reservations(
                fresh, run_reservations, now, args.warmup_seconds
            )
            if args.one_per_gpu:
                new_assigned, queued = schedule_one_per_gpu(
                    queued, fresh, args.gpu_idle_mem_fraction
                )
            else:
                new_assigned, queued = schedule_assignments(
                    queued, fresh, args.gpu_memory_fraction
                )
            if not new_assigned:
                # Nothing placed this round — print a one-liner and keep waiting.
                active_res = len(run_reservations)
                if args.one_per_gpu:
                    n_idle = sum(
                        1
                        for g in fresh
                        if g.total_mb > 0
                        and g.reserved_mb == 0
                        and (g.free_mb / g.total_mb) >= args.gpu_idle_mem_fraction
                    )
                    print(
                        f"[wait-for-gpu:one-per-gpu] still waiting "
                        f"({len(queued)} left; idle_gpus={n_idle}; "
                        f"active_reservations={active_res})",
                        flush=True,
                    )
                else:
                    top = queued[0]
                    headroom = max(
                        (g.available_mb(args.gpu_memory_fraction) for g in fresh),
                        default=0,
                    )
                    print(
                        f"[wait-for-gpu] still waiting ({len(queued)} left; "
                        f"largest GPU avail={headroom} MiB vs need={top.mem_mb} "
                        f"MiB; active_reservations={active_res})",
                        flush=True,
                    )
                continue
            for j, exp in enumerate(new_assigned):
                inner, log_path = build_inner_command(
                    exp,
                    train_script=train_script,
                    log_dir=log_dir,
                    conda_base=conda_base,
                    conda_env=args.conda_env,
                    keep_alive=args.keep_alive,
                    storage_root=args.storage_root,
                    resume=args.resume,
                )
                status = launch_one(exp, inner, log_path, args)
                stats[status] += 1
                if status == "launched" and exp.gpu_index is not None:
                    run_reservations.append(
                        _Reservation(
                            gpu_index=exp.gpu_index,
                            mem_mb=exp.mem_mb,
                            committed_at=time.time(),
                        )
                    )
                    if args.launch_stagger > 0 and j < len(new_assigned) - 1:
                        time.sleep(args.launch_stagger)

    # ── Summary ────────────────────────────────────────────────────────
    if args.dry_run:
        print(
            f"Dry-run complete: {len(assigned)} session(s) would launch, "
            f"{len(queued)} would be queued."
        )
        if queued:
            if args.one_per_gpu:
                print("Queued (no idle GPU currently available):")
                for exp in queued:
                    print(f"  - {exp.session_name}")
            else:
                print("Queued (would not fit with current free memory):")
                for exp in queued:
                    print(f"  - {exp.session_name} (needs {exp.mem_mb} MiB)")
    else:
        print(
            f"Launched: {stats['launched']} | Skipped: {stats['skipped']} | "
            f"Errors: {stats['error']} | Queued (unlaunched): {len(queued)} | "
            f"Total: {len(experiments)}"
        )
        if queued:
            if args.one_per_gpu:
                print(
                    "\nUnlaunched experiments (no idle GPU — wait loop "
                    "exited early, e.g. nvidia-smi became unavailable):"
                )
                for exp in queued:
                    print(f"  - {exp.session_name}")
            else:
                print("\nUnlaunched experiments (no GPU had enough free memory):")
                for exp in queued:
                    print(f"  - {exp.session_name} (needs {exp.mem_mb} MiB)")
                print("  → re-run with --wait-for-gpu to drain as memory frees.")
        print(f"\nLogs directory: {log_dir}")
        if stats["launched"] > 0:
            print(
                "\nUseful commands:\n"
                "  tmux ls                       # list sessions\n"
                "  tmux attach -t <name>         # attach (Ctrl-b d to detach)\n"
                "  tmux kill-session -t <name>   # terminate a session\n"
                f"  tail -f {log_dir}/<name>.log  # watch output"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
