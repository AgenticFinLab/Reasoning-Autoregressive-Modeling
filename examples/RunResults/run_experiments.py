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

    Opt-out: pass ``--no-gpu-schedule`` to disable the whole mechanism
    and launch everything in parallel (the pre-scheduler behaviour).

Usage:
    # Launch two baseline experiments in parallel:
    python3 examples/RunResults/run_experiments.py -m builder -d GSM8K \\
        -e train_builder_Qwen2.5-0.5B_2level.yml \\
           train_builder_Qwen2.5-0.5B_4level.yml

    # Mix baseline + AutoWeighted variants in one launch:
    python3 examples/RunResults/run_experiments.py -m builder -d GSM8K \\
        -e train_builder_Qwen2.5-0.5B_2level.yml \\
           AutoWeighted/train_builder_Qwen2.5-0.5B_2level.yml

    # Keep the launcher alive and drain a large queue as GPUs free up:
    python3 examples/RunResults/run_experiments.py -m builder -d GSM8K \\
        -e AutoWeighted/*.yml --wait-for-gpu

    # Preview the plan (incl. GPU assignments) without touching tmux:
    python3 examples/RunResults/run_experiments.py -m builder -d GSM8K \\
        -e AutoWeighted/*.yml --dry-run

    # Restrict scheduling to GPUs 0 and 2:
    python3 examples/RunResults/run_experiments.py -m builder -d GSM8K \\
        -e AutoWeighted/*.yml --gpus 0,2

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
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ── Project paths ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "nlcpV4"
VALID_MODULES = ("builder", "predictor")


# =====================================================================
# CLI
# =====================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch multiple train_{module}.py experiments in parallel tmux "
            "sessions, with automatic GPU-memory-aware scheduling."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
            "If a tmux session with the target name already exists, kill it "
            "first and relaunch. Without this flag, the experiment is "
            "skipped with a warning."
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
    import json as _json

    result = subprocess.run(
        ["conda", "env", "list", "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = _json.loads(result.stdout)
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
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    reason_cfg = model_cfg["reason_model"]
    pyramid_cfg = model_cfg["pyramid"]
    training_cfg = cfg["training"]

    params_b = parse_model_params_b(reason_cfg["reason_model_name"])
    params = params_b * 1e9

    dtype = reason_cfg.get("torch_dtype", "float32")
    bytes_per_elem = 2 if dtype in ("bfloat16", "float16") else 4

    # LoRA / frozen backbone means no full-model grads / Adam state.
    train_rm = training_cfg.get("reason_model", {}) or {}
    lora_cfg = train_rm.get("lora")
    freeze = bool(train_rm.get("freeze", False))
    lora_or_frozen = (lora_cfg is not None) or freeze

    weights_mb = params * bytes_per_elem / 1e6
    optimizer_mb = weights_mb * (0.2 if lora_or_frozen else 3.0)

    # Activation pressure from the autoregressive reasoning forward:
    # [Q | concepts | S] sequence through the full reason_model.
    batch_size = int(training_cfg.get("batch_size", 8))
    max_seq_len = int(pyramid_cfg.get("max_seq_len", 1024))
    hidden_dim = int(pyramid_cfg.get("hidden_dim", 1024))
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
    result = subprocess.run(
        ["tmux", "has-session", "-t", f"={session_name}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def kill_tmux_session(session_name: str) -> None:
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
    inner_cmd = (
        f"cd {shlex.quote(str(PROJECT_ROOT))} && "
        f"source {shlex.quote(str(conda_sh))} && "
        f"conda activate {shlex.quote(conda_env)} && "
        f"{gpu_prefix}PYTHONUNBUFFERED=1 python "
        f"{shlex.quote(str(train_script))} "
        f"-c {shlex.quote(str(exp.cfg_path))} "
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
    args = parse_args()

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

    # ── Build Experiment objects + estimate memory ─────────────────────
    experiments: list[Experiment] = []
    seen_names: set[str] = set()
    for raw, cfg_path in resolved_experiments:
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

        if args.mem_per_exp_mb is not None:
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
            )
        )

    # ── GPU scheduling ─────────────────────────────────────────────────
    restrict = parse_gpu_restriction(args.gpus)
    gpus: list[GPUInfo] = []
    scheduling_enabled = not args.no_gpu_schedule
    if scheduling_enabled:
        gpus = detect_gpus(restrict_indices=restrict)
        if not gpus:
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
        print(
            f"GPU scheduling : ENABLED (fraction={args.gpu_memory_fraction}, "
            f"safety={args.mem_safety_factor})"
        )
        print(f"Detected GPUs  : {len(gpus)}")
        for g in gpus:
            print(f"  GPU{g.index} ({g.name}): " f"{g.free_mb}/{g.total_mb} MiB free")
    else:
        print("GPU scheduling : DISABLED")
    print()

    # ── Run the scheduler (or skip it) ─────────────────────────────────
    if scheduling_enabled:
        assigned, queued = schedule_assignments(
            experiments, gpus, args.gpu_memory_fraction
        )
    else:
        assigned, queued = experiments, []

    # ── Launch loop: assigned experiments ──────────────────────────────
    stats = {"launched": 0, "skipped": 0, "error": 0, "dry-run": 0}
    for exp in assigned:
        inner, log_path = build_inner_command(
            exp,
            train_script=train_script,
            log_dir=log_dir,
            conda_base=conda_base,
            conda_env=args.conda_env,
            keep_alive=args.keep_alive,
        )
        status = launch_one(exp, inner, log_path, args)
        stats[status] += 1

    # ── Overflow handling: --wait-for-gpu drains the queue ─────────────
    if queued and scheduling_enabled and args.wait_for_gpu and not args.dry_run:
        print(
            f"\n[wait-for-gpu] {len(queued)} experiment(s) queued — polling "
            f"nvidia-smi every {args.poll_interval}s until each fits.\n"
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
            new_assigned, queued = schedule_assignments(
                queued, fresh, args.gpu_memory_fraction
            )
            if not new_assigned:
                # Nothing placed this round — print a one-liner and keep waiting.
                top = queued[0]
                headroom = max(
                    (g.available_mb(args.gpu_memory_fraction) for g in fresh),
                    default=0,
                )
                print(
                    f"[wait-for-gpu] still waiting ({len(queued)} left; "
                    f"largest GPU free={headroom} MiB vs need={top.mem_mb} MiB)",
                    flush=True,
                )
                continue
            for exp in new_assigned:
                inner, log_path = build_inner_command(
                    exp,
                    train_script=train_script,
                    log_dir=log_dir,
                    conda_base=conda_base,
                    conda_env=args.conda_env,
                    keep_alive=args.keep_alive,
                )
                status = launch_one(exp, inner, log_path, args)
                stats[status] += 1

    # ── Summary ────────────────────────────────────────────────────────
    if args.dry_run:
        print(
            f"Dry-run complete: {len(assigned)} session(s) would launch, "
            f"{len(queued)} would be queued."
        )
        if queued:
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
