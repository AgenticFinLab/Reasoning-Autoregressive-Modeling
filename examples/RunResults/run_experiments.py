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
    Examples:
        train_builder_Qwen2.5-0.5B_2level.yml
            -> builder-GSM8K-train_builder_Qwen2.5-0.5B_2level
        AutoWeighted/train_builder_Qwen2.5-0.5B_2level.yml
            -> builder-GSM8K-AutoWeighted-train_builder_Qwen2.5-0.5B_2level

Usage:
    # Launch two baseline experiments in parallel:
    python3 examples/RunResults/run_experiments.py -m builder -d GSM8K \\
        -e train_builder_Qwen2.5-0.5B_2level.yml \\
           train_builder_Qwen2.5-0.5B_4level.yml

    # Mix baseline + AutoWeighted variants in one launch:
    python3 examples/RunResults/run_experiments.py -m builder -d GSM8K \\
        -e train_builder_Qwen2.5-0.5B_2level.yml \\
           AutoWeighted/train_builder_Qwen2.5-0.5B_2level.yml

    # Preview the commands without touching tmux:
    python3 examples/RunResults/run_experiments.py -m builder -d GSM8K \\
        -e AutoWeighted/*.yml --dry-run

    # Replace any pre-existing tmux session with the same name:
    python3 examples/RunResults/run_experiments.py -m builder -d GSM8K \\
        -e train_builder_Qwen2.5-0.5B_2level.yml --kill-existing

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
         training script. This keeps the command line clean:
         ``python examples/nlcpV4/train_{module}.py -c {cfg}``.
    Creating the env happens once in the LAUNCHER process so parallel
    sessions never race each other to create the same env.
"""

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

# ── Project paths ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "nlcpV4"
VALID_MODULES = ("builder", "predictor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch multiple train_{module}.py experiments in parallel tmux sessions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the tmux commands without executing them.",
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
    return parser.parse_args()


# ── Conda helpers ────────────────────────────────────────────────────


def _conda_base() -> Path:
    """Return the conda base directory (``CONDA_PREFIX`` of the base env).

    Uses ``conda info --base`` so it works regardless of where conda was
    installed (miniconda/anaconda, user vs system). The returned path
    hosts ``etc/profile.d/conda.sh`` which the tmux sessions source to
    get ``conda activate`` in a non-interactive shell.
    """
    result = subprocess.run(
        ["conda", "info", "--base"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def _conda_env_exists(env_name: str) -> bool:
    """True iff ``conda env list`` contains an env named exactly ``env_name``.

    Parses ``conda env list --json`` so we match by basename regardless
    of the envs directory layout (user envs, shared envs, etc.).
    """
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
    """Guarantee ``env_name`` exists; create it if missing. Return conda base.

    Raises ``RuntimeError`` (with a clear message) if conda itself is
    not installed — fail-fast so the user does not discover this only
    after the tmux sessions silently die.
    """
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


def resolve_experiment(raw: str, dataset_dir: Path) -> Path:
    """Resolve a user-supplied experiment path to an absolute config path.

    Matches three accepted forms: bare filename, relative path (possibly
    nested), absolute path. Fails fast if the file does not exist — we
    do NOT silently fall back to another directory.
    """
    p = Path(raw)
    if p.is_absolute():
        resolved = p
    else:
        resolved = dataset_dir / p
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
    """Derive the ``{experiment_label}`` part of the tmux session name.

    The label mirrors the config's location under ``dataset_dir``, with
    path separators replaced by ``-`` and the ``.yml`` suffix stripped.
    Configs outside ``dataset_dir`` fall back to just the file stem.
    """
    try:
        rel = config_path.relative_to(dataset_dir)
    except ValueError:
        return config_path.stem
    parts = [*rel.parent.parts, rel.stem]
    # Filter empty parts that appear when rel.parent is '.'
    parts = [p for p in parts if p and p != "."]
    return "-".join(parts)


def sanitize_session_name(name: str) -> str:
    """tmux disallows ``.`` and ``:`` in session names — replace with ``_``.

    Everything else (letters, digits, ``-``, ``_``) is preserved so the
    user can still pattern-match session names with ``tmux ls | grep``.
    """
    return name.replace(".", "_").replace(":", "_")


def tmux_session_exists(session_name: str) -> bool:
    """Return True iff tmux has a session with this exact name."""
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


def main() -> int:
    args = parse_args()

    # ── Validate tmux availability ───────────────────────────────────
    if not args.dry_run and shutil.which("tmux") is None:
        print(
            "[ERROR] tmux is not installed or not on PATH. "
            "Install it (e.g., `brew install tmux`) or pass --dry-run.",
            file=sys.stderr,
        )
        return 2

    # ── Resolve paths ───────────────────────────────────────────────
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

    # ── Conda pre-flight (before log dir, before tmux) ─────────────────
    # Runs in the LAUNCHER process so parallel tmux sessions do not race
    # each other to ``conda create`` the same env. Skipped on --dry-run
    # so plan preview is side-effect-free.
    conda_base: Path | None = None
    if not args.dry_run:
        try:
            conda_base = ensure_conda_env(args.conda_env, args.python_version)
        except (RuntimeError, subprocess.CalledProcessError) as e:
            print(f"[ERROR] conda setup failed: {e}", file=sys.stderr)
            return 2

    # Resolve (and create) the log directory — tmux sessions close as soon
    # as their command exits, so without an on-disk log a quick crash would
    # leave no trace. Every session writes to {log_dir}/{session}.log.
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

    # ── Build launch plan ────────────────────────────────────────────
    plan = []
    seen_names: set[str] = set()
    for raw, cfg_path in resolved_experiments:
        label = experiment_label(cfg_path, dataset_dir)
        session_name = sanitize_session_name(f"{args.module}-{args.dataset}-{label}")
        if session_name in seen_names:
            print(
                f"[ERROR] Duplicate session name within this launch: "
                f"{session_name!r} (from {raw!r}). Each -e must yield a "
                f"unique label.",
                file=sys.stderr,
            )
            return 2
        seen_names.add(session_name)

        # Build the shell command executed inside the tmux session.
        # Sequence (run inside a single ``bash -o pipefail -c '...'``):
        #   1. cd into PROJECT_ROOT (so relative paths + .env resolve)
        #   2. source conda.sh (gives ``conda activate`` to the non-
        #      interactive bash spawned by tmux)
        #   3. conda activate {env}
        #   4. PYTHONUNBUFFERED=1 python <script> -c <cfg> 2>&1 | tee <log>
        # ``bash -o pipefail`` propagates Python's exit code through
        # tee; otherwise tee's success would always mask a crash.
        # PYTHONUNBUFFERED=1 ensures the log captures traceback output
        # that would otherwise be lost in stdout buffers on abort.
        log_path = log_dir / f"{session_name}.log"
        # conda_base is None only on --dry-run; still emit a readable
        # preview by using the placeholder path.
        conda_sh = (
            (conda_base / "etc" / "profile.d" / "conda.sh")
            if conda_base is not None
            else Path("$(conda info --base)/etc/profile.d/conda.sh")
        )
        inner_cmd = (
            f"cd {shlex.quote(str(PROJECT_ROOT))} && "
            f"source {shlex.quote(str(conda_sh))} && "
            f"conda activate {shlex.quote(args.conda_env)} && "
            f"PYTHONUNBUFFERED=1 python {shlex.quote(str(train_script))} "
            f"-c {shlex.quote(str(cfg_path))} 2>&1 | tee {shlex.quote(str(log_path))}"
        )
        piped = f"bash -o pipefail -c {shlex.quote(inner_cmd)}"
        if args.keep_alive:
            # Always keep session open, regardless of exit code.
            tail = (
                f"ec=$?; echo; "
                f"echo '[run_experiments] exited with code '\"$ec\"' — log: {log_path}'; "
                f"exec $SHELL"
            )
        else:
            # Keep the session open ONLY on failure so the user can
            # still inspect the pane; close cleanly on success.
            tail = (
                f"ec=$?; "
                f'if [ "$ec" != "0" ]; then '
                f"echo; "
                f"echo '[run_experiments] FAILED with code '\"$ec\"' — log: {log_path}'; "
                f"exec $SHELL; "
                f"fi"
            )
        inner = f"{piped}; {tail}"
        plan.append((raw, cfg_path, session_name, inner, log_path))

    # ── Preview header ───────────────────────────────────────────────
    print(f"Project root   : {PROJECT_ROOT}")
    print(f"Training script: {train_script.relative_to(PROJECT_ROOT)}")
    print(f"Dataset dir    : {dataset_dir.relative_to(PROJECT_ROOT)}")
    print(f"Conda env      : {args.conda_env}")
    print(f"Experiments    : {len(plan)}")
    print()

    # ── Launch loop ──────────────────────────────────────────────────
    launched = 0
    skipped = 0
    for raw, cfg_path, session_name, inner, log_path in plan:
        rel_cfg = cfg_path.relative_to(PROJECT_ROOT)
        try:
            rel_log = log_path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel_log = log_path
        tmux_cmd = ["tmux", "new-session", "-d", "-s", session_name, inner]
        print(f"[{session_name}]")
        print(f"  config : {rel_cfg}")
        print(f"  log    : {rel_log}")
        print(f"  command: {inner}")

        if args.dry_run:
            print("  (dry-run: not launched)")
            print()
            continue

        if tmux_session_exists(session_name):
            if args.kill_existing:
                print(f"  [info] killing existing session {session_name!r}")
                kill_tmux_session(session_name)
            else:
                print(
                    f"  [SKIP] session already exists. Use --kill-existing to replace."
                )
                skipped += 1
                print()
                continue

        result = subprocess.run(tmux_cmd)
        if result.returncode != 0:
            print(
                f"  [ERROR] tmux new-session failed (exit {result.returncode})",
                file=sys.stderr,
            )
            print()
            continue
        print(f"  [OK] launched detached session {session_name!r}")
        launched += 1
        print()

    # ── Summary ──────────────────────────────────────────────────────
    if args.dry_run:
        print(f"Dry-run complete: {len(plan)} session(s) would have been launched.")
    else:
        print(
            f"Launched: {launched} | Skipped: {skipped} | "
            f"Total planned: {len(plan)}"
        )
        print(f"Logs directory: {log_dir}")
        if launched > 0:
            print(
                "\nUseful commands:\n"
                "  tmux ls                       # list sessions\n"
                "  tmux attach -t <name>         # attach (Ctrl-b d to detach)\n"
                "  tmux kill-session -t <name>   # terminate a session\n"
                f"  tail -f {log_dir}/<name>.log  # watch output even if session closed"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
