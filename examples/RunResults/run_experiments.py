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
            "Keep the tmux session open after the training command exits "
            "(useful to inspect the final stack/log). Default: session "
            "closes automatically when the command finishes."
        ),
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help=(
            "Python interpreter path to use inside each tmux session "
            "(default: the interpreter running this launcher)."
        ),
    )
    return parser.parse_args()


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
        # We cd into PROJECT_ROOT first so relative paths inside the
        # training script (and any .env lookup) resolve correctly.
        inner = (
            f"cd {shlex.quote(str(PROJECT_ROOT))} && "
            f"{shlex.quote(args.python)} {shlex.quote(str(train_script))} "
            f"-c {shlex.quote(str(cfg_path))}"
        )
        if args.keep_alive:
            # `exec $SHELL` replaces the tmux pane's process with an
            # interactive shell so the session stays attached after the
            # training command exits (success or failure).
            inner = f"{inner}; echo; echo '[run_experiments] training exited — shell kept open'; exec $SHELL"
        plan.append((raw, cfg_path, session_name, inner))

    # ── Preview header ───────────────────────────────────────────────
    print(f"Project root   : {PROJECT_ROOT}")
    print(f"Training script: {train_script.relative_to(PROJECT_ROOT)}")
    print(f"Dataset dir    : {dataset_dir.relative_to(PROJECT_ROOT)}")
    print(f"Python         : {args.python}")
    print(f"Experiments    : {len(plan)}")
    print()

    # ── Launch loop ──────────────────────────────────────────────────
    launched = 0
    skipped = 0
    for raw, cfg_path, session_name, inner in plan:
        rel_cfg = cfg_path.relative_to(PROJECT_ROOT)
        tmux_cmd = ["tmux", "new-session", "-d", "-s", session_name, inner]
        print(f"[{session_name}]")
        print(f"  config : {rel_cfg}")
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
        if launched > 0:
            print(
                "\nUseful tmux commands:\n"
                "  tmux ls                  # list sessions\n"
                "  tmux attach -t <name>    # attach (Ctrl-b d to detach)\n"
                "  tmux kill-session -t <name>"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
