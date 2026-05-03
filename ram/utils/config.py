"""
YAML configuration loading for TAR framework.

Supports !include directive for modular configs.
"""

import os
from pathlib import Path
from typing import Any

import yaml

# =============================================================================
# Config Loading with !include Support
# =============================================================================


class IncludeLoader(yaml.SafeLoader):
    """YAML Loader with !include support.

    Supports:
        !include path/to/file.yml        # Include entire file
        !include path/to/file.yml:key    # Include specific key from file
        !include path/to/file.yml:a.b.c  # Include nested key
    """

    def __init__(self, stream):
        self._root = (
            os.path.dirname(stream.name) if hasattr(stream, "name") else os.getcwd()
        )
        super().__init__(stream)


def _include_constructor(loader: IncludeLoader, node: yaml.Node) -> Any:
    """Handle !include directive.

    Args:
        loader: YAML loader instance
        node: YAML node with include path

    Returns:
        Included content (dict, list, or scalar)
    """
    value = loader.construct_scalar(node)

    # Check for key selector: !include file.yml:key
    if ":" in value and not value.startswith("/"):
        # Handle Windows paths (C:\...) vs key selector
        parts = value.rsplit(":", 1)
        if len(parts) == 2 and not parts[0].endswith("\\"):
            filepath, key = parts
        else:
            filepath, key = value, None
    else:
        filepath, key = value, None

    # Resolve relative path
    if not os.path.isabs(filepath):
        filepath = os.path.join(loader._root, filepath)

    # Load included file
    with open(filepath, "r", encoding="utf-8") as f:
        content = yaml.load(f, IncludeLoader)

    # Extract specific key if specified
    if key is not None:
        for k in key.split("."):
            content = content[k]

    return content


IncludeLoader.add_constructor("!include", _include_constructor)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file with !include support.

    Supports:
        !include path/to/file.yml        # Include entire file
        !include path/to/file.yml:key    # Include specific key
        !include path/to/file.yml:a.b.c  # Include nested key

    Example config.yml:
        model:
          encoder: !include encoders/bert.yml
          decoder: !include decoders/gpt2.yml:decoder

    Args:
        config_path: Path to the YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, IncludeLoader)
    return config


# =============================================================================
# Storage-root override
# =============================================================================


_STORAGE_ROOT_KEYS = ("save_folder", "checkpoint_path", "log_path")


def apply_storage_root(config: dict, storage_root: str | os.PathLike) -> dict:
    """Redirect relative output paths under ``config['log']`` to ``storage_root``.

    Rewrites the three well-known output keys in ``config['log']``
    (``save_folder``, ``checkpoint_path``, ``log_path``) so that any
    *relative* value is prepended with ``storage_root``. Absolute paths
    are kept verbatim so the user can always force a specific location
    from YAML.

    This enables a single ``-s/--storage-root`` CLI flag (the default
    across every CLI entry point is ``"./"`` so behaviour is always
    explicit — never silently derived from a project root or some
    implicit working directory). Pass ``-s /Data/<proj>`` on remote
    servers to redirect all relative paths under a specific root
    without editing every YAML.

    ``storage_root`` is REQUIRED (no sentinel/None/empty no-op). Every
    CLI entry point sets the argparse default to ``"./"`` and forwards
    the concrete string here, so callers must always pass an explicit
    value — this keeps the storage contract visible at every call site.

    Example:
        >>> cfg = {"log": {
        ...     "save_folder": "EXPERIMENT/nlcpV4/builder/exp1",
        ...     "checkpoint_path": "EXPERIMENT/nlcpV4/builder/exp1/checkpoints",
        ...     "log_path": "/tmp/force_absolute/logs",
        ... }}
        >>> apply_storage_root(cfg, "/Data/<proj>")  # doctest: +NORMALIZE_WHITESPACE
        {'log': {'save_folder': '/Data/<proj>/EXPERIMENT/nlcpV4/builder/exp1',
                 'checkpoint_path': '/Data/<proj>/EXPERIMENT/nlcpV4/builder/exp1/checkpoints',
                 'log_path': '/tmp/force_absolute/logs'}}

    Args:
        config: Configuration dict produced by ``load_config``. Must
            contain a ``log`` sub-dict with the three keys listed
            above (fail-fast KeyError otherwise).
        storage_root: Directory to prepend to relative output paths.
            Required. Pass ``"./"`` for the CLI default.

    Returns:
        The same ``config`` dict, mutated in place (also returned for
        convenient chaining).
    """
    root = Path(storage_root)
    log_cfg = config["log"]
    for key in _STORAGE_ROOT_KEYS:
        raw = log_cfg[key]
        p = Path(raw)
        if p.is_absolute():
            continue
        log_cfg[key] = str(root / p)
    return config


def print_storage_paths(
    config: dict,
    storage_root: str | os.PathLike,
) -> None:
    """Print a deterministic summary of the resolved log output paths.

    Purpose:
        Every CLI entry point MUST show the user exactly where data
        will be written/read. Silent defaults (e.g. implicit project
        root) are forbidden — if ``-s`` is not supplied the CLI
        default is ``"./"`` and that value (and its concrete absolute
        resolution) is surfaced here.

    Output format (one block, written to stdout)::

        [STORAGE] storage_root = './' (cwd=/abs/path/to/cwd)
        [STORAGE]   save_folder     = EXPERIMENT/nlcpV4/builder/exp1
        [STORAGE]                     (absolute: /abs/path/to/cwd/EXPERIMENT/nlcpV4/builder/exp1)
        [STORAGE]   checkpoint_path = EXPERIMENT/nlcpV4/builder/exp1/checkpoints
        [STORAGE]                     (absolute: /abs/.../checkpoints)
        [STORAGE]   log_path        = EXPERIMENT/nlcpV4/builder/exp1/logs
        [STORAGE]                     (absolute: /abs/.../logs)

    Args:
        config: Config dict containing the ``log`` sub-dict with the
            three ``_STORAGE_ROOT_KEYS``. Typically called AFTER
            ``apply_storage_root`` so the values shown are the ones
            actually used at runtime.
        storage_root: The ``-s`` value. Required. Displayed verbatim
            so the user can verify the flag they passed.
    """
    shown = str(storage_root)
    cwd = Path.cwd().resolve()
    print(f"[STORAGE] storage_root = {shown!r} (cwd={cwd})")
    log_cfg = config["log"]
    # Align column width for scanability.
    width = max(len(k) for k in _STORAGE_ROOT_KEYS)
    for key in _STORAGE_ROOT_KEYS:
        val = log_cfg[key]
        abs_path = Path(val).expanduser()
        if not abs_path.is_absolute():
            abs_path = (cwd / abs_path).resolve()
        print(f"[STORAGE]   {key:<{width}s} = {val}")
        print(f"[STORAGE]   {' ' * width}   (absolute: {abs_path})")
