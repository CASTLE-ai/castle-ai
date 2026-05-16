"""
castle/cli/main.py
Main typer application and subcommand registration.
"""

import json
from pathlib import Path
from typing import Optional

import typer

from castle.cli.storage_util import get_storage

app = typer.Typer(
    name="castle",
    help="CASTLE — Animal Behavior Analysis CLI",
    no_args_is_help=True,
)


def _load_config_file(path: Path) -> dict:
    """Read a JSON or YAML config file into a plain dict.

    Args:
        path: Path to a ``.json``, ``.yaml`` or ``.yml`` file.

    Returns:
        Parsed config as a dict. Returns an empty dict for empty files.

    Raises:
        typer.BadParameter: If the file does not exist, has an unsupported
            extension, or contains invalid syntax.
    """
    if not path.exists():
        raise typer.BadParameter(f"Config file not found: {path}")
    suffix = path.suffix.lower()
    text = path.read_text()
    if suffix == ".json":
        try:
            data = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError as e:
            raise typer.BadParameter(f"Invalid JSON in {path}: {e}") from e
    elif suffix in (".yaml", ".yml"):
        import yaml

        try:
            data = yaml.safe_load(text) or {}
        except yaml.YAMLError as e:
            raise typer.BadParameter(f"Invalid YAML in {path}: {e}") from e
    else:
        raise typer.BadParameter(
            f"Unsupported config extension {suffix!r}; expected .json, .yaml or .yml"
        )
    if not isinstance(data, dict):
        raise typer.BadParameter(
            f"Config file must be a mapping at the top level (got {type(data).__name__})"
        )
    return data


def _apply_device_override(device: str) -> str:
    """Apply ``--device`` to :mod:`castle.core.environment`'s singleton.

    Args:
        device: ``'auto'``, ``'cuda'``, ``'mps'``, or ``'cpu'``.

    Returns:
        The device string that was actually applied (after resolving ``'auto'``).
    """
    from castle.core import environment as _env_mod

    if device == "auto":
        return _env_mod.env.device
    _env_mod.env.device = device
    return device


@app.callback()
def main_callback(
    ctx: typer.Context,
    seed: int = typer.Option(
        42,
        "--seed",
        envvar="CASTLE_SEED",
        help=(
            "Master seed for every stochastic component except UMAP "
            "(UMAP keeps its own re-roll/lock UX). Default: 42. "
            "Override via env CASTLE_SEED."
        ),
    ),
    strict_cuda: bool = typer.Option(
        False,
        "--strict-cuda",
        envvar="CASTLE_STRICT_CUDA",
        help=(
            "Force bit-identical CUDA output (cudnn.deterministic + "
            "use_deterministic_algorithms). ~10%% slower; use for paper-grade runs."
        ),
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        envvar="CASTLE_DEVICE",
        help=(
            "Compute device. 'auto' (default) detects CUDA/MPS/CPU. "
            "Set explicitly to override — e.g. 'cpu' to force CPU even when CUDA is present."
        ),
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        envvar="CASTLE_CONFIG",
        help=(
            "Optional JSON or YAML config file. Loaded values land in ctx.obj['config'] "
            "and are read by subcommands as default overrides (subcommand CLI flags still win)."
        ),
    ),
) -> None:
    """Apply global options before any subcommand runs."""
    from castle.core.seed import set_global_seed

    set_global_seed(seed, strict_cuda=strict_cuda)
    resolved_device = _apply_device_override(device)
    config_data = _load_config_file(config) if config else {}

    ctx.ensure_object(dict)
    ctx.obj["master_seed"] = seed
    ctx.obj["strict_cuda"] = strict_cuda
    ctx.obj["device"] = resolved_device
    ctx.obj["config"] = config_data

# Register subcommands — must appear after app is created (CLI pattern)
from castle.cli import project_cmd, track_cmd, extract_cmd, cluster_cmd, mcp_cmd  # noqa: E402
from castle.cli.ethogram_cmd import app as ethogram_app  # noqa: E402
from castle.cli.compare_cmd import app as compare_app  # noqa: E402
from castle.cli.preprocess_cmd import app as preprocess_app  # noqa: E402
from castle.cli.batch_cmd import app as batch_app  # noqa: E402

app.add_typer(project_cmd.app, name="project", help="Project management")
app.add_typer(cluster_cmd.app, name="cluster", help="Clustering operations")
app.add_typer(mcp_cmd.app, name="mcp", help="MCP (Model Context Protocol) server")
app.add_typer(ethogram_app, name="ethogram")
app.add_typer(compare_app, name="compare")
app.add_typer(batch_app, name="batch", help="Batch processing across multiple experiments")
app.registered_commands += track_cmd.app.registered_commands
app.registered_commands += extract_cmd.app.registered_commands
app.registered_commands += preprocess_app.registered_commands


# Top-level alias: `castle info` → shortcut for project info
@app.command("info")
def info_alias(
    project: str = typer.Argument(..., help="Project name"),
    storage: str = typer.Option(None, "--storage", "-s", help="Storage directory (or set CASTLE_STORAGE env var)"),
):
    """Show project info (alias for 'project info')."""
    storage = get_storage(storage)
    project_cmd.info(project, storage)


@app.command("gui")
def gui(
    storage: str = typer.Option("projects/", "--storage", "-s", help="Storage directory path"),
    project: str = typer.Option(None, "--project", "-p", help="Project name to open"),
):
    """Launch the Desktop GUI."""
    import subprocess
    import sys
    cmd = [sys.executable, "-m", "castle.desktop", "--storage", storage]
    if project:
        cmd += ["--project", project]
    subprocess.Popen(cmd)


if __name__ == "__main__":
    app()
