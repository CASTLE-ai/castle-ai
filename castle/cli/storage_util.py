"""
castle/cli/storage_util.py
Utility for resolving the --storage path from multiple sources.
"""

import os

import typer


def get_storage(storage: str = None) -> str:
    """Resolve storage path from argument, env var, or current directory.

    Priority:
      1. Explicit --storage/-s argument
      2. CASTLE_STORAGE environment variable
      3. Current directory (if castle_config.json exists)

    Returns the resolved path, or exits with a helpful error.
    """
    if storage:
        return storage

    # Check environment variable
    env_storage = os.environ.get("CASTLE_STORAGE")
    if env_storage:
        return env_storage

    # Check current directory for castle projects
    if os.path.isfile("castle_config.json"):
        return os.getcwd()

    # Nothing found — print helpful message and exit
    typer.echo(
        "Error: No storage path. Set CASTLE_STORAGE or use --storage/-s",
        err=True,
    )
    raise typer.Exit(code=1)
