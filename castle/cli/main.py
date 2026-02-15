"""
castle/cli/main.py
Main typer application and subcommand registration.
"""

import typer

app = typer.Typer(
    name="castle",
    help="CASTLE — Animal Behavior Analysis CLI",
    no_args_is_help=True,
)

# Register subcommands
from castle.cli import project_cmd, track_cmd, extract_cmd, cluster_cmd

app.add_typer(project_cmd.app, name="project", help="Project management")
app.add_typer(cluster_cmd.app, name="cluster", help="Clustering operations")
app.registered_commands += track_cmd.app.registered_commands
app.registered_commands += extract_cmd.app.registered_commands


# Top-level alias: `castle info` → shortcut for project info
@app.command("info")
def info_alias(
    project: str = typer.Argument(..., help="Project name"),
    storage: str = typer.Option(..., "--storage", "-s", help="Storage directory path"),
):
    """Show project info (alias for 'project info')."""
    project_cmd.info(project, storage)


if __name__ == "__main__":
    app()
