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
from castle.cli import project_cmd, track_cmd, extract_cmd, cluster_cmd, mcp_cmd
from castle.cli.ethogram_cmd import app as ethogram_app

app.add_typer(project_cmd.app, name="project", help="Project management")
app.add_typer(cluster_cmd.app, name="cluster", help="Clustering operations")
app.add_typer(mcp_cmd.app, name="mcp", help="MCP (Model Context Protocol) server")
app.add_typer(ethogram_app, name="ethogram")
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


@app.command("gui")
def gui(
    storage: str = typer.Option("projects/", "--storage", "-s", help="Storage directory path"),
    project: str = typer.Option(None, "--project", "-p", help="Project name to open"),
):
    """Launch the Desktop GUI."""
    import subprocess, sys
    cmd = [sys.executable, "-m", "castle.desktop", "--storage", storage]
    if project:
        cmd += ["--project", project]
    subprocess.Popen(cmd)


if __name__ == "__main__":
    app()
