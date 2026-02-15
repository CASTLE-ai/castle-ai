"""
castle/cli/project_cmd.py
Project management CLI commands: init, info, add-videos, list.
"""

import os
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from castle.service.project_service import (
    create_project,
    get_project_info,
    add_videos,
    add_videos_from_directory,
    list_projects,
)

console = Console()
app = typer.Typer(no_args_is_help=True)


@app.command("init")
def init(
    name: str = typer.Argument(..., help="Project name"),
    storage: str = typer.Option(..., "--storage", "-s", help="Storage directory path"),
):
    """Create a new CASTLE project."""
    try:
        result = create_project(storage, name)
        console.print(f"[green]✓[/green] Project [bold]{result['name']}[/bold] created at {result['path']}")
    except FileExistsError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(code=1)


@app.command("info")
def info(
    name: str = typer.Argument(..., help="Project name"),
    storage: str = typer.Option(..., "--storage", "-s", help="Storage directory path"),
):
    """Show project information."""
    result = get_project_info(storage, name)
    if 'error' in result:
        console.print(f"[red]✗[/red] {result['error']}")
        raise typer.Exit(code=1)

    table = Table(title=f"Project: {result['name']}")
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Path", result['path'])
    table.add_row("Videos", str(result['video_count']))
    table.add_row("Latent files", str(result['latent_count']))

    console.print(table)

    if result['videos']:
        console.print("\n[bold]Videos:[/bold]")
        for v in result['videos']:
            console.print(f"  • {v}")


@app.command("add-videos")
def add_videos_cmd(
    name: str = typer.Argument(..., help="Project name"),
    source: str = typer.Option(..., "--source", help="Directory or file path(s) to add"),
    storage: str = typer.Option(..., "--storage", "-s", help="Storage directory path"),
):
    """Add video files to a project from a directory or file list."""
    if os.path.isdir(source):
        result = add_videos_from_directory(storage, name, source)
        console.print(
            f"[green]✓[/green] Added {result['success_count']} videos "
            f"({result['fail_count']} failed)"
        )
        for msg in result['messages']:
            console.print(f"  {msg}")
    else:
        # Treat source as a comma-separated list of files
        video_paths = [p.strip() for p in source.split(",")]
        results = add_videos(storage, name, video_paths)
        for r in results:
            status = "[green]✓[/green]" if r['success'] else "[red]✗[/red]"
            console.print(f"  {status} {r['video_name']}: {r['message']}")


@app.command("list")
def list_cmd(
    storage: str = typer.Option(..., "--storage", "-s", help="Storage directory path"),
):
    """List all projects in storage."""
    projects = list_projects(storage)
    if not projects:
        console.print("[yellow]No projects found.[/yellow]")
        return

    table = Table(title="CASTLE Projects")
    table.add_column("#", style="dim")
    table.add_column("Name", style="bold")

    for i, p in enumerate(projects, 1):
        table.add_row(str(i), p)

    console.print(table)
