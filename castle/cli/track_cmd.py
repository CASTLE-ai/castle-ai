"""
castle/cli/track_cmd.py
Tracking CLI command.
"""

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from castle.service.project_service import get_project_info
from castle.service.tracking_service import track_video, get_tracking_status
from castle.cli.storage_util import get_storage

console = Console()
app = typer.Typer()


@app.command("track")
def track(
    project: str = typer.Argument(..., help="Project name"),
    storage: str = typer.Option(None, "--storage", "-s", help="Storage directory (or set CASTLE_STORAGE env var)"),
    model: str = typer.Option("r50_deaotl", "--model", "-m", help="Tracking model (r50_deaotl or swinb_deaotl)"),
    skip_existing: bool = typer.Option(True, "--skip-existing/--no-skip-existing", help="Skip already tracked videos"),
    start: int = typer.Option(0, "--start", help="Start frame"),
    stop: int = typer.Option(-1, "--stop", help="Stop frame (-1 for end)"),
):
    """Run ROI tracking on all project videos."""
    storage = get_storage(storage)
    info = get_project_info(storage, project)
    if 'error' in info:
        console.print(f"[red]✗[/red] {info['error']}")
        raise typer.Exit(code=1)

    videos = info['videos']
    if not videos:
        console.print("[yellow]No videos in project.[/yellow]")
        return

    console.print(f"Tracking {len(videos)} videos with model [bold]{model}[/bold]...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for video_name in videos:
            task = progress.add_task(f"Tracking {video_name}...", total=None)

            status = get_tracking_status(storage, project, video_name)
            if skip_existing and status['tracked']:
                progress.update(task, description=f"[dim]Skipped {video_name} (already tracked)[/dim]")
                progress.stop_task(task)
                continue

            result = track_video(
                storage, project, video_name,
                model=model, start=start, stop=stop,
                skip_existing=skip_existing,
            )

            if result == 'Done':
                progress.update(task, description=f"[green]✓ {video_name}[/green]")
            elif result == 'Skipped':
                progress.update(task, description=f"[dim]Skipped {video_name}[/dim]")
            else:
                progress.update(task, description=f"[red]✗ {video_name}: {result}[/red]")
            progress.stop_task(task)

    console.print("[green]Tracking complete.[/green]")
