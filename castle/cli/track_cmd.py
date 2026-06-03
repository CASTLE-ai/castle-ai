"""
castle/cli/track_cmd.py
Tracking CLI command.
"""

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from castle.service.project_service import get_project_info
from castle.service.tracking_service import track_videos
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

    # track_videos handles skip-existing and spreads whole videos across GPUs
    # when CASTLE_MULTI_GPU is set (>1 CUDA device); otherwise it runs sequentially.
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Tracking…", total=None)

        def _cb(frac: float, desc: str) -> None:
            progress.update(task, description=desc)

        results = track_videos(
            storage, project, videos,
            model=model, start=start, stop=stop,
            skip_existing=skip_existing,
            progress_callback=_cb,
        )

    for video_name in videos:
        result = results.get(video_name, '?')
        if result == 'Done':
            console.print(f"[green]✓ {video_name}[/green]")
        elif result in ('Skipped', 'Skip'):
            console.print(f"[dim]Skipped {video_name}[/dim]")
        elif result == 'Cancel':
            console.print(f"[yellow]Cancelled {video_name}[/yellow]")
        else:
            console.print(f"[red]✗ {video_name}: {result}[/red]")

    console.print("[green]Tracking complete.[/green]")
