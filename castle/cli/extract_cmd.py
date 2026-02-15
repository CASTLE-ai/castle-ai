"""
castle/cli/extract_cmd.py
Latent extraction CLI command.
"""

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from castle.service.project_service import get_project_info
from castle.service.extraction_service import extract_latent, make_preprocess_config

console = Console()
app = typer.Typer()


@app.command("extract")
def extract(
    project: str = typer.Argument(..., help="Project name"),
    storage: str = typer.Option(..., "--storage", "-s", help="Storage directory path"),
    model: str = typer.Option("dinov3_vitb16", "--model", "-m", help="Visual model name"),
    roi: int = typer.Option(1, "--roi", help="ROI ID to extract"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size for extraction"),
    skip_existing: bool = typer.Option(True, "--skip-existing/--no-skip-existing", help="Skip already extracted"),
):
    """Extract latent features from tracked videos."""
    info = get_project_info(storage, project)
    if 'error' in info:
        console.print(f"[red]✗[/red] {info['error']}")
        raise typer.Exit(code=1)

    videos = info['videos']
    if not videos:
        console.print("[yellow]No videos in project.[/yellow]")
        return

    console.print(
        f"Extracting latents for {len(videos)} videos "
        f"with model [bold]{model}[/bold], ROI={roi}, batch_size={batch_size}..."
    )

    preprocess_config = make_preprocess_config()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for video_name in videos:
            task = progress.add_task(f"Extracting {video_name}...", total=None)

            result = extract_latent(
                storage_path=storage,
                project_name=project,
                video_name=video_name,
                model=model,
                roi=roi,
                batch_size=batch_size,
                preprocess_config=preprocess_config,
                skip_existing=skip_existing,
            )

            if result:
                progress.update(task, description=f"[green]✓ {video_name}[/green]")
            else:
                progress.update(task, description=f"[red]✗ {video_name}: no output[/red]")
            progress.stop_task(task)

    console.print("[green]Extraction complete.[/green]")
