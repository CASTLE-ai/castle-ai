"""
castle/cli/preprocess_cmd.py
Stabilized camera preprocessing CLI command.
"""

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from castle.cli.storage_util import get_storage

console = Console()
app = typer.Typer()


@app.command("preprocess")
def preprocess(
    project: str = typer.Argument(..., help="Project name"),
    video: str = typer.Option(..., "--video", "-v", help="Video filename (e.g. animal.mp4)"),
    body_roi: int = typer.Option(..., "--body-roi", help="ROI id for the body"),
    head_roi: int = typer.Option(..., "--head-roi", help="ROI id for the head"),
    storage: str = typer.Option(
        None, "--storage", "-s", help="Storage directory (or set CASTLE_STORAGE env var)"
    ),
    fc: float = typer.Option(0.25, "--fc", help="Low-pass cutoff frequency in Hz"),
    order: int = typer.Option(2, "--order", help="Butterworth filter order"),
    margin: int = typer.Option(75, "--margin", help="Spatial margin around HP residual (px)"),
    min_crop: int = typer.Option(300, "--min-crop", help="Minimum crop side length (px)"),
    output_size: int = typer.Option(
        592,
        "--output-size",
        help="Output frame side length (px). 592 (=37x16) for the default DINOv3 ViT-B/16; 518 (=37x14) for DINOv2 ViT-B/14",
    ),
    preview_duration: float = typer.Option(
        10.0, "--preview-duration", help="Preview clip duration in seconds"
    ),
    mode: str = typer.Option(
        "stabilized-camera",
        "--mode",
        help="Preprocessing mode (currently only 'stabilized-camera' is supported)",
    ),
) -> None:
    """Preprocess a tracked video using the stabilized virtual camera.

    Applies zero-phase Butterworth low-pass filtering to ROI centroid
    trajectories and orientation angles, then extracts dynamically-cropped
    and rotated frames saved as H.264 MP4 under
    ``{storage}/{project}/preprocessed/{video}/stabilized.mp4``.

    Example
    -------
    castle preprocess my_project \\
        --video animal.mp4 --body-roi 1 --head-roi 2 \\
        --fc 0.25 --margin 75 --output-size 592
    """
    if mode != "stabilized-camera":
        console.print(f"[red]Unknown mode: {mode}. Only 'stabilized-camera' is supported.[/red]")
        raise typer.Exit(code=1)

    storage = get_storage(storage)

    console.print(
        f"[bold]Preprocessing[/bold] [cyan]{video}[/cyan] in project [cyan]{project}[/cyan]"
    )
    console.print(
        f"  body_roi={body_roi}, head_roi={head_roi}, fc={fc} Hz, "
        f"order={order}, margin={margin} px, min_crop={min_crop} px, "
        f"output_size={output_size} px"
    )

    from castle.service.preprocessing_service import preprocess_stabilized_camera

    with Progress(
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Initialising…", total=100)

        def _cb(fraction: float, description: str = "") -> None:
            pct = int(fraction * 100)
            progress.update(task, completed=pct, description=description or "Processing…")

        try:
            result = preprocess_stabilized_camera(
                storage_path=storage,
                project_name=project,
                video_name=video,
                body_roi_id=body_roi,
                head_roi_id=head_roi,
                fc=fc,
                order=order,
                margin=margin,
                min_crop=min_crop,
                output_size=output_size,
                preview_duration=preview_duration,
                progress_callback=_cb,
            )
        except FileNotFoundError as exc:
            console.print(f"[red]✗ File not found: {exc}[/red]")
            raise typer.Exit(code=1) from exc
        except Exception as exc:
            console.print(f"[red]✗ Preprocessing failed: {exc}[/red]")
            raise typer.Exit(code=1) from exc

    diag = result["diagnostics"]
    console.print("\n[green]✓ Preprocessing complete.[/green]")
    console.print(f"  Output video : {result['preprocessed_video_path']}")
    console.print(f"  Preview clip : {result['preview_path']}")
    console.print(f"  Frames       : {result['n_frames']}")
    console.print(
        f"  HP residual RMS : {diag['hp_residual_rms']:.2f} px  |  "
        f"% at min_crop : {diag['pct_at_min_crop']:.1f}%  |  "
        f"speed-crop r : {diag['speed_crop_correlation']:.3f}"
    )
