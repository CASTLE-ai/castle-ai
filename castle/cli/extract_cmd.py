"""
castle/cli/extract_cmd.py
Latent extraction CLI command.
"""

from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from castle.service.project_service import get_project_info
from castle.service.extraction_service import extract_latent, make_preprocess_config, latent_gap_summary
from castle.cli.storage_util import get_storage

console = Console()
app = typer.Typer()


@app.command("extract")
def extract(
    project: str = typer.Argument(..., help="Project name"),
    storage: str = typer.Option(None, "--storage", "-s", help="Storage directory (or set CASTLE_STORAGE env var)"),
    model: str = typer.Option("dinov3_vitb16", "--model", "-m", help="Visual model name"),
    roi: int = typer.Option(1, "--roi", help="ROI ID to extract"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Batch size for extraction (default: auto-sized from free VRAM)"),
    skip_existing: bool = typer.Option(True, "--skip-existing/--no-skip-existing", help="Skip already extracted"),
    pooling: str = typer.Option("weighted_average", "--pooling", "-p", help="Pooling method: weighted_average or multiscale"),
    scales: str = typer.Option("", "--scales", help="Comma-separated scales for multiscale pooling, e.g. '1,2,4'"),
    layers: str = typer.Option("", "--layers", help="Comma-separated layer indices for multi-layer extraction, e.g. '3,7,11'"),
    latent_dtype: str = typer.Option("float32", "--latent-dtype", help="Latent storage dtype: float32 (default) or float16 (half the file size / I/O — useful on network storage)"),
):
    """Extract latent features from tracked videos."""
    storage = get_storage(storage)
    info = get_project_info(storage, project)
    if 'error' in info:
        console.print(f"[red]✗[/red] {info['error']}")
        raise typer.Exit(code=1)

    videos = info['videos']
    if not videos:
        console.print("[yellow]No videos in project.[/yellow]")
        return

    # A-06: Parse advanced extraction options
    def _parse_int_csv(raw: str, flag: str):
        if not raw:
            return None
        try:
            return [int(tok.strip()) for tok in raw.split(',') if tok.strip()]
        except ValueError as exc:
            raise typer.BadParameter(
                f"{flag} expects a comma-separated list of integers (e.g. '1,2,4'); got {raw!r}."
            ) from exc

    parsed_scales = _parse_int_csv(scales, "--scales")
    parsed_layers = _parse_int_csv(layers, "--layers")

    if latent_dtype not in ("float32", "float16"):
        console.print(f"[red]✗[/red] --latent-dtype must be float32 or float16, got '{latent_dtype}'")
        raise typer.Exit(code=1)

    extra_info = ""
    if pooling == 'multiscale':
        extra_info += f", pooling=multiscale(scales={parsed_scales or [1,2,4]})"
    if parsed_layers:
        extra_info += f", layers={parsed_layers}"

    # Pre-flight memory check, only when the user pinned an explicit batch size.
    # When --batch-size is omitted the service auto-sizes it from free VRAM, so
    # there is nothing to warn about. (The CLI does not enable the rotation path,
    # so rotate=False here is correct; the service uses the real flag.)
    if batch_size is not None:
        try:
            import torch
            from castle.core.memory_guard import check as _mem_check, suggest_batch_size as _suggest_bs
            _device = "cuda" if torch.cuda.is_available() else "cpu"
            _risky, _warn = _mem_check(model, batch_size, _device, rotate=False)
            if _risky:
                console.print(f"[yellow]{_warn}[/yellow]")
                console.print(f"[yellow]Tip: re-run with --batch-size {_suggest_bs(model, _device, rotate=False)} to stay within safe limits.[/yellow]")
        except Exception:
            pass  # memory check is advisory; never block extraction

    _bs_display = batch_size if batch_size is not None else "auto"
    console.print(
        f"Extracting latents for {len(videos)} videos "
        f"with model [bold]{model}[/bold], ROI={roi}, batch_size={_bs_display}{extra_info}..."
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
                pooling_method=pooling,
                pooling_scales=parsed_scales,
                feature_layers=parsed_layers,
                latent_dtype=latent_dtype,
            )

            if result:
                progress.update(task, description=f"[green]✓ {video_name}[/green]")
            else:
                progress.update(task, description=f"[red]✗ {video_name}: no output[/red]")
            progress.stop_task(task)

            # Surface frames the tracker never tracked (stored as NaN gaps).
            for p in (result or "").split(';'):
                summary = latent_gap_summary(p)
                if summary and summary["n_skipped"]:
                    console.print(
                        f"[yellow]⚠ {video_name}: {summary['n_skipped']}/{summary['n_total']} "
                        f"frame(s) ({summary['frac']:.1%}) had no tracked mask — skipped "
                        f"(stored as gaps, ignored by clustering).[/yellow]"
                    )

    console.print("[green]Extraction complete.[/green]")
