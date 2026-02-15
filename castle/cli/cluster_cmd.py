"""
castle/cli/cluster_cmd.py
Clustering CLI commands: run, export.
"""

import json

import typer
from rich.console import Console
from rich.table import Table

from castle.service.clustering_service import ClusteringSession

console = Console()
app = typer.Typer(no_args_is_help=True)


@app.command("run")
def run(
    project: str = typer.Argument(..., help="Project name"),
    storage: str = typer.Option(..., "--storage", "-s", help="Storage directory path"),
    roi: int = typer.Option(1, "--roi", help="ROI ID"),
    bin_size: int = typer.Option(1, "--bin-size", help="Temporal bin size (frames)"),
    model: str = typer.Option("dinov3_vitb16", "--model", "-m", help="Visual model name"),
    umap_config: str = typer.Option(
        '[{"n_neighbors": 100, "min_dist": 0.0, "n_components": 2, "n_epochs": 5000}]',
        "--umap-config",
        help="UMAP config as JSON string",
    ),
    eps: float = typer.Option(1.0, "--eps", help="DBSCAN epsilon-neighborhood radius"),
):
    """Run full clustering pipeline: UMAP + DBSCAN + auto-label + submit."""
    console.print(f"[bold]Initializing clustering session...[/bold]")

    def notify(msg, level="info"):
        if level == "error":
            console.print(f"[red]{msg}[/red]")
        else:
            console.print(f"  {msg}")

    session = ClusteringSession(
        storage_path=storage,
        project_name=project,
        roi=roi,
        bin_size=bin_size,
        model=model,
        notify=notify,
    )

    # Parse UMAP config
    try:
        cfg = json.loads(umap_config)
    except json.JSONDecodeError as e:
        console.print(f"[red]✗[/red] Invalid UMAP config JSON: {e}")
        raise typer.Exit(code=1)

    # Run UMAP on 'init' cluster (all data)
    console.print("[bold]Running UMAP...[/bold]")
    umap_result = session.run_umap("init", cfg)
    if not umap_result['success']:
        console.print(f"[red]✗[/red] UMAP failed: {umap_result.get('error', 'unknown')}")
        raise typer.Exit(code=1)
    console.print(
        f"  Embedding: {umap_result['n_points']} points → {umap_result['embedding_shape']}"
    )

    # Run DBSCAN
    console.print(f"[bold]Running DBSCAN (eps={eps})...[/bold]")
    dbscan_result = session.run_dbscan(eps)
    if not dbscan_result['success']:
        console.print(f"[red]✗[/red] DBSCAN failed: {dbscan_result.get('error', 'unknown')}")
        raise typer.Exit(code=1)
    console.print(
        f"  Found {dbscan_result['n_clusters']} clusters, "
        f"{dbscan_result['noise_count']} noise points"
    )

    # Auto-label all
    count = session.auto_label_all("root")
    console.print(f"  Auto-labeled {count} clusters")

    # Submit
    console.print("[bold]Submitting results...[/bold]")
    submit_result = session.submit()
    if not submit_result['success']:
        console.print(f"[red]✗[/red] Submit failed: {submit_result.get('error', 'unknown')}")
        raise typer.Exit(code=1)

    console.print(f"[green]✓[/green] ID CSV: {submit_result['id_csv_path']}")
    for p in submit_result['time_series_paths']:
        console.print(f"[green]✓[/green] Time series: {p}")
    for p in submit_result.get('srt_paths', []):
        console.print(f"[green]✓[/green] Subtitle: {p}")
    console.print(f"[green]✓[/green] Embedding: {submit_result['embedding_path']}")
    console.print("\n[green bold]Clustering complete![/green bold]")


@app.command("export")
def export(
    project: str = typer.Argument(..., help="Project name"),
    storage: str = typer.Option(..., "--storage", "-s", help="Storage directory path"),
    roi: int = typer.Option(1, "--roi", help="ROI ID"),
    bin_size: int = typer.Option(1, "--bin-size", help="Temporal bin size"),
    model: str = typer.Option("dinov3_vitb16", "--model", "-m", help="Visual model name"),
    format: str = typer.Option("csv", "--format", "-f", help="Export format (csv)"),
):
    """Export clustering results (restore + show info)."""
    def notify(msg, level="info"):
        if level == "error":
            console.print(f"[red]{msg}[/red]")
        else:
            console.print(f"  {msg}")

    session = ClusteringSession(
        storage_path=storage,
        project_name=project,
        roi=roi,
        bin_size=bin_size,
        model=model,
        notify=notify,
    )

    result = session.restore()
    if not result['success']:
        console.print(f"[red]✗[/red] {result.get('error', 'No session to export')}")
        raise typer.Exit(code=1)

    console.print(f"[green]✓[/green] Restored {result['cluster_count']} clusters")

    # Display cluster info table
    table = Table(title="Cluster Summary")
    table.add_column("ID", style="dim")
    table.add_column("Name", style="bold")
    table.add_column("Color")

    for cid, meta in sorted(session.latents.cluster_meta.items()):
        table.add_row(str(cid), meta['name'], meta['color'])

    console.print(table)

    console.print(f"\nID CSV: {result['id_csv_path']}")
    for p in result['time_series_paths']:
        console.print(f"Time series: {p}")
