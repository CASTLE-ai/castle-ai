"""
castle/cli/cluster_cmd.py
Clustering CLI commands: run, export.
"""

import json

import typer
from rich.console import Console
from rich.table import Table

from castle.defaults import BIN_SIZE, DBSCAN_EPS
from castle.service.clustering_service import ClusteringSession
from castle.cli.storage_util import get_storage

console = Console()
app = typer.Typer(no_args_is_help=True)


@app.command("run")
def run(
    project: str = typer.Argument(..., help="Project name"),
    storage: str = typer.Option(None, "--storage", "-s", help="Storage directory (or set CASTLE_STORAGE env var)"),
    roi: int = typer.Option(1, "--roi", help="ROI ID"),
    bin_size: int = typer.Option(BIN_SIZE, "--bin-size", help="Temporal bin size (frames)"),
    model: str = typer.Option("dinov3_vitb16", "--model", "-m", help="Visual model name"),
    umap_config: str = typer.Option(
        '[{"n_neighbors": 100, "min_dist": 0.0, "n_components": 2, "n_epochs": 5000}]',
        "--umap-config",
        help="UMAP config as JSON string",
    ),
    eps: float = typer.Option(DBSCAN_EPS, "--eps", help="DBSCAN epsilon-neighborhood radius"),
):
    """Run full clustering pipeline: UMAP + DBSCAN + auto-label + submit."""
    storage = get_storage(storage)
    console.print("[bold]Initializing clustering session...[/bold]")

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
    storage: str = typer.Option(None, "--storage", "-s", help="Storage directory (or set CASTLE_STORAGE env var)"),
    roi: int = typer.Option(1, "--roi", help="ROI ID"),
    bin_size: int = typer.Option(1, "--bin-size", help="Temporal bin size"),
    model: str = typer.Option("dinov3_vitb16", "--model", "-m", help="Visual model name"),
    format: str = typer.Option("csv", "--format", "-f", help="Export format (csv)"),
):
    """Export clustering results (restore + show info)."""
    storage = get_storage(storage)
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


@app.command("save-model")
def save_model(
    project: str = typer.Argument(..., help="Project name"),
    storage: str = typer.Option(None, "-s", "--storage", help="Storage directory"),
    output: str = typer.Option(None, "-o", "--output", help="Output model path (.npz)"),
    name: str = typer.Option("", "--name", help="Model name"),
    k: int = typer.Option(5, "--k", help="k-NN neighbors"),
):
    """Save clustering model for transfer to new data."""
    import os
    from castle.service.clustering_service import save_project_cluster_model

    storage = get_storage(storage)
    project_path = os.path.join(storage, project)
    if not os.path.isdir(project_path):
        console.print(f"[red]✗[/red] Project directory not found: {project_path}")
        raise typer.Exit(code=1)

    try:
        saved_path = save_project_cluster_model(
            project_path=project_path,
            output_path=output,
            model_name=name,
            k=k,
        )
        console.print(f"[green]✓[/green] Cluster model saved: {saved_path}")
    except Exception as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(code=1)


@app.command("apply-model")
def apply_model(
    project: str = typer.Argument(..., help="Target project with new latent features"),
    model: str = typer.Option(..., "-m", "--model", help="Path to saved model (.npz)"),
    storage: str = typer.Option(None, "-s", "--storage", help="Storage directory"),
    method: str = typer.Option("knn_feature", "--method", help="knn_feature or knn_umap"),
):
    """Apply saved clustering model to new data."""
    import os
    from castle.service.clustering_service import apply_cluster_model_to_project

    storage = get_storage(storage)
    project_path = os.path.join(storage, project)
    if not os.path.isdir(project_path):
        console.print(f"[red]✗[/red] Project directory not found: {project_path}")
        raise typer.Exit(code=1)

    if not os.path.exists(model):
        console.print(f"[red]✗[/red] Model file not found: {model}")
        raise typer.Exit(code=1)

    try:
        result = apply_cluster_model_to_project(
            model_path=model,
            project_path=project_path,
            method=method,
        )
        console.print(f"[green]✓[/green] Applied model to {result['n_frames']} frames")
        console.print(f"  Mean confidence: {result['mean_confidence']:.3f}")
        console.print(f"  Labels CSV: {result['output_csv']}")
        console.print(f"  ID CSV: {result['id_csv']}")

        # Show cluster distribution
        import numpy as np
        labels = result["labels"]
        table = Table(title="Cluster Distribution")
        table.add_column("ID", style="dim")
        table.add_column("Name", style="bold")
        table.add_column("Count", justify="right")
        table.add_column("Fraction", justify="right")
        for cid in sorted(result["cluster_names"].keys()):
            count = int(np.sum(labels == cid))
            frac = count / len(labels) if len(labels) else 0
            table.add_row(
                str(cid),
                result["cluster_names"][cid],
                str(count),
                f"{frac:.1%}",
            )
        console.print(table)

    except Exception as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(code=1)


@app.command("evaluate")
def evaluate(
    project: str = typer.Argument(..., help="Project name"),
    storage: str = typer.Option(None, "-s", "--storage", help="Storage directory (or set CASTLE_STORAGE env var)"),
    ground_truth: str = typer.Option(None, "--gt", help="Ground truth CSV path"),
):
    """Evaluate clustering quality with automated metrics."""
    import os
    from castle.service.metrics_service import evaluate_project_clustering

    storage = get_storage(storage)
    project_path = os.path.join(storage, project)
    if not os.path.isdir(project_path):
        console.print(f"[red]✗[/red] Project directory not found: {project_path}")
        raise typer.Exit(code=1)

    console.print(f"[bold]Evaluating clustering quality for '{project}'...[/bold]")
    result = evaluate_project_clustering(project_path, ground_truth_path=ground_truth)

    if "error" in result:
        console.print(f"[red]✗[/red] {result['error']}")
        raise typer.Exit(code=1)

    # --- Formatted report ---
    table = Table(title="Clustering Quality Report")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Verdict", f"[{'green' if result['verdict'] == 'GOOD' else 'yellow' if result['verdict'] == 'ACCEPTABLE' else 'red'}]{result['verdict']}[/]")
    table.add_row("Temporal Coherence", f"{result['temporal_coherence']:.4f}")
    table.add_row("Single-frame Bout Ratio", f"{result['single_frame_ratio']:.4f}")
    table.add_row("Median Bout Duration (frames)", f"{result['median_bout_duration_frames']:.1f}")
    table.add_row("Bout Duration CV", f"{result['bout_duration_cv']:.4f}")
    table.add_row("Single-frame Bouts", str(result['n_single_frame_bouts']))

    if result.get("silhouette_sample") is not None:
        table.add_row("Silhouette (sampled)", f"{result['silhouette_sample']:.4f}")
    if result.get("calinski_harabasz") is not None:
        table.add_row("Calinski-Harabasz", f"{result['calinski_harabasz']:.2f}")
    if result.get("davies_bouldin") is not None:
        table.add_row("Davies-Bouldin", f"{result['davies_bouldin']:.4f}")

    if result.get("nmi") is not None:
        table.add_row("NMI", f"{result['nmi']:.4f}")
        table.add_row("ARI", f"{result['ari']:.4f}")
        table.add_row("V-measure", f"{result['v_measure']:.4f}")
        table.add_row("Homogeneity", f"{result['homogeneity']:.4f}")
        table.add_row("Completeness", f"{result['completeness']:.4f}")

    console.print(table)

    if result.get("warnings"):
        console.print("\n[yellow bold]Warnings:[/yellow bold]")
        for w in result["warnings"]:
            console.print(f"  [yellow]⚠[/yellow] {w}")

    console.print(f"\n  Frames: {result.get('n_frames', '?')} | Files: {result.get('n_time_series_files', '?')}")
