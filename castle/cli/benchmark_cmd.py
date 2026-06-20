"""
castle/cli/benchmark_cmd.py
Reproducible accuracy benchmark CLI: `castle benchmark`.
"""

import os

import typer
from rich.console import Console
from rich.table import Table

from castle.cli.storage_util import get_storage

console = Console()
app = typer.Typer(no_args_is_help=True)


@app.command("run")
def run(
    project: str = typer.Argument(..., help="Project name (must already be clustered)"),
    storage: str = typer.Option(None, "--storage", "-s", help="Storage directory (or CASTLE_STORAGE)"),
    gt: str = typer.Option(
        None, "--gt",
        help="Ground-truth CSV with a 'behavior' column aligned to the frames. "
             "Required for accuracy metrics (NMI/ARI/V-measure).",
    ),
    dataset: str = typer.Option(
        None, "--dataset",
        help="Known dataset key for citation (e.g. 'calms21'), or a free-text name.",
    ),
    doi: str = typer.Option(None, "--doi", help="Dataset DOI (overrides the known-dataset DOI)."),
    url: str = typer.Option(None, "--url", help="Dataset URL (overrides the known-dataset URL)."),
    output: str = typer.Option(None, "-o", "--output", help="Report output dir (default <project>/benchmark)."),
):
    """Score a clustered project against ground truth and write a citable report.

    Dataset-agnostic: pass any project + a labelled CSV via ``--gt``. Use
    ``--dataset calms21`` (or ``--doi``) to attribute the result to a public
    DOI'd dataset. Writes ``benchmark_report.json`` + ``.md``.
    """
    from castle.service.benchmark_service import run_accuracy_benchmark

    storage = get_storage(storage)
    project_path = os.path.join(storage, project)
    if not os.path.isdir(project_path):
        console.print(f"[red]✗[/red] Project directory not found: {project_path}")
        raise typer.Exit(code=1)
    if gt and not os.path.isfile(gt):
        console.print(f"[red]✗[/red] Ground-truth CSV not found: {gt}")
        raise typer.Exit(code=1)

    console.print(f"[bold]Benchmarking '{project}'…[/bold]")
    report = run_accuracy_benchmark(
        project_path, ground_truth_path=gt,
        dataset=dataset, dataset_doi=doi, dataset_url=url, output_dir=output,
    )
    if "error" in report:
        console.print(f"[red]✗[/red] {report['error']}")
        raise typer.Exit(code=1)

    m = report["metrics"]
    table = Table(title=f"Accuracy benchmark — {project}")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    if report["ground_truth_provided"]:
        for k, label in (("nmi", "NMI"), ("ari", "ARI"), ("v_measure", "V-measure"),
                         ("homogeneity", "Homogeneity"), ("completeness", "Completeness")):
            if m.get(k) is not None:
                table.add_row(label, f"{m[k]:.4f}")
    else:
        table.add_row("Accuracy", "[yellow]needs --gt[/yellow]")
    if m.get("temporal_coherence") is not None:
        table.add_row("Temporal coherence", f"{m['temporal_coherence']:.4f}")
    if m.get("silhouette_sample") is not None:
        table.add_row("Silhouette (sampled)", f"{m['silhouette_sample']:.4f}")
    table.add_row("Frames", str(m.get("n_frames", "?")))
    console.print(table)

    ds = report.get("dataset")
    if ds and ds.get("doi"):
        console.print(f"[dim]Dataset: {ds.get('name')} — DOI {ds['doi']}[/dim]")
    if not report["ground_truth_provided"]:
        console.print(
            "[yellow]No ground truth given — accuracy metrics (NMI/ARI) were not "
            "computed. Pass --gt <labels.csv> for the accuracy benchmark.[/yellow]"
        )
    if report.get("report_dir"):
        console.print(f"[green]✓[/green] Report: {os.path.join(report['report_dir'], 'benchmark_report.md')}")


@app.command("datasets")
def datasets():
    """List the registered citable benchmark datasets."""
    from castle.service.benchmark_service import KNOWN_DATASETS

    table = Table(title="Known benchmark datasets")
    table.add_column("Key", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("DOI")
    for key, ds in KNOWN_DATASETS.items():
        table.add_row(key, ds.name, ds.doi or "-")
    console.print(table)
    console.print("[dim]Use any of these with `castle benchmark run <project> --dataset <key> --gt <csv>`.[/dim]")
