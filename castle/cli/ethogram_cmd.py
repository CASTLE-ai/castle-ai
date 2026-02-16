"""Ethogram analysis CLI commands."""

import os
import typer

from castle.cli.storage_util import get_storage

app = typer.Typer(help="Ethogram analysis (transition matrix, bout statistics)")


@app.command("analyze")
def analyze(
    project: str = typer.Argument(..., help="Project name"),
    storage: str = typer.Option(None, "--storage", "-s", help="Storage directory (or set CASTLE_STORAGE env var)"),
    fps: float = typer.Option(30.0, help="Frames per second"),
    smooth: bool = typer.Option(False, "--smooth", help="Apply temporal smoothing"),
    smooth_window: int = typer.Option(5, "--smooth-window", help="Smoothing window size"),
    min_bout: int = typer.Option(3, "--min-bout", help="Minimum bout duration in frames"),
):
    """Run complete ethogram analysis on a clustered project."""
    from castle.service.ethogram_service import analyze_ethogram

    storage = get_storage(storage)
    project_path = os.path.join(storage, project)
    result = analyze_ethogram(
        project_path, fps=fps,
        smooth=smooth, smooth_window=smooth_window, min_bout_frames=min_bout,
    )

    if result.get("status") != "success":
        typer.echo(f"Error: {result.get('message', 'unknown error')}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Ethogram Analysis — {project}")
    typer.echo(f"  Frames: {result['n_frames']}  |  Clusters: {result['n_clusters']}  |  FPS: {result['fps']}")
    typer.echo(f"  Temporal coherence: {result['temporal_coherence']}")
    typer.echo(f"  Total bouts: {result['n_bouts_total']}")

    tm = result["transition_matrix"]
    typer.echo(f"\nTransition matrix ({tm['n_transitions']} transitions, entropy={tm['entropy']} bits):")
    names = tm["cluster_names"]
    header = "  {:>12s}".format("") + "".join(f"  {n[:10]:>10s}" for n in names)
    typer.echo(header)
    for i, name in enumerate(names):
        row = "  {:>12s}".format(name[:12])
        for j in range(len(names)):
            val = tm["matrix"][i][j]
            row += f"  {val:>10.3f}"
        typer.echo(row)

    typer.echo("\nBout statistics:")
    for cid_str, bs in sorted(result["bout_stats"].items(), key=lambda x: int(x[0])):
        typer.echo(
            f"  [{cid_str}] {bs['cluster_name']}: "
            f"{bs['n_bouts']} bouts, freq={bs['frequency']:.2%}, "
            f"mean={bs['mean_duration_s']:.3f}s, "
            f"median={bs['median_duration_s']:.3f}s, "
            f"cv={bs['cv_duration']:.2f}"
        )


@app.command("transitions")
def transitions(
    project: str = typer.Argument(..., help="Project name"),
    storage: str = typer.Option(None, "--storage", "-s", help="Storage directory (or set CASTLE_STORAGE env var)"),
):
    """Show transition matrix."""
    from castle.service.ethogram_service import get_transition_matrix

    storage = get_storage(storage)
    project_path = os.path.join(storage, project)
    result = get_transition_matrix(project_path)

    if result.get("status") != "success":
        typer.echo(f"Error: {result.get('message', 'unknown error')}", err=True)
        raise typer.Exit(1)

    names = result["cluster_names"]
    typer.echo(f"Transition Matrix ({result['n_transitions']} transitions)")
    typer.echo(f"Entropy: {result['entropy']} bits  |  Stationarity: {result['stationarity']}")
    typer.echo()

    header = "{:>12s}".format("") + "".join(f"  {n[:10]:>10s}" for n in names)
    typer.echo(header)
    for i, name in enumerate(names):
        row = "{:>12s}".format(name[:12])
        for j in range(len(names)):
            val = result["matrix"][i][j]
            row += f"  {val:>10.3f}"
        typer.echo(row)


@app.command("bouts")
def bouts(
    project: str = typer.Argument(..., help="Project name"),
    storage: str = typer.Option(None, "--storage", "-s", help="Storage directory (or set CASTLE_STORAGE env var)"),
    fps: float = typer.Option(30.0, help="Frames per second"),
):
    """Show bout statistics per cluster."""
    from castle.service.ethogram_service import get_bout_statistics

    storage = get_storage(storage)
    project_path = os.path.join(storage, project)
    result = get_bout_statistics(project_path, fps=fps)

    if result.get("status") != "success":
        typer.echo(f"Error: {result.get('message', 'unknown error')}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Bout Statistics (total bouts: {result['n_bouts_total']})\n")
    for cid_str, bs in sorted(result["bout_stats"].items(), key=lambda x: int(x[0])):
        typer.echo(f"  Cluster {cid_str} — {bs['cluster_name']}")
        typer.echo(f"    Bouts: {bs['n_bouts']}  |  Frequency: {bs['frequency']:.2%}")
        typer.echo(
            f"    Duration: mean={bs['mean_duration_s']:.3f}s  "
            f"median={bs['median_duration_s']:.3f}s  "
            f"std={bs['std_duration_s']:.3f}s  "
            f"CV={bs['cv_duration']:.2f}"
        )
        typer.echo(
            f"    Range: [{bs['min_duration_s']:.3f}s, {bs['max_duration_s']:.3f}s]  "
            f"IBI={bs['mean_inter_bout_interval_s']:.3f}s"
        )
        typer.echo()


@app.command("export")
def export(
    project: str = typer.Argument(..., help="Project name"),
    storage: str = typer.Option(None, "--storage", "-s", help="Storage directory (or set CASTLE_STORAGE env var)"),
    output: str = typer.Option("./ethogram_export", help="Output directory"),
):
    """Export ethogram data to CSV files."""
    from castle.service.ethogram_service import export_ethogram_csv

    storage = get_storage(storage)
    project_path = os.path.join(storage, project)
    out_dir = export_ethogram_csv(project_path, output)
    typer.echo(f"Exported ethogram CSV files to {out_dir}")


@app.command("export-nwb")
def export_nwb(
    project: str = typer.Argument(..., help="Project name"),
    storage: str = typer.Option(None, "--storage", "-s", help="Storage directory path"),
    output: str = typer.Option(None, "--output", "-o", help="Output .nwb file path"),
    description: str = typer.Option(
        "CASTLE behavioral analysis", "--description", help="NWB session description"
    ),
):
    """Export analysis results to NWB (Neurodata Without Borders) format."""
    try:
        from castle.core.nwb_export import _require_pynwb
        _require_pynwb()
    except ImportError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    from castle.service.nwb_service import export_project_nwb

    project_path = os.path.join(storage, project) if storage else project

    try:
        nwb_path = export_project_nwb(
            project_path,
            output_path=output,
            session_description=description,
        )
        typer.echo(f"Exported NWB file to {nwb_path}")
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
