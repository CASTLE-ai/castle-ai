"""Group comparison CLI commands."""

import os
import typer

from castle.cli.storage_util import get_storage

app = typer.Typer(help="Compare behavioral patterns between groups")


@app.command("run")
def run(
    project_a: str = typer.Option(
        ..., "--project-a", "-a", help="Group A project name"
    ),
    project_b: str = typer.Option(
        ..., "--project-b", "-b", help="Group B project name"
    ),
    storage: str = typer.Option(None, "-s", "--storage", help="Storage directory (or set CASTLE_STORAGE env var)"),
    group_a_name: str = typer.Option("Control", "--name-a", help="Display name for group A (default: Control)."),
    group_b_name: str = typer.Option("Treatment", "--name-b", help="Display name for group B (default: Treatment)."),
    fps: float = typer.Option(30.0, "--fps", help="Frames per second used to convert bout durations to seconds (default: 30)."),
    permutations: int = typer.Option(10000, "--permutations", "-n", help="Number of permutations for the statistical test (default: 10000)."),
    output: str = typer.Option(
        None, "--output", "-o", help="Output directory for report"
    ),
    paired: bool = typer.Option(False, "--paired", help="Use paired/within-subject test"),
):
    """Compare behavioral patterns between two experimental groups."""
    from castle.service.comparison_service import (
        compare_projects,
        compare_projects_paired,
        export_comparison_report,
    )

    storage = get_storage(storage)
    path_a = os.path.join(storage, project_a)
    path_b = os.path.join(storage, project_b)

    mode = "paired" if paired else "independent"
    typer.echo(f"Comparing {project_a} ({group_a_name}) vs {project_b} ({group_b_name})  [{mode}]")
    typer.echo(f"  Permutations: {permutations}  |  FPS: {fps}")

    if paired:
        result = compare_projects_paired(
            path_a,
            path_b,
            group_before_name=group_a_name,
            group_after_name=group_b_name,
            fps=fps,
            n_permutations=permutations,
        )
    else:
        result = compare_projects(
            path_a,
            path_b,
            group_a_name=group_a_name,
            group_b_name=group_b_name,
            fps=fps,
            n_permutations=permutations,
        )

    if result.get("status") != "success":
        typer.echo(f"Error: {result.get('message', 'unknown error')}", err=True)
        raise typer.Exit(1)

    # Print summary
    typer.echo("")
    typer.echo(result.get("summary", ""))

    # Export if output specified
    if output:
        paths = export_comparison_report(result, output)
        typer.echo(f"\nExported {len(paths)} file(s) to {output}")
        for p in paths:
            typer.echo(f"  {p}")


@app.command("fingerprint")
def fingerprint(
    project: str = typer.Argument(..., help="Project name"),
    storage: str = typer.Option(None, "-s", "--storage", help="Storage directory (or set CASTLE_STORAGE env var)"),
    fps: float = typer.Option(30.0, "--fps", help="Frames per second used to convert bout durations to seconds (default: 30)."),
):
    """Compute and display behavioral fingerprint for a project."""
    from castle.service.comparison_service import compute_project_fingerprints

    storage = get_storage(storage)
    path = os.path.join(storage, project)
    result = compute_project_fingerprints(path, group_name=project, fps=fps)

    if result.get("status") != "success":
        typer.echo(f"Error: {result.get('message', 'unknown error')}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Behavioral Fingerprints — {project}")
    typer.echo(f"  Animals (videos): {result['n_animals']}")
    typer.echo("")

    for fp in result["fingerprints"]:
        typer.echo(f"  Animal: {fp['animal_id']}")
        typer.echo(f"    Frames: {fp['n_frames']}  |  FPS: {fp['fps']}")
        cnames = fp["cluster_names"]
        freqs = fp["frequencies"]
        bout_counts = fp["bout_counts"]
        for i, name in enumerate(cnames):
            typer.echo(
                f"    [{name}] freq={freqs[i]:.2%}, bouts={int(bout_counts[i])}, "
                f"mean_dur={fp['mean_bout_durations'][i]:.3f}s, "
                f"cv={fp['cv_bout_durations'][i]:.2f}"
            )
        typer.echo("")
