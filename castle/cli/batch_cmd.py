"""
castle/cli/batch_cmd.py
CLI commands for batch processing multiple CASTLE experiments.

Subcommands:
    castle batch run     experiments.yaml [--parallel] [--max-workers N]
    castle batch status  experiments.yaml
    castle batch report  experiments.yaml --output report.html
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Batch processing across multiple experiments / projects")

# Path used to persist the last batch result alongside the YAML file.
_RESULT_SUFFIX = ".batch_result.json"


def _result_path(yaml_path: str) -> str:
    return os.path.splitext(os.path.abspath(yaml_path))[0] + _RESULT_SUFFIX


# ---------------------------------------------------------------------------
# batch run
# ---------------------------------------------------------------------------


@app.command("run")
def batch_run(
    yaml_file: str = typer.Argument(..., help="Path to experiments.yaml"),
    parallel: Optional[bool] = typer.Option(
        None, "--parallel/--no-parallel",
        help="Run projects in parallel (overrides YAML; default: YAML value)"),
    max_workers: Optional[int] = typer.Option(
        None, "--max-workers", help="Number of parallel workers (overrides YAML)"),
    no_save: bool = typer.Option(False, "--no-save", help="Do not save results to disk"),
) -> None:
    """Process all experiments defined in *yaml_file*."""
    from castle.core.batch import BatchConfig, BatchRunner  # noqa: PLC0415

    if not os.path.isfile(yaml_file):
        typer.echo(f"Error: file not found: {yaml_file}", err=True)
        raise typer.Exit(1)

    try:
        config = BatchConfig.from_yaml(yaml_file)
    except Exception as exc:
        typer.echo(f"Error loading YAML: {exc}", err=True)
        raise typer.Exit(1)

    # Override YAML-level parallel settings only when the flag was actually given;
    # a plain bool/int default cannot signal "unset" and would silently clobber
    # the YAML's parallel/max_workers.
    if parallel is not None:
        config.parallel = parallel
    if max_workers is not None:
        config.max_workers = max_workers

    n = len(config.projects)
    typer.echo(f"🚀 Starting batch run: {n} project(s)  [parallel={config.parallel}]")

    def _cb(frac: float, msg: str) -> None:
        pct = int(frac * 100)
        typer.echo(f"  [{pct:3d}%] {msg}")

    runner = BatchRunner(config)
    results = runner.run(progress_callback=_cb)

    typer.echo("")
    typer.echo(runner.generate_summary(results))

    if not no_save:
        out = _result_path(yaml_file)
        _save_results(results, out)
        typer.echo(f"\n📄 Results saved to {out}")


# ---------------------------------------------------------------------------
# batch status
# ---------------------------------------------------------------------------


@app.command("status")
def batch_status(
    yaml_file: str = typer.Argument(..., help="Path to experiments.yaml"),
) -> None:
    """Show status of the last batch run for *yaml_file*."""
    result_file = _result_path(yaml_file)
    if not os.path.isfile(result_file):
        typer.echo(
            f"No results found for '{yaml_file}'.\n"
            "Run 'castle batch run' first.",
            err=True,
        )
        raise typer.Exit(1)

    results = _load_results(result_file)
    if not results:
        typer.echo("Results file is empty or corrupt.", err=True)
        raise typer.Exit(1)

    typer.echo(f"📊 Batch status — {os.path.basename(yaml_file)}")
    typer.echo(f"   Result file: {result_file}")
    typer.echo("")

    for r in results:
        status = r.get("status", "unknown")
        sym = {"done": "✓", "error": "✗", "skipped": "○"}.get(status, "?")
        elapsed = r.get("elapsed_s", 0)
        typer.echo(f"  [{sym}] {r.get('name', '?')}  status={status}  elapsed={elapsed:.1f}s")
        if r.get("error"):
            typer.echo(f"       ↳ Error: {r['error']}")
        tracking = r.get("tracking", {})
        extraction = r.get("extraction", {})
        if tracking:
            typer.echo(f"       ↳ Tracking:   {len(tracking)} video(s)")
        if extraction:
            ok_e = sum(1 for v in extraction.values() if v)
            typer.echo(f"       ↳ Extraction: {ok_e}/{len(extraction)} video(s) extracted")


# ---------------------------------------------------------------------------
# batch report
# ---------------------------------------------------------------------------


@app.command("report")
def batch_report(
    yaml_file: str = typer.Argument(..., help="Path to experiments.yaml"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output HTML path"),
    include_ethogram: bool = typer.Option(True, help="Include ethogram section"),
    include_quality: bool = typer.Option(True, help="Include quality metrics section"),
    include_comparison: bool = typer.Option(False, help="Include group comparison section"),
) -> None:
    """Generate HTML reports for each experiment in *yaml_file*.

    If *--output* ends with ``.html``, a single combined report is generated
    (first project only when multiple projects exist, with batch summary).
    Otherwise *output* is treated as a directory and one report per project
    is written there.
    """
    from castle.core.batch import BatchConfig  # noqa: PLC0415
    from castle.analysis.report import ReportGenerator  # noqa: PLC0415

    if not os.path.isfile(yaml_file):
        typer.echo(f"Error: file not found: {yaml_file}", err=True)
        raise typer.Exit(1)

    try:
        config = BatchConfig.from_yaml(yaml_file)
    except Exception as exc:
        typer.echo(f"Error loading YAML: {exc}", err=True)
        raise typer.Exit(1)

    # Determine output directory
    if output and output.endswith(".html"):
        out_dir = os.path.dirname(os.path.abspath(output)) or "."
        single_output = output
    else:
        out_dir = output or os.path.splitext(os.path.abspath(yaml_file))[0] + "_reports"
        single_output = None

    os.makedirs(out_dir, exist_ok=True)

    generated: list[str] = []
    for spec in config.projects:
        name = spec.get("name", "unnamed")
        project_path = spec.get("project", "")

        if not os.path.isdir(project_path):
            typer.echo(f"  ⚠  Project path not found, skipping: {project_path}", err=True)
            continue

        if single_output and len(generated) == 0:
            out_path: Optional[str] = single_output
        else:
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
            out_path = os.path.join(out_dir, f"report_{safe_name}.html")

        typer.echo(f"  📝 Generating report for '{name}' → {out_path}")
        try:
            gen = ReportGenerator(project_path)
            written = gen.generate(
                output_path=out_path,
                include_ethogram=include_ethogram,
                include_quality=include_quality,
                include_comparison=include_comparison,
            )
            generated.append(written)
            typer.echo(f"     ✓  Written: {written}")
        except Exception as exc:
            typer.echo(f"     ✗  Failed: {exc}", err=True)

    typer.echo(f"\n✅ {len(generated)}/{len(config.projects)} report(s) generated.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _save_results(results: list[dict], path: str) -> None:
    """Persist batch results as JSON."""

    def _clean(obj: object) -> object:
        """Make result dicts JSON-serialisable."""
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        return obj

    with open(path, "w") as fh:
        json.dump(_clean(results), fh, indent=2)


def _load_results(path: str) -> list[dict]:
    """Load persisted batch results from JSON."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return []
