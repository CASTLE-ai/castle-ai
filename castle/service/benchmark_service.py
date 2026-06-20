"""Reproducible accuracy benchmark for CASTLE clustering.

Wraps the clustering-quality metrics (:func:`evaluate_project_clustering`) into a
self-contained, citable report: accuracy vs. ground truth (NMI / ARI / V-measure
when a labelled CSV is given) plus unsupervised quality, stamped with the run
environment and a dataset DOI so a result is reproducible and attributable.

Dataset-agnostic by design — any project + a ground-truth CSV (a ``behavior``
column aligned to the frames) can be scored. CalMS21 is registered as the
citable public-standard target; see ``docs/technical/benchmarking.md``.
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from castle.service.metrics_service import evaluate_project_clustering

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetCitation:
    """A citable benchmark dataset."""
    key: str
    name: str
    doi: Optional[str] = None
    url: Optional[str] = None
    citation: Optional[str] = None
    note: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


# Public DOI'd datasets usable as the standard accuracy target. CalMS21 is the
# de-facto standard for mouse social behaviour; note it is trajectory-centric, so
# running CASTLE on it requires its raw videos + tracking masks.
KNOWN_DATASETS = {
    "calms21": DatasetCitation(
        key="calms21",
        name="CalMS21",
        doi="10.22002/D1.1991",
        url="https://data.caltech.edu/records/1991",
        citation=(
            "Sun, J. J. et al. (2021). The Multi-Agent Behavior Dataset: Mouse "
            "Dyadic Social Interactions. NeurIPS Datasets and Benchmarks."
        ),
        note=(
            "Trajectory/keypoint-centric; CASTLE is video+mask based, so running "
            "it on CalMS21 requires the dataset's raw videos and a tracking pass."
        ),
    ),
}


def resolve_dataset(
    dataset: Optional[str] = None,
    *,
    name: Optional[str] = None,
    doi: Optional[str] = None,
    url: Optional[str] = None,
) -> Optional[DatasetCitation]:
    """Resolve a dataset citation from a registry key and/or explicit overrides.

    A known key (e.g. ``"calms21"``) seeds the citation; any explicit field
    overrides it. Returns ``None`` if nothing is specified.
    """
    base = KNOWN_DATASETS.get(dataset.lower()) if dataset else None
    if base is None and not any([dataset, name, doi, url]):
        return None
    return DatasetCitation(
        key=(base.key if base else (dataset or "custom")),
        name=name or (base.name if base else (dataset or "custom")),
        doi=doi or (base.doi if base else None),
        url=url or (base.url if base else None),
        citation=(base.citation if base else None),
        note=(base.note if base else None),
    )


def _render_markdown(report: dict) -> str:
    m = report.get("metrics", {})
    ds = report.get("dataset")
    lines = ["# CASTLE accuracy benchmark", ""]
    lines.append(f"- Generated: `{report.get('timestamp', '?')}`")
    lines.append(f"- Project: `{report.get('project_path', '?')}`")
    if ds:
        cite = f"**{ds.get('name', '?')}**"
        if ds.get("doi"):
            cite += f" — DOI [`{ds['doi']}`](https://doi.org/{ds['doi']})"
        lines.append(f"- Dataset: {cite}")
        if ds.get("citation"):
            lines.append(f"  - {ds['citation']}")
        if ds.get("note"):
            lines.append(f"  - _Note:_ {ds['note']}")
    lines.append(f"- Ground truth: {'provided' if report.get('ground_truth_provided') else 'NONE (accuracy metrics need a labelled CSV)'}")
    lines.append("")
    lines.append("## Accuracy vs. ground truth")
    if report.get("ground_truth_provided"):
        for k, label in (("nmi", "NMI"), ("ari", "ARI"), ("v_measure", "V-measure"),
                         ("homogeneity", "Homogeneity"), ("completeness", "Completeness")):
            if m.get(k) is not None:
                lines.append(f"- {label}: **{m[k]:.4f}**")
    else:
        lines.append("- _Not computed — supply `--gt <labels.csv>` (a `behavior` column aligned to the frames)._")
    lines.append("")
    lines.append("## Unsupervised quality")
    for k, label in (("verdict", "Verdict"), ("temporal_coherence", "Temporal coherence"),
                     ("silhouette_sample", "Silhouette (sampled)"),
                     ("calinski_harabasz", "Calinski-Harabasz"), ("davies_bouldin", "Davies-Bouldin"),
                     ("median_bout_duration_frames", "Median bout duration (frames)")):
        if m.get(k) is not None:
            val = m[k]
            lines.append(f"- {label}: {val:.4f}" if isinstance(val, float) else f"- {label}: {val}")
    lines.append(f"- Frames: {m.get('n_frames', '?')} | time-series files: {m.get('n_time_series_files', '?')}")
    lines.append("")
    env = report.get("provenance", {})
    lines.append("## Provenance")
    lines.append(f"- CASTLE {env.get('castle_version', '?')} | device `{env.get('device', '?')}` | "
                 f"python {env.get('python', '?')}")
    lines.append("")
    lines.append("_Reproduce: re-run `castle benchmark` with the same project, ground-truth CSV, "
                 "and `--seed`; the full environment is recorded above._")
    return "\n".join(lines) + "\n"


def run_accuracy_benchmark(
    project_path: str,
    ground_truth_path: Optional[str] = None,
    *,
    dataset: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dataset_doi: Optional[str] = None,
    dataset_url: Optional[str] = None,
    output_dir: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> dict:
    """Run the reproducible accuracy benchmark and write a citable report.

    Returns a JSON-serialisable dict (also written to ``benchmark_report.json``
    + ``benchmark_report.md`` under *output_dir*, default ``<project>/benchmark``).
    On a metrics error the error dict is returned unchanged.
    """
    metrics = evaluate_project_clustering(project_path, ground_truth_path=ground_truth_path)
    if "error" in metrics:
        return metrics

    # Provenance is best-effort — a snapshot failure must not sink the benchmark.
    try:
        from castle.core.environment import collect_run_environment
        provenance = collect_run_environment()
    except Exception:  # noqa: BLE001
        provenance = {}

    cite = resolve_dataset(dataset, name=dataset_name, doi=dataset_doi, url=dataset_url)
    gt_provided = bool(ground_truth_path and os.path.isfile(ground_truth_path))

    report = {
        "benchmark": "castle-accuracy-benchmark",
        "schema_version": 1,
        "timestamp": timestamp or datetime.now().isoformat(),
        "project_path": os.path.abspath(project_path),
        "dataset": cite.to_dict() if cite else None,
        "ground_truth": os.path.abspath(ground_truth_path) if gt_provided else None,
        "ground_truth_provided": gt_provided,
        "metrics": metrics,
        "provenance": provenance,
    }

    out_dir = output_dir or os.path.join(project_path, "benchmark")
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "benchmark_report.json"), "w") as f:
            json.dump(report, f, indent=2, default=str)
        with open(os.path.join(out_dir, "benchmark_report.md"), "w") as f:
            f.write(_render_markdown(report))
        report["report_dir"] = out_dir
    except OSError as e:
        logger.warning("Failed to write benchmark report to %s: %s", out_dir, e)

    return report
