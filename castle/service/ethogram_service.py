"""Service layer for ethogram analysis.

Loads cluster data from CASTLE projects and delegates computation to
:mod:`castle.core.ethogram`.
"""

import os
import csv
import logging
import numpy as np

logger = logging.getLogger(__name__)


def _load_cluster_data(project_path: str) -> dict:
    """Load cluster labels and metadata from a project directory.

    Reads ``cluster/id.csv`` for cluster metadata and all
    ``cluster/time_series_*.csv`` for per-frame cluster assignments.

    Returns:
        dict with keys ``labels`` (np.ndarray), ``cluster_names`` (dict),
        ``fps`` (float or None), ``cluster_meta`` (dict).
    """
    cluster_dir = os.path.join(project_path, "cluster")
    if not os.path.isdir(cluster_dir):
        raise FileNotFoundError(
            f"No cluster directory found at {cluster_dir}. "
            "Run clustering first."
        )

    # --- cluster metadata from id.csv ---
    id_csv = os.path.join(cluster_dir, "id.csv")
    cluster_names: dict = {}
    cluster_meta: dict = {}
    if os.path.exists(id_csv):
        with open(id_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = int(row["Id"])
                name = row.get("Name", f"cluster_{cid}")
                color = row.get("Color", "grey")
                cluster_names[cid] = name
                cluster_meta[cid] = {"name": name, "color": color}

    # --- per-frame labels from time_series CSVs ---
    ts_files = sorted(
        f for f in os.listdir(cluster_dir) if f.startswith("time_series_") and f.endswith(".csv")
    )
    if not ts_files:
        raise FileNotFoundError(
            f"No time_series_*.csv files found in {cluster_dir}. "
            "Run clustering and submit first."
        )

    all_labels = []
    for ts_file in ts_files:
        ts_path = os.path.join(cluster_dir, ts_file)
        with open(ts_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_labels.append(int(row["behavior"]))

    labels = np.array(all_labels, dtype=np.int32)

    return {
        "labels": labels,
        "cluster_names": cluster_names,
        "cluster_meta": cluster_meta,
        "fps": None,  # caller should supply; we don't have it in CSV
    }


def _resolve_project_path(project_path: str) -> str:
    """If *project_path* looks like ``storage/project_name``, return as-is."""
    return os.path.abspath(project_path)


def _ethogram_to_dict(ethogram) -> dict:
    """Serialise an :class:`Ethogram` to a JSON-safe dict."""
    tm = ethogram.transition_matrix
    return {
        "n_frames": ethogram.n_frames,
        "fps": ethogram.fps,
        "n_clusters": ethogram.n_clusters,
        "cluster_names": ethogram.cluster_names,
        "temporal_coherence": round(ethogram.temporal_coherence, 4),
        "transition_matrix": {
            "matrix": tm.matrix.tolist(),
            "counts": tm.counts.tolist(),
            "cluster_ids": tm.cluster_ids,
            "cluster_names": tm.cluster_names,
            "n_transitions": tm.n_transitions,
            "entropy": round(tm.entropy, 4),
            "stationarity": round(tm.stationarity, 4),
        },
        "bout_stats": {
            str(cid): {
                "cluster_name": bs.cluster_name,
                "n_bouts": bs.n_bouts,
                "total_frames": bs.total_frames,
                "frequency": round(bs.frequency, 4),
                "mean_duration_s": round(bs.mean_duration_s, 4),
                "median_duration_s": round(bs.median_duration_s, 4),
                "std_duration_s": round(bs.std_duration_s, 4),
                "cv_duration": round(bs.cv_duration, 4),
                "min_duration_s": round(bs.min_duration_s, 4),
                "max_duration_s": round(bs.max_duration_s, 4),
                "mean_inter_bout_interval_s": round(bs.mean_inter_bout_interval_s, 4),
            }
            for cid, bs in ethogram.bout_stats.items()
        },
        "n_bouts_total": len(ethogram.bouts),
    }


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def analyze_ethogram(
    project_path: str,
    fps: float = None,
    smooth: bool = False,
    smooth_window: int = 5,
    min_bout_frames: int = 3,
) -> dict:
    """Run complete ethogram analysis on a clustered project.

    Args:
        project_path: Path to the project directory (``storage/project_name``).
        fps: Frames per second. If *None*, defaults to 30.0.
        smooth: If *True*, apply temporal smoothing before computing the
            ethogram.
        smooth_window: Window size for median smoothing (odd integer).
        min_bout_frames: Minimum bout duration for the bout filter.

    Returns:
        Structured dict suitable for JSON serialisation.
    """
    from castle.core.ethogram import compute_ethogram

    project_path = _resolve_project_path(project_path)
    data = _load_cluster_data(project_path)
    effective_fps = fps or data["fps"] or 30.0

    labels = data["labels"]

    if smooth:
        from castle.core.temporal_smooth import smooth_labels
        labels = smooth_labels(
            labels, method="both",
            window=smooth_window, min_bout_frames=min_bout_frames,
        )

    ethogram = compute_ethogram(
        labels,
        fps=effective_fps,
        cluster_names=data["cluster_names"],
    )
    result = _ethogram_to_dict(ethogram)
    result["status"] = "success"
    result["project_path"] = project_path
    if smooth:
        result["smoothing"] = {
            "applied": True,
            "window": smooth_window,
            "min_bout_frames": min_bout_frames,
        }
    return result


def get_transition_matrix(project_path: str) -> dict:
    """Get transition matrix for a project."""
    from castle.core.ethogram import compute_transition_matrix

    project_path = _resolve_project_path(project_path)
    data = _load_cluster_data(project_path)
    tm = compute_transition_matrix(data["labels"], data["cluster_names"])
    return {
        "status": "success",
        "matrix": tm.matrix.tolist(),
        "counts": tm.counts.tolist(),
        "cluster_ids": tm.cluster_ids,
        "cluster_names": tm.cluster_names,
        "n_transitions": tm.n_transitions,
        "entropy": round(tm.entropy, 4),
        "stationarity": round(tm.stationarity, 4),
    }


def get_bout_statistics(project_path: str, fps: float = None) -> dict:
    """Get per-cluster bout statistics."""
    from castle.core.ethogram import extract_bouts, compute_bout_statistics

    project_path = _resolve_project_path(project_path)
    data = _load_cluster_data(project_path)
    effective_fps = fps or data["fps"] or 30.0

    bouts = extract_bouts(data["labels"], effective_fps)
    stats = compute_bout_statistics(
        bouts, data["labels"], effective_fps, data["cluster_names"]
    )

    return {
        "status": "success",
        "bout_stats": {
            str(cid): {
                "cluster_name": bs.cluster_name,
                "n_bouts": bs.n_bouts,
                "total_frames": bs.total_frames,
                "frequency": round(bs.frequency, 4),
                "mean_duration_s": round(bs.mean_duration_s, 4),
                "median_duration_s": round(bs.median_duration_s, 4),
                "std_duration_s": round(bs.std_duration_s, 4),
                "cv_duration": round(bs.cv_duration, 4),
                "min_duration_s": round(bs.min_duration_s, 4),
                "max_duration_s": round(bs.max_duration_s, 4),
                "mean_inter_bout_interval_s": round(bs.mean_inter_bout_interval_s, 4),
            }
            for cid, bs in stats.items()
        },
        "n_bouts_total": len(bouts),
    }


def compute_ethogram_from_data(labels, fps: float, cluster_names: dict = None):
    """Compute an :class:`~castle.core.ethogram.Ethogram` from pre-loaded data.

    This is the service-layer entry point for frontends that have already
    loaded cluster labels into memory (e.g. from
    :func:`castle.service.annotator_loader.load_annotator_data`).

    Args:
        labels: 1-D array-like of integer cluster assignments (one per bin).
        fps: Frames per second for duration calculations.
        cluster_names: Optional ``{cluster_id: name}`` mapping.

    Returns:
        :class:`~castle.core.ethogram.Ethogram` instance.
    """
    from castle.core.ethogram import compute_ethogram

    return compute_ethogram(labels, fps=fps, cluster_names=cluster_names or {})


def export_ethogram_csv(project_path: str, output_path: str) -> str:
    """Export ethogram data to CSV files.

    Creates:
      - ``bout_stats.csv``  — per-cluster summary statistics
      - ``transition_matrix.csv`` — transition probability matrix
      - ``transition_counts.csv`` — raw transition counts
      - ``bouts.csv`` — every individual bout

    Args:
        project_path: Project directory path.
        output_path: Directory to write CSV files into.

    Returns:
        Path to the output directory.
    """
    from castle.core.ethogram import compute_ethogram

    project_path = _resolve_project_path(project_path)
    data = _load_cluster_data(project_path)
    ethogram = compute_ethogram(data["labels"], fps=30.0, cluster_names=data["cluster_names"])

    os.makedirs(output_path, exist_ok=True)

    # --- bout_stats.csv ---
    stats_path = os.path.join(output_path, "bout_stats.csv")
    fields = [
        "cluster_id", "cluster_name", "n_bouts", "total_frames", "frequency",
        "mean_duration_s", "median_duration_s", "std_duration_s", "cv_duration",
        "min_duration_s", "max_duration_s", "mean_inter_bout_interval_s",
    ]
    with open(stats_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for cid in sorted(ethogram.bout_stats.keys()):
            bs = ethogram.bout_stats[cid]
            writer.writerow({
                "cluster_id": bs.cluster_id,
                "cluster_name": bs.cluster_name,
                "n_bouts": bs.n_bouts,
                "total_frames": bs.total_frames,
                "frequency": round(bs.frequency, 6),
                "mean_duration_s": round(bs.mean_duration_s, 6),
                "median_duration_s": round(bs.median_duration_s, 6),
                "std_duration_s": round(bs.std_duration_s, 6),
                "cv_duration": round(bs.cv_duration, 6),
                "min_duration_s": round(bs.min_duration_s, 6),
                "max_duration_s": round(bs.max_duration_s, 6),
                "mean_inter_bout_interval_s": round(bs.mean_inter_bout_interval_s, 6),
            })

    # --- transition_matrix.csv ---
    tm = ethogram.transition_matrix
    tm_path = os.path.join(output_path, "transition_matrix.csv")
    with open(tm_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + tm.cluster_names)
        for i, name in enumerate(tm.cluster_names):
            writer.writerow([name] + [round(float(x), 6) for x in tm.matrix[i]])

    # --- transition_counts.csv ---
    tc_path = os.path.join(output_path, "transition_counts.csv")
    with open(tc_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + tm.cluster_names)
        for i, name in enumerate(tm.cluster_names):
            writer.writerow([name] + [int(x) for x in tm.counts[i]])

    # --- bouts.csv ---
    bouts_path = os.path.join(output_path, "bouts.csv")
    with open(bouts_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "cluster_name", "start_frame", "end_frame",
                         "duration_frames", "duration_seconds"])
        for b in ethogram.bouts:
            writer.writerow([
                b.cluster_id,
                ethogram.cluster_names.get(b.cluster_id, f"cluster_{b.cluster_id}"),
                b.start_frame,
                b.end_frame,
                b.duration_frames,
                round(b.duration_seconds, 6),
            ])

    logger.info("Exported ethogram CSV files to %s", output_path)
    return output_path
