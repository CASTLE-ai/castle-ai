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
    all_reasons = []
    any_missing_reason = False
    for ts_file in ts_files:
        ts_path = os.path.join(cluster_dir, ts_file)
        labels_f, reason_f = _read_time_series(ts_path)
        all_labels.append(labels_f)
        if reason_f is None:
            any_missing_reason = True
        else:
            all_reasons.append(reason_f)

    labels = (
        np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int32)
    )
    # Only expose a reason array if every video carried the column; otherwise
    # the ethogram falls back to bucketing -1 frames as "unknown".
    exclude_reason = (
        np.concatenate(all_reasons)
        if (all_reasons and not any_missing_reason)
        else None
    )

    return {
        "labels": labels,
        "exclude_reason": exclude_reason,
        "cluster_names": cluster_names,
        "cluster_meta": cluster_meta,
        "fps": None,  # caller should supply; we don't have it in CSV
    }


def _resolve_project_path(project_path: str) -> str:
    """If *project_path* looks like ``storage/project_name``, return as-is."""
    return os.path.abspath(project_path)


# ------------------------------------------------------------------ #
# Per-video helpers (mixed-fps / per-subject ethograms)
#
# A project may hold several videos at different frame rates (e.g. one
# animal per video). Pooling every video's frames into a single sequence
# and applying one fps (a) scales durations wrongly for every video that
# isn't at that fps and (b) merges bouts across video boundaries (a run at
# the end of video A + the start of video B becomes one spurious bout).
# These helpers compute one ethogram per video, each from that video's own
# per-frame ``time_series_{basename}.csv`` and its own fps.
# ------------------------------------------------------------------ #


def _read_behavior_csv(ts_path: str) -> np.ndarray:
    """Read a per-frame ``behavior`` column from a time_series CSV."""
    labels, _ = _read_time_series(ts_path)
    return labels


def _read_time_series(ts_path: str):
    """Read ``behavior`` (+ optional ``exclude_reason``) from a time_series CSV.

    Returns ``(labels, exclude_reason)``. ``exclude_reason`` is ``None`` for
    legacy CSVs that predate the column (the ethogram then buckets every -1 as
    ``"unknown"``).
    """
    labels = []
    reasons = []
    has_reason = False
    with open(ts_path, "r") as f:
        reader = csv.DictReader(f)
        has_reason = reader.fieldnames is not None and "exclude_reason" in reader.fieldnames
        for row in reader:
            labels.append(int(row["behavior"]))
            if has_reason:
                reasons.append(int(row["exclude_reason"]))
    labels_arr = np.array(labels, dtype=np.int32)
    reason_arr = np.array(reasons, dtype=np.int8) if has_reason else None
    return labels_arr, reason_arr


def _list_video_time_series(project_path: str):
    """Yield ``(video_basename, ts_path)`` for each per-video time_series CSV."""
    cluster_dir = os.path.join(project_path, "cluster")
    if not os.path.isdir(cluster_dir):
        return
    for fname in sorted(os.listdir(cluster_dir)):
        if fname.startswith("time_series_") and fname.endswith(".csv"):
            basename = fname[len("time_series_"):-len(".csv")]
            yield basename, os.path.join(cluster_dir, fname)


def _video_fps(project_path: str, video_name: str, default: float = 30.0) -> float:
    """Read a single video's fps from ``sources/``.

    ``video_name`` may be a full filename or a bare basename (the time_series
    files only carry the basename); in the latter case we glob ``sources`` for
    a matching source file. Falls back to ``default`` if unreadable.
    """
    import glob

    from castle.utils.video_io import VideoReader

    sources = os.path.join(project_path, "sources")
    direct = os.path.join(sources, video_name)
    if os.path.exists(direct):
        candidates = [direct]
    else:
        base = os.path.splitext(os.path.basename(video_name))[0]
        candidates = sorted(glob.glob(os.path.join(sources, base + ".*")))
    for path in candidates:
        try:
            with VideoReader(path) as vr:
                fps = float(vr.fps)
            if fps > 0:
                return fps
        except Exception as exc:  # noqa: BLE001 — fps probe must never block analysis
            logger.warning("Could not read fps from %s: %s", path, exc)
    return default


def compute_video_ethogram(
    project_path: str,
    video_name: str,
    cluster_names: dict = None,
    fps: float = None,
    smooth: bool = False,
    smooth_window: int = 5,
    min_bout_frames: int = 3,
):
    """Compute an :class:`~castle.core.ethogram.Ethogram` for ONE video.

    Reads the video's per-frame ``time_series_{basename}.csv`` and uses that
    video's own fps, so bout durations are correct in mixed-fps projects and no
    bout is merged across a video boundary.

    Args:
        project_path: Project directory (``storage/project_name``).
        video_name: Video filename or basename (matched against the
            ``time_series_{basename}.csv`` files).
        cluster_names: Optional ``{cluster_id: name}`` display mapping.
        fps: Override fps; if None, read the video's own fps.
        smooth / smooth_window / min_bout_frames: optional temporal smoothing.

    Raises:
        FileNotFoundError: No time_series CSV for this video.
    """
    from castle.core.ethogram import compute_ethogram

    project_path = _resolve_project_path(project_path)
    basename = os.path.splitext(os.path.basename(video_name))[0]
    ts_path = os.path.join(project_path, "cluster", f"time_series_{basename}.csv")
    if not os.path.exists(ts_path):
        raise FileNotFoundError(
            f"No time_series CSV for video {video_name!r} at {ts_path}. "
            "Run clustering and submit first."
        )

    labels, exclude_reason = _read_time_series(ts_path)
    if smooth:
        from castle.core.temporal_smooth import smooth_labels
        labels = smooth_labels(
            labels, method="both", window=smooth_window, min_bout_frames=min_bout_frames,
        )
        # Smoothing is gap-preserving (the -1 set is invariant), so
        # exclude_reason stays aligned with the smoothed labels.

    effective_fps = fps if fps is not None else _video_fps(project_path, video_name)
    return compute_ethogram(
        labels, fps=effective_fps, cluster_names=cluster_names or {},
        exclude_reason=exclude_reason,
    )


def _ethogram_to_dict(ethogram) -> dict:
    """Serialise an :class:`Ethogram` to a JSON-safe dict."""
    tm = ethogram.transition_matrix
    return {
        "schema_version": ethogram.schema_version,
        "n_frames": ethogram.n_frames,
        "fps": ethogram.fps,
        "n_clusters": ethogram.n_clusters,
        "n_unlabeled": ethogram.n_unlabeled,
        "unlabeled_fraction": round(ethogram.unlabeled_fraction, 4),
        "n_valid_frames": ethogram.n_valid_frames,
        "n_excluded_frames": ethogram.n_unlabeled,
        "valid_frame_fraction": round(ethogram.valid_frame_fraction, 4),
        "excluded_reason_counts": dict(ethogram.excluded_reason_counts),
        "cluster_names": ethogram.cluster_names,
        "temporal_coherence": round(ethogram.temporal_coherence, 4),
        "transition_matrix": {
            "matrix": tm.matrix.tolist(),
            "counts": tm.counts.tolist(),
            "cluster_ids": tm.cluster_ids,
            "cluster_names": tm.cluster_names,
            "n_transitions": tm.n_transitions,
            "entropy": round(tm.entropy, 4),
            "stationarity": round(tm.stationarity, 4),  # deprecated (cosine)
            "stationarity_jsd": _round_or_none(tm.stationarity_jsd, 4),
            "stationarity_status": tm.stationarity_status,
        },
        "bout_stats": {
            str(cid): {
                "cluster_name": bs.cluster_name,
                "n_bouts": bs.n_bouts,
                "total_frames": bs.total_frames,
                "frequency": round(bs.frequency, 4),  # deprecated (/ all frames)
                "frequency_valid_only": round(bs.frequency_valid_only, 4),
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


def _round_or_none(x: float, ndigits: int):
    """Round a float, but preserve NaN as None for JSON-safety."""
    return round(x, ndigits) if x is not None and np.isfinite(x) else None


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
        exclude_reason=data.get("exclude_reason"),
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


def export_ethogram_csv(
    project_path: str,
    output_path: str,
    session_id: str = None,
) -> str:
    """Export per-video ethogram data to CSV files.

    One ethogram is computed per video (each from its own per-frame
    ``time_series_{basename}.csv`` and its own fps), so durations are correct in
    mixed-fps projects and no bout is merged across a video boundary. Creates:

      - ``bout_stats.csv``  — per-cluster summary stats, long-format with a
        leading ``video`` column (one row per video × cluster; ready for
        per-subject group analysis)
      - ``bouts.csv`` — every individual bout, long-format with a ``video`` column
      - ``transition_matrix_{video}.csv`` — per-video transition probabilities
      - ``transition_counts_{video}.csv`` — per-video raw transition counts

    Cluster names follow the Analysis page convention (Bug 11):
    ``"human_label — bm_name"`` when a human annotation exists, else the BM name.

    Args:
        project_path: Project directory path.
        output_path: Directory to write CSV files into.
        session_id: Optional session ID used to locate the annotations CSV.

    Returns:
        Path to the output directory.
    """
    project_path = _resolve_project_path(project_path)
    data = _load_cluster_data(project_path)  # only for cluster_names from id.csv

    # Apply annotation labels when available (Bug 11)
    annotations: dict = {}
    try:
        storage_path = os.path.dirname(project_path)
        project_name = os.path.basename(project_path)
        from castle.service.annotation_service import load_annotations as _load_ann
        annotations = _load_ann(storage_path, project_name, session_id=session_id)
    except Exception as _exc:
        logger.warning("Could not load annotations for ethogram CSV export: %s", _exc)

    def _display_name(bm_name: str) -> str:
        ann = annotations.get(bm_name)
        if ann and ann.get("behavior_label"):
            return f"{ann['behavior_label']} \u2014 {bm_name}"
        return bm_name

    annotated_names = {
        cid: _display_name(name)
        for cid, name in data["cluster_names"].items()
    }

    videos = list(_list_video_time_series(project_path))
    if not videos:
        raise FileNotFoundError(
            f"No time_series_*.csv files in {os.path.join(project_path, 'cluster')}. "
            "Run clustering and submit first."
        )

    os.makedirs(output_path, exist_ok=True)

    # One ethogram per video (own fps, no cross-video bouts/transitions). bout_stats
    # and bouts are written long-format with a `video` column (ready for per-subject
    # group analysis); transition matrices are written per video.
    stats_fields = [
        "video", "cluster_id", "cluster_name", "n_bouts", "total_frames", "frequency",
        "mean_duration_s", "median_duration_s", "std_duration_s", "cv_duration",
        "min_duration_s", "max_duration_s", "mean_inter_bout_interval_s",
    ]
    bouts_fields = [
        "video", "cluster_id", "cluster_name", "start_frame", "end_frame",
        "duration_frames", "duration_seconds",
    ]
    # Per-video summary: unlabeled (noise) frames are excluded from the bout /
    # transition stats above and reported here separately (ready for group analysis).
    summary_fields = [
        "video", "n_frames", "fps", "n_clusters", "n_unlabeled", "unlabeled_fraction",
        "temporal_coherence", "transition_entropy", "stationarity", "n_transitions",
    ]

    stats_path = os.path.join(output_path, "bout_stats.csv")
    bouts_path = os.path.join(output_path, "bouts.csv")
    summary_path = os.path.join(output_path, "video_summary.csv")
    with open(stats_path, "w", newline="") as sf, \
            open(bouts_path, "w", newline="") as bf, \
            open(summary_path, "w", newline="") as mf:
        stats_writer = csv.DictWriter(sf, fieldnames=stats_fields)
        stats_writer.writeheader()
        bouts_writer = csv.writer(bf)
        bouts_writer.writerow(bouts_fields)
        summary_writer = csv.DictWriter(mf, fieldnames=summary_fields)
        summary_writer.writeheader()

        for basename, _ts_path in videos:
            ethogram = compute_video_ethogram(
                project_path, basename, cluster_names=annotated_names,
            )

            tm_summary = ethogram.transition_matrix
            summary_writer.writerow({
                "video": basename,
                "n_frames": ethogram.n_frames,
                "fps": round(ethogram.fps, 6),
                "n_clusters": ethogram.n_clusters,
                "n_unlabeled": ethogram.n_unlabeled,
                "unlabeled_fraction": round(ethogram.unlabeled_fraction, 6),
                "temporal_coherence": round(ethogram.temporal_coherence, 6),
                "transition_entropy": round(tm_summary.entropy, 6),
                "stationarity": round(tm_summary.stationarity, 6),
                "n_transitions": tm_summary.n_transitions,
            })

            for cid in sorted(ethogram.bout_stats.keys()):
                bs = ethogram.bout_stats[cid]
                stats_writer.writerow({
                    "video": basename,
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

            for b in ethogram.bouts:
                bouts_writer.writerow([
                    basename,
                    b.cluster_id,
                    ethogram.cluster_names.get(b.cluster_id, f"cluster_{b.cluster_id}"),
                    b.start_frame,
                    b.end_frame,
                    b.duration_frames,
                    round(b.duration_seconds, 6),
                ])

            tm = ethogram.transition_matrix
            tm_path = os.path.join(output_path, f"transition_matrix_{basename}.csv")
            with open(tm_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([""] + tm.cluster_names)
                for i, name in enumerate(tm.cluster_names):
                    writer.writerow([name] + [round(float(x), 6) for x in tm.matrix[i]])

            tc_path = os.path.join(output_path, f"transition_counts_{basename}.csv")
            with open(tc_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([""] + tm.cluster_names)
                for i, name in enumerate(tm.cluster_names):
                    writer.writerow([name] + [int(x) for x in tm.counts[i]])

    logger.info("Exported per-video ethogram CSV files to %s", output_path)
    return output_path
