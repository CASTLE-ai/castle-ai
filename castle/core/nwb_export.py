"""Export CASTLE results to NWB (Neurodata Without Borders) format.

Requires optional dependency: pip install pynwb

Creates NWB files containing:
- BehavioralTimeSeries: cluster labels per frame
- TimeIntervals: behavioral bouts with start/stop times
- DynamicTable: bout statistics per cluster
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

try:
    import pynwb
    from pynwb import NWBFile, NWBHDF5IO
    from pynwb.behavior import BehavioralTimeSeries
    from pynwb.epoch import TimeIntervals
    from hdmf.common import DynamicTable, VectorData

    HAS_NWB = True
except ImportError:
    HAS_NWB = False


def _require_pynwb():
    """Raise ImportError with helpful message if pynwb is not installed."""
    if not HAS_NWB:
        raise ImportError(
            "pynwb is required for NWB export but is not installed. "
            "Install it with: pip install pynwb"
        )


def _extract_bouts(
    cluster_labels: np.ndarray, fps: float
) -> List[dict]:
    """Extract behavioral bouts from cluster label sequence.

    Returns list of dicts with keys:
        cluster_id, start_frame, stop_frame, start_time, stop_time, duration_s
    """
    labels = np.asarray(cluster_labels).ravel()
    n = len(labels)
    if n == 0:
        return []

    bouts = []
    start = 0
    current = labels[0]

    for i in range(1, n):
        if labels[i] != current:
            bouts.append({
                "cluster_id": int(current),
                "start_frame": int(start),
                "stop_frame": int(i - 1),
                "start_time": float(start / fps),
                "stop_time": float((i - 1) / fps),
                "duration_s": float((i - start) / fps),
            })
            start = i
            current = labels[i]

    # Final bout
    bouts.append({
        "cluster_id": int(current),
        "start_frame": int(start),
        "stop_frame": int(n - 1),
        "start_time": float(start / fps),
        "stop_time": float((n - 1) / fps),
        "duration_s": float((n - start) / fps),
    })

    return bouts


def export_to_nwb(
    output_path: str,
    cluster_labels: np.ndarray,
    fps: float,
    cluster_names: Optional[Dict[int, str]] = None,
    bout_stats: Optional[dict] = None,
    transition_matrix: Optional[np.ndarray] = None,
    session_description: str = "CASTLE behavioral analysis",
    experimenter: str = "",
    subject_id: str = "",
) -> str:
    """Export CASTLE analysis results to NWB file.

    Creates an NWB file containing:
    - BehavioralTimeSeries: cluster labels per frame
    - TimeIntervals: behavioral bouts with start/stop times
    - DynamicTable: bout statistics per cluster (if provided)
    - Transition matrix as a scratch data field (if provided)

    Args:
        output_path: Path for .nwb output file.
        cluster_labels: Per-frame cluster assignments (1-D int array).
        fps: Video frame rate.
        cluster_names: Optional mapping cluster_id → name.
        bout_stats: Optional bout statistics dict (keyed by cluster_id string).
        transition_matrix: Optional K×K transition probability matrix.
        session_description: Description for NWB session.
        experimenter: Experimenter name.
        subject_id: Subject/animal identifier.

    Returns:
        Path to created NWB file.
    """
    _require_pynwb()

    labels = np.asarray(cluster_labels, dtype=np.int32).ravel()
    n_frames = len(labels)

    if cluster_names is None:
        unique_ids = sorted(set(int(x) for x in labels))
        cluster_names = {cid: f"cluster_{cid}" for cid in unique_ids}

    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Create NWB file
    session_start = datetime.now(tz=timezone.utc)
    nwbfile = NWBFile(
        session_description=session_description,
        identifier=f"castle_{session_start.strftime('%Y%m%d_%H%M%S')}",
        session_start_time=session_start,
    )

    if experimenter:
        nwbfile.experimenter = [experimenter]

    if subject_id:
        from pynwb.file import Subject
        nwbfile.subject = Subject(subject_id=subject_id)

    # --- Behavioral time series: cluster labels ---
    from pynwb import TimeSeries

    behavior_mod = nwbfile.create_processing_module(
        "behavior", "CASTLE behavioral analysis results"
    )

    cluster_ts = TimeSeries(
        name="cluster_labels",
        data=labels,
        unit="cluster_id",
        rate=float(fps),
        description="Per-frame behavioral cluster assignments from CASTLE",
    )

    bts = BehavioralTimeSeries(name="behavioral_clusters")
    bts.add_timeseries(cluster_ts)
    behavior_mod.add(bts)

    # --- Time intervals: behavioral bouts ---
    bouts = _extract_bouts(labels, fps)

    if bouts:
        bout_intervals = TimeIntervals(
            name="behavioral_bouts",
            description="Behavioral bout intervals extracted from cluster labels",
        )
        bout_intervals.add_column(
            name="cluster_id", description="Behavioral cluster ID"
        )
        bout_intervals.add_column(
            name="cluster_name", description="Behavioral cluster name"
        )
        bout_intervals.add_column(
            name="duration_s", description="Bout duration in seconds"
        )

        for bout in bouts:
            cname = cluster_names.get(bout["cluster_id"], f"cluster_{bout['cluster_id']}")
            bout_intervals.add_row(
                start_time=bout["start_time"],
                stop_time=bout["stop_time"],
                cluster_id=bout["cluster_id"],
                cluster_name=cname,
                duration_s=bout["duration_s"],
            )

        behavior_mod.add(bout_intervals)

    # --- Bout statistics as a dynamic table ---
    if bout_stats:
        stat_names = []
        stat_n_bouts = []
        stat_freq = []
        stat_mean_dur = []
        stat_median_dur = []
        stat_cv = []

        for cid_str in sorted(bout_stats.keys(), key=lambda x: int(x)):
            bs = bout_stats[cid_str]
            cname = cluster_names.get(int(cid_str), bs.get("cluster_name", f"cluster_{cid_str}"))
            stat_names.append(cname)
            stat_n_bouts.append(bs.get("n_bouts", 0))
            stat_freq.append(bs.get("frequency", 0.0))
            stat_mean_dur.append(bs.get("mean_duration_s", 0.0))
            stat_median_dur.append(bs.get("median_duration_s", 0.0))
            stat_cv.append(bs.get("cv_duration", 0.0))

        bout_stats_table = DynamicTable(
            name="bout_statistics",
            description="Per-cluster bout statistics from CASTLE ethogram analysis",
            columns=[
                VectorData(name="cluster_name", description="Cluster name", data=stat_names),
                VectorData(name="n_bouts", description="Number of bouts", data=stat_n_bouts),
                VectorData(name="frequency", description="Fraction of total time", data=stat_freq),
                VectorData(name="mean_duration_s", description="Mean bout duration (seconds)", data=stat_mean_dur),
                VectorData(name="median_duration_s", description="Median bout duration (seconds)", data=stat_median_dur),
                VectorData(name="cv_duration", description="Coefficient of variation of bout durations", data=stat_cv),
            ],
        )
        behavior_mod.add(bout_stats_table)

    # --- Transition matrix as scratch data ---
    if transition_matrix is not None:
        tm = np.asarray(transition_matrix, dtype=np.float64)
        nwbfile.add_scratch(
            tm,
            name="transition_matrix",
            description="K×K behavioral state transition probability matrix",
        )

    # --- Write NWB file ---
    with NWBHDF5IO(output_path, "w") as io:
        io.write(nwbfile)

    return os.path.abspath(output_path)
