"""Service layer for NWB export.

Loads cluster data from CASTLE projects and delegates to
:mod:`castle.core.nwb_export`.
"""

import os
import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def _safe_stem(video_name: str) -> str:
    """Filename-safe stem for a video name (drop extension + unsafe chars)."""
    stem = os.path.splitext(video_name)[0]
    return re.sub(r"[^A-Za-z0-9._-]", "_", stem) or "video"


def export_project_nwb(
    project_path: str,
    output_path: Optional[str] = None,
    session_description: str = "CASTLE behavioral analysis",
    experimenter: str = "",
) -> List[str]:
    """Export a CASTLE project's results to NWB format — **one file per video**.

    NWB files are per-recording (per-session) by convention, so each video is
    written as its own NWB with its OWN frame rate and its OWN per-video bouts /
    transition matrix. This avoids fusing independent recordings into one fake
    continuous timeline (which would invent cross-video bouts and assign wrong
    absolute times to every video after the first) and removes the mixed-fps
    crash from collapsing several rates into a single scalar.

    Args:
        project_path: Path to the CASTLE project directory.
        output_path: Optional. For a single-video project, the exact ``.nwb``
            path to write. For a multi-video project, treated as the output
            **directory** (or, if it ends in ``.nwb``, its parent dir + filename
            stem are used as a per-video prefix). If None, defaults to
            ``<project_path>/export/`` with ``<project>_<video>.nwb`` filenames.
        session_description: Description for each NWB session.
        experimenter: Experimenter name.

    Returns:
        List of absolute paths to the created NWB files (one per video).

    Raises:
        ImportError: If pynwb is not installed.
        FileNotFoundError: If project has no cluster data.
    """
    from castle.core.nwb_export import export_to_nwb, _require_pynwb

    _require_pynwb()

    project_path = os.path.abspath(project_path)
    project_name = os.path.basename(project_path)

    # Validate cluster data exists + obtain per-video fps.
    from castle.service.ethogram_service import analyze_ethogram, _video_fps

    ethogram_result = analyze_ethogram(project_path)
    if ethogram_result.get("status") != "success":
        raise FileNotFoundError(
            f"Cannot load cluster data from {project_path}: "
            f"{ethogram_result.get('message', 'unknown error')}"
        )

    import numpy as np
    from castle.core.ethogram import compute_ethogram
    from castle.service.comparison_service import _load_per_video_cluster_data

    data = _load_per_video_cluster_data(project_path)
    cluster_names = data["cluster_names"]
    video_fps = ethogram_result.get("video_fps", {}) or {}
    videos = data["videos"]
    multi = len(videos) > 1

    # Resolve one output path per video.
    if output_path is not None and not multi:
        out_paths = [output_path]
    else:
        if output_path is not None:
            if output_path.endswith(".nwb"):
                base_dir = os.path.dirname(os.path.abspath(output_path)) or "."
                prefix = os.path.splitext(os.path.basename(output_path))[0]
            else:
                base_dir = output_path
                prefix = project_name
        else:
            base_dir = os.path.join(project_path, "export")
            prefix = project_name
        os.makedirs(base_dir, exist_ok=True)
        out_paths = [
            os.path.join(base_dir, f"{prefix}_{_safe_stem(v['video_name'])}.nwb")
            for v in videos
        ]

    written: List[str] = []
    for vid, out in zip(videos, out_paths):
        vname = vid["video_name"]
        labels = np.asarray(vid["labels"], dtype=np.int32)

        fps = video_fps.get(vname)
        if fps is None or not np.isfinite(fps) or fps <= 0:
            fps = _video_fps(project_path, vname)
        fps = float(fps)

        # Per-video ethogram → bouts/transition matrix at THIS video's fps.
        etho = compute_ethogram(labels, fps=fps, cluster_names=cluster_names)
        bout_stats = {
            str(cid): {
                "cluster_name": bs.cluster_name,
                "n_bouts": bs.n_bouts,
                "frequency": bs.frequency,
                "frequency_valid_only": bs.frequency_valid_only,
                "mean_duration_s": bs.mean_duration_s,
                "median_duration_s": bs.median_duration_s,
                "cv_duration": bs.cv_duration,
            }
            for cid, bs in etho.bout_stats.items()
        }
        transition_matrix = np.asarray(etho.transition_matrix.matrix)

        result_path = export_to_nwb(
            output_path=out,
            cluster_labels=labels,
            fps=fps,
            cluster_names=cluster_names,
            bout_stats=bout_stats,
            transition_matrix=transition_matrix,
            session_description=session_description,
            experimenter=experimenter,
            subject_id=f"{project_name}/{vname}",
        )
        written.append(result_path)
        logger.info("Exported NWB for video %s to %s", vname, result_path)

    return written
