"""Service layer for NWB export.

Loads cluster data from CASTLE projects and delegates to
:mod:`castle.core.nwb_export`.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def export_project_nwb(
    project_path: str,
    output_path: Optional[str] = None,
    session_description: str = "CASTLE behavioral analysis",
    experimenter: str = "",
) -> str:
    """Export a CASTLE project's results to NWB format.

    Loads cluster labels, ethogram analysis, and transition data from the
    project directory and exports to an NWB file.

    Args:
        project_path: Path to the CASTLE project directory.
        output_path: Path for .nwb output file. If None, defaults to
            ``<project_path>/export/<project_name>.nwb``.
        session_description: Description for the NWB session.
        experimenter: Experimenter name.

    Returns:
        Absolute path to the created NWB file.

    Raises:
        ImportError: If pynwb is not installed.
        FileNotFoundError: If project has no cluster data.
    """
    from castle.core.nwb_export import export_to_nwb, _require_pynwb

    _require_pynwb()

    project_path = os.path.abspath(project_path)
    project_name = os.path.basename(project_path)

    if output_path is None:
        export_dir = os.path.join(project_path, "export")
        os.makedirs(export_dir, exist_ok=True)
        output_path = os.path.join(export_dir, f"{project_name}.nwb")

    # Load cluster data via ethogram service
    from castle.service.ethogram_service import analyze_ethogram

    ethogram_result = analyze_ethogram(project_path)

    if ethogram_result.get("status") != "success":
        raise FileNotFoundError(
            f"Cannot load cluster data from {project_path}: "
            f"{ethogram_result.get('message', 'unknown error')}"
        )

    import numpy as np

    # Reconstruct cluster labels from the raw data
    from castle.service.comparison_service import _load_per_video_cluster_data

    data = _load_per_video_cluster_data(project_path)
    # Concatenate all video labels
    all_labels = np.concatenate([v["labels"] for v in data["videos"]])
    cluster_names = data["cluster_names"]

    fps = ethogram_result.get("fps", 30.0)
    bout_stats = ethogram_result.get("bout_stats", None)

    # Get transition matrix
    tm = ethogram_result.get("transition_matrix", {})
    transition_matrix = None
    if tm and "matrix" in tm:
        transition_matrix = np.array(tm["matrix"])

    result_path = export_to_nwb(
        output_path=output_path,
        cluster_labels=all_labels,
        fps=fps,
        cluster_names=cluster_names,
        bout_stats=bout_stats,
        transition_matrix=transition_matrix,
        session_description=session_description,
        experimenter=experimenter,
        subject_id=project_name,
    )

    logger.info("Exported NWB file to %s", result_path)
    return result_path
