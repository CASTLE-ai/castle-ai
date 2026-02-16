"""Service layer for group comparison.

Loads cluster data from CASTLE projects and delegates computation
to :mod:`castle.core.comparison`.
"""

import os
import csv
import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def _load_per_video_cluster_data(project_path: str) -> Dict:
    """Load per-video cluster labels and metadata from a project.

    Reads ``cluster/id.csv`` for cluster metadata and each
    ``cluster/time_series_*.csv`` as a separate animal/video.

    Returns:
        dict with keys:
            ``videos`` — list of dicts with keys ``video_name``, ``labels``
            ``cluster_names`` — dict of id→name
    """
    cluster_dir = os.path.join(project_path, "cluster")
    if not os.path.isdir(cluster_dir):
        raise FileNotFoundError(
            f"No cluster directory found at {cluster_dir}. "
            "Run clustering first."
        )

    # --- cluster metadata from id.csv ---
    id_csv = os.path.join(cluster_dir, "id.csv")
    cluster_names: Dict[int, str] = {}
    if os.path.exists(id_csv):
        with open(id_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = int(row["Id"])
                name = row.get("Name", f"cluster_{cid}")
                cluster_names[cid] = name

    # --- per-video labels from time_series CSVs ---
    ts_files = sorted(
        f
        for f in os.listdir(cluster_dir)
        if f.startswith("time_series_") and f.endswith(".csv")
    )
    if not ts_files:
        raise FileNotFoundError(
            f"No time_series_*.csv files found in {cluster_dir}. "
            "Run clustering and submit first."
        )

    videos = []
    for ts_file in ts_files:
        ts_path = os.path.join(cluster_dir, ts_file)
        labels = []
        with open(ts_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append(int(row["behavior"]))
        # Derive video name from filename: time_series_<name>.csv
        video_name = ts_file[len("time_series_") : -len(".csv")]
        videos.append(
            {
                "video_name": video_name,
                "labels": np.array(labels, dtype=np.int32),
            }
        )

    return {"videos": videos, "cluster_names": cluster_names}


def _result_to_dict(result) -> dict:
    """Convert a ComparisonResult to a JSON-serialisable dict."""

    def _arr(x):
        if x is None:
            return None
        return [round(float(v), 6) for v in x]

    return {
        "group_a_name": result.group_a_name,
        "group_b_name": result.group_b_name,
        "n_a": result.n_a,
        "n_b": result.n_b,
        "bfa_distance": round(result.bfa_distance, 6),
        "bfa_pvalue": round(result.bfa_pvalue, 6),
        "energy_distance": round(result.energy_distance, 6)
        if result.energy_distance is not None
        else None,
        "energy_pvalue": round(result.energy_pvalue, 6)
        if result.energy_pvalue is not None
        else None,
        "feature_names": result.feature_names,
        "feature_pvalues": _arr(result.feature_pvalues),
        "feature_pvalues_adj": _arr(result.feature_pvalues_adj),
        "feature_effect_sizes": _arr(result.feature_effect_sizes),
        "feature_ci_lower": _arr(result.feature_ci_lower),
        "feature_ci_upper": _arr(result.feature_ci_upper),
        "feature_means_a": _arr(result.feature_means_a),
        "feature_means_b": _arr(result.feature_means_b),
        "significant_features": result.significant_features,
        "summary": result.summary,
    }


def _fingerprint_to_dict(fp) -> dict:
    """Convert a BehavioralFingerprint to a JSON-serialisable dict."""
    return {
        "animal_id": fp.animal_id,
        "group": fp.group,
        "cluster_names": fp.cluster_names,
        "n_frames": fp.n_frames,
        "fps": fp.fps,
        "frequencies": [round(float(v), 6) for v in fp.frequencies],
        "bout_counts": [round(float(v), 6) for v in fp.bout_counts],
        "mean_bout_durations": [round(float(v), 6) for v in fp.mean_bout_durations],
        "median_bout_durations": [
            round(float(v), 6) for v in fp.median_bout_durations
        ],
        "cv_bout_durations": [round(float(v), 6) for v in fp.cv_bout_durations],
        "inter_bout_intervals": [
            round(float(v), 6) for v in fp.inter_bout_intervals
        ],
        "transition_matrix": [
            [round(float(v), 6) for v in row] for row in fp.transition_matrix
        ],
    }


def compare_projects(
    project_a_path: str,
    project_b_path: str,
    group_a_name: str = "Group A",
    group_b_name: str = "Group B",
    fps: float = 30.0,
    n_permutations: int = 10000,
) -> dict:
    """Compare behavioral patterns between two projects (groups).

    Each project represents one group. Individual videos within each project
    are treated as individual animals.

    Args:
        project_a_path: Path to the Group A project directory.
        project_b_path: Path to the Group B project directory.
        group_a_name: Display name for group A.
        group_b_name: Display name for group B.
        fps: Frames per second.
        n_permutations: Number of permutations for statistical tests.

    Returns:
        JSON-serialisable dict with comparison results.
    """
    from castle.core.comparison import compute_fingerprint, compare_groups

    project_a_path = os.path.abspath(project_a_path)
    project_b_path = os.path.abspath(project_b_path)

    data_a = _load_per_video_cluster_data(project_a_path)
    data_b = _load_per_video_cluster_data(project_b_path)

    # Merge cluster names from both projects
    cluster_names = {**data_a["cluster_names"], **data_b["cluster_names"]}

    fps_a = []
    for vid in data_a["videos"]:
        fp = compute_fingerprint(
            animal_id=vid["video_name"],
            group=group_a_name,
            cluster_labels=vid["labels"],
            fps=fps,
            cluster_names=cluster_names,
        )
        fps_a.append(fp)

    fps_b = []
    for vid in data_b["videos"]:
        fp = compute_fingerprint(
            animal_id=vid["video_name"],
            group=group_b_name,
            cluster_labels=vid["labels"],
            fps=fps,
            cluster_names=cluster_names,
        )
        fps_b.append(fp)

    if not fps_a or not fps_b:
        return {
            "status": "error",
            "message": "Each project must have at least one video.",
        }

    result = compare_groups(fps_a, fps_b, n_permutations=n_permutations)

    out = _result_to_dict(result)
    out["status"] = "success"
    out["fingerprints_a"] = [_fingerprint_to_dict(fp) for fp in fps_a]
    out["fingerprints_b"] = [_fingerprint_to_dict(fp) for fp in fps_b]
    return out


def compute_project_fingerprints(
    project_path: str,
    group_name: str = "default",
    fps: float = 30.0,
) -> dict:
    """Compute behavioral fingerprints for all videos in a project.

    Args:
        project_path: Path to the project directory.
        group_name: Group label for these fingerprints.
        fps: Frames per second.

    Returns:
        JSON-serialisable dict with per-video fingerprints.
    """
    from castle.core.comparison import compute_fingerprint

    project_path = os.path.abspath(project_path)
    data = _load_per_video_cluster_data(project_path)

    fingerprints = []
    for vid in data["videos"]:
        fp = compute_fingerprint(
            animal_id=vid["video_name"],
            group=group_name,
            cluster_labels=vid["labels"],
            fps=fps,
            cluster_names=data["cluster_names"],
        )
        fingerprints.append(_fingerprint_to_dict(fp))

    return {
        "status": "success",
        "project_path": project_path,
        "group_name": group_name,
        "n_animals": len(fingerprints),
        "fingerprints": fingerprints,
    }


def compare_projects_paired(
    project_before_path: str,
    project_after_path: str,
    group_before_name: str = "Before",
    group_after_name: str = "After",
    fps: float = 30.0,
    n_permutations: int = 10000,
) -> dict:
    """Paired comparison between two projects (within-subject design).

    Each video in project_before is matched to the corresponding video
    in project_after by order. Both projects must have the same number
    of videos.

    Args:
        project_before_path: Path to the pre-treatment project directory.
        project_after_path: Path to the post-treatment project directory.
        group_before_name: Display name for "before" condition.
        group_after_name: Display name for "after" condition.
        fps: Frames per second.
        n_permutations: Number of permutations for statistical tests.

    Returns:
        JSON-serialisable dict with paired comparison results.
    """
    from castle.core.comparison import compute_fingerprint, compare_paired

    project_before_path = os.path.abspath(project_before_path)
    project_after_path = os.path.abspath(project_after_path)

    data_before = _load_per_video_cluster_data(project_before_path)
    data_after = _load_per_video_cluster_data(project_after_path)

    if len(data_before["videos"]) != len(data_after["videos"]):
        return {
            "status": "error",
            "message": (
                f"Paired comparison requires the same number of videos in both "
                f"projects. Before has {len(data_before['videos'])}, "
                f"after has {len(data_after['videos'])}."
            ),
        }

    cluster_names = {**data_before["cluster_names"], **data_after["cluster_names"]}

    fps_before = []
    for vid in data_before["videos"]:
        fp = compute_fingerprint(
            animal_id=vid["video_name"],
            group=group_before_name,
            cluster_labels=vid["labels"],
            fps=fps,
            cluster_names=cluster_names,
        )
        fps_before.append(fp)

    fps_after = []
    for vid in data_after["videos"]:
        fp = compute_fingerprint(
            animal_id=vid["video_name"],
            group=group_after_name,
            cluster_labels=vid["labels"],
            fps=fps,
            cluster_names=cluster_names,
        )
        fps_after.append(fp)

    result = compare_paired(fps_before, fps_after, n_permutations=n_permutations)

    out = _result_to_dict(result)
    out["status"] = "success"
    out["paired"] = True
    out["n_pairs"] = len(fps_before)
    out["fingerprints_before"] = [_fingerprint_to_dict(fp) for fp in fps_before]
    out["fingerprints_after"] = [_fingerprint_to_dict(fp) for fp in fps_after]
    return out


def export_comparison_report(result: dict, output_dir: str) -> List[str]:
    """Export comparison results to CSV files.

    Creates:
      - ``summary.txt`` — text summary
      - ``omnibus_tests.csv`` — BFA and energy distance results
      - ``feature_tests.csv`` — per-feature test results

    Args:
        result: Comparison result dict (from :func:`compare_projects`).
        output_dir: Directory to write files into.

    Returns:
        List of created file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    created: List[str] = []

    # summary.txt
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(result.get("summary", ""))
    created.append(summary_path)

    # omnibus_tests.csv
    omnibus_path = os.path.join(output_dir, "omnibus_tests.csv")
    with open(omnibus_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test", "statistic", "p_value"])
        writer.writerow(["BFA", result.get("bfa_distance", ""), result.get("bfa_pvalue", "")])
        if result.get("energy_distance") is not None:
            writer.writerow(
                ["Energy", result["energy_distance"], result["energy_pvalue"]]
            )
    created.append(omnibus_path)

    # feature_tests.csv
    if result.get("feature_names"):
        feat_path = os.path.join(output_dir, "feature_tests.csv")
        with open(feat_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "feature",
                    "mean_a",
                    "mean_b",
                    "p_value",
                    "p_adjusted",
                    "hedges_g",
                    "ci_lower",
                    "ci_upper",
                ]
            )
            n_feat = len(result["feature_names"])
            for i in range(n_feat):
                writer.writerow(
                    [
                        result["feature_names"][i],
                        result["feature_means_a"][i] if result.get("feature_means_a") else "",
                        result["feature_means_b"][i] if result.get("feature_means_b") else "",
                        result["feature_pvalues"][i] if result.get("feature_pvalues") else "",
                        result["feature_pvalues_adj"][i]
                        if result.get("feature_pvalues_adj")
                        else "",
                        result["feature_effect_sizes"][i]
                        if result.get("feature_effect_sizes")
                        else "",
                        result["feature_ci_lower"][i]
                        if result.get("feature_ci_lower")
                        else "",
                        result["feature_ci_upper"][i]
                        if result.get("feature_ci_upper")
                        else "",
                    ]
                )
        created.append(feat_path)

    logger.info("Exported comparison report to %s", output_dir)
    return created
