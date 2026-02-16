"""
castle/mcp/server.py
CASTLE MCP Server — exposes CASTLE pipeline as MCP tools and resources.

All heavy imports are lazy (inside each function) to keep startup fast.
Every tool returns a dict with at least {"status": "success"|"error", "message": "..."}.
"""

import json
import os

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Default storage path — can be overridden via CASTLE_STORAGE env var
# ---------------------------------------------------------------------------
_DEFAULT_STORAGE = os.environ.get("CASTLE_STORAGE", "projects/")


def _storage() -> str:
    return os.environ.get("CASTLE_STORAGE", _DEFAULT_STORAGE)


# ---------------------------------------------------------------------------
# FastMCP instance
# ---------------------------------------------------------------------------
mcp = FastMCP("castle", json_response=True)


# ======================================================================== #
#  TOOLS                                                                    #
# ======================================================================== #

@mcp.tool()
def project_create(name: str, source_dir: str = "") -> dict:
    """Create a new CASTLE analysis project.

    Args:
        name: Project name
        source_dir: Optional directory containing video files to import
    """
    try:
        from castle.service.project_service import (
            create_project,
            add_videos_from_directory,
        )

        result = create_project(_storage(), name)
        out = {
            "status": "success",
            "message": f"Project '{name}' created",
            "path": result["path"],
        }

        if source_dir and os.path.isdir(source_dir):
            vresult = add_videos_from_directory(_storage(), name, source_dir)
            out["videos_added"] = vresult["success_count"]
            out["videos_failed"] = vresult["fail_count"]
            out["video_messages"] = vresult["messages"]

        return out
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def project_list() -> dict:
    """List all CASTLE projects."""
    try:
        from castle.service.project_service import list_projects

        names = list_projects(_storage())
        return {
            "status": "success",
            "message": f"Found {len(names)} project(s)",
            "projects": names,
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def project_info(name: str) -> dict:
    """Get detailed information about a project including pipeline status.

    Args:
        name: Project name
    """
    try:
        from castle.service.project_service import get_project_info

        info = get_project_info(_storage(), name)
        if info.get("error"):
            return {"status": "error", "message": info["error"]}

        return {
            "status": "success",
            "message": f"Project '{name}': {info['video_count']} video(s)",
            "name": info["name"],
            "path": info["path"],
            "video_count": info["video_count"],
            "videos": info["videos"],
            "latent_count": info["latent_count"],
            "config": info["config"],
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def track_run(
    project: str,
    video: str = "All",
    model: str = "r50_deaotl",
) -> dict:
    """Run ROI tracking on a project's videos.

    Args:
        project: Project name
        video: Video filename, or "All" for every video
        model: Tracking model (r50_deaotl or swinb_deaotl)
    """
    try:
        from castle.service.project_service import get_project_info
        from castle.service.tracking_service import track_video

        info = get_project_info(_storage(), project)
        if info.get("error"):
            return {"status": "error", "message": info["error"]}

        videos = info["videos"] if video == "All" else [video]
        results = {}
        for vname in videos:
            status = track_video(_storage(), project, vname, model=model)
            results[vname] = status

        errors = {k: v for k, v in results.items() if v.startswith("Error")}
        if errors:
            return {
                "status": "error",
                "message": f"Tracking failed for {len(errors)}/{len(videos)} video(s)",
                "results": results,
            }
        return {
            "status": "success",
            "message": f"Tracked {len(videos)} video(s)",
            "results": results,
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def track_status(project: str, video: str) -> dict:
    """Check tracking status for a video.

    Args:
        project: Project name
        video: Video filename
    """
    try:
        from castle.service.tracking_service import get_tracking_status

        info = get_tracking_status(_storage(), project, video)
        return {"status": "success", "message": "OK", **info}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def extract_run(
    project: str,
    video: str = "All",
    model: str = "dinov2_vitb14_reg",
    roi: int = 1,
    batch_size: int = 32,
    pooling_scales: str = "1",
    feature_layers: str = "",
) -> dict:
    """Extract visual features from tracked ROIs.

    Args:
        project: Project name
        video: Video filename, or "All" for every video
        model: Feature model (dinov2_vitb14_reg, dinov3_vitl16, dinov3_vitb16)
        roi: ROI ID to extract
        batch_size: Batch size for inference
        pooling_scales: Comma-separated scales for spatial pyramid pooling (e.g. "1,2,4")
        feature_layers: Comma-separated ViT layers to extract (e.g. "3,7,11"). Empty = last layer only.
    """
    try:
        from castle.service.extraction_service import extract_latent

        scales = [int(s) for s in pooling_scales.split(",") if s.strip()]
        layers = [int(l) for l in feature_layers.split(",") if l.strip()] or None
        pooling_method = "multiscale" if len(scales) > 1 else "weighted_average"

        paths = extract_latent(
            storage_path=_storage(),
            project_name=project,
            video_name=video,
            model=model,
            roi=roi,
            batch_size=batch_size,
            pooling_method=pooling_method,
            pooling_scales=scales if len(scales) > 1 else None,
            feature_layers=layers,
        )

        path_list = [p for p in paths.split(";") if p]
        return {
            "status": "success",
            "message": f"Extracted {len(path_list)} latent file(s)",
            "paths": path_list,
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def cluster_run(
    project: str,
    roi: int = 1,
    bin_size: int = 1,
    model: str = "dinov2_vitb14_reg",
    n_neighbors: int = 100,
    min_dist: float = 0.0,
    n_components: int = 2,
    n_epochs: int = 5000,
    eps: float = 1.0,
) -> dict:
    """Run UMAP dimensionality reduction and DBSCAN clustering.

    Args:
        project: Project name
        roi: ROI ID
        bin_size: Temporal binning (frames per bin)
        model: Feature model used during extraction
        n_neighbors: UMAP n_neighbors
        min_dist: UMAP min_dist
        n_components: UMAP output dimensions
        n_epochs: UMAP n_epochs
        eps: DBSCAN epsilon parameter
    """
    try:
        from castle.service.clustering_service import ClusteringSession

        session = ClusteringSession(
            storage_path=_storage(),
            project_name=project,
            roi=roi,
            bin_size=bin_size,
            model=model,
        )

        umap_cfg = [{
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "n_components": n_components,
            "n_epochs": n_epochs,
        }]
        umap_result = session.run_umap("init", umap_cfg)
        if not umap_result.get("success"):
            return {
                "status": "error",
                "message": umap_result.get("error", "UMAP failed"),
            }

        dbscan_result = session.run_dbscan(eps=eps)
        if not dbscan_result.get("success"):
            return {
                "status": "error",
                "message": dbscan_result.get("error", "DBSCAN failed"),
            }

        # Auto-label and submit
        session.auto_label_all()
        submit_result = session.submit()

        return {
            "status": "success",
            "message": (
                f"Clustering complete: {dbscan_result['n_clusters']} cluster(s), "
                f"{dbscan_result['noise_count']} noise points"
            ),
            "n_clusters": dbscan_result["n_clusters"],
            "cluster_ids": dbscan_result["cluster_ids"],
            "noise_count": dbscan_result["noise_count"],
            "n_points": umap_result["n_points"],
            "id_csv_path": submit_result.get("id_csv_path", ""),
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def cluster_label(project: str, cluster_name: str, label: str, scheme: str = "") -> dict:
    """Label a behavioral cluster with a human-readable name.

    This saves an annotation mapping a cluster to a behavior label.

    Args:
        project: Project name
        cluster_name: Cluster name (as assigned during clustering)
        label: Behavioral label (e.g. "grooming", "rearing", "walking")
        scheme: Classification scheme name (optional)
    """
    try:
        from castle.service.annotation_service import load_annotations, save_annotations
        import datetime

        annotations = load_annotations(_storage(), project)
        annotations[cluster_name] = {
            "behavior_label": label,
            "scheme": scheme,
            "annotator": "mcp",
            "timestamp": datetime.datetime.now().isoformat(),
        }
        csv_path = save_annotations(_storage(), project, annotations)

        return {
            "status": "success",
            "message": f"Labeled cluster '{cluster_name}' as '{label}'",
            "csv_path": csv_path,
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def cluster_evaluate(project: str, storage: str = "", ground_truth: str = "") -> dict:
    """Evaluate clustering quality with automated metrics.

    Returns temporal coherence, bout quality, and optionally distance-based
    and ground-truth comparison metrics.

    Args:
        project: Project name
        storage: Storage directory (defaults to CASTLE_STORAGE env var)
        ground_truth: Optional path to a ground-truth CSV with a 'behavior' column
    """
    try:
        import os
        from castle.service.metrics_service import evaluate_project_clustering

        storage_dir = storage or _storage()
        project_path = os.path.join(storage_dir, project)
        if not os.path.isdir(project_path):
            return {"status": "error", "message": f"Project directory not found: {project_path}"}

        gt_path = ground_truth if ground_truth else None
        result = evaluate_project_clustering(project_path, ground_truth_path=gt_path)

        if "error" in result:
            return {"status": "error", "message": result["error"]}

        result["status"] = "success"
        result["message"] = f"Clustering quality: {result.get('verdict', 'N/A')}"
        return result
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def ethogram_analyze(
    project: str,
    storage: str = "",
    fps: float = 30.0,
    smooth: bool = False,
    smooth_window: int = 5,
    min_bout_frames: int = 3,
) -> dict:
    """Run ethogram analysis: transition matrix, bout stats, temporal coherence.

    Args:
        project: Project name
        storage: Storage directory (defaults to CASTLE_STORAGE env var)
        fps: Frames per second
        smooth: Apply temporal smoothing before analysis
        smooth_window: Smoothing window size (odd integer, default 5)
        min_bout_frames: Minimum bout duration in frames (default 3)
    """
    try:
        import os
        from castle.service.ethogram_service import analyze_ethogram

        storage_dir = storage or _storage()
        project_path = os.path.join(storage_dir, project)
        return analyze_ethogram(
            project_path, fps=fps,
            smooth=smooth, smooth_window=smooth_window,
            min_bout_frames=min_bout_frames,
        )
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def ethogram_transitions(project: str, storage: str = "") -> dict:
    """Get transition probability matrix.

    Args:
        project: Project name
        storage: Storage directory (defaults to CASTLE_STORAGE env var)
    """
    try:
        import os
        from castle.service.ethogram_service import get_transition_matrix

        storage_dir = storage or _storage()
        project_path = os.path.join(storage_dir, project)
        return get_transition_matrix(project_path)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def ethogram_bouts(project: str, storage: str = "") -> dict:
    """Get per-cluster bout statistics.

    Args:
        project: Project name
        storage: Storage directory (defaults to CASTLE_STORAGE env var)
    """
    try:
        import os
        from castle.service.ethogram_service import get_bout_statistics

        storage_dir = storage or _storage()
        project_path = os.path.join(storage_dir, project)
        return get_bout_statistics(project_path)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def compare_groups_tool(
    project_a: str,
    project_b: str,
    storage: str = "",
    group_a_name: str = "Control",
    group_b_name: str = "Treatment",
    fps: float = 30.0,
) -> dict:
    """Compare behavioral patterns between two groups using BFA and permutation tests.

    Args:
        project_a: Project name for group A
        project_b: Project name for group B
        storage: Storage directory (defaults to CASTLE_STORAGE env var)
        group_a_name: Display name for group A
        group_b_name: Display name for group B
        fps: Frames per second
    """
    try:
        import os
        from castle.service.comparison_service import compare_projects

        storage_dir = storage or _storage()
        path_a = os.path.join(storage_dir, project_a)
        path_b = os.path.join(storage_dir, project_b)
        return compare_projects(
            path_a,
            path_b,
            group_a_name=group_a_name,
            group_b_name=group_b_name,
            fps=fps,
        )
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def compute_fingerprint_tool(
    project: str,
    storage: str = "",
    fps: float = 30.0,
) -> dict:
    """Compute behavioral fingerprint for a project (per-animal summary).

    Args:
        project: Project name
        storage: Storage directory (defaults to CASTLE_STORAGE env var)
        fps: Frames per second
    """
    try:
        import os
        from castle.service.comparison_service import compute_project_fingerprints

        storage_dir = storage or _storage()
        project_path = os.path.join(storage_dir, project)
        return compute_project_fingerprints(project_path, group_name=project, fps=fps)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def device_info() -> dict:
    """Get GPU/device information for CASTLE processing."""
    try:
        from castle.core.environment import Environment, get_num_workers

        env = Environment()
        gpu_name = "N/A"
        if env.device == "cuda":
            try:
                import torch
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                pass

        return {
            "status": "success",
            "message": f"Device: {env.device}",
            "device": str(env.device),
            "gpu_name": gpu_name,
            "num_workers_default": get_num_workers("default"),
            "num_workers_extraction": get_num_workers("extraction"),
            "num_workers_tracking": get_num_workers("tracking"),
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# ======================================================================== #
#  RESOURCES                                                                #
# ======================================================================== #

@mcp.resource("castle://projects")
def list_all_projects() -> str:
    """List all CASTLE projects."""
    try:
        from castle.service.project_service import list_projects

        names = list_projects(_storage())
        return json.dumps({"projects": names})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.resource("castle://project/{name}/status")
def get_project_status(name: str) -> str:
    """Get pipeline status for a project."""
    try:
        from castle.service.project_service import get_project_info
        from castle.service.tracking_service import get_tracking_status

        info = get_project_info(_storage(), name)
        if info.get("error"):
            return json.dumps({"error": info["error"]})

        # Gather tracking status per video
        tracking = {}
        for vname in info["videos"]:
            tracking[vname] = get_tracking_status(_storage(), name, vname)

        return json.dumps({
            "name": name,
            "video_count": info["video_count"],
            "videos": info["videos"],
            "tracking": tracking,
            "latent_count": info["latent_count"],
        })
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.resource("castle://project/{name}/clusters")
def get_project_clusters(name: str) -> str:
    """Get cluster labels and annotations for a project."""
    try:
        from castle.service.annotation_service import load_annotations

        annotations = load_annotations(_storage(), name)

        # Also try to load id.csv for cluster tree
        cluster_meta = {}
        id_csv = os.path.join(_storage(), name, "cluster", "id.csv")
        if os.path.exists(id_csv):
            import csv
            with open(id_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cluster_meta[row["Id"]] = {
                        "name": row.get("Name", ""),
                        "color": row.get("Color", ""),
                    }

        return json.dumps({
            "cluster_meta": cluster_meta,
            "annotations": {
                k: v for k, v in annotations.items()
            },
        })
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.resource("castle://project/{name}/config")
def get_project_config(name: str) -> str:
    """Get project configuration."""
    try:
        from castle.core.project import get_project_config as _get_config

        _, config = _get_config(_storage(), name)
        return json.dumps(config)
    except Exception as exc:
        return json.dumps({"error": str(exc)})
