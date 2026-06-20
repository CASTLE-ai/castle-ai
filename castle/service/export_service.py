"""
castle/service/export_service.py
Shared file-collection helpers for project data export.

These functions are used by the Gradio UI (castle/ui/export_ui.py) to
collect (source_path, archive_name) tuples for packaging into a ZIP
archive.

No UI imports are allowed here — this is pure service-layer logic.
"""

import glob
import json
import logging
import os

logger = logging.getLogger(__name__)


def _collect_masks(project_path: str) -> list:
    """Return list of (src_path, archive_name) for all mask_list.h5 files.

    Args:
        project_path: Absolute path to the project directory.

    Returns:
        List of ``(src_path, archive_name)`` tuples.
    """
    pattern = os.path.join(project_path, "track", "*", "mask_list.h5")
    results = []
    for src in glob.glob(pattern):
        video_name = os.path.basename(os.path.dirname(src))
        results.append((src, os.path.join("track", video_name, "mask_list.h5")))
    return results


def _collect_latent(project_path: str) -> list:
    """Return list of (src_path, archive_name) for all latent feature files.

    Args:
        project_path: Absolute path to the project directory.

    Returns:
        List of ``(src_path, archive_name)`` tuples.
    """
    latent_dir = os.path.join(project_path, "latent")
    results = []
    if not os.path.isdir(latent_dir):
        return results
    for root, _dirs, files in os.walk(latent_dir):
        for f in files:
            src = os.path.join(root, f)
            rel = os.path.relpath(src, project_path)
            results.append((src, rel))
    return results


def _collect_cluster_results(project_path: str) -> list:
    """Return (src, archive_name) for cluster id.csv, cluster_*.npz, time_series_*.csv.

    Args:
        project_path: Absolute path to the project directory.

    Returns:
        List of ``(src_path, archive_name)`` tuples.
    """
    cluster_dir = os.path.join(project_path, "cluster")
    results = []
    if not os.path.isdir(cluster_dir):
        return results
    for pattern in ("id.csv", "cluster_*.npz", "time_series_*.csv"):
        for src in glob.glob(os.path.join(cluster_dir, pattern)):
            rel = os.path.relpath(src, project_path)
            results.append((src, rel))
    return results


def _collect_annotations(project_path: str, session_id: str) -> list:
    """Return (src, archive_name) for the selected session's annotations.csv.

    Args:
        project_path: Absolute path to the project directory.
        session_id: Session identifier string (may be empty/None).

    Returns:
        List of ``(src_path, archive_name)`` tuples (0 or 1 element).
    """
    if not session_id:
        return []
    src = os.path.join(
        project_path, "cluster", "sessions", session_id, "annotations.csv"
    )
    if not os.path.isfile(src):
        return []
    rel = os.path.relpath(src, project_path)
    return [(src, rel)]


def _collect_grid_videos(project_path: str) -> list:
    """Return (src, archive_name) for all grid videos (.mp4).

    Args:
        project_path: Absolute path to the project directory.

    Returns:
        List of ``(src_path, archive_name)`` tuples.
    """
    pattern = os.path.join(project_path, "cluster", "grid_videos", "*.mp4")
    results = []
    for src in glob.glob(pattern):
        rel = os.path.relpath(src, project_path)
        results.append((src, rel))
    return results


def _collect_analysis(project_path: str) -> list:
    """Return (src, archive_name) for analysis outputs (ethogram, metrics).

    Searches both ``analysis/`` and ``cluster/sessions/*/analysis/``.

    Args:
        project_path: Absolute path to the project directory.

    Returns:
        List of ``(src_path, archive_name)`` tuples.
    """
    results = []
    # Top-level analysis outputs
    analysis_dir = os.path.join(project_path, "analysis")
    if os.path.isdir(analysis_dir):
        for root, _dirs, files in os.walk(analysis_dir):
            for f in files:
                src = os.path.join(root, f)
                rel = os.path.relpath(src, project_path)
                results.append((src, rel))
    # Per-session analysis files
    sessions_dir = os.path.join(project_path, "cluster", "sessions")
    if os.path.isdir(sessions_dir):
        for sid in os.listdir(sessions_dir):
            sid_analysis = os.path.join(sessions_dir, sid, "analysis")
            if os.path.isdir(sid_analysis):
                for root, _dirs, files in os.walk(sid_analysis):
                    for f in files:
                        src = os.path.join(root, f)
                        rel = os.path.relpath(src, project_path)
                        results.append((src, rel))
    return results


def _collect_source_videos(project_path: str) -> list:
    """Return (src, archive_name) for all source video files.

    Args:
        project_path: Absolute path to the project directory.

    Returns:
        List of ``(src_path, archive_name)`` tuples.
    """
    sources_dir = os.path.join(project_path, "sources")
    results = []
    if not os.path.isdir(sources_dir):
        return results
    for root, _dirs, files in os.walk(sources_dir):
        for f in files:
            src = os.path.join(root, f)
            rel = os.path.relpath(src, project_path)
            results.append((src, rel))
    return results


def build_run_manifest(
    project_path: str,
    *,
    project_name: str,
    session_id: str = None,
    components=None,
    generated_at: str = None,
) -> dict:
    """Assemble a self-describing provenance manifest for an export bundle.

    A downloaded export otherwise carries no record of *how* it was produced.
    This manifest captures the CASTLE version, the full library/hardware stack
    (so a reproduction can tell cuML-GPU from sklearn-CPU embeddings apart), the
    selected components, the project inventory, and — if a clustering session is
    selected — that session's manifest. Everything is best-effort: missing or
    malformed inputs are skipped rather than raising, so writing the manifest can
    never fail an export.

    Args:
        project_path: Absolute path to the project directory.
        project_name: Project name.
        session_id: Optional clustering session id to embed its manifest.
        components: Iterable of selected component names (e.g. ['latent', ...]).
        generated_at: Optional ISO/stamp string for when the export was built.

    Returns:
        A JSON-serialisable dict.
    """
    from castle.core.environment import collect_run_environment

    manifest: dict = {
        "manifest_schema_version": 1,
        "generated_at": generated_at,
        "project_name": project_name,
        "components": sorted(components or []),
        "environment": collect_run_environment(),
    }

    cfg_path = os.path.join(project_path, "config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            manifest["project"] = {
                "videos": sorted(cfg.get("source", [])),
                "latent_count": len(cfg.get("latent", {})),
            }
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("run_manifest: could not read project config: %s", exc)

    if session_id:
        sm_path = os.path.join(
            project_path, "cluster", "sessions", session_id, "manifest.json"
        )
        if os.path.isfile(sm_path):
            try:
                with open(sm_path, encoding="utf-8") as f:
                    manifest["session"] = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("run_manifest: could not read session manifest: %s", exc)

    return manifest
