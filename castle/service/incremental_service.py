"""
castle/service/incremental_service.py
Incremental update support for CASTLE pipeline.

Provides helpers to detect which videos in a project still need feature
extraction, and to clean up orphaned latent / cluster data when source
videos are deleted.

Typical usage
-------------
# Before a batch run:
pending = get_unprocessed_videos(project_path)
for video in pending:
    run_extraction(video)

# After the user deletes some source videos:
removed = cleanup_deleted_videos(project_path)
print("Cleaned:", removed)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# File extensions recognised as video sources.
_VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_config(project_path: str) -> dict:
    """Load project config.json, returning an empty dict on failure."""
    config_path = os.path.join(project_path, "config.json")
    if not os.path.exists(config_path):
        logger.debug("No config.json found at %s", project_path)
        return {}
    try:
        with open(config_path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load config.json for %s: %s", project_path, exc)
        return {}


def _save_config(project_path: str, config: dict) -> None:
    """Persist *config* to config.json inside *project_path*."""
    config_path = os.path.join(project_path, "config.json")
    tmp_path = config_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)
        os.replace(tmp_path, config_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to save config.json: %s", exc)
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _list_source_videos(project_path: str) -> set:
    """Return the set of video filenames found in {project_path}/sources/."""
    sources_dir = os.path.join(project_path, "sources")
    if not os.path.isdir(sources_dir):
        return set()
    return {
        name
        for name in os.listdir(sources_dir)
        if os.path.splitext(name)[1].lower() in _VIDEO_EXTENSIONS
    }


def _find_latent_file(project_path: str, latent_filename: str) -> Optional[str]:
    """Search for *latent_filename* inside latent/ subdirectories.

    Returns the absolute path if found, *None* otherwise.
    """
    latent_root = os.path.join(project_path, "latent")
    if not os.path.isdir(latent_root):
        return None

    # Latents may live directly in latent/ or in latent/{model_name}/
    direct_path = os.path.join(latent_root, latent_filename)
    if os.path.exists(direct_path):
        return direct_path

    for entry in os.listdir(latent_root):
        sub = os.path.join(latent_root, entry)
        if os.path.isdir(sub):
            candidate = os.path.join(sub, latent_filename)
            if os.path.exists(candidate):
                return candidate

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_unprocessed_videos(project_path: str) -> list:
    """Return video names that do not yet have any cached latents.

    A video is considered "processed" when at least one entry in
    ``config['latent']`` maps to it (regardless of model or ROI).

    Parameters
    ----------
    project_path : str
        Absolute path to the project directory (must contain ``config.json``
        and a ``sources/`` sub-directory).

    Returns
    -------
    list of str
        Sorted list of source video filenames with no latent outputs yet.

    Example
    -------
    >>> pending = get_unprocessed_videos("/data/projects/my_project")
    >>> # ['animal_03.mp4', 'animal_07.mp4']
    """
    source_videos = _list_source_videos(project_path)
    if not source_videos:
        logger.info("get_unprocessed_videos: no source videos found in %s", project_path)
        return []

    config = _load_config(project_path)
    latent_map: dict = config.get("latent", {})

    # Build the set of video names that already have at least one latent
    processed: set = set(latent_map.values())

    unprocessed = sorted(source_videos - processed)
    logger.info(
        "get_unprocessed_videos: %d/%d videos unprocessed in %s",
        len(unprocessed),
        len(source_videos),
        project_path,
    )
    return unprocessed


def cleanup_deleted_videos(project_path: str) -> list:
    """Remove orphaned latent (and cluster) data for deleted source videos.

    A latent file is "orphaned" when its corresponding source video no longer
    exists in ``{project_path}/sources/``.  This function:

    1. Scans ``config['latent']`` for entries referencing missing videos.
    2. Deletes the corresponding ``.npz`` latent files from disk.
    3. Removes those entries from ``config['latent']`` and saves the config.
    4. Removes empty latent sub-directories.

    **Cluster files** stored under ``{project_path}/cluster/`` that begin with
    the base name of a deleted video are also removed.

    Parameters
    ----------
    project_path : str
        Absolute path to the project directory.

    Returns
    -------
    list of str
        Sorted list of source video names whose data was cleaned up.

    Example
    -------
    >>> removed = cleanup_deleted_videos("/data/projects/my_project")
    >>> # ['deleted_animal.mp4']
    """
    source_videos = _list_source_videos(project_path)
    config = _load_config(project_path)
    latent_map: dict = config.get("latent", {})

    if not latent_map:
        logger.info("cleanup_deleted_videos: no latent entries found in %s", project_path)
        return []

    keys_to_remove: list = []
    cleaned_videos: set = set()

    for latent_filename, video_name in list(latent_map.items()):
        if video_name in source_videos:
            continue  # video still exists — keep it

        # Video is gone; remove its latent file
        latent_path = _find_latent_file(project_path, latent_filename)
        if latent_path is not None:
            try:
                os.remove(latent_path)
                logger.info(
                    "cleanup_deleted_videos: removed latent %s (video=%s)",
                    latent_path,
                    video_name,
                )
            except OSError as exc:
                logger.warning(
                    "cleanup_deleted_videos: could not remove %s: %s",
                    latent_path,
                    exc,
                )
        else:
            logger.debug(
                "cleanup_deleted_videos: latent file %s not found on disk",
                latent_filename,
            )

        keys_to_remove.append(latent_filename)
        cleaned_videos.add(video_name)

    # Update config
    if keys_to_remove:
        for key in keys_to_remove:
            latent_map.pop(key, None)
        config["latent"] = latent_map
        _save_config(project_path, config)

    # Remove empty latent sub-directories
    _prune_empty_latent_dirs(project_path)

    # Clean up cluster files for deleted videos
    _cleanup_cluster_files(project_path, cleaned_videos)

    # Clean up cache manifest entries for deleted videos
    _invalidate_cache_entries(project_path, cleaned_videos)

    result = sorted(cleaned_videos)
    logger.info(
        "cleanup_deleted_videos: cleaned %d video(s) from %s: %s",
        len(result),
        project_path,
        result,
    )
    return result


# ---------------------------------------------------------------------------
# Private cleanup helpers
# ---------------------------------------------------------------------------

def _prune_empty_latent_dirs(project_path: str) -> None:
    """Remove empty directories under {project_path}/latent/."""
    latent_root = os.path.join(project_path, "latent")
    if not os.path.isdir(latent_root):
        return

    for entry in os.listdir(latent_root):
        sub = os.path.join(latent_root, entry)
        if os.path.isdir(sub):
            try:
                # Only removes if truly empty (no files, no subdirs)
                os.rmdir(sub)
                logger.debug("cleanup: removed empty latent dir %s", sub)
            except OSError:
                pass  # Not empty — leave it alone


def _cleanup_cluster_files(project_path: str, deleted_videos: set) -> None:
    """Remove cluster files associated with *deleted_videos*.

    Cluster files are heuristically matched by checking whether the filename
    starts with the base name (stem) of a deleted video.
    """
    if not deleted_videos:
        return

    cluster_root = os.path.join(project_path, "cluster")
    if not os.path.isdir(cluster_root):
        return

    deleted_stems = {os.path.splitext(v)[0] for v in deleted_videos}

    for fname in os.listdir(cluster_root):
        for stem in deleted_stems:
            if fname.startswith(stem):
                fpath = os.path.join(cluster_root, fname)
                try:
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                        logger.info(
                            "cleanup: removed cluster file %s (stem=%s)", fpath, stem
                        )
                except OSError as exc:
                    logger.warning("cleanup: could not remove %s: %s", fpath, exc)
                break


def _invalidate_cache_entries(project_path: str, deleted_videos: set) -> None:
    """Remove PipelineCache manifest entries for *deleted_videos*.

    Reads {project_path}/latent/.cache_manifest.json (if present) and removes
    entries whose stored output path starts with the project path or matches a
    deleted video name.
    """
    if not deleted_videos:
        return

    from castle.core.cache import PipelineCache  # local import to avoid circular deps

    latent_dir = os.path.join(project_path, "latent")
    if not os.path.isdir(latent_dir):
        return

    try:
        cache = PipelineCache(latent_dir)
    except Exception as exc:  # noqa: BLE001
        logger.debug("_invalidate_cache_entries: could not open cache: %s", exc)
        return

    deleted_stems = {os.path.splitext(v)[0] for v in deleted_videos}
    keys_to_remove = []

    for key, output_path in list(cache._manifest.items()):
        basename = os.path.basename(output_path)
        for stem in deleted_stems:
            if basename.startswith(stem):
                keys_to_remove.append(key)
                break

    for key in keys_to_remove:
        cache.invalidate(key)

    if keys_to_remove:
        logger.info(
            "_invalidate_cache_entries: invalidated %d cache key(s) for %s",
            len(keys_to_remove),
            deleted_videos,
        )
