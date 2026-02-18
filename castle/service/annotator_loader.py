"""
castle/service/annotator_loader.py
Lightweight loader for cluster data — decoupled from LatentAggregator.

Reads cluster files (id.csv, cluster_.npz) and project config directly
from disk without requiring the clustering workflow to have been run in
the current session.
"""

import os
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from castle.core.project import get_project_config
from castle.utils.video_io import VideoReader

logger = logging.getLogger(__name__)


# ---------------------------
# Data Classes
# ---------------------------

@dataclass
class AnnotatorData:
    """Lightweight cluster data for the Annotator, loaded from files.

    Attributes:
        cluster: 1-D integer array of cluster assignments per bin, from 'cls' NPZ key.
        cluster_meta: Mapping of cluster_id → {'name': str, 'color': str}.
        embedding: 2-D float array of UMAP embeddings, from 'emb' NPZ key.
        bin_size: Number of video frames per temporal bin.
        project_path: Absolute path to the project directory.
        source_path: Absolute path to the project's sources/ directory.
        videos_meta: List of (n_bins, video_name) tuples.
        fps: Frames per second from project videos.
    """

    cluster: np.ndarray
    cluster_meta: Dict[int, Dict[str, str]]
    embedding: np.ndarray
    bin_size: int
    project_path: str
    source_path: str
    videos_meta: List[Tuple[int, str]]
    fps: float

    # Internal cache — excluded from repr / comparisons
    _reader_cache: Dict[str, VideoReader] = field(
        default_factory=dict, compare=False, repr=False
    )
    _cache_lock: threading.Lock = field(
        default_factory=threading.Lock, compare=False, repr=False
    )


# ---------------------------
# Loader
# ---------------------------

def load_annotator_data(
    storage_path: str,
    project_name: str,
    session_id: Optional[str] = None,
) -> AnnotatorData:
    """Load cluster data from disk files.

    Reads:
    - ``cluster/id.csv``          → cluster_meta
    - ``cluster/cluster_.npz``    → cls array + emb array
    - project ``config.json``     → videos_meta (video names), fps
    - session manifest            → bin_size (falls back to 1)

    If *session_id* is provided the session is first activated (its files
    are copied to the cluster/ root) via :class:`SessionManager`.

    Args:
        storage_path: Root storage directory.
        project_name: Name of the project.
        session_id: Optional session ID to activate before loading.

    Returns:
        :class:`AnnotatorData` populated from disk.

    Raises:
        FileNotFoundError: If required files are missing.
    """
    import pandas as pd

    project_path = os.path.join(storage_path, project_name)
    cluster_path = os.path.join(project_path, "cluster")
    source_path = os.path.join(project_path, "sources")

    # --- Activate session if requested ---
    bin_size = 1
    if session_id is not None:
        from castle.service.session_manager import SessionManager

        sm = SessionManager(storage_path, project_name)
        info = sm.activate_session(session_id)
        if info is not None:
            bin_size = int(info.bin_size)
        else:
            logger.warning("Session '%s' not found — loading from cluster/ root", session_id)
    else:
        # Try to read bin_size from the active session manifest
        from castle.service.session_manager import SessionManager

        sm = SessionManager(storage_path, project_name)
        active_id = sm.get_active_session_id()
        if active_id:
            info = sm.get_session(active_id)
            if info is not None:
                bin_size = int(info.bin_size)

    # --- Load cluster_meta from id.csv ---
    id_csv_path = os.path.join(cluster_path, "id.csv")
    if not os.path.exists(id_csv_path):
        raise FileNotFoundError(f"id.csv not found: {id_csv_path}")

    id_df = pd.read_csv(id_csv_path)
    cluster_meta: Dict[int, Dict[str, str]] = {}
    for _, row in id_df.iterrows():
        cid = int(row["Id"])
        cluster_meta[cid] = {
            "name": str(row["Name"]),
            "color": str(row["Color"]),
        }

    # --- Load cluster array + embedding from cluster_.npz ---
    npz_path = os.path.join(cluster_path, "cluster_.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"cluster_.npz not found: {npz_path}")

    npz = np.load(npz_path, allow_pickle=False)
    cls_array: np.ndarray = npz["cls"].astype(np.int32)
    emb_array: np.ndarray = npz["emb"].astype(np.float64)

    # --- Load project config for videos_meta and fps ---
    _, config = get_project_config(storage_path, project_name)

    videos_meta: List[Tuple[int, str]] = []
    fps: float = 30.0

    latent_info = config.get("latent", {})
    if latent_info:
        # Build videos_meta: derive n_bins from cls array length and bin_size.
        # We only know the total bins; distribute per video proportionally using
        # latent file sizes when available, otherwise treat as a single video.
        total_bins = len(cls_array)

        # Collect unique source video names (in order)
        seen: List[str] = []
        for _latent_fname, video_name in latent_info.items():
            if video_name not in seen:
                seen.append(video_name)

        if len(seen) == 1:
            # Single video: all bins belong to it
            videos_meta = [(total_bins, seen[0])]

            # Derive bin_size from actual video frame count / n_bins
            first_video_path = os.path.join(source_path, seen[0])
            if os.path.exists(first_video_path) and total_bins > 0:
                try:
                    with VideoReader(first_video_path) as vr:
                        n_video_frames = len(vr)
                        derived_bin_size = max(1, n_video_frames // total_bins)
                        if derived_bin_size != bin_size:
                            logger.info(
                                "Overriding manifest bin_size=%d with derived=%d "
                                "(video=%d frames, cluster=%d bins)",
                                bin_size, derived_bin_size, n_video_frames, total_bins,
                            )
                            bin_size = derived_bin_size
                except Exception as exc:
                    logger.warning("Could not derive bin_size from video: %s", exc)
        else:
            # Multiple videos: try to derive n_bins per video from latent files
            latent_dir = os.path.join(project_path, "latent")
            video_bins: Dict[str, int] = {}
            for latent_fname, video_name in latent_info.items():
                # Try to find the latent file in any subdirectory
                for dirpath, _dirs, files in os.walk(latent_dir):
                    if latent_fname in files:
                        try:
                            lat_data = np.load(
                                os.path.join(dirpath, latent_fname),
                                allow_pickle=False,
                            )
                            n_frames = lat_data["latent"].shape[0]
                            n_bins_v = n_frames // bin_size
                            video_bins[video_name] = (
                                video_bins.get(video_name, 0) + n_bins_v
                            )
                        except Exception:
                            pass
                        break

            if video_bins:
                videos_meta = [(video_bins[v], v) for v in seen if v in video_bins]
            else:
                # Fallback: divide bins equally
                n_per = total_bins // len(seen)
                videos_meta = [(n_per, v) for v in seen]

        # Read fps from the first available video
        if seen:
            first_video_path = os.path.join(source_path, seen[0])
            if os.path.exists(first_video_path):
                try:
                    with VideoReader(first_video_path) as vr:
                        fps = vr.fps
                except Exception as exc:
                    logger.warning("Could not read fps from %s: %s", first_video_path, exc)
    else:
        # No latent info in config: treat the whole cls array as one unnamed video
        logger.warning("No 'latent' key in project config; videos_meta will be empty.")

    return AnnotatorData(
        cluster=cls_array,
        cluster_meta=cluster_meta,
        embedding=emb_array,
        bin_size=bin_size,
        project_path=project_path,
        source_path=source_path,
        videos_meta=videos_meta,
        fps=fps,
    )


# ---------------------------
# Frame Retrieval
# ---------------------------

def get_annotator_frame(
    annotator_data: AnnotatorData,
    bin_idx: int,
) -> Optional[np.ndarray]:
    """Return the representative video frame for a given global bin index.

    Mirrors the logic of :meth:`LatentAggregator.get_frame`: iterates
    ``videos_meta``, finds the correct video, then computes::

        frame_idx = bin_idx * bin_size + bin_size // 2

    Uses a simple per-:class:`AnnotatorData` VideoReader cache (up to 3
    readers) to avoid re-opening files on every call.

    Args:
        annotator_data: Loaded :class:`AnnotatorData` instance.
        bin_idx: Global bin index (0-based).

    Returns:
        Frame as ``(H, W, 3)`` uint8 numpy array, or *None* on failure.
    """
    if not annotator_data.videos_meta:
        logger.warning("videos_meta is empty — cannot retrieve frame")
        return None

    remaining = int(bin_idx)
    bin_size = annotator_data.bin_size

    for n_bins_in_video, video_name in annotator_data.videos_meta:
        if remaining >= n_bins_in_video:
            remaining -= n_bins_in_video
            continue

        # Found the video that contains this bin
        video_path = os.path.join(annotator_data.source_path, video_name)
        frame_idx = remaining * bin_size + bin_size // 2

        try:
            reader = _get_cached_reader(annotator_data, video_path)
            return reader.get_frame(frame_idx)
        except Exception as exc:
            logger.error(
                "Error reading frame %d from '%s': %s", frame_idx, video_name, exc
            )
            return None

    logger.error("bin_idx %d is out of range (total bins: %d)", bin_idx, len(annotator_data.cluster))
    return None


def _get_cached_reader(annotator_data: AnnotatorData, video_path: str) -> VideoReader:
    """Return a cached VideoReader, evicting the oldest entry when full.

    Cache is stored directly on the :class:`AnnotatorData` instance and is
    protected by a threading lock.
    """
    cache = annotator_data._reader_cache
    lock = annotator_data._cache_lock
    max_size = 3

    with lock:
        if video_path in cache:
            # Move to end (most recently used) — use pop/re-insert trick
            reader = cache.pop(video_path)
            cache[video_path] = reader
            return reader

        # Evict oldest if full
        if len(cache) >= max_size:
            oldest_key = next(iter(cache))
            old_reader = cache.pop(oldest_key)
            try:
                old_reader.close()
            except Exception:
                pass

        reader = VideoReader(video_path)
        cache[video_path] = reader
        return reader
