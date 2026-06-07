"""
castle/service/annotator_loader.py
Lightweight loader for cluster data — decoupled from LatentAggregator.

Reads cluster files (id.csv, time_series_*.csv, cluster_.npz) and project
config directly from disk without requiring the clustering workflow to have
been run in the current session.
"""

import glob
import os
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from castle.core.prepare import WindowedFrameIndexMap
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
        cluster: 1-D integer array of cluster assignments per bin.
        cluster_meta: Mapping of cluster_id → {'name': str, 'color': str}.
        embedding: 2-D float array of UMAP embeddings, from 'emb' NPZ key.
        bin_size: Number of video frames per temporal bin.
        project_path: Absolute path to the project directory.
        source_path: Absolute path to the project's sources/ directory.
        videos_meta: List of (n_bins, video_name) tuples.
        fps: Frames per second from project videos.
        session_id: Optional session ID used to scope annotations on disk.
    """

    cluster: np.ndarray
    cluster_meta: Dict[int, Dict[str, str]]
    embedding: np.ndarray
    bin_size: int
    project_path: str
    source_path: str
    videos_meta: List[Tuple[int, str]]
    fps: float
    session_id: Optional[str] = None
    # Window-aware datapoint<->original-frame map (legacy or prepared); used by
    # get_annotator_frame / bout grid so both paths share one mapping.
    frame_index_map: Optional[WindowedFrameIndexMap] = None

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
) -> "AnnotatorData":
    """Load cluster data from disk files.

    Reads:
    - ``cluster/id.csv``                  → cluster_meta
    - ``cluster/time_series_*.csv``       → cluster array + videos_meta (primary)
    - ``cluster/cluster_.npz``            → emb array; cls fallback if no CSVs
    - project ``config.json``             → video order, fps
    - session manifest                    → bin_size (falls back to 1)

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
    prepare_id = None  # set from the session manifest; non-None => prepared cache
    if session_id is not None:
        from castle.service.session_manager import SessionManager

        sm = SessionManager(storage_path, project_name)
        info = sm.activate_session(session_id)
        if info is not None:
            bin_size = int(info.bin_size)
            prepare_id = getattr(info, "prepare_id", None)
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
                prepare_id = getattr(info, "prepare_id", None)
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

    # --- Load embedding from cluster_{emb_name}.npz (kept for visualization) ---
    npz_candidates = [
        f for f in glob.glob(os.path.join(cluster_path, "cluster_*.npz"))
        if not f.endswith("cluster_data.npz")
    ]
    if not npz_candidates:
        raise FileNotFoundError(f"cluster_*.npz not found: {cluster_path}")
    npz_path = npz_candidates[0]

    npz = np.load(npz_path, allow_pickle=False)
    emb_array: np.ndarray = npz["emb"].astype(np.float64)

    # --- Load project config for video order and fps ---
    _, config = get_project_config(storage_path, project_name)

    videos_meta: List[Tuple[int, str]] = []
    fps: float = 30.0

    latent_info = config.get("latent", {})

    # Collect unique source video names in project config order
    seen: List[str] = []
    for _latent_fname, video_name in latent_info.items():
        if video_name not in seen:
            seen.append(video_name)

    # --- Try to build cluster array from time_series_*.csv files ---
    cls_array = _load_cluster_from_time_series(
        cluster_path=cluster_path,
        video_names=seen,
        bin_size=bin_size,
        videos_meta_out=videos_meta,  # mutated in-place
    )

    if cls_array is not None:
        # Downsample embedding to match cluster array if sizes differ
        if len(emb_array) != len(cls_array) and len(emb_array) > 0 and len(cls_array) > 0:
            ratio = len(emb_array) / len(cls_array)
            if abs(ratio - round(ratio)) < 0.01:
                step = int(round(ratio))
                emb_array = emb_array[::step][:len(cls_array)]
                logger.info(
                    "Downsampled embedding %d → %d (step=%d) to match cluster array",
                    len(emb_array) * step, len(emb_array), step,
                )
        logger.info(
            "Loaded cluster array from time_series CSVs: %d bins total", len(cls_array)
        )
    else:
        # --- Fallback: read cls array from cluster_.npz ---
        logger.info("No time_series CSVs found; falling back to cluster_.npz cls array")
        cls_array = npz["cls"].astype(np.int32)
        total_bins = len(cls_array)

        # Rebuild videos_meta from npz cls + latent info
        if len(seen) == 1:
            videos_meta.clear()
            videos_meta.append((total_bins, seen[0]))

            # Derive bin_size from actual frame count when possible
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
                                bin_size,
                                derived_bin_size,
                                n_video_frames,
                                total_bins,
                            )
                            bin_size = derived_bin_size
                except Exception as exc:
                    logger.warning("Could not derive bin_size from video: %s", exc)
        elif seen:
            latent_dir = os.path.join(project_path, "latent")
            video_bins: Dict[str, int] = {}
            for latent_fname, video_name in latent_info.items():
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
                        except Exception as exc:
                            logger.warning(
                                "Could not read latent file %s for bin count: %s",
                                latent_fname, exc
                            )
                        break

            videos_meta.clear()
            if video_bins:
                videos_meta.extend(
                    (video_bins[v], v) for v in seen if v in video_bins
                )
            else:
                n_per = total_bins // len(seen)
                videos_meta.extend((n_per, v) for v in seen)
        else:
            logger.warning("No 'latent' key in project config; videos_meta will be empty.")

    # --- Read fps from the first available video ---
    if seen:
        first_video_path = os.path.join(source_path, seen[0])
        if os.path.exists(first_video_path):
            try:
                with VideoReader(first_video_path) as vr:
                    fps = vr.fps
            except Exception as exc:
                logger.warning("Could not read fps from %s: %s", first_video_path, exc)

    # --- Datapoint<->original-frame map (+ prepared-session override) ---
    frame_index_map = None
    if prepare_id:
        try:
            import pandas as pd
            from castle.core.prepare import load_prepare
            _pd = load_prepare(os.path.join(cluster_path, "prepared", prepare_id))
            base_map = _pd.index_map
            frame_index_map = base_map.for_window(max(1, int(bin_size)))
            # Prepared datapoints are decimated windows, not uniform bins. The
            # npz only stores per-submit LOCAL labels, so recover per-window
            # GLOBAL labels from the authoritative original-frame time_series CSVs
            # (sampled through the window map); keep the npz emb for bout ranking.
            emb_array = npz["emb"].astype(np.float64)
            seg = []
            vmeta: List[Tuple[int, str]] = []
            for v in range(base_map.n_videos):
                name = base_map.video_names[v]
                nwin = int(frame_index_map.n_windows_per_video[v])
                base_name = os.path.splitext(os.path.basename(name))[0]
                ts_path = os.path.join(cluster_path, f"time_series_{base_name}.csv")
                if os.path.exists(ts_path):
                    beh = pd.read_csv(ts_path)["behavior"].values
                    wl = frame_index_map.windowed_labels_from_orig(beh, v)
                else:
                    wl = np.full(nwin, -1, dtype=np.int64)
                seg.append(np.asarray(wl, dtype=np.int32))
                vmeta.append((nwin, name))
            cls_array = np.concatenate(seg) if seg else np.array([], dtype=np.int32)
            videos_meta = vmeta
            if base_map.n_videos:
                fps = float(base_map.raw_fps[0])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Prepared annotator load failed (%s); using legacy reconstruction.", exc)
            frame_index_map = None
    if frame_index_map is None:
        from castle.core.prepare import build_legacy_index_map
        frame_index_map = build_legacy_index_map(videos_meta, bin_size).for_window(1)

    return AnnotatorData(
        cluster=cls_array,
        cluster_meta=cluster_meta,
        embedding=emb_array,
        bin_size=bin_size,
        project_path=project_path,
        source_path=source_path,
        videos_meta=videos_meta,
        fps=fps,
        session_id=session_id,
        frame_index_map=frame_index_map,
    )


def _load_cluster_from_time_series(
    cluster_path: str,
    video_names: List[str],
    bin_size: int,
    videos_meta_out: List[Tuple[int, str]],
) -> Optional[np.ndarray]:
    """Build a flat cluster assignment array from per-video time_series CSVs.

    Each ``time_series_<basename>.csv`` has one row per *frame* with a
    ``behavior`` column containing the leaf cluster ID.  Down-sampling by
    *bin_size* converts it to per-bin values.

    Args:
        cluster_path: Directory containing the time_series CSV files.
        video_names: Ordered list of video filenames (from project config).
        bin_size: Number of frames per temporal bin.
        videos_meta_out: Output list that is populated with
            ``(n_bins, video_name)`` tuples (mutated in-place).

    Returns:
        Concatenated int32 array of cluster IDs per bin, or *None* if no
        CSV files were found for any of the listed videos.
    """
    import pandas as pd

    segments: List[np.ndarray] = []
    found_any = False

    for video_name in video_names:
        basename = os.path.splitext(os.path.basename(video_name))[0]
        ts_path = os.path.join(cluster_path, f"time_series_{basename}.csv")

        if not os.path.exists(ts_path):
            logger.warning("time_series CSV not found for %s at %s", video_name, ts_path)
            continue

        found_any = True
        ts_df = pd.read_csv(ts_path)

        if "behavior" not in ts_df.columns:
            logger.error("'behavior' column missing in %s — skipping", ts_path)
            continue

        # Down-sample: one value per bin (take first frame of each bin)
        per_bin = ts_df["behavior"].values[::bin_size].astype(np.int32)
        n_bins = len(per_bin)

        videos_meta_out.append((n_bins, video_name))
        segments.append(per_bin)
        logger.debug(
            "Loaded %s: %d frames → %d bins (bin_size=%d)",
            basename,
            len(ts_df),
            n_bins,
            bin_size,
        )

    if not found_any:
        return None

    return np.concatenate(segments) if segments else np.array([], dtype=np.int32)


# ---------------------------
# Frame Retrieval
# ---------------------------

def get_annotator_frame(
    annotator_data: "AnnotatorData",
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

    # Prefer the FrameIndexMap (legacy bin centre OR prepared decimated-window
    # centre); fall back to the old bin arithmetic if it is somehow absent.
    fim = annotator_data.frame_index_map
    if fim is not None:
        try:
            video_idx, frame_idx = fim.dp_to_orig_frame(int(bin_idx))
            video_name = fim.base.video_names[video_idx]
        except (IndexError, ValueError):
            logger.warning("bin_idx %d out of range for frame map", bin_idx)
            return None
        video_path = os.path.join(annotator_data.source_path, video_name)
        try:
            reader = _get_cached_reader(annotator_data, video_path)
            return reader.get_frame(frame_idx)
        except Exception as exc:
            logger.warning("Frame read failed for %s[%d]: %s", video_name, frame_idx, exc)
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

    logger.error(
        "bin_idx %d is out of range (total bins: %d)",
        bin_idx,
        len(annotator_data.cluster),
    )
    return None


def _get_cached_reader(annotator_data: "AnnotatorData", video_path: str) -> VideoReader:
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
