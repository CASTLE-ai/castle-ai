"""
castle/core/cluster.py
Core clustering logic and data aggregation.
"""

import os
import threading
from collections import OrderedDict
import numpy as np
from typing import List, Tuple, Dict, Optional

from castle.core.interfaces import NotificationCallback
from castle.core.logging_config import setup_logger
from castle.core.types import LatentCorruptError
from castle.utils.video_io import VideoReader
from castle.utils.safe_load import load_latent_safe
from castle.utils.latent_metadata import load_latent_metadata
from castle.core.project import get_project_config
from castle.utils.latent_explorer import Latent

logger = setup_logger(__name__)

# PERF-03: VideoReader pool size (was 3; 8 covers the "click clusters from
# different videos" UX without thrashing).
_VIDEO_READER_CACHE_MAX = 8
# Frame LRU upper bound — ~256 frames × ~270 KB each → < 100 MB ceiling.
_FRAME_CACHE_MAX = 256

# PERF-01: aggregated latent memmap threshold (bytes). Override via env var
# CASTLE_MEMMAP_THRESHOLD_GB. Default sourced from castle.defaults.MEMMAP_THRESHOLD_GB.
from castle.defaults import MEMMAP_THRESHOLD_GB as _DEFAULT_MEMMAP_THRESHOLD_GB

# ---------------------------
# Helper Functions
# ---------------------------

def frame_to_timestamp(frame_number: int, fps: float) -> str:
    """Convert frame number to a SubRip ``HH:MM:SS,mmm`` timestamp.

    BUG-15: the old implementation ran ``seconds = frame_number / fps``
    and chained float ``//`` / ``%`` operations to extract HMS+ms. On
    long videos that compounds float rounding so the SRT clock can drift
    by up to one millisecond per hour and ``int(59.99999...)`` rolls
    backwards to 59. This version stays in integer microseconds the
    whole way, which is exact for any (frame, fps) pair representable
    in IEEE-754 doubles.

    Args:
        frame_number: 0-indexed frame.
        fps: Frame rate (frames per second). Must be > 0.

    Returns:
        ``HH:MM:SS,mmm`` string (SRT convention; comma between seconds
        and milliseconds).
    """
    total_us = int(round(frame_number * 1_000_000 / fps))
    hours, rem = divmod(total_us, 3600 * 1_000_000)
    minutes, rem = divmod(rem, 60 * 1_000_000)
    seconds, micro = divmod(rem, 1_000_000)
    millis = micro // 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def _resolve_latent_path(latent_dir_path: str, config_key: str) -> str:
    """Resolve a ``config['latent']`` key to a physical ``.npz`` path.

    Pre-process (KIT) session latents are registered under a *logical*
    ``"{session_id}/{filename}"`` key — the prefix lets
    ``delete_session_with_latent_cleanup`` find a session's latents — but the
    file itself is written *flat* in ``latent/{model}/`` with a disambiguating
    ``_pre-{session_id}`` suffix in the filename. Joining the prefixed key onto
    the latent dir therefore points at a phantom ``latent/{model}/{session_id}/``
    sub-directory that does not exist (this is what surfaced as
    "Latent file missing" / "No latents loaded").

    Try the key as a relative path first (legacy / any future sub-dir layout),
    then fall back to the flat basename so prefixed keys resolve correctly.
    """
    direct = os.path.join(latent_dir_path, config_key)
    if os.path.exists(direct):
        return direct
    return os.path.join(latent_dir_path, os.path.basename(config_key))


def find_nearest_embedding(embedding_data: np.ndarray, x: float, y: float, tree=None) -> Tuple[int, float]:
    """
    Find the nearest point in embedding space using KDTree.
    
    Args:
        embedding_data: 2D embedding array of shape (N, 2)
        x: Query x coordinate
        y: Query y coordinate
        tree: Optional pre-built KDTree (avoids rebuilding each call)
    
    Returns:
        Tuple of (index, distance) to nearest point
    """
    from scipy.spatial import KDTree
    if tree is None:
        tree = KDTree(embedding_data)
    distance, index = tree.query((x, y))
    return int(index), float(distance)

def _memmap_threshold_bytes() -> int:
    """Resolve the memmap threshold in bytes from env var.

    Honors ``CASTLE_MEMMAP_THRESHOLD_GB`` (float, GiB). Defaults to 2 GiB.
    """
    raw = os.environ.get("CASTLE_MEMMAP_THRESHOLD_GB")
    try:
        gb = float(raw) if raw else _DEFAULT_MEMMAP_THRESHOLD_GB
    except ValueError:
        gb = _DEFAULT_MEMMAP_THRESHOLD_GB
    return int(gb * (1024 ** 3))


def _aggregate_latents(
    chunks: List[np.ndarray],
    *,
    cache_dir: str,
    notify: NotificationCallback,
    memmap_filename: str = 'aggregated_latents.dat',
) -> np.ndarray:
    """Concatenate latent chunks into one tall array.

    Behaviour:

    - Pre-allocate ``np.empty((sum(T_i), F))`` once and copy each chunk in
      place — peak RSS is ~1x the final size instead of ~2x for
      ``np.concatenate``.
    - When the total exceeds :func:`_memmap_threshold_bytes`, fall back to a
      disk-backed ``np.memmap`` under ``cache_dir`` so very large projects
      don't OOM. The memmap is reusable across sessions and is overwritten
      each time the aggregator runs.

    Args:
        chunks: Non-empty list of 2D arrays sharing a feature dimension.
        cache_dir: Directory to write the memmap to (created on demand).
        notify: User-facing notification callback for progress messages.
        memmap_filename: Override for the memmap file name; defaults to
            ``aggregated_latents.dat``.  Pass a per-aggregator unique name
            (e.g. embedding (model, roi, bin) hash) when multiple
            ``LatentAggregator`` instances may run concurrently and would
            otherwise overwrite each other's memmap.

    Returns:
        Aggregated ``(total_T, F)`` array. For the memmap path the returned
        object is an ``np.memmap``; downstream UMAP/DBSCAN treat it as a
        normal ndarray.
    """
    if not chunks:
        raise ValueError("Cannot aggregate empty chunk list")

    feature_dim = chunks[0].shape[1]
    total_T = sum(c.shape[0] for c in chunks)
    dtype = chunks[0].dtype
    total_bytes = total_T * feature_dim * np.dtype(dtype).itemsize
    threshold_bytes = _memmap_threshold_bytes()

    if total_bytes > threshold_bytes:
        os.makedirs(cache_dir, exist_ok=True)
        memmap_path = os.path.join(cache_dir, memmap_filename)
        notify(
            f"Aggregated latents {total_bytes / 1e9:.2f} GB exceed "
            f"memmap threshold {threshold_bytes / 1e9:.2f} GB; "
            f"falling back to disk-backed memmap at {memmap_path}.",
            "warning",
        )
        out = np.memmap(memmap_path, dtype=dtype, mode='w+',
                        shape=(total_T, feature_dim))
    else:
        out = np.empty((total_T, feature_dim), dtype=dtype)

    offset = 0
    for chunk in chunks:
        n = chunk.shape[0]
        out[offset:offset + n] = chunk
        offset += n
    # Drop references so Python can reclaim the chunk arrays immediately.
    chunks.clear()

    return out


def auto_generate_cluster_name(parent_name, cluster_id):
    """Auto-generate a hierarchical cluster name based on parent name and cluster ID.
    
    Naming convention: parent_name + next_level_letter + cluster_id
    e.g., root_a0, root_a0_b1, root_a0_b1_c2
    """
    import re
    if parent_name is None:
        parent_name = "root"
    
    # Try to find the last level designator (single letter followed by digits at end)
    match = re.search(r'_([a-z])(\d+)$', parent_name)
    if match:
        last_char = match.group(1)
        next_char = chr(ord(last_char) + 1)
    else:
        next_char = 'a'
        
    return f"{parent_name}_{next_char}{cluster_id}"


# ---------------------------
# Core Class: LatentAggregator
# ---------------------------

class LatentAggregator:
    """
    Aggregates latent features from multiple video files in a project.
    
    This class replaces the legacy MultiVideos class and provides:
    - Loading and concatenation of latent files across videos
    - Frame retrieval by global bin index
    - Subtitle generation for clustered behaviors
    
    Attributes:
        latents: Aggregated latent array of shape (N, feature_dim)
        videos_meta: List of (n_bins, video_name) tuples
        fps: Frames per second from first loaded video
        bin_size: Number of frames per bin
    """
    def __init__(self, storage_path: str, project_name: str, select_roi_id: int, bin_size: int, 
                 model_name: str, notify: Optional[NotificationCallback] = None) -> None:
        """
        Initialize the LatentAggregator.
        
        Args:
            storage_path: Root storage directory path
            project_name: Name of the project
            select_roi_id: ROI ID to filter latent files
            bin_size: Number of frames per temporal bin
            model_name: Name of the model to load latents for
            notify: Optional callback for progress/status notifications
        """
        self.storage_path = storage_path
        self.project_name = project_name
        self.source_path = os.path.join(storage_path, project_name, 'sources')
        self.project_path = os.path.join(storage_path, project_name)
        self.bin_size = int(bin_size)
        self.model_name = model_name
        self.notify = notify or print  # Fallback to print
        
        # C-02 / PERF-03: VideoReader LRU cache — keeps N most-recently-used
        # readers open. Default 8 covers "click cluster from many videos" UX
        # without thrashing.
        self._video_reader_cache: "OrderedDict[str, VideoReader]" = OrderedDict()
        self._cache_max_size: int = _VIDEO_READER_CACHE_MAX
        # PERF-03: per-frame LRU cache so repeated hovers don't re-decode.
        self._frame_cache: "OrderedDict[Tuple[str, int], np.ndarray]" = OrderedDict()
        self._frame_cache_max: int = _FRAME_CACHE_MAX
        # RLock so _get_cached_frame can hold the lock across the full
        # check→fetch→insert sequence (3-E) while still calling helpers
        # like _get_cached_reader that re-acquire the same lock.
        self._cache_lock = threading.RLock()
        
        # Load project configuration
        project_path, project_config = get_project_config(storage_path, project_name)
        self.project_path = project_path
        # Remember the per-aggregator key params so the memmap filename is
        # unique per (model, roi, bin) tuple — prevents two concurrent
        # LatentAggregators (e.g. two Gradio sessions clustering different
        # ROIs) from overwriting each other's aggregated_latents.dat (3-D).
        self.select_roi_id = int(select_roi_id)

        # Filter latents for the selected ROI
        roi_key = f'ROI_{select_roi_id}'
        
        # Latent files are stored in model-specific subdirectories
        latent_dir_path = os.path.join(storage_path, project_name, 'latent', model_name)
        
        self.latents: Optional[np.ndarray] = None
        self.videos_meta: List[Tuple[int, str]] = []
        self.fps: float = 30.0 # Default fallback (first video; see fps_per_video)
        # Per-video frame rate: a project may mix frame rates (e.g. 24 + 60 fps).
        # Keyed by video source name (the same string stored in videos_meta).
        self.fps_per_video: Dict[str, float] = {}
        
        latent_files = []
        if 'latent' in project_config:
            for filename, video_source_name in project_config['latent'].items():
                # Check 1: Must match ROI ID
                if roi_key not in filename:
                    continue
                
                # Check 2: Match Model Name. BUG-14 — prefer the metadata
                # embedded in the npz (or its sidecar .json) over fragile
                # filename splitting. Fall back to the old stem-split when
                # the npz predates the metadata helper.
                latent_file_path = _resolve_latent_path(latent_dir_path, filename)
                model_name_matches = False
                if os.path.exists(latent_file_path):
                    meta = load_latent_metadata(latent_file_path)
                    if meta is not None and meta.get("model_name") == model_name:
                        model_name_matches = True
                    elif meta is not None and meta.get("roi_id") is not None:
                        # Metadata present but model mismatch → definitively skip.
                        continue
                if not model_name_matches:
                    stem = os.path.splitext(os.path.basename(filename))[0]
                    stem_after_roi = stem.split(f'_ROI_{select_roi_id}_', 1)
                    legacy_match = (
                        len(stem_after_roi) > 1
                        and (stem_after_roi[1] == model_name
                             or stem_after_roi[1].startswith(model_name + '_'))
                    )
                    if not legacy_match and not os.path.exists(latent_file_path):
                        continue
                    # Legacy file in model-specific directory — accept it.
                    model_name_matches = legacy_match or os.path.exists(latent_file_path)

                if model_name_matches:
                    latent_files.append((filename, video_source_name))
        
        total_frames_loaded = 0
        latents_buffer: List[np.ndarray] = []  # Buffer for pre-alloc / memmap fill

        # Load and aggregate latents
        for filename, video_source_name in latent_files:
            self.notify(f'Loading latent: {video_source_name}')
            try:
                latent_path = _resolve_latent_path(latent_dir_path, filename)
                if not os.path.exists(latent_path):
                    self.notify(f"Latent file missing: {latent_path}", "warning")
                    continue

                # BUG-10: typed corruption errors instead of cryptic BadZipFile
                latent_chunk = load_latent_safe(latent_path)

                # Truncate to multiple of bin_size
                n_bins = len(latent_chunk) // bin_size
                n_frames_to_keep = n_bins * bin_size

                if n_frames_to_keep == 0:
                    continue

                # Read each video's OWN fps (not just the first). A project may
                # mix frame rates; using one video's fps for all others produces
                # systematically wrong timestamps. self.fps keeps the first
                # successful read as a fallback for any video we can't probe.
                try:
                    video_path = os.path.join(self.source_path, video_source_name)
                    with VideoReader(video_path) as vr:
                        video_fps = vr.fps
                    self.fps_per_video[video_source_name] = video_fps
                    if not latents_buffer:
                        self.fps = video_fps
                except Exception as e:
                    self.notify(f"Warning: Could not read FPS from {video_source_name}, using fallback {self.fps}. Error: {e}", "warning")

                latents_buffer.append(latent_chunk[:n_frames_to_keep])
                self.videos_meta.append((n_bins, video_source_name))
                total_frames_loaded += n_frames_to_keep

            except LatentCorruptError as e:
                # Surface the typed error with the hint, but keep loading the rest
                # — corrupting one video should not break the whole session.
                self.notify(str(e), "error")
            except Exception as e:
                self.notify(f"Error loading {filename}: {e}", "error")

        if latents_buffer:
            # Unique-per-aggregator memmap filename so two concurrent runs
            # (e.g. two Gradio sessions clustering different ROIs / models)
            # do not clobber each other's disk-backed array (3-D).
            safe_model = "".join(c if c.isalnum() else "_" for c in self.model_name)
            memmap_name = (
                f"aggregated_latents_{safe_model}_roi{self.select_roi_id}_bin{self.bin_size}.dat"
            )
            self.latents = _aggregate_latents(
                latents_buffer,
                cache_dir=os.path.join(self.project_path, 'cluster', '_cache'),
                notify=self.notify,
                memmap_filename=memmap_name,
            )
            self.notify(f'Finished loading. Total aggregated latents: {len(self.latents)}')
        else:
            self.notify("Warning: No latents loaded.", "warning")

    def _get_cached_reader(self, video_path: str) -> VideoReader:
        """Get or create a cached VideoReader for a video path.

        Maintains an LRU cache of at most ``_cache_max_size`` open
        ``VideoReader`` instances so repeated frame reads for the same
        video avoid re-opening the file.

        Thread-safe via ``_cache_lock``.
        """
        with self._cache_lock:
            if video_path in self._video_reader_cache:
                self._video_reader_cache.move_to_end(video_path)
                return self._video_reader_cache[video_path]

            if len(self._video_reader_cache) >= self._cache_max_size:
                _, old_reader = self._video_reader_cache.popitem(last=False)
                try:
                    old_reader.close()
                except Exception as exc:  # noqa: BLE001 — cleanup only
                    logger.debug("VideoReader close on evict failed: %s", exc)

            reader = VideoReader(video_path)
            self._video_reader_cache[video_path] = reader
            return reader

    def _get_cached_frame(self, video_path: str, frame_idx: int) -> np.ndarray:
        """Return one frame, served from an LRU frame cache when possible.

        Args:
            video_path: Absolute path to the source video.
            frame_idx: Zero-based frame index inside that video.

        Returns:
            ``(H, W, 3)`` uint8 array. Cache hits avoid PyAV seek+decode
            entirely; misses fall through to ``reader.get_frame``.
        """
        key = (video_path, frame_idx)
        # Hold the lock for the full check → fetch → insert sequence so two
        # concurrent callers asking for the same frame don't both decode and
        # both insert (3-E).  Decoding inside the lock is acceptable — single
        # frame decode is fast relative to the per-click I/O budget, and the
        # lock is uncontended in the common (single-user) case.
        with self._cache_lock:
            if key in self._frame_cache:
                self._frame_cache.move_to_end(key)
                return self._frame_cache[key]

            reader = self._get_cached_reader(video_path)
            frame = reader.get_frame(frame_idx)

            self._frame_cache[key] = frame
            if len(self._frame_cache) > self._frame_cache_max:
                self._frame_cache.popitem(last=False)
            return frame

    def get_frame(self, index: int) -> Optional[np.ndarray]:
        """Retrieve the representative frame for a given global bin index.

        The frame is taken from the centre of the bin (``bin_size // 2``).
        Uses an LRU reader pool + frame cache (see [PERF-03]) so repeated
        clicks / hovers on the same cluster don't re-decode.

        Args:
            index: Global bin index across all aggregated videos.

        Returns:
            ``(H, W, 3)`` numpy frame, or ``None`` if retrieval fails or
            the index is out of bounds (an error is surfaced via
            :attr:`notify` in those cases).
        """
        for n_bins_in_video, video_name in self.videos_meta:
            if index >= n_bins_in_video:
                index -= n_bins_in_video
                continue

            video_path = os.path.join(self.source_path, video_name)
            frame_idx = index * self.bin_size + self.bin_size // 2
            logger.debug('Retrieving frame from %s at index %d', video_name, frame_idx)
            try:
                return self._get_cached_frame(video_path, frame_idx)
            except Exception as e:
                self.notify(f"Error reading frame: {e}", "error")
                return None

        self.notify('Error: Index out of bounds in Aggregator', "error")
        return None

    def close(self) -> None:
        """Close all cached VideoReader instances and release resources.

        Cleanup-phase exceptions are logged at debug level — raising here
        would mask whatever triggered the close in the first place.
        """
        for reader in self._video_reader_cache.values():
            try:
                reader.close()
            except Exception as exc:  # noqa: BLE001 — cleanup only
                logger.debug("VideoReader close on aggregator shutdown failed: %s", exc)
        self._video_reader_cache.clear()
        self._frame_cache.clear()

    def __del__(self) -> None:
        """Clean up cached readers on garbage collection."""
        self.close()

    def get_latent_object(self) -> Latent:
        """Returns the high-level Latent explorer object."""
        if self.latents is None:
            raise ValueError("No latents loaded. Cannot create Latent object.")
        return Latent(self.latents, self.bin_size)

    def generate_subtitles(self, syllables: np.ndarray, meta: Dict) -> List[str]:
        """
        Generates SRT subtitle files based on clustering results (syllables).
        """
        subtitle_output_dir = os.path.join(self.project_path, 'subtitles')
        os.makedirs(subtitle_output_dir, exist_ok=True)
        
        generated_files = []
        cum_bins = 0

        for n_bins_in_video, video_name in self.videos_meta:
            # Use this video's own fps (a project may mix frame rates).
            video_fps = self.fps_per_video.get(video_name, self.fps)

            # Extract syllables corresponding to this video
            # Syllables are per-bin, so we repeat them to match frame-rate if we want per-frame arrays,
            # BUT the logic here seems to iterate changes in bins.

            this_video_syllables_bins = syllables[cum_bins : cum_bins + n_bins_in_video]
            
            # Expand bins to frames for precision? 
            # The original code repeated: data = np.repeat(this_video_syllabels, self.bin_size)
            data = np.repeat(this_video_syllables_bins, self.bin_size)
            
            srt_entries = []
            n_frames = len(data)
            
            # Find indices where behavior changes
            # Prepend -1 and append n-1 to handle start and end
            change_indices = np.arange(n_frames - 1)[data[:-1] != data[1:]]
            change_indices = np.concatenate([[-1], change_indices, [n_frames - 1]])
            
            for i in range(len(change_indices) - 1):
                start_frame = change_indices[i] + 1
                end_frame = change_indices[i+1]
                
                start_time = frame_to_timestamp(start_frame, video_fps)
                end_time = frame_to_timestamp(end_frame, video_fps)
                
                behavior_id = data[start_frame]
                
                if behavior_id == -1:
                    behavior_name = "Unclustered"
                else:
                    # meta keys might be integers or strings, let's try both
                    if behavior_id in meta:
                         behavior_name = meta[behavior_id]['name']
                    elif str(behavior_id) in meta:
                         behavior_name = meta[str(behavior_id)]['name']
                    else:
                         behavior_name = f"Cluster {behavior_id}"

                srt_entries.append(f"{i + 1}\n{start_time} --> {end_time}\n{behavior_name}\n")
            
            srt_content = "\n".join(srt_entries)
            
            video_basename = os.path.splitext(os.path.basename(video_name))[0]
            output_path = os.path.join(subtitle_output_dir, video_basename + '.srt')
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
                
            generated_files.append(output_path)
            cum_bins += n_bins_in_video
            
        return generated_files
