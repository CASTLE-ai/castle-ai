"""
castle/service/bout_service.py
Service for extracting bout video clips from clustered behavior data.

A bout is a consecutive sequence of frames assigned to the same cluster.
"""

import os
import logging
import subprocess
import tempfile
import numpy as np
from typing import List, Optional, Tuple

# Avoid importing AnnotatorData at module level to keep dependency light;
# the type is referenced only in generate_grid_video's signature.

logger = logging.getLogger(__name__)


def _transcode_to_h264(video_path: str) -> None:
    """Re-encode *video_path* in-place to H.264 using ffmpeg libx264.

    The file is written to a temporary path first, then atomically
    replaces the original so that a partial failure leaves the mp4v
    file intact.

    Args:
        video_path: Path to an MP4 file written with the mp4v codec.
    """
    tmp_path = video_path + ".h264tmp.mp4"
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-movflags",
                "+faststart",
                tmp_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            os.replace(tmp_path, video_path)
        else:
            logger.warning(
                "ffmpeg H.264 transcode failed for %s (keeping mp4v). stderr: %s",
                video_path,
                result.stderr[-300:],
            )
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except FileNotFoundError:
        logger.warning("ffmpeg not found — keeping mp4v codec for %s", video_path)
    except Exception as exc:
        logger.warning("H.264 transcode error for %s: %s", video_path, exc)
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def find_bouts(cluster_array: np.ndarray, cluster_id: int) -> List[Tuple[int, int]]:
    """Find all consecutive bout segments for a given cluster ID.

    Args:
        cluster_array: 1D array of cluster assignments per bin.
        cluster_id: Target cluster ID.

    Returns:
        List of (start_bin, end_bin) tuples (end exclusive), sorted longest first.
    """
    mask = (cluster_array == cluster_id).astype(int)
    if mask.sum() == 0:
        return []

    # Find transitions
    diff = np.diff(mask, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    bouts = list(zip(starts.tolist(), ends.tolist()))
    # Sort by length (longest first)
    bouts.sort(key=lambda b: b[1] - b[0], reverse=True)
    return bouts


def extract_cluster_bouts(
    aggregator,
    latents,
    cluster_id: int,
    max_bouts: int = 9,
    max_frames: int = 60,
    output_dir: Optional[str] = None,
    fps: float = 10.0,
) -> List[str]:
    """Extract bout video clips for a specific cluster as MP4 files.

    Args:
        aggregator: LatentAggregator instance (provides get_frame, videos_meta, etc.)
        latents: Latent object with .cluster array
        cluster_id: Target cluster ID to extract bouts for
        max_bouts: Maximum number of bouts to extract
        max_frames: Maximum frames per bout video
        output_dir: Directory to save videos. If None, uses a temp directory.
        fps: Video playback speed (frames per second)

    Returns:
        List of MP4 file paths
    """
    import cv2

    bouts = find_bouts(latents.cluster, cluster_id)
    if not bouts:
        return []

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="castle_bouts_")
    os.makedirs(output_dir, exist_ok=True)

    video_paths = []
    for bout_idx, (start_bin, end_bin) in enumerate(bouts[:max_bouts]):
        bout_len = end_bin - start_bin
        # Sample frames: if bout is longer than max_frames, subsample evenly
        if bout_len > max_frames:
            indices = np.linspace(start_bin, end_bin - 1, max_frames, dtype=int)
        else:
            indices = np.arange(start_bin, end_bin)

        frames = []
        for bin_idx in indices:
            frame = aggregator.get_frame(int(bin_idx))
            if frame is not None:
                # Resize for reasonable video size (max 256px on longest side)
                h, w = frame.shape[:2]
                max_side = 256
                if max(w, h) > max_side:
                    scale = max_side / max(w, h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                frames.append(frame)

        if not frames:
            continue

        # Get behavior name for filename
        cluster_name = "unknown"
        if cluster_id in latents.cluster_meta:
            cluster_name = latents.cluster_meta[cluster_id]['name']

        video_path = os.path.join(
            output_dir,
            f"bout_{cluster_name}_{bout_idx:02d}_bins{start_bin}-{end_bin}.mp4"
        )

        # Save as MP4 video using cv2, then transcode to H.264 for browser compat
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

        for frame in frames:
            # cv2 expects BGR, frames from get_frame are likely RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            out.write(frame_bgr)

        out.release()
        _transcode_to_h264(video_path)
        video_paths.append(video_path)
        logger.info("Saved bout video: %s (%d frames)", video_path, len(frames))

    return video_paths


# ---------------------------
# Grid Video Generation
# ---------------------------


def _select_representative_bouts(
    bouts: List[Tuple[int, int]],
    cluster_id: int,
    embedding: np.ndarray,
    cluster_array: np.ndarray,
    n: int,
) -> List[Tuple[int, int]]:
    """Return the *n* bouts closest to the cluster centroid in embedding space.

    Args:
        bouts: All bouts for this cluster as (start_bin, end_bin) pairs.
        cluster_id: The cluster ID whose centroid we compute.
        embedding: (n_bins, 2) UMAP embedding array.
        cluster_array: (n_bins,) cluster assignment array.
        n: Number of bouts to select.

    Returns:
        Up to *n* (start_bin, end_bin) pairs, sorted by ascending distance
        to the centroid (most representative first).
    """
    if not bouts:
        return []

    # Guard: embedding and cluster array must have same length
    if len(embedding) != len(cluster_array):
        logger.warning(
            "Embedding/cluster size mismatch (%d vs %d); returning first %d bouts",
            len(embedding), len(cluster_array), n,
        )
        return bouts[:n]

    # Cluster centroid from all bins belonging to this cluster
    mask = cluster_array == cluster_id
    centroid = embedding[mask].mean(axis=0)  # shape (2,)

    def _bout_distance(bout: Tuple[int, int]) -> float:
        start, end = bout
        mean_emb = embedding[start:end].mean(axis=0)
        diff = mean_emb - centroid
        return float(np.dot(diff, diff))  # squared L2 — monotone for ranking

    bouts_sorted = sorted(bouts, key=_bout_distance)
    return bouts_sorted[:n]


def _compute_aligned_range(
    bouts: List[Tuple[int, int]],
    pad_bins: int,
    total_bins: int,
) -> int:
    """Compute the half-length for center-aligned, symmetrically-trimmed bouts.

    Each bout's padded span = (end - start) + 2 * pad_bins.  All bouts are
    center-aligned, so we take the *shortest* padded length and halve it.

    Args:
        bouts: Selected (start_bin, end_bin) pairs.
        pad_bins: Padding to add on each side.
        total_bins: Total number of bins in the dataset (for clamping).

    Returns:
        half_len: Half-length (in bins) for extraction around each centre.
    """
    if not bouts:
        return 0

    padded_lengths = []
    for start, end in bouts:
        centre = (start + end) // 2
        padded_start = max(0, start - pad_bins)
        padded_end = min(total_bins, end + pad_bins)
        # Actual available half-length on each side
        left = centre - padded_start
        right = padded_end - centre
        padded_lengths.append(min(left, right))

    return min(padded_lengths)


def generate_grid_video(
    annotator_data,
    cluster_id: int,
    grid_cols: int = 3,
    output_dir: Optional[str] = None,
    target_fps: float = 30.0,
    cell_size: int = 192,
) -> Optional[str]:
    """Generate a grid video of the most representative bouts for a cluster.

    Selects the top-N bouts (N = grid_cols²) closest to the cluster centroid
    in UMAP embedding space, pads each by 0.5 s, center-aligns them, trims
    to the shortest bout, tiles them in a grid, and writes an H.264 MP4.

    The result is cached: if the output file already exists it is returned
    immediately without re-rendering.

    Args:
        annotator_data: :class:`~castle.service.annotator_loader.AnnotatorData`.
        cluster_id: Target cluster ID.
        grid_cols: Side length of the square grid (e.g. 3 → 3×3 = 9 cells).
        output_dir: Directory for the cached grid video.  Defaults to
            ``<project_path>/cluster/grid_videos/``.
        target_fps: Frame-rate of the output video.
        cell_size: Each grid cell is resized so its longest side equals
            *cell_size* pixels before tiling.

    Returns:
        Absolute path to the rendered (or cached) MP4, or *None* on failure.
    """
    import cv2
    from castle.service.annotator_loader import get_annotator_frame

    cluster_meta = annotator_data.cluster_meta
    cluster_name = cluster_meta.get(cluster_id, {}).get("name", f"cluster{cluster_id}")

    # --- Cache check ---
    if output_dir is None:
        output_dir = os.path.join(
            annotator_data.project_path, "cluster", "grid_videos"
        )
    os.makedirs(output_dir, exist_ok=True)

    cache_path = os.path.join(
        output_dir, f"{cluster_name}_grid_{grid_cols}x{grid_cols}.mp4"
    )
    if os.path.exists(cache_path):
        logger.info("Grid video cache hit: %s", cache_path)
        return cache_path

    # --- Find bouts ---
    all_bouts = find_bouts(annotator_data.cluster, cluster_id)
    if not all_bouts:
        logger.warning("No bouts found for cluster %d (%s)", cluster_id, cluster_name)
        return None

    n_cells = grid_cols * grid_cols
    selected = _select_representative_bouts(
        bouts=all_bouts,
        cluster_id=cluster_id,
        embedding=annotator_data.embedding,
        cluster_array=annotator_data.cluster,
        n=n_cells,
    )

    # --- Padding in bins ---
    fps = annotator_data.fps if annotator_data.fps > 0 else 30.0
    bin_size = annotator_data.bin_size if annotator_data.bin_size > 0 else 1
    pad_bins = max(1, int(0.5 * fps / bin_size))

    total_bins = len(annotator_data.cluster)
    half_len = _compute_aligned_range(selected, pad_bins, total_bins)
    if half_len <= 0:
        logger.warning("Computed half_len=%d — cannot generate grid video", half_len)
        return None

    n_frames = half_len * 2  # total frames per cell (in bins)

    # --- Determine output cell dimensions from first available frame ---
    cell_h = cell_size
    cell_w = cell_size
    for start, end in selected:
        centre = (start + end) // 2
        sample_frame = get_annotator_frame(annotator_data, centre)
        if sample_frame is not None:
            fh, fw = sample_frame.shape[:2]
            scale = cell_size / max(fw, fh)
            cell_w = max(2, int(fw * scale) & ~1)
            cell_h = max(2, int(fh * scale) & ~1)
            break

    grid_w = cell_w * grid_cols
    grid_h = cell_h * grid_cols
    # Ensure even dimensions for H.264
    grid_w = grid_w & ~1
    grid_h = grid_h & ~1

    # --- Write raw video with cv2 ---
    raw_path = cache_path + ".raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_path, fourcc, target_fps, (grid_w, grid_h))

    black_cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)

    for frame_offset in range(n_frames):
        grid_rows_imgs = []
        for row in range(grid_cols):
            row_cells = []
            for col in range(grid_cols):
                cell_idx = row * grid_cols + col
                if cell_idx < len(selected):
                    start, end = selected[cell_idx]
                    centre = (start + end) // 2
                    bin_idx = centre - half_len + frame_offset
                    if 0 <= bin_idx < total_bins:
                        frame = get_annotator_frame(annotator_data, int(bin_idx))
                    else:
                        frame = None

                    if frame is not None:
                        fh, fw = frame.shape[:2]
                        scale = cell_size / max(fw, fh)
                        rw = max(2, int(fw * scale) & ~1)
                        rh = max(2, int(fh * scale) & ~1)
                        frame_resized = cv2.resize(
                            frame, (rw, rh), interpolation=cv2.INTER_LANCZOS4
                        )
                        # Pad to exact cell size (top-left anchor)
                        cell_img = black_cell.copy()
                        cell_img[:rh, :rw] = frame_resized
                    else:
                        cell_img = black_cell.copy()
                else:
                    cell_img = black_cell.copy()

                # cv2 expects BGR
                row_cells.append(cv2.cvtColor(cell_img, cv2.COLOR_RGB2BGR))

            grid_rows_imgs.append(np.concatenate(row_cells, axis=1))

        grid_frame = np.concatenate(grid_rows_imgs, axis=0)
        # Safety: ensure exact grid dimensions
        grid_frame = grid_frame[:grid_h, :grid_w]
        writer.write(grid_frame)

    writer.release()
    logger.info(
        "Grid video written: %s (%d cells × %d frames)", raw_path, len(selected), n_frames
    )

    # --- Transcode to H.264 ---
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                raw_path,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                cache_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            try:
                os.unlink(raw_path)
            except OSError:
                pass
            logger.info("Grid video transcoded to H.264: %s", cache_path)
            return cache_path
        else:
            logger.warning(
                "ffmpeg transcode failed (returncode=%d). stderr: %s",
                result.returncode,
                result.stderr[-400:],
            )
            # Fall back to the raw mp4v file
            import shutil
            shutil.copyfile(raw_path, cache_path)
            try:
                os.unlink(raw_path)
            except OSError:
                pass
            return cache_path
    except FileNotFoundError:
        logger.warning("ffmpeg not found — returning mp4v grid video")
        import shutil
        shutil.copyfile(raw_path, cache_path)
        try:
            os.unlink(raw_path)
        except OSError:
            pass
        return cache_path
    except Exception as exc:
        logger.error("Unexpected error during H.264 transcode: %s", exc)
        return None
