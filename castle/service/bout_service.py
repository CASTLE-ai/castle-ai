"""
castle/service/bout_service.py
Service for extracting bout video clips from clustered behavior data.

A bout is a consecutive sequence of frames assigned to the same cluster.
"""

import os
import logging
import tempfile
import numpy as np
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


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

        # Save as MP4 video using cv2
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        
        for frame in frames:
            # cv2 expects BGR, frames from get_frame are likely RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            out.write(frame_bgr)
        
        out.release()
        video_paths.append(video_path)
        logger.info(f"Saved bout video: {video_path} ({len(frames)} frames)")

    return video_paths
