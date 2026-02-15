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
    """Extract bout video clips for a specific cluster as GIF files.

    Args:
        aggregator: LatentAggregator instance (provides get_frame, videos_meta, etc.)
        latents: Latent object with .cluster array
        cluster_id: Target cluster ID to extract bouts for
        max_bouts: Maximum number of bouts to extract
        max_frames: Maximum frames per bout GIF
        output_dir: Directory to save GIFs. If None, uses a temp directory.
        fps: GIF playback speed (frames per second)

    Returns:
        List of GIF file paths
    """
    from PIL import Image

    bouts = find_bouts(latents.cluster, cluster_id)
    if not bouts:
        return []

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="castle_bouts_")
    os.makedirs(output_dir, exist_ok=True)

    gif_paths = []
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
                # Convert to PIL Image
                img = Image.fromarray(frame)
                # Resize for reasonable GIF size (max 256px on longest side)
                max_side = 256
                w, h = img.size
                if max(w, h) > max_side:
                    scale = max_side / max(w, h)
                    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                frames.append(img)

        if not frames:
            continue

        # Get behavior name for filename
        cluster_name = "unknown"
        if cluster_id in latents.cluster_meta:
            cluster_name = latents.cluster_meta[cluster_id]['name']

        gif_path = os.path.join(
            output_dir,
            f"bout_{cluster_name}_{bout_idx:02d}_bins{start_bin}-{end_bin}.gif"
        )

        # Save as animated GIF
        duration_ms = int(1000 / fps)
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
        gif_paths.append(gif_path)
        logger.info(f"Saved bout GIF: {gif_path} ({len(frames)} frames)")

    return gif_paths
