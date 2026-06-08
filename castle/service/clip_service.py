"""Short-clip generation service for the Behavior Microscope (ARCH-01 / P4).

When the user clicks a point in the UMAP scatter plot we render a short
mp4 around the corresponding frame with the ROI contour overlaid. The
implementation is pure :mod:`cv2` + ``ffmpeg`` (via subprocess) and was
historically tangled into ``castle/ui/cluster_handlers.py``.

This module exposes a single public entry point
:func:`generate_clip_with_roi_overlay` plus its supporting helpers, so
the same clip-generation code can be reused from the CLI or a Jupyter
notebook without dragging in the Gradio import path.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from typing import Any, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "generate_clip_with_roi_overlay",
    "get_bin_video_info",
    "apply_roi_overlay",
    "transcode_to_h264",
]


def transcode_to_h264(video_path: str) -> None:
    """Re-encode *video_path* in place to H.264 via ``ffmpeg libx264``.

    Writes to a sibling ``.h264tmp.mp4`` first and then atomically
    replaces the original, so a partial failure leaves the source
    intact. Tolerant of missing ``ffmpeg`` — logs a warning and leaves
    the file as-is.

    Args:
        video_path: Path to an MP4 currently encoded with another codec
            (typically the ``mp4v`` output from :class:`cv2.VideoWriter`).
    """
    tmp_path = video_path + ".h264tmp.mp4"
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-movflags", "+faststart",
                tmp_path,
            ],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            os.replace(tmp_path, video_path)
        else:
            logger.warning(
                "ffmpeg H.264 transcode failed for %s (keeping mp4v). stderr: %s",
                video_path, result.stderr[-300:],
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


def get_bin_video_info(aggregator: Any, bin_index: int) -> Tuple[Optional[str], Optional[int]]:
    """Map a global bin index to ``(video_name, frame_index)``.

    The aggregator stores videos sequentially in a flat bin array; this
    helper walks the videos_meta cumsum to recover the per-video frame
    index for an arbitrary global bin.

    Args:
        aggregator: :class:`LatentAggregator`. Reads ``videos_meta`` and
            ``bin_size``.
        bin_index: Global (across-all-videos) bin index.

    Returns:
        ``(video_name, frame_idx)`` or ``(None, None)`` if ``bin_index``
        falls past the last video.
    """
    # Route through the aggregator's FrameIndexMap so legacy (bin) and prepared
    # (decimated window) datapoints both map to the correct original frame.
    fim = getattr(aggregator, "frame_index_map", None)
    if fim is not None:
        try:
            video_idx, frame_idx = fim.dp_to_orig_frame(int(bin_index))
            return fim.base.video_names[video_idx], frame_idx
        except (IndexError, ValueError):
            return None, None
    idx = bin_index
    for n_bins_in_video, video_name in aggregator.videos_meta:
        if idx >= n_bins_in_video:
            idx -= n_bins_in_video
            continue
        frame_idx = idx * aggregator.bin_size + aggregator.bin_size // 2
        return video_name, frame_idx
    return None, None


def apply_roi_overlay(frame: np.ndarray, mask_h5_path: str, frame_idx: int, thickness: int = 1) -> np.ndarray:
    """Draw an ROI contour outline on *frame* if a mask is available.

    Loads the binary mask for ``frame_idx`` from the HDF5 file at
    ``mask_h5_path``, finds its external contours, rescales them to the
    frame's dimensions, and draws a green outline. Returns the original
    frame unchanged if the mask file is missing or any error occurs.

    Args:
        frame: RGB numpy array of shape ``(H, W, 3)``.
        mask_h5_path: Path to the project's ``mask_list.h5``.
        frame_idx: Integer frame index used as the key inside the HDF5.
        thickness: Contour line thickness in pixels (default 1 — a thin
            outline; the clip is downscaled afterwards so 1px reads as a
            crisp hairline).

    Returns:
        A new RGB frame with the ROI overlay, or the input frame
        unchanged when no overlay could be drawn.
    """
    if not os.path.exists(mask_h5_path):
        return frame
    try:
        import h5py

        with h5py.File(mask_h5_path, 'r') as f:
            key = str(frame_idx)
            if key not in f:
                return frame
            mask = f[key][()]
        if mask.max() == 0:
            return frame
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return frame
        fh, fw = frame.shape[:2]
        mask_h, mask_w = mask.shape[:2]
        frame_out = frame.copy()
        for cnt in contours:
            scaled = cnt.copy().astype(np.float64)
            scaled[:, :, 0] *= fw / mask_w
            scaled[:, :, 1] *= fh / mask_h
            scaled = scaled.astype(np.int32)
            cv2.drawContours(frame_out, [scaled], -1, (0, 255, 0), thickness)
        return frame_out
    except Exception:
        logger.debug("ROI overlay failed for %s[%d]", mask_h5_path, frame_idx, exc_info=True)
        return frame


def generate_clip_with_roi_overlay(
    aggregator: Any,
    center_bin: int,
    *,
    clip_seconds: float = 2.0,
    fps: Optional[float] = None,
    max_frames: int = 300,
    max_size: int = 480,
) -> Optional[str]:
    """Render a smooth MP4 preview clip around the clicked datapoint.

    The datapoint maps to a ``(video, original-frame INTERVAL)``. We read the
    consecutive original frames spanning ~``clip_seconds`` of real time around
    that interval (always covering the whole interval) from that one video, and
    overlay the ROI contour ONLY on the frames inside the datapoint's interval —
    so the green outline marks the datapoint's actual extent while the
    surrounding buffer frames play un-marked. Frames are downscaled so the
    longest side is ``max_size`` for comfortable in-browser viewing.

    Args:
        aggregator: :class:`LatentAggregator` (frame reader + ``frame_index_map``).
        center_bin: Global datapoint index the user clicked.
        clip_seconds: Real-time span of the buffer window (centred on the
            datapoint). The window is always widened to cover the datapoint's
            full interval if that interval is longer.
        fps: Playback frame rate. ``None`` → the source video's own fps
            (natural-speed playback).
        max_frames: Hard cap on frames read, to keep preview generation snappy.
        max_size: Longest-side pixel cap for the output (downscaled after the
            overlay so the contour stays aligned). 0 disables downscaling.

    Returns:
        Absolute path to a temporary ``.mp4`` (re-encoded to H.264 when
        ffmpeg/PyAV is available), or ``None`` if no frames could be assembled.
    """
    fim = getattr(aggregator, "frame_index_map", None)
    if fim is None:
        return None
    try:
        video_idx, dp_first, dp_last = fim.dp_to_orig_span(int(center_bin))
        video_name = fim.base.video_names[video_idx]
    except (IndexError, ValueError, AttributeError):
        return None
    dp_first, dp_last = int(dp_first), int(dp_last)

    video_fps = aggregator.fps_per_video.get(video_name, aggregator.fps) or 30.0
    try:
        n_orig = int(fim.base.n_orig_frames[video_idx])
    except Exception:  # noqa: BLE001 — n_orig_frames may be absent on legacy maps
        n_orig = None

    # Buffer ~clip_seconds around the datapoint, but ALWAYS cover its full
    # interval [dp_first, dp_last) so the marked extent is never cut off.
    dp_center = (dp_first + dp_last) // 2
    half = max(1, int(round(clip_seconds * video_fps / 2.0)))
    start_f = max(0, min(dp_first, dp_center - half))
    end_f = max(dp_last, dp_center + half + 1)
    if n_orig is not None:
        end_f = min(end_f, n_orig)
    end_f = min(end_f, start_f + max_frames)

    # track/ subdirs are named with the full video filename (incl. extension),
    # so do NOT strip it — splitext here points the overlay at a non-existent dir.
    mask_path = os.path.join(
        aggregator.project_path, "track", os.path.basename(video_name), "mask_list.h5",
    )

    def _shrink(fr: Any) -> Any:
        h0, w0 = fr.shape[:2]
        if not max_size or max(h0, w0) <= max_size:
            return fr
        s = max_size / float(max(h0, w0))
        # even dims for H.264; INTER_AREA is the right filter for shrinking
        nw = max(2, (int(round(w0 * s)) // 2) * 2)
        nh = max(2, (int(round(h0 * s)) // 2) * 2)
        return cv2.resize(fr, (nw, nh), interpolation=cv2.INTER_AREA)

    frames = []
    for f in range(start_f, end_f):
        frame = aggregator.get_raw_frame(video_name, f)
        if frame is None:
            continue
        # Overlay ONLY on the datapoint's own interval; buffer frames stay clean.
        if dp_first <= f < dp_last:
            frame = apply_roi_overlay(frame, mask_path, f)
        frames.append(_shrink(frame))  # resize AFTER overlay so the contour aligns

    if not frames:
        return None

    play_fps = min(max(float(fps) if fps else float(video_fps), 1.0), 120.0)
    h, w = frames[0].shape[:2]
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp.name, fourcc, play_fps, (w, h))
    for fr in frames:
        bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR) if len(fr.shape) == 3 else fr
        out.write(bgr)
    out.release()

    transcode_to_h264(tmp.name)
    return tmp.name
