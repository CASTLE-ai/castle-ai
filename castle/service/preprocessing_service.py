"""
castle/service/preprocessing_service.py
Service layer for stabilized camera preprocessing.

All functions take simple types (str, int, float) and return dicts.
No gradio imports.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


def preprocess_stabilized_camera(
    storage_path: str,
    project_name: str,
    video_name: str,
    body_roi_id: int,
    head_roi_id: int,
    fc: float = 0.25,
    order: int = 2,
    margin: int = 75,
    min_crop: int = 300,
    output_size: int = 518,
    preview_duration: float = 10.0,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, object]:
    """Run stabilized camera preprocessing for a video.

    Loads ROI masks from the tracking output, computes smooth centroid
    trajectories via zero-phase Butterworth filtering, then extracts
    dynamically-cropped and rotated frames saved as an H.264 MP4.

    Parameters
    ----------
    storage_path : str
        Root storage directory (e.g. ``"projects/"``).
    project_name : str
        Name of the project folder inside *storage_path*.
    video_name : str
        Filename of the source video (e.g. ``"animal.mp4"``).
    body_roi_id : int
        ROI id for the body (used as centroid and for orientation computation).
    head_roi_id : int
        ROI id for the head (used together with *body_roi_id* for orientation).
    fc : float
        Low-pass cutoff frequency in Hz. Default 0.25 Hz.
    order : int
        Butterworth filter order. Default 2.
    margin : int
        Spatial margin (px) added to the high-pass residual when computing the
        dynamic crop window. Default 75 px.
    min_crop : int
        Minimum crop side length in pixels. Default 300 px.
    output_size : int
        Side length of the square output frames. Default 518 px (DINOv2 ViT-B/14).
    preview_duration : float
        Length of the short preview clip in seconds. Default 10 s.
    progress_callback : callable, optional
        Called as ``progress_callback(fraction: float, description: str)``
        where *fraction* is in [0, 1].

    Returns
    -------
    dict with keys:
        preprocessed_video_path : str
            Absolute path to the full stabilised output video.
        diagnostics : dict
            Diagnostic metrics from :meth:`StabilizedCamera.get_diagnostics`.
        preview_path : str
            Absolute path to the short preview clip.
        n_frames : int
            Total number of processed frames.
    """
    from castle.core.stabilized_camera import (
        StabilizedCamera,
        extract_centroids_from_masks,
        extract_orientations_from_masks,
    )
    from castle.utils.video_io import ReadArray

    # ------------------------------------------------------------------
    # 1. Resolve paths
    # ------------------------------------------------------------------
    project_path = Path(storage_path) / project_name
    mask_h5_path = str(project_path / "track" / video_name / "mask_list.h5")
    video_path = str(project_path / "sources" / video_name)

    out_dir = project_path / "preprocessed" / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_video_path = str(out_dir / "stabilized.mp4")
    out_preview_path = str(out_dir / "stabilized_preview.mp4")

    logger.info(
        "preprocess_stabilized_camera: project=%s, video=%s, body_roi=%d, head_roi=%d",
        project_name, video_name, body_roi_id, head_roi_id,
    )

    # ------------------------------------------------------------------
    # 2. Get frame count and fps from VideoReader
    # ------------------------------------------------------------------
    with ReadArray(video_path) as reader:
        n_frames: int = len(reader)
        fps: float = reader.fps
    logger.info("preprocess_stabilized_camera: n_frames=%d, fps=%.2f", n_frames, fps)

    # ------------------------------------------------------------------
    # 3. Extract centroid positions and orientation angles from masks
    # ------------------------------------------------------------------
    if progress_callback:
        progress_callback(0.0, "Extracting centroids from masks…")

    positions = extract_centroids_from_masks(mask_h5_path, body_roi_id, n_frames)

    if progress_callback:
        progress_callback(0.05, "Extracting orientations from masks…")

    angles = extract_orientations_from_masks(
        mask_h5_path, body_roi_id, head_roi_id, n_frames
    )

    # ------------------------------------------------------------------
    # 4. Create StabilizedCamera (computes filtered trajectory)
    # ------------------------------------------------------------------
    if progress_callback:
        progress_callback(0.10, "Computing stabilised trajectory…")

    cam = StabilizedCamera(
        positions=positions,
        angles=angles,
        fps=fps,
        fc=fc,
        order=order,
        margin=margin,
        min_crop=min_crop,
        output_size=output_size,
    )

    # ------------------------------------------------------------------
    # 5. Generate all preprocessed frames → H.264 MP4
    # ------------------------------------------------------------------
    if progress_callback:
        progress_callback(0.12, "Encoding stabilised video…")

    _encode_stabilized_video(
        video_path=video_path,
        cam=cam,
        out_path=out_video_path,
        fps=fps,
        n_frames=n_frames,
        output_size=output_size,
        progress_callback=progress_callback,
        progress_start=0.12,
        progress_end=0.90,
    )

    # ------------------------------------------------------------------
    # 6. Generate preview clip
    # ------------------------------------------------------------------
    if progress_callback:
        progress_callback(0.90, "Generating preview clip…")

    _encode_stabilized_video(
        video_path=video_path,
        cam=cam,
        out_path=out_preview_path,
        fps=fps,
        n_frames=n_frames,
        output_size=output_size,
        max_frames=int(fps * preview_duration),
        progress_callback=None,
    )

    # ------------------------------------------------------------------
    # 7. Collect diagnostics and return
    # ------------------------------------------------------------------
    if progress_callback:
        progress_callback(1.0, "Done.")

    diagnostics = cam.get_diagnostics()
    logger.info(
        "preprocess_stabilized_camera: done. hp_rms=%.2f px, pct_min_crop=%.1f%%",
        diagnostics["hp_residual_rms"],
        diagnostics["pct_at_min_crop"],
    )

    return {
        "preprocessed_video_path": out_video_path,
        "diagnostics": diagnostics,
        "preview_path": out_preview_path,
        "n_frames": n_frames,
    }


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _encode_stabilized_video(
    video_path: str,
    cam: object,  # StabilizedCamera
    out_path: str,
    fps: float,
    n_frames: int,
    output_size: int,
    max_frames: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
    progress_start: float = 0.0,
    progress_end: float = 1.0,
) -> None:
    """Encode stabilised frames to an H.264 MP4 using PyAV.

    Parameters
    ----------
    video_path : str
        Path to the source video.
    cam : StabilizedCamera
        Configured camera instance.
    out_path : str
        Destination MP4 path.
    fps : float
        Frame rate for the output container.
    n_frames : int
        Total number of frames in the source video.
    output_size : int
        Width/height of the square output frames.
    max_frames : int, optional
        Stop after this many frames (for preview clips). ``None`` = all frames.
    progress_callback : callable, optional
        Progress reporting callback.
    progress_start / progress_end : float
        Range within [0, 1] for this encoding step.
    """
    import av  # type: ignore

    limit = min(max_frames, n_frames) if max_frames is not None else n_frames

    input_container = av.open(video_path)
    input_stream = input_container.streams.video[0]

    output_container = av.open(out_path, mode="w")
    out_stream = output_container.add_stream("h264", rate=int(fps))
    out_stream.width = output_size
    out_stream.height = output_size
    out_stream.pix_fmt = "yuv420p"
    out_stream.options = {"crf": "18", "preset": "fast"}

    try:
        for i, pkt_frame in enumerate(input_container.decode(input_stream)):
            if i >= limit:
                break
            if i >= n_frames:
                break

            img_bgr = pkt_frame.to_ndarray(format="bgr24")
            result = cam.generate_frame(img_bgr, i)

            out_frame = av.VideoFrame.from_ndarray(result, format="bgr24")
            for packet in out_stream.encode(out_frame):
                output_container.mux(packet)

            if progress_callback and i % 30 == 0:
                frac = progress_start + (progress_end - progress_start) * i / limit
                progress_callback(frac, f"Frame {i}/{limit}")

        # Flush encoder
        for packet in out_stream.encode():
            output_container.mux(packet)

    finally:
        output_container.close()
        input_container.close()


# ---------------------------------------------------------------------------
# Convenience wrapper class (matches service layer conventions)
# ---------------------------------------------------------------------------


class PreprocessingService:
    """Thin wrapper that holds project context for preprocessing calls.

    Follows the same pattern as other service classes in this package.
    """

    def __init__(self, storage_path: str, project_name: str) -> None:
        self.storage_path = storage_path
        self.project_name = project_name

    def preprocess_stabilized_camera(
        self,
        video_name: str,
        body_roi_id: int,
        head_roi_id: int,
        fc: float = 0.25,
        order: int = 2,
        margin: int = 75,
        min_crop: int = 300,
        output_size: int = 518,
        preview_duration: float = 10.0,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, object]:
        """Delegate to the module-level :func:`preprocess_stabilized_camera`."""
        return preprocess_stabilized_camera(
            storage_path=self.storage_path,
            project_name=self.project_name,
            video_name=video_name,
            body_roi_id=body_roi_id,
            head_roi_id=head_roi_id,
            fc=fc,
            order=order,
            margin=margin,
            min_crop=min_crop,
            output_size=output_size,
            preview_duration=preview_duration,
            progress_callback=progress_callback,
        )
