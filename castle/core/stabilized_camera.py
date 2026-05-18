"""Stabilized Virtual Camera preprocessing module.

Implements zero-phase Butterworth low-pass filtering of centroid trajectories
and orientation angles, followed by dynamic crop extraction and resize to a
fixed output resolution suitable for DINOv2 feature extraction.

Theory & parameter justification:
    /mnt/AI-Assistant/260210-lowpass-justification/CASTLE_Video_Preprocessing_Justification.md

Pipeline
--------
centroid x(t), orientation θ(t)
        ↓
Zero-phase 2nd-order Butterworth LP (fc = 0.25 Hz, filtfilt)
        ↓
x_c(t), θ_c(t)  — smooth camera trajectory
        ↓
dist = ‖x(t) − x_c(t)‖
crop_size = max(300, 2 × (dist + 75))   [px]
        ↓
warpAffine: translate to x_c, rotate by θ_c − 90°
        ↓
Resize → 518 × 518
        ↓
DINOv2
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import scipy.signal

logger = logging.getLogger(__name__)


class StabilizedCamera:
    """Stabilized virtual camera for animal tracking video preprocessing.

    Applies zero-phase Butterworth low-pass filtering to centroid positions
    and orientations, then extracts dynamically-sized crops centred on the
    filtered trajectory, finally resizing to a fixed output resolution.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 2)
        Raw centroid [x, y] per frame (pixels).
    angles : np.ndarray, shape (N,)
        Unwrapped orientation in degrees.
    fps : float
        Video frame rate (Hz).
    fc : float
        Low-pass cutoff frequency in Hz. Default 0.25 Hz (period ≈ 4 s).
    order : int
        Butterworth filter order. Default 2.
    margin : int
        Spatial margin added around the HP residual displacement when computing
        the dynamic crop window (pixels). Default 75 px.
    min_crop : int
        Minimum crop window side length (pixels). Default 300 px.
    output_size : int
        Side length of the square output frame (pixels). Default 518 px
        (= 37 × 14, optimal for DINOv2 ViT-B/14).
    """

    def __init__(
        self,
        positions: np.ndarray,
        angles: np.ndarray,
        fps: float,
        fc: float = 0.25,
        order: int = 2,
        margin: int = 75,
        min_crop: int = 300,
        output_size: int = 518,
    ) -> None:
        self.positions = np.asarray(positions, dtype=np.float64)
        self.angles = np.asarray(angles, dtype=np.float64)
        self.fps = float(fps)
        self.fc = float(fc)
        self.order = int(order)
        self.margin = int(margin)
        self.min_crop = int(min_crop)
        self.output_size = int(output_size)

        # Computed by compute_trajectory()
        self.pos_filtered: np.ndarray = np.empty(0)
        self.angle_filtered: np.ndarray = np.empty(0)
        self.crop_sizes: np.ndarray = np.empty(0)

        # Design filter coefficients once
        nyquist = 0.5 * self.fps
        fc_safe = self.fc
        if fc_safe >= nyquist:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                f"Cutoff frequency fc={fc_safe} Hz >= Nyquist={nyquist} Hz "
                f"(fps={self.fps}); clamping to {nyquist * 0.99:.4f} Hz."
            )
            fc_safe = nyquist * 0.99
        normal_cutoff = fc_safe / nyquist
        self._b, self._a = scipy.signal.butter(
            self.order, normal_cutoff, btype="low", analog=False
        )

        self.compute_trajectory()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _center_crop(img: np.ndarray, crop_size: int) -> np.ndarray:
        """Crop a square region of ``crop_size`` from the centre of ``img``.

        Zero-pads (BORDER_CONSTANT behaviour) if ``crop_size`` exceeds the
        image dimension in either axis — matches cv2.BORDER_CONSTANT filling
        used when the tracked centroid is near the frame edge.

        Args:
            img: Input array, shape ``(H, W)`` or ``(H, W, C)``.
            crop_size: Side length of the square crop.

        Returns:
            Array of shape ``(crop_size, crop_size)`` or
            ``(crop_size, crop_size, C)``, dtype preserved.
        """
        h, w = img.shape[:2]
        cy_c, cx_c = h // 2, w // 2
        half = crop_size // 2

        y0 = cy_c - half
        y1 = y0 + crop_size
        x0 = cx_c - half
        x1 = x0 + crop_size

        # Valid source bounds
        y0s, y1s = max(0, y0), min(h, y1)
        x0s, x1s = max(0, x0), min(w, x1)

        if img.ndim == 3:
            out = np.zeros((crop_size, crop_size, img.shape[2]), dtype=img.dtype)
        else:
            out = np.zeros((crop_size, crop_size), dtype=img.dtype)

        out[y0s - y0:y0s - y0 + (y1s - y0s),
            x0s - x0:x0s - x0 + (x1s - x0s)] = img[y0s:y1s, x0s:x1s]
        return out

    def _get_warp_params(
        self,
        frame_idx: int,
        frame_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, int]:
        """Compute the shared warpAffine matrix and crop size for one frame.

        The returned matrix ``M`` maps the filtered centroid to the geometric
        centre of a canvas of size ``frame_shape``.  Applying
        ``cv2.warpAffine(img, M, (frame_shape[1], frame_shape[0]))`` followed
        by :meth:`_center_crop` with the returned ``crop_size`` reproduces the
        same visible region as the original pipeline — but both frames and masks
        can share the same ``M``, guaranteeing pixel-perfect alignment.

        Args:
            frame_idx: Zero-based frame index.
            frame_shape: ``(H, W)`` of the image to be warped — passed
                explicitly so this method remains stateless and callers can
                assert alignment independently.

        Returns:
            ``(M, crop_size)`` where ``M`` is a ``(2, 3)`` float64 affine
            matrix and ``crop_size`` is the integer side length of the square
            crop to apply after warping.
        """
        h, w = frame_shape
        cx = float(self.pos_filtered[frame_idx, 0])
        cy = float(self.pos_filtered[frame_idx, 1])
        angle_deg = float(self.angle_filtered[frame_idx]) - 90.0
        crop_size = int(self.crop_sizes[frame_idx])

        # Rotation matrix: rotates around (cx, cy)
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

        # Translate so that the filtered centroid lands at the canvas centre.
        # _center_crop then extracts the crop_size × crop_size region around
        # that centre — equivalent to the original (crop_size, crop_size)
        # warpAffine destination but avoids a separate crop pipeline per
        # image type (frame vs mask).
        M[0, 2] += w / 2.0 - cx
        M[1, 2] += h / 2.0 - cy

        return M, crop_size

    @staticmethod
    def _interpolate_nans(data: np.ndarray) -> np.ndarray:
        """Replace NaN values with linear interpolation.

        For leading/trailing NaNs, uses nearest valid value (forward/back fill).
        """
        if data.ndim == 1:
            mask = np.isnan(data)
            if not mask.any():
                return data
            valid = np.where(~mask)[0]
            if len(valid) == 0:
                data[:] = 0.0
                return data
            data[mask] = np.interp(
                np.where(mask)[0], valid, data[valid]
            )
        else:
            for col in range(data.shape[1]):
                mask = np.isnan(data[:, col])
                if not mask.any():
                    continue
                valid = np.where(~mask)[0]
                if len(valid) == 0:
                    data[:, col] = 0.0
                    continue
                data[mask, col] = np.interp(
                    np.where(mask)[0], valid, data[valid, col]
                )
        return data

    def _apply_zero_phase_lowpass(self, data: np.ndarray) -> np.ndarray:
        """Apply zero-phase (filtfilt) Butterworth low-pass filter.

        Uses ``scipy.signal.filtfilt`` for zero phase delay. Falls back to
        ``method='gust'`` when the signal is too short for the default
        padding length.

        Parameters
        ----------
        data : np.ndarray
            1-D or 2-D (N, C) array to filter along axis 0.

        Returns
        -------
        np.ndarray
            Filtered data with the same shape as ``data``.
        """
        n = data.shape[0]
        # filtfilt default padlen = 3 * max(len(a), len(b)) − 1
        default_padlen = 3 * max(len(self._a), len(self._b)) - 1

        if n <= default_padlen:
            # Too short for default padding — use Gustafsson method which
            # handles short sequences gracefully.
            logger.warning(
                "Signal length %d ≤ default filtfilt padlen %d; "
                "switching to method='gust'.",
                n,
                default_padlen,
            )
            try:
                return scipy.signal.filtfilt(
                    self._b, self._a, data, axis=0, method="gust"
                )
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "filtfilt method='gust' failed (%s); returning raw data.", exc
                )
                return data.copy()

        return scipy.signal.filtfilt(self._b, self._a, data, axis=0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_trajectory(self) -> None:
        """Filter positions and angles; pre-compute per-frame crop sizes.

        Populates ``self.pos_filtered``, ``self.angle_filtered``, and
        ``self.crop_sizes``.
        """
        logger.debug(
            "Computing trajectory: N=%d, fc=%.3f Hz, order=%d",
            len(self.positions),
            self.fc,
            self.order,
        )

        # Interpolate NaN values before filtering
        pos_clean = self._interpolate_nans(self.positions.copy())
        ang_clean = self._interpolate_nans(self.angles.copy())

        self.pos_filtered = self._apply_zero_phase_lowpass(pos_clean)
        self.angle_filtered = self._apply_zero_phase_lowpass(ang_clean)

        # Vectorised crop-size computation
        residuals = self.positions - self.pos_filtered  # (N, 2)
        dists = np.linalg.norm(residuals, axis=1)       # (N,)
        raw_sizes = 2.0 * (dists + self.margin)
        self.crop_sizes = np.maximum(self.min_crop, raw_sizes).astype(np.int32)

        logger.debug(
            "Trajectory computed: crop_size min=%d, median=%d, max=%d",
            int(self.crop_sizes.min()),
            int(np.median(self.crop_sizes)),
            int(self.crop_sizes.max()),
        )

    def get_crop_size(self, frame_idx: int) -> int:
        """Return the dynamic crop window size for a given frame.

        Parameters
        ----------
        frame_idx : int
            Zero-based frame index.

        Returns
        -------
        int
            ``max(min_crop, int(2 * (dist + margin)))`` where *dist* is the
            L2 norm of the high-pass residual at that frame.
        """
        dist = float(
            np.linalg.norm(self.positions[frame_idx] - self.pos_filtered[frame_idx])
        )
        return max(self.min_crop, int(2 * (dist + self.margin)))

    def generate_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Extract a stabilised, rotated, and resized crop for one video frame.

        Uses :meth:`_get_warp_params` to compute the shared affine matrix so
        that the identical ``M`` can also be used by :meth:`generate_mask` —
        guaranteeing frame/mask pixel alignment after KIT transformation.

        Args:
            frame: BGR image array, shape ``(H, W, 3)``.
            frame_idx: Zero-based frame index.

        Returns:
            Processed frame, shape ``(output_size, output_size, 3)``, uint8.
        """
        h, w = frame.shape[:2]
        M, crop_size = self._get_warp_params(frame_idx, (h, w))

        warped = cv2.warpAffine(
            frame,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        cropped = self._center_crop(warped, crop_size)
        return cv2.resize(
            cropped,
            (self.output_size, self.output_size),
            interpolation=cv2.INTER_LINEAR,
        )

    def generate_mask(self, mask: np.ndarray, frame_idx: int) -> np.ndarray:
        """Apply the same warpAffine as :meth:`generate_frame` to a mask.

        Uses INTER_NEAREST interpolation to preserve integer label integrity.
        Because both methods share :meth:`_get_warp_params`, the resulting mask
        is in the same ``output_size × output_size`` coordinate space as the
        corresponding processed frame — they are always pixel-aligned.

        This is the correct approach when KIT is combined with ROI-weighted
        pooling: after stabilisation the coordinate space changes, so the mask
        must be transformed by the same affine ``M`` before pooling.

        Args:
            mask: Integer mask in original video coordinates, shape ``(H, W)``.
                Pixel values encode ROI labels (0 = background).
            frame_idx: Zero-based frame index for trajectory lookup.

        Returns:
            Transformed mask, shape ``(output_size, output_size)``, same dtype
            as input.
        """
        h, w = mask.shape[:2]
        M, crop_size = self._get_warp_params(frame_idx, (h, w))

        warped = cv2.warpAffine(
            mask,
            M,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        cropped = self._center_crop(warped, crop_size)
        return cv2.resize(
            cropped,
            (self.output_size, self.output_size),
            interpolation=cv2.INTER_NEAREST,
        )

    def get_diagnostics(self) -> Dict[str, object]:
        """Return a dictionary of diagnostic metrics for the stabilisation.

        Returns
        -------
        dict with keys:
            crop_sizes : np.ndarray, shape (N,)
                Per-frame crop window sizes (pixels).
            hp_residual_rms : float
                RMS of the high-pass positional residual (pixels).
            pct_at_min_crop : float
                Percentage of frames where ``crop_size == min_crop``.
            speed_crop_correlation : float
                Pearson r between per-frame speed (pixels/frame) and crop size.
        """
        residuals = self.positions - self.pos_filtered
        hp_residual_rms = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))

        pct_at_min_crop = float(
            100.0 * np.mean(self.crop_sizes == self.min_crop)
        )

        # Per-frame speed: L2 distance between consecutive raw positions
        if len(self.positions) > 1:
            diffs = np.diff(self.positions, axis=0)
            speeds = np.linalg.norm(diffs, axis=1)
            # Align lengths (speed has N-1 entries)
            if len(speeds) == len(self.crop_sizes) - 1:
                speeds = np.append(speeds, speeds[-1])
            corr_matrix = np.corrcoef(speeds, self.crop_sizes)
            speed_crop_correlation = float(corr_matrix[0, 1])
        else:
            speed_crop_correlation = float("nan")

        return {
            "crop_sizes": self.crop_sizes,
            "hp_residual_rms": hp_residual_rms,
            "pct_at_min_crop": pct_at_min_crop,
            "speed_crop_correlation": speed_crop_correlation,
        }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def extract_centroids_from_masks(
    mask_h5_path: str,
    roi_id: int,
    n_frames: int,
) -> np.ndarray:
    """Extract per-frame centroids of a ROI from a mask HDF5 file.

    Uses connected components to find the largest component of the specified
    ROI value in each mask frame. Missing frames are handled by linear
    interpolation between valid neighbours.

    Parameters
    ----------
    mask_h5_path : str
        Path to the mask HDF5 file (written by :class:`castle.utils.h5_io.H5IO`).
    roi_id : int
        Integer pixel value identifying the ROI in the mask.
    n_frames : int
        Total number of frames to extract (0 … n_frames−1).

    Returns
    -------
    np.ndarray, shape (n_frames, 2)
        Array of [x, y] centroid coordinates (float64).
        Frames with no mask data are linearly interpolated.
    """
    from castle.utils.h5_io import H5IO  # local import to avoid circular deps

    positions = np.full((n_frames, 2), np.nan, dtype=np.float64)

    with H5IO(mask_h5_path) as h5:
        for i in range(n_frames):
            if not h5.has_mask(i):
                logger.debug("extract_centroids: frame %d missing mask, will interpolate", i)
                continue
            try:
                mask = h5.read_mask(i)
            except Exception as exc:
                logger.warning("extract_centroids: cannot read frame %d (%s)", i, exc)
                continue

            # Isolate the requested ROI
            binary = (mask == int(roi_id)).astype(np.uint8)
            if binary.sum() == 0:
                logger.debug("extract_centroids: frame %d ROI %d absent", i, roi_id)
                continue

            num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
                binary, connectivity=8, ltype=cv2.CV_32S
            )
            if num_labels <= 1:
                continue

            areas = [stats[j, cv2.CC_STAT_AREA] for j in range(1, num_labels)]
            best = int(np.argmax(areas)) + 1  # +1 because label 0 is background
            positions[i, 0] = centroids[best, 0]
            positions[i, 1] = centroids[best, 1]

    # Interpolate NaN frames
    valid = np.where(~np.isnan(positions[:, 0]))[0]
    if len(valid) == 0:
        raise ValueError(
            f"No valid centroids found for roi_id={roi_id} in {mask_h5_path}"
        )

    for col in range(2):
        nan_mask = np.isnan(positions[:, col])
        if nan_mask.any():
            positions[:, col] = np.interp(
                np.arange(n_frames),
                valid,
                positions[valid, col],
            )

    logger.info(
        "extract_centroids: roi=%d, valid frames %d/%d (%.1f%%)",
        roi_id,
        len(valid),
        n_frames,
        100.0 * len(valid) / n_frames,
    )
    return positions


def extract_orientations_from_masks(
    mask_h5_path: str,
    body_roi_id: int,
    head_roi_id: int,
    n_frames: int,
) -> np.ndarray:
    """Compute unwrapped orientation angles from body→head vector.

    Extracts the centroid of *body_roi_id* and *head_roi_id* from each mask
    frame, computes the angle of the body→head vector, then unwraps and
    returns degrees.

    Parameters
    ----------
    mask_h5_path : str
        Path to the mask HDF5 file.
    body_roi_id : int
        ROI id for the body (tail-base or torso centroid).
    head_roi_id : int
        ROI id for the head.
    n_frames : int
        Total number of frames.

    Returns
    -------
    np.ndarray, shape (n_frames,)
        Unwrapped orientation in degrees (float64).
    """
    body_pos = extract_centroids_from_masks(mask_h5_path, body_roi_id, n_frames)
    head_pos = extract_centroids_from_masks(mask_h5_path, head_roi_id, n_frames)

    dx = head_pos[:, 0] - body_pos[:, 0]
    dy = head_pos[:, 1] - body_pos[:, 1]

    angles_rad = np.arctan2(dy, dx)  # (N,)
    angles_unwrapped_rad = np.unwrap(angles_rad)
    angles_deg = np.rad2deg(angles_unwrapped_rad)

    logger.info(
        "extract_orientations: body_roi=%d → head_roi=%d, angle range [%.1f°, %.1f°]",
        body_roi_id,
        head_roi_id,
        float(angles_deg.min()),
        float(angles_deg.max()),
    )
    return angles_deg


def preview_stabilization(
    video_path: str,
    positions: np.ndarray,
    angles: np.ndarray,
    fps: float,
    duration: float = 10.0,
    **kwargs,
) -> str:
    """Generate a short stabilised preview video using H.264 encoding (av).

    Parameters
    ----------
    video_path : str
        Path to the source video file.
    positions : np.ndarray, shape (N, 2)
        Raw centroid [x, y] per frame.
    angles : np.ndarray, shape (N,)
        Unwrapped orientation in degrees.
    fps : float
        Video frame rate.
    duration : float
        Duration of the preview in seconds. Default 10 s.
    **kwargs
        Additional keyword arguments forwarded to :class:`StabilizedCamera`.

    Returns
    -------
    str
        Path to the generated preview MP4 file.
    """
    try:
        import av  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'av' package is required for preview_stabilization. "
            "Install it with: pip install av"
        ) from exc

    cam = StabilizedCamera(positions, angles, fps, **kwargs)

    output_size = cam.output_size
    max_frames = int(fps * duration)

    # Build output path next to input video
    src = Path(video_path)
    out_path = str(src.parent / f"{src.stem}_stabilized_preview.mp4")

    input_container = av.open(video_path)
    input_stream = input_container.streams.video[0]

    output_container = av.open(out_path, mode="w")
    out_stream = output_container.add_stream("h264", rate=int(fps))
    out_stream.width = output_size
    out_stream.height = output_size
    out_stream.pix_fmt = "yuv420p"
    out_stream.options = {"crf": "18", "preset": "slow"}

    logger.info("preview_stabilization: generating '%s' (%.1f s)", out_path, duration)

    try:
        for i, pkt_frame in enumerate(input_container.decode(input_stream)):
            if i >= max_frames:
                break
            if i >= len(positions):
                break

            img_bgr = pkt_frame.to_ndarray(format="bgr24")
            result = cam.generate_frame(img_bgr, i)

            out_frame = av.VideoFrame.from_ndarray(result, format="bgr24")
            for packet in out_stream.encode(out_frame):
                output_container.mux(packet)

            if i % 60 == 0:
                logger.debug("preview_stabilization: frame %d/%d", i, max_frames)

        # Flush encoder
        for packet in out_stream.encode():
            output_container.mux(packet)

    finally:
        output_container.close()
        input_container.close()

    logger.info("preview_stabilization: saved '%s'", out_path)
    return out_path
