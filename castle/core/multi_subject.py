"""Multi-subject tracking and analysis management.

Provides :class:`SubjectTrack` — a container for a single subject's tracking
data — and :class:`MultiSubjectProject` which orchestrates independent
preprocessing, feature extraction, and clustering for every subject in a
shared video.

Usage example::

    project = MultiSubjectProject("/data/projects/social_session", "video01.mp4")
    project.add_subject(subject_id=0, body_roi=1, head_roi=2)
    project.add_subject(subject_id=1, body_roi=3, head_roi=4)
    project.process_all()
    tracks = project.get_subjects()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from castle.core.project_data import ProjectData
from castle.core.stabilized_camera import (
    extract_centroids_from_masks,
    extract_orientations_from_masks,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class SubjectTrack:
    """Single subject's tracking data within a multi-subject video.

    Attributes
    ----------
    subject_id : int
        Unique integer identifier for this subject.
    body_roi_id : int
        ROI pixel-value in the mask HDF5 corresponding to the body.
    head_roi_id : int
        ROI pixel-value in the mask HDF5 corresponding to the head.
    positions : np.ndarray, shape (N, 2)
        Raw centroid [x, y] per frame (pixels).
    angles : np.ndarray, shape (N,)
        Unwrapped orientation in degrees (body→head vector).
    latents : np.ndarray or None, shape (N, D)
        Feature vectors extracted from stabilised frames; set after extraction.
    labels : np.ndarray or None, shape (N,)
        Per-frame cluster assignments; set after clustering.
    """

    subject_id: int
    body_roi_id: int
    head_roi_id: int
    positions: np.ndarray  # (N, 2)
    angles: np.ndarray  # (N,)
    latents: Optional[np.ndarray] = field(default=None)  # (N, D)
    labels: Optional[np.ndarray] = field(default=None)  # (N,)

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=np.float64)
        self.angles = np.asarray(self.angles, dtype=np.float64)
        if self.latents is not None:
            self.latents = np.asarray(self.latents, dtype=np.float64)
        if self.labels is not None:
            self.labels = np.asarray(self.labels, dtype=np.int64)

    @property
    def n_frames(self) -> int:
        """Number of tracked frames."""
        return len(self.positions)

    def set_latents(self, latents: np.ndarray) -> None:
        """Assign extracted latent features.

        Parameters
        ----------
        latents : np.ndarray, shape (N, D)
            Feature matrix produced by a feature extractor.
        """
        if latents.shape[0] != self.n_frames:
            raise ValueError(
                f"Subject {self.subject_id}: latents length {latents.shape[0]} "
                f"!= n_frames {self.n_frames}"
            )
        self.latents = np.asarray(latents, dtype=np.float64)

    def set_labels(self, labels: np.ndarray) -> None:
        """Assign per-frame cluster labels.

        Parameters
        ----------
        labels : np.ndarray, shape (N,)
            Integer cluster assignments produced by a clustering step.
        """
        if labels.shape[0] != self.n_frames:
            raise ValueError(
                f"Subject {self.subject_id}: labels length {labels.shape[0]} "
                f"!= n_frames {self.n_frames}"
            )
        self.labels = np.asarray(labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# Project manager
# ---------------------------------------------------------------------------


class MultiSubjectProject:
    """Manages multiple subjects tracked within a single video.

    Each subject is defined by a pair of ROI IDs (body + head) in the shared
    mask HDF5 file.  After registering subjects via :meth:`add_subject`, call
    :meth:`process_all` to populate positions and angles for every subject.
    Feature extraction and clustering must be applied externally (the
    :class:`SubjectTrack` objects are mutable containers).

    Parameters
    ----------
    project_path : str
        Path to the CASTLE project root directory (must contain ``config.json``
        and a ``track/<video_name>/mask_list.h5`` file).
    video_name : str
        Basename of the source video (e.g. ``"session01.mp4"``).
    fps : float, optional
        Frames per second.  If *None* the value is read from the project's
        video list; falls back to 30.0 if unavailable.
    """

    def __init__(
        self,
        project_path: str,
        video_name: str,
        fps: Optional[float] = None,
    ) -> None:
        self._project = ProjectData.from_path(project_path)
        self._video_name = video_name
        self._mask_h5_path = str(self._project.mask_h5_path(video_name))

        # Resolve FPS
        if fps is not None:
            self._fps = float(fps)
        else:
            self._fps = self._detect_fps()

        # Registered subject specs: subject_id → (body_roi, head_roi)
        self._subject_specs: dict[int, tuple[int, int]] = {}
        # Fully built tracks (populated by process_all)
        self._tracks: dict[int, SubjectTrack] = {}

        logger.info(
            "MultiSubjectProject: project=%s, video=%s, fps=%.2f, mask_h5=%s",
            self._project.name,
            video_name,
            self._fps,
            self._mask_h5_path,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def project(self) -> ProjectData:
        """Underlying :class:`~castle.core.project_data.ProjectData`."""
        return self._project

    @property
    def video_name(self) -> str:
        """Source video basename."""
        return self._video_name

    @property
    def fps(self) -> float:
        """Frames per second."""
        return self._fps

    @property
    def mask_h5_path(self) -> str:
        """Path to the shared mask HDF5 file."""
        return self._mask_h5_path

    # ------------------------------------------------------------------
    # Subject registration
    # ------------------------------------------------------------------

    def add_subject(self, subject_id: int, body_roi: int, head_roi: int) -> None:
        """Register a subject with its body and head ROI IDs.

        Must be called before :meth:`process_all`.  Re-registering an existing
        ``subject_id`` with new ROI IDs will overwrite the previous spec (and
        clear any existing track for that subject).

        Parameters
        ----------
        subject_id : int
            Unique identifier for this subject.
        body_roi : int
            ROI pixel value for the body region in the mask HDF5.
        head_roi : int
            ROI pixel value for the head region in the mask HDF5.
        """
        if subject_id in self._subject_specs:
            logger.warning(
                "MultiSubjectProject: overwriting subject_id=%d (was body=%d, head=%d)",
                subject_id,
                *self._subject_specs[subject_id],
            )
            self._tracks.pop(subject_id, None)
        self._subject_specs[subject_id] = (int(body_roi), int(head_roi))
        logger.debug(
            "MultiSubjectProject: registered subject_id=%d body_roi=%d head_roi=%d",
            subject_id,
            body_roi,
            head_roi,
        )

    # ------------------------------------------------------------------
    # Subject access
    # ------------------------------------------------------------------

    def get_subjects(self) -> list[SubjectTrack]:
        """Return a list of all processed :class:`SubjectTrack` objects.

        Returns an empty list if :meth:`process_all` has not been called yet.

        Returns
        -------
        list[SubjectTrack]
            Tracks in ascending ``subject_id`` order.
        """
        return [self._tracks[sid] for sid in sorted(self._tracks)]

    def get_subject(self, subject_id: int) -> SubjectTrack:
        """Return the :class:`SubjectTrack` for a given *subject_id*.

        Parameters
        ----------
        subject_id : int
            Subject identifier previously registered via :meth:`add_subject`.

        Raises
        ------
        KeyError
            If *subject_id* is not registered or :meth:`process_all` has not
            been called.
        """
        if subject_id not in self._tracks:
            if subject_id in self._subject_specs:
                raise KeyError(
                    f"subject_id={subject_id} is registered but "
                    "process_all() has not been called yet."
                )
            raise KeyError(f"subject_id={subject_id} is not registered.")
        return self._tracks[subject_id]

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process_all(
        self,
        n_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Extract positions and orientations for all registered subjects.

        For each registered subject the method:

        1. Calls :func:`~castle.core.stabilized_camera.extract_centroids_from_masks`
           to build the per-frame position array.
        2. Calls
           :func:`~castle.core.stabilized_camera.extract_orientations_from_masks`
           to build the unwrapped angle array.
        3. Stores the result as a :class:`SubjectTrack`.

        Optionally reports progress via *progress_callback*.

        Parameters
        ----------
        n_frames : int, optional
            Total number of frames to process.  If *None*, inferred from the
            first successfully extracted subject.
        progress_callback : callable(current, total) or None
            Called after each subject is processed with ``(current_index,
            total_subjects)``.
        """
        if not self._subject_specs:
            logger.warning("MultiSubjectProject.process_all: no subjects registered.")
            return

        if not Path(self._mask_h5_path).exists():
            raise FileNotFoundError(
                f"Mask HDF5 file not found: {self._mask_h5_path}"
            )

        # Determine n_frames from metadata if not provided
        resolved_n_frames = n_frames or self._infer_n_frames()
        if resolved_n_frames <= 0:
            raise ValueError(
                "Cannot determine n_frames. Pass it explicitly to process_all()."
            )

        total = len(self._subject_specs)
        for idx, (subject_id, (body_roi, head_roi)) in enumerate(
            sorted(self._subject_specs.items())
        ):
            logger.info(
                "MultiSubjectProject: processing subject_id=%d "
                "(body_roi=%d, head_roi=%d) [%d/%d]",
                subject_id,
                body_roi,
                head_roi,
                idx + 1,
                total,
            )

            positions = extract_centroids_from_masks(
                self._mask_h5_path,
                roi_id=body_roi,
                n_frames=resolved_n_frames,
            )
            angles = extract_orientations_from_masks(
                self._mask_h5_path,
                body_roi_id=body_roi,
                head_roi_id=head_roi,
                n_frames=resolved_n_frames,
            )

            self._tracks[subject_id] = SubjectTrack(
                subject_id=subject_id,
                body_roi_id=body_roi,
                head_roi_id=head_roi,
                positions=positions,
                angles=angles,
            )

            if progress_callback is not None:
                progress_callback(idx + 1, total)

        logger.info(
            "MultiSubjectProject.process_all: completed %d subjects.", total
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_fps(self) -> float:
        """Attempt to read FPS from the project's video list."""
        try:
            for video_info in self._project.list_videos():
                if video_info.name == self._video_name and video_info.fps > 0:
                    logger.debug(
                        "MultiSubjectProject: detected fps=%.2f from video metadata.",
                        video_info.fps,
                    )
                    return float(video_info.fps)
        except Exception as exc:  # noqa: BLE001
            logger.debug("MultiSubjectProject: fps detection failed: %s", exc)
        logger.warning(
            "MultiSubjectProject: cannot detect fps for '%s'; defaulting to 30.0.",
            self._video_name,
        )
        return 30.0

    def _infer_n_frames(self) -> int:
        """Try to infer n_frames from video metadata."""
        try:
            for video_info in self._project.list_videos():
                if video_info.name == self._video_name and video_info.n_frames > 0:
                    return int(video_info.n_frames)
        except Exception as exc:  # noqa: BLE001
            logger.debug("MultiSubjectProject: n_frames inference failed: %s", exc)
        return 0
