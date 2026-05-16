"""
castle/utils/video_reader_simple.py
Simplified video reader built on PyAV.

Why this exists
---------------
:class:`VideoReader` in ``video_io.py`` grew to handle many edge-cases
(binary-search fallback, LRU cache, sequential-read optimisation, …).  That
complexity is valuable for production pipelines, but it makes the code hard
to follow and test.

``SimpleVideoReader`` covers the *common* case:

* Open a local video file.
* Read metadata (fps, resolution, frame count).
* Fetch an arbitrary frame by index.
* Iterate efficiently over a range of frames.

If you need caching, fallback handling, or subtitle generation, use the
full :class:`VideoReader` from ``castle.utils.video_io``.

Usage::

    with SimpleVideoReader("video.mp4") as r:
        print(r.fps, r.width, r.height, len(r))
        frame = r.get_frame(0)          # (H, W, 3) BGR uint8
        for idx, frame in r.iter_frames(start=10, end=100, step=5):
            process(frame)
"""

from __future__ import annotations

import logging
from typing import Generator, Optional, Tuple

import av
import numpy as np

logger = logging.getLogger(__name__)


class SimpleVideoReader:
    """Simplified video reader using PyAV.

    No cv2 fallback complexity.  No LRU cache.  Clean API.

    Frames are returned as BGR uint8 arrays of shape ``(H, W, 3)`` to match
    the convention used by the rest of CASTLE (which was originally OpenCV-
    based).

    Args:
        path: Path to the video file.

    Raises:
        FileNotFoundError: If *path* does not exist.
        RuntimeError:      If the container cannot be opened or has no video
                           stream.
    """

    def __init__(self, path: str) -> None:
        import os

        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")

        self.path = path
        self._container = av.open(path)

        streams = self._container.streams.video
        if not streams:
            self._container.close()
            raise RuntimeError(f"No video stream found in: {path}")

        self._stream = streams[0]

        # ── Metadata ────────────────────────────────────────────────
        self.fps: float = float(self._stream.average_rate or 0)
        self.width: int = self._stream.width
        self.height: int = self._stream.height
        self.n_frames: int = self._resolve_frame_count()

        # pts ↔ frame-index conversion factor
        self._pts2idx = self._stream.time_base * self._stream.average_rate

        logger.debug(
            "SimpleVideoReader: %s  %dx%d  %.2f fps  %d frames",
            path,
            self.width,
            self.height,
            self.fps,
            self.n_frames,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_frames

    def __enter__(self) -> "SimpleVideoReader":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def get_frame(self, index: int) -> np.ndarray:
        """Return a single frame by index.

        Args:
            index: Zero-based frame index.

        Returns:
            ``(H, W, 3)`` BGR uint8 NumPy array.

        Raises:
            IndexError:   If *index* is out of range.
            RuntimeError: If the frame cannot be decoded.
        """
        if index < 0 or index >= self.n_frames:
            raise IndexError(
                f"Frame index {index} out of range [0, {self.n_frames})"
            )

        # Seek to keyframe at or before the target
        timestamp = int(index / self._pts2idx)
        self._container.seek(
            timestamp, stream=self._stream, backward=True, any_frame=False
        )

        for frame in self._container.decode(self._stream):
            idx = int(frame.pts * self._pts2idx)
            if idx == index:
                return self._to_bgr(frame)
            if idx > index:
                # Overshot — return the closest available frame
                logger.debug(
                    "SimpleVideoReader.get_frame: overshot at idx=%d (wanted %d)",
                    idx,
                    index,
                )
                return self._to_bgr(frame)

        raise RuntimeError(f"Could not decode frame {index} from {self.path}")

    def iter_frames(
        self,
        start: int = 0,
        end: Optional[int] = None,
        step: int = 1,
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Iterate over frames in ``[start, end)`` with the given step.

        For *step == 1* the iteration is fully sequential (no seek per frame),
        which is the most efficient access pattern for PyAV.  For *step > 1*
        we seek to each frame individually.

        Args:
            start: First frame index (inclusive).
            end:   Last frame index (exclusive).  Defaults to :attr:`n_frames`.
            step:  Step between returned frames.  Must be ≥ 1.

        Yields:
            ``(index, frame)`` tuples where *frame* is a ``(H, W, 3)`` BGR
            uint8 array.
        """
        if end is None:
            end = self.n_frames
        end = min(end, self.n_frames)

        if step < 1:
            raise ValueError("step must be >= 1")

        if step == 1:
            yield from self._iter_sequential(start, end)
        else:
            for idx in range(start, end, step):
                yield idx, self.get_frame(idx)

    def close(self) -> None:
        """Release the underlying PyAV container.

        Cleanup-phase exceptions are logged at debug level but never re-raised
        — raising here would mask the original error that triggered the close.
        """
        container = getattr(self, "_container", None)
        if container is not None:
            try:
                container.close()
            except Exception as exc:  # noqa: BLE001 — cleanup must not mask original error
                logger.debug("PyAV container.close() failed during cleanup: %s", exc)
            self._container = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_frame_count(self) -> int:
        """Return the total frame count, preferring the stream metadata."""
        count = self._stream.frames
        if count and count > 0:
            return int(count)

        # Fallback: use duration × fps
        if self._stream.duration and self._stream.time_base and self.fps:
            duration_sec = float(
                self._stream.duration * self._stream.time_base
            )
            estimated = int(round(duration_sec * self.fps))
            logger.debug(
                "SimpleVideoReader: stream.frames unavailable; "
                "estimated %d frames from duration",
                estimated,
            )
            return estimated

        logger.warning(
            "SimpleVideoReader: cannot determine frame count for %s; defaulting to 0",
            self.path,
        )
        return 0

    def _iter_sequential(
        self, start: int, end: int
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Efficient sequential iteration without per-frame seeks."""
        timestamp = int(start / self._pts2idx)
        self._container.seek(
            timestamp, stream=self._stream, backward=True, any_frame=False
        )

        for frame in self._container.decode(self._stream):
            idx = int(frame.pts * self._pts2idx)
            if idx < start:
                continue
            if idx >= end:
                break
            yield idx, self._to_bgr(frame)

    @staticmethod
    def _to_bgr(frame: av.VideoFrame) -> np.ndarray:
        """Convert a PyAV VideoFrame to a (H, W, 3) BGR uint8 array."""
        rgb = frame.to_rgb().to_ndarray()  # (H, W, 3) RGB uint8
        return rgb[:, :, ::-1].copy()      # flip to BGR, make contiguous
