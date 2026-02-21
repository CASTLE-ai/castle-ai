"""
castle/core/pipeline_parallel.py
Threaded 3-stage producer-consumer pipeline for feature extraction.

Stage 1 (I/O thread):  VideoReader decode → frame_queue
Stage 2 (CPU thread):  Preprocess (StabilizedCamera.generate_frame) → tensor_queue
Stage 3 (GPU main):    DINOv2 inference → collect latents → return array

Uses threading (NOT multiprocessing) to avoid CUDA fork issues.
Uses queue.Queue with bounded size to control memory.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Optional

import numpy as np

from castle.utils.video_io import VideoReader

logger = logging.getLogger(__name__)

# Sentinel object used to signal stage completion through queues.
_SENTINEL = None


class ParallelExtractor:
    """Three-stage parallel pipeline for feature extraction.

    Stage 1 (I/O thread): VideoReader decode → frame_queue
    Stage 2 (CPU thread): Preprocess (StabilizedCamera.generate_frame) → tensor_queue
    Stage 3 (GPU main):   DINOv2 inference → latent list → returned as (N, D) array

    Parameters
    ----------
    video_path : str
        Path to the source video file.
    stabilized_camera : StabilizedCamera, optional
        Pre-constructed StabilizedCamera instance.  When *None*, frames are
        passed through without preprocessing.
    model : object, optional
        Visual encoder with an ``extract_tensor_batch(frames, masks, roi_id)``
        or ``extract_batch_latent(frames, masks, roi_id)`` method.  When
        *None*, ``run()`` returns the raw preprocessed frames as an array
        (useful for testing).
    batch_size : int
        Number of frames per GPU inference batch.  Default 8.
    queue_size : int
        Maximum number of items buffered in each inter-stage queue.  Default 32.
    roi_id : int
        ROI identifier forwarded to the model's extraction call.  Default 1.
    """

    def __init__(
        self,
        video_path: str,
        stabilized_camera=None,
        model=None,
        batch_size: int = 8,
        queue_size: int = 32,
        roi_id: int = 1,
    ) -> None:
        self.video_path = video_path
        self.stabilized_camera = stabilized_camera
        self.model = model
        self.batch_size = int(batch_size)
        self.queue_size = int(queue_size)
        self.roi_id = int(roi_id)

    # ------------------------------------------------------------------
    # Stage workers (run in background threads)
    # ------------------------------------------------------------------

    def _stage1_io(
        self,
        frame_queue: "queue.Queue[object]",
        error_holder: list,
        error_event: threading.Event,
    ) -> None:
        """Read raw frames from disk and push them onto *frame_queue*."""
        try:
            with VideoReader(self.video_path) as reader:
                total = len(reader)
                for idx in range(total):
                    if error_event.is_set():
                        break
                    frame = reader.get_frame(idx)
                    frame_queue.put((idx, total, frame))
            frame_queue.put(_SENTINEL)
        except Exception as exc:  # noqa: BLE001
            error_holder.append(exc)
            error_event.set()
            frame_queue.put(_SENTINEL)  # unblock stage 2

    def _stage2_preprocess(
        self,
        frame_queue: "queue.Queue[object]",
        tensor_queue: "queue.Queue[object]",
        error_holder: list,
        error_event: threading.Event,
    ) -> None:
        """Apply preprocessing and push tensors onto *tensor_queue*."""
        try:
            while True:
                item = frame_queue.get()
                if item is _SENTINEL:
                    break
                if error_event.is_set():
                    # Drain remaining items so stage 1 is never blocked
                    continue

                idx, total, frame = item
                try:
                    if self.stabilized_camera is not None:
                        processed = self.stabilized_camera.generate_frame(frame, idx)
                    else:
                        processed = frame
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Stage-2 frame %d preprocessing failed: %s", idx, exc)
                    processed = np.zeros_like(frame)

                tensor_queue.put((idx, total, processed))

            tensor_queue.put(_SENTINEL)
        except Exception as exc:  # noqa: BLE001
            error_holder.append(exc)
            error_event.set()
            tensor_queue.put(_SENTINEL)  # unblock stage 3

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> np.ndarray:
        """Run the 3-stage pipeline and return a latent array.

        Parameters
        ----------
        progress_callback : callable, optional
            Called as ``callback(current_frame, total_frames, stage_name)``
            during processing.

        Returns
        -------
        np.ndarray, shape (N, D)
            Stacked latent vectors (one per frame).  When *model* is *None*,
            returns the preprocessed frames stacked along axis 0 instead.

        Raises
        ------
        RuntimeError
            If any pipeline stage raises an unhandled exception.
        """
        frame_queue: queue.Queue[object] = queue.Queue(maxsize=self.queue_size)
        tensor_queue: queue.Queue[object] = queue.Queue(maxsize=self.queue_size)

        error_holder: list = []
        error_event = threading.Event()

        # Launch stage 1 & 2 threads
        t1 = threading.Thread(
            target=self._stage1_io,
            args=(frame_queue, error_holder, error_event),
            name="pipeline-stage1-io",
            daemon=True,
        )
        t2 = threading.Thread(
            target=self._stage2_preprocess,
            args=(frame_queue, tensor_queue, error_holder, error_event),
            name="pipeline-stage2-preprocess",
            daemon=True,
        )
        t1.start()
        t2.start()

        # Stage 3: GPU inference in the main thread (batched)
        latent_list: list = []
        batch_frames: list = []
        batch_indices: list = []
        total_frames = 0

        try:
            while True:
                item = tensor_queue.get()
                if item is _SENTINEL:
                    break

                idx, total_frames, processed = item

                if progress_callback is not None:
                    progress_callback(idx, total_frames, "preprocessing")

                batch_frames.append(processed)
                batch_indices.append(idx)

                if len(batch_frames) >= self.batch_size:
                    latent_list.append(
                        self._run_inference(batch_frames, batch_indices)
                    )
                    if progress_callback is not None:
                        progress_callback(batch_indices[-1], total_frames, "inference")
                    batch_frames = []
                    batch_indices = []

            # Flush the last partial batch
            if batch_frames:
                latent_list.append(
                    self._run_inference(batch_frames, batch_indices)
                )
                if progress_callback is not None and total_frames > 0:
                    progress_callback(total_frames, total_frames, "inference")

        finally:
            # Always join background threads
            t1.join()
            t2.join()

        if error_holder:
            raise RuntimeError(
                f"Pipeline failed with {len(error_holder)} error(s). "
                f"First: {error_holder[0]}"
            ) from error_holder[0]

        if not latent_list:
            logger.warning("ParallelExtractor: no latents produced for %s", self.video_path)
            return np.empty((0,), dtype=np.float32)

        return np.concatenate(latent_list, axis=0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_inference(self, frames: list, indices: list) -> np.ndarray:
        """Run model inference on a batch of preprocessed frames.

        Parameters
        ----------
        frames : list of np.ndarray
            Preprocessed frame arrays.
        indices : list of int
            Corresponding frame indices (for logging).

        Returns
        -------
        np.ndarray, shape (B, D) or (B, H, W, C)
            Latent vectors, or raw stacked frames when *model* is *None*.
        """
        batch = np.stack(frames, axis=0)  # (B, H, W, C)

        if self.model is None:
            # No model: return raw frames (shape stays (B, H, W, C))
            return batch.astype(np.float32)

        # Create dummy masks (all-ones) matching the batch spatial dims
        h, w = batch.shape[1], batch.shape[2]
        masks = np.ones((len(frames), h, w), dtype=np.uint8) * self.roi_id

        try:
            if hasattr(self.model, "extract_tensor_batch"):
                result = self.model.extract_tensor_batch(
                    batch, masks, self.roi_id
                )
            elif hasattr(self.model, "extract_batch_latent"):
                result = self.model.extract_batch_latent(batch, masks, self.roi_id)
            else:
                raise AttributeError(
                    "Model must implement extract_tensor_batch or extract_batch_latent"
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Stage-3 inference failed for frames %s: %s",
                indices,
                exc,
            )
            # Return zero-filled placeholder so we don't lose shape info
            embed_dim = getattr(self.model, "n_feature", 768)
            result = np.zeros((len(frames), embed_dim), dtype=np.float32)

        if isinstance(result, list):
            result = np.array(result)
        return np.asarray(result, dtype=np.float32)
