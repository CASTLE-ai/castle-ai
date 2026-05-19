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
# Use a unique object() rather than None so legitimate None payloads cannot collide.
_SENTINEL = object()


class ParallelExtractor:
    """Three-stage parallel pipeline for feature extraction.

    Stage 1 (I/O thread): VideoReader decode (+ optional H5IO mask read) → frame_queue
    Stage 2 (CPU thread): Preprocess frame (+ optional mask) → tensor_queue
    Stage 3 (GPU main):   DINOv2 inference with ROI-weighted pooling → return (N, D)

    When *mask_path* and *stabilized_camera* are both provided, Stage 2 applies
    :meth:`StabilizedCamera.generate_mask` with the same affine matrix as
    :meth:`StabilizedCamera.generate_frame`, guaranteeing pixel-perfect frame/mask
    alignment in the KIT (Kinematics Info Transfusion) pipeline.

    Parameters
    ----------
    video_path : str
        Path to the source video file.
    stabilized_camera : StabilizedCamera, optional
        Pre-constructed StabilizedCamera instance.  When *None*, frames are
        passed through without preprocessing.
    mask_path : str, optional
        Path to the HDF5 mask file (``mask_list.h5``).  When provided, Stage 1
        reads the corresponding mask for every frame and Stage 2 applies the
        same KIT transform to it.  The transformed mask is then used for
        ROI-weighted pooling in Stage 3 instead of a dummy all-ROI mask.
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
        mask_path: Optional[str] = None,
        model=None,
        batch_size: int = 8,
        queue_size: int = 32,
        roi_id: int = 1,
    ) -> None:
        self.video_path = video_path
        self.stabilized_camera = stabilized_camera
        self.mask_path = mask_path
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
        """Read raw frames (and optionally masks) from disk.

        When ``self.mask_path`` is set, opens the HDF5 mask file alongside the
        video reader and reads the mask for each frame.  Missing masks produce
        ``None`` entries which Stage 2 handles gracefully.
        """
        try:
            h5_ctx = None
            if self.mask_path is not None:
                from castle.utils.h5_io import H5IO
                h5_ctx = H5IO(self.mask_path, read_only=True)

            try:
                with VideoReader(self.video_path) as reader:
                    total = len(reader)
                    for idx in range(total):
                        if error_event.is_set():
                            break
                        frame = reader.get_frame(idx)
                        mask = None
                        if h5_ctx is not None:
                            try:
                                mask = h5_ctx.read_mask(idx)
                            except Exception as exc:  # noqa: BLE001
                                logger.debug("Stage-1: mask read failed frame %d: %s", idx, exc)
                        frame_queue.put((idx, total, frame, mask))
            finally:
                if h5_ctx is not None:
                    try:
                        h5_ctx.close()
                    except Exception:  # noqa: BLE001
                        pass

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
        """Apply StabilizedCamera transform to frame and (optionally) mask.

        When KIT is active (``stabilized_camera`` + ``mask_path`` both set),
        both frame and mask are transformed with the **same** affine matrix via
        :meth:`_get_warp_params` — ensuring pixel-perfect alignment between the
        stabilised frame and the transformed ROI mask.
        """
        try:
            while True:
                item = frame_queue.get()
                if item is _SENTINEL:
                    break
                if error_event.is_set():
                    # Drain remaining items so stage 1 is never blocked
                    continue

                idx, total, frame, mask = item
                try:
                    if self.stabilized_camera is not None:
                        processed_frame = self.stabilized_camera.generate_frame(frame, idx)
                        processed_mask = (
                            self.stabilized_camera.generate_mask(mask, idx)
                            if mask is not None
                            else None
                        )
                    else:
                        processed_frame = frame
                        processed_mask = mask
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Stage-2 frame %d preprocessing failed: %s", idx, exc)
                    processed_frame = np.zeros_like(frame)
                    processed_mask = None

                tensor_queue.put((idx, total, processed_frame, processed_mask))

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
        batch_masks: list = []
        batch_indices: list = []
        total_frames = 0

        try:
            while True:
                item = tensor_queue.get()
                if item is _SENTINEL:
                    break

                idx, total_frames, processed_frame, processed_mask = item

                if progress_callback is not None:
                    progress_callback(idx, total_frames, "preprocessing")

                batch_frames.append(processed_frame)
                batch_masks.append(processed_mask)
                batch_indices.append(idx)

                if len(batch_frames) >= self.batch_size:
                    latent_list.append(
                        self._run_inference(batch_frames, batch_masks, batch_indices)
                    )
                    if progress_callback is not None:
                        progress_callback(batch_indices[-1], total_frames, "inference")
                    batch_frames = []
                    batch_masks = []
                    batch_indices = []

            # Flush the last partial batch
            if batch_frames:
                latent_list.append(
                    self._run_inference(batch_frames, batch_masks, batch_indices)
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

    def _run_inference(
        self,
        frames: list,
        masks: list,
        indices: list,
    ) -> np.ndarray:
        """Run model inference on a batch of preprocessed frames.

        When KIT-transformed masks are available (not ``None``), they are used
        directly for ROI-weighted pooling.  Otherwise a dummy all-ROI mask is
        constructed to maintain backward compatibility.

        Args:
            frames: List of preprocessed frame arrays, each ``(H, W, 3)``.
            masks: List of transformed mask arrays ``(H, W)`` or ``None`` entries.
            indices: Corresponding frame indices (for logging).

        Returns:
            Latent array, shape ``(B, D)``; or stacked frames ``(B, H, W, C)``
            when ``self.model`` is ``None``.
        """
        batch = np.stack(frames, axis=0)  # (B, H, W, C)

        if self.model is None:
            # No model: return raw frames (shape stays (B, H, W, C))
            return batch.astype(np.float32)

        # Use real transformed masks when available; fall back to dummy all-ROI masks.
        h, w = batch.shape[1], batch.shape[2]
        if any(m is not None for m in masks):
            masks_arr = np.stack(
                [m if m is not None else np.full((h, w), self.roi_id, dtype=np.uint8)
                 for m in masks],
                axis=0,
            )
        else:
            masks_arr = np.full((len(frames), h, w), self.roi_id, dtype=np.uint8)

        # Let inference exceptions propagate — substituting zeros would create a
        # spurious cluster of all-zero latents that downstream UMAP/DBSCAN would
        # treat as a real (and very tight) behavioral group.
        if hasattr(self.model, "extract_tensor_batch"):
            result = self.model.extract_tensor_batch(batch, masks_arr, self.roi_id)
        elif hasattr(self.model, "extract_batch_latent"):
            result = self.model.extract_batch_latent(batch, masks_arr, self.roi_id)
        else:
            raise AttributeError(
                "Model must implement extract_tensor_batch or extract_batch_latent"
            )

        if isinstance(result, list):
            result = np.array(result)
        return np.asarray(result, dtype=np.float32)
