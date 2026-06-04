"""
castle/core/extractor.py
Core extraction logic execution engine.
"""

from collections import Counter
from typing import Protocol, Optional
import json
import os
import threading

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Import from our new Core modules
from castle.core.data import OnFrameError, VideoDataset, Preprocess
from castle.core.config import SUPPORTED_MODELS, ERROR_MESSAGES
from castle.core.environment import get_num_workers
from castle.core.logging_config import setup_logger
from castle.core.models import get_visual_encoder
from castle.core.project import get_project_config, save_project_config
from castle.core.seed import make_torch_generator, seed_worker
from castle.core.types import (
    ExtractionError,
    MaskNotFoundError,
    PreprocessingError,
    ROINotFoundError,
    VideoReadError,
)
from castle.utils.video_io import VideoWriter, VideoReader
from castle.utils.h5_io import H5IO
from castle.utils.latent_metadata import save_latent_with_metadata
from castle.utils.video_align import center_roi, get_roi_closest_point_safe, blank_page

# Setup logger
logger = setup_logger(__name__)


# --- Protocol Definition ---
class ProgressCallback(Protocol):
    """Callback protocol for reporting extraction progress.

    Implementations receive a float in [0, 1] and an optional description
    string, enabling progress bars in any frontend (Gradio, CLI, Desktop).
    """

    def __call__(self, progress: float, desc: str = None) -> None: ...

# --- Helper Logic ---
def _load_prescan_cache(path: str, key: dict):
    """Read a PERF-02 tail-ROI sidecar cache if its key matches.

    Returns the cached ``interpolated_points`` dict (keyed by integer frame
    index) on cache hit, or ``None`` when the file is missing / stale /
    unreadable. Reads tolerate any I/O or JSON error — a corrupt cache must
    not block extraction.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get('key') != key:
        return None
    points = payload.get('points')
    if not isinstance(points, dict):
        return None
    try:
        return {int(k): tuple(v) for k, v in points.items()}
    except (TypeError, ValueError):
        return None


def _save_prescan_cache(path: str, key: dict, interpolated_points: dict) -> None:
    """Write a PERF-02 tail-ROI sidecar cache. Best-effort: failures log only."""
    payload = {
        "key": key,
        "points": {str(k): list(v) for k, v in interpolated_points.items()},
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f)
        logger.info("Pre-scan cache written to %s", path)
    except OSError as exc:
        logger.debug("Could not write pre-scan cache at %s: %s", path, exc)


def _enable_cudnn_benchmark_if_not_strict() -> None:
    """Turn cudnn.benchmark on when the user is *not* in strict-CUDA mode.

    Strict mode (see :func:`castle.core.seed.set_global_seed`) sets
    ``cudnn.deterministic = True``; respecting that flag here means the
    deterministic guarantee wins over throughput, while the default
    fast path still gets the heuristic-driven kernel selection that helps
    DINO conv/patch-embed by ~10–20 %% across batches.
    """
    if torch.cuda.is_available() and not torch.backends.cudnn.deterministic:
        torch.backends.cudnn.benchmark = True


class ExtractionCancelled(Exception):
    """Raised inside the per-batch loop when a run's cancel_event is set, so a
    long single-video extraction aborts within ~one batch (the .npz is written
    only after the loop, so nothing partial is saved)."""


def _build_extractor_loader_kwargs(batch_size: int, num_workers: int, pin_memory: bool = True) -> dict:
    """Common DataLoader kwargs for both latent + rotation extraction paths.

    Adds ``persistent_workers`` + ``prefetch_factor`` (PERF-07) and threads
    the master-seed-derived ``torch.Generator`` for reproducibility
    (BUG-09, set by P0-B).
    """
    kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    # Deeper prefetch + persistent workers keep the GPU fed (raised from 4 → 6).
    # These apply whenever we have worker processes, independent of the seed gen.
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 6
    gen = make_torch_generator()
    if gen is not None:
        kwargs["generator"] = gen
        if num_workers > 0:
            kwargs["worker_init_fn"] = seed_worker
    return kwargs


def _get_observer(select_model):
    """Get visual model observer with validation."""
    if select_model not in SUPPORTED_MODELS:
        raise ValueError(ERROR_MESSAGES['unsupported_model'].format(
            model=select_model,
            supported=SUPPORTED_MODELS
        ))
    
    try:
        return get_visual_encoder(select_model)
    except Exception as e:
        raise ImportError(f"Failed to load model {select_model}: {e}")


# --- Interpolation Utility ---
def interpolate_missing_points(valid_points, total_frames):
    """
    對缺失的幀執行線性內插或外插（向量化版本）
    
    Args:
        valid_points: dict {frame_idx: (x, y)} 有效的追蹤點
        total_frames: 總幀數
    
    Returns:
        dict {frame_idx: (x, y)} 包含所有幀的點（內插/外插後）
    """
    if not valid_points:
        raise ValueError("No valid tracking points found for rotate_roi_tail_id")
    
    # Sort valid indices and extract x, y arrays
    sorted_indices = np.array(sorted(valid_points.keys()))
    valid_x = np.array([valid_points[i][0] for i in sorted_indices])
    valid_y = np.array([valid_points[i][1] for i in sorted_indices])
    
    # Use numpy interp for vectorized linear interpolation (handles extrapolation at edges)
    all_indices = np.arange(total_frames)
    interp_x = np.interp(all_indices, sorted_indices, valid_x)
    interp_y = np.interp(all_indices, sorted_indices, valid_y)
    
    # Log extrapolation at edges
    if sorted_indices[0] > 0:
        logger.warning(f"Extrapolating at beginning of video for frames 0-{sorted_indices[0]-1} using frame {sorted_indices[0]}")
    if sorted_indices[-1] < total_frames - 1:
        logger.warning(f"Extrapolating at end of video for frames {sorted_indices[-1]+1}-{total_frames-1} using frame {sorted_indices[-1]}")
    
    result = {idx: (float(interp_x[idx]), float(interp_y[idx])) for idx in range(total_frames)}
    
    return result


def _run_extraction_loop(
    observer,
    loader,
    *,
    roi_id: int,
    pooling_method: str,
    pooling_scales: Optional[list],
    feature_layers: Optional[list],
    on_frame_error: OnFrameError,
    max_batch_failure_rate: float,
    video_name: str,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_event=None,
):
    """Run the per-batch extraction loop over ``loader`` and return the latent array.

    Shared by single-GPU :func:`extract_roi_latent_from_video` and the per-GPU
    workers of :func:`extract_roi_latent_from_video_2gpu`. Timeline integrity
    (P0-2): every batch contributes exactly ``frames.shape[0]`` rows; a tolerated
    failure becomes a NaN placeholder (filled once the feature dim is known) and
    its frame range is recorded, so row index == frame index for this loader's
    range (the caller offsets ranges to global frame coords when merging).

    Returns:
        ``(latent_array, failed_frame_ranges, n_batches_failed)``.
    """
    latent_slots: list = []          # real (B, C) arrays, or None placeholders
    pending_fail: list = []          # (slot_index, n_rows) awaiting feature dim
    failed_frame_ranges: list = []   # [[start, end), ...] loader-local frame indices
    total_batches = len(loader)
    n_batches_failed = 0
    abs_failure_threshold = max(1, int(max_batch_failure_rate * total_batches))
    first_batch_error: Optional[str] = None
    expected_dim: Optional[int] = None
    rows_seen = 0

    def _record_failure(slot_pos: int, n_rows: int, frame_start: int) -> None:
        pending_fail.append((slot_pos, n_rows))
        failed_frame_ranges.append([int(frame_start), int(frame_start + n_rows)])

    for i, (frames, masks) in enumerate(loader):
        # Batch-granular cancel: abort within ~one batch (a single big video can
        # run for tens of minutes). The .npz is saved only after this loop, so a
        # raise here leaves no partial output.
        if cancel_event is not None and cancel_event.is_set():
            raise ExtractionCancelled(f"extraction cancelled during {video_name}")
        n_rows = int(frames.shape[0])
        frame_start = rows_seen
        rows_seen += n_rows
        try:
            if hasattr(observer, 'extract_tensor_batch'):
                 latent_batch = observer.extract_tensor_batch(
                     frames, masks, roi_id,
                     pooling=pooling_method,
                     scales=pooling_scales,
                     layers=feature_layers,
                 )
            else:
                 latent_batch = observer.extract_batch_latent(frames, masks, roi_id)

            latent_slots.append(latent_batch)
            if expected_dim is None:
                expected_dim = latent_batch.shape[1]

        except (ROINotFoundError, PreprocessingError) as e:
            # Strict path re-raises; tolerant path keeps the timeline aligned.
            n_batches_failed += 1
            if on_frame_error == "raise" or n_batches_failed > abs_failure_threshold:
                raise
            if first_batch_error is None:
                first_batch_error = repr(e)
            logger.warning(
                "Batch %d/%d for %s failed (%s); inserting %d NaN placeholder "
                "frame(s) to preserve the timeline.",
                i + 1, total_batches, video_name, e, n_rows,
            )
            latent_slots.append(None)
            _record_failure(len(latent_slots) - 1, n_rows, frame_start)
        except Exception as e:
            n_batches_failed += 1
            if first_batch_error is None:
                first_batch_error = repr(e)
            logger.error(
                "Batch %d/%d failed for %s: %s",
                i + 1, total_batches, video_name, e,
            )
            if n_batches_failed > abs_failure_threshold:
                raise ExtractionError(
                    f"Aborting {video_name}: {n_batches_failed}/{i + 1} batches "
                    f"failed (threshold {abs_failure_threshold} of "
                    f"{total_batches}, max_rate={max_batch_failure_rate:.0%}). "
                    f"Cause: {first_batch_error}."
                ) from e
            logger.warning(
                "Batch %d/%d for %s tolerated after error; inserting %d NaN "
                "placeholder frame(s) to preserve the timeline.",
                i + 1, total_batches, video_name, n_rows,
            )
            latent_slots.append(None)
            _record_failure(len(latent_slots) - 1, n_rows, frame_start)

        if progress_callback:
            progress_callback((i + 1) / total_batches, desc=f"Extracting {video_name}")

    if expected_dim is None:
        raise ExtractionError(
            f"All {total_batches} batches failed for {video_name}. "
            f"No latent file written."
        )

    # BUG-05: validate feature-dimension consistency across the real batches so
    # a model swap mid-extraction fails loudly instead of a cryptic ValueError.
    mismatched = [
        (idx, tuple(b.shape)) for idx, b in enumerate(latent_slots)
        if b is not None and b.shape[1] != expected_dim
    ]
    if mismatched:
        sample = mismatched[:5]
        raise ExtractionError(
            f"Inconsistent feature dimensions across batches for {video_name}. "
            f"Expected dim {expected_dim}; mismatched batches: {sample}"
            + ("..." if len(mismatched) > len(sample) else "")
            + ". This usually indicates a model swap mid-extraction."
        )

    # Fill tolerated failures with NaN placeholders of the correct width.
    for slot_pos, n_rows in pending_fail:
        latent_slots[slot_pos] = np.full((n_rows, expected_dim), np.nan, dtype=np.float32)

    latent_array = np.concatenate(latent_slots, axis=0)
    return latent_array, failed_frame_ranges, n_batches_failed


def _latent_filename(video_name, roi_id, model_name, preprocess_config,
                     pooling_method, pooling_scales, feature_layers, session_id) -> str:
    """Canonical latent ``.npz`` filename.

    Shared by the single-GPU and multi-GPU paths so they produce identical names
    (``skip_existing`` checks and ``config['latent']`` keys must match exactly).
    """
    base_name = os.path.splitext(video_name)[0]
    tags = []
    if preprocess_config.center_roi_switch:
        tags.append("ctr")
    if preprocess_config.remove_background_switch:
        tags.append("rmbg")
    if pooling_method == 'multiscale' and pooling_scales:
        tags.append("spp" + "x".join(str(s) for s in sorted(pooling_scales)))
    if feature_layers:
        tags.append("L" + "x".join(str(layer) for layer in sorted(feature_layers)))
    suffix = "_".join([model_name] + tags)
    pre_tag = f"_pre-{session_id}" if session_id else ""
    return f'{base_name}_ROI_{roi_id}_{suffix}{pre_tag}.npz'


def _resolve_latent_dtype(latent_dtype):
    """Map a UI precision string to a numpy storage dtype (default float32)."""
    return np.float16 if str(latent_dtype).lower() in ("float16", "fp16", "half") else np.float32


# --- Core Function 1: Extract Latent ---
def extract_roi_latent_from_video(
    storage_path: str,
    project_name: str,
    video_name: str,
    roi_id: int,
    model_name: str,
    batch_size: int,
    preprocess_config: Preprocess,
    skip_existing: bool,
    progress_callback: Optional[ProgressCallback] = None,
    pooling_method: str = 'weighted_average',
    pooling_scales: Optional[list] = None,
    feature_layers: Optional[list] = None,
    *,
    source_video_path: Optional[str] = None,
    mask_path_override: Optional[str] = None,
    session_id: Optional[str] = None,
    on_frame_error: OnFrameError = "skip",
    max_batch_failure_rate: float = 0.05,
    device: Optional[str] = None,
    num_workers: Optional[int] = None,
    latent_dtype: str = 'float32',
    cancel_event=None,
) -> str:
    """Extracts latent features from a specific video ROI.

    Args:
        storage_path: Project storage root.
        project_name: Project name.
        video_name: Filename within ``sources/``.
        roi_id: ROI id to extract.
        model_name: Visual encoder name.
        batch_size: DataLoader batch size.
        preprocess_config: Preprocess pipeline configuration.
        skip_existing: Skip if the output ``.npz`` already exists.
        progress_callback: Optional progress reporter.
        pooling_method, pooling_scales, feature_layers: see model docs.
        on_frame_error: ``"skip"`` (default) drops frames whose preprocessing
            raises ``ROINotFoundError`` / ``PreprocessingError``; ``"raise"``
            aborts the whole extraction at the first bad frame.
        max_batch_failure_rate: Abort when more than this fraction of
            DataLoader batches fail. A floor of 1 is always honoured, so a
            single failure in a small (≤ 20-batch) run still aborts.

    Returns:
        Absolute path to the saved latent ``.npz`` file.

    Raises:
        MaskNotFoundError: Tracker mask file is missing — run ``castle track`` first.
        VideoReadError: Source video cannot be opened or read.
        ExtractionError: Model load failure, all batches failed, or the
            batch failure rate exceeded ``max_batch_failure_rate``.
    """
    batch_size = int(batch_size)
    roi_id = int(roi_id)

    # PERF-07: honour strict_cuda; otherwise turn on cudnn benchmark for speed.
    _enable_cudnn_benchmark_if_not_strict()

    # 1. Setup paths
    project_path, config = get_project_config(storage_path, project_name)

    # New Structure: latent/{model_name}/
    latent_dir_path = os.path.join(project_path, 'latent', model_name)
    os.makedirs(latent_dir_path, exist_ok=True)
    
    latent_filename = _latent_filename(
        video_name, roi_id, model_name, preprocess_config,
        pooling_method, pooling_scales, feature_layers, session_id,
    )
    latent_path = os.path.join(latent_dir_path, latent_filename)

    if skip_existing and os.path.exists(latent_path):
        logger.info(f"Skipping existing latent: {latent_path}")
        return latent_path

    # 2. Load Resources
    source_path = source_video_path or os.path.join(storage_path, project_name, 'sources', video_name)
    track_dir_path = os.path.join(project_path, 'track', video_name)
    mask_list_path = mask_path_override or os.path.join(track_dir_path, 'mask_list.h5')

    if not os.path.exists(mask_list_path):
        raise MaskNotFoundError(
            f"Mask file not found for video {video_name!r}. "
            f"Expected at: {mask_list_path}. "
            f"Hint: run `castle track {project_name}` first."
        )

    # 3. Setup Models — pin to an explicit device for video-level multi-GPU
    # (extract_latent runs one video per GPU); device=None / 'cuda' / 'cuda:0'
    # reuse the primary-device singleton via _get_device_encoder.
    try:
        observer = _get_device_encoder(model_name, device) if device else _get_observer(model_name)
    except (ImportError, ValueError, RuntimeError) as e:
        raise ExtractionError(
            f"Failed to load model {model_name!r} for {video_name}: {e}"
        ) from e

    # 4. Processing — when several videos extract concurrently (one per GPU) the
    # caller passes a reduced num_workers (total // n_gpu) so the DataLoaders
    # don't oversubscribe the CPU and starve the GPUs.
    NUM_WORKERS = num_workers if num_workers is not None else get_num_workers('extraction')

    # Get video length
    try:
        with VideoReader(source_path) as vr:
            video_len = len(vr)
    except Exception as e:
        raise VideoReadError(
            f"Failed to open video {source_path}: {e}. "
            f"Hint: try re-encoding with ffmpeg if metadata is corrupt."
        ) from e

    # Pre-scan: if rotate_roi_tail is enabled, scan all frames to collect
    # valid tail ROI points, then interpolate missing ones
    interpolated_points = None
    if preprocess_config.rotate_roi_tail_switch and preprocess_config.center_roi_switch:
        scan_cache_path = os.path.join(track_dir_path, 'tail_roi_scan.json')
        scan_cache_key = {
            "center_roi_id": int(preprocess_config.center_roi_id),
            "rotate_roi_tail_id": int(preprocess_config.rotate_roi_tail_id),
            "video_len": int(video_len),
        }
        cached = _load_prescan_cache(scan_cache_path, scan_cache_key)
        if cached is not None:
            interpolated_points = cached
            logger.info(
                "Pre-scan: cache hit at %s (%d frames); skipping mask sweep.",
                scan_cache_path, video_len,
            )
        else:
            logger.info(f"Pre-scanning {video_name} for tail ROI interpolation...")
            valid_points = {}
            failure_reasons: Counter = Counter()
            tracker_scan = H5IO(mask_list_path, read_only=True)
            BATCH = 256
            try:
                for start in range(0, video_len, BATCH):
                    indices = range(start, min(start + BATCH, video_len))
                    masks_batch = tracker_scan.read_masks_batch(indices)
                    for idx in indices:
                        mask = masks_batch.get(idx)
                        if mask is None:
                            failure_reasons["mask_read"] += 1
                            continue
                        try:
                            m = center_roi(mask, mask, preprocess_config.center_roi_id)
                            point = get_roi_closest_point_safe(m, preprocess_config.rotate_roi_tail_id)
                        except Exception as e:
                            failure_reasons["preprocess"] += 1
                            logger.debug("Pre-scan preprocess failed at %d: %s", idx, e)
                            continue
                        if point is None:
                            failure_reasons["roi_not_found"] += 1
                            continue
                        valid_points[idx] = point
            finally:
                tracker_scan.close()

            logger.info(
                "Pre-scan: %d/%d valid; failures: %s",
                len(valid_points), video_len, dict(failure_reasons),
            )

            if valid_points:
                interpolated_points = interpolate_missing_points(valid_points, video_len)
                logger.info(f"Interpolation complete: all {video_len} frames now have rotation points")
                _save_prescan_cache(scan_cache_path, scan_cache_key, interpolated_points)

    dataset = VideoDataset(
        source_path, video_len, mask_list_path, preprocess_config, roi_id,
        interpolated_points=interpolated_points,
        on_frame_error=on_frame_error,
    )

    # pin_memory on regardless of device — pinned host buffers speed the H2D copy
    # for cuda:0 and cuda:1 alike (was disabled whenever a device was passed).
    loader = DataLoader(dataset, **_build_extractor_loader_kwargs(batch_size, NUM_WORKERS, pin_memory=True))

    latent_array, failed_frame_ranges, n_batches_failed = _run_extraction_loop(
        observer, loader,
        roi_id=roi_id,
        pooling_method=pooling_method,
        pooling_scales=pooling_scales,
        feature_layers=feature_layers,
        on_frame_error=on_frame_error,
        max_batch_failure_rate=max_batch_failure_rate,
        video_name=video_name,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )
    total_batches = len(loader)

    # BUG-14: embed video / ROI / model identity so loaders can stop relying
    # on filename parsing.
    save_latent_with_metadata(
        latent_path,
        latent_array,
        video_name=video_name,
        roi_id=int(roi_id),
        model_name=model_name,
        tags={
            "pooling_method": pooling_method,
            "pooling_scales": list(pooling_scales) if pooling_scales else None,
            "feature_layers": list(feature_layers) if feature_layers else None,
            "rotation": False,
            "failed_frame_ranges": failed_frame_ranges or None,
        },
        dtype=_resolve_latent_dtype(latent_dtype),
    )

    if n_batches_failed:
        n_nan_frames = sum(end - start for start, end in failed_frame_ranges)
        logger.warning(
            "Extraction for %s completed with %d/%d failed batches (below %.0f%% "
            "threshold); %d frame(s) stored as NaN placeholders to keep the "
            "timeline aligned (ranges recorded in metadata; downstream clustering "
            "filters NaN rows).",
            video_name, n_batches_failed, total_batches,
            max_batch_failure_rate * 100, n_nan_frames,
        )

    # Update Config — use atomic read-modify-write context manager so two
    # concurrent extractions writing different videos don't lose updates (3-F).
    from castle.core.project import update_config
    latent_key = f"{session_id}/{latent_filename}" if session_id else latent_filename
    with update_config(storage_path, project_name) as config:
        config.setdefault('latent', {})[latent_key] = video_name

    return latent_path

# --- Core Function 2: Extract Crop Video ---
def extract_roi_crop_video(
    storage_path: str,
    project_name: str,
    video_name: str,
    roi_id: int,
    preprocess_config: Preprocess,
    skip_existing: bool,
    progress_callback: Optional[ProgressCallback] = None,
    *,
    on_frame_error: OnFrameError = "skip",
) -> str:
    """Render a per-frame cropped/preprocessed video for the given ROI.

    Args:
        on_frame_error: ``"skip"`` (default) writes a blank frame for any
            preprocessing failure (preserving the output timeline). ``"raise"``
            aborts on the first bad frame.

    Raises:
        MaskNotFoundError: Tracker mask missing.
    """
    roi_id = int(roi_id)
    project_path, _ = get_project_config(storage_path, project_name)
    latent_dir_path = os.path.join(project_path, 'latent')
    os.makedirs(latent_dir_path, exist_ok=True)

    base_name = os.path.splitext(video_name)[0]
    out_video_path = os.path.join(latent_dir_path, f'{base_name}_ROI_{roi_id}_crop.mp4')

    if skip_existing and os.path.exists(out_video_path):
        return out_video_path

    source_path = os.path.join(storage_path, project_name, 'sources', video_name)
    track_dir_path = os.path.join(project_path, 'track', video_name)
    mask_list_path = os.path.join(track_dir_path, 'mask_list.h5')

    if not os.path.exists(mask_list_path):
        raise MaskNotFoundError(
            f"Mask file not found for video {video_name!r}. "
            f"Expected at: {mask_list_path}. "
            f"Hint: run `castle track {project_name}` first."
        )

    writer = None
    tracker = None

    try:
        tracker = H5IO(mask_list_path, read_only=True)
        with VideoReader(source_path) as source_video:
            fps = source_video.fps
            writer = VideoWriter(out_video_path, fps, crf=15)

            total_frames = len(source_video)

            for i, frame in enumerate(source_video):
                if progress_callback and i % 10 == 0:
                    progress_callback((i + 1) / total_frames, desc=f"Cropping {video_name}")

                try:
                    mask = tracker.read_mask(i)
                    processed_frame, _ = preprocess_config.transform(frame, mask)
                    writer.write_frame(processed_frame)
                except (ROINotFoundError, PreprocessingError) as e:
                    if on_frame_error == "raise":
                        raise
                    logger.warning(
                        "Frame %d in %s: %s; writing blank frame to preserve timeline.",
                        i, video_name, e,
                    )
                    h, w = frame.shape[:2]
                    writer.write_frame(blank_page(h, w))
                except (IOError, OSError, KeyError) as e:
                    if on_frame_error == "raise":
                        raise ExtractionError(
                            f"Frame {i} in {video_name}: mask read failed: {e}"
                        ) from e
                    logger.warning(
                        "Frame %d in %s: mask read failed: %s; writing blank.",
                        i, video_name, e,
                    )
                    h, w = frame.shape[:2]
                    writer.write_frame(blank_page(h, w))
    finally:
        if tracker is not None:
            try:
                tracker.close()
            except Exception as e:
                logger.debug("tracker.close() failed during crop cleanup: %s", e)
        if writer:
            try:
                writer.close()
            except Exception as e:
                logger.debug("writer.close() failed during crop cleanup: %s", e)

    return out_video_path


# --- Helper Classes ---
class RotationDataset(VideoDataset):
    """
    Dataset that returns a batch of rotated views for a single frame.
    Returns: (frames, masks) where frames is (Num_Rotations, H, W, C)
    """
    def __init__(self, *args, num_rotations=7, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_rotations = num_rotations
        self.angles = np.linspace(0, 360, num_rotations, endpoint=False)

    def __getitem__(self, idx: int):
        if self.reader is None:
            self.reader = VideoReader(self.video_path)

        if self.tracker is None:
            self.tracker = H5IO(self.mask_path, read_only=True)

        frame = self.reader[idx]
        mask = self.tracker.read_mask(idx)

        frames_list = []
        masks_list = []

        for deg in self.angles:
            try:
                pf, pm = self.preprocess.transform(frame, mask, int(deg))
            except (ROINotFoundError, PreprocessingError) as e:
                if self.on_frame_error == "raise":
                    raise
                logger.warning(
                    "Dropping rotated view %s for frame %d in %s: %s",
                    deg, idx, self.video_path, e,
                )
                h = self.preprocess.center_roi_crop_height
                w = self.preprocess.center_roi_crop_width
                pf = blank_page(h, w)
                # 2D all-background mask (blank_page is a 3D frame, wrong for masks).
                pm = np.zeros((h, w), dtype=np.uint8)
            frames_list.append(pf)
            masks_list.append(pm)

        # Return Stacked
        return np.stack(frames_list), np.stack(masks_list)


# --- Core Function 3: Extract Rotation Latent ---
def extract_roi_rotation_latent_from_video(
    storage_path: str,
    project_name: str,
    video_name: str,
    roi_id: int,
    model_name: str,
    batch_size: int,
    preprocess_config: Preprocess,
    skip_existing: bool,
    progress_callback: Optional[ProgressCallback] = None,
    *,
    source_video_path: Optional[str] = None,
    mask_path_override: Optional[str] = None,
    session_id: Optional[str] = None,
    on_frame_error: OnFrameError = "skip",
    max_batch_failure_rate: float = 0.05,
) -> str:
    """Extracts rotation-averaged latent features for the given ROI.

    Generates 7 rotated views (0–360 deg) per frame, embeds each, and averages
    the resulting latents.

    Raises:
        MaskNotFoundError: Tracker mask missing.
        VideoReadError: Source video unopenable.
        ExtractionError: Model load failure or all batches failed.
    """
    batch_size = int(batch_size)
    roi_id = int(roi_id)

    _enable_cudnn_benchmark_if_not_strict()

    # 1. Setup paths
    project_path, config = get_project_config(storage_path, project_name)
    latent_dir_path = os.path.join(project_path, 'latent', model_name)
    os.makedirs(latent_dir_path, exist_ok=True)

    base_name = os.path.splitext(video_name)[0]

    pre_tag = f"_pre-{session_id}" if session_id else ""
    latent_filename = f'{base_name}_ROI_{roi_id}_rotation_latent{pre_tag}.npz'

    latent_path = os.path.join(latent_dir_path, latent_filename)

    if skip_existing and os.path.exists(latent_path):
        logger.info(f"Skipping existing latent: {latent_path}")
        return latent_path

    # 2. Load Resources
    source_path = source_video_path or os.path.join(storage_path, project_name, 'sources', video_name)
    track_dir_path = os.path.join(project_path, 'track', video_name)
    mask_list_path = mask_path_override or os.path.join(track_dir_path, 'mask_list.h5')

    if not os.path.exists(mask_list_path):
        raise MaskNotFoundError(
            f"Mask file not found for video {video_name!r}. "
            f"Expected at: {mask_list_path}. "
            f"Hint: run `castle track {project_name}` first."
        )

    # 3. Setup Models
    try:
        observer = _get_observer(model_name)
    except (ImportError, ValueError, RuntimeError) as e:
        raise ExtractionError(
            f"Failed to load model {model_name!r} for {video_name}: {e}"
        ) from e

    embed_dim = observer.n_feature

    # 4. Processing
    NUM_WORKERS = get_num_workers('extraction')

    try:
        with VideoReader(source_path) as vr:
            video_len = len(vr)
    except Exception as e:
        raise VideoReadError(
            f"Failed to open video {source_path}: {e}"
        ) from e

    dataset = RotationDataset(
        video_path=source_path,
        video_len=video_len,
        mask_path=mask_list_path,
        preprocess=preprocess_config,
        select_roi=roi_id,
        num_rotations=7,
        on_frame_error=on_frame_error,
    )

    loader = DataLoader(dataset, **_build_extractor_loader_kwargs(batch_size, NUM_WORKERS))

    # Timeline-preserving slots (see extract_roi_latent_from_video): a tolerated
    # batch failure becomes a NaN placeholder of the right row count rather than
    # being dropped, so row index == frame index is never violated.
    latent_slots: list = []
    failed_frame_ranges: list = []
    total_batches = len(loader)
    n_batches_failed = 0
    abs_failure_threshold = max(1, int(max_batch_failure_rate * total_batches))
    first_batch_error: Optional[str] = None
    rows_seen = 0

    try:
        for i, (frames, masks) in enumerate(loader):
            if progress_callback:
                progress_callback((i + 1) / total_batches, desc=f"Extracting {video_name} (Rot)")

            n_rows = int(frames.shape[0])
            frame_start = rows_seen
            rows_seen += n_rows
            try:
                B, R, H, W, C = frames.shape
                frames_flat = frames.reshape(B * R, H, W, C)
                masks_flat = masks.reshape(B * R, H, W)

                if hasattr(observer, 'extract_tensor_batch'):
                    latent_batch = observer.extract_tensor_batch(frames_flat, masks_flat, roi_id)
                else:
                    latent_batch = observer.extract_batch_latent(frames_flat, masks_flat, roi_id)

                if isinstance(latent_batch, list):
                    latent_batch = np.array(latent_batch)

                latent_reshaped = latent_batch.reshape(B, R, embed_dim)
                latent_averaged = latent_reshaped.mean(axis=1)

                latent_slots.append(latent_averaged)
            except (ROINotFoundError, PreprocessingError) as e:
                n_batches_failed += 1
                if on_frame_error == "raise" or n_batches_failed > abs_failure_threshold:
                    raise
                logger.warning(
                    "Rotation batch %d/%d for %s failed (%s); inserting %d NaN "
                    "placeholder frame(s) to preserve the timeline.",
                    i + 1, total_batches, video_name, e, n_rows,
                )
                latent_slots.append(np.full((n_rows, embed_dim), np.nan, dtype=np.float32))
                failed_frame_ranges.append([int(frame_start), int(frame_start + n_rows)])
            except Exception as e:
                n_batches_failed += 1
                if first_batch_error is None:
                    first_batch_error = repr(e)
                logger.error(
                    "Rotation batch %d/%d failed for %s: %s",
                    i + 1, total_batches, video_name, e,
                )
                if n_batches_failed > abs_failure_threshold:
                    raise ExtractionError(
                        f"Aborting {video_name}: {n_batches_failed}/{i + 1} "
                        f"rotation batches failed (threshold "
                        f"{abs_failure_threshold} of {total_batches}, "
                        f"max_rate={max_batch_failure_rate:.0%}). "
                        f"Cause: {first_batch_error}."
                    ) from e
                logger.warning(
                    "Rotation batch %d/%d for %s tolerated after error; inserting "
                    "%d NaN placeholder frame(s) to preserve the timeline.",
                    i + 1, total_batches, video_name, n_rows,
                )
                latent_slots.append(np.full((n_rows, embed_dim), np.nan, dtype=np.float32))
                failed_frame_ranges.append([int(frame_start), int(frame_start + n_rows)])

        n_failed_frames_total = sum(end - start for start, end in failed_frame_ranges)
        if not latent_slots or n_failed_frames_total >= rows_seen:
            raise ExtractionError(
                f"All {total_batches} rotation batches failed for {video_name}."
            )

        # Concatenate final results
        latent_array = np.concatenate(latent_slots, axis=0)
        # BUG-14: include metadata so loaders can stop relying on filename
        # parsing (rotation files don't carry model_name in the filename).
        save_latent_with_metadata(
            latent_path,
            latent_array,
            video_name=video_name,
            roi_id=int(roi_id),
            model_name=model_name,
            tags={"rotation": True, "failed_frame_ranges": failed_frame_ranges or None},
        )

        # Update Config — atomic RMW per 3-F.
        from castle.core.project import update_config
        with update_config(storage_path, project_name) as config:
            config.setdefault('latent', {})[latent_filename] = video_name

        if n_batches_failed:
            n_nan_frames = sum(end - start for start, end in failed_frame_ranges)
            logger.warning(
                "Rotation extraction for %s completed with %d/%d failed batches; "
                "%d frame(s) stored as NaN placeholders to keep the timeline "
                "aligned (ranges recorded in metadata).",
                video_name, n_batches_failed, total_batches, n_nan_frames,
            )

        return latent_path

    except Exception:
        # Clean up partial file so a retry does not get tricked by skip_existing.
        if os.path.exists(latent_path):
            try:
                os.remove(latent_path)
            except OSError as cleanup_err:
                logger.debug(
                    "Could not remove partial latent %s: %s", latent_path, cleanup_err,
                )
        raise


# ---------------------------------------------------------------------------
# Multi-GPU extraction (opt-in): split one video's frames across GPUs
# ---------------------------------------------------------------------------

_device_encoder_cache: dict = {}
_device_encoder_lock = threading.Lock()


def _get_device_encoder(model_name: str, device: str):
    """Get an encoder pinned to a specific CUDA device for multi-GPU extraction.

    The primary device reuses the shared singleton (``get_visual_encoder`` via
    ``_get_observer``, no reload); other devices get a dedicated encoder built +
    cached here so repeated videos don't reload weights. Two extraction threads
    use *different* encoder objects (no shared mutable state during inference).
    """
    if device in ('cuda', 'cuda:0'):
        return _get_observer(model_name)
    key = (model_name, device)
    with _device_encoder_lock:
        enc = _device_encoder_cache.get(key)
        if enc is None:
            from castle.core.models import DINOv2Encoder, DINOv3Encoder
            if 'dinov3' in model_name:
                enc = DINOv3Encoder(model_name, device=device)
            else:
                enc = DINOv2Encoder(model_name, device=device)
            enc.load_model()
            _device_encoder_cache[key] = enc
            logger.info("Multi-GPU: built %s encoder on %s", model_name, device)
        return enc


def clear_device_encoder_cache() -> None:
    """Evict the multi-GPU per-device encoders and free their GPU memory.

    The primary-device encoder lives in models._model_cache (evicted by
    _evict_model_cache); the cuda:1+ encoders built here for multi-GPU
    extraction were previously never freed, leaking VRAM on the secondary
    GPU for the process lifetime (e.g. across model switches in Gradio).
    """
    import torch
    with _device_encoder_lock:
        for key, enc in list(_device_encoder_cache.items()):
            dev = key[1]
            try:
                if getattr(enc, 'model', None) is not None:
                    del enc.model
                    enc.model = None
            except Exception:  # noqa: BLE001
                pass
            _device_encoder_cache.pop(key, None)
            try:
                if torch.cuda.is_available():
                    with torch.cuda.device(dev):
                        torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass


def extract_roi_latent_from_video_2gpu(
    storage_path: str,
    project_name: str,
    video_name: str,
    roi_id: int,
    model_name: str,
    batch_size: int,
    preprocess_config: Preprocess,
    skip_existing: bool,
    progress_callback: Optional[ProgressCallback] = None,
    pooling_method: str = 'weighted_average',
    pooling_scales: Optional[list] = None,
    feature_layers: Optional[list] = None,
    *,
    source_video_path: Optional[str] = None,
    mask_path_override: Optional[str] = None,
    session_id: Optional[str] = None,
    on_frame_error: OnFrameError = "skip",
    max_batch_failure_rate: float = 0.05,
    device_ids=(0, 1),
    min_frames_for_split: int = 2000,
    latent_dtype: str = 'float32',
    cancel_event=None,
) -> str:
    """Extract one video's ROI latents by splitting frames across GPUs.

    Splits ``[0, N)`` into ``len(device_ids)`` contiguous ranges, runs each
    range's *full* extraction loop on its own GPU + encoder concurrently
    (threads — CUDA releases the GIL), then concatenates the per-range latents in
    frame order. ``row index == frame index`` is preserved (contiguous slices
    concatenated in order); the result is numerically equivalent to single-GPU up
    to per-device float16 kernel rounding.

    Falls back to :func:`extract_roi_latent_from_video` when the video is short
    (``< min_frames_for_split``), the model isn't DINOv3, or ``rotate_roi_tail``
    is enabled (the tail pre-scan path isn't split).

    Same return contract as :func:`extract_roi_latent_from_video`.
    """
    batch_size = int(batch_size)
    roi_id = int(roi_id)
    _enable_cudnn_benchmark_if_not_strict()

    project_path, _config = get_project_config(storage_path, project_name)
    latent_dir_path = os.path.join(project_path, 'latent', model_name)
    os.makedirs(latent_dir_path, exist_ok=True)

    latent_filename = _latent_filename(
        video_name, roi_id, model_name, preprocess_config,
        pooling_method, pooling_scales, feature_layers, session_id,
    )
    latent_path = os.path.join(latent_dir_path, latent_filename)
    if skip_existing and os.path.exists(latent_path):
        logger.info(f"Skipping existing latent: {latent_path}")
        return latent_path

    source_path = source_video_path or os.path.join(storage_path, project_name, 'sources', video_name)
    track_dir_path = os.path.join(project_path, 'track', video_name)
    mask_list_path = mask_path_override or os.path.join(track_dir_path, 'mask_list.h5')
    if not os.path.exists(mask_list_path):
        raise MaskNotFoundError(
            f"Mask file not found for video {video_name!r}. Expected at: "
            f"{mask_list_path}. Hint: run `castle track {project_name}` first."
        )

    try:
        with VideoReader(source_path) as vr:
            video_len = len(vr)
    except Exception as e:
        raise VideoReadError(
            f"Failed to open video {source_path}: {e}."
        ) from e

    # Fall back to single-GPU when splitting won't help or isn't supported here.
    if (len(device_ids) < 2
            or video_len < min_frames_for_split
            or 'dinov3' not in model_name
            or preprocess_config.rotate_roi_tail_switch):
        logger.info(
            "Multi-GPU: falling back to single-GPU for %s "
            "(len=%d, model=%s, rotate_tail=%s).",
            video_name, video_len, model_name, preprocess_config.rotate_roi_tail_switch,
        )
        return extract_roi_latent_from_video(
            storage_path, project_name, video_name, roi_id, model_name, batch_size,
            preprocess_config, skip_existing, progress_callback,
            pooling_method, pooling_scales, feature_layers,
            source_video_path=source_video_path, mask_path_override=mask_path_override,
            session_id=session_id, on_frame_error=on_frame_error,
            max_batch_failure_rate=max_batch_failure_rate,
        )

    # Contiguous frame ranges, one per device.
    n_dev = len(device_ids)
    bounds = [(k * video_len) // n_dev for k in range(n_dev + 1)]
    ranges = [(bounds[k], bounds[k + 1]) for k in range(n_dev)]

    per_thread_workers = max(0, get_num_workers('extraction') // n_dev)
    results: dict = {}
    errors: dict = {}

    def _worker(slot: int, dev_id: int, fr):
        start, end = fr
        try:
            enc = _get_device_encoder(model_name, f'cuda:{dev_id}')
            dataset = VideoDataset(
                source_path, video_len, mask_list_path, preprocess_config, roi_id,
                interpolated_points=None, on_frame_error=on_frame_error,
            )
            sub = Subset(dataset, list(range(start, end)))
            loader = DataLoader(sub, **_build_extractor_loader_kwargs(batch_size, per_thread_workers, pin_memory=True))
            arr, fails, n_failed = _run_extraction_loop(
                enc, loader,
                roi_id=roi_id, pooling_method=pooling_method, pooling_scales=pooling_scales,
                feature_layers=feature_layers, on_frame_error=on_frame_error,
                max_batch_failure_rate=max_batch_failure_rate,
                video_name=f"{video_name}[{start}:{end}]", progress_callback=None,
                cancel_event=cancel_event,
            )
            results[slot] = (arr, fails, n_failed)
        except Exception as exc:  # surfaced to the caller; no partial write
            errors[slot] = exc

    if progress_callback:
        progress_callback(0.02, desc=f"Extracting {video_name} on {n_dev} GPUs")

    threads = [
        threading.Thread(target=_worker, args=(s, device_ids[s], ranges[s]), daemon=True)
        for s in range(n_dev)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        slots = sorted(errors.items())
        if len(slots) > 1:
            detail = '; '.join(f'GPU{device_ids[s]}: {e}' for s, e in slots)
            raise RuntimeError(
                f'Multi-GPU extraction failed on {len(slots)} device(s): {detail}'
            )
        raise slots[0][1]

    # Merge per-range latents in frame order; offset failed ranges to global coords.
    latent_parts = []
    failed_frame_ranges: list = []
    n_batches_failed = 0
    for s in range(n_dev):
        arr, fails, n_failed = results[s]
        offset = ranges[s][0]
        latent_parts.append(arr)
        failed_frame_ranges.extend([[fs + offset, fe + offset] for fs, fe in fails])
        n_batches_failed += n_failed
    latent_array = np.concatenate(latent_parts, axis=0)

    save_latent_with_metadata(
        latent_path,
        latent_array,
        video_name=video_name,
        roi_id=int(roi_id),
        model_name=model_name,
        tags={
            "pooling_method": pooling_method,
            "pooling_scales": list(pooling_scales) if pooling_scales else None,
            "feature_layers": list(feature_layers) if feature_layers else None,
            "rotation": False,
            "failed_frame_ranges": failed_frame_ranges or None,
            "multi_gpu_device_ids": list(device_ids),
        },
        dtype=_resolve_latent_dtype(latent_dtype),
    )

    if n_batches_failed:
        n_nan_frames = sum(end - start for start, end in failed_frame_ranges)
        logger.warning(
            "Multi-GPU extraction for %s: %d batches failed; %d frame(s) stored as "
            "NaN placeholders to keep the timeline aligned (ranges in metadata).",
            video_name, n_batches_failed, n_nan_frames,
        )

    from castle.core.project import update_config
    latent_key = f"{session_id}/{latent_filename}" if session_id else latent_filename
    with update_config(storage_path, project_name) as config:
        config.setdefault('latent', {})[latent_key] = video_name

    logger.info(
        "Multi-GPU extraction complete for %s (%d frames across GPUs %s) -> %s",
        video_name, video_len, list(device_ids), latent_path,
    )
    return latent_path


def extract_roi_latent_from_video_auto(*args, **kwargs) -> str:
    """Dispatch to multi-GPU extraction when opted-in, else single-GPU.

    Multi-GPU runs only when ``CASTLE_MULTI_GPU`` is truthy AND more than one CUDA
    device is visible; otherwise (the default) the single-GPU path runs unchanged.
    Same signature / return as :func:`extract_roi_latent_from_video`.
    """
    from castle.core.gpu_pool import multi_gpu_enabled
    if multi_gpu_enabled():
        return extract_roi_latent_from_video_2gpu(*args, **kwargs)
    return extract_roi_latent_from_video(*args, **kwargs)


