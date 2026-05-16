"""
castle/core/extractor.py
Core extraction logic execution engine.
"""

from collections import Counter
from typing import Protocol, Optional
import os
import numpy as np
from torch.utils.data import DataLoader

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
    on_frame_error: OnFrameError = "skip",
    max_batch_failure_rate: float = 0.05,
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
    
    # 1. Setup paths
    project_path, config = get_project_config(storage_path, project_name)
    
    # New Structure: latent/{model_name}/
    latent_dir_path = os.path.join(project_path, 'latent', model_name)
    os.makedirs(latent_dir_path, exist_ok=True)
    
    base_name = os.path.splitext(video_name)[0]
    
    # Tags logic
    tags = []
    if preprocess_config.center_roi_switch:
        tags.append("ctr")
    if preprocess_config.remove_background_switch:
        tags.append("rmbg")
    # A-06: Add pooling/layer tags to filename
    if pooling_method == 'multiscale' and pooling_scales:
        scales_str = "x".join(str(s) for s in sorted(pooling_scales))
        tags.append(f"spp{scales_str}")
    if feature_layers:
        layers_str = "x".join(str(layer) for layer in sorted(feature_layers))
        tags.append(f"L{layers_str}")
    
    suffix = "_".join([model_name] + tags)
    latent_filename = f'{base_name}_ROI_{roi_id}_{suffix}.npz'
    
    latent_path = os.path.join(latent_dir_path, latent_filename)

    if skip_existing and os.path.exists(latent_path):
        logger.info(f"Skipping existing latent: {latent_path}")
        return latent_path

    # 2. Load Resources
    source_path = os.path.join(storage_path, project_name, 'sources', video_name)
    track_dir_path = os.path.join(project_path, 'track', video_name)
    mask_list_path = os.path.join(track_dir_path, 'mask_list.h5')
    
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

    # 4. Processing
    NUM_WORKERS = get_num_workers('extraction')

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
        logger.info(f"Pre-scanning {video_name} for tail ROI interpolation...")
        valid_points = {}
        failure_reasons: Counter = Counter()
        tracker_scan = H5IO(mask_list_path)

        for idx in range(video_len):
            try:
                mask = tracker_scan.read_mask(idx)
            except (IOError, OSError, KeyError, ValueError) as e:
                failure_reasons["mask_read"] += 1
                logger.debug("Pre-scan mask read failed at %d: %s", idx, e)
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

        del tracker_scan

        failed_count = sum(failure_reasons.values())
        logger.info(
            "Pre-scan: %d/%d valid; failures: %s",
            len(valid_points), video_len, dict(failure_reasons),
        )

        if valid_points:
            interpolated_points = interpolate_missing_points(valid_points, video_len)
            logger.info(f"Interpolation complete: all {video_len} frames now have rotation points")

    dataset = VideoDataset(
        source_path, video_len, mask_list_path, preprocess_config, roi_id,
        interpolated_points=interpolated_points,
        on_frame_error=on_frame_error,
    )
        
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    # BUG-09: seed DataLoader workers so augmentation / shuffle is reproducible
    _gen = make_torch_generator()
    if _gen is not None:
        loader_kwargs["generator"] = _gen
        if NUM_WORKERS > 0:
            loader_kwargs["worker_init_fn"] = seed_worker
    loader = DataLoader(dataset, **loader_kwargs)

    latent_list = []
    total_batches = len(loader)
    n_batches_failed = 0
    abs_failure_threshold = max(1, int(max_batch_failure_rate * total_batches))

    for i, (frames, masks) in enumerate(loader):
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

            latent_list.append(latent_batch)

        except (ROINotFoundError, PreprocessingError):
            # Frame-level error already explained by the dataset layer; raise
            # immediately for the strict path or surface to abort logic.
            n_batches_failed += 1
            if on_frame_error == "raise" or n_batches_failed > abs_failure_threshold:
                raise
        except Exception as e:
            n_batches_failed += 1
            logger.error(
                "Batch %d/%d failed for %s: %s",
                i + 1, total_batches, video_name, e,
            )
            if n_batches_failed > abs_failure_threshold:
                raise ExtractionError(
                    f"Aborting {video_name}: {n_batches_failed}/{i + 1} batches "
                    f"failed (threshold {abs_failure_threshold} of "
                    f"{total_batches}, max_rate={max_batch_failure_rate:.0%}). "
                    f"Common causes: GPU OOM (reduce --batch-size), corrupted "
                    f"mask, all-NaN frames."
                ) from e

        if progress_callback:
            progress_callback((i + 1) / total_batches, desc=f"Extracting {video_name}")

    if not latent_list:
        raise ExtractionError(
            f"All {total_batches} batches failed for {video_name}. "
            f"No latent file written."
        )

    # BUG-05: validate feature-dimension consistency before concat so a model
    # swap mid-extraction fails loudly instead of producing a cryptic ValueError.
    expected_dim = latent_list[0].shape[1]
    mismatched = [(i, tuple(b.shape)) for i, b in enumerate(latent_list) if b.shape[1] != expected_dim]
    if mismatched:
        sample = mismatched[:5]
        raise ExtractionError(
            f"Inconsistent feature dimensions across batches for {video_name}. "
            f"Expected dim {expected_dim}; mismatched batches: {sample}"
            + ("..." if len(mismatched) > len(sample) else "")
            + ". This usually indicates a model swap mid-extraction."
        )

    latent_array = np.concatenate(latent_list, axis=0)

    np.savez_compressed(latent_path, latent=latent_array)

    if n_batches_failed:
        logger.warning(
            "Extraction for %s completed with %d/%d failed batches (below %.0f%% threshold).",
            video_name, n_batches_failed, total_batches, max_batch_failure_rate * 100,
        )

    # Update Config
    _, config = get_project_config(storage_path, project_name)
    config.setdefault('latent', {})[latent_filename] = video_name
    save_project_config(storage_path, project_name, config)

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
        tracker = H5IO(mask_list_path)
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
            self.tracker = H5IO(self.mask_path)

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
                pm = blank_page(h, w)
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

    # 1. Setup paths
    project_path, config = get_project_config(storage_path, project_name)
    latent_dir_path = os.path.join(project_path, 'latent', model_name)
    os.makedirs(latent_dir_path, exist_ok=True)

    base_name = os.path.splitext(video_name)[0]

    latent_filename = f'{base_name}_ROI_{roi_id}_rotation_latent.npz'

    latent_path = os.path.join(latent_dir_path, latent_filename)

    if skip_existing and os.path.exists(latent_path):
        logger.info(f"Skipping existing latent: {latent_path}")
        return latent_path

    # 2. Load Resources
    source_path = os.path.join(storage_path, project_name, 'sources', video_name)
    track_dir_path = os.path.join(project_path, 'track', video_name)
    mask_list_path = os.path.join(track_dir_path, 'mask_list.h5')

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

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    _gen = make_torch_generator()
    if _gen is not None:
        loader_kwargs["generator"] = _gen
        if NUM_WORKERS > 0:
            loader_kwargs["worker_init_fn"] = seed_worker
    loader = DataLoader(dataset, **loader_kwargs)

    latent_list = []
    total_batches = len(loader)
    n_batches_failed = 0
    abs_failure_threshold = max(1, int(max_batch_failure_rate * total_batches))

    try:
        for i, (frames, masks) in enumerate(loader):
            if progress_callback:
                progress_callback((i + 1) / total_batches, desc=f"Extracting {video_name} (Rot)")

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

                latent_list.append(latent_averaged)
            except (ROINotFoundError, PreprocessingError):
                n_batches_failed += 1
                if on_frame_error == "raise" or n_batches_failed > abs_failure_threshold:
                    raise
            except Exception as e:
                n_batches_failed += 1
                logger.error(
                    "Rotation batch %d/%d failed for %s: %s",
                    i + 1, total_batches, video_name, e,
                )
                if n_batches_failed > abs_failure_threshold:
                    raise ExtractionError(
                        f"Aborting {video_name}: {n_batches_failed}/{i + 1} "
                        f"rotation batches failed (threshold "
                        f"{abs_failure_threshold} of {total_batches}, "
                        f"max_rate={max_batch_failure_rate:.0%})."
                    ) from e

        if not latent_list:
            raise ExtractionError(
                f"All {total_batches} rotation batches failed for {video_name}."
            )

        # Concatenate final results
        latent_array = np.concatenate(latent_list, axis=0)
        np.savez_compressed(latent_path, latent=latent_array)

        # Update Config
        _, config = get_project_config(storage_path, project_name)
        config.setdefault('latent', {})[latent_filename] = video_name
        save_project_config(storage_path, project_name, config)

        if n_batches_failed:
            logger.warning(
                "Rotation extraction for %s completed with %d/%d failed batches.",
                video_name, n_batches_failed, total_batches,
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


