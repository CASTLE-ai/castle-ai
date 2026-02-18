"""
castle/core/extractor.py
Core extraction logic execution engine.
"""

from typing import Protocol, Optional
import os
import numpy as np
from torch.utils.data import DataLoader

# Import from our new Core modules
from castle.core.data import VideoDataset, Preprocess
from castle.core.config import SUPPORTED_MODELS, ERROR_MESSAGES
from castle.core.environment import get_num_workers
from castle.core.logging_config import setup_logger
from castle.core.models import get_visual_encoder
from castle.core.project import get_project_config, save_project_config
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
) -> str:
    """
    Extracts latent features from a specific video ROI.
    Returns: Absolute path to the saved latent file.
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
        logger.warning(f"Warning: Mask not found for {video_name}")
        return ""

    # 3. Setup Models
    try:
        observer = _get_observer(model_name)
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return ""
    
    # 4. Processing
    NUM_WORKERS = get_num_workers('extraction')

    # Get video length
    try:
        with VideoReader(source_path) as vr:
            video_len = len(vr)
    except Exception as e:
        logger.error(f"Failed to open video {source_path}: {e}")
        return ""

    # Pre-scan: if rotate_roi_tail is enabled, scan all frames to collect
    # valid tail ROI points, then interpolate missing ones
    interpolated_points = None
    if preprocess_config.rotate_roi_tail_switch and preprocess_config.center_roi_switch:
        logger.info(f"Pre-scanning {video_name} for tail ROI interpolation...")
        valid_points = {}
        failed_count = 0
        tracker_scan = H5IO(mask_list_path)
        
        for idx in range(video_len):
            try:
                mask = tracker_scan.read_mask(idx)
                # Center the mask first (same as transform does)
                m = center_roi(mask, mask, preprocess_config.center_roi_id)
                point = get_roi_closest_point_safe(m, preprocess_config.rotate_roi_tail_id)
                if point is not None:
                    valid_points[idx] = point
                else:
                    failed_count += 1
            except Exception:
                failed_count += 1
        
        del tracker_scan
        
        logger.info(f"Scan complete: {len(valid_points)}/{video_len} valid, {failed_count} missing")
        
        if valid_points:
            interpolated_points = interpolate_missing_points(valid_points, video_len)
            logger.info(f"Interpolation complete: all {video_len} frames now have rotation points")

    dataset = VideoDataset(source_path, video_len, mask_list_path, preprocess_config, roi_id,
                           interpolated_points=interpolated_points)
        
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )

    latent_list = []
    total_batches = len(loader)
    
    for i, (frames, masks) in enumerate(loader):
        if progress_callback:
            progress_callback((i + 1) / total_batches, desc=f"Extracting {video_name}")
            
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

        except Exception as e:
            logger.error(f"Batch {i} failed for video {video_name}: {e}")

    if not latent_list:
        logger.error(f"No latent batches extracted for {video_name}")
        return ""

    latent_array = np.concatenate(latent_list, axis=0)

    np.savez_compressed(latent_path, latent=latent_array)
    
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
    progress_callback: Optional[ProgressCallback] = None
) -> str:
    
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
        logger.warning(f"Mask not found for {video_name}")
        return ""

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
                except Exception as e:
                    logger.error(f"Error processing frame {i} in {video_name}: {e}")
                    h, w = frame.shape[:2]
                    writer.write_frame(blank_page(h, w))
    finally:
        if writer:
            writer.close()
        # H5IO usually doesn't need explicit close but good practice if available
    
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
            pf, pm = self.preprocess.transform(frame, mask, int(deg))
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
    progress_callback: Optional[ProgressCallback] = None
) -> str:

    """
    Extracts latent features using rotation invariance strategy.
    Generates 7 rotated views (0-360), processes them, and averages the latents.
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
        logger.warning(f"Warning: Mask not found for {video_name}")
        return ""

    # 3. Setup Models
    try:
        observer = _get_observer(model_name)
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return ""
        
    embed_dim = observer.n_feature

    # 4. Processing
    NUM_WORKERS = get_num_workers('extraction')
    
    try:
        with VideoReader(source_path) as vr:
            video_len = len(vr)
    except Exception as e:
        logger.error(f"Failed to open video {source_path}: {e}")
        return ""

    dataset = RotationDataset(
        video_path=source_path,
        video_len=video_len,
        mask_path=mask_list_path,
        preprocess=preprocess_config,
        select_roi=roi_id,
        num_rotations=7
    )

    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )

    latent_list = []
    total_batches = len(loader)
    
    try:
        for i, (frames, masks) in enumerate(loader):
            if progress_callback:
                progress_callback((i + 1) / total_batches, desc=f"Extracting {video_name} (Rot)")
            
            B, R, H, W, C = frames.shape
            
            frames_flat = frames.view(B * R, H, W, C)
            masks_flat = masks.view(B * R, H, W)

            if hasattr(observer, 'extract_tensor_batch'):
                 latent_batch = observer.extract_tensor_batch(frames_flat, masks_flat, roi_id)
            else:
                 latent_batch = observer.extract_batch_latent(frames_flat, masks_flat, roi_id)
            
            if isinstance(latent_batch, list):
                latent_batch = np.array(latent_batch)
            
            latent_reshaped = latent_batch.reshape(B, R, embed_dim)
            latent_averaged = latent_reshaped.mean(axis=1)
            
            latent_list.append(latent_averaged)

        # Concatenate final results
        latent_array = np.concatenate(latent_list, axis=0)
        np.savez_compressed(latent_path, latent=latent_array)
        
        # Update Config
        _, config = get_project_config(storage_path, project_name)
        config.setdefault('latent', {})[latent_filename] = video_name
        save_project_config(storage_path, project_name, config)
        
        return latent_path

    except Exception:
        logger.error(f"Rotation extraction failed for {video_name}", exc_info=True)
        # Clean up partial file?
        if os.path.exists(latent_path):
            os.remove(latent_path)
        return ""


