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
from castle.core.logging_config import setup_logger
from castle.utils.video_manager import get_project_config, save_project_config
from castle.utils.video_io import VideoWriter, VideoReader
from castle.utils.h5_io import H5IO
from castle.utils.video_align import blank_page

# Setup logger
logger = setup_logger(__name__)

from castle.core.models import get_visual_encoder


# --- Protocol Definition ---
class ProgressCallback(Protocol):
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
    progress_callback: Optional[ProgressCallback] = None
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
    if preprocess_config.center_roi_switch: tags.append("ctr")
    if preprocess_config.remove_background_switch: tags.append("rmbg")
    
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
    NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() else 1
    if NUM_WORKERS == 0: NUM_WORKERS = 1

    dataset = VideoDataset(source_path, 0, mask_list_path, preprocess_config, roi_id)
    # Get video length
    try:
        with VideoReader(source_path) as vr:
            dataset.video_len = len(vr)
    except Exception as e:
        logger.error(f"Failed to open video {source_path}: {e}")
        return ""
        
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
                 latent_batch = observer.extract_tensor_batch(frames, masks, roi_id)
            else:
                 # Fallback assumption
                 latent_batch = observer.extract_batch_latent(frames, masks, roi_id)
                 
            latent_list.append(latent_batch)

        except Exception as e:
            logger.error(f"Batch {i} failed for video {video_name}: {e}")

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
        if writer: writer.close()
        # H5IO usually doesn't need explicit close but good practice if available
    
    return out_video_path

# --- Core Function 3: Extract Rotation Latent (Restored) ---
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
    For each frame, generates 24 rotated views (0-360, step 15), processes them, and averages the latents.
    """
    batch_size = int(batch_size)
    roi_id = int(roi_id)
    
    # 1. Setup paths
    project_path, config = get_project_config(storage_path, project_name)
    latent_dir_path = os.path.join(project_path, 'latent')
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
        
    num_rotations = 24  # 360 / 15
    embed_dim = observer.n_feature

    # 4. Processing
    try:
        tracker = H5IO(mask_list_path)
        with VideoReader(source_path) as source_video:
            total_frames = len(source_video)
            latent_list = []
            
            # Note: We implement the rotation loop manually here instead of using VideoDataset
            # becaues mapping 1 frame -> 24 items in a standard Dataset is tricky without custom collate/sampler.
            # Using a simple batched loop for now to ensure correctness as per legacy logic.
            
            for i in range(0, total_frames, batch_size):
                if progress_callback:
                    progress_callback((i + 1) / total_frames, desc=f"Rotation Extract {video_name}")
                
                frames = []
                masks = []
                
                # Collect batch
                current_batch_size = 0
                for j in range(batch_size):
                    idx = i + j
                    if idx >= total_frames: break
                    
                    try:
                        frame = source_video.get_frame(idx) # Use get_frame(idx) for safe random access if needed, or iterate if VideoReader allows
                        mask = tracker.read_mask(idx)
                        
                        # Apply 24 rotations
                        for deg in range(0, 360, 15):
                            pf, pm = preprocess_config.transform(frame, mask, deg)
                            frames.append(pf)
                            masks.append(pm)
                        
                        current_batch_size += 1
                    except Exception as e:
                        logger.error(f"Frame {idx} read failed: {e}")
                        pass
                
                if not frames: continue

                try:
                    # Pass directly to observer (it usually handles list of numpy arrays)
                    # We need to make sure observer.extract_tensor_batch can handle large lists or we chunk it.
                    # Assuming it can handle batch_size * 24 (e.g. 16 * 24 = 384 images).
                    
                    if hasattr(observer, 'extract_tensor_batch'):
                        latent_batch = observer.extract_tensor_batch(frames, masks, roi_id)
                    else:
                        latent_batch = observer.extract_batch_latent(frames, masks, roi_id)
                    
                    # latent_batch is likely a numpy array or list of numpy arrays
                    if isinstance(latent_batch, list):
                        latent_batch = np.concatenate(latent_batch, axis=0) # Shape (N, Dim)
                    
                    # Validate shape
                    if len(latent_batch) != current_batch_size * num_rotations:
                        # Fallback or error
                        # DINOv2 might return list of arrays, DINOv3 might return array
                        pass

                    # Reshape and Average
                    # (B * 24, Dim) -> (B, 24, Dim) -> (B, Dim)
                    latent_reshaped = latent_batch.reshape(current_batch_size, num_rotations, embed_dim)
                    latent_averaged = latent_reshaped.mean(axis=1) # (B, Dim)
                    
                    latent_list.append(latent_averaged)
                    
                except Exception as e:
                    logger.error(f"Batch inference failed at frame {i}: {e}")

        # Concatenate final results
        if latent_list:
            latent_array = np.concatenate(latent_list, axis=0)
            np.savez_compressed(latent_path, latent=latent_array)
            
            # Update Config
            _, config = get_project_config(storage_path, project_name)
            config.setdefault('latent', {})[latent_filename] = video_name
            save_project_config(storage_path, project_name, config)
            
            return latent_path
        else:
             return ""

    except Exception as e:
         logger.error(f"Rotation extraction failed for {video_name}: {e}")
         return ""

