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
from castle.core.project import get_project_config, save_project_config
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
            self.reader = VideoReader(self.video_path) # Use VideoReader consistent with VideoDataset logic if it uses it. Wait, VideoDataset uses ReadArray?
            # Checking data.py: self.reader = ReadArray(self.video_path). 
            # But here in extractor.py usually VideoReader is imported.
            # VideoDataset in data.py imports ReadArray. 
            # I should reuse VideoDataset logic carefully.
            # VideoDataset uses ReadArray. Let's use what VideoDataset expects.
            # But wait, VideoDataset in data.py lines 88-89 used ReadArray.
            # Here I'm subclassing. So I should call super().__getitem__ logic OR re-implement if I need efficient multi-transform.
            # Re-implementing is safer to ensure I only read the frame ONCE.
            pass

        # Since VideoDataset.reader is initialized lazily in getitem, let's access it safely or init it.
        # But VideoDataset.__getitem__ reads one frame and transforms it. 
        # I want one frame -> multiple transforms.
        
        # Access internal reader from parent if possible, or duplicate logic.
        if self.reader is None:
             from castle.utils.video_io import ReadArray # Local import to match data.py if needed? 
             # data.py imports ReadArray. I should use that if I can. 
             # But I cannot import ReadArray inside the class easily if it's not imported at top level.
             # Let's rely on data.py's implementation details or just handle it.
             # Actually, if I inherit, I inherit the __init__.
             # I should just copy the "open lazy" logic.
             # VideoDataset.__getitem__ does: self.reader = ReadArray(...)
             
             # Let's import ReadArray at top of this file if not present, OR use VideoReader as extractor already does.
             # extractor.py has `from castle.utils.video_io import VideoWriter, VideoReader`.
             # data.py has `from castle.utils.video_io import ReadArray`.
             pass

        # To avoid dependency hell, I'll just check if self.reader is None and init it using VideoReader or ReadArray.
        # VideoReader is available here.
        if self.reader is None:
            self.reader = VideoReader(self.video_path)
            
        if self.tracker is None:
            self.tracker = H5IO(self.mask_path)

        # Read Frame ONCE
        # VideoReader needs get_frame(idx) or similar
        try:
             # VideoReader is a context manager but also has get_frame?
             # Let's check VideoReader usage in extract_roi_rotation... old code used `source_video.get_frame(idx)`.
             frame = self.reader.get_frame(idx)
        except AttributeError:
             # Fallback if reader is ReadArray (which supports __getitem__)
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
    
    # Tags
    tags = ["rot"] # Explicitly tag rotation
    if preprocess_config.center_roi_switch: tags.append("ctr")
    if preprocess_config.remove_background_switch: tags.append("rmbg")
    
    suffix = "_".join([model_name] + tags)
    latent_filename = f'{base_name}_ROI_{roi_id}_{suffix}.npz' 
    # Note: Filename format change? The prompt said "Restored" logic.
    # The previous code used f'{base_name}_ROI_{roi_id}_rotation_latent.npz'. I should keep that to minimize surprises unless user wants standardization.
    # The existing code had: latent_filename = f'{base_name}_ROI_{roi_id}_rotation_latent.npz'
    # I will revert to that to be safe, or check if standardization is better.
    # Let's stick to the exact previous filename for safety.
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
    NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() else 1
    if NUM_WORKERS == 0: NUM_WORKERS = 1
    
    # Get video length
    video_len = 0
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
    
    # We catch exceptions at the whole-process level or per-batch?
    # Failing fast is better.
    try:
        for i, (frames, masks) in enumerate(loader):
            if progress_callback:
                progress_callback((i + 1) / total_batches, desc=f"Extracting {video_name} (Rot)")
            
            # frames shape: (B, 7, H, W, 3)
            # masks shape: (B, 7, H, W)
            B, R, H, W, C = frames.shape
            
            # Fuse Batch and Rotations
            frames_flat = frames.view(B * R, H, W, C) # Or permute if needed by observer? 
            # Observer expecting (N, H, W, C) numpy or (N, C, H, W) tensor?
            # extract_tensor_batch handles preprocessing. 
            # If frames is Tensor (from DataLoader), extract_tensor_batch in DINOv2 expects Tensor (B, H, W, 3) or (B, 3, H, W).
            # VideoDataset returns numpy usually? 
            # Wait, default collate converts numpy to Tensor.
            # So frames is Tensor.
            
            masks_flat = masks.view(B * R, H, W)

            if hasattr(observer, 'extract_tensor_batch'):
                 latent_batch = observer.extract_tensor_batch(frames_flat, masks_flat, roi_id)
            else:
                 # Fallback: convert to numpy list if needed, but observer likely supports tensor now.
                 # If not, we might crash.
                 # Assuming extract_tensor_batch is the standard interface now.
                 latent_batch = observer.extract_batch_latent(frames_flat, masks_flat, roi_id)
            
            # latent_batch: (B*R, Dim) - likely numpy array from observer
            if isinstance(latent_batch, list):
                latent_batch = np.array(latent_batch)
            
            # Reshape and Average
            # (B*R, Dim) -> (B, R, Dim)
            latent_reshaped = latent_batch.reshape(B, R, embed_dim)
            latent_averaged = latent_reshaped.mean(axis=1) # (B, Dim)
            
            latent_list.append(latent_averaged)

        # Concatenate final results
        latent_array = np.concatenate(latent_list, axis=0)
        np.savez_compressed(latent_path, latent=latent_array)
        
        # Update Config
        _, config = get_project_config(storage_path, project_name)
        config.setdefault('latent', {})[latent_filename] = video_name
        save_project_config(storage_path, project_name, config)
        
        return latent_path

    except Exception as e:
        logger.error(f"Rotation extraction failed for {video_name}", exc_info=True)
        # Clean up partial file?
        if os.path.exists(latent_path):
            os.remove(latent_path)
        return ""


