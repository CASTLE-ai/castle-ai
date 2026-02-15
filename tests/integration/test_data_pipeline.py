
import pytest
import os
import shutil
import numpy as np
import cv2
import torch
from castle.core.data import VideoDataset, Preprocess
from torch.utils.data import DataLoader

@pytest.mark.integration
def test_data_pipeline_flow(dummy_project, device):
    """
    Test VideoDataset -> DataLoader -> Batch Output.
    """
    storage_path, project_name, video_name = dummy_project
    source_path = os.path.join(storage_path, project_name, "sources", video_name)
    track_path = os.path.join(storage_path, project_name, "track", video_name, "mask_list.h5")
    
    # Init Preprocess
    preprocess = Preprocess(
        center_roi_switch=False,
        center_roi_id=1,
        center_roi_crop_width=224,
        center_roi_crop_height=224,
        rotate_roi_tail_switch=False, 
        rotate_roi_tail_id=2,
        remove_background_switch=False
    )
    
    # Init Dataset
    dataset = VideoDataset(
        video_path=source_path,
        video_len=30, # From dummy video fixture
        mask_path=track_path,
        preprocess=preprocess,
        select_roi=1
    )
    
    assert len(dataset) == 30 # Dummy video len
    
    # Init Loader
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Iterate one batch
    batch = next(iter(loader))
    frames, masks = batch
    
    # Check shapes
    # Frames: (B=4, H, W, C=3)
    # Masks: (B=4, H, W)
    assert frames.shape[0] == 4
    assert masks.shape[0] == 4
    
    # Check normalization (if applied in dataset) or raw content
    # VideoDataset returns raw frames (uint8) usually?
    # Let's check type
    assert frames.dtype == torch.uint8
    
