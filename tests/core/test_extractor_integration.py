
import pytest
import os
import numpy as np
from castle.core.extractor import extract_roi_latent_from_video, extract_roi_rotation_latent_from_video
from castle.core.data import Preprocess

class MockProgress:
    def __call__(self, x, desc=None):
        pass

@pytest.mark.integration
def test_extract_roi_latent_real(dummy_project, device):
    """
    Test standard latent extraction with DINOv2.
    """
    storage_path, project_name, video_name = dummy_project
    
    # Preprocess Config
    preprocess = Preprocess(
        center_roi_switch=False,
        center_roi_id=1,
        center_roi_crop_width=224,
        center_roi_crop_height=224,
        rotate_roi_tail_switch=False,
        rotate_roi_tail_id=2,
        remove_background_switch=False
    )
    
    # Use real model name that invokes download
    model_name = "dinov2_vitb14_reg4_pretrain" 
    
    # Run Extraction
    try:
        latent_path = extract_roi_latent_from_video(
            storage_path=storage_path,
            project_name=project_name,
            video_name=video_name,
            roi_id=1,
            model_name=model_name,
            batch_size=4,
            preprocess_config=preprocess,
            skip_existing=False,
            progress_callback=MockProgress()
        )
    except Exception as e:
        if "Google Drive ID" in str(e) or "403" in str(e):
             pytest.skip("Skipping model download failure (likely network/auth issue)")
        raise e

    assert os.path.exists(latent_path)
    
    # Verify Content
    data = np.load(latent_path)
    latent = data['latent']
    
    # Expected: (30 frames, 768 dim)
    assert latent.shape == (30, 768)
    # Check for NaNs
    assert not np.isnan(latent).any()

@pytest.mark.integration
def test_extract_rotation_latent_real(dummy_project, device):
    """
    Test rotation latent extraction (24 views).
    """
    storage_path, project_name, video_name = dummy_project
    
    preprocess = Preprocess(
        center_roi_switch=True, # Rotation usually implies centering
        center_roi_id=1,
        center_roi_crop_width=224,
        center_roi_crop_height=224,
        rotate_roi_tail_switch=False, 
        rotate_roi_tail_id=2,
        remove_background_switch=False
    )
    
    model_name = "dinov2_vitb14_reg4_pretrain"
    
    try:
        latent_path = extract_roi_rotation_latent_from_video(
            storage_path=storage_path,
            project_name=project_name,
            video_name=video_name,
            roi_id=1,
            model_name=model_name,
            batch_size=2, # Small batch due to 24x expansion
            preprocess_config=preprocess,
            skip_existing=False,
            progress_callback=MockProgress()
        )
    except Exception as e:
        if "Google Drive ID" in str(e): pytest.skip("Model download failed")
        raise e

    assert os.path.exists(latent_path)
    data = np.load(latent_path)
    latent = data['latent']
    
    # Should still be (30, 768) despite 24x rotation averaging
    assert latent.shape == (30, 768) 
