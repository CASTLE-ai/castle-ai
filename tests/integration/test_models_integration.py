
import pytest
import torch
import numpy as np
from castle.core.models import get_visual_encoder, DINOv2Encoder, DINOv3Encoder

@pytest.mark.integration
def test_dinov2_load_and_infer(device):
    """
    Load DINOv2 and run a dummy batch.
    """
    model_name = "dinov2_vitb14_reg4_pretrain"
    try:
        encoder = get_visual_encoder(model_name)
        encoder.load_model()
    except Exception as e:
        pytest.skip(f"Model load failed: {e}")

    assert isinstance(encoder, DINOv2Encoder)
    assert encoder.n_feature == 768
    
    # Fake Batch: (B=2, H=518, W=518, C=3)
    # Using numpy uint8
    frames = [np.zeros((518, 518, 3), dtype=np.uint8) for _ in range(2)]
    masks = [np.ones((518, 518), dtype=np.uint8) for _ in range(2)]
    
    latents = encoder.extract_tensor_batch(frames, masks, roi_id=1)
    
    assert latents.shape == (2, 768)

@pytest.mark.integration
def test_dinov3_load_and_infer(device):
    """
    Load DINOv3 and run a dummy batch.
    """
    model_name = "dinov3_vitl16"
    try:
        encoder = get_visual_encoder(model_name)
        encoder.load_model()
    except Exception as e:
        pytest.skip(f"Model load failed: {e}")
        
    assert isinstance(encoder, DINOv3Encoder)
    assert encoder.n_feature == 1024
    
    # Fake Batch: (B=2, H=592, W=592, C=3)
    frames = [np.zeros((592, 592, 3), dtype=np.uint8) for _ in range(2)]
    masks = [np.ones((592, 592), dtype=np.uint8) for _ in range(2)]
    
    latents = encoder.extract_tensor_batch(frames, masks, roi_id=1)
    
    assert latents.shape == (2, 1024)
