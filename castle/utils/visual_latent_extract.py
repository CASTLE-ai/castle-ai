"""
castle/utils/visual_latent_extract.py
Wrapper module for backward compatibility.
Delegates to castle.core.models.
"""

from typing import Optional, Any
import os
import torch
from castle.core.models import get_visual_encoder, VisualEncoder

# Backward compatibility wrappers

def generate_dinov2(model_type: str = 'dinov2_vitb14', **kwargs) -> VisualEncoder:
    """Wrapper to get DINOv2 encoder from core."""
    return get_visual_encoder(model_type)

def generate_dinov3(model_type: str = 'dinov3_vitb16', notify_func=None, **kwargs) -> VisualEncoder:
    """Wrapper to get DINOv3 encoder from core."""
    # notify_func was used for Gradio info, can be ignored or logged
    return get_visual_encoder(model_type)

def download_dinov3_ckpt(model_name: str) -> str:
    """
    Placeholder for DINOv3 download logic.
    Originally this was in this file, but the file was reported missing.
    Please restore the download logic or update castle/core/config.py with IDs.
    """
    import os
    from castle.core.config import DEFAULT_CKPT_DIR
    
    os.makedirs(DEFAULT_CKPT_DIR, exist_ok=True)
    # Check if file exists
    # Assuming filenames based on previous context or standard naming
    # Defaulting to a safe return or error
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Request to download {model_name}. Logic pending restoration.")
    # Return a dummy path or expected path to prevent immediate crashing if file happens to be there
    expected_path = os.path.join(DEFAULT_CKPT_DIR, f"{model_name}.pth")
    if os.path.exists(expected_path):
        return expected_path
    
    # If we knew the IDs, we would use gdown here.
    # raise NotImplementedError("DINOv3 Download logic missing. Please restore or add IDs to config.")
    return expected_path

# Helper alias if the original code used these classes directly (imports typically masked by generate functions)
# But strictly speaking, extract_ui.py imported generate_dinov2, generate_dinov3.
