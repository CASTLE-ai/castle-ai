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
    Downloads DINOv3 checkpoint if not exists.
    """
    from castle.core.config import DEFAULT_CKPT_DIR, CKPT_DINO_IDS, DINOV3_CONSTANTS
    from castle.utils.download import download_with_gdown
    import logging
    
    logger = logging.getLogger(__name__)
    
    os.makedirs(DEFAULT_CKPT_DIR, exist_ok=True)
    
    # Get filename from constants
    filename = DINOV3_CONSTANTS['MODEL_TO_CKPT_FILENAME'].get(model_name, f"{model_name}.pth")
    ckpt_path = DEFAULT_CKPT_DIR / filename
    
    if ckpt_path.exists():
        return str(ckpt_path)
        
    logger.info(f"Downloading {model_name} to {ckpt_path}...")
    
    file_id = CKPT_DINO_IDS.get(model_name)
    if not file_id:
         # Fallback search if model_name mismatch
         pass
         
    if file_id:
        download_with_gdown(file_id, str(ckpt_path))
    else:
        logger.warning(f"No Google ID found for {model_name}, skipping download.")
        
    return str(ckpt_path)

# Helper alias if the original code used these classes directly (imports typically masked by generate functions)
# But strictly speaking, extract_ui.py imported generate_dinov2, generate_dinov3.
