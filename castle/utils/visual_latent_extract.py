"""
castle/utils/visual_latent_extract.py
Wrapper module for backward compatibility.
Delegates to castle.core.models.
"""

from castle.core.models import get_visual_encoder, VisualEncoder

# Backward compatibility wrappers

def generate_dinov2(model_type: str = 'dinov2_vitb14', **kwargs) -> VisualEncoder:
    """Wrapper to get DINOv2 encoder from core."""
    return get_visual_encoder(model_type)


def generate_dinov3(model_type: str = 'dinov3_vitb16', notify_func=None, **kwargs) -> VisualEncoder:
    """Wrapper to get DINOv3 encoder from core.

    Since P0-A' (2026-05-16) DINOv3 weights are pulled from HuggingFace
    on first use via ``transformers.AutoModel.from_pretrained``; there is
    no longer a manual gdown checkpoint step. ``notify_func`` is accepted
    for backwards-compatibility but ignored.
    """
    return get_visual_encoder(model_type)
