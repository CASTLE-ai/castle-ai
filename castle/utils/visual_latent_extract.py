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
    """Wrapper to get the DINOv3 encoder from core.

    DINOv3 weights are loaded from a local checkpoint (fetched via gdown on first
    use) — see :func:`castle.core.models.get_visual_encoder`. ``notify_func`` is
    accepted for backwards-compatibility but ignored.
    """
    return get_visual_encoder(model_type)
