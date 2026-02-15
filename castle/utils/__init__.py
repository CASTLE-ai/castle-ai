from .video_object_segment import generate_aot  # noqa: F401
from .image_segment import generate_sa  # noqa: F401


def __getattr__(name):
    """Lazy import to break circular dependency with castle.core.models."""
    if name in ('generate_dinov2', 'generate_dinov3'):
        from .visual_latent_extract import generate_dinov2, generate_dinov3
        globals()['generate_dinov2'] = generate_dinov2
        globals()['generate_dinov3'] = generate_dinov3
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")