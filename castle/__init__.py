# Set must-be-early env defaults (e.g. HDF5_USE_FILE_LOCKING) BEFORE any heavy
# library can be imported through a CASTLE submodule. Kept stdlib-only so it
# never breaks the "import castle pulls in no torch/numpy/h5py" invariant.
from castle.core._early_env import apply_early_env as _apply_early_env

_apply_early_env()

__version__ = "0.0.18"

# Lazy-load heavy model generators to avoid importing torch/torchvision/SAM/AOT
# on `import castle`. This makes CLI and lightweight usage fast.
_LAZY_IMPORTS = {
    'generate_dinov2': 'castle.utils.visual_latent_extract',
    'generate_dinov3': 'castle.utils.visual_latent_extract',
    'generate_aot': 'castle.utils.video_object_segment',
    'generate_sa': 'castle.utils.image_segment',
}


def __getattr__(name):
    """Lazy-load heavy model generators."""
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module 'castle' has no attribute {name!r}")
