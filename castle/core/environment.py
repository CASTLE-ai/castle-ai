"""
castle/core/environment.py
Environment detection and setup.
"""

import os
import sys
import platform
from functools import lru_cache

import torch

class Environment:
    """Runtime environment detector for CASTLE.

    Detects the operating system, whether running in Google Colab, and the
    best available compute device (CUDA, MPS, or CPU). A global singleton
    ``env`` is created at module level.

    Attributes:
        os_sys: Operating system name (e.g. 'Linux', 'Darwin').
        is_colab: True if running inside Google Colab.
        device: Best available device string ('cuda', 'mps', or 'cpu').
        allowed_paths: Paths whitelisted for Colab file access.
    """

    def __init__(self):
        self.os_sys = platform.uname().system
        self.is_colab = 'google.colab' in sys.modules
        self.device = self._detect_device()
        self.allowed_paths = []
        
    def _detect_device(self):
        # E-01: Robust MPS detection
        if self.os_sys == 'Darwin' and torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        return 'cpu'

    def setup_colab_paths(self, root_path=None):
        """Configures allowed paths for Colab."""
        if not self.is_colab:
            return
            
        # Add basic system paths
        self.allowed_paths = ['/', '.', '/content', '/usr', '/mnt']
        
        # Add root project path if provided
        if root_path:
             self.allowed_paths.append(root_path)

        # HDF5 locking fix for Colab environment
        import logging
        logging.getLogger(__name__).info("Setting HDF5_USE_FILE_LOCKING = FALSE for Colab environment")
        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# Global instance
env = Environment()


def get_device() -> str:
    """Return the detected device string ('cuda', 'mps', or 'cpu').
    
    This is the single canonical source for device detection across CASTLE.
    All modules should use this instead of implementing their own detection.
    """
    return env.device


@lru_cache(maxsize=1)
def collect_run_environment() -> dict:
    """Snapshot the runtime software/hardware stack for output provenance.

    Captured once per process (cached) and embedded into saved artifacts so a
    reproduction attempt — or a journal reviewer — can see exactly which CASTLE
    version, library stack, device, and GPU produced an output. This matters
    because the clustering backend silently resolves cuML-GPU vs umap-learn /
    sklearn-CPU (which give *different* embeddings); recording the resolved
    stack lets a failed reproduction be told apart from a backend mismatch.

    Optional / absent packages record ``None`` rather than raising; the whole
    function is best-effort and never throws.

    Returns:
        A JSON-serialisable dict: castle version, python, platform, resolved
        device, key library versions, torch CUDA/cuDNN, and GPU model name(s).
    """
    import importlib.metadata as _md

    def _ver(*dists: str) -> 'str | None':
        for dist in dists:
            try:
                return _md.version(dist)
            except Exception:
                continue
        return None

    try:
        import castle
        castle_version = getattr(castle, "__version__", "unknown")
    except Exception:
        castle_version = "unknown"

    info: dict = {
        "castle_version": castle_version,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": env.device,
        "packages": {
            "torch": _ver("torch"),
            "torchvision": _ver("torchvision"),
            "numpy": _ver("numpy"),
            "scipy": _ver("scipy"),
            "scikit-learn": _ver("scikit-learn"),
            "umap-learn": _ver("umap-learn"),
            "cuml": _ver("cuml", "cuml-cu12"),
            "transformers": _ver("transformers"),
            "gradio": _ver("gradio"),
            "h5py": _ver("h5py"),
            "av": _ver("av"),
            "opencv": _ver("opencv-python-headless", "opencv-python"),
        },
    }
    # torch CUDA / cuDNN / GPU model names — best-effort, never raise.
    try:
        info["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["cudnn"] = torch.backends.cudnn.version()
            info["gpus"] = [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ]
    except Exception:
        pass
    return info


def _env_int(name: str) -> 'int | None':
    raw = os.environ.get(name, '').strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def get_num_workers(task_type: str = 'default', *, fs_path: 'str | None' = None) -> int:
    """Get optimal number of DataLoader workers based on system resources.

    This is the single canonical source for worker count across CASTLE.
    All modules should use this instead of inline cpu_count calculations.

    Unlike a raw ``os.cpu_count()``, this uses the *usable* core count
    (cgroup/affinity aware via :mod:`castle.core.runtime_env`), so a container
    limited to N cores on a 64-core host sizes pools to N, not 64. It also
    enforces an absolute cap and, when ``fs_path`` is on a network filesystem,
    a lower cap — too many workers thrash a shared FS (HDF5 round-trips) and
    starve the GPU rather than feeding it.

    Args:
        task_type:
            'extraction' — CPU-heavy preprocessing + GPU inference; more
                workers help keep the GPU fed.
            'tracking' — GPU-heavy batch inference; fewer workers to save
                GPU memory and avoid contention.
            'default' — balanced middle-ground.
        fs_path: If given and on a network FS (CephFS/NFS), apply the lower
            ``CASTLE_NETWORK_FS_WORKERS`` cap (default 8).

    Returns:
        Number of workers (always >= 1).

    Env overrides:
        CASTLE_EXTRACTION_WORKERS / CASTLE_NUM_WORKERS — force the count
            (authoritative; bypasses the caps).
        CASTLE_MAX_EXTRACTION_WORKERS — absolute cap (default 16).
        CASTLE_NETWORK_FS_WORKERS — network-FS cap (default 8).
    """
    from castle.core import runtime_env

    override = _env_int('CASTLE_EXTRACTION_WORKERS') if task_type == 'extraction' else None
    if override is None:
        override = _env_int('CASTLE_NUM_WORKERS')
    if override is not None:
        return max(1, override)

    cpu = runtime_env.usable_cpu_count()
    if task_type == 'extraction':
        workers = max(1, cpu // 2)
    elif task_type == 'tracking':
        workers = max(1, min(4, cpu // 4))
    else:
        workers = max(1, cpu // 4)

    cap = _env_int('CASTLE_MAX_EXTRACTION_WORKERS') or 16
    workers = min(workers, cap)

    if fs_path is not None and runtime_env.is_network_fs(fs_path):
        net_cap = _env_int('CASTLE_NETWORK_FS_WORKERS') or 8
        workers = min(workers, net_cap)

    return max(1, workers)
