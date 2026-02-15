"""
castle/core/environment.py
Environment detection and setup.
"""

import os
import sys
import platform
import torch

class Environment:
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

        # HDF5 locking fix
        # E-03: Logging side effects
        print("Setting HDF5_USE_FILE_LOCKING = FALSE for Colab environment")
        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# Global instance
env = Environment()


def get_device() -> str:
    """Return the detected device string ('cuda', 'mps', or 'cpu').
    
    This is the single canonical source for device detection across CASTLE.
    All modules should use this instead of implementing their own detection.
    """
    return env.device
