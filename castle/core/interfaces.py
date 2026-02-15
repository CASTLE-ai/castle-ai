"""
castle/core/interfaces.py
Protocols for Castle AI core components.
"""

from typing import Protocol, Any, Iterable, List, Optional
import numpy as np

class ProgressReporter(Protocol):
    """Protocol for progress reporting, compatible with tqdm."""
    def __call__(self, iterable: Iterable, desc: str = "", total: Optional[int] = None) -> Iterable: ...

class NotificationCallback(Protocol):
    """Interface for general notifications without UI framework dependency."""
    def __call__(self, message: str, level: str = "info") -> None: ...


class ModelObserver(Protocol):
    """Protocol for DINO/Visual models."""
    n_feature: int
    
    def extract_tensor_batch(self, frames: Any, masks: Any, select_roi: int) -> List[Any]:
        """Extract latent from a batch of tensors (efficient)."""
        ...
        
    def extract_batch_latent(self, frames: List[np.ndarray], masks: List[np.ndarray], select_roi: int) -> List[Any]:
        """Extract latent from a batch of numpy arrays."""
        ...
