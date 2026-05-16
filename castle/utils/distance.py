"""GPU-accelerated pairwise distance with a CPU fallback (PERF-04 / P3-D).

``scipy.spatial.distance.cdist`` is fine for tens of points but scales
quadratically — at N=10 000 it is several seconds on a recent CPU. CASTLE
already depends on torch, so we can route through ``torch.cdist`` on
CUDA and get a 10–100× speed-up for the cluster-comparison + social-
feature paths.

The fallback is unconditional: if torch / CUDA is unavailable the
function transparently calls scipy.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

DeviceChoice = Literal["auto", "cuda", "cpu"]


def _resolve_device(device: DeviceChoice) -> str:
    """Pick the actual backend given the user's preference + hardware."""
    if device == "cpu":
        return "cpu"
    try:
        import torch

        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "device='cuda' was requested but torch.cuda.is_available() is False."
                )
            return "cuda"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def pairwise_distance(
    A: np.ndarray,  # [N, D]
    B: np.ndarray,  # [M, D]
    *,
    device: DeviceChoice = "auto",
) -> np.ndarray:
    """Compute the ``(N, M)`` Euclidean distance matrix.

    Args:
        A: First point set. Shape ``(N, D)``.
        B: Second point set. Shape ``(M, D)``. Must share ``D`` with ``A``.
        device: ``'auto'`` (default) routes to CUDA when available and
            falls back to CPU; ``'cuda'`` forces GPU and errors if
            unavailable; ``'cpu'`` forces scipy.

    Returns:
        Distance matrix of shape ``(N, M)``, dtype ``float32`` on the
        CUDA path and matches scipy on CPU.

    Raises:
        RuntimeError: ``device='cuda'`` requested but CUDA unavailable.
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(
            f"pairwise_distance expects 2D arrays; got A.ndim={A.ndim}, B.ndim={B.ndim}."
        )
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"Feature dim mismatch: A.shape[1]={A.shape[1]} vs B.shape[1]={B.shape[1]}."
        )

    backend = _resolve_device(device)
    if backend == "cuda":
        import torch

        a_t = torch.as_tensor(A, dtype=torch.float32, device="cuda")
        b_t = torch.as_tensor(B, dtype=torch.float32, device="cuda")
        with torch.inference_mode():
            d = torch.cdist(a_t, b_t)
        return d.cpu().numpy()

    from scipy.spatial.distance import cdist

    return cdist(A, B)
