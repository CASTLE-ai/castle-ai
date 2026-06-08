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
import os
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

DeviceChoice = Literal["auto", "cuda", "cpu"]


def _idlest_cuda_str() -> str:
    """``'cuda:N'`` for the idlest GPU, honouring ``CASTLE_GPU_DEVICE``; else ``'cuda'``."""
    forced = os.environ.get("CASTLE_GPU_DEVICE", "").strip().lower()
    if forced.startswith("cuda:"):
        return forced
    try:
        from castle.core import runtime_env
        idx = runtime_env.idlest_gpu()
        if idx is not None:
            return f"cuda:{idx}"
    except Exception:  # noqa: BLE001
        pass
    return "cuda"


def _resolve_device(device: DeviceChoice) -> str:
    """Pick the actual backend given the user's preference + hardware.

    On CUDA this returns a concrete ``'cuda:N'`` for the idlest GPU (most free
    VRAM) so this single-GPU op runs on the emptiest card instead of always
    cuda:0. Honours ``CASTLE_GPU_DEVICE=cuda:N``.
    """
    if device == "cpu":
        return "cpu"
    try:
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' was requested but torch.cuda.is_available() is False."
            )
        if device == "cuda" or torch.cuda.is_available():
            return _idlest_cuda_str()
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
        Distance matrix of shape ``(N, M)``. The output dtype matches the
        input dtype on BOTH backends (``float64`` in → ``float64`` out,
        otherwise ``float32``). This makes the CUDA and CPU paths
        device-consistent: with ``float64`` inputs both compute in
        ``float64``, so e.g. ``comparison.energy_distance_test`` yields the
        same p-value on GPU and CPU (the old code computed CUDA in ``float32``
        but CPU in scipy's ``float64``, drifting permutation p-values).

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

    # Preserve input precision so the CUDA and CPU paths agree: float64 inputs
    # compute in float64 on both, instead of CUDA silently downcasting to
    # float32 (which made cross-device permutation p-values disagree).
    out_dtype = np.float64 if np.dtype(A.dtype) == np.float64 else np.float32

    backend = _resolve_device(device)
    if backend == "cuda":
        import torch

        torch_dtype = torch.float64 if out_dtype == np.float64 else torch.float32
        a_t = torch.as_tensor(A, dtype=torch_dtype, device=backend)
        b_t = torch.as_tensor(B, dtype=torch_dtype, device=backend)
        with torch.inference_mode():
            d = torch.cdist(a_t, b_t)
        return d.cpu().numpy()

    from scipy.spatial.distance import cdist

    # scipy.cdist always returns float64 regardless of input dtype; cast back to
    # the input dtype so the contract ("output dtype == input dtype") holds.
    return cdist(A, B).astype(out_dtype, copy=False)
