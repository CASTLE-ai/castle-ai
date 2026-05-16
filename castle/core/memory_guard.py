"""Pre-flight memory guard: estimate RAM/VRAM requirements before extraction.

Call check() before starting a batch extraction to detect OOM risk early,
instead of discovering it mid-run after wasted time.

Memory model
------------
Peak VRAM ≈ (model_static + activation_per_sample × effective_batch) × overhead

Where:
  effective_batch = batch_size × N_ROTATIONS  (if rotate=True)
                  = batch_size                 (if rotate=False)

Key insight:
  - Multiscale SPP (scales=[1,2,4]): model forward pass runs ONCE per batch.
    SPP is post-processing on already-extracted patch tokens — VRAM impact
    is negligible (~2 MB output tensor) and is NOT a multiplier.
  - Rotate based on Tail: RotationDataset flattens B×R frames into one
    batch (B*R, H, W, C) before the forward pass. Default R=7 rotations
    means the effective GPU batch is 7× larger. This IS the dominant factor.
"""
from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Default rotation count in RotationDataset (data.py: num_rotations=7)
N_ROTATIONS: int = 7

# Static model footprint in MB (fp32 — halve for bf16)
_STATIC_MB: dict[str, float] = {
    "dinov3_vitb16": 350.0,
    "dinov3_vitl16": 700.0,
    "dinov2_vitb14_reg4_pretrain": 350.0,
    "dinov2_vitb14": 350.0,
    "dinov2_vitl14": 700.0,
    "dinov3_vits16": 150.0,
}

# Activation MB per sample per forward pass (fp32 — halve for bf16).
# Empirical estimate at 592×592 input with SDPA.
_ACT_MB: dict[str, float] = {
    "dinov3_vitb16": 50.0,
    "dinov3_vitl16": 100.0,
    "dinov2_vitb14_reg4_pretrain": 50.0,
    "dinov2_vitb14": 50.0,
    "dinov2_vitl14": 100.0,
    "dinov3_vits16": 25.0,
}

_FALLBACK_STATIC_MB = 400.0
_FALLBACK_ACT_MB = 60.0
_OVERHEAD = 1.25  # PyTorch allocator fragmentation + misc


def get_available_bytes(device: str) -> Optional[int]:
    """Return free VRAM (cuda) or available RAM (cpu/mps). None if unknown.

    Args:
        device: "cuda", "cpu", or "mps".

    Returns:
        Free bytes, or None when psutil / torch is not available.
    """
    if device == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                return int(torch.cuda.mem_get_info()[0])
        except Exception:
            pass
        return None
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        return None


def _bf16(device: str) -> bool:
    """True when bf16 is supported on this device (Ampere+ CUDA)."""
    if device != "cuda":
        return False
    try:
        import torch
        return torch.cuda.is_bf16_supported()
    except Exception:
        return False


def estimate_bytes(
    model_type: str,
    batch_size: int,
    *,
    rotate: bool = False,
    bf16: bool = False,
) -> int:
    """Estimate peak bytes for one extraction batch (approximate).

    Args:
        model_type: DINOv2/v3 variant name.
        batch_size: Frames per batch (as configured by the user).
        rotate: Whether Rotate based on Tail is enabled. When True,
            RotationDataset expands each batch to batch_size × N_ROTATIONS
            frames before the forward pass, multiplying activation memory.
        bf16: Whether model runs in bfloat16 (halves memory).

    Returns:
        Estimated peak bytes including PyTorch allocator overhead.

    Note:
        Multiscale SPP (pooling_method='multiscale') is NOT accounted for
        separately — SPP runs on already-extracted patch tokens and adds
        negligible VRAM overhead compared to the forward pass.
    """
    static = _STATIC_MB.get(model_type, _FALLBACK_STATIC_MB)
    act = _ACT_MB.get(model_type, _FALLBACK_ACT_MB)
    prec = 0.5 if bf16 else 1.0
    effective_batch = batch_size * (N_ROTATIONS if rotate else 1)
    return int((static * prec + act * prec * effective_batch) * _OVERHEAD * 1_000_000)


def suggest_batch_size(
    model_type: str,
    device: str,
    *,
    rotate: bool = False,
    safety_margin: float = 0.75,
) -> int:
    """Suggest largest safe batch size given available memory.

    Args:
        model_type: DINOv2/v3 variant name.
        device: "cuda" or "cpu".
        rotate: Whether Rotate based on Tail is enabled.
        safety_margin: Fraction of available memory to use. Default 0.75.

    Returns:
        Safe batch size clamped to [1, 64]. Falls back to 8 if memory
        info is unavailable.
    """
    available = get_available_bytes(device)
    if available is None:
        return 8
    bf16 = _bf16(device)
    static = _STATIC_MB.get(model_type, _FALLBACK_STATIC_MB)
    act = _ACT_MB.get(model_type, _FALLBACK_ACT_MB)
    prec = 0.5 if bf16 else 1.0
    rotation_factor = N_ROTATIONS if rotate else 1
    headroom_mb = (available * safety_margin / 1_000_000) / _OVERHEAD - static * prec
    if headroom_mb <= 0 or act <= 0:
        return 1
    return max(1, min(64, int(headroom_mb / (act * prec * rotation_factor))))


def check(
    model_type: str,
    batch_size: int,
    device: str,
    *,
    rotate: bool = False,
    safety_margin: float = 0.75,
) -> tuple[bool, str]:
    """Check OOM risk for an extraction job.

    Args:
        model_type: DINOv2/v3 variant name.
        batch_size: Frames per batch.
        device: "cuda" or "cpu".
        rotate: Whether Rotate based on Tail is enabled.
        safety_margin: Fraction of available memory considered safe.

    Returns:
        (is_risky, message). message is "" when safe.

    Example:
        >>> risky, msg = check("dinov3_vitb16", 32, "cuda", rotate=True)
        >>> if risky:
        ...     print(msg)
    """
    available = get_available_bytes(device)
    if available is None:
        return False, ""
    bf16 = _bf16(device)
    estimated = estimate_bytes(model_type, batch_size, rotate=rotate, bf16=bf16)
    safe_limit = int(available * safety_margin)
    if estimated > safe_limit:
        mem_type = "VRAM" if device == "cuda" else "RAM"
        avail_gb = available / 1e9
        est_gb = estimated / 1e9
        effective = batch_size * (N_ROTATIONS if rotate else 1)
        rotate_note = f" (batch×{N_ROTATIONS} rotations = {effective} effective)" if rotate else ""
        suggested = suggest_batch_size(model_type, device, rotate=rotate, safety_margin=safety_margin)
        return True, (
            f"⚠️ OOM risk: ~{est_gb:.1f} GB estimated{rotate_note}, "
            f"{avail_gb:.1f} GB {mem_type} free. "
            f"Suggested batch size: {suggested}."
        )
    return False, ""
