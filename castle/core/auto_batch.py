"""
castle/core/auto_batch.py
Automatic batch size computation and OOM-safe retry wrapper.

Provides GPU-aware batch size recommendation based on available VRAM
and a generic wrapper that halves batch size on out-of-memory errors.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from castle.core.logging_config import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Model weight footprint estimates (MB) — conservative upper bounds.
# Used to reserve headroom when computing the usable memory budget.
# ---------------------------------------------------------------------------
_MODEL_WEIGHT_MB: dict[str, int] = {
    "dinov2_vitb14": 330,
    "dinov2_vitl14": 1200,
    "dinov2_vits14": 85,
    "dinov2_vitb14_reg": 330,
    "dinov3_vitb16": 330,
    "dinov3_vitl16": 1200,
    "r50_deaotl": 280,
    "r50_deaots": 100,
    "swinb_deaotl": 660,
    "default": 512,
}

# Model input resolution (px²) used for per-frame memory estimation.
_MODEL_INPUT_RES: dict[str, int] = {
    "dinov2_vitb14": 518,
    "dinov2_vitl14": 518,
    "dinov2_vits14": 518,
    "dinov2_vitb14_reg": 518,
    "dinov3_vitb16": 592,
    "dinov3_vitl16": 592,
}

# Multiply raw frame bytes by this factor to account for activations,
# intermediate tensors, and allocator fragmentation.
_OVERHEAD_FACTOR: float = 3.5

# Keep 25 % of available VRAM as headroom.
_SAFETY_FACTOR: float = 0.75

_MAX_BATCH_SIZE: int = 128
_MIN_BATCH_SIZE: int = 1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_device() -> str:
    """Return the canonical device string from the environment module."""
    from castle.core.environment import get_device  # noqa: PLC0415

    return get_device()


def _get_free_vram_mb(device: str) -> int:
    """Return available VRAM in MB for *device*, or 0 if unavailable."""
    if device == "cpu" or not torch.cuda.is_available():
        return 0
    try:
        dev_idx = (
            int(device.split(":")[-1])
            if ":" in device
            else torch.cuda.current_device()
        )
        free, _ = torch.cuda.mem_get_info(dev_idx)
        return free // (1024 * 1024)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_optimal_batch_size(
    model_name: str,
    frame_size: tuple,
    device: str = "auto",
    dtype_bytes: int = 4,
) -> int:
    """Compute an optimal batch size based on available VRAM.

    Strategy:

    1. Resolve available VRAM via :func:`torch.cuda.mem_get_info`.
    2. Estimate per-frame memory: ``input_res² × channels × dtype_bytes × overhead``.
    3. Reserve memory for model weights (see ``_MODEL_WEIGHT_MB``).
    4. ``batch_size = usable_vram / per_frame_mem``, capped at ``_MAX_BATCH_SIZE``.

    Falls back to **4** on CPU or when VRAM information is unavailable.

    Args:
        model_name:  Model name used to look up weight reservation and input
                     resolution (e.g. ``'dinov2_vitb14'``, ``'r50_deaotl'``).
        frame_size:  ``(H, W)`` or ``(H, W, C)`` of the *source* frame in pixels.
                     When the model has a known fixed input resolution that value
                     is used instead of ``max(H, W)``.
        device:      Target device string or ``'auto'`` (auto-detected).
        dtype_bytes: Bytes per tensor element — 4 for float32, 2 for float16.

    Returns:
        Recommended batch size (int ≥ 1).
    """
    if device == "auto":
        device = _resolve_device()

    if device == "cpu":
        logger.debug("compute_optimal_batch_size: CPU device → returning 4.")
        return 4

    free_mb = _get_free_vram_mb(device)
    if free_mb == 0:
        logger.warning(
            "compute_optimal_batch_size: could not read VRAM info → returning 4."
        )
        return 4

    # ---- model weight reservation ----
    name_lower = model_name.lower()
    weight_mb = next(
        (v for k, v in _MODEL_WEIGHT_MB.items() if k in name_lower),
        _MODEL_WEIGHT_MB["default"],
    )

    # ---- per-frame memory estimate ----
    if len(frame_size) >= 3:
        h, w, c = int(frame_size[0]), int(frame_size[1]), int(frame_size[2])
    else:
        h, w = int(frame_size[0]), int(frame_size[1])
        c = 3

    input_res = next(
        (v for k, v in _MODEL_INPUT_RES.items() if k in name_lower),
        max(h, w),
    )

    per_frame_bytes = input_res * input_res * c * dtype_bytes * _OVERHEAD_FACTOR
    per_frame_mb = per_frame_bytes / (1024.0 * 1024.0)

    # ---- compute batch size ----
    usable_mb = (free_mb - weight_mb) * _SAFETY_FACTOR

    if usable_mb <= 0 or per_frame_mb <= 0:
        logger.warning(
            "compute_optimal_batch_size: usable_mb=%.1f per_frame_mb=%.3f → 1.",
            usable_mb,
            per_frame_mb,
        )
        return _MIN_BATCH_SIZE

    batch = int(usable_mb / per_frame_mb)
    batch = max(_MIN_BATCH_SIZE, min(batch, _MAX_BATCH_SIZE))

    logger.info(
        "compute_optimal_batch_size: model=%s free=%dMB weight=%dMB "
        "usable=%.0fMB per_frame=%.2fMB → batch_size=%d",
        model_name,
        free_mb,
        weight_mb,
        usable_mb,
        per_frame_mb,
        batch,
    )
    return batch


def auto_retry_on_oom(
    fn: Callable[..., Any],
    /,
    *args: Any,
    initial_batch: Optional[int] = None,
    batch_kwarg: str = "batch_size",
    min_batch: int = 1,
    **kwargs: Any,
) -> Any:
    """Call *fn* with automatic retry on GPU out-of-memory errors.

    If *fn* raises :class:`torch.cuda.OutOfMemoryError` (or a
    :class:`RuntimeError` whose message contains ``"out of memory"``), the
    batch size is halved and the call is retried.  The retry loop continues
    until *batch_size* would fall below *min_batch*, at which point the
    exception is re-raised.

    Args:
        fn:            The callable to invoke. Must accept *batch_kwarg* as a
                       keyword argument.
        *args:         Positional arguments forwarded to *fn*.
        initial_batch: Override for the starting batch size. If ``None``, the
                       value is read from *kwargs[batch_kwarg]*.
        batch_kwarg:   Name of the batch-size keyword argument (default
                       ``"batch_size"``).
        min_batch:     Minimum permissible batch size before re-raising.
        **kwargs:      Keyword arguments forwarded to *fn*.

    Returns:
        Return value of *fn* on success.

    Raises:
        RuntimeError:  When OOM persists at *min_batch*.
    """
    if initial_batch is not None:
        kwargs[batch_kwarg] = initial_batch

    batch: int = int(kwargs.get(batch_kwarg, 1))

    while True:
        try:
            logger.debug(
                "auto_retry_on_oom: calling %s with %s=%d",
                getattr(fn, "__name__", repr(fn)),
                batch_kwarg,
                batch,
            )
            return fn(*args, **kwargs)

        except RuntimeError as exc:
            is_oom = "out of memory" in str(exc).lower()
            if not is_oom:
                raise

            if batch <= min_batch:
                logger.error(
                    "auto_retry_on_oom: OOM at min batch_size=%d for %s. "
                    "Giving up.",
                    batch,
                    getattr(fn, "__name__", repr(fn)),
                )
                raise RuntimeError(
                    f"GPU OOM persists at minimum batch_size={batch} "
                    f"for {getattr(fn, '__name__', repr(fn))}."
                ) from exc

            new_batch = max(min_batch, batch // 2)
            logger.warning(
                "auto_retry_on_oom: OOM (batch=%d) → retrying with batch=%d.",
                batch,
                new_batch,
            )
            batch = new_batch
            kwargs[batch_kwarg] = batch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
