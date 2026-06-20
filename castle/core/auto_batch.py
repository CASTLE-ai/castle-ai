"""
castle/core/auto_batch.py
OOM-safe retry wrapper for GPU batch work.

Provides a generic wrapper that halves the batch size and retries on
out-of-memory errors. (Up-front VRAM-aware batch-size recommendation lives in
:func:`castle.core.memory_guard.suggest_batch_size`, which is what the live
extraction path uses.)
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from castle.core.logging_config import setup_logger

logger = setup_logger(__name__)


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
