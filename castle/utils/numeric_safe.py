"""Numerically safe normalisation helpers (REPRO-04 / P3-B).

When a feature column has zero variance (e.g. a constant patch in a
masked-out region), the textbook formulas

    (x - mean) / std
    (x - min)  / (max - min)

return ``inf`` / ``NaN`` because the denominator is zero. Those tainted
values then propagate through any downstream UMAP / DBSCAN call and
silently corrupt the cluster output.

``safe_zscore`` and ``safe_minmax`` add an explicit epsilon to the
denominator so constant features collapse to zero instead of NaN. They
are deliberately tiny — call sites that need numerical robustness should
prefer these over hand-rolled formulas.
"""

from __future__ import annotations

import numpy as np

__all__ = ["safe_zscore", "safe_minmax"]


def safe_zscore(
    x: np.ndarray,
    *,
    axis: int = 0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Z-score normalise without dividing by zero.

    Args:
        x: Input array. Any shape.
        axis: Axis along which to compute mean / std. Default ``0`` (per
            feature column).
        eps: Additive guard for the denominator. Default ``1e-8`` is
            small enough to be negligible for well-behaved data and
            large enough to keep constant features finite.

    Returns:
        Z-scored array, same shape as ``x``. Constant features map to
        ``0``. NaN handling: the mean/std are computed with ``nanmean`` /
        ``nanstd`` so a stray NaN in one row does not turn the whole feature
        column into NaN (only the NaN entries themselves stay NaN).
    """
    arr = np.asarray(x)
    mean = np.nanmean(arr, axis=axis, keepdims=True)
    std = np.nanstd(arr, axis=axis, keepdims=True)
    return (arr - mean) / (std + eps)


def safe_minmax(
    x: np.ndarray,
    *,
    axis: int = 0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Min-max normalise without dividing by zero.

    Args:
        x: Input array. Any shape.
        axis: Axis along which to compute min / max. Default ``0`` (per
            feature column).
        eps: Additive guard for the denominator. Default ``1e-8``.

    Returns:
        Min-max scaled array in ``[0, 1]``, same shape as ``x``.
        Constant features map to ``0``. Min/max use ``nanmin`` / ``nanmax``
        so a stray NaN does not collapse the whole column to NaN.
    """
    arr = np.asarray(x)
    mi = np.nanmin(arr, axis=axis, keepdims=True)
    mx = np.nanmax(arr, axis=axis, keepdims=True)
    return (arr - mi) / (mx - mi + eps)
