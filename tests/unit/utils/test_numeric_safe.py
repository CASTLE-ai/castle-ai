"""Tests for :mod:`castle.utils.numeric_safe` (REPRO-04 / P3-B)."""

from __future__ import annotations

import numpy as np

from castle.utils.numeric_safe import safe_minmax, safe_zscore


def test_safe_zscore_matches_classic_for_nonconstant_columns() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 4))

    safe = safe_zscore(x)
    classic = (x - x.mean(axis=0)) / x.std(axis=0)

    # With eps=1e-8 the divergence is bounded by eps * |classic|, so a
    # loose abs tolerance is fine.
    np.testing.assert_allclose(safe, classic, atol=1e-4)


def test_safe_zscore_constant_column_returns_zero_not_nan() -> None:
    x = np.zeros((10, 3), dtype=np.float64)
    x[:, 1] = 5.0   # constant column
    out = safe_zscore(x)
    assert np.all(np.isfinite(out)), "must not produce NaN/Inf"
    # The constant column should collapse to ~0.
    np.testing.assert_allclose(out[:, 1], 0.0, atol=1e-6)


def test_safe_minmax_constant_column_collapses_to_zero() -> None:
    x = np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]])
    out = safe_minmax(x)
    assert np.all(np.isfinite(out))
    # Column 0 spans [1,3] → maps to [0,1]; column 1 is constant → [0,0].
    np.testing.assert_allclose(out[:, 0], [0.0, 0.5, 1.0], atol=1e-6)
    np.testing.assert_allclose(out[:, 1], 0.0, atol=1e-6)


def test_safe_minmax_preserves_range_for_nonconstant_columns() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((50, 3))
    out = safe_minmax(x)
    # eps slightly compresses; min ≈ 0, max ≈ 1 - eps/range
    np.testing.assert_allclose(out.min(axis=0), 0.0, atol=1e-6)
    assert (out.max(axis=0) <= 1.0 + 1e-6).all()
    assert (out.max(axis=0) > 0.99).all()


def test_helpers_accept_axis_1() -> None:
    x = np.tile(np.arange(5.0), (3, 1))  # constant *rows*
    zs = safe_zscore(x, axis=1)
    mm = safe_minmax(x, axis=1)
    assert zs.shape == x.shape
    assert mm.shape == x.shape
    assert np.all(np.isfinite(zs))
    assert np.all(np.isfinite(mm))
