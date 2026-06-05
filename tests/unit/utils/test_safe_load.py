"""Unit tests for :mod:`castle.utils.safe_load` (UX-05 / P2-B).

Some happy-path + missing-key cases already live in
``tests/integration/test_p1_perf_robustness.py``; this file widens the unit
coverage for shape/finite-value behaviour and missing-file errors.
"""

from __future__ import annotations

import numpy as np
import pytest

from castle.core.types import LatentCorruptError
from castle.utils.safe_load import load_latent_safe


def _save_npz(path, latent: np.ndarray) -> None:
    np.savez_compressed(path, latent=latent)


def test_load_latent_safe_happy_path(tmp_path) -> None:
    p = tmp_path / "good.npz"
    rng = np.random.default_rng(0)
    expected = rng.standard_normal((32, 8)).astype(np.float32)
    _save_npz(p, expected)

    arr = load_latent_safe(p)
    assert arr.shape == (32, 8)
    assert arr.dtype == np.float32
    np.testing.assert_array_equal(arr, expected)


def test_load_latent_safe_missing_file_raises(tmp_path) -> None:
    missing = tmp_path / "nope.npz"
    with pytest.raises(LatentCorruptError, match="not found"):
        load_latent_safe(missing)


def test_load_latent_safe_missing_key_raises(tmp_path) -> None:
    p = tmp_path / "wrong_key.npz"
    np.savez_compressed(p, embeddings=np.zeros((4, 2)))
    with pytest.raises(LatentCorruptError, match="missing the 'latent' key"):
        load_latent_safe(p)


def test_load_latent_safe_wrong_shape_raises(tmp_path) -> None:
    p = tmp_path / "1d.npz"
    _save_npz(p, np.zeros(16, dtype=np.float32))
    with pytest.raises(LatentCorruptError, match="expected 2D"):
        load_latent_safe(p)


def test_load_latent_safe_nonfinite_converted_to_nan(tmp_path, caplog) -> None:
    p = tmp_path / "nan.npz"
    arr = np.array([[1.0, 2.0], [np.nan, np.inf]], dtype=np.float32)
    _save_npz(p, arr)

    with caplog.at_level("WARNING"):
        out = load_latent_safe(p)

    assert out.shape == (2, 2)
    assert any("non-finite" in rec.message for rec in caplog.records)
    # Inf is normalised to NaN so EVERY downstream non-finite check excludes the
    # row (np.isnan(inf) was False, letting Inf crash the embedding later).
    assert np.isnan(out[1, 0])              # was NaN
    assert np.isnan(out[1, 1])              # was +inf -> NaN
    assert out[0, 0] == 1.0 and out[0, 1] == 2.0   # finite values untouched


def test_load_latent_safe_corrupt_file_raises(tmp_path) -> None:
    p = tmp_path / "truncated.npz"
    p.write_bytes(b"PK\x03\x04not-a-real-zip")
    with pytest.raises(LatentCorruptError, match="corrupted"):
        load_latent_safe(p)
