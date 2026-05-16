"""Tests for :mod:`castle.utils.distance` (PERF-04 / P3-D)."""

from __future__ import annotations

import numpy as np
import pytest

from castle.utils.distance import pairwise_distance


def test_matches_scipy_on_cpu_path() -> None:
    """Force the CPU backend and confirm bit-identical agreement with scipy."""
    from scipy.spatial.distance import cdist

    rng = np.random.default_rng(0)
    A = rng.standard_normal((40, 6))
    B = rng.standard_normal((30, 6))
    out = pairwise_distance(A, B, device="cpu")
    np.testing.assert_allclose(out, cdist(A, B))


def test_auto_path_returns_correct_shape() -> None:
    """Whatever backend is picked, shape must be (N, M)."""
    rng = np.random.default_rng(1)
    A = rng.standard_normal((10, 4))
    B = rng.standard_normal((7, 4))
    out = pairwise_distance(A, B)
    assert out.shape == (10, 7)


def test_self_distance_diagonal_is_zero() -> None:
    rng = np.random.default_rng(2)
    A = rng.standard_normal((20, 4)).astype(np.float32)
    out = pairwise_distance(A, A, device="cpu")
    np.testing.assert_allclose(np.diag(out), 0.0, atol=1e-5)


def test_dim_mismatch_raises() -> None:
    A = np.zeros((5, 4))
    B = np.zeros((5, 3))
    with pytest.raises(ValueError, match="Feature dim mismatch"):
        pairwise_distance(A, B)


def test_non_2d_input_raises() -> None:
    with pytest.raises(ValueError, match="2D arrays"):
        pairwise_distance(np.zeros(5), np.zeros((5, 1)))


def test_cuda_requested_but_unavailable_raises() -> None:
    """device='cuda' should fail loudly when CUDA is missing."""
    try:
        import torch

        cuda_present = torch.cuda.is_available()
    except ImportError:
        cuda_present = False
    if cuda_present:
        pytest.skip("CUDA available; cannot test unavailable path")
    A = np.zeros((3, 2))
    with pytest.raises(RuntimeError, match="cuda"):
        pairwise_distance(A, A, device="cuda")
