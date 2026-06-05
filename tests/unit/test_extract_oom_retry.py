"""Tests for CUDA-OOM handling in extraction (PR Stage E).

A CUDA OOM must NOT be swallowed as a tolerated per-frame batch failure (which
would NaN-fill the timeline or abort with a misleading error). It must surface
cleanly so auto_retry_on_oom can halve the batch and retry. These exercise the
real `_run_extraction_loop` with fake observers/loaders — no GPU needed.
"""

import math

import numpy as np
import pytest

import castle.core.extractor as ex
from castle.core.auto_batch import auto_retry_on_oom


class _Loader:
    def __init__(self, n, b):
        self.n, self.b = n, b

    def __len__(self):
        return math.ceil(self.n / self.b)

    @property
    def dataset(self):
        class _DS:
            def __len__(_self):
                return self.n
        return _DS()

    def __iter__(self):
        for s in range(0, self.n, self.b):
            k = min(self.b, self.n - s)
            yield (np.zeros((k, 1, 1, 1), np.uint8), np.zeros(k, np.uint8))


class _OomObs:
    """Raises CUDA OOM when the batch is >= oom_if_ge (None → always OOM);
    otherwise returns rows filled with the running frame index."""
    def __init__(self, dim, oom_if_ge=None):
        self.dim, self.oom_if_ge, self._row = dim, oom_if_ge, 0

    def extract_tensor_batch(self, frames, masks, roi_id, **k):
        n = frames.shape[0]
        if self.oom_if_ge is None or n >= self.oom_if_ge:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        start = self._row
        self._row += n
        return np.arange(start, start + n, dtype=np.float32)[:, None].repeat(self.dim, 1)


def _loop(obs, n, b, max_rate=0.9):
    return ex._run_extraction_loop(
        obs, _Loader(n, b),
        roi_id=1, pooling_method='weighted_average', pooling_scales=None,
        feature_layers=None, on_frame_error='skip', max_batch_failure_rate=max_rate,
        video_name='v', out_dir=None,
    )


def test_is_cuda_oom_truth_table():
    assert ex._is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate"))
    assert ex._is_cuda_oom(RuntimeError("out of memory"))
    assert not ex._is_cuda_oom(RuntimeError("some other runtime error"))
    assert not ex._is_cuda_oom(ValueError("out of memory"))  # not a RuntimeError


def test_oom_propagates_instead_of_nan_tolerating(monkeypatch):
    """Even with a generous failure budget, an OOM must raise — not be tolerated
    as a NaN-filled batch (which would corrupt the timeline)."""
    monkeypatch.setenv("CASTLE_MEMMAP_THRESHOLD_GB", "100")  # in-RAM, no temp file
    with pytest.raises(RuntimeError, match="out of memory"):
        _loop(_OomObs(16), 40, 8, max_rate=0.9)


def test_non_oom_error_still_tolerated_as_nan(monkeypatch):
    """A genuine per-frame error remains tolerated (existing behavior preserved)."""
    class _BoomFirst:
        def __init__(self):
            self._row = 0
        def extract_tensor_batch(self, frames, masks, roi_id, **k):
            n = frames.shape[0]
            if self._row == 0:           # only the first batch fails, non-OOM
                self._row += n
                raise RuntimeError("boom")
            start = self._row
            self._row += n
            return np.zeros((n, 16), np.float32)
    monkeypatch.setenv("CASTLE_MEMMAP_THRESHOLD_GB", "100")
    arr, fr, nfail, _ = _loop(_BoomFirst(), 40, 8, max_rate=0.5)
    assert nfail == 1
    assert np.isnan(np.asarray(arr)[0:8]).all()


def test_auto_retry_halves_batch_until_it_fits(monkeypatch):
    """End-to-end: the entrypoint's inner _attempt pattern wrapped in
    auto_retry_on_oom halves the batch on OOM and recovers, preserving order."""
    monkeypatch.setenv("CASTLE_MEMMAP_THRESHOLD_GB", "100")
    dim = 16

    def _attempt(batch_size):
        res = _loop(_OomObs(dim, oom_if_ge=8), 40, batch_size, max_rate=0.5)
        return res, batch_size

    (arr, _fr, _nfail, _tmp), used_batch = auto_retry_on_oom(
        _attempt, batch_size=8, min_batch=1
    )
    assert used_batch == 4  # 8 OOMs → halved to 4, which fits
    assert np.array_equal(np.asarray(arr)[:, 0], np.arange(40, dtype=np.float32))
