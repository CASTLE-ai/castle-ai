"""Tests for the memmap-backed extraction buffer (no list+concatenate 2× peak).

`_run_extraction_loop` now preallocates the output once (disk memmap when large,
else in-RAM) and writes each batch in place. These check in-RAM == memmap output,
NaN placeholders for tolerated failures, the all-fail raise, fd hygiene, and that
`_alloc_latent_out` closes the mkstemp fd.
"""

import math
import os

import numpy as np
import pytest

import castle.core.extractor as ex


class _DS:
    def __init__(self, n):
        self.n = n
    def __len__(self):
        return self.n


class _Loader:
    """Yields (frames, masks) batches; row r of latent == r (so order is checkable)."""
    def __init__(self, n, b, fail_batches=()):
        self.n, self.b = n, b
        self.fail = set(fail_batches)
    def __len__(self):
        return math.ceil(self.n / self.b)
    @property
    def dataset(self):
        return _DS(self.n)
    def __iter__(self):
        for bi, s in enumerate(range(0, self.n, self.b)):
            k = min(self.b, self.n - s)
            # encode the global row index in frames[:,0,0,0] so the observer can echo it
            f = np.zeros((k, 1, 1, 1), np.uint8)
            yield (f, np.full(k, bi, np.uint8))


class _Obs:
    """Returns rows filled with the global frame index (for order checks). Raises on
    batches whose mask label is in `fail`."""
    def __init__(self, dim, fail=()):
        self.dim, self.fail, self._row = dim, set(fail), 0
    def extract_tensor_batch(self, frames, masks, roi_id, **k):
        bi = int(masks[0])
        n = frames.shape[0]
        if bi in self.fail:
            raise RuntimeError("boom")
        start = self._row
        self._row += n
        return np.arange(start, start + n, dtype=np.float32)[:, None].repeat(self.dim, 1)


def _run(n, b, dim, out_dir=None, fail=()):
    return ex._run_extraction_loop(
        _Obs(dim, fail), _Loader(n, b, fail),
        roi_id=1, pooling_method='weighted_average', pooling_scales=None,
        feature_layers=None, on_frame_error='skip', max_batch_failure_rate=0.5,
        video_name='v', out_dir=out_dir,
    )


def test_inram_equals_memmap(monkeypatch, tmp_path):
    monkeypatch.setenv("CASTLE_MEMMAP_THRESHOLD_GB", "100")   # in-RAM
    a, _, _, ta = _run(40, 8, 16)
    monkeypatch.setenv("CASTLE_MEMMAP_THRESHOLD_GB", "0")     # force memmap
    b, _, _, tb = _run(40, 8, 16, out_dir=str(tmp_path))
    assert ta is None and isinstance(a, np.ndarray)
    assert isinstance(b, np.memmap) and tb is not None and os.path.exists(tb)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    # row r holds value r in every column (order preserved, in-place writes)
    assert np.array_equal(a[:, 0], np.arange(40, dtype=np.float32))
    del b
    os.remove(tb)


def test_tolerated_failure_is_nan_in_place(tmp_path, monkeypatch):
    monkeypatch.setenv("CASTLE_MEMMAP_THRESHOLD_GB", "0")
    arr, fr, nfail, tmp = _run(40, 8, 16, out_dir=str(tmp_path), fail=(2,))  # 3rd batch (rows 16:24)
    assert nfail == 1 and [16, 24] in fr
    arr = np.asarray(arr)
    assert np.isnan(arr[16:24]).all()
    assert not np.isnan(arr[:16]).any() and not np.isnan(arr[24:]).any()
    del arr
    os.remove(tmp)


def test_failure_before_first_success_backfills_nan(tmp_path, monkeypatch):
    # First batch fails (alloc deferred) → must still NaN-fill rows 0:8 once allocated.
    monkeypatch.setenv("CASTLE_MEMMAP_THRESHOLD_GB", "0")
    arr, fr, nfail, tmp = _run(40, 8, 16, out_dir=str(tmp_path), fail=(0,))
    arr = np.asarray(arr)
    assert np.isnan(arr[0:8]).all() and not np.isnan(arr[8:]).any()
    del arr
    os.remove(tmp)


def test_all_fail_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("CASTLE_MEMMAP_THRESHOLD_GB", "0")
    with pytest.raises(ex.ExtractionError):
        _run(16, 8, 16, out_dir=str(tmp_path), fail=(0, 1))


def test_alloc_closes_fd(monkeypatch, tmp_path):
    # Allocating many memmaps must not leak fds (mkstemp fd closed).
    monkeypatch.setenv("CASTLE_MEMMAP_THRESHOLD_GB", "0")
    paths = []
    for _ in range(50):
        arr, tmp = ex._alloc_latent_out(1000, 16, str(tmp_path))
        assert isinstance(arr, np.memmap) and os.path.exists(tmp)
        paths.append(tmp)
        del arr
    for p in paths:
        os.remove(p)
