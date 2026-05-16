"""Unit tests for :mod:`castle.utils.h5_io` (UX-05 / P2-B).

Focus: round-trip, batch read semantics, and the context-manager contract.
We already have a richer ``tests/unit/test_h5io.py`` — this file fills the
specific gaps for the new ``read_masks_batch`` API and missing-key handling.
"""

from __future__ import annotations

import numpy as np
import pytest


def _make_mask(value: int) -> np.ndarray:
    """Build a small 8×8 mask filled with ``value``."""
    return np.full((8, 8), value, dtype=np.uint8)


def test_read_masks_batch_returns_dict_of_existing_keys(tmp_path) -> None:
    from castle.utils.h5_io import H5IO

    path = tmp_path / "masks.h5"
    with H5IO(str(path)) as h:
        for i in range(5):
            h.write_mask(i, _make_mask(i + 1))

    with H5IO(str(path)) as h:
        batch = h.read_masks_batch([0, 2, 4])

    assert set(batch.keys()) == {0, 2, 4}
    assert batch[0][0, 0] == 1
    assert batch[2][0, 0] == 3
    assert batch[4][0, 0] == 5


def test_read_masks_batch_skips_missing_keys(tmp_path) -> None:
    """Missing indices are omitted, not exception-raising — pre-scan resilience."""
    from castle.utils.h5_io import H5IO

    path = tmp_path / "masks_sparse.h5"
    with H5IO(str(path)) as h:
        h.write_mask(0, _make_mask(7))
        h.write_mask(3, _make_mask(11))

    with H5IO(str(path)) as h:
        batch = h.read_masks_batch([0, 1, 2, 3, 99])

    assert set(batch.keys()) == {0, 3}


def test_has_mask_distinguishes_present_and_absent(tmp_path) -> None:
    from castle.utils.h5_io import H5IO

    path = tmp_path / "masks.h5"
    with H5IO(str(path)) as h:
        h.write_mask(1, _make_mask(1))
        assert h.has_mask(1) is True
        assert h.has_mask(99) is False


def test_read_mask_missing_raises(tmp_path) -> None:
    from castle.utils.h5_io import H5IO

    path = tmp_path / "masks.h5"
    with H5IO(str(path)) as h:
        h.write_mask(0, _make_mask(0))
        with pytest.raises(ValueError, match="Without mask"):
            h.read_mask(42)
