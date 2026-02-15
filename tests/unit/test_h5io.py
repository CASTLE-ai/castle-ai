import tempfile, os, numpy as np
from castle.utils.h5_io import H5IO


def test_h5io_write_read():
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        path = f.name
    try:
        h5 = H5IO(path)
        mask = np.ones((100, 100), dtype=np.uint8)
        h5.write_mask(0, mask)
        read_mask = h5.read_mask(0)
        assert read_mask.shape == (100, 100)
        h5.close()
    finally:
        os.unlink(path)


def test_h5io_context_manager():
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        path = f.name
    try:
        with H5IO(path) as h5:
            assert h5 is not None
    finally:
        os.unlink(path)


def test_h5io_len_empty():
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        path = f.name
    try:
        with H5IO(path) as h5:
            length = len(h5)
            assert isinstance(length, int)
            assert length == 0
    finally:
        os.unlink(path)
