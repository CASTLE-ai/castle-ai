"""HDF5 I/O for reading and writing tracking masks."""

import os
import logging
import threading
import h5py
import numpy as np

logger = logging.getLogger(__name__)


class H5IO:
    """HDF5 I/O handler for mask storage.

    Thread-safe implementation with periodic flush instead of close/reopen.

    Supports context manager protocol for safe resource management:
        with H5IO('masks.h5') as h5:
            mask = h5.read_mask(0)

    Set ``read_only=True`` for read-only access (extraction pipeline, DataLoader
    workers, video-mix readers).  Read-only handles open ``mode='r'`` so they
    cannot acquire HDF5's exclusive write lock and cannot corrupt the file if
    a sibling writer is active (HDF5 SWMR-style protection).
    """

    def __init__(self, file_path, read_only: bool = False):
        self.file_path = file_path
        self.config = dict()
        self._lock = threading.Lock()
        self.count = 0
        self.read_only = read_only
        if read_only:
            if not os.path.isfile(self.file_path):
                raise FileNotFoundError(
                    f"H5IO(read_only=True) requires existing file: {self.file_path}"
                )
            mode = 'r'
        else:
            mode = 'a' if os.path.isfile(self.file_path) else 'w'
        self.f = h5py.File(self.file_path, mode)

    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure file is closed."""
        self.close()
        return False

    def __setitem__(self, index, mask):
        self.write_mask(index, mask)

    def write_mask(self, index, mask):
        if self.read_only:
            raise PermissionError(
                f"H5IO opened read-only: cannot write_mask({index}) to {self.file_path}"
            )
        with self._lock:
            self.check()
            key = str(index)
            try:
                # Assume the dataset exists and try to overwrite
                dset = self.f[key]
                dset[:] = mask
            except KeyError:
                # If it doesn't exist, create it
                self.f.create_dataset(key, data=mask, dtype='uint8', compression="gzip", compression_opts=3)

    def __getitem__(self, index):
        return self.read_mask(index)
    
    def has_mask(self, index):
        with self._lock:
            return str(index) in self.f

    def read_mask(self, index):
        with self._lock:
            if str(index) not in self.f:
                raise ValueError(f"Without mask at frame {index}")
            return self.f[str(index)][:]

    def read_masks_batch(self, indices):
        """Read multiple masks under a single HDF5 lock acquisition.

        PERF-02: pre-scan paths iterate over 10K+ masks. Acquiring the lock
        once per batch (vs once per call) cuts per-frame overhead from
        ~1 ms HDF5 round-trip to amortized ~tens of µs.

        Args:
            indices: Iterable of integer frame indices.

        Returns:
            ``dict`` mapping index → mask ndarray. Missing indices are
            omitted (no exception); the caller decides how to log them.
        """
        out = {}
        with self._lock:
            for idx in indices:
                key = str(idx)
                if key in self.f:
                    out[idx] = self.f[key][:]
        return out

    def read_config(self, key):
        with self._lock:
            value = self.f[key][()]
            logger.debug('read_config %s = %s', key, value)
            return value

    def write_config(self, key, value):
        if self.read_only:
            raise PermissionError(
                f"H5IO opened read-only: cannot write_config({key}) to {self.file_path}"
            )
        with self._lock:
            logger.debug('write_config %s = %s', key, value)
            try:
                # Assume the dataset exists and try to overwrite
                dset = self.f[key]
                dset[...] = value
            except KeyError:
                # If it doesn't exist, create it
                self.f.create_dataset(key, data=value)

    def check(self):
        """Flush periodically instead of close/reopen.

        MUST be called by a writer that already holds ``self._lock``; the
        flush mutates HDF5 internal state and would race against concurrent
        ``write_mask`` / ``write_config`` if invoked outside the lock.  The
        only call site today is :meth:`write_mask`, which holds the lock.
        """
        self.count += 1
        if self.count % 5000 == 0:
            self.f.flush()

    def get_n_rois(self):
        with self._lock:
            if 'n_rois' in self.f:
                return int(self.f['n_rois'][()])

            # Calculate n_rois from masks if missing
            n_rois = 0
            for key in self.f.keys():
                if key.isdigit():
                    mask = self.f[key][()]
                    if mask is not None and mask.size > 0:
                        n_rois = max(n_rois, int(np.max(mask)))

            # Only persist the computed n_rois when we hold a writable handle.
            # Read-only handles return the computed value without mutating the file.
            if not self.read_only:
                logger.debug('write_config n_rois = %s', n_rois)
                try:
                    self.f['n_rois'][...] = n_rois
                except KeyError:
                    self.f.create_dataset('n_rois', data=n_rois)

        return n_rois
    
    def __len__(self):
        with self._lock:
            if 'total_frames' not in self.f:
                return 0
            return int(self.f['total_frames'][()])
    
    def close(self):
        """Explicitly close the HDF5 file."""
        if hasattr(self, 'f'):
            try:
                if self.f.id.valid:
                    self.f.close()
            except Exception as e:
                logger.warning(f"Failed to close HDF5 file: {e}")

    def __del__(self):
        self.close()


