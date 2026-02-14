import os
import logging
import h5py
import numpy as np

logger = logging.getLogger(__name__)


class H5IO:
    """HDF5 I/O handler for mask storage.
    
    Supports context manager protocol for safe resource management:
        with H5IO('masks.h5') as h5:
            mask = h5.read_mask(0)
    """
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.config = dict()
        self.reset()

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
        return str(index) in self.f

    def read_mask(self, index):
        if str(index) not in self.f:
            raise ValueError(f"Without mask at frame {index}")
        return self.f[str(index)][:]

    def read_config(self, key):
        value = self.f[key][()]
        logger.debug('read_config %s = %s', key, value)
        return value

    def write_config(self, key, value):
        logger.debug('write_config %s = %s', key, value)
        try:
            # Assume the dataset exists and try to overwrite
            dset = self.f[key]
            dset[...] = value
        except KeyError:
            # If it doesn't exist, create it
            self.f.create_dataset(key, data=value)

    def check(self):
        self.reset_count += 1
        if self.reset_count > 5000:
            self.reset()

    def reset(self):
        self.reset_count = 0
        if hasattr(self, 'f'):
            try:
                self.f.close()
            except Exception as e:
                print(f"Warning: Failed to close HDF5 file: {e}")

        mode = 'a' if os.path.isfile(self.file_path) else 'w'
        self.f = h5py.File(self.file_path, mode)

    def get_n_rois(self):
        if 'n_rois' in self.f:
            return int(self.f['n_rois'][()])
        
        # Calculate n_rois from masks if missing
        n_rois = 0
        for key in self.f.keys():
            if key.isdigit():
                mask = self.f[key][()]
                if mask is not None and mask.size > 0:
                    n_rois = max(n_rois, int(np.max(mask)))
        
        # Fix the h5 file
        self.write_config('n_rois', n_rois)
        return n_rois
    
    def __len__(self):
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


