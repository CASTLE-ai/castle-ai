import os
import h5py
import numpy as np


class H5IO:
    def __init__(self, file_path):
        self.file_path = file_path
        self.config = dict()
        self.reset()

    def __setitem__(self, index, mask):
        self.write_mask(index, mask)

    def write_mask(self, index, mask):
        self.check()
        if str(index) in self.f:
            dset = self.f[str(index)] # for Overwrite previous results
            dset[:] = mask
        else:
            dset = self.f.create_dataset(str(index), mask.shape, dtype='uint8', compression="gzip", compression_opts=3)
            dset[:] = mask

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
        print('read_config', key, value)
        return value


    def write_config(self, key, value):
        print('write_config', key, value)
        if key in self.f:
            del self.f[key]

        self.f.create_dataset(key, data=value)


    def check(self):
        self.reset_count += 1
        if self.reset_count > 5000:
            self.reset()

    def reset(self):
        self.reset_count = 0
        if hasattr(self, 'f'):
            self.f.close()

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
    


    def __del__(self):
        if hasattr(self, 'f') and self.f.id.valid:
            self.f.close()
            del self.f


