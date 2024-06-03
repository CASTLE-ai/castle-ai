import os
import h5py

# Maybe can save with format tiff for imageJ

class H5IO:
    def __init__(self, file_path):
        self.file_path = file_path
        self.config = dict()
        self.reset()


    def write_mask(self, index, mask):
        self.check()
        if str(index) in self.f:
            dset = self.f[str(index)] # for Overwrite previous results
            dset[:] = mask
        else:
            dset = self.f.create_dataset(str(index), mask.shape, dtype='uint8', compression="gzip", compression_opts=3)
            dset[:] = mask

    def read_mask(self, index):
        assert str(index) in self.f, f"Without mask at frame {index}"
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


    def __del__(self):
        if hasattr(self, 'f'):
            self.f.close()





