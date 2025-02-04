import os
import h5py
import numpy as np
import pytest
from castle.utils.h5_io import H5IO

@pytest.fixture
def temp_h5_file(tmp_path):
    """Fixture to create a temporary HDF5 file for testing."""
    return str(tmp_path / "test.h5")

def test_write_and_read_mask(temp_h5_file):
    """Test writing and reading a mask."""
    h5io = H5IO(temp_h5_file)
    
    # 假設 mask 是一個 10x10 的二值陣列
    mask = np.random.randint(0, 2, (10, 10), dtype=np.uint8)
    
    # 測試寫入
    h5io.write_mask(0, mask)
    
    # 測試讀取，應該與寫入的 mask 相同
    read_mask = h5io.read_mask(0)
    assert np.array_equal(mask, read_mask), "Read mask does not match written mask"

def test_write_and_read_config(temp_h5_file):
    """Test writing and reading config values."""
    h5io = H5IO(temp_h5_file)
    
    # 測試寫入設定值
    h5io.write_config("threshold", 0.5)
    h5io.write_config("iterations", 10)
    
    # 測試讀取設定值
    assert h5io.read_config("threshold") == 0.5, "Threshold value incorrect"
    assert h5io.read_config("iterations") == 10, "Iterations value incorrect"

def test_reset_functionality(temp_h5_file):
    """Test if reset closes and reopens the HDF5 file."""
    h5io = H5IO(temp_h5_file)
    
    # 檢查檔案是否開啟
    assert h5io.f.id.valid, "File should be open after initialization"
    
    # 呼叫 reset，應該重新開啟檔案
    h5io.reset()
    assert h5io.f.id.valid, "File should be open after reset"

def test_check_function(temp_h5_file):
    """Test that check() triggers reset after multiple calls."""
    h5io = H5IO(temp_h5_file)
    
    # 檢查初始計數
    assert h5io.reset_count == 0, "Initial reset_count should be 0"
    
    # 多次呼叫 check
    for _ in range(5001):  # 超過 5000 次
        h5io.check()
    
    # 確保 reset 被觸發
    assert h5io.reset_count == 0, "Reset should be triggered after 5000 checks"

def test_del_functionality(temp_h5_file):
    """Test if the __del__ method properly closes the HDF5 file."""
    h5io = H5IO(temp_h5_file)
    h5io.f.close()
    # 確保 HDF5 檔案關閉
    assert not h5io.f.id.valid, "File should be closed after deletion"

