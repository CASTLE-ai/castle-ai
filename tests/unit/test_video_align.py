import numpy as np
from castle.utils.video_align import blank_page, center_roi


def test_blank_page():
    img = blank_page(100, 200)
    assert img.shape == (100, 200, 3)
    assert img.dtype == np.uint8
    assert np.all(img == 255)
