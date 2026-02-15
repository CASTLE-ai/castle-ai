import numpy as np
from castle.core.data import Preprocess


def test_preprocess_default():
    p = Preprocess()
    assert p.center_roi_switch == False
    assert p.center_roi_crop_width == 300


def test_preprocess_custom():
    p = Preprocess(center_roi_switch=True, center_roi_crop_width=400)
    assert p.center_roi_switch == True
    assert p.center_roi_crop_width == 400
