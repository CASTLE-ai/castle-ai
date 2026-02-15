"""Unit tests for castle.core.mask_filter."""

import numpy as np
from castle.core.mask_filter import filter_largest_component, filter_by_reference


def test_filter_largest_component_empty_mask():
    mask = np.zeros((100, 100), dtype=np.uint8)
    result = filter_largest_component(mask)
    assert np.all(result == 0)


def test_filter_largest_component_single_object():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:50, 10:50] = 1  # One big component
    result = filter_largest_component(mask)
    assert np.sum(result == 1) == 40 * 40


def test_filter_largest_component_removes_small_fragments():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:50, 10:50] = 1  # Big component (1600 px)
    mask[80:83, 80:83] = 1  # Small fragment (9 px)
    result = filter_largest_component(mask, min_area=50)
    assert np.sum(result == 1) == 40 * 40  # Only big one kept


def test_filter_largest_component_multiple_rois():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:50, 10:50] = 1
    mask[60:90, 60:90] = 2
    mask[95:98, 95:98] = 2  # Small fragment of ROI 2 (9 px)
    result = filter_largest_component(mask, min_area=10)
    assert np.sum(result == 1) > 0
    assert np.sum(result == 2) == 30 * 30  # Only big ROI 2 kept


def test_filter_largest_component_preserves_all_when_single():
    """A single connected component per ROI should always be preserved."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[5:45, 5:45] = 3
    result = filter_largest_component(mask, min_area=10)
    assert np.sum(result == 3) == 40 * 40


def test_filter_by_reference():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:50, 10:50] = 1  # 1600 px
    mask[80:83, 80:83] = 1  # 9 px
    reference_areas = {1: 2000.0}
    result = filter_by_reference(mask, reference_areas, ratio=0.1)
    # Threshold = 200, so 9px fragment should be removed
    assert np.sum(result == 1) == 40 * 40


def test_filter_by_reference_empty_mask():
    mask = np.zeros((100, 100), dtype=np.uint8)
    result = filter_by_reference(mask, {}, ratio=0.1)
    assert np.all(result == 0)


def test_filter_by_reference_fallback_default():
    """ROI not in reference_areas should use default_min_area."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:50, 10:50] = 1  # 1600 px
    mask[80:83, 80:83] = 1  # 9 px
    # No reference for ROI 1 → uses default_min_area=50
    result = filter_by_reference(mask, {}, ratio=0.1, default_min_area=50)
    assert np.sum(result == 1) == 40 * 40


def test_filter_by_reference_keeps_when_above_threshold():
    """Both components above threshold → only largest kept."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 10:30] = 1  # 400 px
    mask[60:90, 60:90] = 1  # 900 px
    reference_areas = {1: 1000.0}
    # Threshold = 100, both above → keep only largest (900 px)
    result = filter_by_reference(mask, reference_areas, ratio=0.1)
    assert np.sum(result == 1) == 30 * 30
