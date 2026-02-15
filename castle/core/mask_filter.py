"""
castle/core/mask_filter.py
Standalone mask filtering utilities (A-03).

Provides reusable mask post-processing that can be used independently
of the tracking pipeline — e.g. during extraction or analysis.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def filter_largest_component(mask: np.ndarray, min_area: int = 50) -> np.ndarray:
    """Filter mask to keep only the largest connected component per ROI.

    Args:
        mask: (H, W) uint8 mask where each value is a ROI ID (0=background)
        min_area: Minimum pixel area threshold

    Returns:
        Filtered mask with only largest components
    """
    if mask.max() == 0:
        return mask

    new_mask = np.zeros_like(mask)
    for obj_id in np.unique(mask):
        if obj_id == 0:
            continue

        binary = (mask == obj_id).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        if num_labels <= 1:
            continue

        # Find largest component above threshold
        best_label = -1
        best_area = -1
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area and area > best_area:
                best_area = area
                best_label = i

        if best_label != -1:
            new_mask[labels == best_label] = obj_id

    return new_mask


def filter_by_reference(
    mask: np.ndarray,
    reference_areas: Dict[int, float],
    ratio: float = 0.1,
    default_min_area: int = 50,
) -> np.ndarray:
    """Filter mask using reference area thresholds.

    For each ROI, the threshold is `reference_area * ratio`. Components
    smaller than this threshold are discarded; only the largest surviving
    component is kept.

    Args:
        mask: (H, W) uint8 mask where each value is a ROI ID (0=background)
        reference_areas: {roi_id: median_reference_area} — typically computed
            from reference frames during tracking initialization.
        ratio: Threshold ratio (default 0.1 = 10% of reference area)
        default_min_area: Fallback minimum area for ROIs not in reference_areas

    Returns:
        Filtered mask with only largest components per ROI
    """
    if mask.max() == 0:
        return mask

    # Pre-compute per-ROI thresholds
    thresholds = {k: v * ratio for k, v in reference_areas.items()}

    new_mask = np.zeros_like(mask)
    for obj_id in np.unique(mask):
        if obj_id == 0:
            continue

        threshold = thresholds.get(int(obj_id), default_min_area)

        binary = (mask == obj_id).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        if num_labels <= 1:
            # Only one component (+ background label 0)
            # Keep it if it meets threshold
            if num_labels == 1:
                continue
            area = stats[1, cv2.CC_STAT_AREA]
            if area >= threshold:
                new_mask[labels == 1] = obj_id
            continue

        # Find largest component above threshold
        best_label = -1
        best_area = -1
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > threshold and area > best_area:
                best_area = area
                best_label = i

        if best_label != -1:
            new_mask[labels == best_label] = obj_id

    return new_mask
