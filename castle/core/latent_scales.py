"""Shared helpers for SPP (spatial-pyramid-pooling) multiscale latents.

A multiscale latent stores the per-scale pooled blocks concatenated in
ascending-scale order: scale ``s`` occupies a contiguous ``s²·C`` column block
(``C`` = base feature dim, 768 for DINOv3 ViT-B/16). Extraction writes one file
per scale (``…_spp{s}.npz``) but a legacy combined file (``…_spp1x2x4.npz``)
holds every block. These helpers parse the scale list and slice a single scale's
block — used by both the clustering aggregator (legacy raw path) and the Prepare
cache builder, which is where scales are combined *before* PCA.
"""

import os
import re
from typing import List, Optional

import numpy as np

from castle.core.types import CastleDataError


def _spp_scales_of(filename: str, scales_hint: Optional[List[int]] = None) -> List[int]:
    """SPP scale list of a latent file, ascending. Prefer the explicit metadata
    hint (``tags.pooling_scales``); otherwise parse the ``spp<AxBx…>`` filename
    tag. A weighted-average file (no ``spp`` tag) returns ``[]``.
    """
    if scales_hint:
        return sorted(int(s) for s in scales_hint)
    m = re.search(r'spp([0-9]+(?:x[0-9]+)*)', os.path.basename(filename).lower())
    if not m:
        return []
    return sorted(int(x) for x in m.group(1).split('x') if x)


def _scale_block(array: np.ndarray, file_scales: List[int], scale: int) -> np.ndarray:
    """Return the ``(N, scale²·C)`` column block for ``scale`` from a latent file
    whose columns are the concatenated multiscale blocks in ascending-scale order
    (``[s1 | s2 | …]``, each ``s²·C`` wide). ``C`` is derived as ``width // Σ s²``.
    Extraction writes scales sorted ascending, so the column order matches
    ``sorted(file_scales)``.
    """
    fs = sorted(int(s) for s in file_scales)
    units = sum(s * s for s in fs)
    if units == 0 or array.shape[1] % units != 0:
        raise CastleDataError(
            f"Latent width {array.shape[1]} is not divisible by Σs²={units} for "
            f"SPP scales {fs}; cannot slice scale {scale}. The file may be a "
            f"different pooling variant than its name implies."
        )
    base_c = array.shape[1] // units
    off = 0
    for s in fs:
        w = s * s * base_c
        if s == int(scale):
            return array[:, off:off + w]
        off += w
    raise KeyError(f"scale {scale} not in file scales {fs}")
