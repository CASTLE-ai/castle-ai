"""Cluster representative frame export (P1 / UX-02).

Once a researcher has built UMAP + DBSCAN in the Behavior Microscope tab,
the natural next question is "what does each cluster *look* like?". This
helper picks N representative frames per cluster and writes them out as
individual PNGs plus a square montage so the researcher can inspect /
publish them.

The helper is service-layer (no Gradio import) so the Behavior Microscope
tab can wire a single button to it and the same code can be reused from a
notebook in future.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SelectionMethod = str  # "medoid" | "random"


def _pick_indices(
    cluster_global_indices: np.ndarray,
    latents_data: np.ndarray,
    *,
    n: int,
    method: SelectionMethod,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick ``n`` representative global bin indices for one cluster.

    Args:
        cluster_global_indices: Integer indices into the global ``data`` array
            for samples belonging to this cluster.
        latents_data: ``(T, F)`` global latent array. Used only for ``medoid``.
        n: Maximum number of representatives requested. Capped at the cluster
            size.
        method: ``"medoid"`` (default, closest to cluster centroid) or
            ``"random"`` (uniform draws without replacement).
        rng: Seeded RNG for the ``"random"`` path.

    Returns:
        Array of selected global indices, length ``min(n, len(cluster_global_indices))``.
    """
    if cluster_global_indices.size == 0:
        return cluster_global_indices
    n = min(int(n), int(cluster_global_indices.size))

    if method == "medoid":
        feats = latents_data[cluster_global_indices]
        center = feats.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(feats - center, axis=1)
        order = np.argsort(dists)[:n]
        return cluster_global_indices[order]
    elif method == "random":
        return rng.choice(cluster_global_indices, size=n, replace=False)
    else:
        raise ValueError(
            f"Unknown selection method {method!r}; expected 'medoid' or 'random'."
        )


def _make_grid(images: List[np.ndarray], target_side: Optional[int] = 240) -> np.ndarray:
    """Tile a list of frames into a square ish montage."""
    if not images:
        raise ValueError("Cannot build grid from empty image list")

    grid_side = int(np.ceil(np.sqrt(len(images))))
    if target_side is None:
        # Match the size of the first image (assume all the same shape)
        target_h, target_w = images[0].shape[:2]
    else:
        target_h = target_w = target_side

    rows = []
    for row in range(grid_side):
        cells: List[np.ndarray] = []
        for col in range(grid_side):
            i = row * grid_side + col
            if i < len(images):
                cell = cv2.resize(images[i], (target_w, target_h))
            else:
                cell = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            cells.append(cell)
        rows.append(np.hstack(cells))
    return np.vstack(rows)


def export_cluster_representatives(
    latents,
    aggregator,
    *,
    output_dir: Path,
    n_per_cluster: int = 9,
    selection: SelectionMethod = "medoid",
    seed: int = 42,
) -> Dict[int, List[Path]]:
    """Save N representative frames per cluster, plus a montage per cluster.

    Args:
        latents: A ``castle.utils.latent_explorer.Latent`` instance, post
            ``import_local_latent`` so its ``cluster`` array carries the
            researcher-labelled IDs.
        aggregator: A ``LatentAggregator`` that can answer
            ``get_frame(global_bin_index)`` for the same bins as
            ``latents.cluster``.
        output_dir: Directory to write into; created on demand. Existing
            files for the same cluster id are overwritten.
        n_per_cluster: Cap on the number of PNG frames written per cluster.
        selection: ``"medoid"`` (default) or ``"random"``.
        seed: Seed for the ``"random"`` selection path. Note: master seed
            applied by :func:`castle.core.seed.set_global_seed` does NOT
            cover this RNG because the picker uses ``np.random.default_rng``
            scoped to this call; pass the same ``seed`` between runs to
            reproduce.

    Returns:
        Mapping cluster id → list of written PNG paths (in selection order).

    Notes:
        Skips the noise / placeholder cluster id ``-1`` automatically.
        Skips clusters whose name is the synthetic ``"init"`` placeholder.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_meta = getattr(latents, 'cluster_meta', {}) or {}
    cluster_array = np.asarray(getattr(latents, 'cluster'))
    data = np.asarray(getattr(latents, 'data'))
    if cluster_array.ndim != 1 or data.ndim != 2:
        raise ValueError(
            f"Unexpected latents state: cluster.shape={cluster_array.shape}, "
            f"data.shape={data.shape}; expected 1D cluster + 2D data."
        )

    rng = np.random.default_rng(seed)

    representatives: Dict[int, List[Path]] = {}
    for cid in sorted({int(c) for c in cluster_meta.keys()}):
        if cid == -1:
            continue
        name = cluster_meta.get(cid, {}).get('name', f'cluster_{cid}')
        if name == 'init':
            continue

        mask = (cluster_array == cid)
        cluster_global_indices = np.flatnonzero(mask)
        if cluster_global_indices.size == 0:
            logger.info("Cluster %d (%s) is empty; skipping.", cid, name)
            continue

        chosen = _pick_indices(
            cluster_global_indices, data,
            n=n_per_cluster, method=selection, rng=rng,
        )

        frames: List[np.ndarray] = []
        written_paths: List[Path] = []
        for idx in chosen:
            frame = aggregator.get_frame(int(idx))
            if frame is None:
                logger.warning(
                    "Could not load frame for cluster %d (%s) bin %d; skipping.",
                    cid, name, int(idx),
                )
                continue
            png_path = output_dir / f"cluster_{cid:03d}_{_safe(name)}_bin{int(idx):06d}.png"
            cv2.imwrite(str(png_path), frame)
            written_paths.append(png_path)
            frames.append(frame)

        if frames:
            grid_path = output_dir / f"cluster_{cid:03d}_{_safe(name)}_grid.png"
            cv2.imwrite(str(grid_path), _make_grid(frames))
            written_paths.append(grid_path)

        representatives[cid] = written_paths

    return representatives


def _safe(name: str) -> str:
    """Make a cluster name filesystem-friendly."""
    return ''.join(c if (c.isalnum() or c in '-_') else '_' for c in name)[:32] or 'cluster'
