"""castle/service/cluster_npz.py

The ``cluster_*.npz`` filename grammar and node-meta sidecar lookup.

CASTLE encodes a parent node's child cluster names into the embedding filename
``cluster_{c1}_{c2}_..._{ck}_.npz`` (each child has ``parent_depth + 1``
underscore-segments). These pure path/string helpers parse and locate those
files. Extracted out of the former ``clustering_service`` god-module so the
filename-as-data-channel grammar lives in one cohesive place; they have no
clustering state and depend only on the standard library.
"""

import glob
import json
import logging
import os
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def load_node_meta(cluster_path: str, parent_cluster_name: str) -> Optional[dict]:
    """Return the persisted sidecar metadata for a parent cluster node, or None.

    The sidecar is written by :func:`submit_local_to_global` when the UI
    submits a fresh round of clustering against a parent node. It holds the
    UMAP config string and DBSCAN eps used at that submission, plus the
    basename of the associated ``cluster_*.npz``.

    Args:
        cluster_path: Directory typically ``<project>/cluster/``.
        parent_cluster_name: Name of the parent cluster (e.g. ``'init_a0'``).

    Returns:
        Parsed dict, or ``None`` if the file is missing or malformed.
    """
    if not parent_cluster_name:
        return None
    meta_path = os.path.join(
        cluster_path, f'node_{parent_cluster_name}_meta.json'
    )
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, 'r') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read node meta %s: %s", meta_path, e)
        return None


def _parent_from_cluster_filename(
    basename: str,
    parent_cluster_name: str,
) -> bool:
    """Return True iff ``basename`` is the embedding npz for the supplied
    parent node.

    The filename is built in :func:`submit_local_to_global` as
    ``cluster_{c1}_{c2}_..._{ck}_.npz`` where every ``c_i`` is an immediate
    child of the parent and therefore has ``parent_depth + 1``
    underscore-segments. Parsing the filename works even after deeper
    splits have evicted intermediate nodes from ``cluster_meta`` (which
    is why an export-name based check breaks for non-deepest parents).
    """
    if not basename.startswith('cluster_') or not basename.endswith('.npz'):
        return False
    if basename == 'cluster_model.npz':
        return False
    core = basename[len('cluster_'):-len('.npz')]
    if not core.endswith('_'):
        return False
    segments = core.rstrip('_').split('_')
    parent_depth = len(parent_cluster_name.split('_'))
    seg_per_child = parent_depth + 1
    if seg_per_child <= 0 or len(segments) % seg_per_child != 0:
        return False
    child_count = len(segments) // seg_per_child
    if child_count < 1:
        return False
    parent_segs = parent_cluster_name.split('_')
    for i in range(child_count):
        chunk = segments[i * seg_per_child:(i + 1) * seg_per_child]
        if chunk[:parent_depth] != parent_segs:
            return False
    return True


def find_cluster_npz_for_parent(
    cluster_path: str,
    parent_cluster_name: str,
    latents: Any,
) -> Optional[str]:
    """Fallback locator: pick the ``cluster_*.npz`` produced when
    ``parent_cluster_name`` was last submitted.

    Used when a node has no ``node_{parent}_meta.json`` sidecar (e.g.
    submissions made before the sidecar feature landed) or when the
    sidecar points at a missing file. The parent is identified by parsing
    the canonical filename ``cluster_{c1}_..._{ck}_.npz`` — see
    :func:`_parent_from_cluster_filename`. When several files match we
    return the most recently modified one.

    Args:
        cluster_path: Directory typically ``<project>/cluster/``.
        parent_cluster_name: Parent node name (e.g. ``'init'``).
        latents: Unused; kept for backwards-compatible call sites.

    Returns:
        Absolute path to the best-matching npz, or ``None``.
    """
    del latents  # filename-only matching no longer needs cluster_meta
    if not parent_cluster_name:
        return None

    candidates = glob.glob(os.path.join(cluster_path, 'cluster_*.npz'))
    best: Optional[str] = None
    best_mtime = -1.0
    for npz in candidates:
        if not _parent_from_cluster_filename(
            os.path.basename(npz), parent_cluster_name,
        ):
            continue
        mt = os.path.getmtime(npz)
        if mt > best_mtime:
            best = npz
            best_mtime = mt
    return best


# ``cluster_*.npz`` files that are NOT per-session embedding exports (they lack
# the emb/cls keys). They must be excluded from any embedding-file glob, else a
# max-mtime pick after "Save Cluster Model" selects cluster_model.npz and reading
# ["emb"] raises KeyError.
_NON_EMBEDDING_NPZ = frozenset({"cluster_model.npz", "cluster_data.npz"})


def _embedding_npz_files(cluster_path: str) -> List[str]:
    """Per-session embedding ``cluster_*.npz`` files, newest first.

    Excludes non-embedding artefacts (cluster_model.npz / cluster_data.npz) that
    share the ``cluster_*`` prefix but have no emb/cls keys.
    """
    files = [
        f for f in glob.glob(os.path.join(cluster_path, "cluster_*.npz"))
        if os.path.basename(f) not in _NON_EMBEDDING_NPZ
    ]
    files.sort(key=os.path.getmtime, reverse=True)
    return files


def find_latest_cluster_npz(cluster_path: str) -> Optional[str]:
    """Return the most recently modified embedding ``cluster_*.npz`` in ``cluster_path``.

    Args:
        cluster_path: Directory typically ``<project>/cluster/``.

    Returns:
        Absolute path to the newest matching embedding file, or ``None`` if none
        exists. Non-embedding artefacts (cluster_model.npz / cluster_data.npz)
        are skipped.
    """
    files = _embedding_npz_files(cluster_path)
    return files[0] if files else None


def _extract_child_names_from_filename(
    basename: str,
    parent_cluster_name: str,
) -> List[str]:
    """Parse child cluster names from a ``cluster_*.npz`` filename.

    The file is named ``cluster_{c1}_{c2}_..._{ck}_.npz`` where each ``c_i``
    is an immediate child of ``parent_cluster_name``.  Because children have
    exactly ``parent_depth + 1`` underscore-segments we can recover the
    ordered list without touching ``cluster_meta``.

    Args:
        basename: Filename (no directory), e.g. ``cluster_init_a0_init_a1_.npz``.
        parent_cluster_name: Parent node name, e.g. ``'init'``.

    Returns:
        Ordered list of child names (empty list on parse failure).
    """
    if not basename.startswith('cluster_') or not basename.endswith('.npz'):
        return []
    core = basename[len('cluster_'):-len('.npz')]
    if not core.endswith('_'):
        return []
    segments = core.rstrip('_').split('_')
    parent_depth = len(parent_cluster_name.split('_'))
    seg_per_child = parent_depth + 1
    if seg_per_child <= 0 or len(segments) % seg_per_child != 0:
        return []
    child_count = len(segments) // seg_per_child
    return [
        '_'.join(segments[i * seg_per_child:(i + 1) * seg_per_child])
        for i in range(child_count)
    ]
