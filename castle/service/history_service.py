"""Undo/Redo history for clustering operations."""
from dataclasses import dataclass
from typing import Optional, List
import copy
import numpy as np


@dataclass
class ClusterSnapshot:
    """A snapshot of clustering state at a point in time."""
    cluster: np.ndarray          # cluster assignment array
    cluster_meta: dict           # {id: {name, color}}
    embedding: Optional[np.ndarray] = None  # UMAP embedding (2D)
    description: str = ""        # Human-readable description of what changed
    parent_cluster: Optional[np.ndarray] = None       # parent Latent .cluster
    parent_cluster_meta: Optional[dict] = None         # parent Latent .cluster_meta


class HistoryManager:
    """Manages undo/redo history for clustering operations."""

    def __init__(self, max_history: int = 50):
        self._undo_stack: List[ClusterSnapshot] = []
        self._redo_stack: List[ClusterSnapshot] = []
        self._max_history = max_history

    def _snapshot_from_latent(self, latent, description: str = "") -> ClusterSnapshot:
        """Create a snapshot from a latent object (Latent or LocalLatent)."""
        # LocalLatent uses .export for metadata; Latent uses .cluster_meta
        cluster = getattr(latent, 'cluster', None)
        if cluster is not None:
            cluster = cluster.copy()
        else:
            cluster = np.array([])

        meta = getattr(latent, 'cluster_meta', None)
        if meta is None:
            meta = getattr(latent, 'export', {})
        meta = copy.deepcopy(meta)

        embedding = None
        if hasattr(latent, 'embedding') and latent.embedding is not None:
            embedding = latent.embedding.copy()

        return ClusterSnapshot(
            cluster=cluster,
            cluster_meta=meta,
            embedding=embedding,
            description=description,
        )

    def _apply_snapshot(self, snapshot: ClusterSnapshot, latent):
        """Apply a snapshot's state onto a latent object (Latent or LocalLatent)."""
        # Always restore cluster — an empty/None array is a valid state (pre-DBSCAN)
        latent.cluster = snapshot.cluster
        # Restore metadata to the correct attribute
        if hasattr(latent, 'cluster_meta'):
            latent.cluster_meta = snapshot.cluster_meta
        elif hasattr(latent, 'export'):
            latent.export = snapshot.cluster_meta
        if snapshot.embedding is not None and hasattr(latent, 'embedding'):
            latent.embedding = snapshot.embedding

    def save_state(self, latent, description: str = "", parent=None):
        """Save current state before a mutation. Optionally also snapshot parent."""
        snapshot = self._snapshot_from_latent(latent, description)
        if parent is not None and hasattr(parent, 'cluster') and hasattr(parent, 'cluster_meta'):
            snapshot.parent_cluster = parent.cluster.copy()
            snapshot.parent_cluster_meta = copy.deepcopy(parent.cluster_meta)
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > self._max_history:
            self._undo_stack.pop(0)
        self._redo_stack.clear()  # New action invalidates redo

    def undo(self, latent, parent=None) -> Optional[str]:
        """Restore previous state. Returns description of undone action, or None."""
        if not self._undo_stack:
            return None

        # Save current state to redo stack
        redo_snap = self._snapshot_from_latent(latent)
        if parent is not None and hasattr(parent, 'cluster') and hasattr(parent, 'cluster_meta'):
            redo_snap.parent_cluster = parent.cluster.copy()
            redo_snap.parent_cluster_meta = copy.deepcopy(parent.cluster_meta)
        self._redo_stack.append(redo_snap)

        # Restore previous state
        prev = self._undo_stack.pop()
        self._apply_snapshot(prev, latent)
        # Also restore parent if snapshot has parent state
        if parent is not None and prev.parent_cluster is not None:
            parent.cluster = prev.parent_cluster
            parent.cluster_meta = prev.parent_cluster_meta
        return prev.description

    def redo(self, latent, parent=None) -> Optional[str]:
        """Re-apply undone action. Returns description, or None."""
        if not self._redo_stack:
            return None

        # Save current state to undo stack
        undo_snap = self._snapshot_from_latent(latent)
        if parent is not None and hasattr(parent, 'cluster') and hasattr(parent, 'cluster_meta'):
            undo_snap.parent_cluster = parent.cluster.copy()
            undo_snap.parent_cluster_meta = copy.deepcopy(parent.cluster_meta)
        self._undo_stack.append(undo_snap)

        next_state = self._redo_stack.pop()
        self._apply_snapshot(next_state, latent)
        if parent is not None and next_state.parent_cluster is not None:
            parent.cluster = next_state.parent_cluster
            parent.cluster_meta = next_state.parent_cluster_meta
        return next_state.description

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    @property
    def undo_description(self) -> str:
        return self._undo_stack[-1].description if self._undo_stack else ""

    @property
    def redo_description(self) -> str:
        return self._redo_stack[-1].description if self._redo_stack else ""

    def clear(self):
        self._undo_stack.clear()
        self._redo_stack.clear()
