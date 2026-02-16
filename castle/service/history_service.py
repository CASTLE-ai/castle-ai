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
        if len(snapshot.cluster) > 0:
            latent.cluster = snapshot.cluster
        # Restore metadata to the correct attribute
        if hasattr(latent, 'cluster_meta'):
            latent.cluster_meta = snapshot.cluster_meta
        elif hasattr(latent, 'export'):
            latent.export = snapshot.cluster_meta
        if snapshot.embedding is not None and hasattr(latent, 'embedding'):
            latent.embedding = snapshot.embedding

    def save_state(self, latent, description: str = ""):
        """Save current state before a mutation."""
        snapshot = self._snapshot_from_latent(latent, description)
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > self._max_history:
            self._undo_stack.pop(0)
        self._redo_stack.clear()  # New action invalidates redo

    def undo(self, latent) -> Optional[str]:
        """Restore previous state. Returns description of undone action, or None."""
        if not self._undo_stack:
            return None

        # Save current state to redo stack
        self._redo_stack.append(self._snapshot_from_latent(latent))

        # Restore previous state
        prev = self._undo_stack.pop()
        self._apply_snapshot(prev, latent)
        return prev.description

    def redo(self, latent) -> Optional[str]:
        """Re-apply undone action. Returns description, or None."""
        if not self._redo_stack:
            return None

        # Save current state to undo stack
        self._undo_stack.append(self._snapshot_from_latent(latent))

        next_state = self._redo_stack.pop()
        self._apply_snapshot(next_state, latent)
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
