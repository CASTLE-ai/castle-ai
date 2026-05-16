"""Clustering protocols and pure-data container (ARCH-02 / P2-C).

The existing :class:`castle.utils.latent_explorer.Latent` mixes a data
container, UMAP execution, DBSCAN execution, cluster metadata, and even
visualisation glue. Adding a new dimensionality reducer or clusterer
(HDBSCAN, spectral, GMM …) requires editing that monolith.

This module introduces the type-level seams the refactor needs:

* :class:`LatentData` — a frozen dataclass that holds **only** the
  numeric state (raw latents, optional 2D embedding, optional cluster
  labels, cluster metadata). Pure container, no method calls.
* :class:`DimensionReducer` — the minimal :class:`typing.Protocol`
  every reducer needs to satisfy (``fit_transform``). UMAP, PCA, and
  any future plug-in all conform structurally.
* :class:`Clusterer` — likewise for ``fit_predict``-style clusterers
  (DBSCAN, HDBSCAN, KMeans …).

These are introduced without yet rewiring ``Latent``. Each entry point
that currently constructs UMAP/DBSCAN directly can migrate to "accept a
``DimensionReducer`` argument with default ``UMAPReducer()``" piecemeal
over later PRs. Defining the surface up front gives reviewers something
concrete to comment on before the big rename starts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, TypedDict, runtime_checkable

import numpy as np


class ClusterMeta(TypedDict, total=False):
    """Per-cluster display metadata.

    Keys mirror the legacy ``Latent.cluster_meta[cid]`` dicts so a
    migration can swap the type without touching call sites.
    """
    name: str
    color: str


@dataclass
class LatentData:
    """Pure numeric state for a clustering session.

    Attributes:
        raw: ``(T, F)`` latent features after temporal binning. ``T`` is
            the number of bins, ``F`` the feature dimension. Required.
        embedding: Optional ``(T, D)`` reduced embedding (usually
            ``D == 2`` for visualisation). ``None`` until a reducer has
            been run.
        cluster_ids: Optional ``(T,)`` integer cluster labels. ``-1``
            marks noise / NaN-tainted samples. ``None`` until a
            clusterer has been run.
        cluster_meta: Mapping from cluster id → ``ClusterMeta``. Empty
            until the user (or auto-label) attaches names + colours.

    Notes:
        Not a ``frozen=True`` dataclass because ``cluster_ids`` and
        ``cluster_meta`` mutate as the researcher labels clusters in
        the Behavior Microscope. The intent is "no behaviour, just a
        bag of named numpy arrays" — equivalent to a record struct.
    """
    raw: np.ndarray  # [T, F]
    embedding: Optional[np.ndarray] = None  # [T, D]
    cluster_ids: Optional[np.ndarray] = None  # [T] int
    cluster_meta: Dict[int, ClusterMeta] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        """Number of latent samples (rows in ``raw``)."""
        return int(self.raw.shape[0])

    @property
    def n_features(self) -> int:
        """Feature dimensionality (columns in ``raw``)."""
        return int(self.raw.shape[1])

    def has_embedding(self) -> bool:
        """``True`` if an embedding has been computed."""
        return self.embedding is not None

    def has_clusters(self) -> bool:
        """``True`` if cluster ids have been assigned."""
        return self.cluster_ids is not None


@runtime_checkable
class DimensionReducer(Protocol):
    """Structural type for a dimensionality reducer.

    Anything implementing :meth:`fit_transform` with a numpy ``(N, F)``
    input and a ``random_state`` keyword can be plugged in — typically
    UMAP, but PCA, t-SNE wrappers, or test stubs all satisfy this.
    """

    def fit_transform(
        self,
        X: np.ndarray,  # [N, F]
        *,
        random_state: int,
    ) -> np.ndarray:  # [N, D]
        """Reduce ``X`` to a lower-dimensional embedding.

        Args:
            X: ``(N, F)`` input features.
            random_state: Seed for the reducer's stochastic
                optimisation. UMAP/t-SNE callers must thread the
                value through to the underlying implementation —
                see [BUG-01] for why this is mandatory.

        Returns:
            ``(N, D)`` embedding. ``D`` is reducer-specific.
        """
        ...


@runtime_checkable
class Clusterer(Protocol):
    """Structural type for a (mostly density-based) clusterer.

    Implements ``fit_predict`` returning a flat integer label array
    where ``-1`` is noise. DBSCAN, HDBSCAN, OPTICS, and the test stubs
    all conform.
    """

    def fit_predict(
        self,
        X: np.ndarray,  # [N, D]
        *,
        random_state: int,
    ) -> np.ndarray:  # [N] int, -1 = noise
        """Assign cluster ids to each row of ``X``.

        Args:
            X: ``(N, D)`` input — typically a UMAP embedding.
            random_state: Seed for stochastic clusterers (KMeans).
                Density-based clusterers may ignore it but should
                still accept the keyword for API uniformity.

        Returns:
            ``(N,)`` integer labels. ``-1`` denotes noise / outlier.
        """
        ...
