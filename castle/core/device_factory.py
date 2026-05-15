"""
castle/core/device_factory.py
Centralized device management for CASTLE.

Eliminates scattered if cpu/mps elif cuda else ... branches that appear
throughout clustering_service.py and other modules. One authoritative
place for all device-specific object construction.

Usage::

    from castle.core.device_factory import DeviceFactory

    umap  = DeviceFactory.get_umap(n_neighbors=300, min_dist=0.0, n_components=2)
    dbscan = DeviceFactory.get_dbscan(eps=0.5, min_samples=5)
    tensor = DeviceFactory.to_tensor(my_numpy_array)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class DeviceFactory:
    """Centralized device management.

    Eliminates scattered ``if cpu/mps elif cuda else ...`` branches.
    One place to define device-specific behaviour.

    The detected device is cached in ``_device`` on first access.  Override
    by calling :meth:`set_device` before any other call (e.g. in tests or
    when the user explicitly picks a device).
    """

    _device: Optional[str] = None

    # ------------------------------------------------------------------
    # Device detection
    # ------------------------------------------------------------------

    @classmethod
    def get_device(cls) -> str:
        """Auto-detect the best available device.

        Detection order: CUDA > MPS > CPU.
        Result is cached; call :meth:`reset` to re-detect.

        Returns:
            One of ``'cuda'``, ``'mps'``, or ``'cpu'``.
        """
        if cls._device is None:
            cls._device = cls._detect()
            logger.debug("DeviceFactory: detected device = %s", cls._device)
        return cls._device

    @classmethod
    def set_device(cls, device: str) -> None:
        """Override the detected device.

        Args:
            device: ``'cuda'``, ``'mps'``, or ``'cpu'``.
        """
        cls._device = device
        logger.debug("DeviceFactory: device overridden to %s", device)

    @classmethod
    def reset(cls) -> None:
        """Clear the cached device so the next call re-detects."""
        cls._device = None

    @classmethod
    def get_torch_device(cls) -> torch.device:
        """Return a :class:`torch.device` for the current device.

        Returns:
            ``torch.device('cuda')``, ``torch.device('mps')``, or
            ``torch.device('cpu')``.
        """
        return torch.device(cls.get_device())

    # ------------------------------------------------------------------
    # Tensor helpers
    # ------------------------------------------------------------------

    @classmethod
    def to_tensor(
        cls,
        data: np.ndarray,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Convert a NumPy array to a tensor on the current device.

        Args:
            data:  Input array.
            dtype: Desired tensor dtype (default ``torch.float32``).

        Returns:
            Tensor placed on the device returned by :meth:`get_device`.
        """
        return torch.from_numpy(np.asarray(data, dtype=np.float32)).to(
            dtype=dtype,
            device=cls.get_torch_device(),
        )

    # ------------------------------------------------------------------
    # Clustering algorithm factories
    # ------------------------------------------------------------------

    @classmethod
    def get_umap(cls, **kwargs):
        """Get a UMAP instance configured for the current device.

        * **GPU (CUDA)** — tries ``cuml.manifold.UMAP`` first; falls back to
          ``umap-learn`` (plain ``umap.UMAP``) if cuML is unavailable.
        * **CPU / MPS** — always uses ``umap-learn``.

        Args:
            **kwargs: Forwarded directly to the chosen UMAP constructor
                (e.g. ``n_neighbors``, ``min_dist``, ``n_components``).

        Returns:
            A UMAP estimator instance (not yet fitted).
        """
        device = cls.get_device()

        if "cuda" in device:
            try:
                from cuml.manifold import UMAP  # type: ignore[import]

                logger.debug("DeviceFactory.get_umap: using cuml.UMAP")
                return UMAP(**kwargs)
            except ImportError:
                logger.debug(
                    "DeviceFactory.get_umap: cuml unavailable, "
                    "falling back to umap-learn"
                )

        # CPU / MPS / cuML unavailable fallback
        from umap import UMAP  # type: ignore[import]  # noqa: PLC0415

        logger.debug("DeviceFactory.get_umap: using umap-learn UMAP")
        return UMAP(**kwargs)

    @classmethod
    def get_dbscan(cls, **kwargs):
        """Get a DBSCAN instance configured for the current device.

        * **GPU (CUDA)** — tries ``cuml.cluster.DBSCAN`` first; falls back to
          ``sklearn``.
        * **CPU / MPS** — always uses ``sklearn``.

        Args:
            **kwargs: Forwarded directly to the chosen DBSCAN constructor
                (e.g. ``eps``, ``min_samples``).

        Returns:
            A DBSCAN estimator instance (not yet fitted).
        """
        device = cls.get_device()

        if "cuda" in device:
            try:
                from cuml.cluster import DBSCAN  # type: ignore[import]

                logger.debug("DeviceFactory.get_dbscan: using cuml.DBSCAN")
                return DBSCAN(**kwargs)
            except ImportError:
                logger.debug(
                    "DeviceFactory.get_dbscan: cuml unavailable, "
                    "falling back to sklearn"
                )

        from sklearn.cluster import DBSCAN  # noqa: PLC0415

        logger.debug("DeviceFactory.get_dbscan: using sklearn.DBSCAN")
        return DBSCAN(**kwargs)

    @classmethod
    def get_hdbscan(cls, **kwargs):
        """Get an HDBSCAN instance configured for the current device.

        * **GPU (CUDA)** — tries ``cuml.cluster.HDBSCAN`` first; falls back to
          ``sklearn`` (≥1.3) or the ``hdbscan`` package.
        * **CPU / MPS** — tries ``sklearn.cluster.HDBSCAN`` (≥1.3), then the
          standalone ``hdbscan`` package.

        Args:
            **kwargs: Forwarded directly to the chosen HDBSCAN constructor.

        Returns:
            An HDBSCAN estimator instance (not yet fitted).
        """
        device = cls.get_device()

        if "cuda" in device:
            try:
                from cuml.cluster import HDBSCAN  # type: ignore[import]

                logger.debug("DeviceFactory.get_hdbscan: using cuml.HDBSCAN")
                return HDBSCAN(**kwargs)
            except ImportError:
                logger.debug(
                    "DeviceFactory.get_hdbscan: cuml unavailable, "
                    "falling back to sklearn/hdbscan"
                )

        # sklearn ≥ 1.3 ships HDBSCAN
        try:
            from sklearn.cluster import HDBSCAN  # noqa: PLC0415

            logger.debug("DeviceFactory.get_hdbscan: using sklearn.HDBSCAN")
            return HDBSCAN(**kwargs)
        except ImportError:
            pass

        # Standalone hdbscan package as last resort
        import hdbscan as _hdbscan  # type: ignore[import]  # noqa: PLC0415

        logger.debug("DeviceFactory.get_hdbscan: using hdbscan package")
        return _hdbscan.HDBSCAN(**kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _detect(cls) -> str:
        """Run device detection and return the best device string."""
        import platform

        os_sys = platform.uname().system

        # MPS — Apple Silicon (macOS only)
        if os_sys == "Darwin" and torch.backends.mps.is_available():
            return "mps"

        # CUDA
        if torch.cuda.is_available():
            return "cuda"

        return "cpu"
