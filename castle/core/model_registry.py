"""
castle/core/model_registry.py
Singleton registry for managing deep learning model lifecycle.

Handles lazy loading, explicit unloading, and CUDA memory management
for SAM, DeAOT, and DINOv2/DINOv3 models between pipeline stages.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Generator, Optional

import torch

from castle.core.logging_config import setup_logger

logger = setup_logger(__name__)


class _TrackingModelSentinel:
    """Placeholder for SAM/DeAOT models owned by ROITracker.

    ROITracker loads and owns its own model weights; the registry
    registers this sentinel so ``unload()`` triggers a GPU memory flush
    after the tracking stage without needing to own the weights directly.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"<TrackingModelSentinel name={self.name!r}>"


class ModelRegistry:
    """Singleton managing lifecycle of SAM, DeAOT, and DINOv2/DINOv3 models.

    Lazy loading: models are only loaded when first needed.
    Explicit unloading: release CUDA memory between pipeline stages.

    Usage::

        registry = ModelRegistry.instance()

        # Explicit load / unload
        model = registry.load("dinov2_vitb14")
        registry.unload("dinov2_vitb14")

        # Context manager (auto-unload on exit)
        with registry.use("dinov2_vitb14") as model:
            latent = model.extract_tensor_batch(frames, masks, roi_id)

        # Bulk operations
        registry.unload_family("sam", "deaot")
        registry.unload_all()

        # VRAM diagnostics
        stats = registry.get_memory_stats()
    """

    _instance: Optional["ModelRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ModelRegistry":
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._models: dict = {}
                cls._instance = inst
        return cls._instance

    @classmethod
    def instance(cls) -> "ModelRegistry":
        """Return the global ModelRegistry singleton."""
        if cls._instance is None:
            cls()
        return cls._instance  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_visual_encoder(name: str) -> bool:
        """True if *name* refers to a DINOv2/DINOv3 visual encoder."""
        n = name.lower()
        return "dinov2" in n or "dinov3" in n

    @staticmethod
    def _is_tracking_model(name: str) -> bool:
        """True if *name* refers to a SAM or DeAOT tracking model."""
        n = name.lower()
        return "sam" in n or "deaot" in n or "aot" in n

    @staticmethod
    def _flush_cuda() -> None:
        """Empty CUDA cache if a GPU is available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("ModelRegistry: torch.cuda.empty_cache() called.")

    def _log_memory(self, tag: str = "") -> None:
        """Log current VRAM usage at DEBUG level."""
        if torch.cuda.is_available():
            stats = self.get_memory_stats()
            logger.debug(
                "VRAM [%s] alloc=%dMB reserved=%dMB free=%dMB total=%dMB",
                tag,
                stats["allocated_mb"],
                stats["reserved_mb"],
                stats["free_mb"],
                stats["total_mb"],
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, name: str, device: str = "auto") -> object:  # noqa: ARG002
        """Load a model by name; return cached instance if already loaded.

        For visual encoders (DINOv2/DINOv3) this delegates to
        :func:`castle.core.models.get_visual_encoder`.  For tracking models
        (SAM/DeAOT) a lightweight sentinel is registered so that memory
        accounting and :meth:`unload` work correctly (the actual weights
        are managed by ``ROITracker``).

        Args:
            name:   Model name, e.g. ``'dinov2_vitb14'`` or ``'r50_deaotl'``.
            device: Target device string or ``'auto'`` (resolved via
                    :func:`castle.core.environment.get_device`).

        Returns:
            Loaded encoder instance or ``_TrackingModelSentinel``.
        """
        if name in self._models:
            logger.debug("ModelRegistry cache hit: %s", name)
            return self._models[name]

        logger.info("ModelRegistry: loading model '%s'", name)

        if self._is_visual_encoder(name):
            from castle.core.models import get_visual_encoder  # noqa: PLC0415

            model: object = get_visual_encoder(name)

        elif self._is_tracking_model(name):
            logger.info(
                "ModelRegistry: tracking model '%s' is managed by ROITracker; "
                "registering sentinel for memory accounting.",
                name,
            )
            model = _TrackingModelSentinel(name)

        else:
            raise ValueError(
                f"Unknown model name: '{name}'. "
                "Expected a DINOv2/DINOv3 encoder name or a tracking model name "
                "(sam, deaot, r50_deaotl, swinb_deaotl, …)."
            )

        self._models[name] = model
        self._log_memory(f"after-load:{name}")
        return model

    def unload(self, name: str) -> None:
        """Unload a model and free CUDA memory.

        If the model is a visual encoder, the underlying ``torch.nn.Module``
        is deleted and the :mod:`castle.core.models` module-level cache is
        also cleared.  A ``torch.cuda.empty_cache()`` call follows regardless
        of model type.

        Args:
            name: Model name to unload.
        """
        if name not in self._models:
            logger.debug("ModelRegistry.unload: '%s' not loaded — skipping.", name)
            return

        logger.info("ModelRegistry: unloading model '%s'", name)
        model = self._models.pop(name)

        if self._is_visual_encoder(name):
            # Free the underlying torch.nn.Module to release GPU tensors.
            from castle.core import models as _models_mod  # noqa: PLC0415

            if hasattr(model, "model") and model.model is not None:
                del model.model
                model.model = None  # type: ignore[attr-defined]
            # Also remove from the models-module cache.
            _models_mod._model_cache.pop(name, None)

        del model
        self._flush_cuda()
        self._log_memory(f"after-unload:{name}")

    def unload_all(self) -> None:
        """Unload all managed models and free CUDA memory."""
        for name in list(self._models.keys()):
            self.unload(name)
        logger.info("ModelRegistry: all models unloaded.")

    def unload_family(self, *families: str) -> None:
        """Unload all models whose name contains any of the given keywords.

        Args:
            *families: Case-insensitive keywords, e.g.
                       ``'sam'``, ``'deaot'``, ``'dinov2'``, ``'dinov3'``.
        """
        to_unload = [
            name
            for name in list(self._models.keys())
            if any(f.lower() in name.lower() for f in families)
        ]
        for name in to_unload:
            self.unload(name)
        if to_unload:
            logger.info("ModelRegistry: unloaded family %s → %s", families, to_unload)
        else:
            # Still flush cache in case ROITracker left GPU residue.
            self._flush_cuda()

    @contextmanager
    def use(
        self,
        name: str,
        device: str = "auto",
        unload_after: bool = True,
    ) -> Generator:
        """Context manager: load a model, yield it, then optionally unload.

        Args:
            name:         Model name.
            device:       Target device or ``'auto'``.
            unload_after: If ``True`` (default), unload on context exit.

        Yields:
            Loaded model instance.
        """
        model = self.load(name, device=device)
        try:
            yield model
        finally:
            if unload_after:
                self.unload(name)

    def get_memory_stats(self) -> dict:
        """Return current GPU/CPU memory usage statistics.

        Returns:
            dict with keys:

            * ``device``        — active device string (``'cpu'`` or ``'cuda:N'``)
            * ``allocated_mb``  — CUDA memory currently allocated (MB)
            * ``reserved_mb``   — CUDA memory reserved by the caching allocator (MB)
            * ``free_mb``       — VRAM not yet claimed (MB)
            * ``total_mb``      — total VRAM on the device (MB)
            * ``loaded_models`` — list of model names currently in the registry
        """
        stats: dict = {
            "device": "cpu",
            "allocated_mb": 0,
            "reserved_mb": 0,
            "free_mb": 0,
            "total_mb": 0,
            "loaded_models": list(self._models.keys()),
        }

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            stats["device"] = f"cuda:{idx}"
            stats["allocated_mb"] = torch.cuda.memory_allocated(idx) // (1024 * 1024)
            stats["reserved_mb"] = torch.cuda.memory_reserved(idx) // (1024 * 1024)
            free, total = torch.cuda.mem_get_info(idx)
            stats["free_mb"] = free // (1024 * 1024)
            stats["total_mb"] = total // (1024 * 1024)

        return stats
