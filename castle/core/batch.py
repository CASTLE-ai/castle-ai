"""Backward-compatibility shim.

``BatchRunner`` orchestrates the pipeline across projects (service-layer), so it
now lives in :mod:`castle.service.batch`. This shim re-exports the public API so
existing imports of ``castle.core.batch`` keep working.
"""

from castle.service.batch import BatchConfig, BatchRunner

__all__ = ["BatchConfig", "BatchRunner"]
