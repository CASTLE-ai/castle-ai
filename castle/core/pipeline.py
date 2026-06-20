"""Backward-compatibility shim.

``Pipeline`` orchestrates tracking + extraction (service-layer operations), so it
now lives in :mod:`castle.service.pipeline` — keeping it in ``core`` made the
core layer import the service layer (a layering inversion). This shim re-exports
the public API so existing imports of ``castle.core.pipeline`` keep working.
"""

from castle.service.pipeline import Pipeline, PipelineConfig

__all__ = ["Pipeline", "PipelineConfig"]
