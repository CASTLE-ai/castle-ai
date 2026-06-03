"""
castle/core/pipeline.py
Multi-stage pipeline orchestrator for CASTLE.

Coordinates the full processing lifecycle:
  1. Tracking stage  (SAM + DeAOT)
  2. GPU memory cleanup  (unload tracking models, empty_cache)
  3. Extraction stage  (DINOv2 / DINOv3)
  4. GPU memory cleanup  (unload visual encoder, empty_cache)

VRAM usage is logged at pipeline boundaries and approximately every
100 "unit steps" (videos × batch iterations) during extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from castle.core import models as _models_mod
from castle.core.data import Preprocess
from castle.core.environment import get_device
from castle.core.logging_config import setup_logger
from castle.core.model_registry import ModelRegistry, _TrackingModelSentinel
from castle.core.project import get_project_config
from castle.service.extraction_service import extract_latent
from castle.service.tracking_service import track_videos

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# VRAM logging helper
# ---------------------------------------------------------------------------


def _log_vram(tag: str = "") -> None:
    """Log current VRAM utilisation at INFO level."""
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(idx) // (1024 * 1024)
        reserved = torch.cuda.memory_reserved(idx) // (1024 * 1024)
        free, total = torch.cuda.mem_get_info(idx)
        free_mb = free // (1024 * 1024)
        total_mb = total // (1024 * 1024)
        logger.info(
            "VRAM [%s] alloc=%dMB reserved=%dMB free=%dMB / %dMB",
            tag,
            alloc,
            reserved,
            free_mb,
            total_mb,
        )
    else:
        logger.info("VRAM [%s] device=cpu (no GPU)", tag)


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Configuration for a full CASTLE pipeline run.

    Attributes:
        storage_path:             Root storage directory.
        project_name:             Project name.
        tracking_model:           DeAOT/AOT variant (e.g. ``'r50_deaotl'``).
        track_start:              First frame index for tracking.
        track_stop:               Last frame index (``-1`` = end of video).
        skip_existing_tracking:   Skip videos already tracked.
        extraction_model:         Visual encoder (e.g. ``'dinov2_vitb14'``).
        roi_id:                   ROI index to extract.
        batch_size:               Extraction batch size.
        skip_existing_extraction: Skip videos with existing latent files.
        pooling_method:           ``'weighted_average'`` or ``'multiscale'``.
        pooling_scales:           Scales for multiscale pooling, e.g. ``[1,2,4]``.
        feature_layers:           Layer indices for multi-layer extraction.
        center_roi_switch:        Enable ROI-centring preprocessing.
        center_roi_id:            ROI to centre on.
        center_roi_crop_width:    Crop width in pixels.
        center_roi_crop_height:   Crop height in pixels.
        remove_background_switch: Mask out background before extraction.
        videos:                   Explicit list of video filenames.
                                  Empty → all videos in project config.
    """

    storage_path: str
    project_name: str

    # Tracking
    tracking_model: str = "r50_deaotl"
    track_start: int = 0
    track_stop: int = -1
    skip_existing_tracking: bool = True

    # Extraction
    extraction_model: str = "dinov2_vitb14"
    roi_id: int = 1
    batch_size: int = 16
    skip_existing_extraction: bool = True
    pooling_method: str = "weighted_average"
    pooling_scales: Optional[list] = None
    feature_layers: Optional[list] = None

    # Preprocessing
    center_roi_switch: bool = False
    center_roi_id: int = 1
    center_roi_crop_width: int = 300
    center_roi_crop_height: int = 300
    remove_background_switch: bool = False

    # Video selection
    videos: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


class Pipeline:
    """Multi-stage CASTLE pipeline with GPU memory management.

    Stages
    ------
    1. **Tracking** — calls :func:`castle.service.tracking_service.track_video`
       for each video.
    2. **Tracking cleanup** — registers sentinels for SAM + DeAOT, calls
       :meth:`ModelRegistry.unload_family`, then ``torch.cuda.empty_cache()``.
    3. **Extraction** — calls :func:`castle.service.extraction_service.extract_latent`
       for each video; logs VRAM every 100 iterations.
    4. **Extraction cleanup** — unloads DINOv2/DINOv3 via the registry and the
       :mod:`castle.core.models` module-level cache, then ``torch.cuda.empty_cache()``.
    """

    def __init__(
        self,
        config: PipelineConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> None:
        self.config = config
        self.progress_callback = progress_callback
        self.registry = ModelRegistry.instance()
        self._device = get_device()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _progress(self, fraction: float, desc: str = "") -> None:
        if self.progress_callback:
            self.progress_callback(fraction, desc)

    def _get_video_list(self) -> list:
        """Resolve the list of videos for this pipeline run."""
        if self.config.videos:
            return list(self.config.videos)
        _, cfg = get_project_config(self.config.storage_path, self.config.project_name)
        return sorted(cfg.get("source", []))

    # ------------------------------------------------------------------
    # Stage 1: Tracking
    # ------------------------------------------------------------------

    def run_tracking_stage(self, video_list: list) -> dict:
        """Run tracking for all videos.

        Args:
            video_list: List of video filenames.

        Returns:
            Dict mapping ``video_name → status string``.
        """
        _log_vram("tracking-start")
        done = {'n': 0}

        def _cb(frac: float, desc: str) -> None:
            # Tracking is the first half of the overall pipeline progress.
            self._progress(frac * 0.5, desc)

        def _on_done(video_name: str, status: str) -> None:
            logger.info("Pipeline: tracking %s → %s", video_name, status)
            done['n'] += 1
            # VRAM log approximately every 5 videos as a rough proxy for frames.
            if done['n'] % 5 == 0:
                _log_vram(f"tracking-video-{done['n']}")

        # track_videos spreads whole videos across GPUs when CASTLE_MULTI_GPU is
        # set (>1 CUDA device), else runs sequentially — identical results.
        results = track_videos(
            storage_path=self.config.storage_path,
            project_name=self.config.project_name,
            video_names=video_list,
            model=self.config.tracking_model,
            start=self.config.track_start,
            stop=self.config.track_stop,
            skip_existing=self.config.skip_existing_tracking,
            progress_callback=_cb,
            on_video_done=_on_done,
        )
        return results

    def _cleanup_tracking(self) -> None:
        """Unload SAM + DeAOT sentinels and flush the CUDA cache."""
        logger.info("Pipeline: cleaning up tracking stage (SAM + DeAOT)…")
        _log_vram("before-tracking-cleanup")

        # Register sentinels so unload_family() flushes GPU memory correctly
        # even though ROITracker owns the actual weights.
        for name in [self.config.tracking_model, "sam"]:
            if name not in self.registry._models:
                self.registry._models[name] = _TrackingModelSentinel(name)

        self.registry.unload_family("sam", "deaot", "aot")

        # Belt-and-suspenders: explicit empty_cache regardless of registry state.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _log_vram("after-tracking-cleanup")

    # ------------------------------------------------------------------
    # Stage 2: Extraction
    # ------------------------------------------------------------------

    def run_extraction_stage(self, video_list: list) -> dict:
        """Run feature extraction for all videos.

        Logs VRAM approximately every 100 video-level iterations (a rough
        proxy for 100 frames when batch_size ≈ 1).

        Args:
            video_list: List of video filenames.

        Returns:
            Dict mapping ``video_name → latent file path`` (empty string on
            failure).
        """
        preprocess_config = Preprocess(
            center_roi_switch=self.config.center_roi_switch,
            center_roi_id=self.config.center_roi_id,
            center_roi_crop_width=self.config.center_roi_crop_width,
            center_roi_crop_height=self.config.center_roi_crop_height,
            remove_background_switch=self.config.remove_background_switch,
        )

        results: dict = {}
        n = len(video_list)
        _log_vram("extraction-start")

        for i, video_name in enumerate(video_list):
            logger.info("Pipeline: extraction [%d/%d] %s", i + 1, n, video_name)
            self._progress(0.5 + (i / n) * 0.5, f"Extracting {video_name}")

            path = extract_latent(
                storage_path=self.config.storage_path,
                project_name=self.config.project_name,
                video_name=video_name,
                model=self.config.extraction_model,
                roi=self.config.roi_id,
                batch_size=self.config.batch_size,
                preprocess_config=preprocess_config,
                skip_existing=self.config.skip_existing_extraction,
                pooling_method=self.config.pooling_method,
                pooling_scales=self.config.pooling_scales,
                feature_layers=self.config.feature_layers,
            )
            results[video_name] = path

            # Log VRAM every ~100 "steps" (video iterations as a proxy).
            if (i + 1) % 100 == 0:
                _log_vram(f"extraction-step-{i + 1}")

        return results

    def _cleanup_extraction(self) -> None:
        """Unload DINOv2/DINOv3 encoder and flush the CUDA cache."""
        logger.info("Pipeline: cleaning up extraction stage (DINOv2/DINOv3)…")
        _log_vram("before-extraction-cleanup")

        # Unload via registry (also strips models._model_cache entry).
        self.registry.unload_family("dinov2", "dinov3")

        # Belt-and-suspenders: evict the models-module cache too.
        _models_mod._evict_model_cache()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _log_vram("after-extraction-cleanup")

    # ------------------------------------------------------------------
    # Full pipeline entry point
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Execute the full pipeline: tracking → cleanup → extraction → cleanup.

        Returns:
            dict with keys:

            * ``"tracking"``      — ``{video_name: status}``
            * ``"extraction"``    — ``{video_name: latent_path}``
            * ``"memory_stats"``  — final :meth:`ModelRegistry.get_memory_stats`
        """
        video_list = self._get_video_list()
        if not video_list:
            logger.warning(
                "Pipeline: no videos found for project '%s'.",
                self.config.project_name,
            )
            return {
                "tracking": {},
                "extraction": {},
                "memory_stats": self.registry.get_memory_stats(),
            }

        logger.info(
            "Pipeline: starting — project='%s' videos=%d device=%s",
            self.config.project_name,
            len(video_list),
            self._device,
        )
        _log_vram("pipeline-start")

        # ---- Stage 1: Tracking ----
        tracking_results = self.run_tracking_stage(video_list)
        self._cleanup_tracking()

        # ---- Stage 2: Extraction ----
        extraction_results = self.run_extraction_stage(video_list)
        self._cleanup_extraction()

        _log_vram("pipeline-end")
        self._progress(1.0, "Pipeline complete")

        return {
            "tracking": tracking_results,
            "extraction": extraction_results,
            "memory_stats": self.registry.get_memory_stats(),
        }
