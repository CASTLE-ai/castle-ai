"""
castle/core/project_config.py
Project-level configuration management (B-05).

Provides typed dataclass-based configuration for all CASTLE processing
parameters. This is separate from the existing config.json which tracks
file inventory (source videos, latent files, etc.).

Usage:
    cfg = ProjectConfig()                    # defaults
    cfg = ProjectConfig.load('castle_config.json')  # from file
    cfg.save('castle_config.json')           # to file

    # Round-trip via dict
    d = cfg.to_dict()
    cfg2 = ProjectConfig.from_dict(d)
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Preprocessing parameters for latent extraction."""
    center_roi: bool = False
    center_roi_id: int = 1
    crop_width: int = 300
    crop_height: int = 300
    rotate_roi_tail: bool = False
    rotate_roi_tail_id: int = 2
    remove_background: bool = False


@dataclass
class UMAPConfig:
    """UMAP dimensionality reduction parameters."""
    n_neighbors: int = 100
    min_dist: float = 0.0
    n_components: int = 2
    n_epochs: int = 5000


@dataclass
class ClusterConfig:
    """Clustering parameters."""
    method: str = 'dbscan'
    eps: float = 1.0
    umap_stages: List[UMAPConfig] = field(default_factory=lambda: [UMAPConfig()])


@dataclass
class TrackingConfig:
    """ROI tracking parameters."""
    model: str = 'r50_deaotl'
    smart_filter_ratio: float = 0.1
    batch_size: int = 16


@dataclass
class ExtractionConfig:
    """Latent extraction parameters."""
    model: str = 'dinov3_vitb16'
    roi_ids: List[int] = field(default_factory=lambda: [1])
    batch_size: int = 32
    bin_size: int = 1
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    # A-06: Multi-scale pooling
    pooling_method: str = 'weighted_average'  # 'weighted_average' or 'multiscale'
    pooling_scales: List[int] = field(default_factory=lambda: [1, 2, 4])
    # A-06: Multi-layer extraction
    feature_layers: Optional[List[int]] = None  # None = last layer only; e.g. [3, 7, 11]


@dataclass
class ProjectConfig:
    """Complete project processing configuration.

    This holds all tunable parameters for the CASTLE pipeline:
    tracking → extraction → clustering.

    NOT to be confused with the per-project config.json which tracks
    file inventory (source videos, latent file paths, etc.).
    """
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    clustering: ClusterConfig = field(default_factory=ClusterConfig)

    # ---- Serialization ----

    def to_dict(self) -> dict:
        """Convert to a plain dict (JSON-serializable)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'ProjectConfig':
        """Reconstruct from a plain dict (inverse of to_dict).

        Handles nested dataclass reconstruction gracefully —
        unknown keys are silently ignored so old configs still load.
        """
        def _build(dc_cls, data):
            if not isinstance(data, dict):
                return data
            # Inspect the dataclass fields and recursively build nested ones
            import dataclasses
            kwargs = {}
            for f in dataclasses.fields(dc_cls):
                if f.name not in data:
                    continue  # use default
                val = data[f.name]
                # Resolve the actual type for nested dataclasses
                ftype = f.type
                # Handle string annotations — resolve via a safe lookup table
                # to avoid arbitrary code execution from malicious config files.
                if isinstance(ftype, str):
                    _SAFE_TYPES = {
                        'int': int, 'float': float, 'str': str,
                        'bool': bool, 'list': list,
                    }
                    ftype = _SAFE_TYPES.get(ftype, str)
                if dataclasses.is_dataclass(ftype):
                    kwargs[f.name] = _build(ftype, val)
                elif hasattr(ftype, '__origin__'):
                    # Handle List[...] generics
                    origin = getattr(ftype, '__origin__', None)
                    args = getattr(ftype, '__args__', ())
                    if origin is list and args and dataclasses.is_dataclass(args[0]):
                        kwargs[f.name] = [_build(args[0], item) for item in val]
                    else:
                        kwargs[f.name] = val
                else:
                    kwargs[f.name] = val
            return dc_cls(**kwargs)

        return _build(cls, d)

    # ---- File I/O ----

    def save(self, path: str) -> None:
        """Save configuration to a JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved project config to {path}")

    @classmethod
    def load(cls, path: str) -> 'ProjectConfig':
        """Load configuration from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        logger.info(f"Loaded project config from {path}")
        return cls.from_dict(data)

    # ---- Convenience ----

    def to_preprocess(self):
        """Convert extraction.preprocess to a castle.core.data.Preprocess object.

        This bridges the new config system with the existing Preprocess class
        used by the extraction pipeline.
        """
        from castle.core.data import Preprocess
        p = self.extraction.preprocess
        return Preprocess(
            center_roi_switch=p.center_roi,
            center_roi_id=p.center_roi_id,
            center_roi_crop_width=p.crop_width,
            center_roi_crop_height=p.crop_height,
            rotate_roi_tail_switch=p.rotate_roi_tail,
            rotate_roi_tail_id=p.rotate_roi_tail_id,
            remove_background_switch=p.remove_background,
        )
