# API Reference

Auto-generated documentation from Python docstrings using [mkdocstrings](https://mkdocstrings.github.io/).

!!! note
    Some modules may have limited docstring coverage. This reference will improve over time.

---

## Service Layer

### Project Service

::: castle.service.project_service
    options:
      show_root_heading: true
      show_source: true

---

### Extraction Service

::: castle.service.extraction_service
    options:
      show_root_heading: true
      show_source: true

---

### Clustering Service

::: castle.service.clustering_service
    options:
      show_root_heading: true
      show_source: true

---

### Tracking Service

::: castle.service.tracking_service
    options:
      show_root_heading: true
      show_source: true

---

### Preprocessing Service ★

::: castle.service.preprocessing_service
    options:
      show_root_heading: true
      show_source: true

---

### Annotation Service

::: castle.service.annotation_service
    options:
      show_root_heading: true
      show_source: true

---

### Bout Service

::: castle.service.bout_service
    options:
      show_root_heading: true
      show_source: true

---

### History Service (Undo/Redo)

::: castle.service.history_service
    options:
      show_root_heading: true
      show_source: true

---

### Ethogram Service

::: castle.service.ethogram_service
    options:
      show_root_heading: true
      show_source: true

---

### Metrics Service

::: castle.service.metrics_service
    options:
      show_root_heading: true
      show_source: true

---

### Comparison Service

::: castle.service.comparison_service
    options:
      show_root_heading: true
      show_source: true

---

### Annotator Loader

::: castle.service.annotator_loader
    options:
      show_root_heading: true
      show_source: true

---

### Session Manager

::: castle.service.session_manager
    options:
      show_root_heading: true
      show_source: true

---

## CLI

::: castle.cli.main
    options:
      show_root_heading: true
      show_source: true

---

## Core

### Project Configuration

::: castle.core.project_config
    options:
      show_root_heading: true
      show_source: true

---

### Environment

::: castle.core.environment
    options:
      show_root_heading: true
      show_source: true

---

### Mask Filter

::: castle.core.mask_filter
    options:
      show_root_heading: true
      show_source: true

---

### Stabilized Camera (Phase 0) ★

The `castle.core.stabilized_camera` module implements the **Phase 0 preprocessing** pipeline.
It applies a zero-phase Butterworth low-pass filter to ROI centroid trajectories and orientation
angles, then extracts dynamically-cropped and head-aligned frames resized to 518×518 for DINOv2.

::: castle.core.stabilized_camera
    options:
      show_root_heading: true
      show_source: true
      members:
        - StabilizedCamera
        - extract_centroids_from_masks
        - extract_orientations_from_masks
        - preview_stabilization

---

### Ethogram Engine

::: castle.core.ethogram
    options:
      show_root_heading: true
      show_source: true

---

### Quality Metrics

::: castle.core.metrics
    options:
      show_root_heading: true
      show_source: true

---

### Group Comparison

::: castle.core.comparison
    options:
      show_root_heading: true
      show_source: true

---

### Extractor

<!-- mkdocstrings fails on castle.core.extractor due to import dependencies (torch, cuml, etc.) -->
<!-- ::: castle.core.extractor -->

!!! note "Build Note"
    Auto-documentation for `castle.core.extractor` is unavailable due to import dependencies that are not installed in the docs build environment. See source code directly: [`castle/core/extractor.py`](https://github.com/CASTLE-ai/castle-ai/blob/main/castle/core/extractor.py)

---

### Data

<!-- mkdocstrings fails on castle.core.data due to import dependencies -->
<!-- ::: castle.core.data -->

!!! note "Build Note"
    Auto-documentation for `castle.core.data` is unavailable due to import dependencies. See source: [`castle/core/data.py`](https://github.com/CASTLE-ai/castle-ai/blob/main/castle/core/data.py)

---

### Models

<!-- mkdocstrings fails on castle.core.models due to import dependencies -->
<!-- ::: castle.core.models -->

!!! note "Build Note"
    Auto-documentation for `castle.core.models` is unavailable due to import dependencies. See source: [`castle/core/models.py`](https://github.com/CASTLE-ai/castle-ai/blob/main/castle/core/models.py)

---

## Utils

### Project Management

::: castle.utils.project_manager
    options:
      show_root_heading: true
      show_source: true

---

### Video I/O

::: castle.utils.video_io
    options:
      show_root_heading: true
      show_source: true

---

### Video Management

::: castle.utils.video_manager
    options:
      show_root_heading: true
      show_source: true

---

### Image Segmentation (SAM)

::: castle.utils.image_segment
    options:
      show_root_heading: true
      show_source: true

---

### ROI Tracking

::: castle.utils.tracking_manager
    options:
      show_root_heading: true
      show_source: true

---

### Feature Extraction

::: castle.utils.visual_latent_extract
    options:
      show_root_heading: true
      show_source: true

---

### Latent Explorer

::: castle.utils.latent_explorer
    options:
      show_root_heading: true
      show_source: true

---

### Video Alignment

::: castle.utils.video_align
    options:
      show_root_heading: true
      show_source: true

---

### HDF5 I/O

::: castle.utils.h5_io
    options:
      show_root_heading: true
      show_source: true

---

## Visualization

### Embedding Plots

::: castle.visualization.embedding_plots
    options:
      show_root_heading: true
      show_source: true

---

### Ethogram Plots

::: castle.visualization.ethogram_plots
    options:
      show_root_heading: true
      show_source: true

---

### Comparison Plots

::: castle.visualization.comparison_plots
    options:
      show_root_heading: true
      show_source: true

---

## Phase 2 Performance Modules ⚡

### Model Registry

`ModelRegistry` is a **thread-safe singleton** that manages the full lifecycle of SAM, DeAOT, and DINOv2/DINOv3 models — lazy loading, explicit unloading, and CUDA memory accounting.

::: castle.core.model_registry
    options:
      show_root_heading: true
      show_source: true
      members:
        - ModelRegistry
        - ModelRegistry.instance
        - ModelRegistry.load
        - ModelRegistry.unload
        - ModelRegistry.unload_all
        - ModelRegistry.unload_family
        - ModelRegistry.use
        - ModelRegistry.get_memory_stats

Quick reference:

```python
from castle.core.model_registry import ModelRegistry

registry = ModelRegistry.instance()

# Explicit load / unload
model = registry.load("dinov2_vitb14")
registry.unload("dinov2_vitb14")

# Context manager (auto-unloads on exit)
with registry.use("dinov2_vitb14") as model:
    latent = model.extract_tensor_batch(frames, masks, roi_id)

# Bulk unload by family keyword
registry.unload_family("sam", "deaot", "aot")
registry.unload_all()

# VRAM diagnostics
stats = registry.get_memory_stats()
# → {"device": "cuda:0", "allocated_mb": 340, "reserved_mb": 512,
#    "free_mb": 8100, "total_mb": 12288, "loaded_models": [...]}
```

---

### Auto Batch Size

VRAM-aware batch size recommendation and automatic OOM retry.

::: castle.core.auto_batch
    options:
      show_root_heading: true
      show_source: true
      members:
        - compute_optimal_batch_size
        - auto_retry_on_oom

Quick reference:

```python
from castle.core.auto_batch import compute_optimal_batch_size, auto_retry_on_oom

# Query free VRAM and return a safe batch size
batch = compute_optimal_batch_size(
    model_name="dinov2_vitb14",
    frame_size=(518, 518, 3),   # (H, W, C)
    device="auto",              # auto-detect cuda/cpu
)

# Wrap any callable; halves batch_size on OOM and retries
result = auto_retry_on_oom(
    extract_fn,
    frames,
    initial_batch=batch,       # override starting batch
    batch_kwarg="batch_size",  # kwarg name passed to extract_fn
    min_batch=1,               # give up below this value
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | — | Model name; used to look up weight reservation and input resolution |
| `frame_size` | — | `(H, W)` or `(H, W, C)` of source frames |
| `device` | `"auto"` | Target device string or `"auto"` |
| `dtype_bytes` | `4` | Bytes per element (4 = float32, 2 = float16) |

Falls back to batch size **4** on CPU or when VRAM info is unavailable.

---

### Pipeline Orchestrator

Full tracking → extraction orchestrator with per-stage GPU memory cleanup.

::: castle.core.pipeline
    options:
      show_root_heading: true
      show_source: true
      members:
        - PipelineConfig
        - Pipeline
        - Pipeline.run
        - Pipeline.run_tracking_stage
        - Pipeline.run_extraction_stage

Quick reference:

```python
from castle.core.pipeline import Pipeline, PipelineConfig

config = PipelineConfig(
    storage_path="/data/storage",
    project_name="my_project",
    tracking_model="r50_deaotl",
    extraction_model="dinov2_vitb14",
    batch_size=16,
)

pipeline = Pipeline(config, progress_callback=lambda f, msg: print(f, msg))
results = pipeline.run()
# results["tracking"]     → {video_name: status}
# results["extraction"]   → {video_name: latent_path}
# results["memory_stats"] → ModelRegistry.get_memory_stats()
```

**Stage sequence:**

1. `run_tracking_stage()` — SAM + DeAOT tracking for every video
2. `_cleanup_tracking()` — unload SAM/DeAOT sentinels, flush CUDA cache
3. `run_extraction_stage()` — DINOv2/DINOv3 extraction for every video
4. `_cleanup_extraction()` — unload visual encoder, flush CUDA cache

---

### Parallel Extractor

Three-stage threaded pipeline for high-throughput feature extraction.

::: castle.core.pipeline_parallel
    options:
      show_root_heading: true
      show_source: true
      members:
        - ParallelExtractor
        - ParallelExtractor.run

Quick reference:

```python
from castle.core.pipeline_parallel import ParallelExtractor

extractor = ParallelExtractor(
    video_path="/data/animal.mp4",
    stabilized_camera=camera,  # optional StabilizedCamera instance
    model=visual_encoder,      # must implement extract_tensor_batch or extract_batch_latent
    batch_size=8,
    queue_size=32,
    roi_id=1,
)

latents = extractor.run(
    progress_callback=lambda cur, total, stage: print(f"{stage}: {cur}/{total}")
)
# → np.ndarray shape (N, D), dtype float32
```

**Pipeline stages:**

| Stage | Thread | Work |
|-------|--------|------|
| 1 — I/O | Background | `VideoReader.get_frame()` → `frame_queue` |
| 2 — Preprocess | Background | `StabilizedCamera.generate_frame()` → `tensor_queue` |
| 3 — Inference | Main (GPU) | Batched `model.extract_tensor_batch()` → latent list |

Uses `threading` (not `multiprocessing`) to avoid CUDA fork issues. Bounded queues (`maxsize=32`) prevent unbounded memory growth.

---

### Pipeline Cache

Content-addressed cache for pipeline outputs.

::: castle.core.cache
    options:
      show_root_heading: true
      show_source: true
      members:
        - PipelineCache
        - PipelineCache.compute_key
        - PipelineCache.is_cached
        - PipelineCache.get
        - PipelineCache.put
        - PipelineCache.invalidate
        - PipelineCache.clear

Quick reference:

```python
from castle.core.cache import PipelineCache

cache = PipelineCache("/data/project/latent")

key = cache.compute_key(
    video_path="/data/project/sources/vid.mp4",
    config={"center_roi": True, "model": "dinov2_vitb14"},
    model_name="dinov2_vitb14",
)

if cache.is_cached(key):          # also validates file existence
    path = cache.get(key)
else:
    path = run_extraction(...)
    cache.put(key, path)

cache.invalidate(key)             # remove one entry
cache.clear()                     # remove all entries
len(cache)                        # number of manifest entries
```

Cache key = SHA-256(`abs_path` + `mtime` + sorted config JSON + `model_name`).  
Manifest written atomically to `{cache_dir}/.cache_manifest.json`.  
Stale entries (file deleted externally) are silently purged on `is_cached()`.

---

### Incremental Service

Helpers for detecting unprocessed videos and cleaning up orphaned data.

::: castle.service.incremental_service
    options:
      show_root_heading: true
      show_source: true
      members:
        - get_unprocessed_videos
        - cleanup_deleted_videos

Quick reference:

```python
from castle.service.incremental_service import (
    get_unprocessed_videos,
    cleanup_deleted_videos,
)

# Before a batch run — returns only videos without any latent output yet
pending = get_unprocessed_videos("/data/projects/my_project")
# → ['animal_03.mp4', 'animal_07.mp4']

# After removing source videos — cleans latent, cluster, and cache data
removed = cleanup_deleted_videos("/data/projects/my_project")
# → ['deleted_animal.mp4']
```

`cleanup_deleted_videos` also:
- Removes empty latent sub-directories
- Removes cluster files whose stem matches a deleted video
- Invalidates `PipelineCache` manifest entries for deleted videos

---

## New Modules (v5.1)

### Temporal Smoothing

::: castle.core.temporal_smooth
    options:
      show_root_heading: true
      show_source: true

---

### Cluster Transfer

::: castle.core.cluster_transfer
    options:
      show_root_heading: true
      show_source: true

---

### Automated Behavior Microscope

::: castle.core.auto_cluster
    options:
      show_root_heading: true
      show_source: true

---

### NWB Export

::: castle.core.nwb_export
    options:
      show_root_heading: true
      show_source: true

---

### NWB Service

::: castle.service.nwb_service
    options:
      show_root_heading: true
      show_source: true

---

## CLI Commands

### Preprocess CLI ★

::: castle.cli.preprocess_cmd
    options:
      show_root_heading: true
      show_source: true

---

### Ethogram CLI

::: castle.cli.ethogram_cmd
    options:
      show_root_heading: true
      show_source: true

---

### Compare CLI

::: castle.cli.compare_cmd
    options:
      show_root_heading: true
      show_source: true

---

### Cluster CLI

::: castle.cli.cluster_cmd
    options:
      show_root_heading: true
      show_source: true

---

## Legacy Utils

### Plot

::: castle.utils.plot
    options:
      show_root_heading: true
      show_source: true
