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

---

## Phase 3 Simplification Modules 🔷

### ProjectData + VideoInfo

`ProjectData` consolidates all project path computation in one place, eliminating scattered `os.path.join(storage_path, project_name, …)` calls throughout the codebase.

::: castle.core.project_data
    options:
      show_root_heading: true
      show_source: true
      members:
        - VideoInfo
        - ProjectData
        - ProjectData.from_path
        - ProjectData.from_storage
        - ProjectData.sources_dir
        - ProjectData.track_dir
        - ProjectData.latent_dir
        - ProjectData.cluster_dir
        - ProjectData.preprocessed_dir
        - ProjectData.config_path
        - ProjectData.video_track_dir
        - ProjectData.mask_h5_path
        - ProjectData.latent_model_dir
        - ProjectData.cluster_session_dir
        - ProjectData.list_videos
        - ProjectData.load_config
        - ProjectData.ensure_dirs

Quick reference:

```python
from castle.core.project_data import ProjectData, VideoInfo

# Load from existing project directory (must contain config.json)
pd = ProjectData.from_path("/data/projects/my_project")

# Legacy (storage_path, project_name) API is still supported
pd = ProjectData.from_storage("/data/projects", "my_project")

# Standard path helpers — no more manual os.path.join
pd.sources_dir           # Path("/data/projects/my_project/sources")
pd.mask_h5_path("v.mp4") # Path(".../track/v.mp4/mask_list.h5")
pd.latent_model_dir("dinov2_vitb14")
pd.cluster_session_dir("session_001")

# List source videos with metadata
videos: list[VideoInfo] = pd.list_videos()
for v in videos:
    print(v.name, v.fps, v.width, v.height, v.n_frames)

# Ensure standard directory tree
pd.ensure_dirs()

# Read project config
config: dict = pd.load_config()
```

---

### ClusterData

`ClusterData` consolidates `cluster_*.npz`, `time_series_*.csv`, `id.csv`, and `annotations.csv` into a single typed container.

::: castle.core.cluster_data
    options:
      show_root_heading: true
      show_source: true
      members:
        - ClusterData
        - ClusterData.load
        - ClusterData.from_arrays
        - ClusterData.save
        - ClusterData.get_cluster_frames
        - ClusterData.n_clusters

Quick reference:

```python
from castle.core.cluster_data import ClusterData
import numpy as np

# Load from project cluster directory
cd = ClusterData.load("/data/projects/my_project/cluster")

# Load with per-session annotations
cd = ClusterData.load(
    "/data/projects/my_project/cluster",
    session_id="session_001",
)

# Query
print(cd.n_clusters())                   # → int (excludes label -1)
frames = cd.get_cluster_frames(2)        # → np.ndarray of frame indices
print(cd.names[0], cd.colors[0])         # → "grooming", (255, 0, 0)

# Construct from freshly computed arrays
embeddings = np.random.randn(1000, 128)
cluster_ids = np.random.randint(0, 5, 1000)
cd2 = ClusterData.from_arrays(embeddings, cluster_ids)

# Persist to disk
cd2.save("/data/projects/my_project/cluster")
cd2.save("/data/projects/my_project/cluster", session_id="session_002")
```

| Method | Description |
|--------|-------------|
| `ClusterData.load(cluster_dir, session_id=None)` | Load from project cluster directory |
| `ClusterData.from_arrays(embeddings, cluster_ids, hierarchy=None)` | Create from raw arrays |
| `.save(cluster_dir, session_id=None)` | Write `id.csv`, `cluster_data.npz`, `annotations.csv` |
| `.get_cluster_frames(cluster_id)` | Frame indices for a given cluster |
| `.n_clusters()` | Count of distinct non-negative cluster IDs |

---

### DeviceFactory

`DeviceFactory` centralises device detection and provides ML algorithm factory methods that automatically choose GPU-accelerated or CPU implementations.

::: castle.core.device_factory
    options:
      show_root_heading: true
      show_source: true
      members:
        - DeviceFactory
        - DeviceFactory.get_device
        - DeviceFactory.set_device
        - DeviceFactory.reset
        - DeviceFactory.get_torch_device
        - DeviceFactory.to_tensor
        - DeviceFactory.get_umap
        - DeviceFactory.get_dbscan
        - DeviceFactory.get_hdbscan

Quick reference:

```python
from castle.core.device_factory import DeviceFactory

# Device detection — cached on first call (CUDA > MPS > CPU)
device = DeviceFactory.get_device()     # → 'cuda' | 'mps' | 'cpu'
t_device = DeviceFactory.get_torch_device()  # → torch.device(...)

# Override (useful in tests or when user picks a device)
DeviceFactory.set_device("cpu")
DeviceFactory.reset()   # clear cache, re-detect on next call

# Algorithm factories — GPU (cuml) on CUDA, sklearn/umap-learn elsewhere
umap   = DeviceFactory.get_umap(n_neighbors=300, min_dist=0.0, n_components=2)
dbscan = DeviceFactory.get_dbscan(eps=0.5, min_samples=5)
hdbscan = DeviceFactory.get_hdbscan(min_cluster_size=10)

# NumPy → Tensor on current device
tensor = DeviceFactory.to_tensor(my_array)                 # float32
tensor = DeviceFactory.to_tensor(my_array, dtype=torch.float16)
```

| Method | GPU (CUDA) | CPU / MPS |
|--------|-----------|-----------|
| `get_umap(**kw)` | `cuml.manifold.UMAP` | `umap.UMAP` |
| `get_dbscan(**kw)` | `cuml.cluster.DBSCAN` | `sklearn.cluster.DBSCAN` |
| `get_hdbscan(**kw)` | `cuml.cluster.HDBSCAN` | `sklearn.cluster.HDBSCAN` or `hdbscan` pkg |

---

### SimpleVideoReader

`SimpleVideoReader` provides a clean, dependency-minimal PyAV-based video reader for the common case: open a file, read metadata, fetch frames.

::: castle.utils.video_reader_simple
    options:
      show_root_heading: true
      show_source: true
      members:
        - SimpleVideoReader
        - SimpleVideoReader.get_frame
        - SimpleVideoReader.iter_frames
        - SimpleVideoReader.close

Quick reference:

```python
from castle.utils.video_reader_simple import SimpleVideoReader

with SimpleVideoReader("video.mp4") as r:
    print(r.fps, r.width, r.height, len(r))  # metadata

    # Random access
    frame = r.get_frame(42)          # (H, W, 3) BGR uint8

    # Sequential iteration (no per-frame seek — most efficient)
    for idx, frame in r.iter_frames():
        process(frame)

    # Range + step
    for idx, frame in r.iter_frames(start=100, end=500, step=5):
        process(frame)
```

**Constructor raises:**

* `FileNotFoundError` — if *path* does not exist
* `RuntimeError` — if no video stream is found in the container

**`get_frame(index)` raises:**

* `IndexError` — if *index* is out of `[0, n_frames)`
* `RuntimeError` — if the frame cannot be decoded

**`iter_frames(start, end, step)` notes:**

* `step=1` uses fully sequential decoding (no seek per frame)
* `step>1` seeks to each frame individually via `get_frame()`
