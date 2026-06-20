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
angles, then extracts dynamically-cropped and head-aligned frames resized to match the encoder patch grid (592×592 for the default DINOv3 ViT-B/16).

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
model = registry.load("dinov3_vitb16")
registry.unload("dinov3_vitb16")

# Context manager (auto-unloads on exit)
with registry.use("dinov3_vitb16") as model:
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

Automatic OOM retry with halved batch size.

::: castle.core.auto_batch
    options:
      show_root_heading: true
      show_source: true
      members:
        - auto_retry_on_oom

Quick reference:

```python
from castle.core.auto_batch import auto_retry_on_oom

# Wrap any callable; halves batch_size on OOM and retries
result = auto_retry_on_oom(
    extract_fn,
    frames,
    initial_batch=batch,       # starting batch size
    batch_kwarg="batch_size",  # kwarg name passed to extract_fn
    min_batch=1,               # give up below this value
)
```

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
    extraction_model="dinov3_vitb16",
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
    config={"center_roi": True, "model": "dinov3_vitb16"},
    model_name="dinov3_vitb16",
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
pd.latent_model_dir("dinov3_vitb16")
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

### Environment / Device Detection

`get_device()` is the single canonical device detector (MPS > CUDA > CPU), computed once in the module-level `env` singleton. Algorithm-class dispatch lives in `castle.core.clustering_backends`.

::: castle.core.environment
    options:
      show_root_heading: true
      show_source: true
      members:
        - get_device
        - get_num_workers

Quick reference:

```python
from castle.core.environment import get_device

# Canonical device detection (MPS > CUDA > CPU)
device = get_device()     # → 'cuda' | 'mps' | 'cpu'
```

---

## Phase 4 Feature Modules 🟢

### Multi-Subject Tracking

`SubjectTrack` and `MultiSubjectProject` provide first-class support for videos containing
multiple independently tracked animals.

::: castle.core.multi_subject
    options:
      show_root_heading: true
      show_source: true
      members:
        - SubjectTrack
        - SubjectTrack.n_frames
        - SubjectTrack.set_latents
        - SubjectTrack.set_labels
        - MultiSubjectProject
        - MultiSubjectProject.add_subject
        - MultiSubjectProject.process_all
        - MultiSubjectProject.get_subjects
        - MultiSubjectProject.get_subject

Quick reference:

```python
from castle.core.multi_subject import MultiSubjectProject

project = MultiSubjectProject("/data/projects/social_session", "video01.mp4")
project.add_subject(subject_id=0, body_roi=1, head_roi=2)
project.add_subject(subject_id=1, body_roi=3, head_roi=4)

# Extracts positions + angles for all subjects from the shared mask HDF5
project.process_all(
    n_frames=None,            # auto-inferred from video metadata
    progress_callback=lambda cur, total: print(f"{cur}/{total}"),
)

tracks = project.get_subjects()   # list[SubjectTrack], sorted by subject_id
track0 = project.get_subject(0)   # single SubjectTrack

print(track0.n_frames)        # → int
print(track0.positions.shape) # → (N, 2)
print(track0.angles.shape)    # → (N,)

# Assign latents + labels after your extraction / clustering step
track0.set_latents(latent_array)   # np.ndarray (N, D)
track0.set_labels(cluster_labels)  # np.ndarray (N,)
```

---

### Social Feature Extraction

::: castle.analysis.social_features
    options:
      show_root_heading: true
      show_source: true
      members:
        - compute_pairwise_distance
        - compute_relative_orientation
        - compute_approach_score
        - detect_social_events

Quick reference:

```python
from castle.analysis.social_features import (
    compute_pairwise_distance,
    compute_relative_orientation,
    compute_approach_score,
    detect_social_events,
)

# All functions take a list[SubjectTrack] — must be synchronised (same n_frames)

dist = compute_pairwise_distance(tracks)
# → np.ndarray shape (N, n_subjects, n_subjects), symmetric, diagonal=0

orient = compute_relative_orientation(tracks)
# → np.ndarray (N, S, S), degrees in (-180, 180]
# orient[t, i, j] = 0° means subject i faces directly toward j at frame t

approach = compute_approach_score(tracks, window=30)
# → np.ndarray (N, S, S); positive = approaching, negative = receding

events = detect_social_events(
    tracks,
    distance_threshold=50.0,  # px
    duration_threshold=15,    # frames
)
# → list of dicts: {type, subjects, start_frame, end_frame, duration}
```

| Function | Output shape | Notes |
|----------|-------------|-------|
| `compute_pairwise_distance` | `(N, S, S)` | Euclidean pixel distance; symmetric |
| `compute_relative_orientation` | `(N, S, S)` | Heading-relative angle in degrees |
| `compute_approach_score(window=W)` | `(N, S, S)` | −mean(Δdist) over W frames; symmetric |
| `detect_social_events(dist_thr, dur_thr)` | `list[dict]` | Sorted by `start_frame` |

---

### Group Ethogram

::: castle.analysis.group_ethogram
    options:
      show_root_heading: true
      show_source: true
      members:
        - build_group_ethogram
        - plot_group_ethogram

Quick reference:

```python
from castle.analysis.group_ethogram import build_group_ethogram, plot_group_ethogram

# Build — requires track.labels to be set on every SubjectTrack
ethogram = build_group_ethogram(
    tracks,
    fps=30.0,
    cluster_names={0: "rest", 1: "groom", 2: "explore"},
    distance_threshold=50.0,
    duration_threshold=15,
)

# ethogram dict keys:
#   fps, n_frames, n_subjects, subject_ids,
#   per_subject: {sid: {ethogram, labels, cluster_names}},
#   social_events: list[dict],
#   time_axis: np.ndarray (N,) seconds

# Visualise
path = plot_group_ethogram(
    ethogram,
    output_path="/tmp/group_ethogram.png",
    figsize=None,                # auto: (14, n_subjects*1.5 + 1.5)
    bar_height=0.8,
    social_event_color="#CC0000",
    dpi=150,
)
print(path)   # absolute path to saved PNG
```

The figure contains one row per subject (colour-coded behaviour raster) plus a bottom row showing social interaction spans.

---

### Batch Config & Runner

::: castle.core.batch
    options:
      show_root_heading: true
      show_source: true
      members:
        - BatchConfig
        - BatchConfig.from_yaml
        - BatchRunner
        - BatchRunner.run
        - BatchRunner.generate_summary

Quick reference:

```python
from castle.core.batch import BatchConfig, BatchRunner

# Load from YAML
config = BatchConfig.from_yaml("experiments.yaml")

# Or construct programmatically
config = BatchConfig(
    projects=[
        {"name": "ctrl", "project": "/data/ctrl", "videos": ["v1.mp4"], "params": {}},
        {"name": "treat", "project": "/data/treat", "videos": [], "params": {"fc": 0.1}},
    ],
    parallel=True,
    max_workers=2,
)

runner  = BatchRunner(config)
results = runner.run(progress_callback=lambda frac, msg: print(f"{frac:.0%} {msg}"))

# results: list[dict] with keys name, project, status, tracking, extraction, elapsed_s, error
summary = runner.generate_summary(results)
print(summary)
```

`BatchRunner.run()` delegates each project to `Pipeline` (from `castle.core.pipeline`).  
With `parallel=True`, projects run concurrently in a `ThreadPoolExecutor`.

**YAML format:**

```yaml
experiments:
  - name: "Control Group"
    project: "/data/control"
    videos: ["mouse1.mp4", "mouse2.mp4"]   # empty = process all source videos
    params:
      fc: 0.25
      n_clusters: 10

parallel: false      # set true to run projects concurrently
max_workers: 2
```

---

### Batch CLI

::: castle.cli.batch_cmd
    options:
      show_root_heading: true
      show_source: true
      members:
        - batch_run
        - batch_status
        - batch_report

Quick reference:

```bash
# Run all experiments defined in a YAML file
castle batch run experiments.yaml

# Run with parallelism
castle batch run experiments.yaml --parallel --max-workers 4

# Check status of the last batch run (reads .batch_result.json)
castle batch status experiments.yaml

# Generate HTML reports for each project
castle batch report experiments.yaml --output reports/

# Generate a single combined HTML report (first project)
castle batch report experiments.yaml --output summary.html \
    --include-ethogram --include-quality
```

| Sub-command | Description |
|-------------|-------------|
| `batch run` | Execute the full pipeline for all experiments |
| `batch status` | Display status of the most recent run |
| `batch report` | Generate HTML reports via `ReportGenerator` |

---

### HTML Report Generator

::: castle.analysis.report
    options:
      show_root_heading: true
      show_source: true
      members:
        - ReportGenerator
        - ReportGenerator.generate

Quick reference:

```python
from castle.analysis.report import ReportGenerator

gen = ReportGenerator(
    project_path="/storage/my_project",
    session_id="exp01",          # optional — shown in report header
)

# Generate full report
path = gen.generate(
    output_path="report.html",   # None → auto path inside project/reports/
    include_ethogram=True,       # ethogram plot + bout stats + transition matrix
    include_quality=True,        # silhouette, CH, DB, inertia + embedding scatter
    include_comparison=False,    # cross-project section (placeholder for single project)
)
print(f"Report saved to {path}")
```

The report is a **self-contained HTML file** — no external dependencies. Sections:

| Section | Content | Requires |
|---------|---------|---------|
| Header | Metadata cards (project, session, frames, clusters, models) | Always |
| Ethogram | Frequency bar chart + bout stats table + transition matrix | Cluster data |
| Quality Metrics | Silhouette / CH / DB / inertia badges + 2-D embedding scatter | Cluster data |
| Group Comparison | Placeholder with link to `BatchRunner.generate_summary()` | `include_comparison=True` |
| Footer | Generation timestamp | Always |
