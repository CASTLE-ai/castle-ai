# Architecture

## System Overview

```
┌────────────────────────────────────────────────────────────────┐
│                        Frontend Layer                          │
│                                                                │
│  castle/cli/                      castle/ui/                   │
│  (typer CLI)                      (Gradio Web UI)              │
└────────────┬───────────────────────────┬──────────────────────┘
             │                           │
             ▼                           ▼
┌────────────────────────────────────────────────────────────────┐
│                      castle/service/                           │
│                      (Service Layer)                           │
│                                                                │
│  project_service.py      — Project CRUD via service layer      │
│  extraction_service.py   — Feature extraction orchestration    │
│  clustering_service.py   — UMAP + DBSCAN management           │
│  tracking_service.py     — Tracking pipeline orchestration     │
│  preprocessing_service.py — Stabilized camera preprocessing ★  │
│  incremental_service.py  — Incremental update helpers ⚡        │
│  annotation_service.py   — Classification scheme management    │
│  bout_service.py         — Behavioral bout analysis            │
│  history_service.py      — Undo/Redo (Command Pattern)         │
│  ethogram_service.py     — Ethogram analysis orchestration     │
│  metrics_service.py      — Clustering quality evaluation       │
│  comparison_service.py   — Group comparison orchestration      │
└──────────────────────────┬─────────────────────────────────────┘
                           │ calls
┌──────────────────────────▼─────────────────────────────────────┐
│                      castle/core/                              │
│                   (Core Business Logic)                        │
│                                                                │
│  extractor.py          — Feature extraction engine             │
│  cluster.py            — LatentAggregator, clustering logic    │
│  data.py               — Preprocess, VideoDataset              │
│  models.py             — VisualEncoder abstraction (DINOv2/v3) │
│  config.py             — Constants, model paths                │
│  project.py            — Project config I/O (file inventory)   │
│  project_config.py     — ProjectConfig dataclass (B-05)        │
│  environment.py        — Device detection, worker count        │
│  mask_filter.py        — Post-tracking mask filtering (A-03)   │
│  stabilized_camera.py  — StabilizedCamera + helpers (P0) ★    │
│  logging_config.py     — Centralized logging setup             │
│  ethogram.py           — Ethogram analysis engine (P1)         │
│  metrics.py            — Clustering quality metrics (P2)       │
│  comparison.py         — Group comparison engine (P4)          │
│  ── Phase 2 Performance Modules ────────────────────────────── │
│  model_registry.py     — ModelRegistry singleton (lazy load,  │
│                          VRAM cleanup between stages) ⚡        │
│  auto_batch.py         — VRAM-aware batch size + OOM retry ⚡  │
│  pipeline.py           — Full pipeline orchestrator with GPU  │
│                          cleanup between stages ⚡             │
│  cache.py              — Content-hash PipelineCache ⚡         │
│  ── Phase 3 Simplification Modules 🔷 ──────────────────────── │
│  project_data.py       — ProjectData + VideoInfo dataclasses   │
│                          (unified path computation)            │
│  cluster_data.py       — ClusterData dataclass (unified        │
│                          cluster artefact container)           │
│  device_factory.py     — DeviceFactory (centralized device     │
│                          management, UMAP/DBSCAN factories)    │
└──────────────────────────┬─────────────────────────────────────┘
                           │ uses
┌──────────────────────────▼─────────────────────────────────────┐
│                      castle/utils/                             │
│                    (Utility Layer)                              │
│                                                                │
│  project_manager.py       — Project CRUD operations            │
│  video_manager.py         — Video import/scan                  │
│  video_io.py              — Video read/write (PyAV)            │
│  video_align.py           — Center, rotate, crop frames        │
│  image_segment.py         — SAM wrapper (Segmentor)            │
│  video_object_segment.py  — DeAOT wrapper                      │
│  tracking_manager.py      — ROI tracking orchestration         │
│  latent_explorer.py       — Latent/LocalLatent classes         │
│  myumap.py                — Custom UMAP (cuml + spectral)      │
│  h5_io.py                 — HDF5 mask storage                  │
│  analysis_utils.py        — Kinematic DataFrame utilities      │
│  roi_manager.py           — ROI utilities                      │
│  download.py              — Checkpoint download (gdown)        │
│  profiler.py              — GPU/CPU performance profiling      │
│  video_reader_simple.py   — SimpleVideoReader (Phase 3 🔷)     │
└──────────────────────────┬─────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────────┐
│                castle/visualization/                           │
│              (Visualization Layer — B-01)                       │
│                                                                │
│  embedding_plots.py  — UMAP scatter, syllable bar,             │
│                        focus embedding, named embedding         │
│  ethogram_plots.py   — Ethogram raster, transition heatmap,    │
│                        bout distributions (P1)                  │
│  comparison_plots.py — Radar, volcano, forest plots,           │
│                        transition diff heatmaps (P4)            │
└────────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────────┐
│                 Vendored / External Models                      │
│  castle/sam/     — Segment Anything Model (Meta)               │
│  castle/aot/     — DeAOT video object segmentation             │
│  castle/dinov2/  — DINOv2 vendored components                  │
│  castle/dinov3/  — DINOv3 vendored components                  │
│  castle/configs/ — model_config.json                           │
└────────────────────────────────────────────────────────────────┘
```

## Module Map

### `castle/cli/` — Command-Line Interface

Built on [Typer](https://typer.tiangolo.com/). Provides a `castle` command for headless pipeline execution.

| Module | Purpose |
|--------|---------|
| `main.py` | Typer app entry point, registers all subcommand groups |
| `project_cmd.py` | `castle project init/info/add-videos/list` |
| `cluster_cmd.py` | `castle cluster run/export/save-model/apply-model/auto/evaluate` |
| `extract_cmd.py` | `castle extract <project>` |
| `track_cmd.py` | `castle track <project>` |
| `preprocess_cmd.py` | `castle preprocess <project> --video … --body-roi … --head-roi …` ★ |
| (in `main.py`) | `castle info <project>` — alias for `castle project info` |
| `ethogram_cmd.py` | `castle ethogram analyze/transitions/bouts/export/export-nwb` |
| `compare_cmd.py` | `castle compare run/fingerprint` |
| `batch_cmd.py` 🟢 | `castle batch run/status/report` — batch processing across multiple experiments (P4) |

### `castle/ui/` — Gradio Web Interface

Built on [Gradio](https://gradio.app/). Each tab has its own module.

8 top-level tabs: **0. Project | 1. Upload Videos | 2. Tracking ROIs | 3. Pre-process (Optional) | 4. Extract Latent | 5. Behavior Microscope | 6. Analysis | 7. Export**

| Module | Tab | Purpose |
|--------|-----|---------|
| `main_ui.py` | — | Creates the top-level app with all 8 tabs |
| `project_ui.py` | 0. Project | Create, open, delete projects |
| `source_ui.py` | 1. Upload Videos | Upload local files or scan server directories |
| `edit_ui.py` | 2. Tracking ROIs | Container for all tracking sub-UIs |
| `view_ui.py` | └─ View | Browse frames with slider |
| `label_ui.py` | └─ Label ROI | Point-and-click segmentation with SAM |
| `knowledge_ui.py` | └─ ROI Prompts | Gallery of all saved ROI labels |
| `track_ui.py` | └─ Tracking | Run DeAOT tracking with progress |
| `post_track_ui.py` | └─ Post-Track | Post-process and review tracking results |
| `batch_track_ui.py` | └─ Batch | Process multiple videos |
| `preprocess_ui.py` | 3. Pre-process (Optional) | Stabilized camera preprocessing (P0) ★ |
| `extract_ui.py` | 4. Extract Latent | Configure and run feature extraction |
| `cluster_page_ui.py` | 5. Behavior Microscope | UMAP + DBSCAN clustering workspace |
| `embedding_scatter.py` | └─ (component) | Plotly embedding scatter widget |
| `cluster_handlers.py` | └─ (component) | Cluster operation callbacks |
| `cluster_tree.py` | └─ (component) | Hierarchical cluster tree view |
| `cluster_input_ui.py` | └─ (component) | Clustering parameter input widgets |
| `annotator_ui.py` | └─ Cluster Annotator | Grid video browser, per-session labels, auto-save |
| `analysis_ui.py` | 6. Analysis | Ethogram, Quality Metrics sub-tabs, Group Comparison placeholder |
| `export_ui.py` | 7. Export | ZIP download with selectable data components |
| `plot_mask_info.py` | (component) | Mask info / contour overlay utilities |
| `HANDLER_GUIDE.md` 🔷 | (guide) | UI handler pattern guide — thin-handler / fat-service convention, anti-patterns, error handling contract |

### `castle/service/` — Service Layer

Clean separation between frontends and business logic. Both frontends (CLI, Gradio) call these services.

| Module | Purpose |
|--------|---------|
| `project_service.py` | Project CRUD (create, list, info, delete) |
| `extraction_service.py` | Feature extraction orchestration |
| `clustering_service.py` | UMAP + DBSCAN session management for the Behavior Microscope (human-in-the-loop) |
| `tracking_service.py` | Tracking pipeline orchestration |
| `preprocessing_service.py` | Stabilized camera preprocessing — `PreprocessingService` + `preprocess_stabilized_camera()` ★ |
| `annotation_service.py` | Classification scheme management |
| `annotator_loader.py` | `AnnotatorData` — loads cluster + video data for Annotator and Analysis UIs |
| `session_manager.py` | `SessionManager` — list/create/activate clustering sessions |
| `bout_service.py` | Behavioral bout analysis and export |
| `history_service.py` | Undo/Redo via Command Pattern |
| `ethogram_service.py` | Ethogram analysis: loads cluster data, delegates to `castle.core.ethogram` |
| `metrics_service.py` | Clustering quality evaluation: loads labels/embedding, delegates to `castle.core.metrics` |
| `comparison_service.py` | Group comparison: loads per-video data, delegates to `castle.core.comparison` |
| `nwb_service.py` | NWB (Neurodata Without Borders) export orchestration |
| `incremental_service.py` ⚡ | `get_unprocessed_videos()` — detect pending videos; `cleanup_deleted_videos()` — remove orphaned latent / cluster / cache data |

### `castle/core/` — Core Business Logic

| Module | Purpose |
|--------|---------|
| `extractor.py` | Feature extraction execution engine |
| `cluster.py` | `LatentAggregator` — multi-video latent loading and frame retrieval |
| `cluster_transfer.py` | Save / apply clustering models to new data |
| `data.py` | `Preprocess` pipeline, `VideoDataset` for batched extraction |
| `models.py` | `VisualEncoder` abstraction: DINOv3 (default `dinov3_vitb16`), DINOv2 (still selectable), multi-scale pooling |
| `config.py` | Constants: checkpoint paths, model IDs, supported models (default `dinov3_vitb16`) |
| `project.py` | Project config read/write (file inventory) |
| `project_config.py` | `ProjectConfig` dataclass — typed processing parameters |
| `environment.py` | Device detection (`cuda`/`mps`/`cpu`), cgroup/network-FS-aware worker count |
| `runtime_env.py` | Cross-environment detection — filesystem type, cgroup-limited CPU count, RAM, GPU/VRAM, node-local scratch dir, RAM-aware latent budget |
| `_early_env.py` | Sets `HDF5_USE_FILE_LOCKING=FALSE` at `import castle` (before h5py loads) |
| `mask_filter.py` | Post-tracking mask filtering — largest component, configurable threshold |
| `stabilized_camera.py` | `StabilizedCamera`, `extract_centroids_from_masks`, `extract_orientations_from_masks`, `preview_stabilization` — Phase 0 preprocessing (P0) ★ |
| `logging_config.py` | Centralized logging setup |
| `temporal_smooth.py` | Temporal smoothing of cluster label sequences |
| `interfaces.py` | Shared abstract interfaces / protocols |
| `ethogram.py` | Ethogram engine — bout extraction, transition matrix, temporal coherence |
| `metrics.py` | Clustering quality metrics — silhouette, CH, DB, temporal coherence, bout quality, external validation |
| `comparison.py` | Group comparison — BFA test, behavioral fingerprint, energy distance, permutation tests, Hedges' g |
| `nwb_export.py` | NWB file creation from CASTLE cluster data |
| `model_registry.py` ⚡ | `ModelRegistry` singleton — lazy load, explicit unload, and CUDA memory accounting for SAM / DeAOT / DINOv2 / DINOv3 |
| `auto_batch.py` ⚡ | `compute_optimal_batch_size()` — VRAM-aware batch size; `auto_retry_on_oom()` — automatic OOM retry with halved batch |
| `pipeline.py` ⚡ | `Pipeline` + `PipelineConfig` — full tracking → extraction orchestrator with per-stage GPU cleanup and VRAM logging |
| `cache.py` ⚡ | `PipelineCache` — SHA-256 content-addressed cache; manifest persisted as JSON; stale-entry auto-invalidation |
| `project_data.py` 🔷 | `ProjectData` + `VideoInfo` dataclasses — unified project path computation, eliminating scattered `os.path.join` calls |
| `cluster_data.py` 🔷 | `ClusterData` dataclass — consolidates `cluster_*.npz`, `time_series_*.csv`, `id.csv`, `annotations.csv` into one typed container |
| `device_factory.py` 🔷 | `DeviceFactory` — centralised device detection and algorithm factory (UMAP, DBSCAN, HDBSCAN) with GPU/CPU/MPS dispatch |
| `multi_subject.py` 🟢 | `SubjectTrack` + `MultiSubjectProject` — multi-subject tracking data containers and pipeline orchestration (P4) |
| `batch.py` 🟢 | `BatchConfig` + `BatchRunner` — YAML-driven multi-project batch processing with optional parallelism and summary reporting (P4) |

### `castle/analysis/` — Analysis Layer (Phase 4) 🟢

Higher-level analysis modules that sit above `castle/core/` and operate on multi-subject tracking data.

| Module | Purpose |
|--------|---------|
| `social_features.py` 🟢 | Pairwise distance, relative orientation, approach/avoidance score, social event detection for multi-subject recordings |
| `group_ethogram.py` 🟢 | `build_group_ethogram` + `plot_group_ethogram` — synchronized multi-subject ethogram construction and publication-quality visualization |
| `report.py` 🟢 | `ReportGenerator` — self-contained HTML report with ethogram plot, quality metrics, transition matrix, optional group comparison |

### `castle/utils/` — Utility Layer

| Module | Purpose |
|--------|---------|
| `project_manager.py` | Project CRUD (create, list, delete) |
| `video_manager.py` | Video import, directory scanning, format detection |
| `video_io.py` | Video read/write using PyAV, subtitle generation |
| `video_align.py` | Frame alignment: center, rotate, crop |
| `image_segment.py` | SAM wrapper (`Segmentor`, `MultiObjectSegmentor`) |
| `video_object_segment.py` | DeAOT wrapper (`AOTTracker`, model loading) |
| `tracking_manager.py` | `ROITracker` — orchestrates multi-frame tracking |
| `latent_explorer.py` | `Latent`, `LocalLatent` — embedding, clustering, visualization |
| `myumap.py` | GPU-accelerated UMAP using cuml + spectral layout |
| `h5_io.py` | `H5IO` — HDF5 file I/O for mask storage |
| `analysis_utils.py` | Kinematic DataFrame construction |
| `roi_manager.py` | ROI color management and utilities |
| `download.py` | Checkpoint download via gdown |
| `profiler.py` | `Profiler`, `TimeBlock`, `SystemMonitor` for performance monitoring |
| `video_reader_simple.py` 🔷 | `SimpleVideoReader` — simplified PyAV-based video reader (no LRU cache, no cv2 fallback); clean `get_frame()` / `iter_frames()` API |

### `castle/visualization/` — Visualization Layer

Separated from utils (B-01) so data classes don't depend on matplotlib/plotly:

| Module | Purpose |
|--------|---------|
| `embedding_plots.py` | UMAP scatter, syllable bar, focus embedding, named embedding |
| `ethogram_plots.py` | Ethogram raster, transition heatmap, bout duration box plots, frequency bar chart (P1) |
| `comparison_plots.py` | Fingerprint radar, transition heatmap diff, volcano plot, forest plot (P4) |

### `castle/sam/` — SAM (Vendored)

Segment Anything Model from Meta AI. Forked from [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything).

### `castle/aot/` — DeAOT (Vendored)

Decoupling features for video object segmentation. Forked from [yoxu515/aot-benchmark](https://github.com/yoxu515/aot-benchmark).

### `castle/configs/` — Configuration

Contains `model_config.json` with paths and settings for all models.

---

## Data Flow

```
Video File (.mp4)
    │
    ▼
[1. SAM] Point clicks → segmentation masks (.npz labels)
    │
    ▼
[2. DeAOT] Propagate masks → tracked masks (mask_list.h5)
    │
    ▼
[3. StabilizedCamera] ★ (optional — Phase 0)
    │  ├─ extract_centroids_from_masks → body centroid x(t)
    │  ├─ extract_orientations_from_masks → heading angle θ(t)
    │  ├─ Zero-phase Butterworth LP (fc=0.25 Hz, filtfilt) → x_c(t), θ_c(t)
    │  ├─ dynamic crop: max(300, 2×(‖x−x_c‖+75)) px
    │  └─ warpAffine + resize → stabilized.mp4  (preprocessed/{video}/)
    │
    ▼
[4. Align] Center + rotate + crop → normalized frames
    │        (or use preprocessed video directly as input)
    │
    ▼
[5. DINOv3] Extract features → latent vectors (.npz)
    │   (default dinov3_vitb16, 768-d; dinov3_vitl16 → 1024-d;
    │    dinov2_vitb14_reg4_pretrain still selectable, 768-d)
    │         ├─ weighted_average pooling (default) → 768-dim
    │         └─ multiscale SPP (A-06) → e.g. 21×768 = 16128-dim
    │              (spatial pyramid: 1×1 + 2×2 + 4×4 grids)
    │         ├─ single layer (default) → last layer features
    │         └─ multi-layer (A-06) → concat e.g. layers [3,7,11]
    │
    ▼
[6. UMAP] Dimensionality reduction → 2D embedding
    │
    ▼
[7. DBSCAN] Clustering → behavioral syllables
    │
    ▼
[Output] CSV labels, SRT subtitles, embedding NPZ
```

!!! note "UMAP reproducibility & input standardization"
    Every UMAP run records its resolved random seed, and each clustering session writes a `umap_log.jsonl` file (one JSON line per UMAP stage, recording the seed plus the resolved config). Reuse a logged seed to reproduce an embedding exactly — take the CPU/deterministic path for bit-identical results.

    The first (raw-feature) UMAP stage now applies **per-feature z-score standardization by default** (`"standardize": true` in the default UMAP config preset). This improves cluster separation but changes embeddings relative to older, unstandardized runs, so the DBSCAN `eps` may need re-tuning. Standardization is configurable in the UMAP config JSON.

---

## Project Directory Structure

After a complete analysis run:

```
projects/my-project/
├── config.json                              # Project metadata (file inventory)
├── castle_config.json                       # ProjectConfig (processing parameters)
├── sources/                                 # Video files
│   ├── video1.mp4
│   └── video2.mp4
├── label/                                   # ROI labels (SAM output)
│   └── video1.mp4/
│       ├── 0.npz                            # Label at frame 0
│       └── 247.npz                          # Label at frame 247
├── track/                                   # Tracking results (DeAOT output)
│   └── video1.mp4/
│       └── mask_list.h5                     # HDF5 with per-frame masks
├── preprocessed/                            # Stabilized camera output (Phase 0) ★
│   └── video1.mp4/
│       ├── stabilized.mp4                   # Full-length stabilised video (592×592)
│       └── stabilized_preview.mp4           # 10-second preview clip
├── crop/                                    # Cropped/aligned videos
│   └── video1.mp4/
│       └── video1_ROI_1_crop.mp4
├── latent/                                  # Extracted features (sub-dir per model)
│   └── dinov3_vitb16/                        # default model (DINOv3)
│       ├── video1_ROI_1_dinov3_vitb16_ctr_rmbg.npz        # default pooling
│       ├── video1_ROI_1_dinov3_vitb16_ctr_spp1x2x4.npz    # multiscale SPP
│       └── video1_ROI_1_dinov3_vitb16_ctr_L3x7x11.npz     # multi-layer
├── cluster/                                 # Clustering outputs
│   ├── id.csv                               # Cluster ID → name mapping (legacy)
│   ├── time_series.csv                      # Frame-by-frame assignments (legacy)
│   ├── cluster_grooming_rearing_.npz        # Embedding + labels
│   ├── grid_videos/                         # Pre-rendered cluster grid videos
│   │   └── <session_id>_cluster0.mp4
│   └── sessions/                            # Per-session clustering state
│       └── <session_id>/
│           ├── session.json                 # Session metadata
│           ├── annotations.csv              # Per-cluster labels + comments
│           ├── time_series_<session>.csv    # Frame assignments for this session
│           └── analysis/                   # Ethogram / metrics outputs
└── analysis/                                # Project-wide analysis outputs
```

---

## Phase 2 Performance Architecture ⚡

Phase 2 adds five `castle/core/` modules and one `castle/service/` module focused on **GPU memory management**, **throughput**, and **incremental processing**.

### GPU Memory Pipeline

```
Pipeline.run()
  │
  ├── run_tracking_stage()          # SAM + DeAOT inference
  │     └── track_video() × N
  │
  ├── _cleanup_tracking()           # ← ModelRegistry.unload_family("sam","deaot","aot")
  │       torch.cuda.empty_cache()  #   + explicit CUDA cache flush
  │
  ├── run_extraction_stage()        # DINOv2/DINOv3 inference
  │     └── extract_latent() × N
  │         └── auto_retry_on_oom() wraps batch inference
  │               (halves batch_size on OOM, retries until success or min_batch)
  │
  └── _cleanup_extraction()         # ← ModelRegistry.unload_family("dinov2","dinov3")
          models._evict_model_cache()   + CUDA cache flush
```

VRAM utilisation is logged at every stage boundary (`pipeline-start`, `before-tracking-cleanup`, `after-tracking-cleanup`, `extraction-start`, `after-extraction-cleanup`, `pipeline-end`) and approximately every 100 video iterations during extraction.

!!! tip "Opt-in multi-GPU extraction"
    Set `CASTLE_MULTI_GPU=1` in the environment to split a single video's frames by range across all available CUDA GPUs during latent extraction. Each GPU runs the full decode → preprocess → encode on its frame range, and the partial latents are merged back in original frame order. The merged output is **bit-identical** to the single-GPU result on identical GPUs and runs ~1.9× faster on 2 GPUs. Activates only when the variable is truthy **and** ≥ 2 CUDA GPUs are present; the default is single-GPU.

### ModelRegistry Singleton

```python
registry = ModelRegistry.instance()   # thread-safe singleton

# Lazy load (cached on first call)
model = registry.load("dinov3_vitb16")

# Context manager — auto-unloads on exit
with registry.use("dinov3_vitb16") as model:
    latent = model.extract_tensor_batch(frames, masks, roi_id)

# Bulk unload by family keyword
registry.unload_family("sam", "deaot")

# VRAM diagnostics
stats = registry.get_memory_stats()
# → {"device": "cuda:0", "allocated_mb": 340, "free_mb": 8100, ...}
```

### Auto Batch Size

```python
from castle.core.auto_batch import compute_optimal_batch_size, auto_retry_on_oom

# Query VRAM and return recommended batch size
batch = compute_optimal_batch_size("dinov3_vitb16", frame_size=(518, 518, 3))

# Wrap any callable; retries with halved batch on OOM
result = auto_retry_on_oom(extract_fn, frames, batch_size=batch)
```

`compute_optimal_batch_size` uses conservative per-model weight estimates and a 25 % VRAM safety margin. Falls back to **4** on CPU or when VRAM information is unavailable.

### PipelineCache

```python
from castle.core.cache import PipelineCache

cache = PipelineCache("/data/project/latent")
key = cache.compute_key(video_path, preprocess_config, "dinov3_vitb16")

if cache.is_cached(key):
    path = cache.get(key)
else:
    path = run_extraction(...)
    cache.put(key, path)
```

Key = SHA-256(abs_path + mtime + sorted config JSON + model_name).  
Manifest is written atomically (`os.replace`) to `{cache_dir}/.cache_manifest.json`.

### Incremental Processing

```python
from castle.service.incremental_service import (
    get_unprocessed_videos,
    cleanup_deleted_videos,
)

# Before a batch run — returns only videos without latent output
pending = get_unprocessed_videos("/data/projects/my_project")

# After the user deletes some source videos — removes latent, cluster, and cache data
removed = cleanup_deleted_videos("/data/projects/my_project")
```

---

## Phase 3 Simplification Architecture 🔷

Phase 3 adds three `castle/core/` modules and one `castle/utils/` module focused on **code clarity**, **reducing scattered path logic**, and **clean algorithm dispatch**.

### Unified Project Data (`ProjectData` + `VideoInfo`)

```python
from castle.core.project_data import ProjectData

# Old way (scattered across many files)
mask_path = os.path.join(storage_path, project_name, "track", video_name, "mask_list.h5")

# Phase 3 way — no more manual path joining
pd = ProjectData.from_path("/data/projects/my_project")
mask_path = pd.mask_h5_path("animal01.mp4")   # <root>/track/animal01.mp4/mask_list.h5

# List all source videos with metadata
for video in pd.list_videos():
    print(video.name, video.fps, video.width, video.height, video.n_frames)

# Ensure standard directory tree exists
pd.ensure_dirs()
```

**`VideoInfo`** — lightweight metadata container returned by `list_videos()`:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Video filename (basename) |
| `path` | `Path` | Absolute path to the video file |
| `fps` | `float` | Frames per second (0.0 if unknown) |
| `width` | `int` | Frame width in pixels |
| `height` | `int` | Frame height in pixels |
| `n_frames` | `int` | Total frame count (0 if unknown) |

**`ProjectData`** standard directories and path helpers:

| Property / Method | Returns |
|-------------------|---------|
| `.sources_dir` | `<root>/sources/` |
| `.track_dir` | `<root>/track/` |
| `.latent_dir` | `<root>/latent/` |
| `.cluster_dir` | `<root>/cluster/` |
| `.preprocessed_dir` | `<root>/preprocessed/` |
| `.config_path` | `<root>/config.json` |
| `.video_track_dir(name)` | `<root>/track/<name>/` |
| `.mask_h5_path(name)` | `<root>/track/<name>/mask_list.h5` |
| `.latent_model_dir(model)` | `<root>/latent/<model>/` |
| `.cluster_session_dir(sid)` | `<root>/cluster/sessions/<sid>/` |

---

### Unified Cluster Data (`ClusterData`)

```python
from castle.core.cluster_data import ClusterData

# Load from existing project cluster directory
cd = ClusterData.load("/data/projects/my_project/cluster")
print(cd.n_clusters())           # → int
frames = cd.get_cluster_frames(0)  # → np.ndarray of frame indices

# Construct from freshly computed arrays
cd2 = ClusterData.from_arrays(embeddings, cluster_ids, hierarchy=tree)

# Persist back to disk
cd2.save("/data/projects/my_project/cluster")
```

**`ClusterData`** fields:

| Field | Type | Description |
|-------|------|-------------|
| `labels` | `np.ndarray (N,)` | Flat cluster ID per temporal bin; `-1` = unassigned |
| `hierarchy` | `dict` | Optional hierarchical tree from clustering |
| `names` | `dict[int, str]` | Cluster ID → human-readable name |
| `colors` | `dict[int, tuple]` | Cluster ID → `(R, G, B)` tuple |
| `annotations` | `dict[int, str]` | Cluster ID → annotation label string |

`ClusterData.load()` reads files in order of precedence:
1. `id.csv` → names + colors
2. `cluster_*.npz` → flat label array + hierarchy
3. `time_series_*.csv` → fallback label source
4. `annotations.csv` (or `sessions/<session_id>/annotations.csv`)

---

### Centralised Device Management (`DeviceFactory`)

```python
from castle.core.device_factory import DeviceFactory

# Auto-detect device (cached after first call)
device = DeviceFactory.get_device()   # → 'cuda' | 'mps' | 'cpu'

# Get UMAP for the current device (cuml on GPU, umap-learn otherwise)
umap = DeviceFactory.get_umap(n_neighbors=300, min_dist=0.0, n_components=2)

# Get DBSCAN (cuml on GPU, sklearn otherwise)
dbscan = DeviceFactory.get_dbscan(eps=0.5, min_samples=5)

# Get HDBSCAN (cuml on GPU, sklearn ≥1.3 or hdbscan package otherwise)
hdbscan = DeviceFactory.get_hdbscan(min_cluster_size=10)

# Convert NumPy array to device tensor
tensor = DeviceFactory.to_tensor(my_array)
```

Detection order: **CUDA > MPS (Apple Silicon) > CPU**.  
Override with `DeviceFactory.set_device("cpu")` (e.g. in tests).

---

### Simplified Video Reader (`SimpleVideoReader`)

```python
from castle.utils.video_reader_simple import SimpleVideoReader

with SimpleVideoReader("video.mp4") as r:
    print(r.fps, r.width, r.height, len(r))

    # Random access — seeks to keyframe then decodes to target
    frame = r.get_frame(42)          # (H, W, 3) BGR uint8

    # Sequential iteration — no per-frame seek, most efficient
    for idx, frame in r.iter_frames(start=0, end=500):
        process(frame)

    # Strided iteration — seeks per frame
    for idx, frame in r.iter_frames(start=0, end=500, step=5):
        process(frame)
```

`SimpleVideoReader` vs `VideoReader` (from `castle.utils.video_io`):

| Feature | `SimpleVideoReader` | `VideoReader` |
|---------|---------------------|---------------|
| Dependency | PyAV only | PyAV + optional cv2 |
| LRU frame cache | ✗ | ✓ |
| Binary-search fallback | ✗ | ✓ |
| Sequential read optimisation | ✓ | ✓ |
| Use case | Simple pipelines, tests | Production, caching |

---

### UI Handler Pattern (`castle/ui/HANDLER_GUIDE.md`)

Phase 3 establishes a mandatory **thin-handler / fat-service** convention for all Gradio UI handlers.

```
Gradio Input
    │
    ▼
[Handler]          # ≤ 15 lines; zero business logic
    │  calls once
    ▼
[Service Layer]    # algorithm, I/O, validation
    │  returns
    ▼
[Handler unpacks]  # Gradio output tuple + gr.Error on exception
    │
    ▼
Gradio Output
```

**Error-handling contract:**

| Exception | Handler action |
|-----------|----------------|
| `ValueError` (bad input) | `raise gr.Error(str(e))` |
| `FileNotFoundError` | `raise gr.Error(f"File not found: {e}")` |
| Unexpected / bug | log + `raise gr.Error("Unexpected error — check logs")` |

See `castle/ui/HANDLER_GUIDE.md` for full examples and anti-patterns.

---

## Phase 4 Feature Modules 🟢

Phase 4 adds social/multi-subject analysis, batch processing, and HTML report generation.

### Multi-Subject Tracking (`castle/core/multi_subject.py`)

```python
from castle.core.multi_subject import SubjectTrack, MultiSubjectProject

project = MultiSubjectProject("/data/projects/social_session", "video01.mp4")
project.add_subject(subject_id=0, body_roi=1, head_roi=2)
project.add_subject(subject_id=1, body_roi=3, head_roi=4)
project.process_all()

tracks = project.get_subjects()   # list[SubjectTrack]
```

**`SubjectTrack`** — immutable data container per individual:

| Field | Type | Description |
|-------|------|-------------|
| `subject_id` | `int` | Unique identifier |
| `body_roi_id` | `int` | Body ROI pixel value in mask HDF5 |
| `head_roi_id` | `int` | Head ROI pixel value in mask HDF5 |
| `positions` | `np.ndarray (N, 2)` | Raw centroid [x, y] per frame |
| `angles` | `np.ndarray (N,)` | Unwrapped orientation (degrees) |
| `latents` | `np.ndarray (N, D) or None` | Feature vectors (set after extraction) |
| `labels` | `np.ndarray (N,) or None` | Cluster assignments (set after clustering) |

**`MultiSubjectProject`** orchestrates per-subject preprocessing from the shared mask HDF5. Call `process_all()` to populate `positions` and `angles`; feature extraction and clustering are applied externally via `set_latents()` / `set_labels()`.

---

### Social Feature Extraction (`castle/analysis/social_features.py`)

Operates on synchronised lists of `SubjectTrack` objects (same `n_frames`).

```python
from castle.analysis.social_features import (
    compute_pairwise_distance,
    compute_relative_orientation,
    compute_approach_score,
    detect_social_events,
)

dist    = compute_pairwise_distance(tracks)      # (N, S, S)
orient  = compute_relative_orientation(tracks)   # (N, S, S) degrees
approach = compute_approach_score(tracks, window=30)  # (N, S, S)
events  = detect_social_events(tracks, distance_threshold=50, duration_threshold=15)
```

| Function | Output | Description |
|----------|--------|-------------|
| `compute_pairwise_distance` | `(N, S, S)` float | Symmetric pixel distance matrix per frame |
| `compute_relative_orientation` | `(N, S, S)` float | Angle (°) from subject *i* heading toward *j*; 0° = facing |
| `compute_approach_score` | `(N, S, S)` float | Negative Δdistance over sliding window; positive = approaching |
| `detect_social_events` | `list[dict]` | Proximity events ≥ `duration_threshold` frames |

Each event dict contains `type`, `subjects`, `start_frame`, `end_frame`, `duration`.

---

### Group Ethogram (`castle/analysis/group_ethogram.py`)

```python
from castle.analysis.group_ethogram import build_group_ethogram, plot_group_ethogram

ethogram = build_group_ethogram(tracks, fps=30.0, cluster_names={0: "rest", 1: "groom"})
path = plot_group_ethogram(ethogram, output_path="/tmp/group_ethogram.png")
```

`build_group_ethogram` returns a dict with `fps`, `n_frames`, `n_subjects`, `subject_ids`, `per_subject` (per-subject ethogram + labels), `social_events`, and `time_axis`.

`plot_group_ethogram` renders a colour-coded raster (one row per subject) plus a social-event shading row at the bottom. Saved as PNG/SVG depending on the extension.

---

### Batch Processing (`castle/core/batch.py`)

YAML-driven multi-project batch runner.

```yaml
# experiments.yaml
experiments:
  - name: "Control Group"
    project: "/data/control"
    videos: ["mouse1.mp4", "mouse2.mp4"]
    params:
      fc: 0.25
      n_clusters: 10
  - name: "Treatment Group"
    project: "/data/treatment"
    videos: ["mouse3.mp4"]

parallel: false
max_workers: 2
```

```python
from castle.core.batch import BatchConfig, BatchRunner

config  = BatchConfig.from_yaml("experiments.yaml")
runner  = BatchRunner(config)
results = runner.run(progress_callback=lambda f, m: print(f"{f:.0%} {m}"))
print(runner.generate_summary(results))
```

Each result dict contains `name`, `project`, `status` (`done`/`error`/`skipped`), `tracking`, `extraction`, `elapsed_s`, `error`.  
Set `config.parallel = True` + `config.max_workers = N` for concurrent project execution.

---

### Batch CLI (`castle/cli/batch_cmd.py`)

```bash
# Run all experiments
castle batch run experiments.yaml [--parallel] [--max-workers 4]

# Show status of last run
castle batch status experiments.yaml

# Generate HTML reports
castle batch report experiments.yaml --output reports/
castle batch report experiments.yaml --output combined.html
```

Results are persisted as `experiments.batch_result.json` alongside the YAML file.

---

### HTML Report Generator (`castle/analysis/report.py`)

```python
from castle.analysis.report import ReportGenerator

gen  = ReportGenerator("/storage/my_project", session_id="exp01")
path = gen.generate(
    output_path="report.html",
    include_ethogram=True,
    include_quality=True,
    include_comparison=False,
)
```

Reports are **self-contained HTML files** (no external CSS/JS dependencies) with:
- Project metadata cards (project name, session, frame count, cluster count, model names)
- Inline base64 PNG ethogram frequency bar chart
- Bout statistics table (frequency %, mean/median duration, CV)
- Transition matrix with entropy and stationarity metrics
- Clustering quality badges (Silhouette, Calinski-Harabász, Davies-Bouldin, Inertia)
- Optional 2-D embedding scatter plot (if `embedding_2d` available from metrics service)
- Group comparison placeholder (links to `BatchRunner.generate_summary()`)

`generate()` falls back gracefully if `matplotlib` plots fail (e.g. no cluster data) — it logs a warning and omits the plot section rather than aborting the report.
