# Architecture

## System Overview

```
┌────────────────────────────────────────────────────────────────┐
│                        Frontend Layer                          │
│                                                                │
│  castle/cli/          castle/ui/           castle/desktop/     │
│  (typer CLI)          (Gradio Web UI)      (PyQt6 App)        │
└────────────┬───────────────┬───────────────────┬──────────────┘
             │               │                   │
             ▼               ▼                   ▼
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
│  pipeline_parallel.py  — 3-stage threaded extractor ⚡         │
│  cache.py              — Content-hash PipelineCache ⚡         │
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
| `info_cmd.py` | `castle info <project>` |
| `ethogram_cmd.py` | `castle ethogram analyze/transitions/bouts/export/export-nwb` |
| `compare_cmd.py` | `castle compare run/fingerprint` |
| `gui_cmd.py` | `castle gui` — launch the PyQt6 desktop app |
| `mcp_cmd.py` | `castle mcp start` — start the MCP server |

### `castle/ui/` — Gradio Web Interface

Built on [Gradio](https://gradio.app/). Each tab has its own module.

7 top-level tabs: **0. Project | 1. Upload Videos | 2. Tracking ROIs | 3. Extract Latent | 4. Behavior Microscope | 5. Analysis | 6. Export**

| Module | Tab | Purpose |
|--------|-----|---------|
| `main_ui.py` | — | Creates the top-level app with all 7 tabs |
| `project_ui.py` | 0. Project | Create, open, delete projects |
| `source_ui.py` | 1. Upload Videos | Upload local files or scan server directories |
| `edit_ui.py` | 2. Tracking ROIs | Container for all tracking sub-UIs |
| `view_ui.py` | └─ View | Browse frames with slider |
| `label_ui.py` | └─ Label ROI | Point-and-click segmentation with SAM |
| `knowledge_ui.py` | └─ ROI Prompts | Gallery of all saved ROI labels |
| `track_ui.py` | └─ Tracking | Run DeAOT tracking with progress |
| `post_track_ui.py` | └─ Post-Track | Post-process and review tracking results |
| `batch_track_ui.py` | └─ Batch | Process multiple videos |
| `preprocess_ui.py` | └─ Preprocessing | Stabilized camera preprocessing (P0) ★ |
| `extract_ui.py` | 3. Extract Latent | Configure and run feature extraction |
| `cluster_page_ui.py` | 4. Behavior Microscope | UMAP + DBSCAN clustering workspace |
| `embedding_scatter.py` | └─ (component) | Plotly embedding scatter widget |
| `cluster_handlers.py` | └─ (component) | Cluster operation callbacks |
| `cluster_tree.py` | └─ (component) | Hierarchical cluster tree view |
| `cluster_input_ui.py` | └─ (component) | Clustering parameter input widgets |
| `annotator_ui.py` | └─ Cluster Annotator | Grid video browser, per-session labels, auto-save |
| `analysis_ui.py` | 5. Analysis | Ethogram, Quality Metrics sub-tabs, Group Comparison placeholder |
| `export_ui.py` | 6. Export | ZIP download with selectable data components |
| `plot_mask_info.py` | (component) | Mask info / contour overlay utilities |

### `castle/desktop/` — PyQt6 Desktop Application

Native desktop GUI using [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) and [pyqtgraph](https://pyqtgraph.readthedocs.io/).

8 tabs: **0. Project | 1. Upload Videos | 2. Tracking ROIs | 3. Extract Latent | 4. Behavior Microscope | 5. Annotator | 6. Analysis | 7. Export**

| Module | Purpose |
|--------|---------|
| `main_window.py` | Main window with 8 tabs |
| `services/worker_threads.py` | QThread `ServiceWorker` for background tasks |
| `components/syllable_bar.py` | pyqtgraph-based syllable bar widget |
| `components/embedding_view.py` | Embedding scatter view |
| `components/video_player.py` | In-app video player |
| `components/cluster_tree.py` | Hierarchical cluster tree widget |
| `widgets/project_panel.py` | Tab 0 — Project management |
| `widgets/source_panel.py` | Tab 1 — Video source |
| `widgets/tracking_panel.py` | Tab 2 — ROI tracking |
| `widgets/extract_panel.py` | Tab 3 — Feature extraction |
| `widgets/cluster_panel.py` | Tab 4 — Behavior Microscope (UMAP + DBSCAN) |
| `widgets/annotator_panel.py` | Tab 5 — Cluster Annotator (grid video, labels, comments, auto-save) |
| `widgets/analysis_panel.py` | Tab 6 — Analysis (Ethogram + Quality Metrics sub-tabs) |
| `widgets/export_panel.py` | Tab 7 — Export (ZIP with selectable components) |

### `castle/service/` — Service Layer

Clean separation between frontends and business logic. All three frontends (CLI, Gradio, Desktop) call these services.

| Module | Purpose |
|--------|---------|
| `project_service.py` | Project CRUD (create, list, info, delete) |
| `extraction_service.py` | Feature extraction orchestration |
| `clustering_service.py` | UMAP + DBSCAN session management, recursive auto-clustering |
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
| `auto_cluster.py` | Recursive hierarchical Behavior Microscope — automated multi-level clustering |
| `data.py` | `Preprocess` pipeline, `VideoDataset` for batched extraction |
| `models.py` | `VisualEncoder` abstraction: DINOv2, DINOv3, multi-scale pooling |
| `config.py` | Constants: checkpoint paths, model IDs, supported models |
| `project.py` | Project config read/write (file inventory) |
| `project_config.py` | `ProjectConfig` dataclass — typed processing parameters |
| `environment.py` | Device detection (`cuda`/`mps`/`cpu`), worker count |
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
| `pipeline_parallel.py` ⚡ | `ParallelExtractor` — 3-stage producer-consumer pipeline (I/O thread → CPU preprocess thread → GPU inference) |
| `cache.py` ⚡ | `PipelineCache` — SHA-256 content-addressed cache; manifest persisted as JSON; stale-entry auto-invalidation |

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
[5. DINOv2/v3] Extract features → latent vectors (.npz)
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
│       ├── stabilized.mp4                   # Full-length stabilised video (518×518)
│       └── stabilized_preview.mp4           # 10-second preview clip
├── crop/                                    # Cropped/aligned videos
│   └── video1.mp4/
│       └── video1_ROI_1_crop.mp4
├── latent/                                  # Extracted features
│   └── dinov2_vitb14_reg/
│       ├── video1_ROI_1_dinov2_vitb14_reg_ctr_rmbg.npz        # default pooling
│       ├── video1_ROI_1_dinov2_vitb14_reg_ctr_spp1x2x4.npz    # multiscale SPP
│       └── video1_ROI_1_dinov2_vitb14_reg_ctr_L3x7x11.npz     # multi-layer
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

### ModelRegistry Singleton

```python
registry = ModelRegistry.instance()   # thread-safe singleton

# Lazy load (cached on first call)
model = registry.load("dinov2_vitb14")

# Context manager — auto-unloads on exit
with registry.use("dinov2_vitb14") as model:
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
batch = compute_optimal_batch_size("dinov2_vitb14", frame_size=(518, 518, 3))

# Wrap any callable; retries with halved batch on OOM
result = auto_retry_on_oom(extract_fn, frames, batch_size=batch)
```

`compute_optimal_batch_size` uses conservative per-model weight estimates and a 25 % VRAM safety margin. Falls back to **4** on CPU or when VRAM information is unavailable.

### ParallelExtractor (3-Stage Pipeline)

```
Thread 1 (I/O)       Thread 2 (CPU)       Main thread (GPU)
VideoReader.get_frame  StabilizedCamera     DINOv2 batched
      │   → frame_queue →   .generate_frame   inference
      │                      │  → tensor_queue → np.concatenate → (N, D)
```

- Uses `threading` (not `multiprocessing`) to avoid CUDA fork issues.
- Bounded `queue.Queue(maxsize=32)` controls peak memory.
- Preprocessing errors produce zero-filled frames; inference errors produce zero-filled latents — both logged as warnings so a single bad frame never aborts the run.

### PipelineCache

```python
from castle.core.cache import PipelineCache

cache = PipelineCache("/data/project/latent")
key = cache.compute_key(video_path, preprocess_config, "dinov2_vitb14")

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
