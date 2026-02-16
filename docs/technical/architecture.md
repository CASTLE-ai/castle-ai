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
│  extractor.py        — Feature extraction engine               │
│  cluster.py          — LatentAggregator, clustering logic      │
│  data.py             — Preprocess, VideoDataset                │
│  models.py           — VisualEncoder abstraction (DINOv2/v3)   │
│  config.py           — Constants, model paths                  │
│  project.py          — Project config I/O (file inventory)     │
│  project_config.py   — ProjectConfig dataclass (B-05)          │
│  environment.py      — Device detection, worker count          │
│  mask_filter.py      — Post-tracking mask filtering (A-03)     │
│  logging_config.py   — Centralized logging setup               │
│  ethogram.py         — Ethogram analysis engine (P1)           │
│  metrics.py          — Clustering quality metrics (P2)         │
│  comparison.py       — Group comparison engine (P4)            │
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
| `project_cmd.py` | `castle project create/list/info/delete` |
| `cluster_cmd.py` | `castle cluster run/list/evaluate` |
| `extract_cmd.py` | `castle extract run` |
| `track_cmd.py` | `castle track run` |
| `info_cmd.py` | `castle info status/devices` |
| `ethogram_cmd.py` | `castle ethogram analyze/transitions/bouts/export` |
| `compare_cmd.py` | `castle compare run/fingerprint` |

### `castle/ui/` — Gradio Web Interface

Built on [Gradio](https://gradio.app/). Each tab has its own module:

| Module | Tab | Purpose |
|--------|-----|---------|
| `main_ui.py` | — | Creates the top-level app with all tabs |
| `project_ui.py` | 0. Project | Create, open, delete projects |
| `source_ui.py` | 1. Upload Videos | Upload local files or scan server directories |
| `edit_ui.py` | 2. Tracking ROIs | Container for all tracking sub-UIs |
| `view_ui.py` | └─ View | Browse frames with slider |
| `label_ui.py` | └─ Label ROI | Point-and-click segmentation with SAM |
| `knowledge_ui.py` | └─ ROI Prompts | Gallery of all saved ROI labels |
| `track_ui.py` | └─ Tracking | Run DeAOT tracking with progress |
| `post_track_ui.py` | └─ Analysis | Post-process and review tracking |
| `batch_track_ui.py` | └─ Batch | Process multiple videos |
| `extract_ui.py` | 3. Extract Latent | Configure and run feature extraction |
| `cluster_page_ui.py` | 4. Behavior Microscope | UMAP + DBSCAN analysis |
| `embedding_scatter.py` | └─ (component) | Plotly embedding scatter widget |
| `cluster_handlers.py` | └─ (component) | Cluster operation callbacks |
| `cluster_tree.py` | └─ (component) | Hierarchical cluster tree view |

### `castle/desktop/` — PyQt6 Desktop Application

Native desktop GUI using [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) and [pyqtgraph](https://pyqtgraph.readthedocs.io/).

| Module | Purpose |
|--------|---------|
| `main_window.py` | Main window with 5 tabs |
| `workers.py` | QThread workers for background tasks |
| `components/syllable_bar.py` | pyqtgraph-based syllable bar widget |
| `widgets/project_panel.py` | Project management panel |
| `widgets/source_panel.py` | Video source panel |
| `widgets/tracking_panel.py` | ROI tracking panel |
| `widgets/extract_panel.py` | Feature extraction panel |
| `widgets/cluster_panel.py` | Behavior clustering panel |

### `castle/service/` — Service Layer

Clean separation between frontends and business logic. All three frontends (CLI, Gradio, Desktop) call these services.

| Module | Purpose |
|--------|---------|
| `project_service.py` | Project CRUD (create, list, info, delete) |
| `extraction_service.py` | Feature extraction orchestration |
| `clustering_service.py` | UMAP + DBSCAN session management |
| `tracking_service.py` | Tracking pipeline orchestration |
| `annotation_service.py` | Classification scheme management |
| `bout_service.py` | Behavioral bout analysis and export |
| `history_service.py` | Undo/Redo via Command Pattern |
| `ethogram_service.py` | Ethogram analysis: loads cluster data, delegates to `castle.core.ethogram` |
| `metrics_service.py` | Clustering quality evaluation: loads labels/embedding, delegates to `castle.core.metrics` |
| `comparison_service.py` | Group comparison: loads per-video data, delegates to `castle.core.comparison` |

### `castle/core/` — Core Business Logic

| Module | Purpose |
|--------|---------|
| `extractor.py` | Feature extraction execution engine |
| `cluster.py` | `LatentAggregator` — multi-video latent loading and frame retrieval |
| `data.py` | `Preprocess` pipeline, `VideoDataset` for batched extraction |
| `models.py` | `VisualEncoder` abstraction: DINOv2, DINOv3, multi-scale pooling |
| `config.py` | Constants: checkpoint paths, model IDs, supported models |
| `project.py` | Project config read/write (file inventory) |
| `project_config.py` | `ProjectConfig` dataclass — typed processing parameters (B-05) |
| `environment.py` | Device detection (`cuda`/`mps`/`cpu`), worker count |
| `mask_filter.py` | Post-tracking mask filtering — largest component, configurable threshold (A-03) |
| `logging_config.py` | Centralized logging setup |
| `ethogram.py` | Ethogram engine — bout extraction, transition matrix, temporal coherence (P1) |
| `metrics.py` | Clustering quality metrics — silhouette, CH, DB, temporal coherence, bout quality, external validation (P2) |
| `comparison.py` | Group comparison — BFA test, behavioral fingerprint, energy distance, permutation tests, Hedges' g (P4) |

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
[3. Align] Center + rotate + crop → normalized frames
    │
    ▼
[4. DINOv2/v3] Extract features → latent vectors (.npz)
    │         ├─ weighted_average pooling (default) → 768-dim
    │         └─ multiscale SPP (A-06) → e.g. 21×768 = 16128-dim
    │              (spatial pyramid: 1×1 + 2×2 + 4×4 grids)
    │         ├─ single layer (default) → last layer features
    │         └─ multi-layer (A-06) → concat e.g. layers [3,7,11]
    │
    ▼
[5. UMAP] Dimensionality reduction → 2D embedding
    │
    ▼
[6. DBSCAN] Clustering → behavioral syllables
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
├── crop/                                    # Cropped/aligned videos
│   └── video1.mp4/
│       └── video1_ROI_1_crop.mp4
├── latent/                                  # Extracted features
│   └── dinov2_vitb14_reg/
│       ├── video1_ROI_1_dinov2_vitb14_reg_ctr_rmbg.npz        # default pooling
│       ├── video1_ROI_1_dinov2_vitb14_reg_ctr_spp1x2x4.npz    # A-06 multiscale
│       └── video1_ROI_1_dinov2_vitb14_reg_ctr_L3x7x11.npz     # A-06 multi-layer
└── cluster/                                 # Analysis results
    ├── id.csv                               # Cluster ID → name mapping
    ├── time_series.csv                      # Frame-by-frame assignments
    └── cluster_grooming_rearing_.npz        # Embedding + labels
```
