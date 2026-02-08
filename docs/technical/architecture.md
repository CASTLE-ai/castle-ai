# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                      app.py (Entry)                     │
│                    Gradio Application                    │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                     castle/ui/                          │
│  main_ui ─┬─ project_ui    (0. Project)                │
│           ├─ source_ui     (1. Upload Videos)           │
│           ├─ edit_ui       (2. Tracking ROIs)           │
│           │   ├─ view_ui        View frames             │
│           │   ├─ label_ui       Label ROIs (SAM)        │
│           │   ├─ knowledge_ui   ROI prompt gallery      │
│           │   ├─ track_ui       Run tracking (DeAOT)    │
│           │   ├─ post_track_ui  Post-process results    │
│           │   └─ batch_track_ui Batch processing        │
│           ├─ extract_ui    (3. Extract Latent)          │
│           └─ cluster_page_ui (4. Behavior Microscope)   │
└──────────────────────────┬──────────────────────────────┘
                           │ calls
┌──────────────────────────▼──────────────────────────────┐
│                   castle/core/                          │
│  extractor.py   ─ Feature extraction engine             │
│  cluster.py     ─ LatentAggregator, clustering logic    │
│  data.py        ─ Preprocess, VideoDataset              │
│  models.py      ─ Visual encoder abstraction            │
│  config.py      ─ Constants, model paths                │
│  project.py     ─ Project config I/O                    │
└──────────────────────────┬──────────────────────────────┘
                           │ uses
┌──────────────────────────▼──────────────────────────────┐
│                   castle/utils/                         │
│  project_manager.py      Project CRUD operations        │
│  video_manager.py        Video import/scan              │
│  video_io.py             Video read/write (PyAV)        │
│  video_align.py          Center, rotate, crop frames    │
│  image_segment.py        SAM wrapper (Segmentor)        │
│  video_object_segment.py DeAOT wrapper                  │
│  tracking_manager.py     ROI tracking orchestration     │
│  visual_latent_extract.py DINOv2/v3 wrapper             │
│  latent_explorer.py      Latent class, UMAP, clustering │
│  myumap.py               Custom UMAP (cuml + spectral)  │
│  h5_io.py                HDF5 mask storage              │
│  plot.py                 Visualization helpers           │
│  roi_manager.py          ROI utilities                   │
│  download.py             Checkpoint download (gdown)     │
└──────────────────────────┬──────────────────────────────┘
                           │ wraps
┌──────────────────────────▼──────────────────────────────┐
│              Vendored / External Models                  │
│  castle/sam/    ─ Segment Anything Model (Meta)          │
│  castle/aot/    ─ DeAOT video object segmentation        │
│  castle/configs/─ model_config.json                      │
└─────────────────────────────────────────────────────────┘
```

## Module Map

### `castle/ui/` — User Interface

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

### `castle/core/` — Core Business Logic

| Module | Purpose |
|--------|---------|
| `extractor.py` | Feature extraction execution engine |
| `cluster.py` | `LatentAggregator` — multi-video latent loading and frame retrieval |
| `data.py` | `Preprocess` dataclass, `VideoDataset` for batched extraction |
| `models.py` | `VisualEncoder` abstraction, model registry |
| `config.py` | Constants: checkpoint paths, model IDs, supported models |
| `project.py` | Project config read/write |

### `castle/utils/` — Utility Layer

| Module | Purpose |
|--------|---------|
| `project_manager.py` | Project CRUD (create, list, delete) |
| `video_manager.py` | Video import, directory scanning, format detection |
| `video_io.py` | Video read/write using PyAV, subtitle generation |
| `video_align.py` | Frame alignment: center, rotate, crop |
| `image_segment.py` | SAM wrapper (`Segmentor` class) |
| `video_object_segment.py` | DeAOT wrapper (model loading) |
| `tracking_manager.py` | `ROITracker` — orchestrates multi-frame tracking |
| `visual_latent_extract.py` | DINOv2/v3 wrapper functions |
| `latent_explorer.py` | `Latent` class — embedding, clustering, visualization |
| `myumap.py` | Custom UMAP using cuml + spectral layout |
| `h5_io.py` | `H5IO` — HDF5 file I/O for mask storage |
| `plot.py` | Visualization: frame+mask overlay, dot annotations |
| `roi_manager.py` | ROI color management and utilities |
| `download.py` | Checkpoint download via gdown |

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
├── config.json                              # Project metadata
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
│   └── dinov2_vitb14_reg4_pretrain/
│       └── video1_ROI_1_dinov2_vitb14_reg4_pretrain.npz
└── cluster/                                 # Analysis results
    ├── id.csv                               # Cluster ID → name mapping
    ├── time_series.csv                      # Frame-by-frame assignments
    └── cluster_grooming_rearing_.npz        # Embedding + labels
```
