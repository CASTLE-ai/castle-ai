# Data Formats

Reference for all file formats CASTLE reads and writes.

---

## Input

### Video Files

CASTLE uses [PyAV](https://pyav.org/) (FFmpeg wrapper) for video I/O.

**Supported formats**: MP4, AVI, MOV, WMV, FLV, MKV

**Supported extensions**: `.mp4`, `.avi`, `.mov`, `.wmv`, `.flv`, `.mkv`

**Recommended**: MP4 with H.264 codec for best compatibility and performance.

No resolution or frame rate limitations — CASTLE processes whatever the video contains.

---

## Intermediate Files

### Label Files (`.npz`)

Created by the **Label ROI** step. Stored in `project/label/{video_name}/`.

```python
import numpy as np
data = np.load('0.npz')

data['frame']  # np.ndarray, shape (H, W, 3), dtype uint8 — RGB frame
data['mask']   # np.ndarray, shape (H, W, N_ROIS), dtype uint8 — per-ROI masks
```

- Filename is the frame index (e.g., `0.npz`, `247.npz`)
- Each ROI is a separate channel in the mask array
- ROI colors are assigned sequentially

### Tracked Masks (`mask_list.h5`)

Created by the **Tracking** step. Stored in `project/track/{video_name}/`.

HDF5 file using `H5IO` wrapper:

```python
from castle.utils.h5_io import H5IO

tracker = H5IO('mask_list.h5')
mask = tracker[frame_index]       # np.ndarray, shape (H, W), dtype uint8
n_frames = len(tracker)           # Total number of frames
n_rois = tracker.get_n_rois()     # Number of tracked ROIs
```

- Each frame is stored as a gzip-compressed dataset keyed by frame index (string)
- Metadata keys: `total_frames`, `n_rois`
- Mask values encode ROI IDs as pixel colors

### Cropped Video (`.mp4`)

Created by **Extract Crop Video**. Stored in `project/crop/{video_name}/`.

- Filename: `{video_basename}_ROI_{id}_crop.mp4`
- Standard MP4 video of the aligned/cropped ROI

---

## Output Files

### Latent Features (`.npz`)

Created by **Extract Latent**. Stored in `project/latent/{model_name}/`.

```python
import numpy as np
data = np.load('video_ROI_1_dinov3_vitb16.npz')

data['latent']  # np.ndarray, shape (n_frames, feature_dim), dtype float32
```

- **Feature dimensions**: 768 (ViT-B models, e.g. the default `dinov3_vitb16`) or 1024 (ViT-L models, e.g. `dinov3_vitl16`)
- Frames with empty masks → NaN vectors
- **Filename pattern**: `{video}_ROI_{roi_id}_{model}_{tags}.npz`
- **Tags**: `ctr` (centered), `rmbg` (background removed)

### Cluster ID Mapping (`id.csv`)

Created by **Submit** in Behavior Microscope. Stored in `project/cluster/`.

```csv
Id,Name
0,init
1,grooming
2,rearing
3,locomotion
```

| Column | Type | Description |
|--------|------|-------------|
| `Id` | int | Cluster numeric ID |
| `Name` | string | Human-assigned behavior name |

### Time Series (`time_series.csv`)

Frame-by-frame behavioral state assignments. Stored in `project/cluster/`.

```csv
,behavior
0,1
1,1
2,1
3,3
4,2
```

| Column | Type | Description |
|--------|------|-------------|
| (index) | int | Frame index |
| `behavior` | int | Cluster ID for this frame |

- `-1` indicates unclassified / noise frames
- When `time_window > 1`, values are repeated for each frame in the window (expanded to per-frame resolution)

### SRT Subtitles (`.srt`)

Standard subtitle format for video overlay. Generated per-video.

```srt
1
00:00:00,000 --> 00:00:01,500
grooming

2
00:00:01,500 --> 00:00:03,200
locomotion
```

### Embedding NPZ

Saved UMAP coordinates and cluster labels. Stored in `project/cluster/`.

```python
import numpy as np
data = np.load('cluster_grooming_rearing_.npz')

data['emb']     # np.ndarray, shape (n_samples, 2) — UMAP 2D coordinates
data['cls']     # np.ndarray, shape (n_samples,), dtype int16 — cluster IDs
data['config']  # UMAP configuration used (includes the resolved random seed)
```

- NaN in `emb` → frame excluded from analysis
- `-1` in `cls` → unclassified frame

!!! tip "Reproducible embeddings"
    Each clustering session also writes a `umap_log.jsonl` next to the session
    files — one JSON line per UMAP stage recording the resolved random seed and
    the config for that stage. Re-running with the logged seed reproduces the
    embedding (use the CPU/deterministic path for bit-identical results). Note
    that per-feature z-score **standardization is ON by default** for the first
    (raw-feature) UMAP stage (`"standardize": true`), which improves cluster
    separation but changes coordinates versus older runs, so the DBSCAN `eps`
    may need re-tuning. Set `"standardize": false` in the stage-0 UMAP config to
    disable it.

---

## Project Directory Layout

Complete structure after a full analysis:

```
projects/my-project/
├── config.json
├── sources/
│   ├── video1.mp4
│   └── video2.mp4
├── label/
│   ├── video1.mp4/
│   │   ├── 0.npz
│   │   └── 247.npz
│   └── video2.mp4/
│       └── 0.npz
├── track/
│   ├── video1.mp4/
│   │   └── mask_list.h5
│   └── video2.mp4/
│       └── mask_list.h5
├── crop/
│   └── video1.mp4/
│       └── video1_ROI_1_crop.mp4
├── latent/
│   └── dinov3_vitb16/
│       ├── video1_ROI_1_dinov3_vitb16.npz
│       └── video2_ROI_1_dinov3_vitb16.npz
└── cluster/
    ├── id.csv
    ├── time_series.csv
    └── cluster_grooming_rearing_.npz
```
