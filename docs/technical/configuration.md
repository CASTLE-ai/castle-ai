# Configuration Reference

All configurable parameters in CASTLE, consolidated in one place.

---

## Launch Options

### Gradio Web UI (`app.py`)

```bash
python app.py [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--root PATH` | Custom project storage path | `projects/` |
| `--share` | Enable Gradio public URL (auto-enabled on Colab) | `False` |

### PyQt6 Desktop App

```bash
castle gui [OPTIONS]
# or: python -m castle.desktop [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--storage PATH` | Custom project storage path | `projects/` |
| `--project NAME` | Auto-open this project on startup | `None` |

### CLI

```bash
castle [COMMAND] [OPTIONS]
```

Use `castle --help` or `castle <command> --help` for full option reference. Key environment variable:

| Variable | Description |
|----------|-------------|
| `CASTLE_STORAGE` | Default project storage directory |

---

## Model Configuration (`castle/configs/model_config.json`)

Defines paths and parameters for all models:

### SAM (Segmentation)

```json
{
    "sam_args": {
        "sam_checkpoint": "ckpt/sam_vit_b_01ec64.pth",
        "model_type": "vit_b",
        "generator_args": {
            "points_per_side": 16,
            "pred_iou_thresh": 0.8,
            "stability_score_thresh": 0.9,
            "crop_n_layers": 1,
            "crop_n_points_downscale_factor": 2,
            "min_mask_region_area": 200
        }
    }
}
```

| Parameter | Description |
|-----------|-------------|
| `sam_checkpoint` | Path to SAM weights |
| `model_type` | SAM variant (`vit_b`) |
| `points_per_side` | Auto-mask grid density |
| `pred_iou_thresh` | IoU prediction threshold |
| `stability_score_thresh` | Mask stability filter |
| `min_mask_region_area` | Minimum mask size (pixels) |

### DeAOT (Tracking)

```json
{
    "r50_deaotl": {
        "phase": "PRE_YTB_DAV",
        "model": "r50_deaotl",
        "model_path": "ckpt/R50_DeAOTL_PRE_YTB_DAV.pth"
    },
    "swinb_deaotl": {
        "phase": "PRE_YTB_DAV",
        "model": "swinb_deaotl",
        "model_path": "ckpt/SwinB_DeAOTL_PRE_YTB_DAV.pth"
    }
}
```

### DINOv2 / DINOv3 (Feature Extraction)

```json
{
    "dinov2_args": {
        "name": "dinov2_vitb14_reg",
        "path": "ckpt/dinov2_vitb14_reg4_pretrain.pth"
    },
    "dinov3_args": {
        "name": "dinov3_vitl16",
        "path": "ckpt/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    },
    "dinov3_vitb16_args": {
        "name": "dinov3_vitb16",
        "path": "ckpt/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    }
}
```

---

## Extraction Parameters

Configured in the **3. Extract Latent** tab:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Visual Model | `dinov3_vitb16` | 3 options | Feature extraction backbone (`dinov3_vitb16` default, 768-d; `dinov3_vitl16`, 1024-d; `dinov2_vitb14_reg4_pretrain`, 768-d) |
| ROI ID | `1` | Any tracked ROI | Which ROI to extract |
| Batch Size | `32` | 1–256+ | Frames per batch (limited by VRAM) |
| Skip Existing | `True` | — | Don't re-extract existing files |

### Preprocessing

| Parameter | Default | Description |
|-----------|---------|-------------|
| Center ROI | `False` | Crop centered on ROI centroid |
| Center ROI ID | `1` | Which ROI to center on |
| Crop Width | `300` px | Crop window width |
| Crop Height | `300` px | Crop window height |
| Rotate (Tail) | `False` | Normalize orientation via tail ROI |
| Tail ROI ID | `2` | Which ROI defines tail direction |
| Remove Background | `False` | Zero out non-ROI pixels |

---

## UMAP Parameters

Configured in the **4. Behavior Microscope** tab. See [Behavior Analysis Tutorial](../tutorials/step4-analysis.md) for preset details.

### Custom Configuration Format

JSON array of UMAP stages:

```json
[
    {
        "n_neighbors": 100,
        "min_dist": 0.0,
        "n_components": 2,
        "standardize": true
    }
]
```

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `n_neighbors` | Nearest neighbors (local vs global structure) | 25–1000 |
| `min_dist` | Minimum distance in embedding | 0.0 (for clustering) |
| `n_components` | Output dimensions per stage | 2, 5, or 10 |
| `standardize` | Per-feature z-score of the raw features before stage 0 (first stage only). Default `true`. | `true` / `false` |

**Internal parameter** (in `myumap.py`): `n_epochs = 20000` — number of optimization epochs. Not exposed in the UI.

!!! note "Input standardization is on by default"
    The first (raw-feature) UMAP stage now z-scores each feature before fitting (`"standardize": true` in the default preset). This improves cluster separation but **changes embeddings versus older runs**, so the DBSCAN `eps` may need re-tuning. Set `"standardize": false` in the stage-0 config to reproduce a pre-standardization layout. Later stages run on low-dim UMAP output and are not standardized.

!!! tip "Reproducible UMAP runs"
    Every UMAP run records its resolved random seed, and each clustering session writes a `umap_log.jsonl` file — one JSON line per UMAP stage capturing the seed and resolved config. To reproduce an embedding exactly, reuse the logged seed (set `random_state` in the stage config) and run the CPU/deterministic path for bit-identical results.

---

## DBSCAN Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `eps` | `1.0` | 0.1–10.0 | Epsilon-neighborhood radius |

Smaller eps → more clusters. Larger eps → fewer clusters.

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `COLAB_GPU` | Auto-detected in Google Colab; enables `--share` and Colab-specific paths |
| `HDF5_USE_FILE_LOCKING` | Set to `FALSE` by `app.py` to avoid HDF5 locking issues |
| `CASTLE_MULTI_GPU` | Opt-in multi-GPU latent extraction (Gradio app, desktop app, or CLI). Default OFF. Set to `1` to enable. |

!!! tip "Multi-GPU latent extraction"
    When `CASTLE_MULTI_GPU=1` **and** at least two CUDA GPUs are visible, latent extraction for a single video splits its frames by range across the GPUs (each GPU runs the full decode → preprocess → encode on its half; results are merged in original order). Output is bit-identical to the single-GPU path on identical GPUs, and roughly 1.9× faster on two GPUs. The default (unset, or `0`/`false`/`off`) is single-GPU.

---

## ProjectConfig (B-05)

The `ProjectConfig` dataclass (`castle/core/project_config.py`) provides typed, serializable configuration for the entire CASTLE processing pipeline. It is stored as `castle_config.json` in the project directory.

```python
from castle.core.project_config import ProjectConfig

cfg = ProjectConfig()                    # defaults
cfg = ProjectConfig.load('castle_config.json')  # from file
cfg.save('castle_config.json')           # to file
```

### Extraction Parameters (A-06)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `extraction.model` | `str` | `'dinov3_vitb16'` | Visual encoder model name |
| `extraction.roi_ids` | `List[int]` | `[1]` | ROI IDs to extract |
| `extraction.batch_size` | `int` | `32` | Frames per GPU batch |
| `extraction.bin_size` | `int` | `1` | Temporal binning (frames per sample) |
| `extraction.pooling_method` | `str` | `'weighted_average'` | `'weighted_average'` or `'multiscale'` |
| `extraction.pooling_scales` | `List[int]` | `[1, 2, 4]` | Grid divisions for multi-scale SPP (only used when `pooling_method='multiscale'`) |
| `extraction.feature_layers` | `Optional[List[int]]` | `None` | Layer indices for multi-layer extraction (e.g. `[3, 7, 11]`). `None` = last layer only. |

**Multi-scale pooling (A-06)** divides the patch grid into s×s regions for each scale s, computes mask-weighted average in each region, then concatenates. For example, `pooling_scales=[1, 2, 4]` produces a `(1 + 4 + 16) × 768 = 16128`-dim vector.

**Multi-layer extraction (A-06)** concatenates features from multiple transformer layers (e.g. shallow layers capture texture/edges, deep layers capture semantics). For example, `feature_layers=[3, 7, 11]` on the default DINOv3-ViT-B/16 (`dinov3_vitb16`, 768-d) produces `3 × 768 = 2304`-dim features per patch.

### Preprocessing Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `extraction.preprocess.center_roi` | `bool` | `False` | Center crop on ROI centroid |
| `extraction.preprocess.center_roi_id` | `int` | `1` | ROI to center on |
| `extraction.preprocess.crop_width` | `int` | `300` | Crop width (px) |
| `extraction.preprocess.crop_height` | `int` | `300` | Crop height (px) |
| `extraction.preprocess.rotate_roi_tail` | `bool` | `False` | Normalize orientation via tail ROI |
| `extraction.preprocess.rotate_roi_tail_id` | `int` | `2` | Tail ROI ID |
| `extraction.preprocess.remove_background` | `bool` | `False` | Zero out non-ROI pixels |

### Tracking Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tracking.model` | `str` | `'r50_deaotl'` | DeAOT model variant |
| `tracking.smart_filter_ratio` | `float` | `0.1` | Mask area filter threshold |
| `tracking.batch_size` | `int` | `16` | Tracking batch size |

### Clustering Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `clustering.method` | `str` | `'dbscan'` | Clustering algorithm |
| `clustering.eps` | `float` | `1.0` | DBSCAN epsilon |
| `clustering.umap_stages` | `List[UMAPConfig]` | `[{n_neighbors:100, ..., standardize:true}]` | Multi-stage UMAP configs |

The stage-0 `UMAPConfig` defaults to `standardize: true` (per-feature z-score of the raw features). The top-level `master_seed` field (default `42`) seeds every stochastic component; resolved per-stage UMAP seeds are recorded in the session's `umap_log.jsonl` for reproducibility (see [UMAP Parameters](#umap-parameters)).

---

## Device Selection

CASTLE auto-detects the compute device:

| Platform | Device |
|----------|--------|
| macOS (Apple Silicon) | `mps` |
| Linux/Windows with NVIDIA GPU | `cuda` |
| No GPU | `cpu` |
