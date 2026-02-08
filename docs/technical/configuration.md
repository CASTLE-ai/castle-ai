# Configuration Reference

All configurable parameters in CASTLE, consolidated in one place.

---

## Launch Options (`app.py`)

```bash
python app.py [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--root PATH` | Custom project storage path | `projects/` |
| `--share` | Enable Gradio public URL (auto-enabled on Colab) | `False` |

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
| Visual Model | `dinov2_vitb14_reg4_pretrain` | 3 options | Feature extraction backbone |
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
        "n_components": 2
    }
]
```

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `n_neighbors` | Nearest neighbors (local vs global structure) | 25–1000 |
| `min_dist` | Minimum distance in embedding | 0.0 (for clustering) |
| `n_components` | Output dimensions per stage | 2, 5, or 10 |

**Internal parameter** (in `myumap.py`): `n_epochs = 20000` — number of optimization epochs. Not exposed in the UI.

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

---

## Device Selection

CASTLE auto-detects the compute device:

| Platform | Device |
|----------|--------|
| macOS (Apple Silicon) | `mps` |
| Linux/Windows with NVIDIA GPU | `cuda` |
| No GPU | `cpu` |
