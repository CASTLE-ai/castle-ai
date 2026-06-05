# Step 3: Extract Latent Features

The **4. Extract Latent** tab uses visual foundation models to extract feature representations from tracked ROIs. These features encode the animal's posture and movement in each frame as a high-dimensional vector.

---

## Overview

Feature extraction transforms your tracked video into numerical data suitable for clustering:

```
Tracked Video (frames + masks) → Preprocessing → Visual Model → Latent Vectors (.npz)
```

---

## Configuration

When you switch to the Extract Latent tab, the interface shows three columns:

### Model & Target Settings (Left Column)

| Parameter | Description | Default |
|-----------|-------------|---------|
| **Select Visual Model** | Feature extraction backbone | `dinov3_vitb16` |
| **Enter ROI ID** | Which tracked ROI to extract features from | `1` |
| **Batch size** | Frames processed per batch (increase if VRAM allows) | `32` |
| **Select Target Video** | Specific video or "All" | `All` |
| **Skip existing files** | Don't re-extract if output already exists | ✅ Enabled |

Available models:

- **`dinov3_vitb16`** — Meta's DINOv3 ViT-B/16 (default, 768-dim output)
- **`dinov3_vitl16`** — DINOv3 ViT-L/16 (larger model, 1024-dim, higher quality but slower)
- **`dinov2_vitb14_reg4_pretrain`** — DINOv2 ViT-B/14 with registers (768-dim, well-tested alternative)

### Preprocessing Settings (Middle Column)

These settings control how frames are preprocessed before feature extraction:

| Parameter | Description | Default |
|-----------|-------------|---------|
| **Center ROI** | Crop frames centered on a reference ROI | `False` |
| **Center ROI ID** | Which ROI to center on | `1` |
| **Width / Height** | Crop dimensions in pixels | `300 × 300` |
| **Rotate based on Tail** | Normalize orientation using a tail ROI | `False` |
| **Tail ROI ID** | Which ROI defines the tail direction | `2` |
| **Remove Background** | Mask out pixels outside the ROI | `False` |

!!! warning "Click Apply First"
    After changing preprocessing settings, you **must click the Apply button** before extracting. The preview image shows the result of your preprocessing configuration on the first frame.

![Preprocessing preview](../assets/screenshots/tutorial-step3-preprocess.png)

### Preprocessing Recommendations

| Scenario | Center ROI | Rotate | Remove BG |
|----------|-----------|--------|-----------|
| **General behavior** | ✅ On | ❌ Off | ❌ Off |
| **Posture analysis** | ✅ On | ✅ On | ✅ On |
| **Locomotion patterns** | ✅ On | ✅ On | ❌ Off |
| **No preprocessing** | ❌ Off | ❌ Off | ❌ Off |

---

## Extraction Types

CASTLE offers three extraction modes, each triggered by a different button:

### Extract (Standard Latent Extraction)

The primary extraction mode. Runs the selected visual model on preprocessed frames and saves latent vectors.

- **Output**: `.npz` file in `project/latent/model-name/`
- **Filename pattern**: `{video}_ROI_{id}_{model}_{tags}.npz`
- **Tags**: `ctr` (centered), `rmbg` (background removed)

### Extract Crop Video

Exports the preprocessed (centered, rotated, cropped) video as an MP4 file. Useful for:

- Visual verification of preprocessing
- Sharing aligned videos with collaborators
- Input to external analysis tools

- **Output**: `.mp4` file in `project/crop/video-name/`

### Extract Rotation Latent

Extracts features specifically capturing rotational information. Used when orientation is a key behavioral variable.

- **Output**: `.npz` file with rotation-specific features

---

## Running Extraction

1. Configure model, ROI, and preprocessing settings
2. Click **Apply** to confirm preprocessing
3. Click **Extract** (or the appropriate extraction button)
4. Monitor progress in the log output area

The log shows:

- Pre-flight check (which videos need processing)
- Per-video progress
- Final summary with success/failure counts

![Extraction progress](../assets/screenshots/tutorial-step3-extract.png)

!!! tip "Multi-GPU extraction (opt-in)"
    On a machine with two or more CUDA GPUs you can speed up extraction by setting the environment variable `CASTLE_MULTI_GPU=1` in your environment before launching CASTLE (the Gradio app and the CLI both honour it). CASTLE then splits a single video's frames by range across the available GPUs (each GPU runs the full decode → preprocess → encode on its half, and the results are merged in original order). On two identical GPUs this is bit-identical to the single-GPU output and roughly **1.9× faster**. The feature is **off by default** (single GPU).

    ```bash
    CASTLE_MULTI_GPU=1 python app.py
    ```

---

## Output Format

The standard latent extraction produces `.npz` files containing:

```python
import numpy as np

data = np.load('video_ROI_1_dinov3_vitb16.npz')
latent_vectors = data['latent']  # Shape: (n_frames, feature_dim)
```

- **Feature dimension**: depends on the model (768 for ViT-B, 1024 for ViT-L)
- **NaN values**: frames where the ROI mask was empty produce NaN vectors

---

## Processing Time

Processing time depends on video length, GPU, and model size.

**Approximate benchmarks** (RTX 4090, 720×720 @ 30fps, batch_size=5):

| Video Length | DINOv3 ViT-B (per ROI) | DINOv3 ViT-L (per ROI) |
|-------------|----------------------|----------------------|
| 10 min | ~4 min | ~12 min |
| 30 min | ~12 min | ~36 min |
| 60 min | ~24 min | ~72 min |

With 7 ROIs on a 30-min video, DINOv3 ViT-B extraction takes ~84 min total.

!!! tip "Speed Tips"
    - Increase **batch size** if you have spare VRAM (e.g., 64 or 128 on a 24 GB card)
    - Use **Skip existing** when re-running after adding new videos
    - The ViT-B models (`dinov3_vitb16`, `dinov2_vitb14_reg4_pretrain`) are fastest; `dinov3_vitl16` is slowest but potentially highest quality
    - On a multi-GPU machine, set `CASTLE_MULTI_GPU=1` for roughly 1.9× faster extraction on 2 GPUs
    - Feature extraction is the pipeline bottleneck — plan accordingly for large datasets

---

## Next Step

Once features are extracted for all videos, proceed to [**Step 4: Behavior Analysis**](step4-analysis.md).
