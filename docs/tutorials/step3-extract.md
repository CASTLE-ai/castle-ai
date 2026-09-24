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

When you switch to the Extract Latent tab, the interface shows a **Pre-process Session** selector followed by the **Model** and **Source & Options** sections:

### Model & Target Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| **Pre-process Session** | Session from [Step 2.5](step2_5-preprocessing.md) to extract from, or the raw source video | `(None — use raw source)` |
| **Visual Model** | Feature extraction backbone | `dinov3_vitb16` |
| **ROI ID** | Which tracked ROI to extract features from | `1` |
| **Batch Size** | Frames processed per batch (increase if VRAM allows; **Auto Batch Size** picks one) | `32` |
| **Videos to extract** | Which project videos to process | All checked |
| **Skip existing files** | Don't re-extract if output already exists | ✅ Enabled |
| **Remove Background** | Mask out pixels outside the ROI | `False` |

Available models:

- **`dinov3_vitb16`** — Meta's DINOv3 ViT-B/16 (default, 768-dim output)
- **`dinov3_vitl16`** — DINOv3 ViT-L/16 (larger model, 1024-dim, higher quality but slower)
- **`dinov2_vitb14_reg4_pretrain`** — DINOv2 ViT-B/14 with registers (768-dim, well-tested alternative)

### Preprocessing Settings

Centering, cropping and rotation are not set on this tab. They are produced in the **3. Pre-process (Optional)** tab as a session (**KIT** or **Center ROI + Crop**, see [Step 2.5](step2_5-preprocessing.md)) and applied here by choosing that session in **Pre-process Session**.

![Preprocessing preview](../assets/screenshots/tutorial-step3-preprocess.png)

### Preprocessing Recommendations

| Scenario | Pre-process Session | Remove BG |
|----------|---------------------|-----------|
| **General behavior** | Center ROI + Crop | ❌ Off |
| **Posture analysis** | KIT (centered + rotated) | ✅ On |
| **Locomotion patterns** | KIT (centered + rotated) | ❌ Off |
| **No preprocessing** | None (raw source) | ❌ Off |

---

## Extraction Types

CASTLE offers two extraction modes:

### Extract (Standard Latent Extraction)

The primary extraction mode. Runs the selected visual model on preprocessed frames and saves latent vectors.

- **Output**: `.npz` file in `project/latent/model-name/`
- **Filename pattern**: `{video}_ROI_{id}_{model}_{tags}_pre-{session_id}.npz` (`_{tags}` only when tags apply; `_pre-{session_id}` only when a pre-process session is selected)
- **Tags**: `ctr` (centered), `rmbg` (background removed), `spp…` (multiscale pooling), `L…` (feature layers)

### Extract Rotation Latent

Enabled with **Eliminate Rotation Asymmetry** under **Advanced Extraction Options**; runs after the standard extraction. Embeds 7 rotated views of each frame and averages them to reduce orientation bias in the latent space.

- **Output**: `{video}_ROI_{id}_rotation_latent.npz` in `project/latent/model-name/` (with the same `_pre-{session_id}` suffix when a session is selected)

---

## Running Extraction

1. Select the pre-process session, model, ROI, and options
2. Click **Extract**
3. Monitor progress in the **Log Output** area

The log shows:

- Pre-flight check (which videos need processing)
- Per-video progress
- Final summary with success/failure counts

![Extraction progress](../assets/screenshots/tutorial-step3-extract.png)

!!! tip "Multi-GPU extraction"
    On a machine with two or more CUDA GPUs, check **Use multiple GPUs** on the Extract Latent tab (enabled and checked by default only when two or more GPUs are detected). With several videos selected, each GPU processes whole videos; with a single video, CASTLE splits its frames by range across the available GPUs (each GPU runs the full decode → preprocess → encode on its part, and the results are merged in original order). On two identical GPUs this is bit-identical to the single-GPU output and roughly **1.9× faster**. The CLI stays single-GPU unless the environment variable `CASTLE_MULTI_GPU=1` is set.

    ```bash
    CASTLE_MULTI_GPU=1 castle extract my_project
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
    - On a multi-GPU machine, keep **Use multiple GPUs** checked (CLI: set `CASTLE_MULTI_GPU=1`) for roughly 1.9× faster extraction on 2 GPUs
    - Feature extraction is the pipeline bottleneck — plan accordingly for large datasets

---

## Next Step

Once features are extracted for all videos, proceed to [**Step 4: Behavior Analysis**](step4-analysis.md).
