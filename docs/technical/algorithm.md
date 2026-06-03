# Algorithm & Methodology

This page describes each algorithm in the CASTLE pipeline, how CASTLE uses it, and provides references to the original papers. For a conceptual overview, see the [Workflow Overview](../tutorials/overview.md).

---

## 1. Image Segmentation — SAM

### What It Does

The **Segment Anything Model (SAM)** generates pixel-level segmentation masks from minimal user input (point clicks or bounding boxes). It is a foundation model trained on 11 million images and 1 billion masks.

### CASTLE's Usage

- **Model variant**: ViT-B (`vit_b`)
- **Checkpoint**: `sam_vit_b_01ec64.pth`
- **Interface**: Point-and-click in the Label ROI sub-tab
- **Implementation**: `castle/utils/image_segment.py` → `Segmentor` class

The `Segmentor` wraps SAM's `SamAutomaticMaskGenerator` and interactive predictor:

1. User clicks on the frame → point coordinates sent to SAM
2. SAM returns a binary mask for the clicked object
3. Multiple clicks refine the mask (add/remove regions)
4. Multiple ROIs can be labeled sequentially (e.g., body, head, tail)

**Auto-mask generator parameters** (from `model_config.json`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `points_per_side` | 16 | Grid density for auto-segmentation |
| `pred_iou_thresh` | 0.8 | IoU prediction threshold |
| `stability_score_thresh` | 0.9 | Mask stability threshold |
| `crop_n_layers` | 1 | Number of crop layers |
| `min_mask_region_area` | 200 | Minimum mask area in pixels |

### Reference

> Kirillov, A., et al. (2023). *Segment Anything.* ICCV 2023.
> [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)

---

## 2. Video Object Tracking — DeAOT

### What It Does

**DeAOT** (Decoupling features in Associating Objects with Transformers) propagates segmentation masks from reference frames across an entire video. It handles object deformation, occlusion, and appearance changes.

### CASTLE's Usage

- **Model variants**:
    - `R50_DeAOTL` — ResNet-50 backbone (faster)
    - `SwinB_DeAOTL` — Swin Transformer-B backbone (more accurate)
- **Checkpoints**: `R50_DeAOTL_PRE_YTB_DAV.pth`, `SwinB_DeAOTL_PRE_YTB_DAV.pth`
- **Pretrained on**: YouTube-VOS + DAVIS datasets
- **Implementation**: `castle/utils/tracking_manager.py` → `ROITracker` class

The tracking process:

1. Load all ROI labels (frame + mask pairs) from the project
2. Initialize DeAOT with the selected model
3. For each frame in the specified range:
    - If labels exist for this frame, inject them
    - Otherwise, propagate masks from previous frame
4. Store masks in HDF5 format (`mask_list.h5`) via `H5IO`

**Iterative refinement workflow**: If tracking fails at frame N, the user labels frame N and re-runs tracking starting from that frame. This progressively improves prompt diversity.

### Reference

> Yang, L., et al. (2022). *Decoupling Features in Hierarchical Propagation for Video Object Segmentation.* NeurIPS 2022.
> [arXiv:2210.09782](https://arxiv.org/abs/2210.09782)

---

## 3. Video Alignment

### What It Does

Normalizes ROI position and orientation across frames so that features reflect **posture and movement**, not location in the frame.

### CASTLE's Usage

- **Implementation**: `castle/utils/video_align.py`

Available transformations:

| Function | Description |
|----------|-------------|
| `center_roi(frame, mask, roi_color)` | Translates frame so the ROI centroid is at the image center |
| `rotate_based_on_roi_closest_center_point(frame, mask, roi_color)` | Rotates frame so the closest contour point of a secondary ROI (e.g., tail) points upward |
| `crop(frame, crop_h, crop_w)` | Crops a fixed-size window centered on the frame |

The alignment pipeline (configured in the Extract Latent tab):

1. **Center** — find the largest connected component of the specified ROI, compute its centroid, translate the frame
2. **Rotate** — using a secondary ROI (e.g., tail), compute the angle from center to the closest contour point, rotate to normalize orientation
3. **Crop** — extract a fixed-size window (default 300×300 pixels)
4. **Remove background** (optional) — zero out pixels outside the ROI mask

---

## 4. Visual Feature Extraction — DINOv2 / DINOv3

### What It Does

Self-supervised Vision Transformers (ViT) extract rich visual features without task-specific training. Each frame is converted to a high-dimensional feature vector that captures visual patterns.

### CASTLE's Usage

- **Default model**: DINOv3 ViT-B/16 (`dinov3_vitb16`, 768-dim output)
- **Models**:

    | Model | Checkpoint | Feature Dim | |
    |-------|-----------|-------------|---|
    | DINOv3 ViT-B/16 (`dinov3_vitb16`) | `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` | 768 | **default** |
    | DINOv3 ViT-L/16 (`dinov3_vitl16`) | `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | 1024 | |
    | DINOv2 ViT-B/14 with registers (`dinov2_vitb14_reg4_pretrain`) | `dinov2_vitb14_reg4_pretrain.pth` | 768 | |

    All three are selectable in the Extract Latent tab. The default `dinov3_vitb16` and the DINOv2 option both produce 768-dim vectors; the larger `dinov3_vitl16` produces 1024-dim vectors.

- **Implementation**: `castle/core/extractor.py` + `castle/core/models.py`

Extraction process:

1. Load aligned video frames as a `VideoDataset` (PyTorch `Dataset`)
2. Batch frames through the visual encoder (default batch size: 32)
3. Extract CLS token or pooled features per frame
4. Save as `.npz` file with shape `(n_frames, feature_dim)`
5. Frames with empty masks produce NaN vectors

!!! tip "Optional multi-GPU extraction"
    Set the environment variable `CASTLE_MULTI_GPU=1` in your environment before running extraction. When it is set **and** at least two CUDA GPUs are present, latent extraction for a single video splits that video's frames by range across the GPUs — each GPU runs the full decode → preprocess → encode on its half, and the results are merged back in original frame order. Output is bit-identical to the single-GPU result on identical GPUs and runs ~1.9× faster on two GPUs. The default is OFF (single GPU).

### Reference

> Oquab, M., et al. (2024). *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2024.
> [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)

---

## 5. Dimensionality Reduction — UMAP

### What It Does

**UMAP** (Uniform Manifold Approximation and Projection) reduces high-dimensional data to 2D for visualization while preserving local and global structure.

### CASTLE's Usage

- **Implementation**: `castle/utils/myumap.py` — custom UMAP using cuml (GPU-accelerated) with spectral layout initialization
- **Called from**: `castle/utils/latent_explorer.py` → `Latent.build_embedding()`

CASTLE's UMAP implementation uses:

- **cuml** `fuzzy_simplicial_set` for graph construction
- **Spectral layout** or **PCA** for initialization
- **cuml** `simplicial_set_embedding` for optimization
- Default: 20,000 epochs for convergence

**Hierarchical multi-stage UMAP**: CASTLE supports chaining multiple UMAP stages to progressively reduce dimensions:

| Magnification | Stages | Dimensions |
|--------------|--------|------------|
| Low | 1 stage | → 2D |
| Intermediate | 2 stages | → 5D → 2D |
| High | 2 stages | → 10D → 2D |

The `n_neighbors` parameter controls the scale of structure preserved — higher values capture more global patterns, lower values capture finer local details.

**Input standardization (default ON)**: The first (raw-feature) UMAP stage now applies per-feature z-score standardization by default — the default UMAP config preset has `"standardize": true`. This improves cluster separation, but it **changes embeddings relative to older runs**, so the DBSCAN `eps` may need re-tuning. To opt out, set `"standardize": false` in the stage-0 UMAP config.

!!! note "Reproducibility"
    Every UMAP run records its resolved random seed. For each clustering session, CASTLE writes a `umap_log.jsonl` file with one JSON line per UMAP stage, capturing the seed and that stage's config. To reproduce an embedding exactly, reuse the logged seed (run on the CPU/deterministic path for bit-identical results; GPU optimization can still vary slightly even with a fixed seed).

### Reference

> McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.*
> [arXiv:1802.03426](https://arxiv.org/abs/1802.03426)

---

## 6. Clustering — DBSCAN

### What It Does

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) discovers clusters of arbitrary shape based on point density. Points in sparse regions are labeled as noise (-1).

### CASTLE's Usage

- **Implementation**: `castle/utils/latent_explorer.py` → `Latent.build_cluster()`
- **Key parameter**: `eps` (epsilon-neighborhood radius)
    - Smaller eps → more, smaller clusters
    - Larger eps → fewer, larger clusters
    - Typical range: 0.1–10.0 (default: 1.0)

Clustering is applied to the 2D UMAP embedding, not the original high-dimensional features. This makes it fast and allows interactive parameter tuning.

### Reference

> Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). *A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise.* KDD 1996.
