# Workflow Overview

## What CASTLE Does

CASTLE (**Combined Approach for Segmentation and Tracking with Latent Extraction**) takes raw video of animal behavior and produces:

- **Segmented and tracked regions of interest (ROIs)** — precise masks of animals or body parts across all frames
- **Visual feature representations** — high-dimensional latent vectors capturing posture and movement
- **Behavioral clusters** — unsupervised discovery of behavioral syllables
- **Visualizations** — UMAP embeddings, ethograms, and cluster summaries for exploration and publication

All of this is achieved **without any training data** — CASTLE leverages pretrained foundation models to work out of the box on any species or experimental setup.

---

## The Pipeline

```
Raw Video → SAM (segment) → DeAOT (track) → Preprocess (stabilize) → DINOv3 (features) → UMAP + DBSCAN (cluster)
```

![CASTLE Pipeline Flowchart](../assets/screenshots/flowchart.png)

### 1. Segmentation (SAM)

The user marks regions of interest on a reference frame — clicking on the animal's body, head, or other features. The **Segment Anything Model (SAM)** generates precise segmentation masks from these clicks.

- Point-and-click interface: click to add, click to remove
- Multiple ROIs per frame (e.g., body centroid + head + tail)
- Labels are saved as `.npz` files containing frame and mask data

### 2. Tracking (DeAOT)

The initial masks are propagated across all video frames using **DeAOT** (Decoupling features in Associating Objects with Transformers).

- Two model options: **R50** (faster) and **SwinB** (more accurate)
- Handles occlusion, deformation, and appearance changes
- Real-time progress monitoring with cancel capability
- Iterative refinement: add labels on failure frames and re-track

### 2.5. Stabilized Camera Preprocessing *(optional — Phase 0)*

Before feature extraction, the **StabilizedCamera** module normalises each frame around the tracked animal to eliminate camera drift and orientation variation:

- **Zero-phase Butterworth low-pass filter** (`filtfilt`, fc = 0.25 Hz, order 2) applied to centroid x(t) and heading angle θ(t) — no temporal delay
- **Dynamic crop window**: `max(300, 2 × (‖residual‖ + 75))` px, adapting to fast movements
- **Output**: 592×592 px MP4 saved to `preprocessed/{video}/stabilized.mp4`
- Available via CLI (`castle preprocess`) and Gradio UI (Tab 3: Pre-process)

This ensures that the extracted features encode **posture and movement** rather than arena position or heading direction.

### 3. Video Alignment

Before feature extraction, tracked ROIs are preprocessed:

- **Center ROI**: crop the video around a reference ROI (e.g., body centroid)
- **Rotate**: normalize orientation using a secondary ROI (e.g., tail direction)
- **Remove background**: mask out non-ROI pixels

This normalization ensures that features reflect **posture and movement**, not position or orientation in the frame.

### 4. Feature Extraction (DINOv3)

Visual foundation models extract latent features from each aligned frame:

- **DINOv3 ViT-B/16** (`dinov3_vitb16`, 768-d) — Meta's self-supervised vision transformer (**default**)
- **DINOv3 ViT-L/16** (`dinov3_vitl16`, 1024-d) — larger model with richer representations
- **DINOv2 ViT-B/14** (`dinov2_vitb14_reg4_pretrain`, 768-d) — still supported as a selectable alternative
- Each frame produces a high-dimensional feature vector
- ROI masking ensures only the animal contributes to the representation
- Batch processing with configurable batch size

!!! tip "Multi-GPU extraction (opt-in)"
    Set `CASTLE_MULTI_GPU=1` in your environment before running extraction. When two or more CUDA GPUs are present, latent extraction for a single video splits its frames by range across the GPUs (each runs the full decode → preprocess → encode on its half; results are merged in original order). Output is bit-identical to the single-GPU path on identical GPUs and roughly **1.9× faster on 2 GPUs**. Default is off (single GPU).

### 5. Behavior Analysis (UMAP + DBSCAN)

The high-dimensional features are reduced and clustered to discover behavioral patterns:

- **UMAP** (Uniform Manifold Approximation and Projection) reduces dimensions for visualization
- **DBSCAN** clusters the embedding into behavioral syllables
- **Hierarchical exploration**: three magnification levels (low → intermediate → high) for progressively finer behavioral categories
- Interactive click-to-explore on the UMAP plot

CASTLE is a **human-in-the-loop** tool: cluster boundaries and labels must be reviewed by the user in the Behavior Microscope tab before they are scientifically meaningful. There is intentionally no "one-click cluster" entry point.

!!! note "Reproducibility & standardization"
    Input **standardization (per-feature z-score) is on by default** for the first (raw-feature) UMAP stage (`"standardize": true` in the default UMAP config). It improves cluster separation but changes embeddings relative to older runs, so the DBSCAN `eps` may need re-tuning; it is configurable in the UMAP config JSON. Every UMAP run also records its resolved random seed, and each clustering session writes a `umap_log.jsonl` file (one JSON line per UMAP stage with seed + config). Reuse the logged seed — via the CPU/deterministic path — to reproduce an embedding exactly.

### 6. Cluster Annotator

After clustering, annotate discovered clusters with behavior labels:

- **Grid video browser** — watch representative video clips for each cluster
- **Per-session annotations** — label name and comment saved per clustering session
- **Auto-save** — labels and comments saved automatically on change / focus-out
- **Mask contour overlay** — ROI contours drawn over video frames

### 7. Downstream Analysis

- **Ethogram** — raster plot, transition matrix heatmap, bout duration distributions, written **per video** (one ethogram per animal/video)
- **Quality Metrics** — silhouette score, Calinski-Harabasz, Davies-Bouldin, V-measure, NMI, ARI
- **Group Comparison** — behavioral fingerprint radar, BFA test, energy distance, permutation tests

### 8. Export

Package results as a ZIP archive with selectable components (masks, latent features, cluster results, annotations, grid videos, analysis outputs).

---

## GUI vs Programmatic

CASTLE offers two interfaces:

### Gradio Web UI (Recommended for exploration)

```bash
python app.py
```

Interactive web interface at `http://localhost:7860` with **8 tabs** following the pipeline:

| Tab | Name | Purpose |
|-----|------|---------|
| 0 | Project | Create / open projects |
| 1 | Upload Videos | Import video files |
| 2 | Tracking ROIs | SAM labeling + DeAOT tracking + batch |
| 3 | Pre-process (Optional) | Stabilized-camera preprocessing |
| 4 | Extract Latent | DINOv3 feature extraction (DINOv2 still selectable) |
| 5 | Behavior Microscope | UMAP + DBSCAN clustering + Cluster Annotator |
| 6 | Analysis | Ethogram, Quality Metrics, Group Comparison |
| 7 | Export | ZIP download with selectable data components |

### Jupyter Notebooks

The `notebooks/` directory contains step-by-step notebooks for programmatic use:

| Notebook | Description |
|----------|-------------|
| `step1_image_segment.ipynb` | Interactive segmentation |
| `step2_video_segment.ipynb` | Video tracking |
| `step3_video_align.ipynb` | Video alignment |
| `step4_latent_extraction.ipynb` | Feature extraction |
| `step5_latent_explore.ipynb` | UMAP + clustering |

Best for custom workflows, batch processing, or integration with other analysis pipelines.

---

## Next Steps

Follow the tutorials in order:

1. [**Create Project & Upload Videos**](step1-project.md)
2. [**Track ROIs**](step2-tracking.md)
3. [**Stabilized Camera Preprocessing**](step2_5-preprocessing.md) *(optional)*
4. [**Extract Features**](step3-extract.md)
5. [**Behavior Analysis**](step4-analysis.md)
6. [**Export Results**](step5-export.md)

!!! note "Coming soon"
    Dedicated tutorials for the Cluster Annotator and Analysis tabs are planned.
