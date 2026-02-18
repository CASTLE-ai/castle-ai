# Step 4: Behavior Analysis

The **4. Behavior Microscope** tab is where CASTLE's core analysis happens — transforming latent features into interpretable behavioral categories through dimensionality reduction and clustering.

The tab contains two sub-tabs:

- **Clustering** — UMAP + DBSCAN workspace
- **Cluster Annotator** — grid video browser with behavior labeling, comments, and auto-save

---

## Overview

The analysis workflow:

```
Latent Vectors → Initialize → Select Cluster → UMAP Embedding → DBSCAN Clustering → Label & Submit
                                                                                           ↓
                                                              Cluster Annotator (grid video, label, comment)
```

This is an **iterative, hierarchical** process. You start with broad categories (low magnification) and progressively zoom in to discover finer behavioral syllables.

!!! tip "Automated alternative"
    For large datasets, use `castle cluster auto <project>` to run recursive hierarchical clustering fully automatically from the CLI — no manual UMAP/DBSCAN interaction required.

---

## Getting Started

### Initialize

1. Switch to the **4. Behavior Microscope** tab
2. In the **Input Setting** accordion, configure:

    | Parameter | Description | Default |
    |-----------|-------------|---------|
    | **Select Visual Model** | Must match the model used in Step 3 | `dinov3_vitb16` |
    | **Enter ROI ID** | Comma-separated list (e.g., `1` or `1,2,3`) | `1` |
    | **Time window (frame)** | Number of frames to aggregate per data point | `1` |

3. Click **Initialize**

!!! note "Time Window"
    A time window of `1` means each data point represents a single frame. Higher values (e.g., `5` or `10`) aggregate consecutive frames, which can smooth noise and capture temporal patterns but reduces temporal resolution.

---

## UMAP Configuration

CASTLE provides **magnification presets** that control how the UMAP dimensionality reduction is performed. The key idea: different `n_neighbors` values reveal structure at different scales.

### Presets

#### Low Magnification (Single-Stage UMAP)

Broad behavioral categories. Single UMAP step reducing directly to 2D.

Available presets with different `n_neighbors` values:

| Preset | n_neighbors | Use Case |
|--------|------------|----------|
| Low-magnification objective 1000 | 1000 | Very broad categories, large datasets |
| Low-magnification objective 500 | 500 | Broad categories |
| Low-magnification objective 300 | 300 | Moderate categories |
| Low-magnification objective 100 | 100 | Default starting point |
| Low-magnification objective 50 | 50 | Finer categories |
| Low-magnification objective 25 | 25 | Fine categories, small datasets |

Configuration format:
```json
[
    {
        "n_neighbors": 100,
        "min_dist": 0.0,
        "n_components": 2
    }
]
```

#### Intermediate Magnification (Two-Stage UMAP)

Two-step reduction: first to 5D, then to 2D. Captures more structure than single-stage.

| Preset | Stage 1 n_neighbors | Stage 2 n_neighbors |
|--------|--------------------|--------------------|
| Intermediate (1000, 500) | 1000 | 500 |
| Intermediate (500, 300) | 500 | 300 |
| Intermediate (300, 100) | 300 | 100 |
| Intermediate (100, 50) | 100 | 50 |
| Intermediate (50, 25) | 50 | 25 |

Configuration format:
```json
[
    {"n_neighbors": 300, "min_dist": 0.0, "n_components": 5},
    {"n_neighbors": 100, "min_dist": 0.0, "n_components": 2}
]
```

#### High Magnification (Two-Stage, Higher Initial Dimension)

Two-step reduction: first to 10D, then to 2D. Preserves the most structure for fine-grained analysis.

| Preset | Stage 1 n_neighbors | Stage 2 n_neighbors |
|--------|--------------------|--------------------|
| High (1000, 500) | 1000 | 500 |
| High (500, 300) | 500 | 300 |
| High (300, 100) | 300 | 100 |
| High (100, 50) | 100 | 50 |
| High (50, 25) | 50 | 25 |

Configuration format:
```json
[
    {"n_neighbors": 300, "min_dist": 0.0, "n_components": 10},
    {"n_neighbors": 100, "min_dist": 0.0, "n_components": 2}
]
```

### Custom Configuration

You can edit the UMAP config JSON directly for full control. The format is a list of UMAP stages, each with:

- `n_neighbors`: number of nearest neighbors (larger = broader structure)
- `min_dist`: minimum distance between points in embedding (0.0 for clustering)
- `n_components`: output dimensions for that stage

---

## Running the Analysis

### 1. Generate Embedding

1. Select a cluster from the **Select Cluster** dropdown (starts with `init` — the full dataset)
2. Choose a UMAP preset or edit the config manually
3. Click **Generate Embedding**

The UMAP scatter plot appears on the right. Each point represents a data point (frame or time window).

![UMAP embedding](../assets/screenshots/tutorial-step4-umap.png)

!!! tip "Interactive Exploration"
    Click on any point in the UMAP plot to see the corresponding video frame. This helps you understand what each region of the embedding represents.

### 2. Cluster the Embedding

1. Set the **epsilon-neighborhood radius** (eps) — controls cluster granularity
    - Smaller eps → more clusters (finer categories)
    - Larger eps → fewer clusters (broader categories)
    - Range: 0.1 to 10.0 (default: 1.0)
2. Click **Generate Cluster**

The plot updates with colors indicating cluster assignments.

### 3. Label Clusters

For each cluster you want to name:

1. Enter the **Cluster ID** (number shown in the plot)
2. Enter a **Cluster Name** (e.g., "grooming", "rearing", "locomotion")
3. Click **Enter**

!!! tip
    Click on points within each cluster to view representative frames. This helps you identify what behavior each cluster represents.

### 4. Submit

Click **Submit** to:

- Import the labeled clusters into the main analysis
- Generate a syllable plot (ethogram)
- Export CSV files (behavior IDs and time series)
- Generate SRT subtitle files for video overlay
- Save the embedding data

---

## Hierarchical Analysis

The power of CASTLE's "Behavior Microscope" comes from iterative refinement:

1. **Start broad**: use Low Magnification to identify major behavioral categories
2. **Zoom in**: select a specific cluster, then re-run UMAP at Intermediate or High Magnification
3. **Refine**: each cluster can be further subdivided into finer syllables
4. **Repeat**: continue until you reach the desired granularity

This mirrors how a microscope works — you start with low magnification to find areas of interest, then zoom in for detail.

![Hierarchical classification](../assets/screenshots/tutorial-step4-hierarchy.png)

---

## Outputs

After submitting, the following are generated:

| Output | Format | Description |
|--------|--------|-------------|
| **Syllable Plot** | Interactive plot | Timeline of behavioral states |
| **Behavior ID CSV** | `.csv` | Mapping of cluster IDs to names |
| **Time Series CSV** | `.csv` | Frame-by-frame cluster assignments |
| **SRT Subtitles** | `.srt` | Behavioral labels as video subtitles |
| **Embedding NPZ** | `.npz` | UMAP coordinates and cluster labels |

---

## Cluster Annotator Sub-tab

After generating clusters, use the **Cluster Annotator** sub-tab to review and label them:

1. Switch to the **Cluster Annotator** sub-tab within **4. Behavior Microscope**
2. Select the clustering session from the dropdown and click **Load Cluster Data**
3. A list of clusters appears on the left panel
4. Click a cluster to load its **grid video** — a mosaic of representative video clips

### Labeling

- Enter a **behavior name** in the label field (replaces the default cluster name)
- Optionally add a **comment** to describe the behavior or note uncertainty
- Labels are **auto-saved** immediately on change; comments are auto-saved on focus-out

### Mask Contour Overlay

When tracking data is available, ROI mask contours are drawn over the video frames so you can confirm the tracked region corresponds to the animal.

### Speed Control (Desktop App)

The PyQt6 desktop app includes a **playback speed** selector (0.25×, 0.5×, 1×, 2×) for reviewing grid videos.

---

## Automated Clustering (CLI)

For headless pipelines, the recursive auto-clustering command runs the full hierarchical Behavior Microscope automatically:

```bash
castle cluster auto <project> \
    --max-depth 6 \
    --min-frames 100 \
    --eps 0.3,0.5,0.7,1.0,1.5,2.0,3.0
```

This mirrors the manual workflow — `select cluster → UMAP → DBSCAN → split → recurse` — but runs unattended. UMAP config is auto-selected per depth (Low at depth 0, Intermediate at depth 1+).

---

## Tips

- **Start with Low Magnification 100** as your first exploration
- **eps = 1.0** is a good starting point for clustering
- If clusters are too noisy, try a **larger n_neighbors** value
- If behaviors are merged together, try **higher magnification** or **smaller eps**
- Use **multiple videos** for more robust clustering — single-video embeddings can overfit to that animal's quirks
- The first round of labeling (cold start) takes ~30 min of interactive exploration; subsequent rounds are much faster (~3 min) once you know what to look for
- Save your UMAP settings once you find a configuration that works for your experimental paradigm — you can reuse them across projects
- If the UMAP plot looks like a single blob with no structure, try a **smaller n_neighbors** (e.g., 25–50) or switch to **Intermediate/High magnification**

---

## Next Step

After labeling clusters, explore downstream analysis in **5. Analysis** (Ethogram, Quality Metrics, Group Comparison), then proceed to [**Step 5: Export Results**](step5-export.md).
