# Quick Start

!!! tip "Prerequisites"
    Complete the [Installation](installation.md) guide first, including downloading model checkpoints.

This guide walks you through a complete analysis using a demo video in under 5 minutes.

---

## 1. Launch CASTLE

=== "Gradio Web UI"

    ```bash
    python app.py
    ```

    The Gradio web UI opens at [http://localhost:7860](http://localhost:7860).

=== "PyQt6 Desktop App"

    ```bash
    castle gui
    # or
    python -m castle.desktop
    ```

    A native desktop window opens with 8 tabs.

---

## 2. Create a Project

1. In the **0. Project** tab, select **New Project**
2. Enter a project name (e.g., `my-first-project`)
3. Click **Create**

![Create a new project in the Project tab](../assets/screenshots/quickstart-create-project.png)

---

## 3. Upload a Video

1. Switch to the **1. Upload Videos** tab
2. Upload one of the included demo videos:
    - `demo/case1-reaching-task/reaching-task-raw.mp4` — rat reaching task
    - `demo/case2-openfield/openfield-1min-raw.mp4` — open field experiment
    - `demo/case5-gait/gait-raw.mp4` — gait analysis
    - `demo/Reach-and-Grasp-demo.mp4` — reach and grasp demo

![Upload a demo video](../assets/screenshots/quickstart-upload-video.png)

!!! tip
    Start with `reaching-task-raw.mp4` — it's short and works well for a first run.

---

## 4. Track ROIs

1. Switch to the **2. Tracking ROIs** tab
2. In the **Label ROI** sub-tab, define your regions of interest on the first frame
3. Use the **ROI Prompts** sub-tab to set prompts for the segmentation model
4. Go to **Tracking** to run the tracker across all frames

![Define and track ROIs](../assets/screenshots/quickstart-tracking.png)

---

## 4.5. Preprocessing (Optional)

Stabilized Camera Preprocessing normalises each frame around the tracked body centroid — removing camera drift and head orientation — to produce a clean 518×518 video optimal for DINOv2 feature extraction.

=== "Gradio Web UI"

    1. Switch to the **2. Tracking ROIs** tab
    2. Open the **Preprocessing** sub-tab (inside the Tracking tab)
    3. Select a video and enter the body ROI id and head ROI id
    4. Adjust filter parameters if needed (defaults work well for most videos)
    5. Click **Run Preprocessing**

=== "CLI"

    ```bash
    castle preprocess <project> \
        --video animal.mp4 --body-roi 1 --head-roi 2
    ```

    Full options:

    ```bash
    castle preprocess <project> \
        --video animal.mp4 --body-roi 1 --head-roi 2 \
        --fc 0.25 --order 2 --margin 75 \
        --min-crop 300 --output-size 518
    ```

    | Option | Default | Description |
    |--------|---------|-------------|
    | `--fc` | `0.25` | Low-pass cutoff frequency (Hz). Period ≈ 4 s. |
    | `--order` | `2` | Butterworth filter order |
    | `--margin` | `75` | Spatial margin around HP residual (px) |
    | `--min-crop` | `300` | Minimum crop side length (px) |
    | `--output-size` | `518` | Output frame side length (px) — 518 for DINOv2 ViT-B/14 |

!!! note "Output"
    The stabilised video is saved to `{storage}/{project}/preprocessed/{video}/stabilized.mp4`.
    A short 10-second preview clip is also generated at `stabilized_preview.mp4`.

!!! tip "When to use preprocessing"
    Use preprocessing when your animal moves freely in the arena and you want DINOv2 features
    to capture **posture** rather than position or orientation. Skip it for fixed-camera or
    already-cropped videos.

---

## 5. Extract Features

1. Switch to the **3. Extract Latent** tab
2. Select your tracked ROIs
3. Run feature extraction — this uses DINOv2/v3 to compute latent representations for each frame

![Extract latent features](../assets/screenshots/quickstart-extract.png)

!!! tip "Auto Batch Size & Cache (Phase 2)"
    CASTLE automatically selects a safe GPU batch size based on available VRAM, so you don't need to tune `batch_size` manually.
    If a GPU out-of-memory error occurs mid-run the batch is halved and the extraction retries transparently.

    Extraction results are also **cached by content hash** (video path + modification time + preprocessing config + model name).
    Re-running extraction on an unchanged video is nearly instant — the cached `.npz` is reused without re-inferring.

    To run extraction only on videos not yet processed (incremental mode):

    ```python
    from castle.service.incremental_service import get_unprocessed_videos

    pending = get_unprocessed_videos("/data/projects/my_project")
    # → only videos with no latent output yet
    ```

---

## 6. Analyze Behavior

1. Switch to the **4. Behavior Microscope** tab
2. Configure clustering parameters
3. Run the analysis to discover behavioral categories

![Behavior analysis results](../assets/screenshots/quickstart-analysis.png)

---

## 7. Annotate Clusters (Optional)

1. Within the **4. Behavior Microscope** tab, switch to the **Cluster Annotator** sub-tab
2. Select a clustering session and click **Load Cluster Data**
3. Browse grid videos for each cluster to verify its behavior
4. Assign behavior labels and (optionally) add a comment
5. Labels are **auto-saved** when changed or when the comment field loses focus

---

## 8. Export Results

1. Switch to the **6. Export** tab
2. Select the data components you want (masks, latent, cluster results, annotations, grid videos)
3. Click **Package ZIP** to download a ZIP archive

---

## What's Next?

- **[Tutorials](../tutorials/overview.md)** — Detailed step-by-step guides for each stage of the pipeline
- **[Technical Guide](../technical/architecture.md)** — Understand how CASTLE works under the hood
- **[Configuration](../technical/configuration.md)** — Customize parameters for your experiments
