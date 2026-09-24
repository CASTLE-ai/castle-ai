# Quick Start

!!! tip "Prerequisites"
    Complete the [Installation](installation.md) guide first, including downloading model checkpoints.

This guide walks you through a complete analysis using a demo video in under 5 minutes.

---

## 1. Launch CASTLE

=== "Gradio Web UI"

    From the CASTLE install folder:

    ```bash
    ./.venv/bin/python app.py              # macOS / Linux
    .\.venv\Scripts\python.exe app.py      # Windows (PowerShell)
    ```

    Then open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

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
3. Use the **ROI Prompts** sub-tab to review (or delete) the labeled frames used as tracking prompts
4. Go to **Tracking** to run the tracker across all frames

![Define and track ROIs](../assets/screenshots/quickstart-tracking.png)

---

## 4.5. Preprocessing (Optional)

Stabilized Camera Preprocessing normalises each frame around the tracked body centroid — removing camera drift and head orientation — to produce a clean 592×592 video optimal for DINOv3 feature extraction.

=== "Gradio Web UI"

    1. Switch to the **3. Pre-process (Optional)** tab
    2. Select a video and choose the body ROI id and head ROI id
    3. Adjust filter parameters if needed (defaults work well for most videos)
    4. Click **▶ Run Pre-process**

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
        --min-crop 300 --output-size 592
    ```

    | Option | Default | Description |
    |--------|---------|-------------|
    | `--fc` | `0.25` | Low-pass cutoff frequency (Hz). Period ≈ 4 s. |
    | `--order` | `2` | Butterworth filter order |
    | `--margin` | `75` | Spatial margin around HP residual (px) |
    | `--min-crop` | `300` | Minimum crop side length (px) |
    | `--output-size` | `592` | Output frame side length (px). 592 (= 37×16) for the default DINOv3 ViT-B/16; 518 (= 37×14) for DINOv2 ViT-B/14. |

!!! note "Output"
    The stabilised video is saved as a pre-process session, at
    `{storage}/{project}/preprocessed/sessions/{session_id}/{video}/stabilized.mp4`
    (with a matching `mask_list.h5`). Select the session in **4. Extract Latent**.

!!! tip "When to use preprocessing"
    Use preprocessing when your animal moves freely in the arena and you want DINOv3 features
    to capture **posture** rather than position or orientation. Skip it for fixed-camera or
    already-cropped videos.

---

## 5. Extract Features

1. Switch to the **4. Extract Latent** tab
2. Select your tracked ROIs
3. Run feature extraction — by default this uses **DINOv3** (`dinov3_vitb16`, 768-d) to compute latent representations for each frame. You can also select `dinov3_vitl16` (1024-d) or `dinov2_vitb14_reg4_pretrain` (768-d).

![Extract latent features](../assets/screenshots/quickstart-extract.png)

!!! tip "Multi-GPU extraction (opt-in)"
    With **≥2 CUDA GPUs**, the **Use multiple GPUs** checkbox in the **4. Extract Latent** tab (on by
    default when several GPUs are detected) spreads videos across GPUs (one video per GPU) and splits a
    single video's frames by range across the GPUs (each GPU decodes, preprocesses, and encodes its
    part; results are merged in original order). For the CLI, set the environment variable
    `CASTLE_MULTI_GPU=1` before running extraction; it is off by default there. Frame-split output is
    bit-identical to single-GPU on identical GPUs, and ~1.9× faster on 2 GPUs.

!!! tip "Auto Batch Size & Cache (Phase 2)"
    CASTLE can pick a safe GPU batch size from available VRAM: the CLI does this when `--batch-size` is omitted,
    and in the UI the **Auto Batch Size** button fills in the **Batch Size** field (default `32`).
    If a GPU out-of-memory error occurs mid-run the batch is halved and the extraction retries transparently.

    With **Skip existing files** checked (the default), re-running extraction skips videos that already have a
    latent `.npz` for the same ROI, model and settings — the existing file is reused without re-inferring.

    To run extraction only on videos not yet processed (incremental mode):

    ```python
    from castle.service.incremental_service import get_unprocessed_videos

    pending = get_unprocessed_videos("/data/projects/my_project")
    # → only videos with no latent output yet
    ```

---

## 6. Analyze Behavior

1. Switch to the **5. Behavior Microscope** tab
2. Configure clustering parameters
3. Run the analysis to discover behavioral categories

![Behavior analysis results](../assets/screenshots/quickstart-analysis.png)

!!! note "Reproducibility"
    A `"standardize"` key in a UMAP config is a **legacy no-op** — per-feature z-score
    standardization was removed, so the key is accepted but ignored and has no effect on the
    embedding. Every UMAP run records its resolved random seed in a per-session `umap_log.jsonl`
    (one JSON line per stage). Reuse that seed — on the CPU/deterministic path — to reproduce an
    embedding exactly.

---

## 7. Annotate Clusters (Optional)

1. Within the **5. Behavior Microscope** tab, switch to the **Cluster Annotator** sub-tab
2. Select a clustering session and click **Load Cluster Data**
3. Browse grid videos for each cluster to verify its behavior
4. Assign behavior labels and (optionally) add a comment
5. Labels are **auto-saved** when changed or when the comment field loses focus

---

## 8. Export Results

1. Switch to the **7. Export** tab
2. Select the data components you want (masks, latent, cluster results, annotations, grid videos, analysis results, source videos)
3. Click **📦 Export**, then download the ZIP archive

---

---

## 9. Batch Processing (Phase 4)

If you have multiple experiments or projects to process in one go, use the batch pipeline.

### Create an `experiments.yaml` file

```yaml
experiments:
  - name: "Control Group"
    project: "/data/control"
    videos: ["mouse1.mp4", "mouse2.mp4"]
    params:
      fc: 0.25
  - name: "Treatment Group"
    project: "/data/treatment"
    videos: ["mouse3.mp4", "mouse4.mp4"]

parallel: false
max_workers: 2
```

### Run the batch

```bash
# Process all projects
castle batch run experiments.yaml

# Check status of the last run
castle batch status experiments.yaml

# Generate HTML reports for each project
castle batch report experiments.yaml --output reports/
```

!!! tip "Parallelism"
    Set `parallel: true` and `max_workers: N` in the YAML (or use `--parallel --max-workers N`)
    to run multiple projects concurrently on a multi-GPU or multi-core machine.

---

## 10. Multi-Subject Analysis (Phase 4)

For videos containing **multiple animals**, CASTLE provides `MultiSubjectProject` to manage
independent preprocessing, feature extraction, and clustering per subject, plus social feature
analysis across subjects.

```python
from castle.core.multi_subject import MultiSubjectProject
from castle.analysis.social_features import (
    compute_pairwise_distance,
    detect_social_events,
)
from castle.analysis.group_ethogram import build_group_ethogram, plot_group_ethogram

# Register subjects by their ROI IDs in the mask HDF5
project = MultiSubjectProject("/data/projects/social_session", "video01.mp4")
project.add_subject(subject_id=0, body_roi=1, head_roi=2)
project.add_subject(subject_id=1, body_roi=3, head_roi=4)
project.process_all()

tracks = project.get_subjects()

# Social features
dist   = compute_pairwise_distance(tracks)   # (N, S, S) pixel distances
events = detect_social_events(tracks, distance_threshold=50, duration_threshold=15)

# After setting labels on each track (from your clustering step):
# track.set_labels(cluster_labels)
ethogram = build_group_ethogram(tracks, fps=30.0)
plot_group_ethogram(ethogram, output_path="/tmp/group_ethogram.png")
```

---

## What's Next?

- **[Tutorials](../tutorials/overview.md)** — Detailed step-by-step guides for each stage of the pipeline
- **[Technical Guide](../technical/architecture.md)** — Understand how CASTLE works under the hood
- **[Configuration](../technical/configuration.md)** — Customize parameters for your experiments
- **[API Reference](../reference/api.md)** — Full API docs for Phase 4 modules (batch, multi-subject, social features, reports)
