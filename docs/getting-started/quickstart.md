# Quick Start

!!! tip "Prerequisites"
    Complete the [Installation](installation.md) guide first, including downloading model checkpoints.

This guide walks you through a complete analysis using a demo video in under 5 minutes.

---

## 1. Launch CASTLE

```bash
python app.py
```

The Gradio web UI opens at [http://localhost:7860](http://localhost:7860).

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

## 5. Extract Features

1. Switch to the **3. Extract Latent** tab
2. Select your tracked ROIs
3. Run feature extraction — this uses DINOv2/v3 to compute latent representations for each frame

![Extract latent features](../assets/screenshots/quickstart-extract.png)

---

## 6. Analyze Behavior

1. Switch to the **4. Behavior Microscope** tab
2. Configure clustering parameters
3. Run the analysis to discover behavioral categories

![Behavior analysis results](../assets/screenshots/quickstart-analysis.png)

---

## What's Next?

- **[Tutorials](../tutorials/overview.md)** — Detailed step-by-step guides for each stage of the pipeline
- **[Technical Guide](../technical/architecture.md)** — Understand how CASTLE works under the hood
- **[Configuration](../technical/configuration.md)** — Customize parameters for your experiments
