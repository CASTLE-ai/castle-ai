# Step 1: Create a Project & Upload Videos

This step covers the first two tabs of the CASTLE interface: **0. Project** and **1. Upload Videos**.

---

## Create a New Project

1. Open the **0. Project** tab
2. Click the **New Project** sub-tab
3. Enter a project name (or leave blank for an auto-generated timestamp name like `2026-02-08-14-30-00-Project`)
4. Click **Create**

![Create a new project](../assets/screenshots/tutorial-step1-create-project.png)

### Open an Existing Project

1. In the **Open Project** sub-tab, select a project from the dropdown
2. Click **Open**

!!! tip "Storage Location"
    By default, projects are stored in `projects/` relative to the CASTLE directory. You can change this by expanding the **Change Storage Location** accordion at the top of the Project tab.

### Project Structure on Disk

When you create a project, CASTLE creates a directory with a `config.json` file:

```
projects/
└── my-project/
    ├── config.json          # Project metadata
    ├── sources/             # Uploaded video files
    ├── label/               # ROI label data (.npz files)
    │   └── video-name/      # Per-video label subdirectories
    ├── track/               # Tracking results
    │   └── video-name/      # Per-video tracking data (mask_list.h5)
    ├── latent/              # Extracted features (.npz files)
    │   └── model-name/      # Per-model subdirectories
    ├── crop/                # Cropped/aligned videos
    └── cluster/             # Clustering results (CSV, embeddings)
```

---

## Upload Videos

Switch to the **1. Upload Videos** tab. There are two methods:

### Method A: Upload Local Files

1. Click the upload area
2. Select one or more video files from your computer
3. Files are copied into the project's `sources/` directory

![Upload local videos](../assets/screenshots/tutorial-step1-upload.png)

### Method B: Scan Server Directory

For videos already on the server (common in lab setups):

1. Enter the directory path in the **Add Videos from Folder** field (default: `demo/`)
2. The system automatically scans and lists found videos
3. Click **Add All Videos** to import them into the project

This method creates symlinks or copies rather than requiring re-upload, making it efficient for large video files.

### Supported Formats

CASTLE uses PyAV and OpenCV for video I/O, supporting most common formats:

- MP4, AVI, MOV, MKV
- Most codecs supported by FFmpeg

---

## Using Demo Data

CASTLE includes demo videos for testing and learning:

| Demo | File | Description |
|------|------|-------------|
| Reaching Task | `demo/case1-reaching-task/reaching-task-raw.mp4` | Mouse reaching task |
| Open Field | `demo/case2-openfield/openfield-1min-raw.mp4` | Open field test (1 min) |
| Gait | `demo/case5-gait/gait-raw.mp4` | Gait analysis |
| Reach & Grasp | `demo/Reach-and-Grasp-demo.mp4` | Reach and grasp demo |

!!! tip "Start Here"
    For your first time, use `reaching-task-raw.mp4` — it's short and processes quickly.

!!! note "Shared ROI Prompts"
    All videos within a project share the same ROI prompts. This means you can build a robust set of prompts on one video and apply them to all others via batch tracking.

---

## Next Step

Once you have a project with videos, proceed to [**Step 2: Track ROIs**](step2-tracking.md).
