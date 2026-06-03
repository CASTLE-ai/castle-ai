# Step 2.5: Stabilized Camera Preprocessing *(Optional)*

After tracking and before feature extraction, CASTLE can apply **Stabilized Camera Preprocessing**
to produce a head-fixed, dynamically-cropped video that is optimal for DINOv3 feature extraction.

!!! note "When to use this step"
    This step is *optional* but strongly recommended when:

    - The animal moves freely across the arena (not a fixed-camera close-up)
    - You want DINOv3 features to capture **posture** rather than position or heading direction
    - You see poor cluster separation due to orientation-dependent features

    Skip it for fixed-camera or already-aligned videos.

---

## Overview

The `StabilizedCamera` module applies a **zero-phase Butterworth low-pass filter** to the body
centroid trajectory and heading angle extracted from the tracking masks, then extracts a
dynamically-sized crop centred on the filtered trajectory and resized to 518×518 px.

```
mask_list.h5  (tracking output)
    │
    ├─ extract_centroids_from_masks → body centroid x(t)
    ├─ extract_orientations_from_masks (body→head vector) → θ(t)
    │
    ▼
Zero-phase Butterworth LP  (fc=0.25 Hz, order=2, filtfilt)
    │
    ├─ x_c(t), θ_c(t)  — smooth camera trajectory
    │
    ▼
crop_size = max(300, 2 × (‖x(t) − x_c(t)‖ + 75))   [px]
    │
    ▼
warpAffine: translate → x_c(t), rotate → θ_c(t) − 90°
    │
    ▼
Resize → 592 × 592 px   (DINOv3 ViT-B/16 default; 518 = 37 × 14 suits DINOv2 ViT-B/14)
    │
    ▼
preprocessed/{video}/stabilized.mp4
```

### Why zero-phase filtering?

`scipy.signal.filtfilt` applies the filter **forward and backward**, resulting in zero net
phase shift — i.e. the filtered trajectory is perfectly time-aligned with the original video.
This matters because any phase delay would shift the crop relative to the animal's actual
position in the frame.

### Why dynamic crop?

The crop window is proportional to the animal's instantaneous **high-pass residual** — the
component of motion that is *faster* than the smoothed trajectory. When the animal makes a
fast jerk, the window expands to keep the animal fully in frame. During slow movement it
contracts to the minimum (300 px), giving a tighter zoom.

---

## Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fc` | `0.25` Hz | Low-pass cutoff. Period ≈ 4 s. Removes slow arena drift. |
| `order` | `2` | Butterworth filter order. Higher = steeper roll-off. |
| `margin` | `75` px | Spatial buffer added around the HP residual. |
| `min_crop` | `300` px | Floor for the dynamic crop window size. |
| `output_size` | `592` px | Output frame side length. Match the encoder patch grid: 592 = 37 × 16 for the default DINOv3 ViT-B/16; 518 = 37 × 14 for DINOv2 ViT-B/14. |

---

## ROI IDs

You need two ROI ids from your tracking setup:

- **`--body-roi`** — the ROI used as the body centroid (determines crop centre and orientation base)
- **`--head-roi`** — the ROI used as the head (together with `body-roi`, determines heading angle θ)

These ids correspond to the integer pixel values in `mask_list.h5`. If you labelled Body as
ROI 1 and Head as ROI 2 in the tracking step, use `--body-roi 1 --head-roi 2`.

---

## Running Preprocessing

=== "CLI (recommended for batch)"

    ```bash
    castle preprocess <project> \
        --video animal.mp4 --body-roi 1 --head-roi 2
    ```

    With custom filter parameters:

    ```bash
    castle preprocess my_project \
        --video animal.mp4 \
        --body-roi 1 --head-roi 2 \
        --fc 0.25 --order 2 \
        --margin 75 --min-crop 300 --output-size 518
    ```

    The command prints a progress bar and diagnostic metrics when finished:

    ```
    ✓ Preprocessing complete.
      Output video : projects/my_project/preprocessed/animal.mp4/stabilized.mp4
      Preview clip : projects/my_project/preprocessed/animal.mp4/stabilized_preview.mp4
      Frames       : 18000
      HP residual RMS : 12.34 px  |  % at min_crop : 73.2%  |  speed-crop r : 0.812
    ```

=== "Gradio Web UI"

    1. Switch to the **2. Tracking ROIs** tab
    2. Open the **Preprocessing** sub-tab
    3. Select the project video from the dropdown
    4. Enter Body ROI id and Head ROI id
    5. Adjust filter parameters if needed
    6. Click **Run Preprocessing**

    A preview clip is shown inline when processing completes.

=== "Python API"

    ```python
    from castle.service.preprocessing_service import PreprocessingService

    svc = PreprocessingService(storage_path="projects/", project_name="my_project")

    result = svc.preprocess_stabilized_camera(
        video_name="animal.mp4",
        body_roi_id=1,
        head_roi_id=2,
        # Optional — override defaults:
        fc=0.25,
        order=2,
        margin=75,
        min_crop=300,
        output_size=518,
        preview_duration=10.0,
    )

    print(result["preprocessed_video_path"])   # full stabilised video
    print(result["preview_path"])              # 10-second preview
    print(result["n_frames"])                  # number of processed frames

    diag = result["diagnostics"]
    print(f"HP residual RMS: {diag['hp_residual_rms']:.2f} px")
    print(f"Frames at min_crop: {diag['pct_at_min_crop']:.1f}%")
    print(f"Speed-crop correlation: {diag['speed_crop_correlation']:.3f}")
    ```

    You can also call the module-level function directly:

    ```python
    from castle.service.preprocessing_service import preprocess_stabilized_camera

    result = preprocess_stabilized_camera(
        storage_path="projects/",
        project_name="my_project",
        video_name="animal.mp4",
        body_roi_id=1,
        head_roi_id=2,
    )
    ```

---

## Output

The preprocessed files are saved under the project directory:

```
projects/my_project/
└── preprocessed/
    └── animal.mp4/
        ├── stabilized.mp4          # Full-length stabilised video (output_size × output_size)
        └── stabilized_preview.mp4  # Short preview clip (first preview_duration seconds)
```

Use `stabilized.mp4` as the video source in the **3. Extract Latent** tab.

---

## Diagnostic Metrics

After preprocessing, CASTLE reports three metrics to help you judge quality:

| Metric | Description |
|--------|-------------|
| `hp_residual_rms` | RMS of the high-pass positional residual (px). High values → animal moves fast relative to the smoothed trajectory. |
| `pct_at_min_crop` | % of frames where the crop was clamped to `min_crop`. High values → animal is mostly slow/stationary. |
| `speed_crop_correlation` | Pearson r between per-frame speed and crop size. Should be positive (faster → larger crop). |

---

## Tips

- **Check the preview clip first** — open `stabilized_preview.mp4` to visually verify that the
  animal is centred and the orientation is correct before processing the full video.
- **Increase `min_crop`** if the animal is large relative to the frame (the default 300 px may
  clip body parts for big animals).
- **Lower `fc`** (e.g. 0.1 Hz) for very slow animals or longer-period drift; raise it (e.g. 0.5 Hz)
  if you want finer movement detail preserved.
- **Increase `margin`** if the dynamic crop is clipping the animal during fast movements.

---

## Next Step

Once preprocessing is complete, proceed to
[**Step 3: Extract Features**](step3-extract.md) using the stabilised video as input.
