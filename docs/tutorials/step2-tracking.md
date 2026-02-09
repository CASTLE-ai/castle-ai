# Step 2: Track Regions of Interest

The **2. Tracking ROIs** tab is the most complex part of CASTLE. It contains two major sections: **Single Video Tracking** (for building and refining your ROI prompts) and **Batch Videos Tracking** (for scaling to multiple videos).

---

## Overview

The tracking workflow has two phases:

### Phase 1: Single Video Tracking (Build Your ROI Prompts)

1. **Label ROIs** on reference frames using SAM
2. **Track** across all frames using DeAOT
3. **Review** results and fix errors iteratively

### Phase 2: Batch Video Tracking (Scale Up)

Once prompts work well on one video, apply them to all videos in the project.

---

## Getting Started

1. Switch to the **2. Tracking ROIs** tab
2. Select a video from the **Select Video** dropdown
3. Click **Edit** to load the video

![Select video for tracking](../assets/screenshots/tutorial-step2-select-video.png)

---

## Single Video Tracking

### Label ROI

The **Label ROI** sub-tab is where you define what to track using point-and-click segmentation.

1. Navigate to the frame you want to label using the frame slider
2. **Click on the animal** (or body part) to create a segmentation mask
    - SAM generates a mask from your click
    - Click mode defaults to **Add** — each click refines the mask
    - Use **Change Mode** to switch to **Remove** mode for excluding regions
3. To track multiple ROIs (e.g., body + head), click **Label Next ROI** to start a new ROI
4. Click **Save ROIs** when satisfied

![Label ROI with SAM](../assets/screenshots/tutorial-step2-label.png)

!!! tip "What is an ROI?"
    An ROI (Region of Interest) is a segmented area you want to track. Common examples:

    - **Body centroid**: the entire animal body
    - **Head**: for head direction analysis
    - **Tail**: used as orientation reference for video alignment

!!! warning "Label on Multiple Frames"
    For robust tracking, label ROIs on **multiple frames** — especially frames where the animal's appearance changes significantly (different postures, partial occlusion, etc.). The more diverse your labels, the more robust the tracking.

### ROI Prompts

The **ROI Prompts** sub-tab displays a gallery of all saved labels across the project. This is where you can review the diversity of your prompt set.

- Labels are loaded from `.npz` files in the project's `label/` directory
- Each entry shows the frame with the segmentation mask overlaid
- Labels from all videos in the project are shown together

!!! tip "Prompt Guidelines"
    - **Diversity = Stability**: varied examples improve tracking robustness
    - **Fewer = Faster**: fewer prompts mean faster execution
    - **Sweet spot**: 5–30 prompts for optimal balance

### Tracking

The **Tracking** sub-tab runs DeAOT to propagate masks across frames.

1. Select a tracking model:
    - **R50** (ResNet-50): faster, good for most cases
    - **SwinB** (Swin Transformer-B): more accurate, better for challenging videos
2. Set **Start Frame** and **Stop Frame** (defaults to the full video)
3. Optionally check **Skip existing** to avoid re-processing
4. Click to start tracking

![Run tracking](../assets/screenshots/tutorial-step2-tracking.png)

!!! warning "Monitor in Real-Time"
    Watch the tracking progress carefully. If you see errors (mask drifting, losing the animal), **cancel immediately** — don't waste time on bad tracking. Instead:

    1. Go back to **Label ROI**
    2. Label the frame where tracking failed
    3. Adjust the **Start Frame** in Tracking settings
    4. Re-run tracking from that point

!!! note "Display Timing"
    During real-time monitoring, the displayed frame and mask may appear off by one frame. This is normal due to processing timing. Verify final results in the **View** tab after tracking completes.

### View

The **View** sub-tab lets you browse tracking results frame by frame. Use the slider to scrub through frames and verify that masks are accurate.

### Analysis (Post-Processing)

The **Analysis** sub-tab provides tools for reviewing and analyzing tracking quality after tracking is complete.

---

## Batch Videos Tracking

Once your ROI prompts successfully track one video, use **Batch Videos Tracking** to process all videos in the project.

1. Switch to the **Batch Videos Tracking** sub-tab
2. The system uses all saved ROI prompts from the project
3. Each video is tracked independently using the shared prompt set
4. Progress is displayed for each video

![Batch tracking](../assets/screenshots/tutorial-step2-batch.png)

!!! tip "Iterative Refinement"
    If batch tracking fails on some videos, add labels from those failure frames and re-run. Your prompt set becomes more robust over time.

---

## Tips and Best Practices

- **Start with one video**: build a solid prompt set before scaling to batch
- **Quality over quantity**: a few well-chosen prompts across diverse frames work better than many similar ones
- **R50 vs SwinB**: start with R50 for speed; switch to SwinB if tracking quality is insufficient
- **Frame selection**: label frames with different postures, lighting conditions, and occlusion scenarios
- **Cancel early**: if tracking starts failing, cancel and add more labels rather than waiting for it to finish

---

## Next Step

Once all videos are tracked, proceed to [**Step 3: Extract Features**](step3-extract.md).
