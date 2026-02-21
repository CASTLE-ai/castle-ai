# API Reference

Auto-generated documentation from Python docstrings using [mkdocstrings](https://mkdocstrings.github.io/).

!!! note
    Some modules may have limited docstring coverage. This reference will improve over time.

---

## Service Layer

### Project Service

::: castle.service.project_service
    options:
      show_root_heading: true
      show_source: true

---

### Extraction Service

::: castle.service.extraction_service
    options:
      show_root_heading: true
      show_source: true

---

### Clustering Service

::: castle.service.clustering_service
    options:
      show_root_heading: true
      show_source: true

---

### Tracking Service

::: castle.service.tracking_service
    options:
      show_root_heading: true
      show_source: true

---

### Preprocessing Service ★

::: castle.service.preprocessing_service
    options:
      show_root_heading: true
      show_source: true

---

### Annotation Service

::: castle.service.annotation_service
    options:
      show_root_heading: true
      show_source: true

---

### Bout Service

::: castle.service.bout_service
    options:
      show_root_heading: true
      show_source: true

---

### History Service (Undo/Redo)

::: castle.service.history_service
    options:
      show_root_heading: true
      show_source: true

---

### Ethogram Service

::: castle.service.ethogram_service
    options:
      show_root_heading: true
      show_source: true

---

### Metrics Service

::: castle.service.metrics_service
    options:
      show_root_heading: true
      show_source: true

---

### Comparison Service

::: castle.service.comparison_service
    options:
      show_root_heading: true
      show_source: true

---

### Annotator Loader

::: castle.service.annotator_loader
    options:
      show_root_heading: true
      show_source: true

---

### Session Manager

::: castle.service.session_manager
    options:
      show_root_heading: true
      show_source: true

---

## CLI

::: castle.cli.main
    options:
      show_root_heading: true
      show_source: true

---

## Core

### Project Configuration

::: castle.core.project_config
    options:
      show_root_heading: true
      show_source: true

---

### Environment

::: castle.core.environment
    options:
      show_root_heading: true
      show_source: true

---

### Mask Filter

::: castle.core.mask_filter
    options:
      show_root_heading: true
      show_source: true

---

### Stabilized Camera (Phase 0) ★

The `castle.core.stabilized_camera` module implements the **Phase 0 preprocessing** pipeline.
It applies a zero-phase Butterworth low-pass filter to ROI centroid trajectories and orientation
angles, then extracts dynamically-cropped and head-aligned frames resized to 518×518 for DINOv2.

::: castle.core.stabilized_camera
    options:
      show_root_heading: true
      show_source: true
      members:
        - StabilizedCamera
        - extract_centroids_from_masks
        - extract_orientations_from_masks
        - preview_stabilization

---

### Ethogram Engine

::: castle.core.ethogram
    options:
      show_root_heading: true
      show_source: true

---

### Quality Metrics

::: castle.core.metrics
    options:
      show_root_heading: true
      show_source: true

---

### Group Comparison

::: castle.core.comparison
    options:
      show_root_heading: true
      show_source: true

---

### Extractor

<!-- mkdocstrings fails on castle.core.extractor due to import dependencies (torch, cuml, etc.) -->
<!-- ::: castle.core.extractor -->

!!! note "Build Note"
    Auto-documentation for `castle.core.extractor` is unavailable due to import dependencies that are not installed in the docs build environment. See source code directly: [`castle/core/extractor.py`](https://github.com/CASTLE-ai/castle-ai/blob/main/castle/core/extractor.py)

---

### Data

<!-- mkdocstrings fails on castle.core.data due to import dependencies -->
<!-- ::: castle.core.data -->

!!! note "Build Note"
    Auto-documentation for `castle.core.data` is unavailable due to import dependencies. See source: [`castle/core/data.py`](https://github.com/CASTLE-ai/castle-ai/blob/main/castle/core/data.py)

---

### Models

<!-- mkdocstrings fails on castle.core.models due to import dependencies -->
<!-- ::: castle.core.models -->

!!! note "Build Note"
    Auto-documentation for `castle.core.models` is unavailable due to import dependencies. See source: [`castle/core/models.py`](https://github.com/CASTLE-ai/castle-ai/blob/main/castle/core/models.py)

---

## Utils

### Project Management

::: castle.utils.project_manager
    options:
      show_root_heading: true
      show_source: true

---

### Video I/O

::: castle.utils.video_io
    options:
      show_root_heading: true
      show_source: true

---

### Video Management

::: castle.utils.video_manager
    options:
      show_root_heading: true
      show_source: true

---

### Image Segmentation (SAM)

::: castle.utils.image_segment
    options:
      show_root_heading: true
      show_source: true

---

### ROI Tracking

::: castle.utils.tracking_manager
    options:
      show_root_heading: true
      show_source: true

---

### Feature Extraction

::: castle.utils.visual_latent_extract
    options:
      show_root_heading: true
      show_source: true

---

### Latent Explorer

::: castle.utils.latent_explorer
    options:
      show_root_heading: true
      show_source: true

---

### Video Alignment

::: castle.utils.video_align
    options:
      show_root_heading: true
      show_source: true

---

### HDF5 I/O

::: castle.utils.h5_io
    options:
      show_root_heading: true
      show_source: true

---

## Visualization

### Embedding Plots

::: castle.visualization.embedding_plots
    options:
      show_root_heading: true
      show_source: true

---

### Ethogram Plots

::: castle.visualization.ethogram_plots
    options:
      show_root_heading: true
      show_source: true

---

### Comparison Plots

::: castle.visualization.comparison_plots
    options:
      show_root_heading: true
      show_source: true

---

## New Modules (v5.1)

### Temporal Smoothing

::: castle.core.temporal_smooth
    options:
      show_root_heading: true
      show_source: true

---

### Cluster Transfer

::: castle.core.cluster_transfer
    options:
      show_root_heading: true
      show_source: true

---

### Automated Behavior Microscope

::: castle.core.auto_cluster
    options:
      show_root_heading: true
      show_source: true

---

### NWB Export

::: castle.core.nwb_export
    options:
      show_root_heading: true
      show_source: true

---

### NWB Service

::: castle.service.nwb_service
    options:
      show_root_heading: true
      show_source: true

---

## CLI Commands

### Preprocess CLI ★

::: castle.cli.preprocess_cmd
    options:
      show_root_heading: true
      show_source: true

---

### Ethogram CLI

::: castle.cli.ethogram_cmd
    options:
      show_root_heading: true
      show_source: true

---

### Compare CLI

::: castle.cli.compare_cmd
    options:
      show_root_heading: true
      show_source: true

---

### Cluster CLI

::: castle.cli.cluster_cmd
    options:
      show_root_heading: true
      show_source: true

---

## Legacy Utils

### Plot

::: castle.utils.plot
    options:
      show_root_heading: true
      show_source: true
