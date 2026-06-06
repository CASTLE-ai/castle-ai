

![CASTLE 標誌](assets/logo.png)
[![arXiv](https://img.shields.io/badge/biorxiv-2025.08.22.671685v2-<COLOR>.svg)](https://www.biorxiv.org/content/10.1101/2025.08.22.671685v2)
[![PyPI version](https://badge.fury.io/py/castle-ai.svg)](https://badge.fury.io/py/castle-ai)
[![CI](https://github.com/CASTLE-ai/castle-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/CASTLE-ai/castle-ai/actions/workflows/ci.yml)

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-castle--ai.github.io-blue)](https://castle-ai.github.io/castle-ai/)
[![PyPI Downloads](https://static.pepy.tech/badge/castle-ai/month)](https://pepy.tech/projects/castle-ai)
[![PyPI Downloads](https://static.pepy.tech/badge/castle-ai)](https://pepy.tech/projects/castle-ai)


![CASTLE Flowchart](assets/Flowchart.png)


**CASTLE (Combined Approach for Segmentation and Tracking with Latent Extraction)** is a training-free framework that combines segmentation models, tracking algorithms, and visual foundation models to automatically discover animal behaviors from video. Through focused latent extraction and hierarchical clustering, it achieves expert-level accuracy across multiple species without manual labeling, while uncovering previously hidden behavioral patterns that keypoint methods miss.

<p align="center">
  <img src="assets/Reaching_demo.gif" alt="Reaching Demo">
</p>

## Documentation

📚 **Full documentation**: [castle-ai.github.io/castle-ai](https://castle-ai.github.io/castle-ai/)

- [Installation Guide](https://castle-ai.github.io/castle-ai/getting-started/installation/)
- [Quick Start](https://castle-ai.github.io/castle-ai/getting-started/quickstart/)
- [Tutorials](https://castle-ai.github.io/castle-ai/tutorials/overview/)
- [API Reference](https://castle-ai.github.io/castle-ai/reference/api/)

## Latest Updates
- **2026-02: Phase 4 — Multi-Subject, Batch Processing & Reports** 🟢
  - **`SubjectTrack` + `MultiSubjectProject`** (`castle/core/multi_subject.py`): First-class support for videos with multiple animals. Each subject is defined by body + head ROI IDs in the shared mask HDF5. `process_all()` extracts per-subject positions and angles; the resulting `SubjectTrack` objects hold latents and cluster labels set by your extraction/clustering steps.
  - **Social Feature Extraction** (`castle/analysis/social_features.py`): `compute_pairwise_distance`, `compute_relative_orientation`, `compute_approach_score`, and `detect_social_events` — all operating on synchronised `SubjectTrack` lists and returning `(N, S, S)` arrays or event dicts.
  - **Group Ethogram** (`castle/analysis/group_ethogram.py`): `build_group_ethogram` assembles a synchronised multi-subject ethogram dict (per-subject bouts + shared social events). `plot_group_ethogram` renders a publication-quality colour-coded raster with social interaction spans.
  - **Batch Processing** (`castle/core/batch.py` + `castle/cli/batch_cmd.py`): `BatchConfig.from_yaml("experiments.yaml")` + `BatchRunner.run()` executes the full CASTLE pipeline across multiple projects/experiments with optional parallelism. `castle batch run/status/report` CLI subcommands included.
  - **HTML Report Generator** (`castle/analysis/report.py`): `ReportGenerator.generate()` produces self-contained HTML reports with inline base64 plots (ethogram frequency chart, 2-D embedding scatter), bout statistics, transition matrix, and quality metric badges — no external CSS/JS required.

- **2026-02: Phase 3 — Simplification & Code Clarity**
  - **`ProjectData` + `VideoInfo`** (`castle/core/project_data.py`): Unified project path dataclass — eliminates all scattered `os.path.join(storage_path, project_name, …)` calls. Single `from_path()` constructor computes every standard directory and per-video path.
  - **`ClusterData`** (`castle/core/cluster_data.py`): Typed dataclass that consolidates `cluster_*.npz`, `time_series_*.csv`, `id.csv`, and `annotations.csv` into one container with `load()`, `save()`, `from_arrays()`, `get_cluster_frames()`, and `n_clusters()`.
  - **`DeviceFactory`** (`castle/core/device_factory.py`): Centralised device detection (CUDA > MPS > CPU) with cached result. Factory methods `get_umap()`, `get_dbscan()`, `get_hdbscan()` automatically dispatch to GPU-accelerated cuML on CUDA or sklearn/umap-learn on CPU/MPS — no more scattered `if cuda … elif mps … else` branches.
  - **`SimpleVideoReader`** (`castle/utils/video_reader_simple.py`): Clean PyAV-based video reader with no cv2 dependency, no LRU cache complexity. Provides `get_frame(index)`, `iter_frames(start, end, step)`, and context-manager support. Sequential iteration uses no per-frame seeks for maximum throughput.
  - **UI Handler Pattern Guide** (`castle/ui/HANDLER_GUIDE.md`): Documents the target thin-handler / fat-service convention for all Gradio UI callbacks — handler ≤ 15 lines, zero algorithmic logic, one service call, convert domain exceptions to `gr.Error`.

- **2026-02: Developer Branch - Major Architecture Overhaul**
  - **Stabilized Camera Preprocessing** (Phase 0): Zero-phase Butterworth low-pass filtering of centroid trajectories + dynamic crop extraction → 518×518 stabilised MP4 ready for DINOv2 (`castle preprocess`)
  - **Service Layer**: Clean separation between UI and business logic
  - **CLI Frontend**: Full command-line interface via `castle` command
  - **Cluster Annotator** (sub-tab in Gradio 5): Grid video browser, per-session annotations, comment field, auto-save on label change / comment blur, mask contour overlay, speed control
  - **Analysis Tab** (Tab 6 Gradio): Ethogram (raster + transition matrix + bout stats), Quality Metrics (silhouette, CH, DB, V-measure, NMI, ARI), Group Comparison placeholder
  - **Export Tab** (Tab 7 Gradio): ZIP download with selectable components (masks, latent, cluster results, annotations, grid videos, analysis outputs)
  - **Multi-scale Pooling**: Spatial pyramid pooling for richer latent representations
  - **Session Management**: Multiple clustering sessions per project with `SessionManager`
  - **NWB Export**: `castle ethogram export-nwb` — Neurodata Without Borders format
  - **Undo/Redo**: Command Pattern history for clustering operations
  - **260 Unit Tests**: Comprehensive test coverage
  - **Code Quality**: Zero ruff warnings in non-vendored code

- **2025-12: Phase 2 — Performance & GPU Memory Management**
  - **ModelRegistry**: Thread-safe singleton for lazy model loading and explicit VRAM cleanup between pipeline stages (`castle/core/model_registry.py`)
  - **Auto Batch Size**: VRAM-aware `compute_optimal_batch_size()` + `auto_retry_on_oom()` wrapper that halves batch size on GPU OOM and retries automatically (`castle/core/auto_batch.py`)
  - **Pipeline Orchestrator**: `Pipeline` class with per-stage GPU memory cleanup — tracking cleanup before extraction, extraction cleanup after (`castle/core/pipeline.py`)
  - **Content-Hash Cache**: `PipelineCache` with SHA-256 keying and atomic JSON manifest — skip already-computed extractions across runs (`castle/core/cache.py`)
  - **Incremental Processing**: `get_unprocessed_videos()` and `cleanup_deleted_videos()` for efficient batch runs and orphan cleanup (`castle/service/incremental_service.py`)

- **2024-09: Public Release**
  - Initial public release of the CASTLE tool.


## Quick Start

### Option 1 (Colab)
[![Open In Colab (free accounts are vary slow)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CASTLE-ai/castle-ai/blob/main/notebooks/colab.ipynb)
[![CASTLE Quick start @Colab](https://img.shields.io/badge/YouTube-CASTLE%20Demo-red?logo=youtube)](https://youtu.be/qzZlixEaKvQ)

### Option 2 (Local Installation)

1.  **Clone & Environment**:
    ```bash
    git clone https://github.com/CASTLE-ai/castle-ai.git
    cd castle-ai
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Download Checkpoints**:
    Sometime the ckpt download may be blocked by Google. So you can download the models from the web by copying the links to the Chrome browser and downloading them.
    ```text
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth
    https://drive.google.com/file/d/1g4E-F0RPOx9Nd6J7tU9AE1TjsouL4oZq/edit
    https://drive.google.com/file/d/1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ/edit
    ```
    Alternatively, you can use the download_ckpt.sh script:
    ```bash
    ./download_ckpt.sh
    ```
    Format:
    ```text
    castle-ai
    ├── castle
    └── ckpt
        ├── dinov2_vitb14_reg4_pretrain.pth
        ├── R50_DeAOTL_PRE_YTB_DAV.pth
        ├── sam_vit_b_01ec64.pth
        └── SwinB_DeAOTL_PRE_YTB_DAV.pth
    ```

### Option 3 (Docker — GPU recommended)

> **Requirements**: [Docker](https://docs.docker.com/get-docker/) ≥ 24, and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU mode.

#### GPU (recommended)

```bash
# Pull or build
git clone https://github.com/CASTLE-ai/castle-ai.git && cd castle-ai

# Build + run with GPU support
docker compose up --build
```

The Gradio UI will be available at [http://localhost:7860](http://localhost:7860).  
Model checkpoints are downloaded automatically on first start and cached in a Docker volume.

#### CPU-only (no NVIDIA GPU required)

```bash
docker compose -f docker-compose.cpu.yml up --build
```

#### Quick one-liner (pre-built image)

```bash
# GPU (recommended)
docker run --gpus all -p 7860:7860 -v $(pwd)/projects:/data castle-ai/castle

# CPU only
docker run -p 7860:7860 -v $(pwd)/projects:/data castle-ai/castle:cpu
```

#### Pre-embed checkpoints at build time (~4 GB larger image, zero first-run delay)

```bash
docker build --build-arg DOWNLOAD_CKPT=1 -t castle-ai/castle .
```

---

## Run App

=== "Gradio Web UI"

```bash
python app.py
```

Opens at [http://localhost:7860](http://localhost:7860) with 8 tabs:
`0. Project | 1. Upload Videos | 2. Tracking ROIs | 3. Pre-process (Optional) | 4. Extract Latent | 5. Behavior Microscope | 6. Analysis | 7. Export`

=== "CLI"

```bash
castle --help
# Key commands:
castle project init <name>
castle track <project>
castle preprocess <project> --video <name> --body-roi 1 --head-roi 2
castle extract <project>
castle cluster run <project>
castle ethogram analyze <project>
castle compare run <project_a> <project_b>
castle batch run experiments.yaml         # Batch: run full pipeline across experiments
castle batch status experiments.yaml      # Batch: show last run status
castle batch report experiments.yaml -o reports/  # Batch: generate HTML reports
```

## Performance Benchmarks

The following benchmarks were measured on a workstation with **Intel i7-12700 + RTX 3060 (12GB)**. Time consumption is expressed as a multiple of the video's actual duration (assuming 30 FPS).

| Task / Model | Video Res | Model Res | **Ratio** | Notes |
| :--- | :--- | :--- | :--- |
| **GMFlow** | 720x720 | 720x720 | **4.50x** | Essential for fine movement (Residual Motion). |
| **DINOv2b** (ViT-B/14) | 720x720 | 518x518 | **2.20x** | Standard vision foundation model. |
| **DeAOT** (Tracking) | 720x720 | 720x720 | **2.11x** | ROI segmentation and tracking. |
| **DINOv3b** (ViT-B/16) | 720x720 | 592x592 | **0.91x** | **Faster than real-time**. Highly optimized. |

> [!TIP]
> **Hardware Scaling**: Higher-end GPUs like the **RTX 4090** are estimated to provide approximately **3.5x - 5x** speedup compared to the RTX 3060, enabling real-time processing for most modules.

---

## GPU Memory Management (Phase 2)

CASTLE Phase 2 introduces a set of modules that keep VRAM usage predictable across long multi-video runs.

### ModelRegistry — Singleton Model Lifecycle

```python
from castle.core.model_registry import ModelRegistry

registry = ModelRegistry.instance()          # global singleton

# Context manager: auto-unload on exit
with registry.use("dinov3_vitb16") as model:   # current default encoder
    latents = model.extract_tensor_batch(frames, masks, roi_id)

# Bulk unload by family keyword (e.g. after the tracking stage)
registry.unload_family("sam", "deaot", "aot")

# Check VRAM usage
stats = registry.get_memory_stats()
print(stats["free_mb"], "MB free on", stats["device"])
```

The `Pipeline` class calls `unload_family` automatically between the tracking and extraction stages, ensuring SAM/DeAOT weights are evicted before the DINOv3 encoder is loaded.

### Auto Batch Size & OOM Retry

```python
from castle.core.auto_batch import compute_optimal_batch_size, auto_retry_on_oom

# Queries free VRAM and returns a safe batch size for the given model + frame size
batch = compute_optimal_batch_size("dinov3_vitb16", frame_size=(592, 592, 3))

# If an OOM error occurs, halves the batch and retries automatically
result = auto_retry_on_oom(extract_fn, frames, initial_batch=batch)
```

### Content-Hash Cache

```python
from castle.core.cache import PipelineCache

cache = PipelineCache("projects/my_project/latent")
key   = cache.compute_key(video_path, preprocess_config, "dinov2_vitb14")

if not cache.is_cached(key):
    path = run_extraction(...)
    cache.put(key, path)
```

The cache key is `SHA-256(abs_path + mtime + config + model_name)`. Re-running extraction on an unchanged video is a near-instant cache hit.

---

## About us

CASTLE is a project by the [Wu Lab](https://www.yuweiwu.org/), a research group at the [Academia Sinica](https://www.sinica.edu.tw/en).


## Credits & Licenses

This project incorporates code and methodologies from the following sources:

- SAM (Segment Anything Model): https://github.com/facebookresearch/segment-anything (Apache License 2.0)
- DeAOT (Decoupling Features in Hierarchical Propagation): https://github.com/yoxu515/aot-benchmark (BSD 3-Clause License)
- DINOv2 (Self-Supervised Vision Transformer): https://github.com/facebookresearch/dinov2 (Apache License 2.0)
- DINOv3 (Vision Transformer): https://github.com/facebookresearch/dinov3 (Creative Commons Attribution-NonCommercial 4.0 International)

This work is distributed under the terms of the Apache License 2.0.


## Citation

If you find this work useful, please consider citing:

```bibtex
@article{CASTLE,
  title={CASTLE: a training‑free foundation‑model pipeline for unsupervised, cross‑species behavioral classification},
  author={Liu, Yu-Shun and Yeh, Han-Yuan and Hu, Yu-Ting and Wu, Bing-Shiuan and Chen, Yi-Fang and Yang, Jia-Bin and Jasmin, Sureka and Hsu, Ching-Lung and Lin, Suewei and Chen, Chun-Hao and Wu, Yu-Wei},
  journal={bioRxiv},
  year={2025}
}
```
