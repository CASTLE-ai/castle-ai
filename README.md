

![CASTLE 標誌](assets/logo.png)
[![arXiv](https://img.shields.io/badge/biorxiv-2025.08.22.671685v2-<COLOR>.svg)](https://www.biorxiv.org/content/10.1101/2025.08.22.671685v2)
[![PyPI version](https://badge.fury.io/py/castle-ai.svg)](https://badge.fury.io/py/castle-ai)
[![CI](https://github.com/CASTLE-ai/castle-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/CASTLE-ai/castle-ai/actions/workflows/ci.yml)

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![PyPI Downloads](https://static.pepy.tech/badge/castle-ai/month)](https://pepy.tech/projects/castle-ai)
[![PyPI Downloads](https://static.pepy.tech/badge/castle-ai)](https://pepy.tech/projects/castle-ai)


![CASTLE Flowchart](assets/Flowchart.png)


**CASTLE (Combined Approach for Segmentation and Tracking with Latent Extraction)** is a training-free framework that combines segmentation models, tracking algorithms, and visual foundation models to automatically discover animal behaviors from video. Through focused latent extraction and hierarchical clustering, it achieves expert-level accuracy across multiple species without manual labeling, while uncovering previously hidden behavioral patterns that keypoint methods miss.

<p align="center">
  <img src="assets/Reaching_demo.gif" alt="Reaching Demo">
</p>

## Latest Updates
- **2025-12: Performance & Stability Update**
  - **High-Performance Pipeline**: Optimized CPU/GPU batch processing for both Tracking and Extraction.

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

## Run App
```bash
python app.py
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
