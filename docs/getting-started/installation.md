# Installation

CASTLE requires **Python 3.10+** and an **NVIDIA GPU with CUDA 12.x** for full functionality. CPU-only mode is possible but significantly slower.

---

## Method 1: Local Installation (Recommended)

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.x drivers (recommended)
- ~4 GB disk space for model checkpoints
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/CASTLE-ai/castle-ai.git
cd castle-ai
```

### Step 2: Create a Virtual Environment

=== "Linux / macOS"

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

=== "Windows"

    ```powershell
    python -m venv .venv
    .venv\Scripts\activate
    ```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

!!! warning "Do NOT use `pip install castle-ai`"
    The `requirements.txt` is designed for installation from source. Install directly from the cloned repository — the PyPI package may be outdated.

!!! note "Key dependencies"
    The main dependencies include:

    - **PyTorch** with CUDA 12.6 support (`torch`, `torchvision`)
    - **cuml-cu12** — NVIDIA RAPIDS for GPU-accelerated clustering
    - **Gradio** — web UI framework
    - **xFormers** — efficient attention for vision models
    - **OpenCV, NumPy, Matplotlib, Plotly** — data processing and visualization

### Step 4: Download Model Checkpoints

CASTLE uses several pretrained models. You can download them automatically or manually.

#### Option A: Automatic Download (Script)

```bash
pip install gdown  # Required for Google Drive downloads
bash download_ckpt.sh
```

#### Option B: Manual Download

Download each file and place it in the `ckpt/` directory:

| Checkpoint | Source | Download Link |
|------------|--------|---------------|
| `sam_vit_b_01ec64.pth` | Meta (SAM) | [Download](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) |
| `dinov2_vitb14_reg4_pretrain.pth` | Meta (DINOv2) | [Download](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth) |
| `R50_DeAOTL_PRE_YTB_DAV.pth` | Google Drive (DeAOT) | [Download](https://drive.google.com/uc?id=1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ) |
| `SwinB_DeAOTL_PRE_YTB_DAV.pth` | Google Drive (DeAOT) | [Download](https://drive.google.com/uc?id=1g4E-F0RPOx9Nd6J7tU9AE1TjsouL4oZq) |
| `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` | Google Drive (DINOv3) | [Download](https://drive.google.com/uc?id=18doehnHWWnz9zBtOdgYZ3XMTpgPYbYZ6) |
| `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | Google Drive (DINOv3) | [Download](https://drive.google.com/uc?id=195H5UHKJ0r4qRDY7Ly6WJrXGnpdlHMSu) |

!!! tip "Google Drive download issues"
    If Google Drive links fail due to quota limits, try:

    1. Open the link in a browser and download manually
    2. Wait a few hours and retry (quota resets periodically)
    3. Use `gdown` with the `--fuzzy` flag: `gdown --fuzzy '<URL>'`

### Step 5: Verify Installation

=== "Gradio Web UI"

    ```bash
    python app.py
    ```

    If everything is set up correctly, a Gradio web UI will launch at [http://localhost:7860](http://localhost:7860).

=== "CLI"

    ```bash
    castle --help
    ```

---

## Method 2: Docker

Docker provides a self-contained environment with all dependencies pre-configured.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support)

### Step 1: Clone and Prepare Checkpoints

```bash
git clone https://github.com/CASTLE-ai/castle-ai.git
cd castle-ai
pip install gdown
bash download_ckpt.sh
```

!!! note
    Checkpoints must be in the `ckpt/` directory before running the container. They are mounted as a volume, so you only need to download them once.

### Step 2: Build the Image

```bash
docker compose build
```

### Step 3: Run the Container

```bash
docker compose up
```

### Step 4: Access the UI

Open [http://localhost:7860](http://localhost:7860) in your browser.

The Docker setup automatically:

- Mounts `ckpt/` for model checkpoints (persistent)
- Mounts `projects/` for your analysis data (persistent)
- Mounts `demo/` for demo videos
- Reserves 1 NVIDIA GPU

---

## Method 3: Google Colab

The easiest way to try CASTLE — no local setup required.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CASTLE-ai/castle-ai/blob/main/notebooks/colab.ipynb)

!!! note "Performance"
    Free-tier Colab GPUs are significantly slower than local setups. For production use, a local installation or Docker is recommended.

A video walkthrough is also available:

[![CASTLE Quick Start](https://img.shields.io/badge/YouTube-CASTLE%20Demo-red?logo=youtube)](https://youtu.be/qzZlixEaKvQ)

---

## Troubleshooting

### CUDA version mismatch

```
RuntimeError: The NVIDIA driver on your system is too old
```

Ensure your NVIDIA driver supports CUDA 12.x. Check with `nvidia-smi`.

### PyTorch not finding GPU

```python
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))
```

If `False`, reinstall PyTorch with the correct CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).

### gdown failing (quota exceeded)

Google Drive has daily download quotas. Options:

1. Download manually via browser
2. Wait and retry later
3. Try: `gdown --fuzzy '<URL>'`

### cuml installation failures

`cuml-cu12` requires an NVIDIA GPU and CUDA 12.x. If installation fails:

- Verify CUDA is installed: `nvcc --version`
- Check CUDA version matches (12.x required)
- On systems without GPU, you may need to skip cuml (some clustering features will be unavailable)

### ffmpeg not found

Some video processing features require ffmpeg:

=== "Linux (Ubuntu/Debian)"

    ```bash
    sudo apt install ffmpeg
    ```

=== "macOS"

    ```bash
    brew install ffmpeg
    ```

=== "Windows"

    Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.
