# System Requirements

## Hardware

### GPU (Recommended)

CASTLE is designed for NVIDIA GPUs with CUDA 12.x support.

- **Minimum VRAM:** 6 GB (e.g., RTX 2060, RTX 3060)
- **Recommended VRAM:** 8+ GB (e.g., RTX 3070+, RTX 4060+)

Models are loaded sequentially (not simultaneously), so peak VRAM is determined by the largest single model:

| Model | VRAM Usage |
|-------|-----------|
| SAM ViT-B | ~2 GB |
| DeAOT R50 | ~3 GB |
| DINOv2 ViT-B | ~2 GB |
| **Peak (any single stage)** | **~4–5 GB** |

!!! note "CPU-Only Mode"
    CASTLE can run without a GPU, but performance will be significantly slower — particularly for tracking (SAM + DeAOT) and feature extraction (DINOv2/v3). GPU-accelerated clustering via cuml will be unavailable.

### RAM

- **Minimum:** 16 GB
- **Recommended (large datasets):** 32 GB

### Storage

- **~4 GB** for model checkpoints
- Additional space for project data (videos, extracted features, results)

---

## Software

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| CUDA | 12.x (12.6 tested in Docker) |
| OS | Linux (primary), macOS (limited GPU support), Windows (limited) |
| Docker (optional) | 20.10+ with NVIDIA Container Toolkit |

---

## Tested Configurations

| Configuration | GPU | VRAM | RAM | OS | Status |
|---------------|-----|------|-----|----|--------|
| Dev workstation | NVIDIA RTX 4090 | 24 GB | 32 GB DDR4 3200 MHz | Linux (Ubuntu) | ✅ Verified |

---

## Processing Time Benchmarks

Benchmarked on: Intel i7-12700, NVIDIA RTX 4090, 32 GB DDR4 3200 MHz.
Test video: 720×720, 30 fps, 30 minutes long.

### Per-Stage Timing (30-min video)

| Stage | Time | Notes |
|-------|------|-------|
| SAM + DeAOT tracking | ~25 min | 0.42× video length per ROI (batch_size=16) |
| Mask statistics | ~3 min | 0.1× video length |
| DINOv2 ViT-B extraction | ~84 min | 0.4× video length per ROI × 7 ROIs (batch_size=5) |
| Manual labeling | ~3 min | Negligible (routine) |
| Behavior Microscope | ~30 min | Interactive exploration |
| **Total** | **~130 min machine + ~33 min human** | **Routine workflow** |

!!! tip "DINOv2 is the bottleneck"
    Feature extraction dominates the machine time. Increasing batch size (if VRAM allows) or reducing the number of ROIs can significantly reduce total time.

### Comparison with Other Frameworks

**Cold Start** — first-time setup including training/labeling (30-min video):

| Framework | Machine Time | Human Time | Total |
|-----------|-------------|------------|-------|
| **CASTLE** | 2.17 h | 1.38 h | **3.55 h** |
| KPMS | 3.21 h | 3.50 h | 6.71 h |
| B-SOiD | 4.81 h | 5.77 h | 10.58 h |

**Routine** — subsequent analyses after initial setup (30-min video):

| Framework | Machine Time | Human Time | Total |
|-----------|-------------|------------|-------|
| CASTLE | 2.17 h | 0.55 h | 2.72 h |
| **KPMS** | 0.81 h | 0.53 h | **1.34 h** |
| B-SOiD | 1.06 h | 0.35 h | 1.41 h |

!!! note "Cold Start vs Routine"
    CASTLE is the fastest for **cold-start** scenarios (new experiment, no prior labels) because it requires no keypoint annotation or classifier training. For **routine** analyses of the same paradigm, supervised methods like KPMS and B-SOiD are faster since they reuse trained models, while CASTLE must re-extract DINOv2 features each time.

---

## Notes

- **Linux** is the primary development and testing platform
- **macOS** has limited support — Apple Silicon GPUs are not supported (no CUDA); CPU-only mode works
- **Windows** should work but is less tested; WSL2 with GPU passthrough is recommended
