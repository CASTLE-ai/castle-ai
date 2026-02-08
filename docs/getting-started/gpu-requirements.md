# System Requirements

## Hardware

### GPU (Recommended)

CASTLE is designed for NVIDIA GPUs with CUDA 12.x support.

- **Minimum VRAM:** [HUMAN TO CONFIRM]
- **Recommended VRAM:** [HUMAN TO CONFIRM]

!!! note "CPU-Only Mode"
    CASTLE can run without a GPU, but performance will be significantly slower — particularly for tracking (SAM + DeAOT) and feature extraction (DINOv2/v3). GPU-accelerated clustering via cuml will be unavailable.

### RAM

- **Minimum:** [HUMAN TO CONFIRM]
- **Recommended (large datasets):** [HUMAN TO CONFIRM]

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

!!! note "Under Construction"
    The following table will be populated with confirmed working configurations.

| Configuration | GPU | VRAM | RAM | OS | Status |
|---------------|-----|------|-----|----|--------|
| [HUMAN TO CONFIRM] | | | | | |

---

## Notes

- **Linux** is the primary development and testing platform
- **macOS** has limited support — Apple Silicon GPUs are not supported (no CUDA); CPU-only mode works
- **Windows** should work but is less tested; WSL2 with GPU passthrough is recommended
