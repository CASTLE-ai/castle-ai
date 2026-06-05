# Configuration Reference

All configurable parameters in CASTLE, consolidated in one place.

---

## Launch Options

### Gradio Web UI (`app.py`)

```bash
python app.py [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--root PATH` | Custom project storage path | `projects/` |
| `--share` | Enable Gradio public URL (auto-enabled on Colab) | `False` |

### CLI

```bash
castle [COMMAND] [OPTIONS]
```

Use `castle --help` or `castle <command> --help` for full option reference. Key environment variable:

| Variable | Description |
|----------|-------------|
| `CASTLE_STORAGE` | Default project storage directory |

---

## Model Configuration (`castle/configs/model_config.json`)

Defines paths and parameters for all models:

### SAM (Segmentation)

```json
{
    "sam_args": {
        "sam_checkpoint": "ckpt/sam_vit_b_01ec64.pth",
        "model_type": "vit_b",
        "generator_args": {
            "points_per_side": 16,
            "pred_iou_thresh": 0.8,
            "stability_score_thresh": 0.9,
            "crop_n_layers": 1,
            "crop_n_points_downscale_factor": 2,
            "min_mask_region_area": 200
        }
    }
}
```

| Parameter | Description |
|-----------|-------------|
| `sam_checkpoint` | Path to SAM weights |
| `model_type` | SAM variant (`vit_b`) |
| `points_per_side` | Auto-mask grid density |
| `pred_iou_thresh` | IoU prediction threshold |
| `stability_score_thresh` | Mask stability filter |
| `min_mask_region_area` | Minimum mask size (pixels) |

### DeAOT (Tracking)

```json
{
    "r50_deaotl": {
        "phase": "PRE_YTB_DAV",
        "model": "r50_deaotl",
        "model_path": "ckpt/R50_DeAOTL_PRE_YTB_DAV.pth"
    },
    "swinb_deaotl": {
        "phase": "PRE_YTB_DAV",
        "model": "swinb_deaotl",
        "model_path": "ckpt/SwinB_DeAOTL_PRE_YTB_DAV.pth"
    }
}
```

### DINOv2 / DINOv3 (Feature Extraction)

```json
{
    "dinov2_args": {
        "name": "dinov2_vitb14_reg",
        "path": "ckpt/dinov2_vitb14_reg4_pretrain.pth"
    },
    "dinov3_args": {
        "name": "dinov3_vitl16",
        "path": "ckpt/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    },
    "dinov3_vitb16_args": {
        "name": "dinov3_vitb16",
        "path": "ckpt/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    }
}
```

---

## Extraction Parameters

Configured in the **4. Extract Latent** tab:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Visual Model | `dinov3_vitb16` | 3 options | Feature extraction backbone (`dinov3_vitb16` default, 768-d; `dinov3_vitl16`, 1024-d; `dinov2_vitb14_reg4_pretrain`, 768-d) |
| ROI ID | `1` | Any tracked ROI | Which ROI to extract |
| Batch Size | auto | 1–256+ | Frames per batch. The CLI auto-sizes from free VRAM when `--batch-size` is omitted (and halves & retries on OOM); the UI default is 32 with an "Auto Batch Size" button. |
| Latent dtype | `float32` | float32 / float16 | Storage precision of the latent `.npz` (CLI `--latent-dtype`). `float16` halves file size and I/O. |
| Skip Existing | `True` | — | Don't re-extract existing files |

### Preprocessing

| Parameter | Default | Description |
|-----------|---------|-------------|
| Center ROI | `False` | Crop centered on ROI centroid |
| Center ROI ID | `1` | Which ROI to center on |
| Crop Width | `300` px | Crop window width |
| Crop Height | `300` px | Crop window height |
| Rotate (Tail) | `False` | Normalize orientation via tail ROI |
| Tail ROI ID | `2` | Which ROI defines tail direction |
| Remove Background | `False` | Zero out non-ROI pixels |

---

## UMAP Parameters

Configured in the **5. Behavior Microscope** tab. See [Behavior Analysis Tutorial](../tutorials/step4-analysis.md) for preset details.

### Custom Configuration Format

JSON array of UMAP stages:

```json
[
    {
        "n_neighbors": 100,
        "min_dist": 0.0,
        "n_components": 2,
        "standardize": true
    }
]
```

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `n_neighbors` | Nearest neighbors (local vs global structure) | 25–1000 |
| `min_dist` | Minimum distance in embedding | 0.0 (for clustering) |
| `n_components` | Output dimensions per stage | 2, 5, or 10 |
| `standardize` | Per-feature z-score of the raw features before stage 0 (first stage only). Default `true`. | `true` / `false` |

**Internal parameter** (in `myumap.py`): `n_epochs = 20000` — number of optimization epochs. Not exposed in the UI.

!!! note "Input standardization is on by default"
    The first (raw-feature) UMAP stage now z-scores each feature before fitting (`"standardize": true` in the default preset). This improves cluster separation but **changes embeddings versus older runs**, so the DBSCAN `eps` may need re-tuning. Set `"standardize": false` in the stage-0 config to reproduce a pre-standardization layout. Later stages run on low-dim UMAP output and are not standardized.

!!! tip "Reproducible UMAP runs"
    Every UMAP run records its resolved random seed, and each clustering session writes a `umap_log.jsonl` file — one JSON line per UMAP stage capturing the seed and resolved config. To reproduce an embedding exactly, reuse the logged seed (set `random_state` in the stage config) and run the CPU/deterministic path for bit-identical results.

---

## DBSCAN Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `eps` | `1.0` | 0.1–10.0 | Epsilon-neighborhood radius |

Smaller eps → more clusters. Larger eps → fewer clusters.

---

## Environment Variables

CASTLE auto-detects the runtime environment at startup — filesystem type (local
vs network such as CephFS/NFS), usable CPU count (honouring container cgroup
limits, not just the host core count), total/available RAM, and GPU/VRAM — and
applies safe defaults. Every decision can be overridden with the variables
below; set them in the shell **before** launching CASTLE.

**General**

| Variable | Description |
|----------|-------------|
| `COLAB_GPU` | Auto-detected in Google Colab; enables `--share` and Colab-specific paths |
| `HDF5_USE_FILE_LOCKING` | Forced to `FALSE` at `import castle` — POSIX file locking is unnecessary (CASTLE is single-writer / multi-reader) and hangs on network filesystems (CephFS/NFS). Set explicitly to override. |

**Multi-GPU (latent extraction)**

| Variable | Description |
|----------|-------------|
| `CASTLE_MULTI_GPU` | Opt-in multi-GPU for tracking **and** latent extraction (Gradio app or CLI). Default OFF. Set to `1` to enable. |
| `CASTLE_MULTI_GPU_DETERMINISTIC` | Opt-in: force cuDNN-deterministic during multi-GPU runs for per-GPU reproducibility, at a throughput cost. Default OFF (speed). |

**Worker & thread tuning** (auto-sized from usable CPUs; override only if needed)

| Variable | Default | Description |
|----------|---------|-------------|
| `CASTLE_EXTRACTION_WORKERS` / `CASTLE_NUM_WORKERS` | auto | Force the extraction DataLoader worker count. |
| `CASTLE_MAX_EXTRACTION_WORKERS` | `16` | Absolute cap on extraction workers. |
| `CASTLE_NETWORK_FS_WORKERS` | `8` | Lower worker cap when the mask file is on a network FS (fewer concurrent HDF5 readers). |
| `CASTLE_PREFETCH_FACTOR` | `2` | DataLoader `prefetch_factor`. |
| `CASTLE_PIN_MEMORY` | `1` | DataLoader `pin_memory` (set `0` to disable on memory-tight hosts). |
| `CASTLE_TORCH_THREADS` | auto | Cap torch intra-op threads during extraction (auto-capped on many-core / cgroup-limited boxes). |
| `CASTLE_RESERVED_CORES` | `4` | Cores left free for the OS when sizing pre-process pools. |
| `CASTLE_CENTROID_WORKERS` / `CASTLE_CENTROID_MAX_WORKERS` | auto / `16` | Pre-process centroid pool size / cap. |
| `CASTLE_PREPROCESS_WARP_WORKERS` | auto | Pre-process warpAffine / mask-transform pool size. |

**Latent buffering & scratch** (RAM-aware; big-RAM hosts keep latents resident in RAM instead of spilling to slow disk)

| Variable | Default | Description |
|----------|---------|-------------|
| `CASTLE_SCRATCH_DIR` | auto | Node-local directory for large temp files (latent memmaps, NVENC probes). Auto picks `/dev/shm` on big-RAM hosts, else local `/tmp`; **never** a network FS. |
| `CASTLE_MEMMAP_THRESHOLD_GB` | RAM-aware | Force the latent-buffer → disk spill threshold (overrides the RAM-aware default). |
| `CASTLE_LATENT_RAM_FRACTION` | `0.5` | On big-RAM hosts, fraction of available RAM the latent buffer may use before spilling. |
| `CASTLE_BIG_RAM_GB` | `128` | Total-RAM threshold (GiB) above which a host counts as "big RAM". |

**Video encode**

| Variable | Default | Description |
|----------|---------|-------------|
| `CASTLE_VIDEO_ENCODER` | `auto` | H.264 encoder for pre-process / mix videos: `auto` (NVENC, falling back to libx264), `nvenc`, or `x264`. |

**Detection overrides** (rarely needed — for exotic mounts / mis-detected containers)

| Variable | Description |
|----------|-------------|
| `CASTLE_FORCE_NETWORK_FS` | `1`/`0` to force network-filesystem handling on/off. |
| `CASTLE_USABLE_CPUS` | Override the detected usable CPU count. |
| `CASTLE_TOTAL_RAM_GB` | Override the detected total RAM (GiB). |

!!! tip "Multi-GPU (tracking + extraction)"
    When `CASTLE_MULTI_GPU=1` **and** ≥2 CUDA GPUs are visible:

    - **Batches** (track-all / extract-`All`, the CLI, `castle batch`) run **whole videos across GPUs** — one video per GPU concurrently (DeAOT tracking is sequential *within* a video, so this is the only way to parallelise it). Tracking is ~3× faster on two GPUs; extraction scales with batch size.
    - A **single** video's extraction still uses the within-video frame-range split across GPUs.

    The default uses fast cuDNN autotuning + fp16, exactly like single-GPU — so a video processed on the second GPU can differ from the single-GPU result by **fp16-rounding noise (~1e-2)**, which is negligible for downstream clustering (UMAP standardises it). For **exact per-GPU reproducibility** set `CASTLE_MULTI_GPU_DETERMINISTIC=1` (pins cuDNN deterministic; slower). Tracking masks are near-identical (mean-IoU ≈ 0.9999) either way.

    !!! warning
        A large multi-GPU batch is memory-heavy. **Stop the Gradio app first** if you launch one from the CLI — both competing for GPU + host RAM can exhaust memory.

---

## ProjectConfig (B-05)

The `ProjectConfig` dataclass (`castle/core/project_config.py`) provides typed, serializable configuration for the entire CASTLE processing pipeline. It is stored as `castle_config.json` in the project directory.

```python
from castle.core.project_config import ProjectConfig

cfg = ProjectConfig()                    # defaults
cfg = ProjectConfig.load('castle_config.json')  # from file
cfg.save('castle_config.json')           # to file
```

### Extraction Parameters (A-06)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `extraction.model` | `str` | `'dinov3_vitb16'` | Visual encoder model name |
| `extraction.roi_ids` | `List[int]` | `[1]` | ROI IDs to extract |
| `extraction.batch_size` | `int` | `32` | Frames per GPU batch |
| `extraction.bin_size` | `int` | `1` | Temporal binning (frames per sample) |
| `extraction.pooling_method` | `str` | `'weighted_average'` | `'weighted_average'` or `'multiscale'` |
| `extraction.pooling_scales` | `List[int]` | `[1, 2, 4]` | Grid divisions for multi-scale SPP (only used when `pooling_method='multiscale'`) |
| `extraction.feature_layers` | `Optional[List[int]]` | `None` | Layer indices for multi-layer extraction (e.g. `[3, 7, 11]`). `None` = last layer only. |

**Multi-scale pooling (A-06)** divides the patch grid into s×s regions for each scale s, computes mask-weighted average in each region, then concatenates. For example, `pooling_scales=[1, 2, 4]` produces a `(1 + 4 + 16) × 768 = 16128`-dim vector.

**Multi-layer extraction (A-06)** concatenates features from multiple transformer layers (e.g. shallow layers capture texture/edges, deep layers capture semantics). For example, `feature_layers=[3, 7, 11]` on the default DINOv3-ViT-B/16 (`dinov3_vitb16`, 768-d) produces `3 × 768 = 2304`-dim features per patch.

### Preprocessing Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `extraction.preprocess.center_roi` | `bool` | `False` | Center crop on ROI centroid |
| `extraction.preprocess.center_roi_id` | `int` | `1` | ROI to center on |
| `extraction.preprocess.crop_width` | `int` | `300` | Crop width (px) |
| `extraction.preprocess.crop_height` | `int` | `300` | Crop height (px) |
| `extraction.preprocess.rotate_roi_tail` | `bool` | `False` | Normalize orientation via tail ROI |
| `extraction.preprocess.rotate_roi_tail_id` | `int` | `2` | Tail ROI ID |
| `extraction.preprocess.remove_background` | `bool` | `False` | Zero out non-ROI pixels |

### Tracking Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tracking.model` | `str` | `'r50_deaotl'` | DeAOT model variant |
| `tracking.smart_filter_ratio` | `float` | `0.1` | Mask area filter threshold |
| `tracking.batch_size` | `int` | `16` | Tracking batch size |

### Clustering Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `clustering.method` | `str` | `'dbscan'` | Clustering algorithm |
| `clustering.eps` | `float` | `1.0` | DBSCAN epsilon |
| `clustering.umap_stages` | `List[UMAPConfig]` | `[{n_neighbors:100, ..., standardize:true}]` | Multi-stage UMAP configs |

The stage-0 `UMAPConfig` defaults to `standardize: true` (per-feature z-score of the raw features). The top-level `master_seed` field (default `42`) seeds every stochastic component; resolved per-stage UMAP seeds are recorded in the session's `umap_log.jsonl` for reproducibility (see [UMAP Parameters](#umap-parameters)).

---

## Device Selection

CASTLE auto-detects the compute device:

| Platform | Device |
|----------|--------|
| macOS (Apple Silicon) | `mps` |
| Linux/Windows with NVIDIA GPU | `cuda` |
| No GPU | `cpu` |
