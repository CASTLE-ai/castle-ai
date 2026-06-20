# Environment Variables

CASTLE reads a number of `CASTLE_*` environment variables at runtime to tune
device selection, parallelism, memory budgets, and storage. **All are optional** —
each has a built-in default — and they are read fresh from the OS environment (not
from any config file). Boolean variables accept `1/true/yes/on` for true and
`0/false/no/off` for false unless noted otherwise.

You usually don't need any of these: CASTLE auto-detects the device, usable CPU
count (cgroup/affinity aware), and RAM budget. Reach for them when adapting to an
unusual environment — a shared cluster, a container with capped resources, a
network filesystem, or a reproducibility run.

## Device & determinism

| Variable | Default | Type | Effect |
|----------|---------|------|--------|
| `CASTLE_DEVICE` | `auto` | str | Global compute-device preference for the CLI (`auto`/`cuda`/`cpu`); `auto` detects CUDA/MPS/CPU. Also the `--device` CLI option. |
| `CASTLE_GPU_DEVICE` | idlest GPU | str | Pins single-GPU ops to a specific card via `cuda:N`. Honored by SAM segmentation, tracking, cuML clustering, and distance helpers; when set it overrides automatic "idlest GPU" selection. |
| `CASTLE_PREPARE_DEVICE` | auto | str | Forces the Prepare/PCA solve onto `cpu` or `cuda`; unset, CASTLE picks CUDA only if a GPU has enough free VRAM for the D×D solve, else CPU. |
| `CASTLE_STRICT_CUDA` | `False` | bool | Forces bit-identical CUDA output (`cudnn.deterministic` + `use_deterministic_algorithms`); ~10% slower. Also the `--strict-cuda` CLI option. |
| `CASTLE_SEED` | `42` | int | Master seed for every stochastic component except UMAP (which keeps its own seed/re-roll UX). Read by both the CLI (`--seed`) and the Gradio app at startup. |
| `CASTLE_MULTI_GPU` | off | bool | Opt-in gate for multi-GPU extraction/tracking; only enables when set truthy **and** more than one CUDA device is visible. |
| `CASTLE_MULTI_GPU_DETERMINISTIC` | on | bool | When multi-GPU is active, controls the slower per-GPU-reproducible cuDNN-deterministic path. **Inverted sense:** treated as enabled unless explicitly set falsey, so leaving it unset keeps determinism on. |

**Device precedence.** `CASTLE_DEVICE` is the coarse CLI-level preference (cuda vs cpu
vs auto). `CASTLE_GPU_DEVICE` is finer-grained — it pins *which* CUDA card single-GPU
ops land on, overriding the automatic idlest-GPU pick, but does not by itself force
CUDA on/off. `CASTLE_PREPARE_DEVICE` overrides only the Prepare/PCA stage's
cpu-vs-cuda decision. They are independent layers.

## Workers & parallelism

| Variable | Default | Type | Effect |
|----------|---------|------|--------|
| `CASTLE_USABLE_CPUS` | detected | int | Overrides the affinity/cgroup/`os.cpu_count()`-derived usable-core count that all pools size against. |
| `CASTLE_RESERVED_CORES` | `4` | int | Cores kept free when sizing default worker pools (`pool size = cpu_count − reserved`). |
| `CASTLE_EXTRACTION_WORKERS` / `CASTLE_NUM_WORKERS` | auto | int | Force the DataLoader worker count for extraction (authoritative; bypasses the caps below). |
| `CASTLE_MAX_EXTRACTION_WORKERS` | `16` | int | Absolute cap on extraction workers. |
| `CASTLE_NETWORK_FS_WORKERS` | `8` | int | Lower worker cap applied when the working path is on a network filesystem (too many workers thrash a shared FS). |
| `CASTLE_MIX_WORKERS` | auto | int | Pool size for the video-mix stage; `1` forces serial. |
| `CASTLE_CENTROID_WORKERS` | auto | int | Pool size for centroid / body-head extraction; `1` forces serial. |
| `CASTLE_CENTROID_MAX_WORKERS` | `16` | int | Hard cap on centroid workers (HDF5 I/O saturates before more processes help). |
| `CASTLE_PREPROCESS_WARP_WORKERS` | auto | int | Pool size for preprocessing encode / mask-warp work; `1` forces serial. |
| `CASTLE_TORCH_THREADS` | auto | int | Overrides intra-op torch thread count during extraction; unset, capped to ≤8 on large/cgroup-limited hosts. |
| `CASTLE_PREPARE_LOADERS` | `4` | int | Cap on concurrent file-loader threads during Prepare. |
| `CASTLE_PREPARE_PREFETCH` | on | bool | Set falsey to disable Prepare's background prefetch (no load/GPU overlap). |
| `CASTLE_PREFETCH_FACTOR` | `2` | int | DataLoader `prefetch_factor` for extraction; raise to hide read latency on high-latency network filesystems. |
| `CASTLE_TRACK_WARMUP_FRAMES` | `256` | int | Warmup frame count for the dual-GPU tracking split decision. |

## Memory & RAM budgets

| Variable | Default | Type | Effect |
|----------|---------|------|--------|
| `CASTLE_TOTAL_RAM_GB` | detected | float | Overrides total-RAM detection (otherwise from `/proc/meminfo`, capped by any cgroup `memory.max`). |
| `CASTLE_BIG_RAM_GB` | `128` | float | RAM threshold (GiB) above which the host is treated as a "big-RAM box" — enabling RAM-resident buffers and `/dev/shm` spill instead of slow disk. |
| `CASTLE_MEMMAP_THRESHOLD_GB` | `2.0` | float | Latent-buffer size (GiB) above which CASTLE spills to a disk memmap instead of RAM. When set it is honored exactly (and disables the big-RAM auto-scaling). |
| `CASTLE_LATENT_RAM_FRACTION` | `0.5` | float | On a big-RAM box (and only when `CASTLE_MEMMAP_THRESHOLD_GB` is unset), fraction of available RAM allowed for resident latent buffers before spilling. |
| `CASTLE_UMAP_RAM_FRACTION` | `0.85` | float | Host-RAM fraction the UMAP guard treats as usable (CPU/host side). |
| `CASTLE_UMAP_VRAM_FRACTION` | `0.85` | float | VRAM fraction the GPU UMAP guard treats as usable before falling back. |
| `CASTLE_PREPARE_RAM_MARGIN_GB` | `10` | float | RAM headroom (GiB) Prepare keeps free when deciding how many files to inflate concurrently (the OOM guard). |
| `CASTLE_PIN_MEMORY` | caller's choice | bool | Forces DataLoader pinned-memory allocation on/off during extraction; disable on memory-pressured / cgroup-limited boxes. |

## Storage & scratch

| Variable | Default | Type | Effect |
|----------|---------|------|--------|
| `CASTLE_STORAGE` | unset | path | Default project storage root for the CLI when `--storage/-s` is not given. |
| `CASTLE_CONFIG` | unset | path | Optional JSON/YAML config file loaded as subcommand default overrides (also `--config/-c`). |
| `CASTLE_SCRATCH_DIR` | auto | path | Node-local directory for large temp files (memmaps, encode probes); unset, CASTLE auto-picks `/dev/shm` on big-RAM boxes or a non-network tmpdir — never silently a network FS. |
| `CASTLE_FORCE_NETWORK_FS` | auto-probe | bool | Overrides network-filesystem detection (`1` = force "network", `0` = force "local"); for exotic mounts and tests. |

## Encoding & diagnostics

| Variable | Default | Type | Effect |
|----------|---------|------|--------|
| `CASTLE_VIDEO_ENCODER` | `auto` | str | Video encoder for preprocessing/mix output ∈ `{auto, nvenc, x264}`; `auto`/`nvenc` validate NVENC at the real frame size and fall back to libx264 if needed. |
| `CASTLE_PREPROCESS_ENCODER` | unset | str | Legacy alias for `CASTLE_VIDEO_ENCODER`, read only as a fallback when the latter is unset. |
| `CASTLE_FAULTHANDLER` | on | bool | Enables the stdlib faulthandler so `kill -USR1 <pid>` dumps all Python thread stacks (diagnose hangs without py-spy/root); set falsey to opt out. |

!!! tip "Reproducibility runs"
    For a paper-grade, deterministic run: set a fixed `CASTLE_SEED`, add
    `CASTLE_STRICT_CUDA=1` (bit-identical CUDA, ~10% slower), and prefer the CPU
    UMAP path. Each saved artifact (latent sidecar, prepare-cache manifest, export
    `run_manifest.json`) records the resolved library/GPU stack so a reproduction
    can be told apart from a backend mismatch.
