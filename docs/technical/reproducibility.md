# Reproducibility

CASTLE is designed so a clustering result can be reproduced and so every saved
artifact records *how* it was produced. This page explains the seed controls, the
deterministic path, and where the provenance lives.

## Capturing your environment

Print (and save) the exact software/hardware stack that produced a result:

```bash
castle env
```

This is the same provenance CASTLE stamps into its outputs (see below). Paste it
into bug reports and keep it alongside published results.

## Seeds

| Control | Scope |
|---|---|
| `CASTLE_SEED` (env) / `--seed` (CLI) | Master seed for every stochastic component **except** UMAP. Default `42`. Applied by the CLI and at Gradio app startup. |
| UMAP seed box (Behavior Microscope) | UMAP keeps its own per-stage seed. Leave blank to draw a fresh seed each run; enter an integer to lock it. The resolved seed is logged one line per stage to the session's `umap_log.jsonl`. |

To reproduce an embedding, reuse the seed recorded in `umap_log.jsonl`.

## Determinism

- **CPU UMAP path** is deterministic given a fixed seed — use it for bit-identical
  embeddings.
- **GPU (cuML) UMAP** is *not* bit-reproducible even with a fixed seed (documented
  RAPIDS behavior). It is fast; use the CPU path when you need exact reproduction.
- **`CASTLE_STRICT_CUDA=1`** forces bit-identical CUDA elsewhere (cuDNN
  deterministic + `use_deterministic_algorithms`), ~10% slower — use for
  paper-grade runs.
- cuML-GPU and sklearn/umap-learn-CPU produce **different** embeddings. Which
  backend actually ran is recorded in the provenance (below), so a non-reproduction
  can be told apart from a backend mismatch.

## Where provenance lives

Every CASTLE artifact carries the run environment (CASTLE version, Python, key
library versions — torch/numpy/scikit-learn/umap-learn/cuml — CUDA/cuDNN, and GPU
model):

| Artifact | Provenance |
|---|---|
| Latent feature files | `*.npz.json` sidecar — `environment` block + model name + seed |
| Prepare cache | `prepared/<id>/meta.json` — `environment` + PCA/seed/sources |
| Export bundle | `run_manifest.json` inside the ZIP — environment + selected components + project/session info |
| HTML report | footer line — CASTLE + library/GPU stack |
| NWB export | `source_script` field; `session_start_time` is the recording start (pass it explicitly for a spec-correct file) |

## Reproducing a published result — checklist

1. Recreate the environment from the recorded versions (`castle env` / the artifact
   `environment` blocks). Match `torch`, `numpy`, `scikit-learn`, `umap-learn`, and
   (if used) `cuml`.
2. Use the **same backend** the result was produced on (GPU vs CPU — see the
   recorded `device`). For bit-identical embeddings, use the **CPU UMAP path**.
3. Set `CASTLE_SEED` to the recorded master seed and the UMAP seed box to the seed
   logged in `umap_log.jsonl`.
4. For bit-identical CUDA on the rest of the pipeline, set `CASTLE_STRICT_CUDA=1`.
5. Compare against the recorded `cluster_*.npz` / `time_series_*.csv`.
