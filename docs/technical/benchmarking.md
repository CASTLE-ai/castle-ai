# Accuracy benchmarking

`castle benchmark` scores a clustered project against ground-truth labels and
writes a **reproducible, citable report** (accuracy + unsupervised quality +
the full run environment). It is dataset-agnostic: any project plus a labelled
CSV can be scored, and a result can be attributed to a public DOI'd dataset.

## Quick start

```bash
# Accuracy vs. a ground-truth CSV (a `behavior` column aligned to the frames)
castle benchmark run my_project --gt labels.csv

# Attribute the result to a public DOI'd dataset
castle benchmark run my_project --gt labels.csv --dataset calms21

# List the registered citable datasets
castle benchmark datasets
```

The report is written to `<project>/benchmark/benchmark_report.{json,md}`.

## What it reports

When `--gt` is supplied, the **accuracy** metrics compare CASTLE's per-frame
cluster labels to the ground truth:

- **NMI** — normalized mutual information
- **ARI** — adjusted Rand index
- **V-measure**, **Homogeneity**, **Completeness**

Always reported (no ground truth needed) — **unsupervised quality**: temporal
coherence, silhouette (sampled), Calinski-Harabasz, Davies-Bouldin, median bout
duration. Without `--gt`, only these are computed and the report says so.

Ground truth and labels are aligned per **original frame** (truncated to the
shorter of the two). The labelled CSV needs a single `behavior` column.

## Reproducibility

Every report embeds the run environment (CASTLE version, device, library and
GPU stack) via the same provenance used across CASTLE artifacts. To reproduce a
number: re-run with the same project, the same ground-truth CSV, and the same
`--seed` (see [reproducibility](reproducibility.md)). cuML-GPU and CPU UMAP give
different embeddings, so the recorded stack tells a non-reproduction apart from
a backend mismatch.

## Datasets

The harness is dataset-agnostic — point `--gt` at any labelled CSV. For a
citable public standard it registers:

| Key | Name | DOI |
|-----|------|-----|
| `calms21` | CalMS21 (mouse dyadic social interactions) | [`10.22002/D1.1991`](https://doi.org/10.22002/D1.1991) |

!!! note "CalMS21 is trajectory-centric"
    CalMS21 ships keypoint trajectories; CASTLE is video+mask based, so running
    it on CalMS21 requires the dataset's raw videos and a tracking pass. Use it
    as the citable target once the videos are obtained, or benchmark on your own
    video + human-annotated labels and deposit the dataset (e.g. Zenodo) to mint
    a DOI. Pass a custom citation with `--dataset <name> --doi <doi> --url <url>`.
