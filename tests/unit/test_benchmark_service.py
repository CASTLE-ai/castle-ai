"""Reproducible accuracy benchmark (`castle benchmark`).

Guards the citable-report shape, the dataset registry, and — critically — the
per-frame ground-truth alignment (a perfect GT must score NMI/ARI = 1.0; a prior
bug compared per-datapoint labels to per-frame GT and reported ~0).
"""

import json
import os

import numpy as np
import pandas as pd

from castle.service.benchmark_service import (
    KNOWN_DATASETS, resolve_dataset, run_accuracy_benchmark,
)


def test_calms21_registered_with_doi():
    assert "calms21" in KNOWN_DATASETS
    assert KNOWN_DATASETS["calms21"].doi == "10.22002/D1.1991"


def test_resolve_dataset_known_and_overrides():
    cb = resolve_dataset("calms21")
    assert cb.name == "CalMS21" and cb.doi == "10.22002/D1.1991"
    # explicit overrides win; unknown key becomes a custom citation
    custom = resolve_dataset("myset", doi="10.1/x", url="http://x")
    assert custom.doi == "10.1/x" and custom.url == "http://x"
    assert resolve_dataset(None) is None


def _make_clustered_project(tmp_path, labels):
    proj = tmp_path / "proj"
    cluster = proj / "cluster"
    cluster.mkdir(parents=True)
    pd.DataFrame({"behavior": labels}).to_csv(cluster / "time_series_v1.csv", index=False)
    return str(proj)


def test_perfect_ground_truth_scores_one(tmp_path):
    # 3 behaviours over 300 frames; GT identical to labels -> NMI/ARI == 1.0
    labels = np.repeat([0, 1, 2], 100)
    proj = _make_clustered_project(tmp_path, labels)
    gt = tmp_path / "gt.csv"
    pd.DataFrame({"behavior": labels}).to_csv(gt, index=False)

    report = run_accuracy_benchmark(
        proj, ground_truth_path=str(gt), dataset="calms21", timestamp="t",
    )
    assert "error" not in report
    assert report["ground_truth_provided"] is True
    assert report["metrics"]["nmi"] == 1.0
    assert report["metrics"]["ari"] == 1.0
    assert report["dataset"]["doi"] == "10.22002/D1.1991"


def test_unrelated_ground_truth_scores_low(tmp_path):
    labels = np.repeat([0, 1, 2], 100)
    proj = _make_clustered_project(tmp_path, labels)
    rng = np.random.default_rng(0)
    gt = tmp_path / "gt.csv"
    pd.DataFrame({"behavior": rng.integers(0, 3, size=300)}).to_csv(gt, index=False)

    report = run_accuracy_benchmark(proj, ground_truth_path=str(gt), timestamp="t")
    assert report["metrics"]["nmi"] < 0.2  # random GT -> near-zero mutual info


def test_report_files_written_and_no_gt_path(tmp_path):
    labels = np.repeat([0, 1], 50)
    proj = _make_clustered_project(tmp_path, labels)
    report = run_accuracy_benchmark(proj, ground_truth_path=None, timestamp="t")
    assert report["ground_truth_provided"] is False
    rdir = report["report_dir"]
    assert os.path.isfile(os.path.join(rdir, "benchmark_report.json"))
    assert os.path.isfile(os.path.join(rdir, "benchmark_report.md"))
    saved = json.loads(open(os.path.join(rdir, "benchmark_report.json")).read())
    assert saved["benchmark"] == "castle-accuracy-benchmark"
    assert "provenance" in saved


def test_missing_cluster_dir_returns_error(tmp_path):
    report = run_accuracy_benchmark(str(tmp_path / "nope"), timestamp="t")
    assert "error" in report
