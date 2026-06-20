"""Unit tests for the castle.service.export_service file collectors.

These collectors decide what lands in an export ZIP. They were previously
exercised only through the Gradio UI; these CPU tests pin the (src, archive_name)
contracts directly so a refactor can't silently drop a category or mis-name a path.
"""

import os

from castle.service import export_service as es


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("x")


def _archive_names(pairs):
    return sorted(arc for _src, arc in pairs)


def test_collect_masks(tmp_path):
    pp = str(tmp_path)
    _touch(os.path.join(pp, "track", "vidA.mp4", "mask_list.h5"))
    _touch(os.path.join(pp, "track", "vidB.mp4", "mask_list.h5"))
    out = es._collect_masks(pp)
    assert _archive_names(out) == [
        os.path.join("track", "vidA.mp4", "mask_list.h5"),
        os.path.join("track", "vidB.mp4", "mask_list.h5"),
    ]
    assert all(os.path.isfile(src) for src, _ in out)


def test_collect_latent_walks_tree(tmp_path):
    pp = str(tmp_path)
    _touch(os.path.join(pp, "latent", "dinov3_vitb16", "vidA_ROI_1.npz"))
    _touch(os.path.join(pp, "latent", "dinov3_vitb16", "vidA_ROI_1.npz.json"))
    out = es._collect_latent(pp)
    names = _archive_names(out)
    assert os.path.join("latent", "dinov3_vitb16", "vidA_ROI_1.npz") in names
    assert os.path.join("latent", "dinov3_vitb16", "vidA_ROI_1.npz.json") in names


def test_collect_cluster_results_globs_expected_patterns(tmp_path):
    pp = str(tmp_path)
    _touch(os.path.join(pp, "cluster", "id.csv"))
    _touch(os.path.join(pp, "cluster", "cluster_init_.npz"))
    _touch(os.path.join(pp, "cluster", "time_series_vidA.csv"))
    _touch(os.path.join(pp, "cluster", "ignored.txt"))  # not collected
    names = _archive_names(es._collect_cluster_results(pp))
    assert os.path.join("cluster", "id.csv") in names
    assert os.path.join("cluster", "cluster_init_.npz") in names
    assert os.path.join("cluster", "time_series_vidA.csv") in names
    assert os.path.join("cluster", "ignored.txt") not in names


def test_collect_annotations_only_for_session(tmp_path):
    pp = str(tmp_path)
    _touch(os.path.join(pp, "cluster", "sessions", "session_001", "annotations.csv"))
    assert es._collect_annotations(pp, "") == []          # no session → nothing
    assert es._collect_annotations(pp, "missing") == []   # wrong session → nothing
    out = es._collect_annotations(pp, "session_001")
    assert _archive_names(out) == [
        os.path.join("cluster", "sessions", "session_001", "annotations.csv")
    ]


def test_collect_grid_videos_and_sources(tmp_path):
    pp = str(tmp_path)
    _touch(os.path.join(pp, "cluster", "grid_videos", "g0.mp4"))
    _touch(os.path.join(pp, "sources", "vidA.mp4"))
    assert _archive_names(es._collect_grid_videos(pp)) == [
        os.path.join("cluster", "grid_videos", "g0.mp4")
    ]
    assert _archive_names(es._collect_source_videos(pp)) == [
        os.path.join("sources", "vidA.mp4")
    ]


def test_collect_analysis_top_level_and_per_session(tmp_path):
    pp = str(tmp_path)
    _touch(os.path.join(pp, "analysis", "ethogram.png"))
    _touch(os.path.join(pp, "cluster", "sessions", "session_001", "analysis", "metrics.json"))
    names = _archive_names(es._collect_analysis(pp))
    assert os.path.join("analysis", "ethogram.png") in names
    assert os.path.join("cluster", "sessions", "session_001", "analysis", "metrics.json") in names


def test_collectors_empty_on_missing_dirs(tmp_path):
    pp = str(tmp_path / "empty")
    os.makedirs(pp)
    assert es._collect_masks(pp) == []
    assert es._collect_latent(pp) == []
    assert es._collect_cluster_results(pp) == []
    assert es._collect_grid_videos(pp) == []
    assert es._collect_analysis(pp) == []
    assert es._collect_source_videos(pp) == []
