"""Per-video ethogram tests (2026-06 manual-test feedback).

The Analysis ethogram used to pool every video's frames into one sequence with a
single fps, which (a) scaled durations wrongly for any video not at that fps and
(b) merged bouts across video boundaries. These tests lock the per-video fix:
each video's ethogram uses its own fps and bouts never cross a video boundary.
"""

import csv
import os

import pandas as pd
import pytest


def _write_project(tmp_path, time_series, clusters=((0, "rest", "#111111"), (1, "move", "#222222"))):
    """Create a minimal project with id.csv + per-video per-frame time_series CSVs."""
    proj = tmp_path / "proj"
    (proj / "cluster").mkdir(parents=True)
    (proj / "sources").mkdir(parents=True)
    pd.DataFrame(
        [{"Id": c[0], "Name": c[1], "Color": c[2]} for c in clusters]
    ).to_csv(proj / "cluster" / "id.csv", index=False)
    for basename, labels in time_series.items():
        pd.DataFrame({"behavior": labels}).to_csv(
            proj / "cluster" / f"time_series_{basename}.csv", index=False
        )
    return str(proj)


def test_compute_video_ethogram_uses_given_fps(tmp_path):
    from castle.service.ethogram_service import compute_video_ethogram

    proj = _write_project(tmp_path, {"A": [0, 0, 0, 1, 1]})
    etho = compute_video_ethogram(proj, "A", cluster_names={0: "rest", 1: "move"}, fps=24.0)

    assert etho.fps == 24.0
    bs = etho.bout_stats[1]
    assert bs.n_bouts == 1                       # one label-1 run
    assert abs(bs.mean_duration_s - 2 / 24.0) < 1e-9   # 2 frames @ 24 fps


def test_per_video_fps_and_no_cross_video_merge(tmp_path, monkeypatch):
    from castle.service import ethogram_service as es

    # Both videos carry a label-1 run touching the A-end / B-start boundary.
    proj = _write_project(tmp_path, {
        "A": [0, 0, 0, 1, 1],   # label 1 at the END
        "B": [1, 1, 0, 0, 0],   # label 1 at the START
    })

    fps_map = {"A": 24.0, "B": 60.0}
    monkeypatch.setattr(
        es, "_video_fps",
        lambda project_path, name, default=30.0: fps_map[os.path.splitext(os.path.basename(name))[0]],
    )

    out = tmp_path / "etho_out"
    es.export_ethogram_csv(str(proj), str(out))

    # bout_stats.csv: long-format with a `video` column, both videos present.
    with open(out / "bout_stats.csv") as f:
        stats = list(csv.DictReader(f))
    assert "video" in stats[0]
    assert {r["video"] for r in stats} == {"A", "B"}

    # bouts.csv: label-1 must be TWO separate bouts (one per video), NOT one
    # merged 4-frame bout spanning the boundary.
    with open(out / "bouts.csv") as f:
        label1 = [r for r in csv.DictReader(f) if int(r["cluster_id"]) == 1]
    by_video = {r["video"]: r for r in label1}
    assert set(by_video) == {"A", "B"}
    assert int(by_video["A"]["duration_frames"]) == 2
    assert int(by_video["B"]["duration_frames"]) == 2
    # Per-video fps: identical 2-frame bout → different seconds (24 vs 60 fps).
    assert abs(float(by_video["A"]["duration_seconds"]) - 2 / 24.0) < 1e-6
    assert abs(float(by_video["B"]["duration_seconds"]) - 2 / 60.0) < 1e-6

    # Per-video transition matrices are written.
    assert (out / "transition_matrix_A.csv").exists()
    assert (out / "transition_matrix_B.csv").exists()


def test_compute_video_ethogram_missing_csv_raises(tmp_path):
    from castle.service.ethogram_service import compute_video_ethogram

    proj = _write_project(tmp_path, {"A": [0, 1, 0]})
    with pytest.raises(FileNotFoundError):
        compute_video_ethogram(proj, "does_not_exist", fps=30.0)
