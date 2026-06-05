"""Golden behavior-data-contract tests (PR1 acceptance criteria).

These lock the semantic contract in ``docs/behavior_data_contract.md`` against a
minimal multi-video, mixed-fps, with-noise project. They are written FIRST and
marked ``xfail`` until the PR1 implementation (Stage 1-3) lands; the xfail
markers are removed in the final PR1 commit once everything passes.

Golden project::

    video1: fps=10, labels=[0, 0, -1, 1, 1]   # -1 (noise) in the middle
    video2: fps=20, labels=[1, 1,  0, 0]      # label-1 run at the START

The label-1 run touches the video1-end / video2-start boundary: pooling merges
them into one spurious 4-frame bout; the contract requires TWO separate bouts.
"""

import os

import pandas as pd
import pytest

# Remove this marker (and the import) in the final PR1 commit once Stage 1-3 land.
pr1_pending = pytest.mark.xfail(
    reason="behavior-data contract implemented across PR1 Stage 1-3",
    strict=False,
)


def _write_project(tmp_path, time_series, reasons=None,
                   clusters=((0, "rest", "#111111"), (1, "move", "#222222"))):
    """Minimal project: id.csv + per-video time_series CSVs (+ optional reasons)."""
    proj = tmp_path / "proj"
    (proj / "cluster").mkdir(parents=True)
    (proj / "sources").mkdir(parents=True)
    pd.DataFrame(
        [{"Id": c[0], "Name": c[1], "Color": c[2]} for c in clusters]
    ).to_csv(proj / "cluster" / "id.csv", index=False)
    for basename, labels in time_series.items():
        cols = {"behavior": labels}
        if reasons is not None and basename in reasons:
            cols["exclude_reason"] = reasons[basename]
        pd.DataFrame(cols).to_csv(
            proj / "cluster" / f"time_series_{basename}.csv", index=False
        )
    return str(proj)


@pytest.fixture
def golden_project(tmp_path, monkeypatch):
    from castle.service import ethogram_service as es

    proj = _write_project(
        tmp_path,
        time_series={"video1": [0, 0, -1, 1, 1], "video2": [1, 1, 0, 0]},
        reasons={"video1": [0, 0, 1, 0, 0], "video2": [0, 0, 0, 0]},  # 1 == dbscan_noise
    )
    fps_map = {"video1": 10.0, "video2": 20.0}
    monkeypatch.setattr(
        es, "_video_fps",
        lambda project_path, name, default=30.0:
            fps_map[os.path.splitext(os.path.basename(name))[0]],
    )
    return proj


@pr1_pending
def test_invariant_no_cross_video_bout_merge(golden_project):
    """C-3: label-1 is two per-video bouts, never one merged 4-frame bout."""
    from castle.service.ethogram_service import analyze_ethogram

    result = analyze_ethogram(golden_project)
    assert result["bout_stats"]["1"]["n_bouts"] == 2


@pr1_pending
def test_invariant_no_minus_one_bout(golden_project):
    """C-1: -1 is never emitted as a behavioral bout."""
    from castle.service.ethogram_service import analyze_ethogram

    result = analyze_ethogram(golden_project)
    assert "-1" not in result["bout_stats"]


@pr1_pending
def test_invariant_frequency_valid_only_sums_to_one(golden_project):
    """C-3: per-cluster frequency_valid_only sums to 1 when n_valid_frames > 0."""
    from castle.service.ethogram_service import analyze_ethogram

    result = analyze_ethogram(golden_project)
    assert result["n_valid_frames"] > 0
    total = sum(bs["frequency_valid_only"] for bs in result["bout_stats"].values())
    assert abs(total - 1.0) < 1e-6


@pr1_pending
def test_invariant_coverage_and_reason_counts(golden_project):
    """C-1: coverage fields present; reason_counts sum to n_excluded_frames."""
    from castle.service.ethogram_service import analyze_ethogram

    result = analyze_ethogram(golden_project)
    assert result["n_excluded_frames"] == 1          # the single -1
    assert result["valid_frame_fraction"] == pytest.approx(8 / 9)
    counts = result["excluded_reason_counts"]
    assert sum(counts.values()) == result["n_excluded_frames"]
    assert counts.get("dbscan_noise") == 1


@pr1_pending
def test_invariant_mixed_fps_reported(golden_project):
    """C-2/C-3: per-video fps is reported, mixed_fps flagged."""
    from castle.service.ethogram_service import analyze_ethogram

    result = analyze_ethogram(golden_project)
    assert result["fps_policy"] == "per_video"
    assert result["mixed_fps"] is True
    assert result["video_fps"] == {"video1": 10.0, "video2": 20.0}


@pr1_pending
def test_invariant_schema_version_present(golden_project):
    """C-7: every public ethogram output carries a schema_version."""
    from castle.service.ethogram_service import analyze_ethogram

    result = analyze_ethogram(golden_project)
    assert "schema_version" in result


@pr1_pending
def test_fps_zero_raises(golden_project):
    """C-2: fps <= 0 is not a legal override — it raises CastleDataError."""
    from castle.core.types import CastleDataError
    from castle.service.ethogram_service import analyze_ethogram

    with pytest.raises(CastleDataError):
        analyze_ethogram(golden_project, fps=0.0)


@pr1_pending
def test_backward_compat_missing_reason_column(tmp_path, monkeypatch):
    """C-1: a CSV without exclude_reason reads -1 under an 'unknown' bucket."""
    from castle.service import ethogram_service as es
    from castle.service.ethogram_service import analyze_ethogram

    proj = _write_project(tmp_path, {"video1": [0, 0, -1, 1, 1]})  # no reasons
    monkeypatch.setattr(es, "_video_fps", lambda *a, **k: 10.0)

    result = analyze_ethogram(proj)
    assert result["n_excluded_frames"] == 1
    assert result["excluded_reason_counts"].get("unknown") == 1
