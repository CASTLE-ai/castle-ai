"""Test the self-describing time_series CSV sidecar metadata builder."""

import json

from castle.service.clustering_service import build_timeseries_meta


def test_build_timeseries_meta_fields():
    m = build_timeseries_meta(29.97, {0: "walk", 2: "groom"}, 100)
    assert m["schema_version"] == 1
    assert m["fps"] == 29.97
    assert m["n_frames"] == 100
    assert m["cluster_id_to_name"] == {0: "walk", 2: "groom"}
    assert "behavior" in m["columns"] and "exclude_reason" in m["columns"]
    assert m["castle_version"]
    json.dumps(m)  # must be JSON-serialisable


def test_build_timeseries_meta_coerces_keys_and_types():
    # int-like keys / non-str names are coerced so JSON round-trips cleanly.
    import numpy as np
    m = build_timeseries_meta(np.float64(30.0), {np.int64(1): "x"}, np.int64(5))
    assert m["fps"] == 30.0 and m["n_frames"] == 5
    assert m["cluster_id_to_name"] == {1: "x"}
    json.dumps(m)


def test_write_timeseries_csvs_shared_by_both_submit_paths(tmp_path):
    """The shared writer that BOTH ClusteringSession.submit (CLI) and
    submit_local_to_global (UI) now call — run-verifies the exact logic the
    (otherwise integration-gated) UI submit path uses."""
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    from castle.service.clustering_service import _write_timeseries_csvs

    # 2 videos × 3 bins; bin 2 has a non-finite latent + noise label (-1).
    cluster = np.array([0, 1, -1, 0, 1, 1], dtype=np.int16)
    data = np.ones((6, 2), dtype=np.float64)
    data[2] = [np.nan, 1.0]
    latents = SimpleNamespace(
        cluster=cluster, data=data, time_window=2,
        cluster_meta={0: {"name": "walk", "color": "#111"},
                      1: {"name": "rear", "color": "#222"}},
    )
    aggregator = SimpleNamespace(
        videos_meta=[(3, "vidA.mp4"), (3, "vidB.mp4")],
        frame_index_map=None,
        fps_per_video={"vidA.mp4": 30.0},  # vidB falls back to .fps
        fps=24.0,
    )

    paths = _write_timeseries_csvs(latents, aggregator, str(tmp_path))
    assert len(paths) == 2

    # vidA: 3 bins × time_window 2 = 6 frames; behavior + exclude_reason columns
    dfa = pd.read_csv(tmp_path / "time_series_vidA.csv")
    assert list(dfa.columns) == ["behavior", "exclude_reason"]
    assert dfa["behavior"].tolist() == [0, 0, 1, 1, -1, -1]
    # bin 2 (non-finite latent) → reason 2, expanded over its 2 frames
    assert dfa["exclude_reason"].tolist() == [0, 0, 0, 0, 2, 2]

    # self-describing meta: per-video fps + name map
    metaA = json.loads((tmp_path / "time_series_vidA.meta.json").read_text())
    assert metaA["fps"] == 30.0
    assert metaA["cluster_id_to_name"] == {"0": "walk", "1": "rear"}  # JSON keys → str
    metaB = json.loads((tmp_path / "time_series_vidB.meta.json").read_text())
    assert metaB["fps"] == 24.0  # fallback fps


def test_write_timeseries_csvs_exclude_reason_failure_is_safe(tmp_path):
    """If exclude_reason can't be derived, the writer must not crash submit —
    it falls back to 0 and still emits the 2-column CSV."""
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    from castle.service.clustering_service import _write_timeseries_csvs

    latents = SimpleNamespace(
        cluster=np.array([0, 1], dtype=np.int16),
        data=None,  # derive_exclude_reason will raise → fallback to 0
        time_window=1,
        cluster_meta={0: {"name": "a", "color": "#111"}, 1: {"name": "b", "color": "#222"}},
    )
    aggregator = SimpleNamespace(
        videos_meta=[(2, "v.mp4")], frame_index_map=None, fps_per_video={}, fps=30.0,
    )
    paths = _write_timeseries_csvs(latents, aggregator, str(tmp_path))
    df = pd.read_csv(tmp_path / "time_series_v.csv")
    assert list(df.columns) == ["behavior", "exclude_reason"]
    assert df["exclude_reason"].tolist() == [0, 0]  # safe fallback
