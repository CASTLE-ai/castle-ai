"""Unit tests for castle.core.environment.

Note: environment.py imports torch at module level for device detection.
We mock torch to avoid requiring GPU for unit tests.
"""

import sys
from unittest.mock import MagicMock, patch


def test_get_device_returns_string():
    """get_device() should return one of 'cuda', 'cpu', 'mps'."""
    # Import after torch is available (it's already imported at module level
    # in environment.py, so we just test the cached value)
    from castle.core.environment import get_device
    d = get_device()
    assert isinstance(d, str)
    assert d in ('cuda', 'cpu', 'mps')


def test_environment_singleton():
    """env should be a module-level singleton."""
    from castle.core.environment import env
    assert hasattr(env, 'device')
    assert hasattr(env, 'os_sys')


# ---- get_num_workers (C-03) ----

def test_get_num_workers_extraction():
    from castle.core.environment import get_num_workers
    w = get_num_workers('extraction')
    assert isinstance(w, int)
    assert w >= 1


def test_get_num_workers_tracking():
    from castle.core.environment import get_num_workers
    w = get_num_workers('tracking')
    assert isinstance(w, int)
    assert w >= 1
    assert w <= 4  # tracking caps at 4


def test_get_num_workers_default():
    from castle.core.environment import get_num_workers
    w = get_num_workers('default')
    assert isinstance(w, int)
    assert w >= 1


def test_get_num_workers_unknown_falls_back_to_default():
    from castle.core.environment import get_num_workers
    w = get_num_workers('unknown_task')
    assert isinstance(w, int)
    assert w >= 1


def test_get_num_workers_with_mocked_cpu_count(monkeypatch):
    """Verify formulas with a known usable CPU count (cgroup/affinity aware)."""
    monkeypatch.setenv('CASTLE_USABLE_CPUS', '16')
    from castle.core.environment import get_num_workers
    assert get_num_workers('extraction') == 8   # 16 // 2, under cap 16
    assert get_num_workers('tracking') == 4     # min(4, 16 // 4)
    assert get_num_workers('default') == 4      # 16 // 4


def test_get_num_workers_caps_on_many_core_box(monkeypatch):
    """A 64-core cloud host must not spawn 32 extraction workers — capped at 16."""
    monkeypatch.setenv('CASTLE_USABLE_CPUS', '64')
    monkeypatch.delenv('CASTLE_MAX_EXTRACTION_WORKERS', raising=False)
    from castle.core.environment import get_num_workers
    assert get_num_workers('extraction') == 16   # min(64//2=32, cap 16)


def test_get_num_workers_network_fs_reduces(monkeypatch):
    """On a network FS the extraction workers drop to the lower (8) cap."""
    monkeypatch.setenv('CASTLE_USABLE_CPUS', '64')
    monkeypatch.setenv('CASTLE_FORCE_NETWORK_FS', '1')
    from castle.core.environment import get_num_workers
    assert get_num_workers('extraction', fs_path='/sharedfs/proj') == 8


def test_get_num_workers_explicit_override_wins(monkeypatch):
    """CASTLE_EXTRACTION_WORKERS is authoritative — bypasses the caps."""
    monkeypatch.setenv('CASTLE_USABLE_CPUS', '64')
    monkeypatch.setenv('CASTLE_EXTRACTION_WORKERS', '24')
    from castle.core.environment import get_num_workers
    assert get_num_workers('extraction', fs_path='/sharedfs/proj') == 24


def test_get_num_workers_dev_box_unchanged(monkeypatch):
    """20-core dev box, local FS: extraction stays 10 (== 20 // 2), as before."""
    monkeypatch.setenv('CASTLE_USABLE_CPUS', '20')
    monkeypatch.delenv('CASTLE_FORCE_NETWORK_FS', raising=False)
    from castle.core.environment import get_num_workers
    assert get_num_workers('extraction') == 10


# ---- collect_run_environment (Phase-2 provenance) ----

def test_collect_run_environment_shape():
    """Provenance snapshot carries version/platform/device + a package map."""
    import json
    from castle.core.environment import collect_run_environment

    info = collect_run_environment()
    assert isinstance(info, dict)
    for key in ("castle_version", "python", "platform", "device", "packages"):
        assert key in info, f"missing provenance key {key!r}"
    assert info["device"] in ("cuda", "cpu", "mps")
    assert isinstance(info["packages"], dict)
    # numpy/torch always present in a CASTLE env; values are version strings or None.
    assert "numpy" in info["packages"] and "torch" in info["packages"]
    # Must be JSON-serialisable (it is embedded into output artifacts).
    json.dumps(info)


def test_collect_run_environment_is_cached():
    """Cached per process so embedding it in every artifact write is cheap."""
    from castle.core.environment import collect_run_environment
    assert collect_run_environment() is collect_run_environment()


def test_latent_sidecar_embeds_environment(tmp_path):
    """save_latent_with_metadata stamps the run environment into the sidecar."""
    import numpy as np
    from castle.utils.latent_metadata import (
        save_latent_with_metadata, load_latent_metadata,
    )

    arr = np.zeros((5, 8), dtype=np.float32)
    npz = tmp_path / "vid_ROI_1.npz"
    save_latent_with_metadata(
        str(npz), arr, video_name="vid.mp4", roi_id=1, model_name="dinov3_vitb16",
    )
    meta = load_latent_metadata(str(npz))
    assert meta is not None
    assert meta["schema_version"] == 2
    assert "environment" in meta
    assert meta["environment"]["castle_version"]
    assert "packages" in meta["environment"]
