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
