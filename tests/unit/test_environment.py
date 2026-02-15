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


def test_get_num_workers_with_mocked_cpu_count():
    """Verify formulas with a known cpu_count."""
    with patch('castle.core.environment.os.cpu_count', return_value=16):
        from castle.core.environment import get_num_workers
        assert get_num_workers('extraction') == 8   # 16 // 2
        assert get_num_workers('tracking') == 4     # min(4, 16 // 4)
        assert get_num_workers('default') == 4      # 16 // 4
