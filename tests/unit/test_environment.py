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
