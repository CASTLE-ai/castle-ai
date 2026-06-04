"""Tests for the CPU worker-count policy (castle/core/cpu_pool.py).

Policy: cpu_count − reserved (default 4), env-overridable, clamped to [1, cpu],
NO hardcoded cap — it must scale with whatever host CASTLE is deployed on.
"""

import os

import castle.core.cpu_pool as cp


def test_reserved_cores_default_and_override(monkeypatch):
    monkeypatch.delenv("CASTLE_RESERVED_CORES", raising=False)
    assert cp.reserved_cores() == 4
    monkeypatch.setenv("CASTLE_RESERVED_CORES", "6")
    assert cp.reserved_cores() == 6
    monkeypatch.setenv("CASTLE_RESERVED_CORES", "garbage")
    assert cp.reserved_cores() == 4  # bad value → default


def test_default_workers_leaves_reserved_free(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 20)
    monkeypatch.delenv("CASTLE_RESERVED_CORES", raising=False)
    assert cp.default_workers() == 16          # 20 - 4
    monkeypatch.setenv("CASTLE_RESERVED_CORES", "8")
    assert cp.default_workers() == 12          # 20 - 8
    # Never below 1 even on tiny / over-reserved hosts.
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    monkeypatch.setenv("CASTLE_RESERVED_CORES", "8")
    assert cp.default_workers() == 1


def test_resolve_workers_override_and_clamp(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 20)
    monkeypatch.delenv("CASTLE_RESERVED_CORES", raising=False)
    monkeypatch.delenv("MY_WORKERS", raising=False)
    assert cp.resolve_workers("MY_WORKERS") == 16     # falls back to default
    monkeypatch.setenv("MY_WORKERS", "1")
    assert cp.resolve_workers("MY_WORKERS") == 1      # explicit serial
    monkeypatch.setenv("MY_WORKERS", "100")
    assert cp.resolve_workers("MY_WORKERS") == 20     # clamped to cpu_count (no magic cap)
    monkeypatch.setenv("MY_WORKERS", "bad")
    assert cp.resolve_workers("MY_WORKERS") == 16     # bad → default


def test_no_hardcoded_cap_scales_up(monkeypatch):
    # A 64-core box with the default reserve should give 60 workers, not 8.
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    monkeypatch.delenv("CASTLE_RESERVED_CORES", raising=False)
    monkeypatch.delenv("MY_WORKERS", raising=False)
    assert cp.resolve_workers("MY_WORKERS") == 60
