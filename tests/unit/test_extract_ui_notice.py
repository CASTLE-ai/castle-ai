"""Tests for the Extract-tab cross-environment notice (PR Stage F).

The banner is shown only on a network filesystem and summarises the auto-applied
safe defaults. It must never raise during UI build.
"""

import pytest

from castle.ui.extract_ui import _env_notice_md


def test_env_notice_empty_on_local_fs(monkeypatch):
    monkeypatch.setenv("CASTLE_FORCE_NETWORK_FS", "0")
    assert _env_notice_md(".") == ""


def test_env_notice_present_on_network_fs(monkeypatch):
    monkeypatch.setenv("CASTLE_FORCE_NETWORK_FS", "1")
    note = _env_notice_md("/home/u/sharedfs/proj")
    assert "Cloud / network filesystem detected" in note
    assert "HDF5 file locking" in note
    assert "DataLoader workers" in note


def test_env_notice_never_raises(monkeypatch):
    # Even if detection blows up, UI build must not break.
    monkeypatch.setattr("castle.core.runtime_env.summary",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert _env_notice_md("/whatever") == ""
