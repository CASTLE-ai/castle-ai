"""Backbone version pinning + checkpoint hash verification (reproducibility).

A re-run must pull the SAME backbone code (pinned torch.hub commit) and the SAME
weights (verified SHA-256), or results are not reproducible. See project
decision 2026-06-20 (backbone pinning).
"""

import hashlib

import pytest

from castle.core import config


def test_torch_hub_refs_are_full_commit_shas():
    refs = config.TORCH_HUB_REFS
    assert {'dinov2', 'dinov3'} <= set(refs)
    for name, sha in refs.items():
        assert len(sha) == 40 and all(c in '0123456789abcdef' for c in sha.lower()), \
            f"{name} ref is not a full 40-char commit SHA: {sha!r}"


def test_dinov3_checkpoint_hashes_present_and_match_filename():
    sha = config.DINOV3_CONSTANTS['MODEL_TO_SHA256']
    fn = config.DINOV3_CONSTANTS['MODEL_TO_CKPT_FILENAME']
    assert {'dinov3_vitb16', 'dinov3_vitl16'} <= set(sha)
    for model, digest in sha.items():
        assert len(digest) == 64
        # the checkpoint filename embeds the first 8 hex of the digest (…-73cec8be)
        assert digest[:8] in fn[model], f"{model}: {digest[:8]} not in {fn[model]}"


def test_hub_repo_pins_commit_with_env_override(monkeypatch):
    from castle.core.models import _hub_repo
    monkeypatch.delenv('CASTLE_DINOV2_REF', raising=False)
    assert _hub_repo('dinov2') == f"facebookresearch/dinov2:{config.TORCH_HUB_REFS['dinov2']}"
    monkeypatch.setenv('CASTLE_DINOV2_REF', 'deadbeef')
    assert _hub_repo('dinov2') == 'facebookresearch/dinov2:deadbeef'
    monkeypatch.setenv('CASTLE_DINOV2_REF', '')  # explicit empty -> track main
    assert _hub_repo('dinov2') == 'facebookresearch/dinov2'


def test_verify_ckpt_sha256(tmp_path, monkeypatch):
    from castle.core.models import _verify_ckpt_sha256
    monkeypatch.delenv('CASTLE_ALLOW_UNVERIFIED_CKPT', raising=False)
    f = tmp_path / "w.pth"
    f.write_bytes(b"weights")
    good = hashlib.sha256(b"weights").hexdigest()

    _verify_ckpt_sha256(str(f), good)  # matches -> no raise

    with pytest.raises(RuntimeError, match="hash mismatch"):
        _verify_ckpt_sha256(str(f), "0" * 64)

    monkeypatch.setenv('CASTLE_ALLOW_UNVERIFIED_CKPT', '1')
    _verify_ckpt_sha256(str(f), "0" * 64)  # opt-out -> no raise
