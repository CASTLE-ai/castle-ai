"""Robustness tests for clustering_service.restore_local_latent_from_npz.

Restoring a clustering session must never crash the UI when it meets a partially
written or foreign ``.npz`` (crashed runs leave torn files; cluster_model.npz
shares the directory but uses a different schema). The contract is "return
(None, None) rather than raise". These CPU tests pin that contract.
"""

import numpy as np

from castle.service.clustering_service import restore_local_latent_from_npz


def test_missing_file_returns_none(tmp_path):
    out = restore_local_latent_from_npz(str(tmp_path / "does_not_exist.npz"), latents=None)
    assert out == (None, None)


def test_foreign_schema_npz_is_skipped(tmp_path):
    # e.g. a cluster_model.npz-style artefact lacking emb/cls/config keys.
    p = tmp_path / "cluster_model.npz"
    np.savez(p, weights=np.zeros(3), meta=np.array(["x"]))
    out = restore_local_latent_from_npz(str(p), latents=None)
    assert out == (None, None)


def test_corrupt_npz_returns_none(tmp_path):
    # A truncated / non-npz file must be tolerated, not raised.
    p = tmp_path / "cluster_init_.npz"
    p.write_bytes(b"not a real npz")
    out = restore_local_latent_from_npz(str(p), latents=None)
    assert out == (None, None)
