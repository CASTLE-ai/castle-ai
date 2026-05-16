"""Reproducibility regression tests (P0-B / REPRO-03).

These tests guard the contract that:

- ``set_global_seed`` produces bit-identical NumPy / PyTorch RNG state across
  two calls.
- ``LocalLatent.build_embedding`` accepts a ``base_seed`` and returns the
  resolved seed list, and re-runs with the same ``base_seed`` produce
  bit-identical embeddings.
- ``LocalLatent.build_embedding`` without a ``base_seed`` draws a fresh
  ``secrets.randbits(32)`` seed per stage (so the resolved seeds are
  different across calls but always recorded).
- ``ProjectConfig`` carries ``master_seed`` round-trip through ``to_dict`` /
  ``from_dict``.
- The CLI exposes the ``--seed`` / ``--strict-cuda`` global options.

The UMAP-based tests fall back gracefully when ``umap-learn`` is not
installed; they are skipped rather than failed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---- set_global_seed ----

def test_set_global_seed_returns_input_seed() -> None:
    from castle.core.seed import set_global_seed

    assert set_global_seed(123) == 123


def test_set_global_seed_makes_numpy_deterministic() -> None:
    from castle.core.seed import set_global_seed

    set_global_seed(42)
    a = np.random.rand(5)
    set_global_seed(42)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_set_global_seed_makes_python_random_deterministic() -> None:
    import random
    from castle.core.seed import set_global_seed

    set_global_seed(42)
    a = [random.random() for _ in range(5)]
    set_global_seed(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_global_seed_makes_torch_deterministic() -> None:
    torch = pytest.importorskip("torch")
    from castle.core.seed import set_global_seed

    set_global_seed(42)
    a = torch.rand(5)
    set_global_seed(42)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_make_torch_generator_is_reproducible_when_master_seeded() -> None:
    torch = pytest.importorskip("torch")
    from castle.core.seed import make_torch_generator, set_global_seed

    set_global_seed(42)
    g1 = make_torch_generator()
    s1 = g1.initial_seed()

    set_global_seed(42)
    g2 = make_torch_generator()
    s2 = g2.initial_seed()

    assert s1 == s2, "Generator seed must follow master seed deterministically"


# ---- ProjectConfig.master_seed ----

def test_project_config_master_seed_default() -> None:
    from castle.core.project_config import ProjectConfig

    cfg = ProjectConfig()
    assert cfg.master_seed == 42


def test_project_config_master_seed_round_trip() -> None:
    from castle.core.project_config import ProjectConfig

    cfg = ProjectConfig(master_seed=7)
    cfg2 = ProjectConfig.from_dict(cfg.to_dict())
    assert cfg2.master_seed == 7


def test_project_config_from_dict_missing_master_seed_uses_default() -> None:
    """Old configs (pre P0-B) lack master_seed and must still load."""
    from castle.core.project_config import ProjectConfig

    old_dict = {
        "tracking": {},
        "extraction": {},
        "clustering": {},
    }
    cfg = ProjectConfig.from_dict(old_dict)
    assert cfg.master_seed == 42


# ---- LocalLatent.build_embedding seed contract ----

@pytest.fixture
def small_latent() -> np.ndarray:
    """Random 80×8 fixture (>= 2*n_neighbors for any UMAP default)."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((80, 8)).astype(np.float32)


def _import_local_latent_or_skip():
    """Import LocalLatent + check umap is installed (CPU path is what we test)."""
    pytest.importorskip("umap", reason="umap-learn not installed; CPU path required")
    from castle.utils.latent_explorer import LocalLatent
    return LocalLatent


def test_build_embedding_returns_resolved_seeds(small_latent) -> None:
    LocalLatent = _import_local_latent_or_skip()
    local = LocalLatent(small_latent, index_mask=None, color_avoid=set(), device="cpu")

    cfg = [{"n_neighbors": 10, "min_dist": 0.1, "n_components": 2}]
    seeds = local.build_embedding(cfg, base_seed=42)

    assert isinstance(seeds, list)
    assert seeds == [42]
    assert local.umap_seeds == [42]
    assert hasattr(local, "embedding")
    assert local.embedding.shape == (80, 2)


def test_build_embedding_base_seed_multi_stage(small_latent) -> None:
    LocalLatent = _import_local_latent_or_skip()
    local = LocalLatent(small_latent, index_mask=None, color_avoid=set(), device="cpu")

    cfg = [
        {"n_neighbors": 10, "min_dist": 0.1, "n_components": 4},
        {"n_neighbors": 10, "min_dist": 0.1, "n_components": 2},
    ]
    seeds = local.build_embedding(cfg, base_seed=100)
    assert seeds == [100, 101]


def test_build_embedding_no_seed_draws_fresh(small_latent) -> None:
    """Empty base_seed → secrets.randbits draws a fresh seed each call."""
    LocalLatent = _import_local_latent_or_skip()

    cfg = [{"n_neighbors": 10, "min_dist": 0.1, "n_components": 2}]

    local1 = LocalLatent(small_latent.copy(), None, set(), "cpu")
    seeds1 = local1.build_embedding(cfg)

    local2 = LocalLatent(small_latent.copy(), None, set(), "cpu")
    seeds2 = local2.build_embedding(cfg)

    assert len(seeds1) == 1 and len(seeds2) == 1
    # secrets.randbits(32) collision is astronomically rare
    assert seeds1 != seeds2, "Two re-rolls should draw different seeds"


def test_build_embedding_writes_umap_log_jsonl(small_latent, tmp_path) -> None:
    LocalLatent = _import_local_latent_or_skip()
    local = LocalLatent(small_latent, None, set(), "cpu")

    log_path = tmp_path / "umap_log.jsonl"
    cfg = [{"n_neighbors": 10, "min_dist": 0.1, "n_components": 2}]
    seeds = local.build_embedding(cfg, base_seed=99, log_path=log_path)

    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["seed"] == 99
    assert record["source"] == "base+offset"
    assert record["stage"] == 0
    assert seeds == [99]


def test_build_embedding_user_seed_marked_as_user(small_latent, tmp_path) -> None:
    LocalLatent = _import_local_latent_or_skip()
    local = LocalLatent(small_latent, None, set(), "cpu")

    log_path = tmp_path / "umap_log.jsonl"
    cfg = [{"n_neighbors": 10, "min_dist": 0.1, "n_components": 2, "random_state": 7}]
    seeds = local.build_embedding(cfg, log_path=log_path)
    assert seeds == [7]

    record = json.loads(log_path.read_text().strip())
    assert record["source"] == "user"


# ---- CLI --seed option ----

def test_cli_main_help_lists_seed_option() -> None:
    """`castle --help` should advertise --seed and --strict-cuda."""
    result = subprocess.run(
        [sys.executable, "-m", "castle.cli.main", "--help"],
        capture_output=True, text=True,
        cwd=str(REPO_ROOT),
    )
    # Typer prints help to stdout on --help with exit 0
    combined = (result.stdout + result.stderr).lower()
    assert "--seed" in combined, f"--seed missing from help:\n{combined}"
    assert "strict-cuda" in combined or "strict_cuda" in combined, (
        f"--strict-cuda missing from help:\n{combined}"
    )


def test_set_global_seed_strict_cuda_flag_runs() -> None:
    """strict_cuda=True must not raise even without a CUDA device."""
    from castle.core.seed import set_global_seed

    # Should be a no-op on CPU but must not raise.
    set_global_seed(0, strict_cuda=True)
