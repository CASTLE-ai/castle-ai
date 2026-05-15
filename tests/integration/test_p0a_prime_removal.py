"""Integration tests verifying that Auto-Cluster has been removed (P0-A').

The Auto-Cluster module (`castle/core/auto_cluster.py`) and its entry
points across the CLI, MCP server, Gradio UI, service layer, and tests
were removed on 2026-05-16 to align CASTLE with its human-in-the-loop
design philosophy. These tests guard against accidental reintroduction
without launching Gradio / PyQt.

Auto-Label (a separate cluster naming utility) is intentionally preserved
and is NOT covered by these tests.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CASTLE_DIR = REPO_ROOT / "castle"


def _grep_castle(pattern: str) -> str:
    """Run `grep -rn PATTERN castle/ --include=*.py` and return stdout."""
    result = subprocess.run(
        ["grep", "-rn", pattern, str(CASTLE_DIR), "--include=*.py"],
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_no_auto_cluster_references_in_castle_package() -> None:
    """Ensure no Python file under castle/ references auto_cluster symbols."""
    pattern = (
        r"auto_cluster\|MICROSCOPE_PRESETS\|run_auto_cluster\|TreeNode\|"
        r"cluster_auto\|DEFAULT_EPS_VALUES\|select_umap_config\|find_best_eps"
    )
    output = _grep_castle(pattern)
    # Allowlist the device_factory.py comment that mentions
    # "clustering_service.py and other modules" — it does not name auto_cluster.
    offending_lines = [
        line for line in output.splitlines()
        if line.strip() and "device_factory.py" not in line
    ]
    assert not offending_lines, (
        "Found auto_cluster references in castle/ package — should have been "
        f"removed by P0-A':\n" + "\n".join(offending_lines)
    )


def test_auto_cluster_file_does_not_exist() -> None:
    """The auto-cluster core module file should not exist."""
    assert not (CASTLE_DIR / "core" / "auto_cluster.py").exists(), (
        "castle/core/auto_cluster.py was supposed to be removed by P0-A'"
    )


def test_auto_cluster_unit_test_does_not_exist() -> None:
    """The auto-cluster unit test should not exist."""
    assert not (REPO_ROOT / "tests" / "unit" / "test_auto_cluster.py").exists(), (
        "tests/unit/test_auto_cluster.py was supposed to be removed by P0-A'"
    )


def test_clustering_service_does_not_expose_auto_cluster() -> None:
    """ClusteringSession must no longer have an auto_cluster() method."""
    from castle.service.clustering_service import ClusteringSession

    assert not hasattr(ClusteringSession, "auto_cluster"), (
        "ClusteringSession still exposes auto_cluster() method"
    )


def test_cluster_handlers_does_not_export_run_auto_cluster() -> None:
    """The Gradio handler module must not expose run_auto_cluster anymore."""
    pytest.importorskip("gradio")
    from castle.ui import cluster_handlers

    assert not hasattr(cluster_handlers, "run_auto_cluster"), (
        "castle.ui.cluster_handlers still exposes run_auto_cluster"
    )


def test_auto_label_is_preserved() -> None:
    """Auto-Label (cluster-naming helper) MUST remain — it is not Auto-Cluster."""
    # auto_label_all is a method on ClusteringSession that auto-generates
    # cluster names from the cluster id; it does NOT bypass the HITL loop.
    from castle.service.clustering_service import ClusteringSession

    assert hasattr(ClusteringSession, "auto_label_all"), (
        "auto_label_all is the cluster-naming helper (NOT Auto-Cluster); "
        "it must remain after P0-A'"
    )
