"""Integration tests verifying that the Quick Start Wizard has been removed.

These tests guard against accidental reintroduction of the wizard module
(`castle/ui/wizard_ui.py`) that was removed as part of P0-A on 2026-05-16. The
removal aligns CASTLE with its human-in-the-loop design philosophy by
eliminating any "one-click" entry point that bypasses the Behavior Microscope
clustering workflow.

Each test uses subprocess grep or module attribute inspection so they run
without launching Gradio and complete in under a second.
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


def test_no_wizard_references_in_castle_package() -> None:
    """Ensure no Python file under castle/ references the wizard modules."""
    pattern = r"WizardPanel\|WizardUI\|wizard_panel\|wizard_ui"
    output = _grep_castle(pattern)
    assert not output, (
        "Found wizard references in castle/ package — should have been "
        f"removed by P0-A:\n{output}"
    )


def test_wizard_ui_file_does_not_exist() -> None:
    """The Gradio wizard module file should not exist."""
    assert not (CASTLE_DIR / "ui" / "wizard_ui.py").exists(), (
        "castle/ui/wizard_ui.py was supposed to be removed by P0-A"
    )


def test_main_ui_module_does_not_import_wizard() -> None:
    """`castle.ui.main_ui` must no longer import WizardUI."""
    pytest.importorskip("gradio")
    import castle.ui.main_ui as main_ui

    assert not hasattr(main_ui, "WizardUI"), (
        "castle.ui.main_ui still exposes WizardUI symbol"
    )
