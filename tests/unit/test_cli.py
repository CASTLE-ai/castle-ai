"""Unit tests for castle.cli.main."""

import tempfile
from typer.testing import CliRunner
from castle.cli.main import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # Check the app name/help text is present
    assert "castle" in result.stdout.lower() or "CASTLE" in result.stdout


def test_project_help():
    result = runner.invoke(app, ["project", "--help"])
    assert result.exit_code == 0
    assert "init" in result.stdout or "list" in result.stdout


def test_project_list_empty():
    with tempfile.TemporaryDirectory() as tmp:
        result = runner.invoke(app, ["project", "list", "--storage", tmp])
        assert result.exit_code == 0


def test_project_init():
    with tempfile.TemporaryDirectory() as tmp:
        result = runner.invoke(app, ["project", "init", "my-proj", "--storage", tmp])
        assert result.exit_code == 0
        assert "my-proj" in result.stdout


def test_project_init_duplicate():
    with tempfile.TemporaryDirectory() as tmp:
        runner.invoke(app, ["project", "init", "dup", "--storage", tmp])
        result = runner.invoke(app, ["project", "init", "dup", "--storage", tmp])
        assert result.exit_code == 1


def test_project_info():
    with tempfile.TemporaryDirectory() as tmp:
        runner.invoke(app, ["project", "init", "info-test", "--storage", tmp])
        result = runner.invoke(app, ["project", "info", "info-test", "--storage", tmp])
        assert result.exit_code == 0
        assert "info-test" in result.stdout


def test_cluster_help():
    result = runner.invoke(app, ["cluster", "--help"])
    assert result.exit_code == 0
