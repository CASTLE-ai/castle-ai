"""
tests/unit/test_mcp.py
Tests for the CASTLE MCP server.
"""

import asyncio
import json
import os
from unittest.mock import patch, MagicMock

import pytest


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ------------------------------------------------------------------
# Import / registration tests
# ------------------------------------------------------------------

class TestMCPImport:
    """Verify MCP server can be imported and is properly configured."""

    def test_import_server(self):
        from castle.mcp.server import mcp
        assert mcp is not None
        assert mcp.name == "castle"

    def test_import_init(self):
        import castle.mcp
        assert hasattr(castle.mcp, "__doc__")


class TestMCPToolRegistration:
    """Verify all expected tools are registered."""

    EXPECTED_TOOLS = [
        "project_create",
        "project_list",
        "project_info",
        "track_run",
        "track_status",
        "extract_run",
        "cluster_run",
        "cluster_label",
        "device_info",
    ]

    def test_all_tools_registered(self):
        from castle.mcp.server import mcp
        tools = _run(mcp.list_tools())
        tool_names = [t.name for t in tools]
        for name in self.EXPECTED_TOOLS:
            assert name in tool_names, f"Tool '{name}' not registered"

    def test_tool_count(self):
        from castle.mcp.server import mcp
        tools = _run(mcp.list_tools())
        assert len(tools) >= len(self.EXPECTED_TOOLS)


class TestMCPResourceRegistration:
    """Verify all expected resources are registered."""

    EXPECTED_RESOURCES = [
        "castle://projects",
    ]

    EXPECTED_TEMPLATES = [
        "castle://project/{name}/status",
        "castle://project/{name}/clusters",
        "castle://project/{name}/config",
    ]

    def test_static_resources_registered(self):
        from castle.mcp.server import mcp
        resources = _run(mcp.list_resources())
        uris = [str(r.uri) for r in resources]
        for uri in self.EXPECTED_RESOURCES:
            assert uri in uris, f"Resource '{uri}' not registered"

    def test_resource_templates_registered(self):
        from castle.mcp.server import mcp
        templates = _run(mcp.list_resource_templates())
        template_uris = [str(t.uriTemplate) for t in templates]
        for uri in self.EXPECTED_TEMPLATES:
            assert uri in template_uris, f"Resource template '{uri}' not registered"


# ------------------------------------------------------------------
# Tool function tests (with mocked service layer)
# ------------------------------------------------------------------

class TestProjectCreateTool:
    """Test project_create tool."""

    @patch("castle.service.project_service.create_project")
    def test_success(self, mock_create):
        from castle.mcp.server import project_create
        mock_create.return_value = {"path": "/tmp/test", "name": "test", "created": True}
        result = project_create("test", "")
        assert result["status"] == "success"
        assert "test" in result["message"]

    @patch("castle.service.project_service.create_project")
    def test_with_nonexistent_source_dir(self, mock_create):
        from castle.mcp.server import project_create
        mock_create.return_value = {"path": "/tmp/test", "name": "test", "created": True}
        # source_dir doesn't exist, so no video import
        result = project_create("test", "/nonexistent")
        assert result["status"] == "success"
        assert "videos_added" not in result

    @patch("castle.service.project_service.create_project", side_effect=FileExistsError("exists"))
    def test_error(self, mock_create):
        from castle.mcp.server import project_create
        result = project_create("test", "")
        assert result["status"] == "error"
        assert "exists" in result["message"]


class TestProjectListTool:
    """Test project_list tool."""

    @patch("castle.service.project_service.list_projects")
    def test_success(self, mock_list):
        from castle.mcp.server import project_list
        mock_list.return_value = ["proj_a", "proj_b"]
        result = project_list()
        assert result["status"] == "success"
        assert result["projects"] == ["proj_a", "proj_b"]
        assert "2" in result["message"]


class TestProjectInfoTool:
    """Test project_info tool."""

    @patch("castle.service.project_service.get_project_info")
    def test_success(self, mock_info):
        from castle.mcp.server import project_info
        mock_info.return_value = {
            "name": "proj",
            "path": "/tmp/proj",
            "videos": ["v1.mp4"],
            "video_count": 1,
            "latent_count": 0,
            "config": {"source": ["v1.mp4"]},
        }
        result = project_info("proj")
        assert result["status"] == "success"
        assert result["video_count"] == 1

    @patch("castle.service.project_service.get_project_info")
    def test_not_found(self, mock_info):
        from castle.mcp.server import project_info
        mock_info.return_value = {
            "name": "nope",
            "path": "/tmp/nope",
            "videos": [],
            "video_count": 0,
            "latent_count": 0,
            "config": {},
            "error": "Project not found",
        }
        result = project_info("nope")
        assert result["status"] == "error"


class TestDeviceInfoTool:
    """Test device_info tool."""

    @patch("castle.core.environment.Environment")
    def test_returns_device(self, mock_env_cls):
        from castle.mcp.server import device_info
        mock_env = MagicMock()
        mock_env.device = "cpu"
        mock_env_cls.return_value = mock_env
        result = device_info()
        assert result["status"] == "success"
        assert "device" in result


class TestClusterLabelTool:
    """Test cluster_label tool."""

    @patch("castle.service.annotation_service.save_annotations")
    @patch("castle.service.annotation_service.load_annotations")
    def test_success(self, mock_load, mock_save):
        from castle.mcp.server import cluster_label
        mock_load.return_value = {}
        mock_save.return_value = "/tmp/annotations.csv"
        result = cluster_label("proj", "init_0", "grooming")
        assert result["status"] == "success"
        assert "grooming" in result["message"]
        mock_save.assert_called_once()


class TestTrackStatusTool:
    """Test track_status tool."""

    @patch("castle.service.tracking_service.get_tracking_status")
    def test_success(self, mock_status):
        from castle.mcp.server import track_status
        mock_status.return_value = {
            "tracked": True,
            "mask_path": "/tmp/mask.h5",
            "n_rois": 2,
            "n_frames": 100,
            "csv_path": "",
            "mix_video_path": "",
        }
        result = track_status("proj", "video.mp4")
        assert result["status"] == "success"
        assert result["tracked"] is True
        assert result["n_rois"] == 2


# ------------------------------------------------------------------
# CLI entry-point test
# ------------------------------------------------------------------

class TestMCPCLI:
    """Test MCP CLI command registration."""

    def test_cli_import(self):
        from castle.cli.mcp_cmd import app
        assert app is not None

    def test_registered_in_main(self):
        """Ensure 'mcp' is registered in main CLI app."""
        from castle.cli.main import app
        # Typer stores registered groups; check by name
        group_names = []
        for group in getattr(app, "registered_groups", []):
            typer_instance = group.typer_instance
            if hasattr(typer_instance, "info") and typer_instance.info:
                group_names.append(group.name)
            else:
                group_names.append(group.name)
        assert "mcp" in group_names
