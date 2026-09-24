"""Unit tests for castle.service.project_service."""

import os
import tempfile
from castle.service.project_service import create_project, list_projects, get_project_info


def test_create_project():
    with tempfile.TemporaryDirectory() as tmp:
        result = create_project(tmp, "test-project")
        assert result['name'] == 'test-project'
        assert result['created'] is True
        assert os.path.exists(os.path.join(tmp, 'test-project'))
        assert os.path.exists(os.path.join(tmp, 'test-project', 'config.json'))
        assert os.path.exists(os.path.join(tmp, 'test-project', 'sources'))


def test_create_project_duplicate_raises():
    import pytest
    with tempfile.TemporaryDirectory() as tmp:
        create_project(tmp, "dup")
        with pytest.raises(FileExistsError):
            create_project(tmp, "dup")


def test_list_projects():
    with tempfile.TemporaryDirectory() as tmp:
        create_project(tmp, "proj1")
        create_project(tmp, "proj2")
        projects = list_projects(tmp)
        assert len(projects) >= 2
        assert "proj1" in projects
        assert "proj2" in projects


def test_list_projects_empty():
    with tempfile.TemporaryDirectory() as tmp:
        projects = list_projects(tmp)
        assert projects == []


def test_list_projects_nonexistent():
    projects = list_projects("/tmp/nonexistent_castle_dir_xyz")
    assert projects == []


def test_list_projects_ignores_non_projects():
    """Directories without config.json should not be listed."""
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "not-a-project"))
        create_project(tmp, "real-project")
        projects = list_projects(tmp)
        assert projects == ["real-project"]


def test_get_project_info():
    with tempfile.TemporaryDirectory() as tmp:
        create_project(tmp, "test")
        info = get_project_info(tmp, "test")
        assert info is not None
        assert info['name'] == 'test'
        assert info['video_count'] == 0
        assert info['latent_count'] == 0
        assert 'error' not in info


def test_get_project_info_not_found():
    with tempfile.TemporaryDirectory() as tmp:
        info = get_project_info(tmp, "nonexistent")
        assert 'error' in info


def test_create_project_rejects_windows_illegal_names():
    import pytest
    with tempfile.TemporaryDirectory() as tmp:
        for bad in ["a:b", "x/y", 'q"', "trail.", "trail ", "CON", "  "]:
            with pytest.raises(ValueError):
                create_project(tmp, bad)
        assert os.listdir(tmp) == []
        create_project(tmp, "小鼠 實驗_(1)")  # non-ASCII, spaces, brackets are fine


def test_invalid_name_reason():
    from castle.core.project import invalid_name_reason
    assert invalid_name_reason("walking") is None
    assert invalid_name_reason("理毛") is None
    assert "contain" in invalid_name_reason("rear?ing")
    assert "reserved" in invalid_name_reason("nul")


def test_clustering_service_label_rejects_illegal_name():
    import pytest
    from castle.service.clustering_service import ClusteringSession
    svc = ClusteringSession.__new__(ClusteringSession)
    svc.local_latents = object()
    with pytest.raises(ValueError, match="contain"):
        svc.label_cluster(0, "a|b")
