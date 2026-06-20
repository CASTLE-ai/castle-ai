"""Unit tests for SessionManager."""

import json
import time
import pandas as pd
import numpy as np

from castle.service.session_manager import SessionManager


def test_create_session(tmp_path):
    """Test creating a new session."""
    sm = SessionManager(str(tmp_path), "test_project")
    
    info = sm.create_session(
        model="dinov3_vitb16",
        roi_id=1,
        bin_size=5,
        total_frames=1000,
        name="Test Session"
    )
    
    assert info.session_id == "session_001"
    assert info.name == "Test Session"
    assert info.model == "dinov3_vitb16"
    assert info.roi_id == 1
    assert info.bin_size == 5
    assert info.n_clusters == 0
    assert info.total_frames == 1000
    assert info.status == "in_progress"
    
    # Check manifest exists
    manifest_path = tmp_path / "test_project/cluster/sessions/session_001/manifest.json"
    assert manifest_path.exists()
    
    with open(manifest_path) as f:
        data = json.load(f)
    assert data["session_id"] == "session_001"
    assert data["model"] == "dinov3_vitb16"
    
    # Check active session is set
    assert sm.get_active_session_id() == "session_001"


def test_list_sessions(tmp_path):
    """Test listing sessions with multiple sessions sorted by updated_at."""
    sm = SessionManager(str(tmp_path), "test_project")
    
    # Create first session
    sm.create_session(
        model="dinov3_vitb16", roi_id=1, bin_size=1,
        total_frames=100, name="Session 1"
    )
    time.sleep(0.01)  # Ensure different timestamps
    
    # Create second session
    sm.create_session(
        model="dinov3_vitb16", roi_id=1, bin_size=1,
        total_frames=200, name="Session 2"
    )
    time.sleep(0.01)
    
    # Update first session (should move it to top)
    sm.save_session_state("session_001", n_clusters=5)
    
    sessions = sm.list_sessions()
    
    assert len(sessions) == 2
    # Most recently updated should be first
    assert sessions[0].session_id == "session_001"
    assert sessions[0].n_clusters == 5
    assert sessions[1].session_id == "session_002"


def test_list_sessions_empty(tmp_path):
    """Test listing sessions when no sessions exist."""
    sm = SessionManager(str(tmp_path), "test_project")
    sessions = sm.list_sessions()
    assert sessions == []


def test_get_session(tmp_path):
    """Test retrieving a specific session by ID."""
    sm = SessionManager(str(tmp_path), "test_project")
    
    created = sm.create_session(
        model="dinov3_vitb16", roi_id=2, bin_size=3,
        total_frames=500, name="My Session"
    )
    
    retrieved = sm.get_session("session_001")
    
    assert retrieved is not None
    assert retrieved.session_id == created.session_id
    assert retrieved.name == "My Session"
    assert retrieved.roi_id == 2
    assert retrieved.bin_size == 3
    
    # Test non-existent session
    assert sm.get_session("session_999") is None


def test_activate_session(tmp_path):
    """Test activating a session copies files to cluster/ root."""
    sm = SessionManager(str(tmp_path), "test_project")
    
    # Create session
    sm.create_session(
        model="dinov3_vitb16", roi_id=1, bin_size=1,
        total_frames=100, name="Test"
    )
    
    # Create mock files in session directory
    session_dir = tmp_path / "test_project/cluster/sessions/session_001"
    
    # Create id.csv
    id_csv = session_dir / "id.csv"
    df = pd.DataFrame({
        'cluster_id': [0, 1, 2],
        'count': [10, 20, 30]
    })
    df.to_csv(id_csv, index=False)
    
    # Create cluster npz files
    cluster_npz = session_dir / "cluster_001.npz"
    np.savez(cluster_npz, data=np.array([1, 2, 3]))
    
    # Activate session
    result = sm.activate_session("session_001")
    
    assert result is not None
    assert result.session_id == "session_001"
    
    # Check files were copied to cluster/ root
    cluster_root = tmp_path / "test_project/cluster"
    assert (cluster_root / "id.csv").exists()
    assert (cluster_root / "cluster_001.npz").exists()
    
    # Verify content
    copied_df = pd.read_csv(cluster_root / "id.csv")
    assert len(copied_df) == 3
    
    # Check active session is set
    assert sm.get_active_session_id() == "session_001"


def test_snapshot_to_session(tmp_path):
    """Test saving current cluster/ state to session."""
    sm = SessionManager(str(tmp_path), "test_project")
    
    # Create session
    sm.create_session(
        model="dinov3_vitb16", roi_id=1, bin_size=1,
        total_frames=100, name="Test"
    )
    
    # Create mock files in cluster/ root
    cluster_root = tmp_path / "test_project/cluster"
    
    # Create id.csv
    id_csv = cluster_root / "id.csv"
    df = pd.DataFrame({
        'cluster_id': [0, 1],
        'count': [5, 10]
    })
    df.to_csv(id_csv, index=False)
    
    # Create cluster npz
    cluster_npz = cluster_root / "cluster_002.npz"
    np.savez(cluster_npz, data=np.array([4, 5, 6]))
    
    # Snapshot to session
    sm.snapshot_to_session("session_001")
    
    # Check files were copied to session directory
    session_dir = tmp_path / "test_project/cluster/sessions/session_001"
    assert (session_dir / "id.csv").exists()
    assert (session_dir / "cluster_002.npz").exists()
    
    # Verify content
    copied_df = pd.read_csv(session_dir / "id.csv")
    assert len(copied_df) == 2


def test_delete_session(tmp_path):
    """Test deleting a session removes its directory."""
    sm = SessionManager(str(tmp_path), "test_project")
    
    # Create session
    sm.create_session(
        model="dinov3_vitb16", roi_id=1, bin_size=1,
        total_frames=100, name="To Delete"
    )
    
    session_dir = tmp_path / "test_project/cluster/sessions/session_001"
    assert session_dir.exists()
    
    # Delete session
    result = sm.delete_session("session_001")
    
    assert result is True
    assert not session_dir.exists()
    assert sm.get_session("session_001") is None
    
    # Test deleting non-existent session
    assert sm.delete_session("session_999") is False


def test_migrate_legacy(tmp_path):
    """Test migrating existing cluster/ data into a session."""
    sm = SessionManager(str(tmp_path), "test_project")
    
    # Create legacy id.csv in cluster/ root (dir already exists from SessionManager init)
    cluster_root = tmp_path / "test_project/cluster"
    
    id_csv = cluster_root / "id.csv"
    df = pd.DataFrame({
        'cluster_id': [0, 1, 2, 3],
        'count': [10, 20, 30, 40]
    })
    df.to_csv(id_csv, index=False)
    
    # Create legacy npz file
    cluster_npz = cluster_root / "cluster_001.npz"
    np.savez(cluster_npz, data=np.array([1, 2, 3]))
    
    # Migrate
    info = sm.migrate_legacy(model="dinov3_vitb16", roi_id=1, bin_size=5)
    
    assert info is not None
    assert info.session_id == "session_001"
    assert info.name == "Migrated Session"
    assert info.n_clusters == 4  # one cluster per id.csv row (no -1 noise row in this fixture)
    assert info.model == "dinov3_vitb16"
    assert info.roi_id == 1
    assert info.bin_size == 5
    
    # Check session directory exists and has files
    session_dir = tmp_path / "test_project/cluster/sessions/session_001"
    assert session_dir.exists()
    assert (session_dir / "id.csv").exists()
    assert (session_dir / "cluster_001.npz").exists()
    assert (session_dir / "manifest.json").exists()


def test_migrate_legacy_no_data(tmp_path):
    """Test migration returns None when no id.csv exists."""
    sm = SessionManager(str(tmp_path), "test_project")
    
    # No id.csv exists
    info = sm.migrate_legacy()
    
    assert info is None
    assert len(sm.list_sessions()) == 0


def test_migrate_legacy_already_migrated(tmp_path):
    """Test migration returns None when sessions already exist."""
    sm = SessionManager(str(tmp_path), "test_project")
    
    # Create a session first
    sm.create_session(
        model="dinov3_vitb16", roi_id=1, bin_size=1,
        total_frames=100, name="Existing"
    )
    
    # Create legacy id.csv
    cluster_root = tmp_path / "test_project/cluster"
    id_csv = cluster_root / "id.csv"
    df = pd.DataFrame({'cluster_id': [0, 1], 'count': [10, 20]})
    df.to_csv(id_csv, index=False)
    
    # Try to migrate
    info = sm.migrate_legacy()
    
    assert info is None  # Should not migrate if sessions already exist
    assert len(sm.list_sessions()) == 1  # Only the original session


def test_get_active_session_id(tmp_path):
    """Test reading active session ID."""
    sm = SessionManager(str(tmp_path), "test_project")
    
    # No active session initially
    assert sm.get_active_session_id() is None
    
    # Create session (sets active)
    sm.create_session(
        model="dinov3_vitb16", roi_id=1, bin_size=1,
        total_frames=100, name="First"
    )
    
    assert sm.get_active_session_id() == "session_001"
    
    # Create another session
    sm.create_session(
        model="dinov3_vitb16", roi_id=1, bin_size=1,
        total_frames=100, name="Second"
    )
    
    assert sm.get_active_session_id() == "session_002"
    
    # Manually set active
    sm.set_active_session("session_001")
    assert sm.get_active_session_id() == "session_001"
