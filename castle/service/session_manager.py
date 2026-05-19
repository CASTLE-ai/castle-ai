"""Session management for Behavior Microscope clustering workflows."""

import json
import logging
import os
import shutil
import glob
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """Metadata for a clustering session."""
    session_id: str           # e.g. "session_001"
    name: str                 # user-friendly name, default = auto-generated
    created_at: str           # ISO-8601
    updated_at: str           # ISO-8601
    model: str                # e.g. "dinov3_vitb16"
    roi_id: int
    bin_size: int
    n_clusters: int           # current cluster count
    total_frames: int
    status: str               # "in_progress" or "completed"
    description: str = ""


class SessionManager:
    """Manages multiple clustering sessions per project.
    
    Directory structure:
        project/cluster/sessions/
            session_001/
                manifest.json
                id.csv
                cluster_*.npz
            session_002/
                ...
            _active.txt          ← contains active session_id
    """
    
    def __init__(self, storage_path: str, project_name: str):
        self.project_path = os.path.join(storage_path, project_name)
        self.cluster_path = os.path.join(self.project_path, 'cluster')
        self.sessions_path = os.path.join(self.cluster_path, 'sessions')
        os.makedirs(self.sessions_path, exist_ok=True)
    
    def list_sessions(self) -> List[SessionInfo]:
        """List all sessions, sorted by updated_at descending.

        Skips manifests that fail to parse (JSON corrupted, missing keys,
        schema mismatch) so a single bad manifest cannot prevent the
        clustering page from loading.  Each skip is logged at WARNING.
        """
        sessions = []
        if not os.path.exists(self.sessions_path):
            return sessions
        for d in os.listdir(self.sessions_path):
            manifest_path = os.path.join(self.sessions_path, d, 'manifest.json')
            if not os.path.isfile(manifest_path):
                continue
            try:
                with open(manifest_path) as f:
                    data = json.load(f)
                sessions.append(SessionInfo(**data))
            except (json.JSONDecodeError, TypeError, KeyError) as exc:
                logger.warning(
                    "Skipping unreadable session manifest %s: %s",
                    manifest_path, exc,
                )
                continue
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions
    
    def get_active_session_id(self) -> Optional[str]:
        """Get the currently active session ID."""
        active_file = os.path.join(self.sessions_path, '_active.txt')
        if os.path.exists(active_file):
            with open(active_file) as f:
                return f.read().strip()
        return None
    
    def set_active_session(self, session_id: str):
        """Set the active session."""
        active_file = os.path.join(self.sessions_path, '_active.txt')
        with open(active_file, 'w') as f:
            f.write(session_id)
    
    def create_session(self, model: str, roi_id: int, bin_size: int, 
                       total_frames: int, name: str = "") -> SessionInfo:
        """Create a new session."""
        # Generate ID — use max existing numeric ID + 1 to avoid collisions
        # after deletions.  Skip non-numeric / malformed session_ids (e.g. a
        # user-renamed directory) instead of crashing the whole UI.
        existing = self.list_sessions()
        existing_ids = []
        for s in existing:
            try:
                existing_ids.append(int(s.session_id.split('_')[1]))
            except (IndexError, ValueError) as exc:
                logger.warning(
                    "Ignoring non-numeric session_id %r while computing next id: %s",
                    s.session_id, exc,
                )
                continue
        next_num = max(existing_ids) + 1 if existing_ids else 1
        session_id = f"session_{next_num:03d}"

        now = datetime.now().isoformat()
        if not name:
            name = f"Session {next_num} ({datetime.now().strftime('%m/%d %H:%M')})"
        
        info = SessionInfo(
            session_id=session_id,
            name=name,
            created_at=now,
            updated_at=now,
            model=model,
            roi_id=roi_id,
            bin_size=bin_size,
            n_clusters=0,
            total_frames=total_frames,
            status="in_progress",
        )
        
        session_dir = os.path.join(self.sessions_path, session_id)
        os.makedirs(session_dir, exist_ok=True)
        self._save_manifest(info)
        self.set_active_session(session_id)
        return info
    
    def save_session_state(self, session_id: str, n_clusters: int):
        """Update session metadata (call after clustering changes)."""
        info = self.get_session(session_id)
        if info is None:
            return
        info.n_clusters = n_clusters
        info.updated_at = datetime.now().isoformat()
        self._save_manifest(info)
    
    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get a specific session's info."""
        manifest_path = os.path.join(self.sessions_path, session_id, 'manifest.json')
        if not os.path.isfile(manifest_path):
            return None
        with open(manifest_path) as f:
            return SessionInfo(**json.load(f))
    
    def get_session_dir(self, session_id: str) -> str:
        """Get the directory path for a session."""
        return os.path.join(self.sessions_path, session_id)
    
    # Files that belong to a session and must be synced on snapshot / activate.
    _SESSION_PATTERNS = ['cluster_*.npz', 'time_series_*.csv', 'node_*_meta.json']
    _SESSION_FILES    = ['id.csv']

    def _clear_cluster_root(self) -> None:
        """Remove all session-owned files from the cluster/ root directory."""
        for fname in self._SESSION_FILES:
            fpath = os.path.join(self.cluster_path, fname)
            try:
                os.unlink(fpath)
            except OSError:
                pass
        for pattern in self._SESSION_PATTERNS:
            for fpath in glob.glob(os.path.join(self.cluster_path, pattern)):
                try:
                    os.unlink(fpath)
                except OSError:
                    pass

    def activate_session(self, session_id: str) -> Optional[SessionInfo]:
        """Switch to a session: atomically clear cluster/ root then copy session files."""
        info = self.get_session(session_id)
        if info is None:
            return None

        session_dir = self.get_session_dir(session_id)

        # Step 1: wipe ALL session-owned files from cluster/ root so no stale
        # data from a previous session can leak into this one.
        self._clear_cluster_root()

        # Step 2: copy this session's saved files back into cluster/ root.
        for fname in self._SESSION_FILES:
            src = os.path.join(session_dir, fname)
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(self.cluster_path, fname))
        for pattern in self._SESSION_PATTERNS:
            for src in glob.glob(os.path.join(session_dir, pattern)):
                shutil.copyfile(src, os.path.join(self.cluster_path, os.path.basename(src)))

        self.set_active_session(session_id)
        return info

    def snapshot_to_session(self, session_id: str):
        """Copy the full current cluster/ state into the session directory."""
        session_dir = self.get_session_dir(session_id)
        os.makedirs(session_dir, exist_ok=True)

        for fname in self._SESSION_FILES:
            src = os.path.join(self.cluster_path, fname)
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(session_dir, fname))
        for pattern in self._SESSION_PATTERNS:
            for src in glob.glob(os.path.join(self.cluster_path, pattern)):
                shutil.copyfile(src, os.path.join(session_dir, os.path.basename(src)))
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and deactivate it if it was the active one."""
        session_dir = self.get_session_dir(session_id)
        if not os.path.exists(session_dir):
            return False
        shutil.rmtree(session_dir)
        # If this was the active session, switch to the next most-recent one (or clear).
        if self.get_active_session_id() == session_id:
            remaining = self.list_sessions()
            if remaining:
                self.set_active_session(remaining[0].session_id)
            else:
                active_file = os.path.join(self.sessions_path, '_active.txt')
                if os.path.exists(active_file):
                    os.unlink(active_file)
        return True
    
    def migrate_legacy(self, model: str = "dinov3_vitb16", roi_id: int = 1, 
                       bin_size: int = 1) -> Optional[SessionInfo]:
        """Migrate existing cluster/ data (pre-session-manager) into a session.
        
        Only migrates if:
        - cluster/id.csv exists (there IS legacy data)
        - No sessions exist yet (haven't migrated before)
        """
        id_csv = os.path.join(self.cluster_path, 'id.csv')
        if not os.path.exists(id_csv):
            return None
        if self.list_sessions():
            return None  # Already have sessions
        
        import pandas as pd
        id_df = pd.read_csv(id_csv)
        n_clusters = len(id_df)
        
        info = self.create_session(
            model=model, roi_id=roi_id, bin_size=bin_size,
            total_frames=0, name="Migrated Session"
        )
        info.n_clusters = n_clusters
        info.status = "in_progress"
        self._save_manifest(info)
        self.snapshot_to_session(info.session_id)
        return info
    
    def _save_manifest(self, info: SessionInfo):
        """Save session manifest."""
        session_dir = os.path.join(self.sessions_path, info.session_id)
        os.makedirs(session_dir, exist_ok=True)
        manifest_path = os.path.join(session_dir, 'manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(asdict(info), f, indent=2)
