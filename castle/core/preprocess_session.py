"""
castle/core/preprocess_session.py
Session management for Pre-process tab.

A session represents one parameter set applied to one or more videos.
Directory is named with 8-char SHA256 hash of the full human-readable session name.
All writes are atomic (tmp + rename) to prevent corruption.

Concurrent writers to ``session_meta.json`` — e.g. two extraction jobs racing
to add their video to the same session — would otherwise lose updates (both
read videos=[], both write videos=[A] / videos=[B], one survives).  A
cross-platform ``filelock.FileLock`` on a sidecar guards the read-modify-write
sequence in ``add_video_to_session``.  ``filelock`` works on Windows, macOS
and Linux; do NOT switch to Linux-only ``fcntl``.
"""

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from filelock import FileLock

logger = logging.getLogger(__name__)

# How long to wait for the session_meta lock before giving up.  Five seconds
# is generous for an in-memory read-modify-write that takes ~milliseconds.
_SESSION_LOCK_TIMEOUT = 5.0


def _sessions_root(storage_path: str, project_name: str) -> Path:
    return Path(storage_path) / project_name / "preprocessed" / "sessions"


def session_name_from_params(method: str, params: dict) -> str:
    """Build deterministic human-readable session name. All floats rounded to 6dp."""
    if method == "KIT":
        fc = round(float(params["fc"]), 6)
        return (
            f"KIT_a{params['anterior_roi_id']}_p{params['posterior_roi_id']}"
            f"_fc{fc:.4g}_sz{params['output_size']}"
        )
    elif method == "CenterROI":
        return (
            f"CenterROI_r{params['roi_id']}"
            f"_w{params['crop_width']}_h{params['crop_height']}"
        )
    raise ValueError(f"Unknown preprocessing method: {method!r}")


def session_id_from_name(session_name: str) -> str:
    """Return 8-char SHA256 hex digest of session_name."""
    return hashlib.sha256(session_name.encode()).hexdigest()[:8]


def get_session_dir(storage_path: str, project_name: str, session_id: str) -> Path:
    return _sessions_root(storage_path, project_name) / session_id


def load_session_meta(
    storage_path: str, project_name: str, session_id: str
) -> Optional[dict]:
    meta_path = get_session_dir(storage_path, project_name, session_id) / "session_meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def save_session_meta(
    storage_path: str, project_name: str, session_id: str, meta: dict
) -> None:
    """Atomic write via tmp + rename."""
    session_dir = get_session_dir(storage_path, project_name, session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    meta_path = session_dir / "session_meta.json"
    tmp = meta_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    tmp.rename(meta_path)


def list_sessions(storage_path: str, project_name: str) -> list[dict]:
    """Return session meta dicts sorted by created_at descending (newest first)."""
    root = _sessions_root(storage_path, project_name)
    if not root.exists():
        return []
    metas = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        meta = load_session_meta(storage_path, project_name, d.name)
        if meta:
            metas.append(meta)
    return sorted(metas, key=lambda m: m.get("created_at", ""), reverse=True)


def _session_lock_path(storage_path: str, project_name: str, session_id: str) -> Path:
    """Return the sidecar lock path for a session.

    The lock file lives next to session_meta.json so cross-process writers all
    contend on the same OS-level file lock.  Created on demand by ``FileLock``.
    """
    return get_session_dir(storage_path, project_name, session_id) / "session_meta.lock"


def find_or_create_session(
    storage_path: str, project_name: str, method: str, params: dict
) -> str:
    """Return session_id, creating session_meta.json if it doesn't yet exist.

    Holds the per-session FileLock for the check-then-act so two concurrent
    callers with the same params don't both decide to create.
    """
    import castle

    name = session_name_from_params(method, params)
    sid = session_id_from_name(name)
    # Ensure the directory exists so we can drop the lock file alongside.
    get_session_dir(storage_path, project_name, sid).mkdir(parents=True, exist_ok=True)
    lock_path = _session_lock_path(storage_path, project_name, sid)
    with FileLock(str(lock_path), timeout=_SESSION_LOCK_TIMEOUT):
        if load_session_meta(storage_path, project_name, sid) is None:
            meta = {
                "session_id": sid,
                "session_name": name,
                "method": method,
                "params": params,
                "videos": [],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "castle_version": getattr(castle, "__version__", ""),
            }
            save_session_meta(storage_path, project_name, sid, meta)
    return sid


def add_video_to_session(
    storage_path: str, project_name: str, session_id: str, video_name: str
) -> None:
    """Append video_name to videos list if not already present.

    Cross-platform ``FileLock`` guards the load → mutate → save sequence so two
    extraction jobs writing the same session don't lose each other's video.
    """
    lock_path = _session_lock_path(storage_path, project_name, session_id)
    # Lock file path may not yet exist if the session dir hasn't been built;
    # FileLock creates the file lazily but the parent must exist.
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path), timeout=_SESSION_LOCK_TIMEOUT):
        meta = load_session_meta(storage_path, project_name, session_id) or {}
        if video_name not in meta.get("videos", []):
            meta.setdefault("videos", []).append(video_name)
            save_session_meta(storage_path, project_name, session_id, meta)


def video_is_preprocessed(
    storage_path: str, project_name: str, session_id: str, video_name: str
) -> bool:
    """Check disk artifacts — both the video file and mask must exist."""
    video_dir = get_session_dir(storage_path, project_name, session_id) / video_name
    has_video = (video_dir / "stabilized.mp4").exists() or (video_dir / "cropped.mp4").exists()
    has_mask = (video_dir / "mask_list.h5").exists()
    return has_video and has_mask


def get_preprocessed_paths(
    storage_path: str, project_name: str, session_id: str, video_name: str
) -> tuple[Path, Path]:
    """Return (video_path, mask_path). Raises FileNotFoundError if artifacts are missing."""
    video_dir = get_session_dir(storage_path, project_name, session_id) / video_name
    for vname in ("stabilized.mp4", "cropped.mp4"):
        vpath = video_dir / vname
        if vpath.exists():
            mpath = video_dir / "mask_list.h5"
            if not mpath.exists():
                raise FileNotFoundError(f"mask_list.h5 missing in {video_dir}")
            return vpath, mpath
    raise FileNotFoundError(f"No preprocessed video found in {video_dir}")


def delete_session(storage_path: str, project_name: str, session_id: str) -> None:
    """Remove session directory from disk. Caller must also clean config['latent']."""
    session_dir = get_session_dir(storage_path, project_name, session_id)
    if session_dir.exists():
        shutil.rmtree(session_dir)
