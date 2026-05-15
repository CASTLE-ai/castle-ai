"""
castle/service/annotation_service.py
Service for managing behavior classification schemes and annotations.
"""

import os
import json
import logging
import pandas as pd
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Default classification schemes
DEFAULT_SCHEMES = {
    "mice-5-class": ["Running", "Walking", "Immobile", "Sniffing", "Other"],
    "mice-10-class": [
        "Sniffing", "Turn Right", "Turn Left",
        "Supported Rearing", "Unsupported Rearing", "Grooming",
        "Running", "Walking", "Immobile", "Other",
    ],
}


def _schemes_path(storage_path: str, project_name: str, session_id: Optional[str] = None) -> str:
    """Path to the custom classification schemes JSON file.

    If *session_id* is given the file lives inside the session directory;
    otherwise it falls back to the cluster root for backward compatibility.

    Args:
        storage_path: Root storage directory.
        project_name: Project name.
        session_id: Optional session ID for per-session storage.

    Returns:
        Absolute path to ``classification_schemes.json``.
    """
    if session_id:
        return os.path.join(
            storage_path, project_name,
            'cluster', 'sessions', session_id,
            'classification_schemes.json',
        )
    return os.path.join(storage_path, project_name, 'cluster', 'classification_schemes.json')


def _annotations_path(storage_path: str, project_name: str, session_id: Optional[str] = None) -> str:
    """Path to the annotations CSV file.

    If *session_id* is given the file lives inside the session directory;
    otherwise it falls back to the cluster root for backward compatibility.

    Args:
        storage_path: Root storage directory.
        project_name: Project name.
        session_id: Optional session ID for per-session storage.

    Returns:
        Absolute path to ``annotations.csv``.
    """
    if session_id:
        return os.path.join(
            storage_path, project_name,
            'cluster', 'sessions', session_id,
            'annotations.csv',
        )
    return os.path.join(storage_path, project_name, 'cluster', 'annotations.csv')


def list_schemes(
    storage_path: str,
    project_name: str,
    session_id: Optional[str] = None,
) -> Dict[str, List[str]]:
    """List available classification schemes (default + custom).

    Args:
        storage_path: Root storage directory.
        project_name: Project name.
        session_id: Optional session ID for per-session custom schemes.

    Returns:
        Dict mapping scheme name to list of label strings.
    """
    schemes = dict(DEFAULT_SCHEMES)

    custom_path = _schemes_path(storage_path, project_name, session_id)
    if os.path.exists(custom_path):
        try:
            with open(custom_path, 'r') as f:
                custom = json.load(f)
            schemes.update(custom)
        except Exception as e:
            logger.warning(f"Failed to load custom schemes: {e}")

    return schemes


def get_scheme_labels(
    storage_path: str,
    project_name: str,
    scheme_name: str,
    session_id: Optional[str] = None,
) -> List[str]:
    """Get labels for a specific classification scheme.

    Args:
        storage_path: Root storage directory.
        project_name: Project name.
        scheme_name: Name of the scheme.
        session_id: Optional session ID for per-session custom schemes.

    Returns:
        List of label strings, or empty list if not found.
    """
    schemes = list_schemes(storage_path, project_name, session_id)
    return schemes.get(scheme_name, [])


def save_scheme(
    storage_path: str,
    project_name: str,
    name: str,
    labels: List[str],
    session_id: Optional[str] = None,
) -> None:
    """Save a custom classification scheme.

    Args:
        storage_path: Root storage directory.
        project_name: Project name.
        name: Scheme name.
        labels: List of behavior label strings.
        session_id: Optional session ID for per-session storage.
    """
    custom_path = _schemes_path(storage_path, project_name, session_id)
    os.makedirs(os.path.dirname(custom_path), exist_ok=True)

    existing = {}
    if os.path.exists(custom_path):
        try:
            with open(custom_path, 'r') as f:
                existing = json.load(f)
        except Exception:
            pass

    existing[name] = labels
    with open(custom_path, 'w') as f:
        json.dump(existing, f, indent=2)

    logger.info(f"Saved classification scheme '{name}' with {len(labels)} labels")


# ---------------------------------------------------------------------------
# Project-level active scheme persistence
# ---------------------------------------------------------------------------

_DEFAULT_SCHEME = "mice-10-class"


def _annotation_config_path(storage_path: str, project_name: str) -> str:
    return os.path.join(storage_path, project_name, "cluster", "annotation_config.json")


def get_active_scheme(storage_path: str, project_name: str) -> str:
    """Return the project-level active classification scheme name.

    Falls back to ``_DEFAULT_SCHEME`` when no config has been saved yet.
    """
    path = _annotation_config_path(storage_path, project_name)
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f).get("classification_scheme", _DEFAULT_SCHEME)
        except Exception as exc:
            logger.warning("Could not read annotation_config.json: %s", exc)
    return _DEFAULT_SCHEME


def save_active_scheme(storage_path: str, project_name: str, scheme_name: str) -> None:
    """Persist the active classification scheme name at the project level."""
    if not storage_path or not project_name or not scheme_name:
        return
    path = _annotation_config_path(storage_path, project_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing: dict = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = json.load(f)
        except Exception:
            pass
    existing["classification_scheme"] = scheme_name
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


def load_annotations(
    storage_path: str,
    project_name: str,
    session_id: Optional[str] = None,
) -> Dict[str, dict]:
    """Load existing annotations from annotations.csv.

    Args:
        storage_path: Root storage directory.
        project_name: Project name.
        session_id: Optional session ID for per-session annotations.

    Returns:
        Dict mapping cluster_name to annotation dict:
            {'behavior_label': str, 'scheme': str, 'annotator': str, 'timestamp': str}
    """
    csv_path = _annotations_path(storage_path, project_name, session_id)
    if not os.path.exists(csv_path):
        return {}

    try:
        df = pd.read_csv(csv_path)
        annotations = {}
        for _, row in df.iterrows():
            annotations[row['cluster_name']] = {
                'behavior_label': row.get('behavior_label', ''),
                'scheme': row.get('scheme', ''),
                'annotator': row.get('annotator', ''),
                'timestamp': row.get('timestamp', ''),
            }
        return annotations
    except Exception as e:
        logger.warning(f"Failed to load annotations: {e}")
        return {}


def save_annotations(
    storage_path: str,
    project_name: str,
    annotations: Dict[str, dict],
    session_id: Optional[str] = None,
) -> str:
    """Save annotations to annotations.csv.

    Args:
        storage_path: Root storage directory.
        project_name: Project name.
        annotations: Dict mapping cluster_name to annotation dict.
        session_id: Optional session ID for per-session storage.

    Returns:
        Path to saved CSV file.
    """
    csv_path = _annotations_path(storage_path, project_name, session_id)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    rows = []
    for cluster_name, ann in annotations.items():
        rows.append({
            'cluster_name': cluster_name,
            'behavior_label': ann.get('behavior_label', ''),
            'scheme': ann.get('scheme', ''),
            'annotator': ann.get('annotator', ''),
            'timestamp': ann.get('timestamp', ''),
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(rows)} annotations to {csv_path}")
    return csv_path
