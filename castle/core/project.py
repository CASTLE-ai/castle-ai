"""
castle/core/project.py
Core Project Management Logic.

Concurrency
-----------
``config.json`` is mutated by extraction (registers latent paths), session
deletion (clears latent entries), KIT save, and others.  A read-modify-write
on it must be atomic per project, both within one Python process (multiple
threads, e.g. two Gradio sessions) and across processes (two ``python app.py``
instances pointing at the same storage).

- ``_get_config_lock(project_path)`` returns a per-project ``threading.Lock``
  protecting same-process writers.
- ``update_config(storage_path, project_name)`` is a context manager that
  holds *both* the thread lock and a cross-platform ``filelock.FileLock`` on
  ``config.json.lock`` for the full load → mutate → save sequence.  Use this
  from any code path that performs read-modify-write on the config.
"""

import contextlib
import os
import json
import logging
import threading
from typing import Dict, Iterator, Optional, Tuple

from filelock import FileLock

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG: Dict = {}

# Per-project threading lock registry.  Two threads in the same process editing
# the same project's config.json contend on the same Lock object.  Different
# projects get independent locks so they don't serialize.
_PROJECT_LOCKS: Dict[str, threading.Lock] = {}
_PROJECT_LOCKS_GUARD = threading.Lock()
_CONFIG_FILELOCK_TIMEOUT = 5.0


def _get_config_lock(project_path: str) -> threading.Lock:
    """Return (and lazily create) the per-project in-process threading lock."""
    with _PROJECT_LOCKS_GUARD:
        lock = _PROJECT_LOCKS.get(project_path)
        if lock is None:
            lock = threading.Lock()
            _PROJECT_LOCKS[project_path] = lock
        return lock


def _config_filelock_path(project_path: str) -> str:
    return os.path.join(project_path, 'config.json.lock')


@contextlib.contextmanager
def update_config(storage_path: str, project_name: str) -> Iterator[Dict]:
    """Atomically read-modify-write ``config.json`` for a project.

    Usage::

        with update_config(storage_path, project_name) as config:
            config["latent"][k] = v
        # save happens on context exit

    Holds the per-project in-process ``threading.Lock`` AND a cross-process
    ``filelock.FileLock`` for the full load → mutate → save sequence so two
    concurrent extractors writing different keys do not lose updates.
    """
    project_path = os.path.join(storage_path, project_name)
    os.makedirs(project_path, exist_ok=True)
    thread_lock = _get_config_lock(project_path)
    file_lock = FileLock(_config_filelock_path(project_path), timeout=_CONFIG_FILELOCK_TIMEOUT)
    with thread_lock, file_lock:
        _, config = get_project_config(storage_path, project_name)
        yield config
        save_project_config(storage_path, project_name, config)


def get_project_config(storage_path: str, project_name: str) -> Tuple[str, Dict]:
    """Load project configuration file.

    Args:
        storage_path: Path to the storage directory
        project_name: Name of the project

    Returns:
        tuple: (project_path, config_dict)

    Raises:
        FileNotFoundError: If the config file is missing (re-raised with a
            friendlier message after logging).
    """
    project_path = os.path.join(storage_path, project_name)
    config_path = os.path.join(project_path, 'config.json')

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.warning(
            f"Config file not found for project '{project_name}' at {config_path}. "
            "Returning default empty config."
        )
        config = dict(_DEFAULT_CONFIG)
    except json.JSONDecodeError as exc:
        logger.warning(
            f"Malformed JSON in config file {config_path}: {exc}. "
            "Returning default empty config."
        )
        config = dict(_DEFAULT_CONFIG)

    return project_path, config


def save_project_config(storage_path: str, project_name: str, config: Dict) -> None:
    """Save project configuration file atomically.

    Writes to ``config.json.tmp`` first, then ``os.replace`` swaps it into
    place.  Prevents a crash mid-write from leaving the project with a
    truncated/empty ``config.json``.

    Args:
        storage_path: Path to the storage directory
        project_name: Name of the project
        config: Configuration dictionary to save
    """
    project_path = os.path.join(storage_path, project_name)
    config_path = os.path.join(project_path, 'config.json')
    tmp_path = config_path + '.tmp'

    with open(tmp_path, 'w') as f:
        json.dump(config, f, indent=2)
    os.replace(tmp_path, config_path)


# ---------------------------------------------------------------------------
# KIT (Kinematics Info Transfusion) parameter persistence
# ---------------------------------------------------------------------------

KIT_PARAMS_KEY = "kinematics_transfusion"


def save_kit_params(storage_path: str, project_name: str, params: dict) -> None:
    """Persist KIT parameters to the project config.

    Writes ``params`` under the ``"kinematics_transfusion"`` top-level key in
    ``config.json``.  Existing keys in the config are preserved.

    Args:
        storage_path: Path to the storage directory.
        project_name: Name of the project.
        params: KIT parameter dict.  Expected keys: ``body_roi_id``,
            ``head_roi_id``, ``fc``, ``order``, ``margin``, ``min_crop``,
            ``output_size``.

    Example:
        >>> save_kit_params('/data', 'my_exp', {
        ...     'body_roi_id': 1, 'head_roi_id': 2,
        ...     'fc': 0.25, 'order': 2, 'margin': 75,
        ...     'min_crop': 300, 'output_size': 518,
        ... })
    """
    with update_config(storage_path, project_name) as config:
        config[KIT_PARAMS_KEY] = params
    logger.info(
        "Saved KIT params for project '%s': %s",
        project_name,
        {k: v for k, v in params.items()},
    )


def load_kit_params(storage_path: str, project_name: str) -> Optional[dict]:
    """Load KIT parameters from the project config.

    Args:
        storage_path: Path to the storage directory.
        project_name: Name of the project.

    Returns:
        The KIT parameter dict if previously saved, otherwise ``None``.

    Example:
        >>> p = load_kit_params('/data', 'my_exp')
        >>> p is None or p['output_size'] in (518, 592)
        True
    """
    _, config = get_project_config(storage_path, project_name)
    return config.get(KIT_PARAMS_KEY, None)
