"""
castle/core/project.py
Core Project Management Logic.
"""

import os
import json
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG: Dict = {}


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
    """Save project configuration file.
    
    Args:
        storage_path: Path to the storage directory
        project_name: Name of the project
        config: Configuration dictionary to save
    """
    project_path = os.path.join(storage_path, project_name)
    config_path = os.path.join(project_path, 'config.json')
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


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
    project_path, config = get_project_config(storage_path, project_name)
    config[KIT_PARAMS_KEY] = params
    save_project_config(storage_path, project_name, config)
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
