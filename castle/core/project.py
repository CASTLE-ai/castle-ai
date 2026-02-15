"""
castle/core/project.py
Core Project Management Logic.
"""

import os
import json
from typing import Tuple, Dict

def get_project_config(storage_path: str, project_name: str) -> Tuple[str, Dict]:
    """Load project configuration file.
    
    Args:
        storage_path: Path to the storage directory
        project_name: Name of the project
        
    Returns:
        tuple: (project_path, config_dict)
    """
    project_path = os.path.join(storage_path, project_name)
    config_path = os.path.join(project_path, 'config.json')
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
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
