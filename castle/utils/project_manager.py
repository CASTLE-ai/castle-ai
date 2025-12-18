"""Project management utilities for Castle AI."""

import os
import json
import shutil
import datetime
import logging

logger = logging.getLogger(__name__)


def list_projects(storage_path):
    """List all project directories in the storage path.
    
    Args:
        storage_path: Path to the storage directory
        
    Returns:
        List of project directory names, sorted if more than one exists
    """
    if not os.path.exists(storage_path):
        return []
    
    projects = [
        d for d in os.listdir(storage_path) 
        if os.path.isdir(os.path.join(storage_path, d))
    ]
    
    if len(projects) > 1:
        return sorted(projects, reverse=True)
    return projects


def create_project(storage_path, project_name):
    """Create a new project with config file.
    
    Args:
        storage_path: Path to the storage directory
        project_name: Name of the new project
        
    Raises:
        FileExistsError: If project already exists
    """
    project_path = os.path.join(storage_path, project_name)
    
    if os.path.exists(project_path):
        raise FileExistsError(f"Project '{project_name}' already exists")
    
    os.makedirs(project_path)
    
    config_path = os.path.join(project_path, 'config.json')
    config = {
        'project_name': project_name,
        'created_at': datetime.datetime.now().isoformat()
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def delete_project(storage_path, project_name):
    """Delete a project directory.
    
    Args:
        storage_path: Path to the storage directory
        project_name: Name of the project to delete
        
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    project_path = os.path.join(storage_path, project_name)
    
    if os.path.exists(project_path):
        shutil.rmtree(project_path)
        logger.info(f"Deleted project: {project_path}")
        return True
    else:
        logger.warning(f"Project not found: {project_path}")
        return False


def generate_default_project_name(custom_name=''):
    """Generate a default project name with timestamp.
    
    Args:
        custom_name: Optional custom name to use
        
    Returns:
        str: Project name
    """
    if custom_name and len(custom_name.strip()) > 0:
        return custom_name.strip()
    
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    return f'{timestamp}-Project'


def initialize_storage(root_path):
    """Initialize storage directory if it doesn't exist.
    
    Args:
        root_path: Path to the root storage directory
        
    Returns:
        str: Normalized storage path with trailing separator
    """
    if root_path is None or root_path == '':
        root_path = os.path.join('projects', '')
    
    # Ensure path ends with separator
    if not root_path.endswith(os.sep):
        root_path += os.sep
    
    # Create directory if it doesn't exist
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    
    return root_path

