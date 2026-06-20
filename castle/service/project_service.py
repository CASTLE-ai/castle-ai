"""
castle/service/project_service.py
Service layer for project CRUD operations.

All functions take simple types and return dicts/lists.
No gradio imports.
"""

import os
import json
import logging
from typing import List

from castle.core.project import get_project_config
from castle.utils.video_manager import (
    add_video_to_project,
    list_videos_in_directory,
    add_videos_batch,
)

logger = logging.getLogger(__name__)


def create_project(storage_path: str, name: str) -> dict:
    """
    Create a new CASTLE project directory with initial config.
    
    Args:
        storage_path: Root storage directory
        name: Project name
    
    Returns:
        dict with keys: 'path', 'name', 'created'
    
    Raises:
        FileExistsError: If project already exists
    """
    project_path = os.path.join(storage_path, name)
    if os.path.exists(project_path):
        raise FileExistsError(f"Project '{name}' already exists at {project_path}")
    
    os.makedirs(project_path, exist_ok=True)
    os.makedirs(os.path.join(project_path, 'sources'), exist_ok=True)
    
    config = {'source': []}
    config_path = os.path.join(project_path, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return {
        'path': project_path,
        'name': name,
        'created': True,
    }


def list_projects(storage_path: str) -> list:
    """
    List all projects in the storage directory.
    
    A directory is considered a project if it contains a config.json.
    
    Args:
        storage_path: Root storage directory
    
    Returns:
        List of project name strings
    """
    if not os.path.exists(storage_path):
        return []
    
    projects = []
    for entry in sorted(os.listdir(storage_path)):
        project_path = os.path.join(storage_path, entry)
        config_path = os.path.join(project_path, 'config.json')
        if os.path.isdir(project_path) and os.path.exists(config_path):
            projects.append(entry)
    return projects


def add_videos(storage_path: str, project_name: str, video_paths: List[str]) -> List[dict]:
    """
    Add video files to a project.
    
    Args:
        storage_path: Root storage directory
        project_name: Project name
        video_paths: List of absolute paths to video files to add
    
    Returns:
        List of dicts, each with keys: 'video_name', 'success', 'message'
    """
    results = []
    for path in video_paths:
        video_name = os.path.basename(path)
        success, message = add_video_to_project(storage_path, project_name, path, video_name)
        results.append({
            'video_name': video_name,
            'success': success,
            'message': message,
        })
    return results


# NOTE: Not yet exposed via CLI or UI
def add_videos_from_directory(storage_path: str, project_name: str, 
                              video_directory: str) -> dict:
    """
    Scan a directory for videos and add them all to a project.
    
    Args:
        storage_path: Root storage directory
        project_name: Project name
        video_directory: Directory containing video files
    
    Returns:
        dict with keys: 'success_count', 'fail_count', 'messages'
    """
    video_list = list_videos_in_directory(video_directory)
    if not video_list:
        return {'success_count': 0, 'fail_count': 0, 'messages': ['No videos found']}
    
    success_count, fail_count, messages = add_videos_batch(
        storage_path, project_name, video_directory, video_list
    )
    return {
        'success_count': success_count,
        'fail_count': fail_count,
        'messages': messages,
    }


def get_project_info(storage_path: str, project_name: str) -> dict:
    """
    Get project information.
    
    Args:
        storage_path: Root storage directory
        project_name: Project name
    
    Returns:
        dict with keys: 'name', 'path', 'videos', 'video_count', 'latent_count',
        'config'. If the project does not exist, the same keys are returned with
        empty values plus an 'error' key explaining that the project was not found.
    """
    # Detect a missing project explicitly. get_project_config() deliberately
    # swallows a missing/!malformed config.json and returns an empty default
    # config (for resilience mid-pipeline), so without this check a nonexistent
    # project would silently come back looking like a valid empty one.
    project_path = os.path.join(storage_path, project_name)
    if not os.path.exists(os.path.join(project_path, 'config.json')):
        return {
            'name': project_name,
            'path': project_path,
            'videos': [],
            'video_count': 0,
            'latent_count': 0,
            'config': {},
            'error': 'Project not found',
        }
    try:
        project_path, config = get_project_config(storage_path, project_name)
        videos = sorted(config.get('source', []))
        
        # Check for tracking/extraction status
        latent_info = config.get('latent', {})
        
        return {
            'name': project_name,
            'path': project_path,
            'videos': videos,
            'video_count': len(videos),
            'latent_count': len(latent_info),
            'config': config,
        }
    except FileNotFoundError:
        return {
            'name': project_name,
            'path': os.path.join(storage_path, project_name),
            'videos': [],
            'video_count': 0,
            'latent_count': 0,
            'config': {},
            'error': 'Project not found',
        }
