"""Video management utilities for Castle AI."""

import os
import json
import shutil
import logging

logger = logging.getLogger(__name__)


# Supported video file extensions
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv']


def is_video_file(file_path):
    """Check if a file is a video based on its extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        bool: True if the file is a video, False otherwise
    """
    _, ext = os.path.splitext(file_path)
    return ext.lower() in VIDEO_EXTENSIONS



from castle.core.project import get_project_config, save_project_config



def add_video_to_project(storage_path, project_name, video_source_path, video_name):
    """Add a video file to the project.
    
    Args:
        storage_path: Path to the storage directory
        project_name: Name of the project
        video_source_path: Source path of the video file
        video_name: Name of the video file
        
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        project_path, config = get_project_config(storage_path, project_name)
        
        # Initialize source list if not exists
        if 'source' not in config:
            config['source'] = []
        
        # Check if video already exists in project
        if video_name in config['source']:
            return False, "Video already exists in this project"
        
        # Create sources directory if not exists
        source_dir_path = os.path.join(project_path, 'sources')
        os.makedirs(source_dir_path, exist_ok=True)
        
        # Copy video file to project
        destination_path = os.path.join(source_dir_path, video_name)
        shutil.copyfile(video_source_path, destination_path)
        
        # Update configuration
        config['source'].append(video_name)
        save_project_config(storage_path, project_name, config)
        
        return True, f"Successfully added video: {video_name}"
    
    except FileNotFoundError as e:
        return False, f"File not found: {str(e)}"
    except Exception as e:
        return False, f"Failed to add video: {str(e)}"


def list_videos_in_directory(directory_path):
    """List all video files in a directory.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        list: Sorted list of video file names
    """
    if not os.path.exists(directory_path):
        return []
    
    try:
        videos = [
            f for f in os.listdir(directory_path)
            if os.path.isfile(os.path.join(directory_path, f)) and is_video_file(f)
        ]
        return sorted(videos)
    except Exception as e:
        logger.error(f"Error listing videos: {e}")
        return []


def get_project_videos(storage_path, project_name):
    """Get list of videos in a project.
    
    Args:
        storage_path: Path to the storage directory
        project_name: Name of the project
        
    Returns:
        list: List of video file names in the project, sorted
    """
    try:
        _, config = get_project_config(storage_path, project_name)
        videos = config.get('source', [])
        return sorted(videos) if videos else []
    except Exception as e:
        logger.error(f"Error getting project videos: {e}")
        return []


def load_video_metadata(storage_path, project_name, video_name):
    """Load video file and return metadata.
    
    Args:
        storage_path: Path to the storage directory
        project_name: Name of the project
        video_name: Name of the video file
        
    Returns:
        tuple: (video_path, frame_count) or (None, 0) if error
    """
    try:
        from castle.utils.video_io import ReadArray
        
        video_path = os.path.join(storage_path, project_name, 'sources', video_name)
        
        if not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}")
            return None, 0
        
        # Load video to get frame count
        video_array = ReadArray(video_path)
        frame_count = len(video_array)
        
        return video_path, frame_count
    
    except Exception as e:
        logger.error(f"Error loading video metadata: {e}")
        return None, 0


def scan_video_directory(directory_path):
    """Scan a directory and return video statistics.
    
    Args:
        directory_path: Path to the directory to scan
        
    Returns:
        tuple: (video_list, summary_text) where video_list is the list of all videos
               and summary_text is a formatted string showing statistics
    """
    if not directory_path or not os.path.exists(directory_path):
        return [], "Directory not found or invalid path"
    
    try:
        videos = list_videos_in_directory(directory_path)
        
        if not videos:
            return [], "No video files found in this directory"
        
        # Create summary text
        video_count = len(videos)
        preview_count = min(5, video_count)
        preview_videos = videos[:preview_count]
        
        summary = f"Found {video_count} video file(s)\n\n"
        summary += "Preview (first 5 files):\n"
        for i, video in enumerate(preview_videos, 1):
            summary += f"{i}. {video}\n"
        
        if video_count > 5:
            summary += f"\n... and {video_count - 5} more file(s)"
        
        return videos, summary
    
    except Exception as e:
        return [], f"Error scanning directory: {str(e)}"


def add_videos_batch(storage_path, project_name, video_directory, video_list):
    """Add multiple videos to a project in batch.
    
    Args:
        storage_path: Path to the storage directory
        project_name: Name of the project
        video_directory: Source directory containing videos
        video_list: List of video file names to add
        
    Returns:
        tuple: (success_count, fail_count, messages)
    """
    if not video_list:
        return 0, 0, ["No videos to add"]
    
    success_count = 0
    fail_count = 0
    messages = []
    
    for video_name in video_list:
        video_source_path = os.path.join(video_directory, video_name)
        success, message = add_video_to_project(
            storage_path,
            project_name,
            video_source_path,
            video_name
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
            messages.append(f"Failed: {video_name} - {message}")
    
    # Add summary message
    summary = f"Added {success_count} video(s) successfully"
    if fail_count > 0:
        summary += f", {fail_count} failed"
    messages.insert(0, summary)
    
    return success_count, fail_count, messages
