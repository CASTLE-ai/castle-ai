import os
import subprocess
import urllib.request


def download_file(url, destination):
    """Download file using urllib instead of wget subprocess."""
    # Ensure the destination directory exists
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    # Check if the file does not exist before downloading
    if not os.path.isfile(destination):
        print(f"Downloading {os.path.basename(destination)}...")
        urllib.request.urlretrieve(url, destination)
    else:
        print(f"{os.path.basename(destination)} already downloaded.")

def download_with_gdown(file_id, destination, notify_func=None):
    """
    Download file from Google Drive using gdown
    
    Args:
        file_id: Google Drive file ID
        destination: Target file path
        notify_func: Optional notification function for displaying messages in Gradio (e.g., gr.Info)
    """
    # Ensure the destination directory exists
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Check if the file does not exist before downloading
    if not os.path.isfile(destination):
        message = f"Downloading {os.path.basename(destination)}..."
        print(message)
        if notify_func:
            try:
                notify_func(message)
            except Exception:
                pass  # If notification fails, continue execution
        
        # Use --id flag to ensure correct file ID handling
        result = subprocess.run(['gdown', '--id', file_id, '--output', destination], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            error_msg = f"Failed to download {os.path.basename(destination)}"
            print(f"Error downloading {os.path.basename(destination)}:")
            print(result.stderr)
            if notify_func:
                try:
                    notify_func(error_msg)
                except Exception:
                    pass
            raise RuntimeError(f"Failed to download {os.path.basename(destination)}")
        
        success_msg = f"Successfully downloaded {os.path.basename(destination)}"
        print(success_msg)
        if notify_func:
            try:
                notify_func(success_msg)
            except Exception:
                pass
    else:
        message = f"{os.path.basename(destination)} already exists, skipping download"
        print(message)
        if notify_func:
            try:
                notify_func(message)
            except Exception:
                pass
