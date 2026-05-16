"""File download utilities (HTTP, Google Drive)."""

import os
import subprocess
import urllib.request

from castle.core.logging_config import setup_logger

logger = setup_logger(__name__)


def download_file(url, destination):
    """Download file using urllib instead of wget subprocess."""
    # Ensure the destination directory exists
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    # Check if the file does not exist before downloading
    if not os.path.isfile(destination):
        logger.info(f"Downloading {os.path.basename(destination)}...")
        urllib.request.urlretrieve(url, destination)
    else:
        logger.info(f"{os.path.basename(destination)} already downloaded.")

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
        logger.info(message)
        _safe_notify(notify_func, message)

        # Use --id flag to ensure correct file ID handling
        result = subprocess.run(['gdown', '--id', file_id, '--output', destination],
                              capture_output=True, text=True)
        if result.returncode != 0:
            error_msg = f"Failed to download {os.path.basename(destination)}"
            logger.error(f"Error downloading {os.path.basename(destination)}: {result.stderr}")
            _safe_notify(notify_func, error_msg)
            raise RuntimeError(f"Failed to download {os.path.basename(destination)}")

        success_msg = f"Successfully downloaded {os.path.basename(destination)}"
        logger.info(success_msg)
        _safe_notify(notify_func, success_msg)
    else:
        message = f"{os.path.basename(destination)} already exists, skipping download"
        logger.info(message)
        _safe_notify(notify_func, message)


def _safe_notify(notify_func, message):
    """Invoke a Gradio (or similar) notification callback without breaking the
    download on callback failure.

    The download path must keep working even when the GUI is wired
    incorrectly; we therefore swallow the callback exception but log it at
    warning level so it surfaces in CI / debug output. Previously this
    pattern used a bare ``except Exception: pass`` which left no trail at all.
    """
    if notify_func is None:
        return
    try:
        notify_func(message)
    except Exception as exc:  # noqa: BLE001 — callback errors must not break download
        logger.warning("notify_func failed (download continues): %s", exc)
