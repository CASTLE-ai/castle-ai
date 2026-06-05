import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import platform
import atexit
from argparse import ArgumentParser
import gradio as gr
from castle.ui import create_ui
from castle.ui.main_ui import CASTLE_JS, CASTLE_CSS

# System configuration
OS_SYS = platform.uname().system
COLAB_GPU = 'COLAB_GPU' in os.environ

# Parse command line arguments
parser = ArgumentParser()
parser.add_argument("--root", dest="root")
parser.add_argument("--share", action="store_true", default=False, 
                    help="Share the Gradio app via public URL")
args = parser.parse_args()

# Create application
app = create_ui(OS_SYS, args.root)

# Enable the Gradio queue at module scope.  Generators, gr.Progress(), and
# `.then()` chains all require the queue; keeping it inside the __main__
# guard disabled all streaming whenever app.py was imported by a production
# server (uvicorn/gunicorn) instead of run directly.
app.queue(max_size=20)


def _castle_shutdown():
    """Best-effort resource reclamation on exit / Ctrl+C.

    CASTLE is one long-lived Gradio server; work runs on daemon threads whose
    *child processes* (centroid ProcessPools) and CUDA context are NOT reaped
    deterministically when the server stops. This hook force-terminates any
    live preprocessing pool and releases CUDA caches so a Ctrl+C doesn't leave
    orphaned workers holding RAM/VRAM/file-handles. Idempotent; never raises.
    """
    try:
        from castle.core.stabilized_camera import shutdown_live_pools
        shutdown_live_pools()
    except Exception:
        pass
    try:
        from castle.core.extractor import clear_device_encoder_cache
        clear_device_encoder_cache()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


atexit.register(_castle_shutdown)

if __name__ == '__main__':
    # Set allowed_paths to resolve Colab path permission issues
    allowed_paths = []
    if COLAB_GPU:
        # In Colab environment, allow access to the following paths
        allowed_paths = [
            "/content/drive/MyDrive/castle-projects",  # Google Drive project directory
            "/tmp",  # Temporary directory
            "/content",  # Colab content directory
        ]
    
    if args.root:
        allowed_paths.append(args.root)
    
    try:
        app.launch(
            server_name='0.0.0.0',
            share=COLAB_GPU or args.share,
            allowed_paths=allowed_paths if allowed_paths else None,
            theme=gr.themes.Soft(),
            js=CASTLE_JS,
            css=CASTLE_CSS,
        )
    except KeyboardInterrupt:
        # uvicorn re-raises SIGINT as KeyboardInterrupt; swallow so cleanup runs
        # quietly instead of dumping a traceback on a normal Ctrl+C.
        pass
    finally:
        _castle_shutdown()