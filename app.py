import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import platform
from argparse import ArgumentParser

from castle.ui import create_ui

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

if __name__ == '__main__':
    app.queue(max_size=20)
    
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
    
    app.launch(
        server_name='0.0.0.0',
        share=COLAB_GPU or args.share,
        allowed_paths=allowed_paths if allowed_paths else None
    )