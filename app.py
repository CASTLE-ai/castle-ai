import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import platform
import atexit
from argparse import ArgumentParser

# System configuration
OS_SYS = platform.uname().system
COLAB_GPU = 'COLAB_GPU' in os.environ

# Windows starts DataLoader/ProcessPool workers with *spawn*, which re-imports
# this file as ``__mp_main__`` in every worker. Without this guard each worker
# re-imports Gradio, re-parses argv and rebuilds the whole UI — slow, and enough
# extra RAM per worker to push an 8 GB laptop into swap. Imports by uvicorn/gunicorn
# (``__name__ == 'app'``) and direct runs still build the app as before.
if __name__ != '__mp_main__':
    import gradio as gr
    from castle.ui import create_ui
    from castle.ui.main_ui import CASTLE_JS, CASTLE_CSS

    # Seed all global RNGs at startup so GUI-driven runs are reproducible by default
    # (honours CASTLE_SEED; mirrors the CLI's --seed). This does NOT lock UMAP — the
    # Behavior Microscope keeps its own per-stage seed / re-roll control.
    from castle.core.seed import set_global_seed
    try:
        _CASTLE_SEED = int(os.environ.get('CASTLE_SEED', '42'))
    except ValueError:
        _CASTLE_SEED = 42
    set_global_seed(_CASTLE_SEED)

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
            prevent_thread_lock=True,
        )
        # Gradio prints the bind address (http://0.0.0.0:...), which browsers on
        # Windows cannot open; students then fell back to --share links. Print
        # the address to use on this computer after the server is up.
        print(f"\n  CASTLE is running. Open http://127.0.0.1:{app.server_port} "
              "in your browser.\n", flush=True)
        app.block_thread()
    except KeyboardInterrupt:
        # uvicorn re-raises SIGINT as KeyboardInterrupt; swallow so cleanup runs
        # quietly instead of dumping a traceback on a normal Ctrl+C.
        pass
    finally:
        _castle_shutdown()