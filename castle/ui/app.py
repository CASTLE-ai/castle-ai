"""
castle/ui/app.py
Entry-point shim — delegates to main_ui.create_ui().
"""

from .main_ui import create_ui

__all__ = ["create_ui"]
