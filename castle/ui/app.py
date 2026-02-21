"""
castle/ui/app.py
Entry-point shim — delegates to main_ui.create_ui().

The wizard tab (🧭 Quick Start) is inserted as Tab 0 by main_ui;
all legacy pipeline tabs follow in order.
"""

from .main_ui import create_ui

__all__ = ["create_ui"]
