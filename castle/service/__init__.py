"""
castle.service
Service Layer for CASTLE — provides clean, Gradio-independent interfaces
for all core operations (C-01).

This layer sits between the UI (Gradio/CLI/Desktop) and the Core modules.
Service functions:
- Take simple types (str, int, dict) as input
- Return dicts/dataclasses, never gr.update() objects
- Do NOT import gradio
- Manage state (project lifecycle, model lifetime, etc.)
"""

from castle.service import project_service, extraction_service, clustering_service
