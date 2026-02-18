"""
Entry point for: python -m castle.desktop
"""

import sys
import os

# Ensure the project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from castle.desktop.app import main  # noqa: E402

if __name__ == '__main__':
    main()
