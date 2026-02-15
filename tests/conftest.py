"""
tests/conftest.py
Shared fixtures for the CASTLE test suite.

This file is kept lightweight — no torch, no gradio, no model loading.
Heavy fixtures for integration tests live in tests/integration/conftest.py.
"""

import os
import pytest
import shutil
import tempfile
import numpy as np

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'


# ---- Lightweight fixtures (safe for unit tests) ----

@pytest.fixture
def tmp_storage():
    """Create a temporary storage directory for project tests."""
    d = tempfile.mkdtemp(prefix="castle_unit_")
    yield d
    shutil.rmtree(d, ignore_errors=True)
