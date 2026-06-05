"""Tests for castle.core._early_env + the import-order invariants it protects.

HDF5_USE_FILE_LOCKING must be FALSE by the time h5py loads (it hangs on CephFS
otherwise), and `import castle` must stay light enough that this setter runs
before any heavy library — which means `import castle` must NOT pull in torch /
numpy / h5py itself.
"""

import subprocess
import sys


def test_apply_early_env_setdefault(monkeypatch):
    from castle.core import _early_env
    # Force a re-run on a clean slate.
    monkeypatch.setattr(_early_env, "_APPLIED", False)
    monkeypatch.delenv("HDF5_USE_FILE_LOCKING", raising=False)
    _early_env.apply_early_env()
    assert _early_env.os.environ["HDF5_USE_FILE_LOCKING"] == "FALSE"


def test_apply_early_env_respects_explicit_value(monkeypatch):
    from castle.core import _early_env
    monkeypatch.setattr(_early_env, "_APPLIED", False)
    monkeypatch.setenv("HDF5_USE_FILE_LOCKING", "TRUE")  # operator override
    _early_env.apply_early_env()
    assert _early_env.os.environ["HDF5_USE_FILE_LOCKING"] == "TRUE"


def test_import_castle_sets_hdf5_locking_and_stays_light():
    """`import castle` (clean interpreter, no pre-set env) must:
    1. leave HDF5_USE_FILE_LOCKING == FALSE, and
    2. NOT import torch / numpy / h5py (the lazy-import invariant the early-env
       hook depends on to win the race against h5py's import).
    """
    code = (
        "import os; os.environ.pop('HDF5_USE_FILE_LOCKING', None);\n"
        "import sys, castle;\n"
        "print(os.environ.get('HDF5_USE_FILE_LOCKING'));\n"
        "print('torch' in sys.modules, 'numpy' in sys.modules, 'h5py' in sys.modules)"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True).splitlines()
    assert out[0] == "FALSE"
    assert out[1] == "False False False"
