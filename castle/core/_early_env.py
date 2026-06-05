"""Process-wide environment defaults that MUST be set before heavy libraries load.

Some environment variables are only read once, when a native library is first
imported — setting them afterwards is a silent no-op. The prime example is
``HDF5_USE_FILE_LOCKING``: the HDF5 C library reads it at ``import h5py``.

``import castle`` deliberately pulls in no heavy libraries (see
``castle/__init__.py``'s lazy ``__getattr__``), so calling :func:`apply_early_env`
at the very top of that module guarantees these win the race against any
``import h5py`` / ``import numpy`` that later flows through a CASTLE submodule.

Stdlib-only and side-effect-light on purpose; safe to import from anywhere.
"""

from __future__ import annotations

import os

_APPLIED = False


def apply_early_env() -> None:
    """Set must-be-early environment defaults. Idempotent.

    Currently disables HDF5 POSIX file locking. CASTLE's HDF5 access is
    single-writer / multi-reader (writes are serialised by ``H5IO``'s in-process
    lock; DataLoader workers open read-only), so advisory file locking buys
    nothing and *hangs or serialises catastrophically on network filesystems*
    (CephFS / NFS) — exactly the cloud deployment target. ``setdefault`` keeps an
    explicit operator value (e.g. ``HDF5_USE_FILE_LOCKING=TRUE``) authoritative.

    Thread oversubscription is intentionally NOT handled here: pinning
    ``OMP_NUM_THREADS`` process-wide would also throttle the clustering stage,
    which legitimately wants many cores. Extraction caps torch intra-op threads
    locally instead (see ``extractor._apply_extraction_thread_cap``).
    """
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
