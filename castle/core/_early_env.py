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

import faulthandler
import os
import signal

_APPLIED = False

_FALSEY = {"0", "false", "no", "off"}


def _enable_fault_diagnostics() -> None:
    """Register an in-process thread-stack dumper for freeze diagnosis.

    Extraction over a flaky network filesystem (CephFS / NFS) can wedge a read
    indefinitely, leaving the process alive but silent — no exception, no log,
    GPU idle, every thread parked in ``futex_wait``. When that happens,
    ``kill -USR1 <pid>`` now makes the process dump *every Python thread's*
    stack to stderr, so the stuck call site is identifiable WITHOUT root or
    py-spy (hosts with ``kernel.yama.ptrace_scope=2`` block py-spy attach — the
    exact wall we hit diagnosing the cloud freeze). ``faulthandler.enable()``
    additionally dumps on fatal signals (SIGSEGV / SIGABRT). Opt out with
    ``CASTLE_FAULTHANDLER=0``.

    stdlib-only (``faulthandler`` / ``signal``); preserves the "import castle
    pulls in no heavy libraries" invariant. Signal registration only works on
    the main thread and SIGUSR1 only exists on POSIX, so a non-main-thread or
    Windows import silently skips registration (the ``enable()`` dump-on-crash
    still applies). DataLoader workers re-import castle (spawn) or inherit the
    handler (fork), so ``kill -USR1`` works on a wedged worker PID too.
    """
    if os.environ.get("CASTLE_FAULTHANDLER", "1").strip().lower() in _FALSEY:
        return
    try:
        faulthandler.enable()
    except (ValueError, OSError, RuntimeError):
        # stderr may be detached (no fileno) — diagnostics are best-effort.
        pass
    sigusr1 = getattr(signal, "SIGUSR1", None)
    if sigusr1 is not None:
        try:
            faulthandler.register(sigusr1, all_threads=True, chain=True)
        except (ValueError, OSError, RuntimeError):
            # Not the main thread, or platform without SIGUSR1 support.
            pass


def apply_early_env() -> None:
    """Set must-be-early environment defaults. Idempotent.

    Disables HDF5 POSIX file locking. CASTLE's HDF5 access is
    single-writer / multi-reader (writes are serialised by ``H5IO``'s in-process
    lock; DataLoader workers open read-only), so advisory file locking buys
    nothing and *hangs or serialises catastrophically on network filesystems*
    (CephFS / NFS) — exactly the cloud deployment target. ``setdefault`` keeps an
    explicit operator value (e.g. ``HDF5_USE_FILE_LOCKING=TRUE``) authoritative.

    Also registers an in-process freeze diagnostic (SIGUSR1 → all-thread stack
    dump) via :func:`_enable_fault_diagnostics`; see there for rationale.

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
    _enable_fault_diagnostics()
