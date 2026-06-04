"""CPU worker-count policy for the pre-process parallel stages.

Stdlib-only and lightweight (no cv2/torch/h5py) so it is cheap to import from a
``forkserver`` worker template. The policy: use as many cores as available but
**leave a few free** for the OS and other apps — never a hardcoded worker count,
since CASTLE deploys to machines with very different core counts.

Knobs (env):
  CASTLE_RESERVED_CORES         cores to leave free (default 4)
  CASTLE_CENTROID_WORKERS       centroid pool size override ("1" → serial)
  CASTLE_PREPROCESS_WARP_WORKERS  encode/mask pool size override ("1" → serial)
"""

import os

_RESERVED_CORES = 4  # leave headroom for OS + other programs


def reserved_cores() -> int:
    """How many cores to keep free. ``CASTLE_RESERVED_CORES`` overrides the default 4."""
    raw = os.environ.get("CASTLE_RESERVED_CORES", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return _RESERVED_CORES


def default_workers() -> int:
    """Default pool size = ``cpu_count - reserved_cores`` (≥1), derived at runtime."""
    return max(1, (os.cpu_count() or 1) - reserved_cores())


def resolve_workers(env_var: str = "") -> int:
    """Resolve a pool size: an explicit ``env_var`` override (incl. ``"1"`` → serial)
    wins; otherwise :func:`default_workers`. Always clamped to ``[1, cpu_count]`` —
    no hardcoded cap, so it scales with the host."""
    cpu = os.cpu_count() or 1
    raw = os.environ.get(env_var, "").strip() if env_var else ""
    if raw:
        try:
            n = int(raw)
        except ValueError:
            n = default_workers()
    else:
        n = default_workers()
    return max(1, min(n, cpu))
