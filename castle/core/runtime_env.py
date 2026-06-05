"""Runtime environment detection for cross-environment robustness.

CASTLE runs on very different machines: a dev box (few cores, ~32 GB RAM, local
NVMe, dual small GPUs) and cloud VMs (many cores, ~1 TB RAM, a single large GPU,
and a **shared network filesystem** like CephFS). The defaults that are right on
one are wrong on the other. This module is the single, lightweight source of
truth the rest of CASTLE consults to adapt.

Stdlib-only by design (``torch`` is imported lazily, only inside
:func:`gpu_info`) so it is cheap to import from the very top of
``castle/__init__.py`` — before ``h5py``/``numpy`` are pulled in — and from
``forkserver`` worker templates.

Detection is best-effort: every probe degrades gracefully (returns a safe
fallback) when a file/syscall is unavailable, and every value can be forced via
an environment variable so deployments and tests never depend on real cgroups /
CephFS being present.

Knobs (env):
  CASTLE_FORCE_NETWORK_FS   "1"/"0" — force is_network_fs() regardless of probe
  CASTLE_USABLE_CPUS        int — override the detected usable CPU count
  CASTLE_TOTAL_RAM_GB       float — override detected total RAM (GiB)
  CASTLE_SCRATCH_DIR        path — explicit node-local scratch dir for temp files
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from typing import Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Injectable for tests (point at a fixture instead of the real procfs/sysfs).
_MOUNTS_PATH = "/proc/mounts"
_MEMINFO_PATH = "/proc/meminfo"
_CGROUP_BASE = "/sys/fs/cgroup"

_GiB = 1024 ** 3

# Filesystem types that round-trip to a storage server: memmap page faults,
# HDF5 POSIX locks and small-file I/O are orders of magnitude slower on these.
_NETWORK_FSTYPE_PREFIXES = (
    "nfs", "ceph", "lustre", "cifs", "smb", "glusterfs", "beegfs", "afs", "9p",
)
# FUSE multiplexes many backends; only the network ones count. Bare "fuse"
# (gocryptfs, mergerfs, …) must NOT be flagged or we needlessly throttle local
# work — so we match on the named sub-type only.
_NETWORK_FUSE_TOKENS = (
    "ceph", "nfs", "sshfs", "glusterfs", "beegfs", "davfs", "s3", "smb",
)


# ---------------------------------------------------------------------------
# Filesystem type
# ---------------------------------------------------------------------------

def _unescape_mount(field: str) -> str:
    """Decode the octal escapes the kernel uses in /proc/mounts (\\040 = space)."""
    if "\\" not in field:
        return field
    out = []
    i = 0
    while i < len(field):
        if field[i] == "\\" and i + 3 < len(field) + 1 and field[i + 1:i + 4].isdigit():
            try:
                out.append(chr(int(field[i + 1:i + 4], 8)))
                i += 4
                continue
            except ValueError:
                pass
        out.append(field[i])
        i += 1
    return "".join(out)


def _iter_mounts() -> Iterator[Tuple[str, str]]:
    """Yield ``(mountpoint, fstype)`` from /proc/mounts. Empty on non-Linux."""
    try:
        with open(_MOUNTS_PATH, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 3:
                    yield _unescape_mount(parts[1]), parts[2]
    except OSError:
        return


def fs_type(path: str) -> str:
    """Filesystem type string for *path* (e.g. ``"ext4"``, ``"ceph"``, ``"nfs4"``).

    Resolves the longest mount-point prefix of ``realpath(path)`` in
    /proc/mounts. Returns ``"unknown"`` when procfs is unavailable (non-Linux)
    or no mount matches.
    """
    try:
        target = os.path.realpath(path)
    except OSError:
        target = path

    best_mp = ""
    best_fstype = "unknown"
    for mp, fstype in _iter_mounts():
        if mp == "/" or target == mp or target.startswith(mp.rstrip("/") + "/"):
            if len(mp) >= len(best_mp):
                best_mp = mp
                best_fstype = fstype
    return best_fstype


def _is_network_fstype(fstype: str) -> bool:
    fstype = fstype.lower()
    if fstype.startswith("fuse."):
        return any(tok in fstype for tok in _NETWORK_FUSE_TOKENS)
    return fstype.startswith(_NETWORK_FSTYPE_PREFIXES)


def is_network_fs(path: str) -> bool:
    """True when *path* lives on a network/shared filesystem.

    ``CASTLE_FORCE_NETWORK_FS`` (``"1"``/``"0"``) overrides the probe — useful
    for deployments on exotic mounts and for tests.
    """
    forced = os.environ.get("CASTLE_FORCE_NETWORK_FS", "").strip().lower()
    if forced in ("1", "true", "yes", "on"):
        return True
    if forced in ("0", "false", "no", "off"):
        return False
    return _is_network_fstype(fs_type(path))


# ---------------------------------------------------------------------------
# Usable CPU count (cgroup / affinity aware)
# ---------------------------------------------------------------------------

def _affinity_cpus() -> Optional[int]:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


def _read_int(path: str) -> Optional[int]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def _cgroup_cpu_quota() -> Optional[int]:
    """CPU count implied by a cgroup quota, or None when unlimited/absent.

    cgroup v2: ``cpu.max`` = ``"<quota> <period>"`` (or ``"max <period>"``).
    cgroup v1: ``cpu/cpu.cfs_quota_us`` (``-1`` = unlimited) and
    ``cpu/cpu.cfs_period_us``.
    """
    # v2
    try:
        with open(os.path.join(_CGROUP_BASE, "cpu.max"), "r", encoding="utf-8") as fh:
            quota_s, _, period_s = fh.read().strip().partition(" ")
        if quota_s != "max":
            quota, period = int(quota_s), int(period_s or 100000)
            if quota > 0 and period > 0:
                return max(1, math.ceil(quota / period))
    except (OSError, ValueError):
        pass
    # v1
    q = _read_int(os.path.join(_CGROUP_BASE, "cpu", "cpu.cfs_quota_us"))
    p = _read_int(os.path.join(_CGROUP_BASE, "cpu", "cpu.cfs_period_us"))
    if q is not None and q > 0 and p and p > 0:
        return max(1, math.ceil(q / p))
    return None


def usable_cpu_count() -> int:
    """Cores this process may actually use — the min of affinity, cgroup quota
    and ``os.cpu_count()``. This is what CASTLE sizes worker pools against, so a
    64-core host that gives a JupyterLab container an 8-CPU quota yields 8, not
    64. ``CASTLE_USABLE_CPUS`` overrides. Always >= 1.
    """
    override = os.environ.get("CASTLE_USABLE_CPUS", "").strip()
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    candidates = [c for c in (_affinity_cpus(), _cgroup_cpu_quota(), os.cpu_count()) if c]
    return max(1, min(candidates)) if candidates else 1


# ---------------------------------------------------------------------------
# RAM
# ---------------------------------------------------------------------------

def _meminfo_bytes(key: str) -> Optional[int]:
    try:
        with open(_MEMINFO_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) * 1024  # value is in kB
    except (OSError, ValueError, IndexError):
        return None
    return None


def _cgroup_mem_max() -> Optional[int]:
    try:
        with open(os.path.join(_CGROUP_BASE, "memory.max"), "r", encoding="utf-8") as fh:
            raw = fh.read().strip()
        if raw != "max":
            return int(raw)
    except (OSError, ValueError):
        pass
    return None


def total_ram_bytes() -> Optional[int]:
    """Total usable RAM in bytes, capped by a cgroup ``memory.max`` if present.

    ``CASTLE_TOTAL_RAM_GB`` overrides. None when nothing can be read.
    """
    override = os.environ.get("CASTLE_TOTAL_RAM_GB", "").strip()
    if override:
        try:
            return int(float(override) * _GiB)
        except ValueError:
            pass
    candidates = [v for v in (_meminfo_bytes("MemTotal"), _cgroup_mem_max()) if v]
    if not candidates:
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            return None
    return min(candidates)


def available_ram_bytes() -> Optional[int]:
    """Currently-available RAM in bytes (live, not cached).

    Read from ``/proc/meminfo`` ``MemAvailable`` (capped by a cgroup
    ``memory.max`` if present). None on non-Linux / when procfs is unreadable —
    callers then fall back to a conservative default.
    """
    avail = _meminfo_bytes("MemAvailable")
    if avail is None:
        return None
    cap = _cgroup_mem_max()
    return min(avail, cap) if cap else avail


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

def gpu_info() -> List[Dict[str, object]]:
    """Per-device GPU info: ``[{index, name, total_bytes, free_bytes}, …]``.

    Empty list when CUDA / torch is unavailable. ``torch`` is imported lazily so
    this module stays import-light.
    """
    out: List[Dict[str, object]] = []
    try:
        import torch  # noqa: PLC0415
        if not torch.cuda.is_available():
            return out
        for idx in range(torch.cuda.device_count()):
            try:
                free, total = torch.cuda.mem_get_info(idx)
            except Exception:
                props = torch.cuda.get_device_properties(idx)
                free = total = int(props.total_memory)
            out.append({
                "index": idx,
                "name": torch.cuda.get_device_name(idx),
                "total_bytes": int(total),
                "free_bytes": int(free),
            })
    except Exception:
        return out
    return out


# ---------------------------------------------------------------------------
# Node-local scratch directory (for temp memmap / probe files)
# ---------------------------------------------------------------------------

def _free_bytes(path: str) -> int:
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize
    except OSError:
        return 0


def _shm_is_good_target(min_free_bytes: int) -> bool:
    """Whether ``/dev/shm`` (RAM-backed tmpfs) is a sane spill target here.

    Spilling to /dev/shm consumes RAM — great on a big-RAM cloud box (RAM is
    abundant, the real disk is slow CephFS), but counter-productive on a small-
    RAM workstation where we spill *because* RAM is tight. So require both a
    large total RAM (``CASTLE_SHM_MIN_TOTAL_RAM_GB``, default 128) and >= 1.5x
    free headroom for the request.
    """
    shm = "/dev/shm"
    if not (os.path.isdir(shm) and fs_type(shm) == "tmpfs"):
        return False
    total = total_ram_bytes()
    try:
        min_total = float(os.environ.get("CASTLE_SHM_MIN_TOTAL_RAM_GB", "128")) * _GiB
    except ValueError:
        min_total = 128 * _GiB
    if total is None or total < min_total:
        return False  # small-RAM box → spill to real local disk instead
    return min_free_bytes <= 0 or _free_bytes(shm) >= int(min_free_bytes * 1.5)


def scratch_dir(min_free_bytes: int = 0) -> str:
    """Resolve a node-local directory for large temporary files (memmaps, encode
    probes). NEVER returns a network filesystem silently.

    Order:
      1. ``CASTLE_SCRATCH_DIR`` (if it exists / is creatable).
      2. ``/dev/shm`` when it is a RAM-backed tmpfs AND the box has lots of RAM
         with >= 1.5x headroom — ideal on big-RAM cloud boxes (spill to RAM, not
         the slow network disk); skipped on small-RAM workstations.
      3. ``tempfile.gettempdir()`` when it is NOT a network FS.
      4. Fallback to ``gettempdir()`` with a loud warning (everything looked
         network-backed) so a multi-GB memmap never lands on CephFS unannounced.
    """
    explicit = os.environ.get("CASTLE_SCRATCH_DIR", "").strip()
    if explicit:
        try:
            os.makedirs(explicit, exist_ok=True)
            return explicit
        except OSError:
            logger.warning("CASTLE_SCRATCH_DIR=%s is not usable; ignoring.", explicit)

    if _shm_is_good_target(min_free_bytes):
        return "/dev/shm"

    tmp = tempfile.gettempdir()
    if not is_network_fs(tmp):
        return tmp

    logger.warning(
        "Scratch dir %s is on a network filesystem (%s); large temporary files "
        "will be slow. Set CASTLE_SCRATCH_DIR to node-local storage (e.g. a "
        "local SSD or /dev/shm).", tmp, fs_type(tmp),
    )
    return tmp


# ---------------------------------------------------------------------------
# One-shot summary (startup log + Gradio notice)
# ---------------------------------------------------------------------------

def summary(storage_path: Optional[str] = None) -> Dict[str, object]:
    """Snapshot of the detected environment + key applied defaults.

    Args:
        storage_path: The directory whose filesystem matters (the project
            storage root). Defaults to the current working directory.
    """
    probe = storage_path or os.getcwd()
    net = is_network_fs(probe)
    total = total_ram_bytes()
    avail = available_ram_bytes()
    gpus = gpu_info()
    scratch = scratch_dir()
    return {
        "storage_root": probe,
        "fs_type": fs_type(probe),
        "network_fs_detected": net,
        "usable_cpus": usable_cpu_count(),
        "total_ram_gb": round(total / _GiB, 1) if total else None,
        "available_ram_gb": round(avail / _GiB, 1) if avail else None,
        "gpu_count": len(gpus),
        "gpus": gpus,
        "scratch_dir": scratch,
        "scratch_on_network_fs": is_network_fs(scratch),
        "hdf5_file_locking": os.environ.get("HDF5_USE_FILE_LOCKING", "(unset)"),
    }
