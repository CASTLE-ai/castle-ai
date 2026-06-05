"""Unit tests for castle.core.runtime_env (cross-environment detection).

Detection probes (/proc/mounts, /proc/meminfo, /sys/fs/cgroup, sched_getaffinity,
statvfs) are injected via the module's path globals or monkeypatched, so these
tests never depend on real CephFS / cgroups / GPUs being present.
"""

import os

import pytest

from castle.core import runtime_env as rt


# ---------------------------------------------------------------------------
# filesystem type / network detection
# ---------------------------------------------------------------------------

def _write_mounts(tmp_path, lines):
    p = tmp_path / "mounts"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


@pytest.fixture(autouse=True)
def _clear_force(monkeypatch):
    # Ensure no leaked override from the ambient environment.
    monkeypatch.delenv("CASTLE_FORCE_NETWORK_FS", raising=False)
    monkeypatch.delenv("CASTLE_USABLE_CPUS", raising=False)
    monkeypatch.delenv("CASTLE_TOTAL_RAM_GB", raising=False)
    monkeypatch.delenv("CASTLE_SCRATCH_DIR", raising=False)


def test_fs_type_longest_prefix_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "_MOUNTS_PATH", _write_mounts(tmp_path, [
        "/dev/sda1 / ext4 rw 0 0",
        "10.0.0.1:/share /home/u/sharedfs ceph rw 0 0",
    ]))
    # realpath of a path under the ceph mount → ceph (longer prefix beats "/")
    monkeypatch.setattr(rt.os.path, "realpath", lambda p: p)
    assert rt.fs_type("/home/u/sharedfs/proj/latent") == "ceph"
    assert rt.fs_type("/var/tmp") == "ext4"


def test_fs_type_unknown_when_no_mounts(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "_MOUNTS_PATH", str(tmp_path / "does_not_exist"))
    assert rt.fs_type("/anything") == "unknown"


def test_fs_type_decodes_octal_escaped_mountpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "_MOUNTS_PATH", _write_mounts(tmp_path, [
        "/dev/sda1 / ext4 rw 0 0",
        r"x /mnt/my\040share nfs4 rw 0 0",
    ]))
    monkeypatch.setattr(rt.os.path, "realpath", lambda p: p)
    assert rt.fs_type("/mnt/my share/data") == "nfs4"


@pytest.mark.parametrize("fstype,expected", [
    ("ext4", False), ("xfs", False), ("tmpfs", False),
    ("nfs", True), ("nfs4", True), ("ceph", True), ("lustre", True),
    ("cifs", True), ("smb3", True), ("glusterfs", True), ("beegfs", True),
    ("fuse.ceph-fuse", True), ("fuse.sshfs", True), ("fuse.glusterfs", True),
    ("fuse", False), ("fuse.gocryptfs", False), ("fuse.mergerfs", False),
])
def test_is_network_fstype(fstype, expected):
    assert rt._is_network_fstype(fstype) is expected


def test_is_network_fs_force_override(monkeypatch, tmp_path):
    monkeypatch.setattr(rt, "_MOUNTS_PATH", _write_mounts(tmp_path, ["/dev/sda1 / ext4 rw 0 0"]))
    monkeypatch.setattr(rt.os.path, "realpath", lambda p: p)
    assert rt.is_network_fs("/x") is False
    monkeypatch.setenv("CASTLE_FORCE_NETWORK_FS", "1")
    assert rt.is_network_fs("/x") is True
    monkeypatch.setenv("CASTLE_FORCE_NETWORK_FS", "0")
    assert rt.is_network_fs("/x") is False


# ---------------------------------------------------------------------------
# usable CPU count
# ---------------------------------------------------------------------------

def test_usable_cpu_count_takes_min_of_signals(monkeypatch):
    monkeypatch.setattr(rt.os, "sched_getaffinity", lambda pid: set(range(40)), raising=False)
    monkeypatch.setattr(rt.os, "cpu_count", lambda: 40)
    monkeypatch.setattr(rt, "_cgroup_cpu_quota", lambda: 8)
    assert rt.usable_cpu_count() == 8


def test_usable_cpu_count_no_cgroup_uses_affinity(monkeypatch):
    monkeypatch.setattr(rt.os, "sched_getaffinity", lambda pid: set(range(12)), raising=False)
    monkeypatch.setattr(rt.os, "cpu_count", lambda: 20)
    monkeypatch.setattr(rt, "_cgroup_cpu_quota", lambda: None)
    assert rt.usable_cpu_count() == 12


def test_usable_cpu_count_override(monkeypatch):
    monkeypatch.setenv("CASTLE_USABLE_CPUS", "3")
    assert rt.usable_cpu_count() == 3


def test_cgroup_cpu_quota_v2(tmp_path, monkeypatch):
    (tmp_path / "cpu.max").write_text("150000 100000\n")
    monkeypatch.setattr(rt, "_CGROUP_BASE", str(tmp_path))
    assert rt._cgroup_cpu_quota() == 2  # ceil(1.5)


def test_cgroup_cpu_quota_v2_unlimited(tmp_path, monkeypatch):
    (tmp_path / "cpu.max").write_text("max 100000\n")
    monkeypatch.setattr(rt, "_CGROUP_BASE", str(tmp_path))
    assert rt._cgroup_cpu_quota() is None


def test_cgroup_cpu_quota_v1(tmp_path, monkeypatch):
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("400000\n")
    (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    monkeypatch.setattr(rt, "_CGROUP_BASE", str(tmp_path))
    assert rt._cgroup_cpu_quota() == 4


def test_cgroup_cpu_quota_v1_unlimited(tmp_path, monkeypatch):
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("-1\n")
    (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    monkeypatch.setattr(rt, "_CGROUP_BASE", str(tmp_path))
    assert rt._cgroup_cpu_quota() is None


# ---------------------------------------------------------------------------
# RAM
# ---------------------------------------------------------------------------

def test_total_ram_from_meminfo(tmp_path, monkeypatch):
    (tmp_path / "meminfo").write_text("MemTotal:       32768000 kB\nMemAvailable:    8000000 kB\n")
    monkeypatch.setattr(rt, "_MEMINFO_PATH", str(tmp_path / "meminfo"))
    monkeypatch.setattr(rt, "_cgroup_mem_max", lambda: None)
    assert rt.total_ram_bytes() == 32768000 * 1024


def test_total_ram_capped_by_cgroup(tmp_path, monkeypatch):
    (tmp_path / "meminfo").write_text("MemTotal:       1073741824 kB\n")
    monkeypatch.setattr(rt, "_MEMINFO_PATH", str(tmp_path / "meminfo"))
    monkeypatch.setattr(rt, "_cgroup_mem_max", lambda: 16 * rt._GiB)
    assert rt.total_ram_bytes() == 16 * rt._GiB


def test_total_ram_override(monkeypatch):
    monkeypatch.setenv("CASTLE_TOTAL_RAM_GB", "64")
    assert rt.total_ram_bytes() == 64 * rt._GiB


def test_available_ram_from_meminfo(tmp_path, monkeypatch):
    (tmp_path / "meminfo").write_text("MemTotal: 32768000 kB\nMemAvailable: 8000000 kB\n")
    monkeypatch.setattr(rt, "_MEMINFO_PATH", str(tmp_path / "meminfo"))
    monkeypatch.setattr(rt, "_cgroup_mem_max", lambda: None)
    assert rt.available_ram_bytes() == 8000000 * 1024


# ---------------------------------------------------------------------------
# scratch_dir
# ---------------------------------------------------------------------------

def test_scratch_dir_explicit_override(tmp_path, monkeypatch):
    target = tmp_path / "scratch"
    monkeypatch.setenv("CASTLE_SCRATCH_DIR", str(target))
    assert rt.scratch_dir() == str(target)
    assert target.is_dir()


def test_scratch_dir_prefers_local_tmp(monkeypatch, tmp_path):
    # /dev/shm not tmpfs (force unknown), gettempdir local → returns tmp
    monkeypatch.setattr(rt.tempfile, "gettempdir", lambda: "/tmp")
    monkeypatch.setattr(rt, "fs_type", lambda p: "tmpfs" if p == "/tmp" else "ext4")
    monkeypatch.setattr(rt, "is_network_fs", lambda p: False)
    monkeypatch.setattr(rt.os.path, "isdir", lambda p: p != "/dev/shm")
    assert rt.scratch_dir() == "/tmp"


def test_scratch_dir_uses_dev_shm_on_big_ram_box(monkeypatch):
    # Big-RAM cloud box: /dev/shm is the right spill target.
    monkeypatch.setattr(rt.os.path, "isdir", lambda p: p == "/dev/shm")
    monkeypatch.setattr(rt, "fs_type", lambda p: "tmpfs")
    monkeypatch.setattr(rt, "total_ram_bytes", lambda: 1000 * rt._GiB)
    monkeypatch.setattr(rt, "_free_bytes", lambda p: 500 * rt._GiB)
    assert rt.scratch_dir(min_free_bytes=10 * rt._GiB) == "/dev/shm"


def test_scratch_dir_skips_dev_shm_on_small_ram_box(monkeypatch):
    # 32GB dev box: spilling to /dev/shm would consume scarce RAM → use /tmp.
    monkeypatch.setattr(rt.os.path, "isdir", lambda p: True)
    monkeypatch.setattr(rt, "fs_type", lambda p: "tmpfs" if p == "/dev/shm" else "ext4")
    monkeypatch.setattr(rt, "total_ram_bytes", lambda: 32 * rt._GiB)
    monkeypatch.setattr(rt, "_free_bytes", lambda p: 16 * rt._GiB)
    monkeypatch.setattr(rt.tempfile, "gettempdir", lambda: "/tmp")
    monkeypatch.setattr(rt, "is_network_fs", lambda p: False)
    assert rt.scratch_dir(min_free_bytes=4 * rt._GiB) == "/tmp"


def test_scratch_dir_warns_when_only_network_fs(monkeypatch, caplog):
    monkeypatch.setattr(rt.os.path, "isdir", lambda p: False)  # no /dev/shm
    monkeypatch.setattr(rt.tempfile, "gettempdir", lambda: "/home/u/sharedfs/tmp")
    monkeypatch.setattr(rt, "is_network_fs", lambda p: True)
    monkeypatch.setattr(rt, "fs_type", lambda p: "ceph")
    with caplog.at_level("WARNING"):
        out = rt.scratch_dir(min_free_bytes=rt._GiB)
    assert out == "/home/u/sharedfs/tmp"
    assert any("network filesystem" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# RAM-aware latent budget (the core cloud-vs-dev lever)
# ---------------------------------------------------------------------------

def test_is_big_ram_box(monkeypatch):
    monkeypatch.setattr(rt, "total_ram_bytes", lambda: 1000 * rt._GiB)
    assert rt.is_big_ram_box() is True
    monkeypatch.setattr(rt, "total_ram_bytes", lambda: 32 * rt._GiB)
    assert rt.is_big_ram_box() is False
    monkeypatch.setattr(rt, "total_ram_bytes", lambda: None)
    assert rt.is_big_ram_box() is False


def test_latent_budget_small_ram_box_keeps_default(monkeypatch):
    # dev box: 32 GB → conservative ~2 GiB default, spills big buffers to disk.
    monkeypatch.delenv("CASTLE_MEMMAP_THRESHOLD_GB", raising=False)
    monkeypatch.setattr(rt, "total_ram_bytes", lambda: 32 * rt._GiB)
    monkeypatch.setattr(rt, "available_ram_bytes", lambda: 26 * rt._GiB)
    assert rt.latent_ram_budget_bytes() == 2 * rt._GiB


def test_latent_budget_big_ram_box_scales(monkeypatch):
    # cloud box: 1 TB → ~half of available RAM, so latents stay resident.
    monkeypatch.delenv("CASTLE_MEMMAP_THRESHOLD_GB", raising=False)
    monkeypatch.delenv("CASTLE_LATENT_RAM_FRACTION", raising=False)
    monkeypatch.setattr(rt, "total_ram_bytes", lambda: 1000 * rt._GiB)
    monkeypatch.setattr(rt, "available_ram_bytes", lambda: 900 * rt._GiB)
    assert rt.latent_ram_budget_bytes() == int(0.5 * 900 * rt._GiB)


def test_latent_budget_explicit_override_wins(monkeypatch):
    # Even on a big-RAM box, an explicit pin is honoured exactly (dev / tests).
    monkeypatch.setenv("CASTLE_MEMMAP_THRESHOLD_GB", "4")
    monkeypatch.setattr(rt, "total_ram_bytes", lambda: 1000 * rt._GiB)
    assert rt.latent_ram_budget_bytes() == 4 * (1024 ** 3)


def test_scratch_dir_fallback_when_tmp_too_small(monkeypatch):
    # local /tmp lacks room → fall back to the caller's output dir (dev safety).
    monkeypatch.setattr(rt.os.path, "isdir", lambda p: p != "/dev/shm")
    monkeypatch.setattr(rt.tempfile, "gettempdir", lambda: "/tmp")
    monkeypatch.setattr(rt, "is_network_fs", lambda p: False)
    monkeypatch.setattr(rt, "_free_bytes", lambda p: 1 * rt._GiB)  # tiny /tmp
    monkeypatch.setattr(rt, "total_ram_bytes", lambda: 32 * rt._GiB)
    out = rt.scratch_dir(min_free_bytes=10 * rt._GiB, fallback="/proj/latent")
    assert out == "/proj/latent"


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def test_summary_shape(monkeypatch):
    monkeypatch.setattr(rt, "is_network_fs", lambda p: True)
    monkeypatch.setattr(rt, "fs_type", lambda p: "ceph")
    monkeypatch.setattr(rt, "usable_cpu_count", lambda: 8)
    monkeypatch.setattr(rt, "total_ram_bytes", lambda: 1000 * rt._GiB)
    monkeypatch.setattr(rt, "available_ram_bytes", lambda: 900 * rt._GiB)
    monkeypatch.setattr(rt, "gpu_info", lambda: [{"index": 0, "name": "RTX 4090",
                                                  "total_bytes": 24 * rt._GiB, "free_bytes": 23 * rt._GiB}])
    monkeypatch.setattr(rt, "scratch_dir", lambda: "/dev/shm")
    s = rt.summary("/home/u/sharedfs/proj")
    assert s["network_fs_detected"] is True
    assert s["fs_type"] == "ceph"
    assert s["usable_cpus"] == 8
    assert s["total_ram_gb"] == 1000.0
    assert s["gpu_count"] == 1
    assert s["scratch_dir"] == "/dev/shm"
