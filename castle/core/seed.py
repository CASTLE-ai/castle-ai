"""Global seed management for CASTLE (P0-B / REPRO-01).

This module controls every stochastic component CASTLE uses EXCEPT UMAP's
optimization seed, which is intentionally exposed via its own re-roll /
lock UX (see ``castle.utils.latent_explorer.LocalLatent.build_embedding``
and ``castle.utils.myumap.UMAP``).

Why UMAP is excluded:
    UMAP's seed has a meaningful UX role — researchers may want to re-roll
    layouts during exploration, then lock a specific seed once they like
    one. The master seed here governs everything else (DataLoader workers,
    PyTorch CUDA, sklearn KMeans, Python random) so that the *rest* of the
    pipeline is deterministic regardless of which UMAP layout is chosen.

Typical use::

    from castle.core.seed import set_global_seed
    set_global_seed(42)                         # fast, fast determinism
    set_global_seed(42, strict_cuda=True)       # bit-identical CUDA (slower)
"""

from __future__ import annotations

import os
import random as _random
from typing import Optional

import numpy as np


DEFAULT_MASTER_SEED: int = 42


def set_global_seed(seed: int = DEFAULT_MASTER_SEED, *, strict_cuda: bool = False) -> int:
    """Seed every stochastic component CASTLE uses except UMAP.

    Args:
        seed: Master seed applied to Python ``random``, NumPy, PyTorch (CPU
            and all CUDA devices), and the ``PYTHONHASHSEED`` env var.

            Note: UMAP has its own ``random_state`` parameter
            (see :class:`castle.utils.myumap.UMAP` and
            :meth:`castle.utils.latent_explorer.LocalLatent.build_embedding`)
            so that the "re-roll layout" UX can stay meaningful. Calling
            ``set_global_seed`` does NOT lock UMAP.
        strict_cuda: If ``True``, also disable cuDNN benchmark, enable
            deterministic algorithms, and set ``CUBLAS_WORKSPACE_CONFIG``.
            Costs roughly 10% throughput on Ampere; use for paper-grade runs.

    Returns:
        The seed that was set (echoed back for logging convenience).

    Example:
        >>> from castle.core.seed import set_global_seed
        >>> seed = set_global_seed(42)
        >>> seed
        42
    """
    _random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if strict_cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except (AttributeError, RuntimeError):
                pass
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` for reproducible multi-worker data loading.

    Each worker derives its seed from PyTorch's per-worker initial seed (which
    is itself derived from the parent ``torch.Generator`` passed to
    ``DataLoader(generator=...)``). This guarantees identical multi-worker
    shuffles / augmentation across two runs that share the same master seed.

    Args:
        worker_id: 0-indexed DataLoader worker id (unused but required by the
            ``worker_init_fn`` signature).

    Example:
        >>> import torch
        >>> from torch.utils.data import DataLoader
        >>> from castle.core.seed import seed_worker
        >>> g = torch.Generator()
        >>> g.manual_seed(42)
        >>> # loader = DataLoader(ds, worker_init_fn=seed_worker, generator=g)
    """
    del worker_id
    try:
        import torch
        worker_seed = torch.initial_seed() % 2 ** 32
    except ImportError:
        return
    np.random.seed(worker_seed)
    _random.seed(worker_seed)


def make_torch_generator(seed: Optional[int] = None):
    """Build a seeded ``torch.Generator`` for DataLoader reproducibility.

    Args:
        seed: Explicit seed. If ``None``, derive a seed from the current
            (master-seeded) torch RNG via ``torch.randint`` so that the
            DataLoader workers' RNG follows the master seed transitively.

    Returns:
        A ``torch.Generator`` instance with ``manual_seed`` applied,
        or ``None`` if torch is not available.
    """
    try:
        import torch
    except ImportError:
        return None
    if seed is None:
        seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g
