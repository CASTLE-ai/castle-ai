"""Default adapter implementations of the clustering protocols (ARCH-02).

The :mod:`castle.core.clustering_protocols` module defines the structural
type seam — ``DimensionReducer`` and ``Clusterer``. This module is the
**concrete other side** of that seam: every reducer / clusterer CASTLE
currently uses (UMAP from ``umap-learn`` / ``cuml`` / ``myumap``, DBSCAN
from ``sklearn`` / ``cuml``, optionally HDBSCAN) lives here, packaged
as a class that satisfies the Protocol.

Why this matters:

* Adding a new reducer / clusterer (HDBSCAN, GMM, spectral …) is now a
  matter of writing one class that conforms to the relevant Protocol.
  Nothing in :class:`castle.utils.latent_explorer.LocalLatent` needs to
  change — pass a factory or instance into ``build_embedding`` /
  ``build_cluster`` and the existing pipeline picks it up.
* The device-aware UMAP / DBSCAN class resolution (umap-learn vs cuml
  vs the in-repo ``myumap`` fallback) is encapsulated in a single
  helper, not duplicated across call sites.

Backwards compatibility:

The default behaviour of ``LocalLatent.build_embedding`` /
``build_cluster`` calls into ``UMAPReducer`` / ``DBSCANClusterer`` here.
The class resolution table inside the adapters reproduces the exact
priority order the previous inline code used, so existing
reproducibility tests stay bit-identical.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import os
from typing import Any, Optional, cast

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "UMAPReducer",
    "DBSCANClusterer",
    "HDBSCANClusterer",
    "resolve_umap_class",
    "resolve_dbscan_class",
    "build_default_clusterer",
    "nn_descent_graph_degree",
    "umap_peak_bytes",
    "umap_host_bytes",
    "target_cuda_free_bytes",
    "free_cuda_memory_pools",
]


# ---------------------------------------------------------------------------
# Backend class resolution
# ---------------------------------------------------------------------------


_GPU_UMAP_CHECKED = False
_GPU_UMAP_REASON: Optional[str] = None


def gpu_umap_unavailable_reason() -> Optional[str]:
    """Cached probe: ``None`` if cuML GPU UMAP imports cleanly, else a short reason.

    cuML can fail to load for many reasons — no GPU, not installed, or (commonly
    with pip-installed RAPIDS) a CUDA library not on the linker path, e.g. a cu12
    wheel's ``libcublas.so.12`` shadowed by an older system CUDA on
    ``LD_LIBRARY_PATH``. The first probe is cached so the UI can *tell* the user
    UMAP will run on CPU instead of silently falling back. See INSTALLATION.md.
    """
    global _GPU_UMAP_CHECKED, _GPU_UMAP_REASON
    if not _GPU_UMAP_CHECKED:
        try:
            from cuml.manifold import UMAP  # noqa: F401
            _GPU_UMAP_REASON = None
        except Exception as exc:  # noqa: BLE001 — any import/load failure -> CPU
            first = str(exc).splitlines()[0] if str(exc) else ""
            _GPU_UMAP_REASON = f"{type(exc).__name__}: {first[:200]}"
        _GPU_UMAP_CHECKED = True
    return _GPU_UMAP_REASON


def resolve_umap_class(device: str) -> Any:
    """Pick the UMAP class to use for ``device``.

    Mirrors the priority order the legacy inline code in
    :meth:`LocalLatent.build_embedding` used:

    * CPU / MPS → :class:`umap.UMAP`.
    * CUDA → ``cuml.manifold.UMAP`` → ``castle.utils.myumap.UMAP`` →
      ``umap.UMAP`` (each falling back on ANY import/load error).

    A CUDA→CPU fallback is logged at WARNING: CPU ``umap-learn`` is dramatically
    slower on large datasets (minutes → hours at ~1M points), so a silent
    fallback reads as "UMAP hung" rather than "GPU unavailable". The UI surfaces
    the same via :func:`gpu_umap_unavailable_reason`.

    Args:
        device: ``'cpu'``, ``'mps'``, or ``'cuda'`` / ``'cuda:N'``.

    Returns:
        The UMAP class object (not an instance).

    Raises:
        ValueError: Unsupported device string.
    """
    if device in ('cpu', 'mps'):
        from umap import UMAP
        return UMAP
    if 'cuda' in device:
        reason = gpu_umap_unavailable_reason()
        if reason is None:
            from cuml.manifold import UMAP  # cached in sys.modules — cheap
            return UMAP
        try:
            from castle.utils.myumap import UMAP
            return UMAP
        except Exception:  # noqa: BLE001 — myumap also needs cuML
            pass
        logger.warning(
            "GPU UMAP (cuML) unavailable (%s); falling back to CPU umap-learn — "
            "this is MUCH slower for large datasets. See INSTALLATION.md "
            "(GPU UMAP / cuML).", reason,
        )
        from umap import UMAP
        return UMAP
    raise ValueError(
        f"Unsupported device {device!r}; expected 'cpu', 'mps', or 'cuda'."
    )


def resolve_dbscan_class(device: str) -> Any:
    """Pick the DBSCAN class for ``device``.

    CUDA path prefers ``cuml.cluster.DBSCAN`` and falls back to
    ``sklearn.cluster.DBSCAN`` if cuml isn't available — matches the
    legacy inline behaviour.

    Args:
        device: ``'cpu'``, ``'mps'``, or ``'cuda'``.

    Returns:
        The DBSCAN class object.
    """
    if device in ('cpu', 'mps'):
        from sklearn.cluster import DBSCAN
        return DBSCAN
    if 'cuda' in device:
        try:
            from cuml.cluster import DBSCAN
            return DBSCAN
        except ImportError:
            from sklearn.cluster import DBSCAN
            return DBSCAN
    raise ValueError(
        f"Unsupported device {device!r}; expected 'cpu', 'mps', or 'cuda'."
    )


# ---------------------------------------------------------------------------
# DimensionReducer adapters
# ---------------------------------------------------------------------------


# cuML's build_algo='auto' reverts to exact brute_force_knn O(N^2) whenever a
# random_state is set — and CASTLE always sets one — so the approximate kNN
# (nn_descent) only runs if requested explicitly. We gate that request to a VERY
# high row count, because nn_descent's approximate graph has LOW RECALL on real
# high-dimensional behavior latents (continuous pose manifold, near-duplicate
# frames, heavy-tailed neighbour distances): it builds a fragmented k-NN graph
# and UMAP then lays the points out as a scattered cloud with no clusters —
# observed in the field at >150k, while exact kNN below that produced clean
# clusters (the only size-dependent branch in the path, so unambiguous). Exact
# brute kNN is fast enough across the practical range (~22s at 200k x 1362 dims,
# measured) and is what well-behaved runs already used, so prefer it; nn_descent
# only engages past here, where brute is genuinely infeasible (and the user can
# subsample instead). A build_algo in the UMAP config still overrides this.
_NN_DESCENT_MIN_ROWS = 1_000_000


def _resolve_cuda_index(device: Optional[str]) -> Optional[int]:
    """The CUDA device index a cuML op for *device* will run on, or ``None``.

    Honours an explicit ``cuda:N`` in *device* or in ``CASTLE_GPU_DEVICE``,
    else the idlest GPU (most free VRAM). ``None`` when *device* isn't CUDA or no
    GPU is selectable. The VRAM guard reads the SAME index so it checks the card
    the fit actually uses.
    """
    if not (isinstance(device, str) and device.startswith("cuda")):
        return None
    for src in (device, os.environ.get("CASTLE_GPU_DEVICE", "").strip().lower()):
        if src.startswith("cuda:"):
            try:
                return int(src.split(":", 1)[1])
            except ValueError:
                pass
    try:
        from castle.core import runtime_env
        return runtime_env.idlest_gpu()
    except Exception:  # noqa: BLE001
        return None


def target_cuda_free_bytes(device: str = "cuda") -> Optional[int]:
    """Free VRAM (bytes) of the GPU a cuML op for *device* will run on, or None."""
    idx = _resolve_cuda_index(device)
    if idx is None:
        return None
    try:
        from castle.core import runtime_env
        for d in runtime_env.gpu_info():
            if int(cast(int, d["index"])) == idx:
                return int(cast(int, d["free_bytes"]))
    except Exception:  # noqa: BLE001
        return None
    return None


def nn_descent_graph_degree(n_neighbors: int) -> int:
    """nn_descent intermediate-graph degree: recall headroom over n_neighbors,
    bounded (+128) so large n_neighbors don't explode VRAM. Single source of
    truth shared by the build_kwds AND the memory estimator."""
    nn = int(n_neighbors)
    return min(2 * nn, nn + 128)


def umap_peak_bytes(n: int, d: int, n_neighbors: int) -> float:
    """Rough peak memory (bytes) of one cuML UMAP fit on *n* points x *d* dims.

    Models the dominant buffers (whichever device): input ``n*d*4``; the
    nn_descent kNN graph + equal-degree intermediate graph (~24 bytes/point/
    degree); the fuzzy simplicial set (~32 bytes/point/n_neighbors). This is a
    RAW estimate — no safety multiplier. The single safety margin lives in the
    guard's free-VRAM fraction (default 0.85 → ~1.18x headroom); stacking a 1.5x
    factor here too made the guard ~2x over-conservative and refused fits that
    comfortably fit. Drives the pre-flight guard.
    """
    gd = nn_descent_graph_degree(n_neighbors)
    return float(n) * (4.0 * int(d) + 24.0 * gd + 32.0 * int(n_neighbors))


def umap_host_bytes(
    n_total: int,
    n_fit: int,
    d: int,
    n_components: int = 2,
    *,
    full_copy: bool = False,
) -> float:
    """*Marginal* host-RAM (bytes) a UMAP + label-propagation run allocates
    **beyond the already-resident input matrix**.

    The ``n_total x d`` latent matrix is materialised by ``select()`` *before*
    this guard runs, so it is already counted as used RAM (excluded from
    MemAvailable) — re-counting it would double-count and refuse fits that
    actually run. Model only what the step newly allocates in host RAM:

    * ``full_copy`` — the float32/contiguity conversion duplicates the whole
      ``n_total x d`` matrix. Only happens when the selected latents aren't
      already float32 + C-contiguous (``select()`` usually yields both, so this
      is normally False and contributes nothing).
    * the subsample draws an ``n_fit x d`` copy (only when ``n_fit < n_total``).
    * the length-``n_total`` embedding output (``n_total x n_components``).

    On the GPU path the UMAP buffers themselves live on the *device* (covered by
    the VRAM guard via :func:`umap_peak_bytes`); on the CPU path those buffers
    dominate host RAM and are covered by :func:`umap_peak_bytes` against RAM
    instead. RAW estimate; the guard's free-RAM fraction is the only margin.
    """
    host = 0.0
    if full_copy:
        host += float(n_total) * 4.0 * int(d)         # f32/contiguity conversion
    if int(n_fit) < int(n_total):
        host += float(n_fit) * 4.0 * int(d)           # subsample draw
    host += float(n_total) * 4.0 * int(n_components)  # embedding output
    return host


def _cuda_device_ctx(device: Optional[str]) -> Any:
    """Pin a single-GPU cuML / cupy op to the idlest CUDA device.

    cuML (and the in-repo :mod:`castle.utils.myumap`) choose the *physical* GPU
    from the active **cupy** current-device — not from any torch device, and not
    from the ``'cuda:N'`` string (that string only selects the backend *class*).
    So wrap the fit/fit_predict in ``cupy.cuda.Device(idlest)``. No-op when
    *device* isn't CUDA, cupy is unavailable, or no GPU is selectable. Kept tight
    around the single call so the per-thread cupy current device cannot leak.
    """
    idx = _resolve_cuda_index(device)
    if idx is None:
        return contextlib.nullcontext()
    try:
        import cupy  # noqa: PLC0415
        return cupy.cuda.Device(idx)
    except Exception:  # noqa: BLE001 — cupy absent (CPU-only) → no-op
        return contextlib.nullcontext()


def free_cuda_memory_pools() -> None:
    """Return cached device VRAM to the CUDA driver after a GPU op finishes.

    cuML/cupy don't hand freed device blocks straight back to the driver: cupy
    caches them in its default memory pool, and cuML objects only release their
    buffers once garbage-collected. So after a UMAP fit / kNN propagation
    completes, ``nvidia-smi`` — and our own pre-flight VRAM guard, which reads
    driver-level free via :func:`target_cuda_free_bytes` /
    ``torch.cuda.mem_get_info`` — keep counting that VRAM as in use, which can
    falsely refuse the *next* run. Force a GC pass so dropped cuML objects are
    finalised, then drain cupy's device + pinned pools. No-op when cupy is
    absent (CPU-only path).
    """
    import gc  # noqa: PLC0415
    gc.collect()
    try:
        import cupy  # noqa: PLC0415
        # MemoryPool.free_all_blocks() drains only the CURRENT device's free
        # list, so iterate every device — the op may have run on the idlest GPU
        # (not device 0), and the caller's current device is unspecified here.
        pool = cupy.get_default_memory_pool()
        for d in range(cupy.cuda.runtime.getDeviceCount()):
            with cupy.cuda.Device(d):
                pool.free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()  # host pinned, global
    except Exception:  # noqa: BLE001 — cupy absent or no device → nothing to drain
        pass


class UMAPReducer:
    """:class:`DimensionReducer` adapter wrapping the device-appropriate UMAP.

    Stores the per-stage config dict at construction (minus any
    ``random_state`` entry — that goes through ``fit_transform``'s
    explicit kwarg so :class:`castle.utils.latent_explorer.LocalLatent`
    can manage per-stage seed resolution).
    """

    def __init__(self, cfg: dict, device: str = 'cpu'):
        """Build a reducer for a single UMAP stage.

        Args:
            cfg: UMAP config dict (``n_neighbors``, ``min_dist``, etc.).
                Any ``random_state`` key is dropped — pass it through
                ``fit_transform`` instead.
            device: Compute device. Resolves which UMAP class to use.
        """
        # Drop keys that are not UMAP constructor kwargs: ``random_state`` is
        # threaded through ``fit_transform``; ``standardize`` is consumed by
        # ``LocalLatent.build_embedding`` (applied to the raw features once).
        self.cfg = {k: v for k, v in cfg.items() if k not in ('random_state', 'standardize')}
        # CPU path: auto-inject n_jobs so pynndescent (the k-NN builder) can
        # use multiple cores. umap-learn's SGD stays single-threaded (numba
        # loop), so the seed still controls the optimisation deterministically.
        # Reserve 2 cores for the OS / GUI so the machine stays responsive.
        # On a 2-core machine this yields 1 (floor at 1); on 4-core → 2;
        # on 16-core → 14. Users can override by setting 'n_jobs' in cfg.
        if device == 'cpu' and 'n_jobs' not in self.cfg:
            import os
            _cpu = os.cpu_count() or 1
            self.cfg['n_jobs'] = max(1, _cpu - 2)
        self.device = device
        self._umap_cls = resolve_umap_class(device)
        # cuML estimator? (vs umap-learn / in-repo myumap) — gates cuML-only
        # kwargs like build_algo and the verbose log-level.
        self._is_cuml = getattr(self._umap_cls, '__module__', '').startswith('cuml')
        # Does the resolved UMAP accept build_algo? cuML >= 25.08 yes; umap-learn
        # and the in-repo myumap do not (passing it there would TypeError).
        try:
            self._supports_build_algo = (
                'build_algo' in inspect.signature(self._umap_cls).parameters
            )
        except (ValueError, TypeError):  # signature unavailable on some C types
            self._supports_build_algo = False

    def fit_transform(
        self,
        X: np.ndarray,  # [N, F]
        *,
        random_state: int,
    ) -> np.ndarray:  # [N, D]
        """Run UMAP on ``X`` with the stored config + given seed.

        Args:
            X: Input features, shape ``(N, F)``.
            random_state: Seed for UMAP's stochastic optimisation.

        Returns:
            ``(N, D)`` embedding, where ``D`` comes from
            ``cfg['n_components']`` (default 2).
        """
        full_cfg = {**self.cfg, 'random_state': int(random_state)}
        # Large-N cuML GPU path: request approximate kNN (nn_descent) explicitly.
        # cuML's 'auto' reverts to brute-force O(N^2) kNN when random_state is set
        # (CASTLE always sets it), and that kNN is the dominant cost at ~1M points
        # (measured ~38s of ~62s). nn_descent is non-deterministic — consistent
        # with the GPU backend already being non-deterministic. A user-supplied
        # build_algo, small N, or a non-cuML class all keep the exact default.
        if (self._supports_build_algo and self.device.startswith('cuda')
                and 'build_algo' not in full_cfg
                and X.shape[0] > _NN_DESCENT_MIN_ROWS):
            full_cfg['build_algo'] = 'nn_descent'
            # nn_descent builds its own intermediate kNN graph of degree
            # nnd_graph_degree, which MUST be >= n_neighbors. Left unset cuML
            # clamps it to exactly n_neighbors (low recall -> fragmented layout),
            # so add recall headroom — but BOUNDED: the graph costs ~N*degree*8
            # bytes of VRAM, so a flat 2x explodes for large n_neighbors (e.g.
            # n_neighbors=1000 -> degree 2000 -> multi-GB CUDA OOM). Cap the
            # headroom at +128. nnd_intermediate_graph_degree must be >= graph
            # degree; set it equal (the minimum) to save VRAM and silence cuML's
            # auto-bump warning. User-supplied build_kwds wins.
            if 'build_kwds' not in full_cfg:
                deg = nn_descent_graph_degree(int(full_cfg.get('n_neighbors', 15)))
                full_cfg['build_kwds'] = {
                    'nnd_graph_degree': deg,
                    'nnd_intermediate_graph_degree': deg,
                }
        # Silence cuML's per-fit "[CUML] [info] build_algo set to brute_force_knn
        # because random_state is given" line — CASTLE always sets a seed, so it
        # printed every stage. Only the estimator-level verbose suppresses it
        # (the global logger set_level does not). cuML only; umap-learn/myumap
        # take a bool verbose, so leave theirs untouched.
        if self._is_cuml and 'verbose' not in full_cfg:
            try:
                from cuml.internals.logger import level_enum  # noqa: PLC0415
                full_cfg['verbose'] = level_enum.warn
            except Exception:  # noqa: BLE001 — cuML internals moved; not fatal
                pass

        # cuML/myumap run on the idlest GPU (cupy current-device); no-op on CPU.
        # NB: do NOT drain the cupy pool here — this runs once PER STAGE, and
        # draining between stages forces cuML to re-cudaMalloc from the driver
        # (slow, device-synchronising). The pool is drained once at the end of
        # the whole run in LocalLatent.build_embedding instead.
        with _cuda_device_ctx(self.device):
            return np.asarray(self._umap_cls(**full_cfg).fit_transform(X))


# ---------------------------------------------------------------------------
# Clusterer adapters
# ---------------------------------------------------------------------------


class DBSCANClusterer:
    """:class:`Clusterer` adapter wrapping the device-appropriate DBSCAN.

    DBSCAN itself is deterministic — the ``random_state`` parameter in
    :meth:`fit_predict` is accepted for API uniformity and ignored.
    """

    def __init__(self, *, eps: float = 1.0, device: str = 'cpu', **kwargs: Any):
        """Build a DBSCAN clusterer.

        Args:
            eps: Neighbourhood radius. See ``sklearn.cluster.DBSCAN``.
            device: Compute device.
            **kwargs: Forwarded to the underlying DBSCAN constructor
                (e.g. ``min_samples``, ``metric``).
        """
        self.eps = float(eps)
        self.device = device
        self.kwargs = dict(kwargs)
        self._dbscan_cls = resolve_dbscan_class(device)

    def fit_predict(
        self,
        X: np.ndarray,  # [N, D]
        *,
        random_state: int = 0,
    ) -> np.ndarray:  # [N] int, -1 = noise
        """Assign cluster ids to each row of ``X``.

        Args:
            X: ``(N, D)`` input — typically a UMAP embedding.
            random_state: Ignored; DBSCAN is deterministic.

        Returns:
            ``(N,)`` integer labels with ``-1`` denoting noise.
        """
        del random_state  # DBSCAN is deterministic — accepted for protocol parity
        # cuML DBSCAN runs on the idlest GPU (cupy current-device); no-op on CPU.
        with _cuda_device_ctx(self.device):
            cl = self._dbscan_cls(eps=self.eps, **self.kwargs)
            return np.asarray(cl.fit_predict(X)).astype(int)


class HDBSCANClusterer:
    """:class:`Clusterer` adapter wrapping :mod:`hdbscan`.

    Optional adapter — only constructable if the ``hdbscan`` package is
    installed. Importing the class without ``hdbscan`` raises
    :class:`ImportError` at ``__init__`` time so the
    :class:`Clusterer` protocol check on the *class itself* still works.
    """

    def __init__(
        self,
        *,
        min_cluster_size: int = 10,
        min_samples: Optional[int] = None,
        **kwargs: Any,
    ):
        """Build an HDBSCAN clusterer.

        Args:
            min_cluster_size: Minimum cluster size; see
                :class:`hdbscan.HDBSCAN`.
            min_samples: HDBSCAN ``min_samples`` (defaults to ``min_cluster_size``
                when ``None``).
            **kwargs: Forwarded to ``hdbscan.HDBSCAN`` constructor.

        Raises:
            ImportError: ``hdbscan`` is not installed.
        """
        try:
            import hdbscan  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "HDBSCANClusterer requires the 'hdbscan' package. "
                "Install with: pip install hdbscan"
            ) from exc
        self.min_cluster_size = int(min_cluster_size)
        self.min_samples = min_samples
        self.kwargs = dict(kwargs)

    def fit_predict(
        self,
        X: np.ndarray,
        *,
        random_state: int = 0,
    ) -> np.ndarray:
        """Run HDBSCAN. ``random_state`` is accepted for protocol parity.

        HDBSCAN itself doesn't use a random_state — its tree-building
        algorithm is deterministic. We accept the keyword for uniform
        :class:`Clusterer` signature.
        """
        del random_state
        import hdbscan

        cl = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            **self.kwargs,
        )
        return np.asarray(cl.fit_predict(X)).astype(int)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def build_default_clusterer(method: str, configs: dict, device: str = 'cpu') -> Any:
    """Build the default :class:`Clusterer` for a legacy ``method`` string.

    Bridges the old ``LocalLatent.build_cluster(method='dbscan', configs=...)``
    API into the new Protocol-based world. Currently DBSCAN is the only
    legacy method; pass a :class:`HDBSCANClusterer` instance directly via
    ``build_cluster(clusterer=...)`` for anything else.

    Args:
        method: Legacy method name (``'dbscan'`` for now).
        configs: Method-specific config dict.
        device: Compute device.

    Returns:
        A :class:`Clusterer` instance.

    Raises:
        ValueError: Unknown method name.
    """
    if method == 'dbscan':
        eps = configs.get('eps', 1.0)
        extra = {k: v for k, v in configs.items() if k != 'eps'}
        return DBSCANClusterer(eps=eps, device=device, **extra)
    raise ValueError(
        f"Unknown clusterer method {method!r}. Pass a Clusterer instance "
        f"via `clusterer=` for non-DBSCAN backends."
    )
