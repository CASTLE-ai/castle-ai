"""
castle/core/batch.py
Batch processing of multiple CASTLE projects/videos.

Provides :class:`BatchConfig` (dataclass driven by an ``experiments.yaml``)
and :class:`BatchRunner` which runs the full pipeline across all configured
experiments and produces a summary report.

Usage::

    from castle.core.batch import BatchConfig, BatchRunner

    config = BatchConfig.from_yaml("experiments.yaml")
    runner = BatchRunner(config)
    results = runner.run(progress_callback=lambda p, m: print(f"{p:.0%} {m}"))
    print(runner.generate_summary(results))

YAML format::

    experiments:
      - name: "Control Group"
        project: "/data/control"
        videos: ["mouse1.mp4", "mouse2.mp4"]
        params:
          fc: 0.25
          n_clusters: 10
      - name: "Treatment Group"
        project: "/data/treatment"
        videos: ["mouse3.mp4", "mouse4.mp4"]
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BatchConfig
# ---------------------------------------------------------------------------


@dataclass
class BatchConfig:
    """Configuration for batch processing multiple projects.

    Attributes:
        projects:    List of project spec dicts, each containing:
                     ``name`` (str), ``project`` (str path),
                     ``videos`` (list[str], optional),
                     ``params`` (dict, optional).
        parallel:    Whether to run projects concurrently.
        max_workers: Thread pool size when ``parallel=True``.
    """

    projects: list[dict]
    parallel: bool = False
    max_workers: int = 2

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "BatchConfig":
        """Load :class:`BatchConfig` from an ``experiments.yaml`` file.

        Args:
            yaml_path: Path to the YAML experiments file.

        Returns:
            Populated :class:`BatchConfig` instance.

        Raises:
            FileNotFoundError: If *yaml_path* does not exist.
            KeyError:          If the YAML is missing the ``experiments`` key.
        """
        import yaml  # type: ignore[import-untyped]  # noqa: PLC0415

        yaml_path = os.path.abspath(yaml_path)
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"experiments file not found: {yaml_path}")

        with open(yaml_path) as fh:
            raw = yaml.safe_load(fh)

        if "experiments" not in raw:
            raise KeyError(
                f"YAML file '{yaml_path}' must contain a top-level 'experiments' key."
            )

        projects: list[dict] = []
        for exp in raw["experiments"]:
            projects.append(
                {
                    "name": exp.get("name", "unnamed"),
                    "project": exp["project"],
                    "videos": list(exp.get("videos") or []),
                    "params": dict(exp.get("params") or {}),
                }
            )

        parallel = bool(raw.get("parallel", False))
        max_workers = int(raw.get("max_workers", 2))

        return cls(projects=projects, parallel=parallel, max_workers=max_workers)


# ---------------------------------------------------------------------------
# BatchRunner
# ---------------------------------------------------------------------------


class BatchRunner:
    """Run the CASTLE pipeline across multiple projects/videos.

    Args:
        config: :class:`BatchConfig` describing all experiments.
    """

    def __init__(self, config: BatchConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> list[dict]:
        """Process all projects defined in :attr:`config`.

        For each project, the full CASTLE pipeline (tracking + extraction)
        is executed.  Projects are run sequentially by default; set
        ``config.parallel = True`` for concurrent execution.

        Args:
            progress_callback: Optional ``(fraction: float, message: str) → None``
                               called periodically with overall progress.

        Returns:
            List of per-project result dicts with keys:
            ``name``, ``project``, ``status``, ``tracking``,
            ``extraction``, ``elapsed_s``, ``error``.
        """
        projects = self.config.projects
        n = len(projects)
        results: list[dict] = []

        def _run_one(idx: int, spec: dict) -> dict:
            return self._process_project(idx, n, spec, progress_callback)

        # Parallel projects share one process-wide ModelRegistry singleton and
        # one default CUDA device: one project's cleanup unloads a model another
        # is still using (data race / freed-module inference), and several big
        # models on one GPU OOM. The GPU pipeline is not safe to run concurrently
        # this way, so on CUDA we force sequential execution + warn (CPU-only
        # parallelism is unaffected). See review theme C / robustness.
        use_parallel = self.config.parallel and n > 1
        if use_parallel:
            try:
                import torch
                if torch.cuda.is_available():
                    logger.warning(
                        "BatchRunner: parallel=True is unsafe with a shared CUDA "
                        "device (projects would unload each other's models / OOM); "
                        "running the %d projects sequentially instead.", n,
                    )
                    use_parallel = False
            except Exception:  # noqa: BLE001 — torch import/probe must not block the run
                pass

        if use_parallel:
            futures_map = {}
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
                for idx, spec in enumerate(projects):
                    fut = pool.submit(_run_one, idx, spec)
                    futures_map[fut] = spec["name"]

                for fut in as_completed(futures_map):
                    try:
                        results.append(fut.result())
                    except Exception as exc:  # noqa: BLE001
                        name = futures_map[fut]
                        logger.error("BatchRunner: project '%s' failed: %s", name, exc)
                        results.append(
                            {
                                "name": name,
                                "project": "",
                                "status": "error",
                                "error": str(exc),
                                "tracking": {},
                                "extraction": {},
                                "elapsed_s": 0,
                            }
                        )
        else:
            for idx, spec in enumerate(projects):
                results.append(_run_one(idx, spec))

        if progress_callback:
            progress_callback(1.0, "Batch complete")

        return results

    def generate_summary(self, results: list[dict]) -> str:
        """Generate a plain-text summary report across all project results.

        Args:
            results: List of dicts as returned by :meth:`run`.

        Returns:
            Formatted summary string suitable for terminal output.
        """
        lines: list[str] = [
            "=" * 60,
            "CASTLE Batch Processing Summary",
            "=" * 60,
            f"Total projects: {len(results)}",
            "",
        ]

        ok = [r for r in results if r.get("status") == "done"]
        err = [r for r in results if r.get("status") == "error"]
        skip = [r for r in results if r.get("status") == "skipped"]

        lines.append(f"  Completed : {len(ok)}")
        lines.append(f"  Errors    : {len(err)}")
        lines.append(f"  Skipped   : {len(skip)}")
        lines.append("")

        for r in results:
            status_sym = {"done": "✓", "error": "✗", "skipped": "○"}.get(r.get("status", ""), "?")
            elapsed = r.get("elapsed_s", 0)
            lines.append(f"  [{status_sym}] {r.get('name', '?')}  ({elapsed:.1f}s)")

            if r.get("error"):
                lines.append(f"       Error: {r['error']}")

            tracking = r.get("tracking", {})
            extraction = r.get("extraction", {})
            if tracking:
                ok_t = sum(1 for v in tracking.values() if "skip" not in str(v).lower())
                lines.append(f"       Tracking  : {ok_t}/{len(tracking)} videos")
            if extraction:
                ok_e = sum(1 for v in extraction.values() if v)
                lines.append(f"       Extraction: {ok_e}/{len(extraction)} videos")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_project(
        self,
        idx: int,
        total: int,
        spec: dict,
        progress_callback: Optional[Callable[[float, str], None]],
    ) -> dict:
        """Run the pipeline for a single project spec.

        Args:
            idx:               Zero-based index of this project in the list.
            total:             Total number of projects.
            spec:              Project spec dict (name, project, videos, params).
            progress_callback: Optional overall-progress callback.

        Returns:
            Result dict for this project.
        """
        from castle.service.pipeline import Pipeline, PipelineConfig  # noqa: PLC0415

        name = spec.get("name", f"project_{idx}")
        project_path = spec.get("project", "")
        videos = spec.get("videos") or []
        params = spec.get("params") or {}

        logger.info("BatchRunner [%d/%d]: starting '%s'", idx + 1, total, name)

        if progress_callback:
            base_frac = idx / total
            progress_callback(base_frac, f"Starting {name} ({idx + 1}/{total})")

        # Resolve storage_path + project_name from project_path
        storage_path = os.path.dirname(os.path.abspath(project_path))
        project_name = os.path.basename(os.path.abspath(project_path))

        # Build PipelineConfig, mapping known params
        cfg = PipelineConfig(
            storage_path=storage_path,
            project_name=project_name,
            videos=list(videos),
        )
        # Apply extra params from YAML (only known PipelineConfig fields)
        for k, v in params.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        def _sub_cb(frac: float, msg: str) -> None:
            if progress_callback:
                overall = (idx + frac) / total
                progress_callback(overall, f"[{name}] {msg}")

        t0 = time.monotonic()
        try:
            pipeline = Pipeline(cfg, progress_callback=_sub_cb)
            pipe_result = pipeline.run()
            elapsed = time.monotonic() - t0
            logger.info("BatchRunner [%d/%d]: '%s' done in %.1fs", idx + 1, total, name, elapsed)
            return {
                "name": name,
                "project": project_path,
                "status": "done",
                "tracking": pipe_result.get("tracking", {}),
                "extraction": pipe_result.get("extraction", {}),
                "memory_stats": pipe_result.get("memory_stats", {}),
                "elapsed_s": elapsed,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - t0
            logger.error(
                "BatchRunner [%d/%d]: '%s' FAILED after %.1fs: %s",
                idx + 1,
                total,
                name,
                elapsed,
                exc,
            )
            return {
                "name": name,
                "project": project_path,
                "status": "error",
                "tracking": {},
                "extraction": {},
                "elapsed_s": elapsed,
                "error": str(exc),
            }
