"""
castle/core/project_data.py
Unified project data structure.

All project paths are computed from a single root, eliminating scattered
``os.path.join(storage_path, project_name, …)`` concatenations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Video file extensions recognised as source videos.
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


@dataclass
class VideoInfo:
    """Lightweight metadata container for a single source video.

    Attributes:
        name:     Filename (basename), e.g. ``"animal01.mp4"``.
        path:     Absolute path to the video file.
        fps:      Frames per second (0.0 if unknown).
        width:    Frame width in pixels (0 if unknown).
        height:   Frame height in pixels (0 if unknown).
        n_frames: Total frame count (0 if unknown).
    """

    name: str
    path: Path
    fps: float = 0.0
    width: int = 0
    height: int = 0
    n_frames: int = 0

    def __post_init__(self) -> None:
        # Normalise path to a Path object regardless of what was passed in.
        object.__setattr__(self, "path", Path(self.path))


@dataclass
class ProjectData:
    """Unified project data structure.

    All project paths are computed from *root*.  No more manual path joining.

    Attributes:
        root: Absolute path to the project root directory
              (e.g. ``Path("/data/projects/my_project")``).
        name: Human-readable project name (usually the directory basename).

    Example::

        pd = ProjectData.from_path("/data/projects/my_project")
        mask = pd.mask_h5_path("video01.mp4")
        pd.ensure_dirs()
        videos = pd.list_videos()
    """

    root: Path
    name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    # ------------------------------------------------------------------
    # Standard directory properties
    # ------------------------------------------------------------------

    @property
    def sources_dir(self) -> Path:
        """Directory that holds raw source video files."""
        return self.root / "sources"

    @property
    def track_dir(self) -> Path:
        """Root directory for per-video tracking outputs."""
        return self.root / "track"

    @property
    def latent_dir(self) -> Path:
        """Root directory for per-model latent feature files."""
        return self.root / "latent"

    @property
    def cluster_dir(self) -> Path:
        """Directory for clustering outputs (id.csv, time_series_*.csv, …)."""
        return self.root / "cluster"

    @property
    def preprocessed_dir(self) -> Path:
        """Directory for pre-processed data (e.g. stabilised frames)."""
        return self.root / "preprocessed"

    @property
    def config_path(self) -> Path:
        """Path to the project configuration file."""
        return self.root / "config.json"

    # ------------------------------------------------------------------
    # Per-video path helpers
    # ------------------------------------------------------------------

    def video_track_dir(self, video_name: str) -> Path:
        """Return the tracking sub-directory for a specific video.

        Args:
            video_name: Basename of the video file (e.g. ``"session01.mp4"``).

        Returns:
            ``<root>/track/<video_name>``
        """
        return self.track_dir / video_name

    def mask_h5_path(self, video_name: str) -> Path:
        """Return the mask HDF5 file path for a specific video.

        Args:
            video_name: Basename of the video file.

        Returns:
            ``<root>/track/<video_name>/mask_list.h5``
        """
        return self.track_dir / video_name / "mask_list.h5"

    def latent_model_dir(self, model_name: str) -> Path:
        """Return the latent sub-directory for a specific model.

        Args:
            model_name: Model identifier string.

        Returns:
            ``<root>/latent/<model_name>``
        """
        return self.latent_dir / model_name

    def cluster_session_dir(self, session_id: str) -> Path:
        """Return the cluster session sub-directory.

        Args:
            session_id: Session identifier string.

        Returns:
            ``<root>/cluster/sessions/<session_id>``
        """
        return self.cluster_dir / "sessions" / session_id

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_path(cls, project_path: str | Path) -> "ProjectData":
        """Load :class:`ProjectData` from an existing project directory.

        The project must contain a ``config.json`` file.

        Args:
            project_path: Path to the project root directory.

        Returns:
            A :class:`ProjectData` instance.

        Raises:
            FileNotFoundError: If the directory or ``config.json`` is missing.
        """
        root = Path(project_path).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Project directory not found: {root}")
        config_path = root / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Project config not found: {config_path}")
        return cls(root=root, name=root.name)

    @classmethod
    def from_storage(
        cls,
        storage_path: str | Path,
        project_name: str,
    ) -> "ProjectData":
        """Convenience constructor using the legacy ``(storage_path, project_name)`` pair.

        Args:
            storage_path: Root storage directory.
            project_name: Project directory name inside *storage_path*.

        Returns:
            A :class:`ProjectData` instance.
        """
        return cls.from_path(Path(storage_path) / project_name)

    # ------------------------------------------------------------------
    # Instance methods
    # ------------------------------------------------------------------

    def list_videos(self) -> list[VideoInfo]:
        """List all recognised video files in ``sources/``.

        Only files with extensions in :data:`_VIDEO_EXTENSIONS` are included.
        Metadata (fps, width, height, n_frames) is filled lazily via
        :mod:`castle.utils.video_io` when available; defaults to 0 otherwise.

        Returns:
            Sorted list of :class:`VideoInfo` objects (by filename).
        """
        if not self.sources_dir.is_dir():
            return []

        results: list[VideoInfo] = []
        for p in sorted(self.sources_dir.iterdir()):
            if p.suffix.lower() not in _VIDEO_EXTENSIONS:
                continue
            info = VideoInfo(name=p.name, path=p)
            # Attempt to populate metadata without hard-depending on PyAV.
            try:
                from castle.utils.video_io import VideoReader  # noqa: PLC0415

                with VideoReader(p) as reader:
                    info.fps = float(reader.fps)
                    info.width = int(reader.width)
                    info.height = int(reader.height)
                    info.n_frames = int(reader.frame_count)
            except Exception:  # noqa: BLE001
                pass
            results.append(info)
        return results

    def load_config(self) -> dict:
        """Read and return the project configuration dict.

        Returns:
            Parsed JSON as a :class:`dict`.

        Raises:
            FileNotFoundError: If ``config.json`` does not exist.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"config.json not found: {self.config_path}")
        with self.config_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def ensure_dirs(self) -> None:
        """Create all standard project directories if they don't exist.

        Directories created:

        * ``sources/``
        * ``track/``
        * ``latent/``
        * ``cluster/``
        * ``preprocessed/``
        """
        for directory in (
            self.sources_dir,
            self.track_dir,
            self.latent_dir,
            self.cluster_dir,
            self.preprocessed_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
