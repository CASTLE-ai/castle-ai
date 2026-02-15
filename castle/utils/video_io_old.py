"""Deprecated: Use castle.utils.video_io instead."""
import warnings
warnings.warn(
    "castle.utils.video_io_old is deprecated. Use castle.utils.video_io instead.",
    DeprecationWarning,
    stacklevel=2
)
from castle.utils.video_io import VideoReader as ReadArray, VideoWriter as WriteArray
