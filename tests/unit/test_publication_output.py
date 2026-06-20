"""Publication-quality figure output + safe batch-frame error handling (Phase 4-5)."""

import os

import numpy as np
import pytest


def test_publication_dpi_constants():
    from castle.core import config
    assert config.PUBLICATION_DPI >= 300       # journal raster standard
    assert config.REPORT_EMBED_DPI >= 150
    assert config.FIGURE_VECTOR_FORMAT in ("svg", "pdf")


def test_save_publication_figure_emits_raster_plus_vector(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from castle.visualization.figure_io import save_publication_figure

    fig = plt.figure()
    fig.add_subplot(111).plot([0, 1], [0, 1])

    png = tmp_path / "fig.png"
    written = save_publication_figure(fig, str(png))
    assert os.path.isfile(png)
    assert os.path.isfile(tmp_path / "fig.svg")   # vector sibling for publication
    assert len(written) == 2
    plt.close(fig)


def test_save_publication_figure_vector_path_no_duplicate(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from castle.visualization.figure_io import save_publication_figure

    fig = plt.figure()
    fig.add_subplot(111).plot([0, 1], [0, 1])
    svg = tmp_path / "fig.svg"
    written = save_publication_figure(fig, str(svg))
    assert written == [str(svg)]  # a vector path is written once, no sibling
    plt.close(fig)


class _FlakyReader:
    """Minimal stand-in for VideoReader: frame 1 fails to decode."""
    height = 4
    width = 4

    def get_frame(self, idx):
        if idx == 1:
            raise RuntimeError("decode error")
        return np.full((4, 4, 3), idx, dtype=np.uint8)


def test_get_batch_frames_raises_by_default():
    from castle.utils.video_io import VideoReader
    with pytest.raises(RuntimeError, match="decode error"):
        VideoReader.get_batch_frames(_FlakyReader(), [0, 1, 2])


def test_get_batch_frames_zero_fill_is_opt_in():
    from castle.utils.video_io import VideoReader
    frames = VideoReader.get_batch_frames(_FlakyReader(), [0, 1, 2], on_error="zero")
    assert len(frames) == 3                 # index-aligned
    assert frames[1].shape == (4, 4, 3)
    assert int(frames[1].sum()) == 0        # the placeholder is all-black


def test_get_batch_frames_skip_drops_failed():
    from castle.utils.video_io import VideoReader
    frames = VideoReader.get_batch_frames(_FlakyReader(), [0, 1, 2], on_error="skip")
    assert len(frames) == 2                 # frame 1 dropped


def test_get_batch_frames_bad_mode():
    from castle.utils.video_io import VideoReader
    with pytest.raises(ValueError):
        VideoReader.get_batch_frames(_FlakyReader(), [0], on_error="bogus")
