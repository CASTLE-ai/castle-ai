"""Unit tests for :mod:`castle.utils.video_io` (UX-05 / P2-B)."""

from __future__ import annotations

import numpy as np
import pytest


def test_reader_yields_correct_frame_count(synthetic_video) -> None:
    """``len(VideoReader)`` matches the number of encoded frames."""
    from castle.utils.video_io import VideoReader

    with VideoReader(synthetic_video) as reader:
        assert len(reader) == 10


def test_reader_first_and_last_frame_content(synthetic_video) -> None:
    """Red-channel encoding survives the round-trip via libx264."""
    from castle.utils.video_io import VideoReader

    with VideoReader(synthetic_video) as reader:
        first = reader[0]
        last = reader[9]

    assert first.shape == (64, 64, 3)
    assert first.dtype == np.uint8

    # frame 0 red ≈ 0, frame 9 red ≈ 225 (allow generous ±15 for codec rounding)
    first_red = float(first[..., 0].mean())
    last_red = float(last[..., 0].mean())
    assert first_red < 15, f"frame 0 red mean {first_red} should be near 0"
    assert last_red > 200, f"frame 9 red mean {last_red} should be near 225"


def test_reader_random_access_matches_sequential(synthetic_video) -> None:
    """Indexing out of order returns the same frames as sequential reads."""
    from castle.utils.video_io import VideoReader

    with VideoReader(synthetic_video) as reader:
        sequential = [reader[i] for i in range(len(reader))]
    with VideoReader(synthetic_video) as reader:
        shuffled = {i: reader[i] for i in (4, 0, 9, 2)}

    for i, frame in shuffled.items():
        # Channels mean should match the sequential read of the same index.
        np.testing.assert_allclose(
            frame.mean(axis=(0, 1)),
            sequential[i].mean(axis=(0, 1)),
            atol=2.0,
        )


def test_reader_context_manager_closes(synthetic_video) -> None:
    """Exiting the context manager releases the underlying av container."""
    from castle.utils.video_io import VideoReader

    reader = VideoReader(synthetic_video)
    assert len(reader) == 10
    reader.__exit__(None, None, None)
    # Second close should not raise (idempotent)
    reader.__exit__(None, None, None)


def test_writer_roundtrip(tmp_path) -> None:
    """Frames written by VideoWriter can be read back by VideoReader."""
    from castle.utils.video_io import VideoReader, VideoWriter

    out = tmp_path / "roundtrip.mp4"
    written = np.zeros((5, 32, 32, 3), dtype=np.uint8)
    for i in range(5):
        written[i, :, :, 1] = (i + 1) * 40  # green channel

    with VideoWriter(out, fps=24.0, crf=18) as w:
        for frame in written:
            w.write_frame(frame)

    with VideoReader(out) as r:
        assert len(r) == 5
        first_green = float(r[0][..., 1].mean())
        last_green = float(r[4][..., 1].mean())

    # Codec rounding tolerance ±10
    assert abs(first_green - 40) < 15
    assert abs(last_green - 200) < 15


def _write_video_with_pts(path, pts_ms) -> None:
    """Encode 32x32 frames at the given millisecond timestamps (VFR if uneven)."""
    import av
    from fractions import Fraction

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = Fraction(1, 1000)
        for i, t in enumerate(pts_ms):
            img = np.full((32, 32, 3), (i * 3) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = t
            frame.time_base = Fraction(1, 1000)
            for pkt in stream.encode(frame):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)


def test_find_unindexable_frame_accepts_constant_rate(synthetic_video) -> None:
    from castle.utils.video_io import find_unindexable_frame

    assert find_unindexable_frame(synthetic_video) is None


def test_find_unindexable_frame_matches_reader_on_vfr(tmp_path) -> None:
    """A burst of dense frames followed by sparse ones is flagged at the same
    index where VideoReader random access actually fails."""
    from castle.utils.video_io import VideoReader, find_unindexable_frame

    path = tmp_path / "vfr.mp4"
    _write_video_with_pts(path, [10 * i for i in range(50)] + [500 + 500 * i for i in range(20)])

    bad = find_unindexable_frame(path)
    assert bad is not None
    with VideoReader(path) as reader:
        with pytest.raises(Exception):
            reader[bad]


def test_add_video_rejects_vfr(tmp_path) -> None:
    from castle.utils.video_manager import add_video_to_project

    project = tmp_path / "proj"
    project.mkdir()
    (project / "config.json").write_text('{"source": []}', encoding="utf-8")
    path = tmp_path / "vfr.mp4"
    _write_video_with_pts(path, [10 * i for i in range(50)] + [500 + 500 * i for i in range(20)])

    ok, msg = add_video_to_project(str(tmp_path), "proj", str(path), "vfr.mp4")
    assert not ok
    assert "variable frame rate" in msg
