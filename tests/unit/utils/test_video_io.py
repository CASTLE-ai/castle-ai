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
