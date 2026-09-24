"""Regression: session_meta.json must be rewritable on Windows (WinError 183)."""

import json
import pathlib


def test_save_session_meta_overwrites_without_rename(tmp_path, monkeypatch):
    from castle.core.preprocess_session import get_session_dir, save_session_meta

    # Windows' rename refuses to overwrite an existing file; make it fail here too.
    def _no_overwrite(self, target):
        if pathlib.Path(target).exists():
            raise FileExistsError(183, "Cannot create a file when that file already exists")
        return pathlib.Path.replace(self, target)

    monkeypatch.setattr(pathlib.Path, "rename", _no_overwrite)
    save_session_meta(str(tmp_path), "proj", "s1", {"step": 1})
    save_session_meta(str(tmp_path), "proj", "s1", {"step": 2})

    session_dir = get_session_dir(str(tmp_path), "proj", "s1")
    assert json.loads((session_dir / "session_meta.json").read_text(encoding="utf-8")) == {"step": 2}
    assert not (session_dir / "session_meta.tmp").exists()
