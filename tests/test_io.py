import errno
import json
from pathlib import Path

import pytest

from remarl.io import append_json


@pytest.mark.parametrize("existing", [False, True])
def test_append_json_on_filesystem_requiring_separate_creation(tmp_path, monkeypatch, existing):
    path = tmp_path / "training.jsonl"
    previous = [{"step": 0}] if existing else []
    if existing:
        path.write_text(json.dumps(previous[0]) + "\n")

    original_open = Path.open

    def open_existing_for_append(self, mode="r", *args, **kwargs):
        if "a" in mode and not self.exists():
            raise OSError(errno.EINVAL, "Cannot create a new log in append mode")
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_existing_for_append)
    append_json(str(path), {"step": 1})
    append_json(path, {"step": 2})

    assert [json.loads(line) for line in path.read_text().splitlines()] == (
        previous + [{"step": 1}, {"step": 2}]
    )
