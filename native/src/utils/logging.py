from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def make_run_dir(base_dir: str | Path, prefix: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = Path(base_dir)
    # Concurrent jobs that start in the same second would otherwise collide,
    # so fall back to a pid/counter suffix until we find a free directory.
    candidate = base / f"{prefix}-{stamp}"
    suffix = 0
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            suffix += 1
            candidate = base / f"{prefix}-{stamp}-{os.getpid()}-{suffix}"


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_json(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)

