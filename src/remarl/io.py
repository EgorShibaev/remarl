"""Atomic output and content hashes."""

import hashlib
import json
import os
from pathlib import Path
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(state, temp)
    temp.replace(path)


def append_json(path, value):
    path = Path(path)
    # Some Lustre mounts reject O_CREAT | O_APPEND for a new striped file.
    # Create it separately before opening it in append mode.
    if not path.exists():
        path.touch()
    with path.open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
