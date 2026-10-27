from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    # Full trainer states contain executable pickle objects (e.g. NumPy RNG state).
    # Evaluation deliberately accepts only the separately exported tensor/primitive payload.
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint must contain a dict: {path}")
    return payload
