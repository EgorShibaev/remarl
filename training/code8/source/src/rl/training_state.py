"""Full frozen-planner resume state and safe checkpoint publication."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def environment_fingerprint(env) -> str:
    digest = hashlib.sha256()
    for p in env.problems:
        for t in (p.input_ids, p.attention_mask, p.answer_span):
            digest.update(t.detach().cpu().numpy().tobytes())
        digest.update(json.dumps([p.y_star, p.nfe_max], sort_keys=True).encode())
    return digest.hexdigest()


def capture_rng(device: torch.device) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }


def restore_rng(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if device.type != "cuda" or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same visible CUDA-device count")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def atomic_torch_save(payload: dict, path: Path, max_bytes: int = 16 * 1024**2) -> None:
    """Replace only this run's last checkpoint after a complete, flushed write."""
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    if temporary.stat().st_size > max_bytes:
        raise RuntimeError("Checkpoint exceeded its declared storage cap")
    os.replace(temporary, path)


def grad_norm(tensors) -> float:
    return sum(float(t.detach().float().square().sum()) for t in tensors if t is not None) ** 0.5
