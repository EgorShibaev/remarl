from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass
class LatencyStats:
    name: str
    elapsed: float = 0.0


@contextmanager
def measure_latency(name: str) -> Iterator[LatencyStats]:
    stats = LatencyStats(name=name)
    start = time.perf_counter()
    try:
        yield stats
    finally:
        stats.elapsed = time.perf_counter() - start

