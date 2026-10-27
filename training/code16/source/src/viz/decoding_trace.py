from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def save_step_count_plot(traces: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = [int(trace["step"]) for trace in traces]
    remaining = [int(trace["masked_remaining"]) for trace in traces]
    plt.figure(figsize=(7, 4))
    plt.plot(steps, remaining, marker="o")
    plt.xlabel("decoding step")
    plt.ylabel("masked remaining")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

