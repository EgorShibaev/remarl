from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def save_conflict_heatmap(
    matrix: torch.Tensor,
    path: str | Path,
    title: str = "Predicted conflict probability",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix.detach().cpu().numpy(), vmin=0.0, vmax=1.0, cmap="magma")
    plt.colorbar(label="conflict")
    plt.title(title)
    plt.xlabel("masked node")
    plt.ylabel("masked node")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

