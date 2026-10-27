"""Shared Weights & Biases helpers.

Centralizes run init + tag construction so training and all eval entrypoints log
to W&B consistently. Logging always fails soft: a missing package, missing key,
or network error must never break an experiment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def standard_tags(config: dict[str, Any], **extra: Any) -> list[str]:
    """Build a flat ``key:value`` tag list from the standardized schema.

    Pulls the conventional axes (backbone, dataset, model_type, mask_ratio, tau,
    anchors, seed, selection, theta, budget, label_scale) from ``config`` where
    present, plus any ``extra`` overrides (e.g. ``decoder=...``). Values that are
    None are skipped.
    """
    backbone = config.get("backbone", {}) or {}
    data = config.get("data", {}) or {}
    model = config.get("model", {}) or {}
    labels = config.get("labels", {}) or {}
    decoding = config.get("decoding", {}) or {}

    candidates: dict[str, Any] = {
        "backbone": backbone.get("type") or backbone.get("name"),
        "dataset": data.get("dataset_name"),
        "model_type": model.get("type"),
        "tau": labels.get("tau"),
        "anchors": labels.get("anchors_per_example"),
        "seed": config.get("seed"),
        "label_scale": data.get("max_examples"),
        "budget": decoding.get("budget_fraction"),
    }
    candidates.update(extra)
    tags: list[str] = []
    for key, value in candidates.items():
        if value is None:
            continue
        tags.append(f"{key}:{value}")
    # free-form tags from the tracking block
    for tag in (config.get("tracking", {}) or {}).get("tags", []) or []:
        tags.append(str(tag))
    return tags


def log_artifact(
    wandb_run,
    path: Path | str,
    name: str,
    artifact_type: str = "model",
    metadata: dict[str, Any] | None = None,
    aliases: list[str] | None = None,
) -> bool:
    """Upload a file as a W&B artifact; returns whether it was uploaded.

    Doubles as an off-cluster **backup** of trained weights: everything under
    ``/data/horse/ws`` is scratch storage, so checkpoints only tracked there can vanish.
    Fails soft -- a broken upload must never take down a training run.
    """
    if wandb_run is None:
        return False
    path = Path(path)
    if not path.exists():
        return False
    try:
        import wandb

        artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
        artifact.add_file(str(path))
        wandb_run.log_artifact(artifact, aliases=aliases)
        return True
    except Exception as exc:  # noqa: BLE001 - artifact upload must not break experiments
        print(f"[tracking] wandb artifact upload failed ({exc})")
        return False


def init_wandb(
    config: dict[str, Any],
    run_dir: Path,
    job_type: str | None = None,
    extra_tags: dict[str, Any] | None = None,
):
    """Initialize a W&B run if ``tracking.wandb`` is true; else return None."""
    tracking = config.get("tracking", {}) or {}
    if not bool(tracking.get("wandb", False)):
        return None
    try:
        import wandb
    except ImportError:
        print("[tracking] wandb requested but not installed; continuing without it")
        return None
    tags = standard_tags(config, **(extra_tags or {}))
    try:
        return wandb.init(
            project=tracking.get("project", os.environ.get("WANDB_PROJECT", "gnn-conflict")),
            entity=tracking.get("entity") or os.environ.get("WANDB_ENTITY"),
            name=tracking.get("run_name") or run_dir.name,
            group=tracking.get("group"),
            job_type=job_type or tracking.get("job_type"),
            tags=tags or None,
            config=config,
            dir=str(tracking.get("dir") or os.environ.get("WANDB_DIR") or run_dir),
            mode=tracking.get("mode", os.environ.get("WANDB_MODE", "online")),
        )
    except Exception as exc:  # noqa: BLE001 - logging must never break experiments
        print(f"[tracking] wandb init failed ({exc}); continuing without it")
        return None
