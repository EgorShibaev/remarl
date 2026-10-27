"""Run one sweep cell end-to-end from a single merged config.

A "cell" is one point in a sweep grid. Its config carries every section
(backbone, data, labels, features, model, training, metrics, decoding, tracking)
plus a ``cell`` block describing the working directory, which stages to run, and
which decoders to evaluate. Stages call the existing pipeline functions directly
(no subprocess), with all paths rooted at ``cell.workdir`` so cells never
collide and can run fully in parallel.

Stages: labels -> cache -> train -> eval_recon -> eval_labels
"""

from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path
from typing import Any

from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.logging import write_json


def _labels_dir(workdir: Path) -> Path:
    return workdir / "labels"


def _run_labels(config: dict[str, Any], workdir: Path, splits: list[str]) -> None:
    from src.labels.generate_counterfactual_labels import generate

    label_cfg = copy.deepcopy(config)
    label_cfg["output_dir"] = str(_labels_dir(workdir))
    for split in splits:
        generate(label_cfg, split)
    # alias validation -> dev for the trainer
    dev = _labels_dir(workdir) / "dev"
    val = _labels_dir(workdir) / "validation"
    if val.exists() and not dev.exists():
        dev.symlink_to("validation")


def _run_cache(config: dict[str, Any], workdir: Path) -> None:
    from src.backbones.registry import build_backbone
    from src.graph.features import FeatureConfig
    from src.labels.feature_cache import cache_features

    device = resolve_device(config.get("device", "auto"))
    backbone = build_backbone(config.get("backbone", {})).to(device)
    feature_config = FeatureConfig.from_raw(config.get("features", {}))
    for split in ("train", "dev"):
        label_dir = _labels_dir(workdir) / split
        if label_dir.exists():
            cache_features(label_dir, workdir / "cache" / split, backbone, feature_config, device)


def _label_root(cell: dict[str, Any], workdir: Path) -> Path:
    """Where labels + feature cache live: a shared labelset (reused across model x
    seed cells) when ``cell.labelset`` is set, else the cell's own workdir."""
    labelset = cell.get("labelset")
    return Path("outputs/labelsets") / labelset if labelset else workdir


def _train_paths(config: dict[str, Any], workdir: Path, label_root: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg["output_dir"] = str(workdir / "runs")
    cfg.setdefault("data", {})
    cfg["data"]["label_dir"] = str(label_root / "labels" / "train")
    cfg["data"]["dev_label_dir"] = str(label_root / "labels" / "dev")
    cfg["data"]["feature_cache_dir"] = str(label_root / "cache" / "train")
    cfg["data"]["dev_feature_cache_dir"] = str(label_root / "cache" / "dev")
    return cfg


def _run_train(config: dict[str, Any], workdir: Path, label_root: Path) -> Path:
    from src.train.train_conflict_model import train

    cfg = _train_paths(config, workdir, label_root)
    checkpoint_path = train(cfg)
    dst = workdir / "checkpoints" / "conflict_gnn.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(checkpoint_path, dst)
    return dst


def _run_eval_recon(
    config: dict[str, Any], workdir: Path, decoders: list[str], eval_max_examples: int | None
) -> None:
    from src.eval.evaluate_reconstruction import evaluate

    cfg = copy.deepcopy(config)
    cfg["output_dir"] = str(workdir / "runs")
    cfg.setdefault("checkpoint", {})["path"] = str(workdir / "checkpoints" / "conflict_gnn.pt")
    if eval_max_examples is not None:
        cfg.setdefault("data", {})["max_examples"] = int(eval_max_examples)
    for decoder in decoders:
        try:
            evaluate(cfg, decoder)
        except Exception as exc:  # noqa: BLE001 - one decoder must not abort the cell
            print(f"[cell] eval decoder '{decoder}' failed: {type(exc).__name__}: {exc}")


def _run_eval_labels(config: dict[str, Any], workdir: Path, label_root: Path) -> None:
    from src.eval.evaluate_conflict_labels import evaluate_conflict_labels

    cfg = _train_paths(config, workdir, label_root)
    checkpoint = str(workdir / "checkpoints" / "conflict_gnn.pt")
    evaluate_conflict_labels(cfg, checkpoint)


def run_cell(config: dict[str, Any]) -> None:
    cell = config.get("cell", {}) or {}
    workdir = Path(cell.get("workdir", "outputs/sweeps/cell"))
    workdir.mkdir(parents=True, exist_ok=True)
    write_json(workdir / "cell_config.json", config)
    stages = cell.get("stages", ["labels", "cache", "train", "eval_recon", "eval_labels"])
    splits = cell.get("splits", ["train", "validation", "test"])
    decoders = cell.get("eval_decoders", ["confidence", "gnn"])
    label_root = _label_root(cell, workdir)

    print(f"[cell] workdir={workdir} stages={stages} label_root={label_root}")
    if "labels" in stages:
        _run_labels(config, workdir, splits)
    if "cache" in stages:
        _run_cache(config, workdir)
    if "train" in stages:
        _run_train(config, workdir, label_root)
    if "eval_recon" in stages:
        _run_eval_recon(config, workdir, decoders, cell.get("eval_max_examples"))
    if "eval_labels" in stages:
        _run_eval_labels(config, workdir, label_root)
    print(f"[cell] done: {workdir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="merged cell config (yaml)")
    args = parser.parse_args()
    run_cell(load_config(args.config))


if __name__ == "__main__":
    main()
