"""Held-out conflict-label evaluation (Phase 0/3 diagnostics).

Runs a trained conflict head over the dev label shards and reports how well it
predicts the counterfactual conflict labels, independent of decoding:

- classification: ROC-AUC, average precision (AP), F1 / precision / recall at a
  threshold, Brier score, soft-label correlation
- label diagnostics: positive (conflict) rate = graph density, pair-distance
  distribution, same-span vs cross-span conflict rates

This is the metric to track when sweeping label definitions / features, because
it is cheap and does not depend on the decode schedule.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from src.graph.graph_builder import build_full_graph, build_full_graph_from_cached
from src.labels.pair_sampling import same_span_mask
from src.labels.storage import LabelRecordDataset
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.logging import make_run_dir, write_json
from src.utils.seed import set_seed
from src.utils.tracking import init_wandb


def _binary_metrics(probs: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict[str, float]:
    """ROC-AUC / AP / F1 etc. Falls back gracefully when a class is missing."""
    import numpy as np
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y = labels.detach().cpu().numpy().astype(int)
    p = probs.detach().cpu().numpy().astype(float)
    out: dict[str, float] = {
        "num_pairs": int(y.size),
        "positive_rate": float(y.mean()) if y.size else 0.0,
        "predicted_positive_rate": float((p > threshold).mean()) if p.size else 0.0,
        "brier": float(np.mean((p - y) ** 2)) if p.size else 0.0,
        "threshold": float(threshold),
    }
    has_both = y.size > 0 and 0 < int(y.sum()) < int(y.size)
    if has_both:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["average_precision"] = float(average_precision_score(y, p))
        preds = (p > threshold).astype(int)
        out["f1"] = float(f1_score(y, preds, zero_division=0))
        out["precision"] = float(precision_score(y, preds, zero_division=0))
        out["recall"] = float(recall_score(y, preds, zero_division=0))
    else:
        out["roc_auc"] = float("nan")
        out["average_precision"] = float("nan")
        out["f1"] = float("nan")
        out["precision"] = float("nan")
        out["recall"] = float("nan")
    return out


@torch.no_grad()
def evaluate_conflict_labels(config: dict[str, Any], checkpoint_path: str) -> Path:
    set_seed(int(config.get("seed", 13)))
    device = resolve_device(config.get("device", "auto"))
    run_dir = make_run_dir(config.get("output_dir", "outputs/runs"), "eval-conflict-labels")
    write_json(run_dir / "resolved_config.json", config)

    # Lazy import to avoid a hard dependency cycle with the decoder module.
    from src.decoding.gnn_conflict_decoder import build_conflict_model_from_checkpoint

    model, feature_config = build_conflict_model_from_checkpoint(checkpoint_path, device)

    data_cfg = config.get("data", {})
    label_dir = data_cfg.get("dev_label_dir") or data_cfg.get("label_dir")
    feature_cache = data_cfg.get("dev_feature_cache_dir") or data_cfg.get("feature_cache_dir")
    dataset = LabelRecordDataset(label_dir, feature_cache_dir=feature_cache)

    backbone = None  # built lazily only if some record lacks cached features

    all_probs: list[torch.Tensor] = []
    all_binary: list[torch.Tensor] = []
    all_soft: list[torch.Tensor] = []
    all_distance: list[torch.Tensor] = []
    all_same_span: list[torch.Tensor] = []

    for record in dataset:
        masked_positions = record["masked_positions"].to(device)
        pairs = record["pairs"].to(device)
        if pairs.numel() == 0 or masked_positions.numel() < 2:
            continue
        timestep = int(record.get("timestep", 0))
        cached = record.get("cached_features")
        if cached is not None:
            graph = build_full_graph_from_cached(
                cached=cached,
                masked_positions=masked_positions,
                seq_len=int(record["attention_mask"].shape[0]),
                timestep=timestep,
                config=feature_config,
                pairs=pairs,
            )
        else:
            if backbone is None:
                from src.backbones.registry import build_backbone

                backbone = build_backbone(config.get("backbone", {})).to(device)
            out = backbone.forward(
                record["input_ids_masked"].unsqueeze(0).to(device),
                record["attention_mask"].unsqueeze(0).to(device),
                timestep=timestep,
            )
            graph = build_full_graph(
                hidden_states=out["hidden_states"][0],
                logits=out["logits"][0],
                masked_positions=masked_positions,
                timestep=timestep,
                config=feature_config,
                pairs=pairs,
            )
        logits = model(graph.node_features, graph.edge_index, graph.edge_features)
        probs = torch.sigmoid(logits)
        all_probs.append(probs.detach().cpu())
        all_binary.append(record["binary_labels"].float().cpu())
        all_soft.append(record["soft_labels"].float().cpu())
        signed = masked_positions[pairs[:, 0]] - masked_positions[pairs[:, 1]]
        all_distance.append(signed.abs().float().cpu())
        all_same_span.append(same_span_mask(masked_positions, pairs).cpu())

    if not all_probs:
        raise ValueError("No labeled pairs found for conflict-label evaluation")

    probs = torch.cat(all_probs)
    binary = torch.cat(all_binary)
    soft = torch.cat(all_soft)
    distance = torch.cat(all_distance)
    same_span = torch.cat(all_same_span)

    threshold = float(config.get("metrics", {}).get("predicted_conflict_theta", 0.5))
    summary = _binary_metrics(probs, binary, threshold)
    # soft-label correlation (how well probs track the continuous JS-based target)
    if probs.numel() > 1 and float(soft.std()) > 0 and float(probs.std()) > 0:
        summary["soft_label_corr"] = float(torch.corrcoef(torch.stack([probs, soft]))[0, 1].item())
    else:
        summary["soft_label_corr"] = float("nan")
    summary["distance_mean"] = float(distance.mean().item())
    summary["distance_median"] = float(distance.median().item())
    same = same_span.bool()
    summary["same_span_conflict_rate"] = (
        float(binary[same].mean().item()) if int(same.sum()) > 0 else float("nan")
    )
    summary["cross_span_conflict_rate"] = (
        float(binary[~same].mean().item()) if int((~same).sum()) > 0 else float("nan")
    )
    summary["same_span_pair_fraction"] = float(same.float().mean().item())
    summary["examples"] = len(all_probs)
    summary["checkpoint"] = checkpoint_path

    write_json(run_dir / "conflict_label_metrics.json", summary)

    wandb_run = init_wandb(config, run_dir, job_type="eval-conflict-labels", extra_tags={"eval": "labels"})
    if wandb_run is not None:
        wandb_run.summary.update({f"labels/{k}": v for k, v in summary.items() if isinstance(v, (int, float))})
        wandb_run.finish()

    print(f"[conflict-labels] AUC={summary['roc_auc']:.4f} AP={summary['average_precision']:.4f} "
          f"F1={summary['f1']:.4f} pos_rate={summary['positive_rate']:.4f}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="conflict-model checkpoint (.pt)")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = args.checkpoint or config.get("checkpoint", {}).get("path", "outputs/checkpoints/conflict_gnn.pt")
    evaluate_conflict_labels(config, checkpoint)


if __name__ == "__main__":
    main()
