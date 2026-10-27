"""Pre-compute backbone-derived node features for training shards.

Backbone is frozen during training, so its forward pass on each shard depends
only on the (input_ids_masked, attention_mask) of the label record. We compute
it once and persist the slim derived features needed by the graph builder:

- ``hidden_at_masked``  [num_masked, hidden_dim]
- ``confidence`` / ``margin`` / ``entropy``  [num_masked]
- ``topk_ids``         [num_masked, k]            (k = min(top_k, vocab))
- ``topk_probs``       [num_masked, top_k]        (padded with zeros if k < top_k)

This avoids 5-10x of training cost vs. running the backbone every epoch.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from src.backbones.registry import build_backbone
from src.eval.metrics import confidence_margin_entropy
from src.graph.features import FeatureConfig
from src.labels.storage import LabelRecord, load_label_shard
from src.utils.config import load_config
from src.utils.device import resolve_device


FeatureRecord = dict[str, Any]


@torch.no_grad()
def cache_record(
    backbone: Any,
    record: LabelRecord,
    feature_config: FeatureConfig,
    device: torch.device,
) -> FeatureRecord:
    input_ids = record["input_ids_masked"].unsqueeze(0).to(device)
    attention_mask = record["attention_mask"].unsqueeze(0).to(device)
    masked_positions = record["masked_positions"].to(device)
    timestep = int(record.get("timestep", 0))
    out = backbone.forward(input_ids, attention_mask, timestep=timestep)
    hidden = out["hidden_states"][0]
    logits = out["logits"][0]
    if masked_positions.numel() == 0:
        hidden_dim = int(hidden.shape[-1])
        return {
            "hidden_at_masked": torch.empty((0, hidden_dim), dtype=torch.float32),
            "confidence": torch.empty((0,), dtype=torch.float32),
            "margin": torch.empty((0,), dtype=torch.float32),
            "entropy": torch.empty((0,), dtype=torch.float32),
            "topk_ids": torch.empty((0, feature_config.top_k), dtype=torch.long),
            "topk_probs": torch.empty((0, feature_config.top_k), dtype=torch.float32),
            "timestep": timestep,
        }
    hidden_at_masked = hidden.index_select(0, masked_positions)
    row_logits = logits.index_select(0, masked_positions)
    confidence, margin, entropy = confidence_margin_entropy(row_logits)
    k = min(feature_config.top_k, row_logits.shape[-1])
    topk = torch.topk(torch.softmax(row_logits, dim=-1), k=k, dim=-1)
    topk_ids = topk.indices
    topk_probs = topk.values
    if topk_probs.shape[-1] < feature_config.top_k:
        pad_probs = topk_probs.new_zeros(
            (topk_probs.shape[0], feature_config.top_k - topk_probs.shape[-1])
        )
        topk_probs = torch.cat([topk_probs, pad_probs], dim=-1)
    return {
        "hidden_at_masked": hidden_at_masked.detach().to(torch.float32).cpu(),
        "confidence": confidence.detach().to(torch.float32).cpu(),
        "margin": margin.detach().to(torch.float32).cpu(),
        "entropy": entropy.detach().to(torch.float32).cpu(),
        "topk_ids": topk_ids.detach().to(torch.long).cpu(),
        "topk_probs": topk_probs.detach().to(torch.float32).cpu(),
        "timestep": timestep,
    }


def cache_label_shard(
    input_path: Path,
    output_path: Path,
    backbone: Any,
    feature_config: FeatureConfig,
    device: torch.device,
) -> None:
    records = load_label_shard(input_path)
    cached = [cache_record(backbone, record, feature_config, device) for record in records]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cached, output_path)


def cache_features(
    label_dir: str | Path,
    output_dir: str | Path,
    backbone: Any,
    feature_config: FeatureConfig,
    device: torch.device,
) -> Path:
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(label_dir.glob("labels-*.pt"))
    if not shards:
        raise ValueError(f"No label shards found in {label_dir}")
    for shard in tqdm(shards, desc=f"cache:{label_dir.name}"):
        output_path = output_dir / shard.name.replace("labels-", "features-")
        cache_label_shard(shard, output_path, backbone, feature_config, device)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML with backbone and features sections")
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device", "auto"))
    backbone = build_backbone(cfg.get("backbone", {})).to(device)
    feature_config = FeatureConfig.from_raw(cfg.get("features", {}))
    cache_features(args.label_dir, args.output_dir, backbone, feature_config, device)


if __name__ == "__main__":
    main()
