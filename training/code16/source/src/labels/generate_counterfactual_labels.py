from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.backbones.registry import build_backbone
from src.data.datasets import collate_tokenized, load_text_dataset, resolve_masking
from src.data.masking import apply_mask
from src.eval.metrics import confidence_margin_entropy
from src.labels.dependency_scores import (
    js_divergence_from_logits,
    soft_conflict_label,
    top1_flip,
    true_token_logprob_change,
)
from src.labels.storage import LabelRecord, save_label_shard
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.seed import set_seed


def _choose_mask_ratio(buckets: list[list[float]]) -> float:
    low, high = random.choice(buckets)
    return random.uniform(float(low), float(high))


def _special_token_ids(tokenizer: Any) -> set[int]:
    ids = set(tokenizer.all_special_ids)
    if tokenizer.mask_token_id is not None:
        ids.add(int(tokenizer.mask_token_id))
    if tokenizer.pad_token_id is not None:
        ids.add(int(tokenizer.pad_token_id))
    return ids


@torch.no_grad()
def make_label_record(
    backbone: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_ratio: float,
    max_masked_positions: int,
    anchors_per_example: int,
    tau: float,
    temperature: float,
    topk_for_js: int | None,
    timestep: int = 0,
    signal: str = "js",
    aggregate: str = "max",
    strategy: str = "random",
    restrict_token_ids: set[int] | None = None,
) -> LabelRecord | None:
    masked = apply_mask(
        input_ids=input_ids.unsqueeze(0),
        attention_mask=attention_mask.unsqueeze(0),
        mask_token_id=backbone.mask_token_id,
        mask_ratio=mask_ratio,
        special_token_ids=_special_token_ids(backbone.tokenizer),
        max_masked_positions=max_masked_positions,
        strategy=strategy,
        restrict_token_ids=restrict_token_ids,
    )
    masked_positions = masked.masked_positions[0]
    if masked_positions.numel() < 2:
        return None
    input_ids_masked = masked.input_ids_masked.to(input_ids.device)
    clean_input_ids = masked.clean_input_ids.to(input_ids.device)
    attention = masked.attention_mask.to(input_ids.device)
    base = backbone.forward(input_ids_masked, attention, timestep=timestep)
    base_logits = base["logits"][0, masked_positions]
    confidence, _, _ = confidence_margin_entropy(base_logits)
    num_masked = int(masked_positions.numel())
    anchor_count = min(anchors_per_example, num_masked)
    anchor_local = torch.randperm(num_masked, device=input_ids.device)[:anchor_count]
    score_matrix = torch.zeros((num_masked, num_masked), dtype=torch.float32, device=input_ids.device)
    observed = torch.zeros((num_masked, num_masked), dtype=torch.bool, device=input_ids.device)

    for anchor_idx in anchor_local:
        anchor = int(anchor_idx.item())
        anchor_pos = masked_positions[anchor]
        counterfactual_ids = input_ids_masked.clone()
        counterfactual_ids[0, anchor_pos] = clean_input_ids[0, anchor_pos]
        cf = backbone.forward(counterfactual_ids, attention, timestep=timestep)
        target_local = torch.tensor(
            [idx for idx in range(num_masked) if idx != anchor],
            dtype=torch.long,
            device=input_ids.device,
        )
        target_pos = masked_positions[target_local]
        base_targets = base_logits.index_select(0, target_local)
        cf_targets = cf["logits"][0, target_pos]
        if signal == "js":
            scores = js_divergence_from_logits(base_targets, cf_targets, topk=topk_for_js)
        elif signal == "logprob_change":
            true_ids = clean_input_ids[0, target_pos]
            scores = true_token_logprob_change(base_targets, cf_targets, true_ids)
        elif signal == "top1_flip":
            scores = top1_flip(base_targets, cf_targets)
        else:
            raise ValueError(f"Unknown label signal: {signal}")
        score_matrix[anchor, target_local] = scores
        observed[anchor, target_local] = True

    pairs: list[list[int]] = []
    scores_acc: list[torch.Tensor] = []
    if aggregate == "directed":
        # keep both orderings as separate labeled (directed) edges
        for i in range(num_masked):
            for j in range(num_masked):
                if i == j or not bool(observed[i, j]):
                    continue
                pairs.append([i, j])
                scores_acc.append(score_matrix[i, j])
    else:
        for i in range(num_masked):
            for j in range(i + 1, num_masked):
                values = []
                if bool(observed[i, j]):
                    values.append(score_matrix[i, j])
                if bool(observed[j, i]):
                    values.append(score_matrix[j, i])
                if not values:
                    continue
                pairs.append([i, j])
                stacked = torch.stack(values)
                scores_acc.append(stacked.mean() if aggregate == "mean" else stacked.max())
    if not pairs:
        return None
    dependency_scores = torch.stack(scores_acc)
    soft_labels = soft_conflict_label(dependency_scores, tau=tau, temperature=temperature)
    binary_labels = dependency_scores.gt(tau).long()
    return {
        "input_ids_masked": input_ids_masked.squeeze(0).cpu(),
        "clean_input_ids": clean_input_ids.squeeze(0).cpu(),
        "attention_mask": attention.squeeze(0).cpu(),
        "timestep": int(timestep),
        "masked_positions": masked_positions.cpu(),
        "pairs": torch.tensor(pairs, dtype=torch.long),
        "soft_labels": soft_labels.cpu(),
        "binary_labels": binary_labels.cpu(),
        "dependency_scores": dependency_scores.cpu(),
        "base_confidence": confidence.cpu(),
        "metadata": {"mask_ratio": float(mask_ratio), "anchors": [int(x) for x in anchor_local.cpu()]},
    }


def generate(config: dict[str, Any], split: str) -> Path:
    set_seed(int(config.get("seed", 13)))
    device = resolve_device(config.get("device", "auto"))
    backbone = build_backbone(config.get("backbone", {})).to(device)
    data_cfg = config["data"]
    labels_cfg = config["labels"]
    dataset = load_text_dataset(
        tokenizer=backbone.tokenizer,
        data_cfg=data_cfg,
        split=split,
        max_examples=int(data_cfg.get("max_examples", 128)),
        max_length=int(data_cfg.get("max_length", 128)),
    )
    strategy, restrict_token_ids = resolve_masking(data_cfg, backbone.tokenizer)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_tokenized)
    output_dir = Path(config.get("output_dir", "data/conflict_labels")) / split
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_size = int(labels_cfg.get("shard_size", 128))
    masks_per_example = max(1, int(labels_cfg.get("masks_per_example", 1)))
    target_labels = labels_cfg.get("target_labels")  # optional hard cap on total records
    records: list[LabelRecord] = []
    shard_index = 0
    total = 0
    stop = False
    for batch in tqdm(loader, desc=f"labels:{split}"):
        if stop:
            break
        input_ids = batch["input_ids"][0].to(device)
        attention_mask = batch["attention_mask"][0].to(device)
        for _ in range(masks_per_example):
            record = make_label_record(
                backbone=backbone,
                input_ids=input_ids,
                attention_mask=attention_mask,
                mask_ratio=_choose_mask_ratio(data_cfg.get("mask_ratio_buckets", [[0.25, 0.60]])),
                max_masked_positions=int(labels_cfg.get("max_masked_positions", 64)),
                anchors_per_example=int(labels_cfg.get("anchors_per_example", 8)),
                tau=float(labels_cfg.get("tau", 0.05)),
                temperature=float(labels_cfg.get("temperature", 0.02)),
                topk_for_js=int(labels_cfg.get("topk_for_js", 128)),
                signal=str(labels_cfg.get("signal", "js")),
                aggregate=str(labels_cfg.get("aggregate", "max")),
                strategy=strategy,
                restrict_token_ids=restrict_token_ids,
            )
            if record is None:
                continue
            records.append(record)
            total += 1
            if len(records) >= shard_size:
                save_label_shard(output_dir / f"labels-{shard_index:05d}.pt", records)
                records = []
                shard_index += 1
            if target_labels is not None and total >= int(target_labels):
                stop = True
                break
    if records:
        save_label_shard(output_dir / f"labels-{shard_index:05d}.pt", records)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    generate(load_config(args.config), args.split)


if __name__ == "__main__":
    main()

