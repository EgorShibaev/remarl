from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.backbones.registry import build_backbone
from src.graph.batching import batch_graphs
from src.graph.features import FeatureConfig, FeatureSpec
from src.graph.graph_builder import ConflictGraph, build_full_graph, build_full_graph_from_cached
from src.labels.storage import LabelRecord, LabelRecordDataset
from src.models.conflict_gnn import ConflictGNN, ConflictGNNConfig
from src.models.pairwise_mlp import PairwiseMLP, PairwiseMLPConfig
from src.utils.checkpointing import save_checkpoint
from src.utils.config import load_config, parse_scalar, set_dotted
from src.utils.device import resolve_device
from src.utils.logging import JsonlLogger, make_run_dir, write_json
from src.utils.seed import set_seed
from src.utils.tracking import init_wandb


def _collate_records(records: list[LabelRecord]) -> list[LabelRecord]:
    return records


@torch.no_grad()
def records_to_graphs(
    records: list[LabelRecord],
    backbone: Any,
    feature_config: FeatureConfig,
    device: torch.device,
) -> tuple[list[ConflictGraph], torch.Tensor]:
    graphs: list[ConflictGraph] = []
    labels: list[torch.Tensor] = []
    for record in records:
        masked_positions = record["masked_positions"].to(device)
        pairs = record["pairs"].to(device)
        timestep = int(record.get("timestep", 0))
        cached = record.get("cached_features")
        if cached is not None:
            seq_len = int(record["attention_mask"].shape[0])
            graph = build_full_graph_from_cached(
                cached=cached,
                masked_positions=masked_positions,
                seq_len=seq_len,
                timestep=timestep,
                config=feature_config,
                pairs=pairs,
            )
        else:
            input_ids = record["input_ids_masked"].unsqueeze(0).to(device)
            attention_mask = record["attention_mask"].unsqueeze(0).to(device)
            out = backbone.forward(input_ids, attention_mask, timestep=timestep)
            graph = build_full_graph(
                hidden_states=out["hidden_states"][0],
                logits=out["logits"][0],
                masked_positions=masked_positions,
                timestep=timestep,
                config=feature_config,
                pairs=pairs,
            )
        graphs.append(graph)
        labels.append(record["soft_labels"].float().to(device))
    return graphs, torch.cat(labels, dim=0)


def _build_model(config: dict[str, Any], feature_spec: FeatureSpec) -> nn.Module:
    model_cfg = config.get("model", {})
    model_type = str(model_cfg.get("type", "gnn"))
    common = {
        "input_node_dim": feature_spec.node_dim,
        "input_edge_dim": feature_spec.edge_dim,
        "node_dim": int(model_cfg.get("node_dim", 256)),
        "edge_dim": int(model_cfg.get("edge_dim", 64)),
        "hidden_dim": int(model_cfg.get("hidden_dim", 256)),
        "dropout": float(model_cfg.get("dropout", 0.1)),
    }
    if model_type == "pairwise_mlp":
        return PairwiseMLP(PairwiseMLPConfig(**common))
    if model_type == "gnn":
        return ConflictGNN(
            ConflictGNNConfig(**common, layers=int(model_cfg.get("layers", 2)))
        )
    raise ValueError(f"Unknown model type: {model_type}")


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    backbone: Any,
    feature_config: FeatureConfig,
    device: torch.device,
) -> float:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    losses: list[float] = []
    for records in loader:
        graphs, labels = records_to_graphs(records, backbone, feature_config, device)
        batched = batch_graphs(graphs)
        logits = model(batched.node_features, batched.edge_index, batched.edge_features)
        loss = criterion(logits, labels)
        losses.append(float(loss.item()))
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def evaluate_dev_metrics(
    model: nn.Module,
    loader: DataLoader,
    backbone: Any,
    feature_config: FeatureConfig,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Dev BCE loss + conflict-classification AUC/AP/positive-rate."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    losses: list[float] = []
    probs_all: list[torch.Tensor] = []
    binary_all: list[torch.Tensor] = []
    for records in loader:
        graphs, labels = records_to_graphs(records, backbone, feature_config, device)
        batched = batch_graphs(graphs)
        logits = model(batched.node_features, batched.edge_index, batched.edge_features)
        losses.append(float(criterion(logits, labels).item()))
        probs_all.append(torch.sigmoid(logits).detach().cpu())
        binary_all.append(torch.cat([r["binary_labels"].float().cpu() for r in records]))
    out = {"dev_loss": sum(losses) / max(len(losses), 1)}
    if probs_all:
        probs = torch.cat(probs_all).numpy()
        binary = torch.cat(binary_all).numpy().astype(int)
        out["dev_positive_rate"] = float(binary.mean()) if binary.size else 0.0
        if binary.size and 0 < int(binary.sum()) < int(binary.size):
            from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

            out["dev_auc"] = float(roc_auc_score(binary, probs))
            out["dev_ap"] = float(average_precision_score(binary, probs))
            out["dev_f1"] = float(f1_score(binary, (probs > threshold).astype(int), zero_division=0))
    return out


def train(config: dict[str, Any]) -> Path:
    set_seed(int(config.get("seed", 13)))
    device = resolve_device(config.get("device", "auto"))
    run_dir = make_run_dir(config.get("output_dir", "outputs/runs"), "train-conflict")
    write_json(run_dir / "resolved_config.json", config)
    logger = JsonlLogger(run_dir / "metrics.jsonl")
    wandb_run = init_wandb(config, run_dir, job_type="train")

    backbone = build_backbone(config.get("backbone", {})).to(device)
    train_dataset = LabelRecordDataset(
        config["data"]["label_dir"],
        feature_cache_dir=config["data"].get("feature_cache_dir"),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["data"].get("batch_size", 2)),
        shuffle=True,
        collate_fn=_collate_records,
    )
    dev_loader = None
    if config["data"].get("dev_label_dir"):
        try:
            dev_dataset = LabelRecordDataset(
                config["data"]["dev_label_dir"],
                feature_cache_dir=config["data"].get("dev_feature_cache_dir"),
            )
            dev_loader = DataLoader(
                dev_dataset,
                batch_size=int(config["data"].get("batch_size", 2)),
                shuffle=False,
                collate_fn=_collate_records,
            )
        except ValueError:
            dev_loader = None

    feature_config = FeatureConfig.from_raw(config.get("features", {}))
    feature_spec = FeatureSpec.from_feature_config(backbone.hidden_dim, feature_config)
    model = _build_model(config, feature_spec).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model={config.get('model', {}).get('type', 'gnn')} params={num_params:,}")
    if wandb_run is not None:
        wandb_run.summary["model/num_params"] = int(num_params)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("lr", 3e-4)),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )
    criterion = nn.BCEWithLogitsLoss()
    max_grad_norm = float(config["training"].get("max_grad_norm", 1.0))
    epochs = int(config["training"].get("epochs", 3))

    global_step = 0
    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        iterator = tqdm(train_loader, desc=f"train:{epoch}")
        for records in iterator:
            graphs, labels = records_to_graphs(records, backbone, feature_config, device)
            batched = batch_graphs(graphs)
            logits = model(batched.node_features, batched.edge_index, batched.edge_features)
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            losses.append(float(loss.item()))
            iterator.set_postfix(loss=f"{losses[-1]:.4f}")
            global_step += 1
            if wandb_run is not None:
                wandb_run.log({"train/loss_step": losses[-1], "epoch": epoch}, step=global_step)
        train_loss = sum(losses) / max(len(losses), 1)
        record = {"epoch": epoch, "train_loss": train_loss}
        dev_metrics: dict[str, float] = {}
        if dev_loader is not None:
            dev_metrics = evaluate_dev_metrics(model, dev_loader, backbone, feature_config, device)
            record.update(dev_metrics)
        logger.write(record)
        if wandb_run is not None:
            epoch_log = {"train/loss_epoch": train_loss, "epoch": epoch}
            for key, value in dev_metrics.items():
                epoch_log[f"dev/{key.replace('dev_', '')}"] = value
            wandb_run.log(epoch_log, step=global_step)

    checkpoint_path = run_dir / "conflict_model.pt"
    save_checkpoint(
        checkpoint_path,
        {
            "model_state": model.state_dict(),
            "model_config": config.get("model", {}),
            "feature_config": feature_config.__dict__,
            "feature_spec": {
                "hidden_dim": feature_spec.hidden_dim,
                "top_k": feature_spec.top_k,
                "distance_buckets": feature_spec.distance_buckets,
                "include_hidden": feature_spec.include_hidden,
                "include_scalar": feature_spec.include_scalar,
                "include_timestep": feature_spec.include_timestep,
                "include_position": feature_spec.include_position,
                "include_runs": feature_spec.include_runs,
                "include_edge_features": feature_spec.include_edge_features,
            },
            "input_node_dim": feature_spec.node_dim,
            "input_edge_dim": feature_spec.edge_dim,
        },
    )
    if wandb_run is not None:
        try:
            tracking = config.get("tracking", {}) or {}
            if bool(tracking.get("log_checkpoint", True)):
                import wandb

                artifact = wandb.Artifact(
                    name=f"conflict-model-{wandb_run.id}",
                    type="model",
                    metadata={"model_type": str(config.get("model", {}).get("type", "gnn"))},
                )
                artifact.add_file(str(checkpoint_path))
                wandb_run.log_artifact(artifact)
        except Exception as exc:  # noqa: BLE001 - artifact upload must not break training
            print(f"[tracking] wandb artifact upload failed ({exc})")
        finally:
            wandb_run.finish()
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], help="dotted=value overrides")
    args = parser.parse_args()
    config = load_config(args.config)
    for item in args.overrides:
        key, _, val = item.partition("=")
        set_dotted(config, key, parse_scalar(val))
    train(config)


if __name__ == "__main__":
    main()

