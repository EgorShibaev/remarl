"""Set-level oracle imitation: train a set scorer to pick a low-conflict unmask
set, comparing GNN message passing vs node pooling.

Offline: uses an existing labelset (labels with ``dependency_scores`` + feature
cache). For each masked example we build candidate sets + their oracle set cost,
then train a scorer to predict cost (MSE + ranking). The HEADLINE metric is the
true oracle conflict of the set the scorer selects (argmin predicted cost) on the
dev split - reported for ``pool`` and ``gnn`` plus confidence/oracle references.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.graph.features import FeatureConfig
from src.graph.graph_builder import build_full_graph_from_cached
from src.labels.candidate_sets import generate_candidate_sets
from src.labels.storage import LabelRecordDataset
from src.models.set_scorer import SetScorer, SetScorerConfig
from src.utils.checkpointing import save_checkpoint
from src.utils.config import load_config, parse_scalar, set_dotted
from src.utils.device import resolve_device
from src.utils.logging import make_run_dir, write_json
from src.utils.seed import set_seed
from src.utils.tracking import init_wandb


class PreparedRecord:
    __slots__ = ("node_features", "edge_index", "edge_features", "sets")

    def __init__(self, node_features, edge_index, edge_features, sets):
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_features = edge_features
        self.sets = sets  # list of (indices LongTensor, cost float)


def _induce_subgraph(node_features, edge_index, edge_features, idx):
    """Subgraph induced by sorted local indices ``idx`` (remapped to 0..m-1)."""
    n = node_features.shape[0]
    idx = idx.sort().values
    in_s = torch.zeros(n, dtype=torch.bool, device=node_features.device)
    in_s[idx] = True
    remap = torch.full((n,), -1, dtype=torch.long, device=node_features.device)
    remap[idx] = torch.arange(idx.numel(), device=node_features.device)
    if edge_index.numel() == 0:
        ei = edge_index.new_zeros((2, 0))
        ef = edge_features[:0]
    else:
        src, dst = edge_index
        mask = in_s[src] & in_s[dst]
        ei = torch.stack([remap[src[mask]], remap[dst[mask]]], dim=0)
        ef = edge_features[mask]
    return node_features.index_select(0, idx), ei, ef


def _prepare(records, feature_config, candidate_cfg, device, generator) -> list[PreparedRecord]:
    prepared: list[PreparedRecord] = []
    for record in records:
        masked_positions = record["masked_positions"].to(device)
        pairs = record["pairs"].to(device)
        if masked_positions.numel() < 3 or pairs.numel() == 0:
            continue
        cached = record.get("cached_features")
        if cached is None:
            continue
        graph = build_full_graph_from_cached(
            cached=cached,
            masked_positions=masked_positions,
            seq_len=int(record["attention_mask"].shape[0]),
            timestep=int(record.get("timestep", 0)),
            config=feature_config,
            pairs=pairs,
        )
        confidence = record["base_confidence"].to(device)
        cands = generate_candidate_sets(
            confidence=confidence,
            pairs=pairs,
            dependency_scores=record["dependency_scores"].to(device),
            budget_fraction=float(candidate_cfg.get("budget_fraction", 0.4)),
            num_random=int(candidate_cfg.get("num_random", 2)),
            conflict_theta=float(candidate_cfg.get("conflict_theta", 0.05)),
            generator=generator,
        )
        sets = [(c.indices.to(device), c.cost, c.name) for c in cands if c.n_observed >= 1]
        if len(sets) < 2:
            continue
        prepared.append(
            PreparedRecord(graph.node_features, graph.edge_index, graph.edge_features, sets)
        )
    return prepared


def _load_dagger_records(dirs, device) -> list[PreparedRecord]:
    """Load rollout records saved by collect_rollout_candidates into PreparedRecords."""
    out: list[PreparedRecord] = []
    for d in dirs:
        root = Path(d)
        for shard in sorted(root.glob("*.pt")):
            for rec in torch.load(shard, map_location="cpu"):
                sets = [
                    (idx.to(device), float(cost), name)
                    for idx, cost, name in rec["sets"]
                    if idx.numel() >= 1
                ]
                if len(sets) < 2:
                    continue
                out.append(
                    PreparedRecord(
                        rec["node_features"].to(device),
                        rec["edge_index"].to(device),
                        rec["edge_features"].to(device),
                        sets,
                    )
                )
    return out


def _predict_sets(model, rec: PreparedRecord) -> torch.Tensor:
    preds = []
    for idx, _cost, _name in rec.sets:
        nf, ei, ef = _induce_subgraph(rec.node_features, rec.edge_index, rec.edge_features, idx)
        preds.append(model(nf, ei, ef))
    return torch.stack(preds)


def _set_loss(preds: torch.Tensor, costs: torch.Tensor, train_cfg: dict) -> torch.Tensor:
    """Loss over one candidate pool. The scorer predicts set cost (lower=better);
    selection is argmin(pred), so the goodness score is s = -pred.

    Modes (``training.loss``):
    - ``mse_rank`` (default): MSE(pred, cost) + lambda * pairwise hinge (legacy).
    - ``listnet``  : listwise top-one cross-entropy between softmax(-cost/tau_t)
                     and softmax(-pred/tau) (+ optional mse_aux * MSE).
    - ``infonce``  : top-1 contrastive; positive(s) = argmin-cost candidate(s),
                     loss = -log sum_{i in argmin} softmax(-pred/tau)_i (tie-safe).
    """
    name = str(train_cfg.get("loss", "mse_rank"))
    if name == "mse_rank":
        lam = float(train_cfg.get("ranking_lambda", 1.0))
        margin = float(train_cfg.get("ranking_margin", 0.05))
        loss = nn.functional.mse_loss(preds, costs)
        m = costs.numel()
        ia, ib = torch.triu_indices(m, m, offset=1, device=preds.device)
        sign = torch.sign(costs[ib] - costs[ia])  # +1 if b costlier
        diff = (preds[ib] - preds[ia]) * sign
        return loss + lam * torch.relu(margin - diff).mean()

    tau = float(train_cfg.get("loss_temperature", 0.1))
    mse_aux = float(train_cfg.get("mse_aux", 0.0))
    s = -preds / tau
    log_p = torch.log_softmax(s, dim=0)
    if name == "listnet":
        tau_t = float(train_cfg.get("target_temperature", train_cfg.get("loss_temperature", 0.1)))
        target = torch.softmax(-costs / tau_t, dim=0)
        loss = -(target * log_p).sum()
    elif name == "infonce":
        pos = costs <= (costs.min() + 1e-9)  # tie-safe positive set
        loss = -torch.logsumexp(log_p[pos], dim=0)
    else:
        raise ValueError(f"Unknown training.loss: {name}")
    if mse_aux > 0.0:
        loss = loss + mse_aux * nn.functional.mse_loss(preds, costs)
    return loss


def _build_scorer_config(mode, input_dims, model_cfg) -> SetScorerConfig:
    aggs = model_cfg.get("pna_aggregators", ("mean", "max", "min", "std"))
    return SetScorerConfig(
        input_node_dim=input_dims[0],
        input_edge_dim=input_dims[1],
        node_dim=int(model_cfg.get("node_dim", 128)),
        edge_dim=int(model_cfg.get("edge_dim", 64)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        layers=int(model_cfg.get("layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        mode=mode,
        pna_aggregators=tuple(aggs),
        gat_heads=int(model_cfg.get("gat_heads", 4)),
        gat_topk=int(model_cfg.get("gat_topk", 0)),
        st_heads=int(model_cfg.get("st_heads", 4)),
        st_blocks=int(model_cfg.get("st_blocks", 1)),
    )


def _train_mode(mode, prepared, input_dims, model_cfg, train_cfg, device, seed):
    set_seed(seed)
    model = SetScorer(_build_scorer_config(mode, input_dims, model_cfg)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 5e-4)),
                            weight_decay=float(train_cfg.get("weight_decay", 0.01)))
    epochs = int(train_cfg.get("epochs", 5))
    loss_name = str(train_cfg.get("loss", "mse_rank"))
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(prepared))
        total = 0.0
        for ridx in order.tolist():
            rec = prepared[ridx]
            costs = torch.tensor([c for _i, c, _n in rec.sets], device=device)
            preds = _predict_sets(model, rec)
            loss = _set_loss(preds, costs, train_cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"[set-scorer:{mode}] loss={loss_name} epoch {epoch} avg={total / max(len(prepared),1):.4f}")
    return model


@torch.no_grad()
def _eval_mode(model, prepared, device) -> dict[str, float]:
    model.eval()
    selected, conf_ref, oracle_ref, random_ref = [], [], [], []
    all_pred, all_true = [], []
    for rec in prepared:
        preds = _predict_sets(model, rec)
        costs = [c for _i, c, _n in rec.sets]
        names = [n for _i, _c, n in rec.sets]
        all_pred.extend([float(p) for p in preds]); all_true.extend(costs)
        sel = int(torch.argmin(preds).item())
        selected.append(costs[sel])
        # references by candidate name
        by_name = {n: c for n, c in zip(names, costs)}
        if "confidence_topk" in by_name:
            conf_ref.append(by_name["confidence_topk"])
        if "oracle_greedy" in by_name:
            oracle_ref.append(by_name["oracle_greedy"])
        rnd = [c for n, c in zip(names, costs) if n.startswith("random")]
        if rnd:
            random_ref.append(sum(rnd) / len(rnd))

    def mean(x):
        return float(sum(x) / len(x)) if x else float("nan")

    out = {
        "selected_oracle_cost": mean(selected),
        "confidence_oracle_cost": mean(conf_ref),
        "oracle_greedy_cost": mean(oracle_ref),
        "random_oracle_cost": mean(random_ref),
        "n_records": len(prepared),
    }
    # Spearman (rank corr) pred vs true
    if len(all_pred) > 2:
        import numpy as np
        p = np.argsort(np.argsort(np.array(all_pred)))
        t = np.argsort(np.argsort(np.array(all_true)))
        out["spearman"] = float(np.corrcoef(p, t)[0, 1])
    return out


def train(config: dict[str, Any]) -> Path:
    device = resolve_device(config.get("device", "auto"))
    run_dir = make_run_dir(config.get("output_dir", "outputs/runs"), "set-scorer")
    write_json(run_dir / "resolved_config.json", config)
    seed = int(config.get("seed", 13))

    labelset = config["cell"]["labelset"] if "cell" in config else config["labelset"]
    root = Path("outputs/labelsets") / labelset
    feature_config = FeatureConfig.from_raw(config.get("features", {}))
    cand_cfg = config.get("candidate", {})
    max_train = int(config.get("data", {}).get("max_records", 8000))
    max_dev = int(config.get("data", {}).get("max_dev_records", 2000))

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    train_records = list(LabelRecordDataset(root / "labels" / "train", feature_cache_dir=root / "cache" / "train"))[:max_train]
    dev_records = list(LabelRecordDataset(root / "labels" / "dev", feature_cache_dir=root / "cache" / "dev"))[:max_dev]
    train_prep = _prepare(train_records, feature_config, cand_cfg, device, gen)
    dev_prep = _prepare(dev_records, feature_config, cand_cfg, device, gen)
    dagger_dirs = (config.get("dagger", {}) or {}).get("dirs", []) or []
    n_offline = len(train_prep)
    if dagger_dirs:
        extra = _load_dagger_records(dagger_dirs, device)
        train_prep.extend(extra)
        print(f"[set-scorer] DAgger aggregation: +{len(extra)} rollout records "
              f"from {len(dagger_dirs)} dir(s) (offline={n_offline})")
    if not train_prep or not dev_prep:
        raise ValueError(f"No usable records for labelset {labelset}")
    input_dims = (train_prep[0].node_features.shape[-1], train_prep[0].edge_features.shape[-1])
    print(f"[set-scorer] labelset={labelset} train={len(train_prep)} dev={len(dev_prep)} dims={input_dims}")

    wandb_run = init_wandb(config, run_dir, job_type="set-scorer",
                           extra_tags={"labelset": labelset, "experiment": "set-level"})
    model_cfg = config.get("model", {})
    save_ckpt = bool(config.get("save_checkpoint", True))
    # versioned subdir so S1 (set_scorer/) is preserved and DAgger rounds get
    # distinct checkpoints (set_scorer_d1, set_scorer_d2, ...).
    ckpt_dir = root / config.get("set_scorer_subdir", "set_scorer")
    loss_name = str(config.get("training", {}).get("loss", "mse_rank"))
    modes = list((config.get("set_scorer", {}) or {}).get("modes", ("pool", "gnn")))
    results: dict[str, Any] = {"labelset": labelset, "seed": seed, "loss": loss_name, "modes": {}}
    for mode in modes:
        model = _train_mode(mode, train_prep, input_dims, model_cfg,
                            config.get("training", {}), device, seed)
        metrics = _eval_mode(model, dev_prep, device)
        results["modes"][mode] = metrics
        print(f"[set-scorer:{mode}] selected_oracle_cost={metrics['selected_oracle_cost']:.4f} "
              f"(conf={metrics['confidence_oracle_cost']:.4f} oracle={metrics['oracle_greedy_cost']:.4f} "
              f"random={metrics['random_oracle_cost']:.4f}) spearman={metrics.get('spearman', float('nan')):.3f}")
        if save_ckpt:
            payload = {
                "model_state": model.state_dict(),
                "set_scorer_config": vars(_build_scorer_config(mode, input_dims, model_cfg)),
                "feature_config": config.get("features", {}),
                "input_node_dim": input_dims[0],
                "input_edge_dim": input_dims[1],
                "mode": mode,
                "labelset": labelset,
                "seed": seed,
                "loss": str(config.get("training", {}).get("loss", "mse_rank")),
            }
            save_checkpoint(run_dir / f"set-scorer-{mode}.pt", payload)
            save_checkpoint(ckpt_dir / f"{mode}.pt", payload)
        if wandb_run is not None:
            wandb_run.summary.update({f"{mode}/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})

    if "pool" in results["modes"] and "gnn" in results["modes"]:
        pool_c = results["modes"]["pool"]["selected_oracle_cost"]
        gnn_c = results["modes"]["gnn"]["selected_oracle_cost"]
        results["gnn_minus_pool_selected_cost"] = gnn_c - pool_c
        print(f"[set-scorer] VERDICT {labelset}: gnn-pool selected cost = {gnn_c - pool_c:+.4f} "
              f"(negative => GNN selects safer sets)")
    write_json(run_dir / "summary.json", results)
    if wandb_run is not None and "gnn_minus_pool_selected_cost" in results:
        wandb_run.summary["gnn_minus_pool_selected_cost"] = results["gnn_minus_pool_selected_cost"]
    if wandb_run is not None:
        wandb_run.finish()
    return run_dir


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
