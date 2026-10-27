"""DAgger-lite rollout collector: in-loop candidate relabeling.

The Pack S2 failure mode is a train/decode distribution mismatch: the set-scorer
trains on oracle-greedy / hard-negative candidate sets but at decode time ranks a
confidence+stochastic pool. This script closes that gap by rolling out the decoder
with the CURRENT policy, recording the candidate pools it actually visits, and
relabeling each candidate with the TRUE counterfactual oracle set cost. The
resulting records (same shape as ``PreparedRecord`` in the set-scorer trainer) are
aggregated with the offline labelset and used to retrain (DAgger).

Usage (see run_dagger.sh):
    python -m src.labels.collect_rollout_candidates --config <eval_setdecode>.yaml \
        --set dagger.policy_ckpt=<gnn.pt> dagger.out_dir=outputs/dagger/<ls>/round1 \
              data.split=train data.max_examples=64 seed=13
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.backbones.registry import build_backbone
from src.data.datasets import collate_tokenized, load_text_dataset, resolve_masking
from src.data.masking import apply_mask
from src.decoding.set_selector_decoder import (
    SetSelectorDecoderConfig,
    _induce_subgraph,
    build_edge_model_from_checkpoint,
    generate_inference_candidates,
    load_set_scorer_from_checkpoint,
    oracle_pool_matrix,
    set_cost,
)
from src.decoding.selection import node_scores
from src.graph.graph_builder import build_full_graph, edge_vector_to_matrix
from src.utils.config import load_config, parse_scalar, set_dotted
from src.utils.device import resolve_device
from src.utils.logging import write_json
from src.utils.seed import set_seed


def _decoder_config(decode_cfg: dict, oracle_cfg: dict) -> SetSelectorDecoderConfig:
    topk = oracle_cfg.get("topk_for_js")
    return SetSelectorDecoderConfig(
        max_steps=int(decode_cfg.get("max_steps", 8)),
        force_complete=bool(decode_cfg.get("force_complete", True)),
        fallback_top1=bool(decode_cfg.get("fallback_top1", True)),
        scoring="model",
        pool_size=int(decode_cfg.get("pool_size", 12)),
        budgets=tuple(float(b) for b in decode_cfg.get("set_budgets", (0.25, 0.4, 0.6, 0.8, 1.0))),
        conservative_sizes=tuple(int(s) for s in decode_cfg.get("conservative_sizes", (1, 2))),
        num_stochastic=int(decode_cfg.get("num_stochastic", 3)),
        stochastic_temperature=float(decode_cfg.get("stochastic_temperature", 0.5)),
        edge_thetas=tuple(float(t) for t in decode_cfg.get("edge_thetas", (0.3, 0.5))),
        tau=float(oracle_cfg.get("tau", 0.05)),
        topk_for_js=int(topk) if topk is not None else None,
        seed=int(decode_cfg.get("seed", 0)),
    )


@torch.no_grad()
def collect(config: dict[str, Any]) -> Path:
    device = resolve_device(config.get("device", "auto"))
    seed = int(config.get("seed", 13))
    set_seed(seed)

    dagger_cfg = config.get("dagger", {})
    policy_ckpt = dagger_cfg.get("policy_ckpt")
    if not policy_ckpt:
        raise ValueError("collect requires dagger.policy_ckpt")
    out_dir = Path(dagger_cfg.get("out_dir", "outputs/dagger/round"))
    out_dir.mkdir(parents=True, exist_ok=True)

    backbone = build_backbone(config["backbone"]).to(device)
    scorer, feature_config = load_set_scorer_from_checkpoint(policy_ckpt, device)
    sd = _decoder_config(config.get("decoding", {}), config.get("metrics", {}).get("oracle", {}) or {})

    # Optional edge-greedy candidate generator: align the rollout pool with the
    # decode-time pool (which also gets edge-greedy candidates when configured).
    edge_model = None
    edge_feature_config = None
    edge_ckpt = dagger_cfg.get("edge_model") or (config.get("checkpoint", {}) or {}).get("edge_model")
    if edge_ckpt:
        edge_model, edge_feature_config = build_edge_model_from_checkpoint(edge_ckpt, device)
        print(f"[dagger-collect] edge-greedy candidates from {edge_ckpt}")

    data_cfg = config["data"]
    dataset = load_text_dataset(
        tokenizer=backbone.tokenizer,
        data_cfg=data_cfg,
        split=data_cfg.get("split", "train"),
        max_examples=int(data_cfg.get("max_examples", 64)),
        max_length=int(data_cfg.get("max_length", 128)),
    )
    strategy, restrict_token_ids = resolve_masking(data_cfg, backbone.tokenizer)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_tokenized)
    special_ids = set(backbone.tokenizer.all_special_ids)
    mask_ratio = float(data_cfg.get("mask_ratio", 0.5))
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    records: list[dict[str, Any]] = []
    for _idx, batch in enumerate(tqdm(loader, desc=f"dagger-collect:{out_dir.name}")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        masked = apply_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_token_id=backbone.mask_token_id,
            mask_ratio=mask_ratio,
            special_token_ids=special_ids,
            strategy=strategy,
            restrict_token_ids=restrict_token_ids,
        )
        current = masked.input_ids_masked.clone()
        clean = masked.clean_input_ids
        amask = masked.attention_mask
        for step in range(sd.max_steps):
            positions = torch.nonzero(current[0].eq(backbone.mask_token_id), as_tuple=False).flatten()
            if positions.numel() == 0:
                break
            out = backbone.forward(current, amask, timestep=0)
            logits = out["logits"]
            if sd.force_complete and step == sd.max_steps - 1:
                current[0, positions] = logits[0, positions].argmax(dim=-1)
                break

            n_all = int(positions.numel())
            graph_full = build_full_graph(
                hidden_states=out["hidden_states"][0],
                logits=logits[0],
                masked_positions=positions,
                timestep=0,
                config=feature_config,
            )
            conf_all = graph_full.node_aux["confidence"]
            pool_size_eff = min(sd.pool_size, n_all)
            pool_local = torch.topk(conf_all, k=pool_size_eff).indices.sort().values
            pool_positions = positions[pool_local]
            graph = build_full_graph(
                hidden_states=out["hidden_states"][0],
                logits=logits[0],
                masked_positions=pool_positions,
                timestep=0,
                config=feature_config,
            )
            conf = graph.node_aux["confidence"]
            scores = node_scores(conf, graph.node_aux["margin"], graph.node_aux["entropy"])
            conflict_matrices: dict[str, torch.Tensor] | None = None
            if edge_model is not None:
                eg = graph
                if edge_feature_config is not feature_config:
                    eg = build_full_graph(
                        hidden_states=out["hidden_states"][0],
                        logits=logits[0],
                        masked_positions=pool_positions,
                        timestep=0,
                        config=edge_feature_config,
                    )
                edge_logits = edge_model(eg.node_features, eg.edge_index, eg.edge_features)
                conflict_matrices = {
                    "edge": edge_vector_to_matrix(
                        eg.edge_index, torch.sigmoid(edge_logits), num_nodes=int(pool_positions.numel())
                    )
                }
            candidates = generate_inference_candidates(
                confidence=conf, scores=scores, config=sd, conflict_matrices=conflict_matrices, generator=generator
            )
            if not candidates:
                candidates = [("top1", torch.topk(conf, k=1).indices)]

            # relabel the visited candidate pool with the true oracle set cost
            oracle_mat = oracle_pool_matrix(
                backbone,
                row_input_ids=current,
                attention_mask_row=amask,
                clean_row=clean,
                pool_positions=pool_positions,
                base_logits_full=logits[0],
                tau=sd.tau,
                topk_for_js=sd.topk_for_js,
                timestep=0,
            )
            if pool_positions.numel() >= 2 and len(candidates) >= 2:
                sets = [
                    (idx_l.detach().cpu(), set_cost(idx_l, oracle_mat), name)
                    for name, idx_l in candidates
                ]
                records.append(
                    {
                        "node_features": graph.node_features.detach().cpu(),
                        "edge_index": graph.edge_index.detach().cpu(),
                        "edge_features": graph.edge_features.detach().cpu(),
                        "sets": sets,
                    }
                )

            # advance the state with the current policy (model argmin predicted cost)
            preds = []
            for _name, idx_l in candidates:
                nf, ei, ef = _induce_subgraph(graph.node_features, graph.edge_index, graph.edge_features, idx_l)
                preds.append(float(scorer(nf, ei, ef)))
            chosen = candidates[int(min(range(len(preds)), key=lambda k: preds[k]))][1]
            if chosen.numel() == 0:
                chosen = torch.topk(conf, k=1).indices
            sel_positions = pool_positions[chosen]
            current[0, sel_positions] = logits[0, sel_positions].argmax(dim=-1)

    shard_path = out_dir / "shard_0000.pt"
    torch.save(records, shard_path)
    write_json(
        out_dir / "meta.json",
        {
            "policy_ckpt": str(policy_ckpt),
            "edge_model": str(edge_ckpt) if edge_ckpt else None,
            "n_records": len(records),
            "seed": seed,
            "dataset": data_cfg.get("dataset_name"),
            "mask_ratio": mask_ratio,
            "pool_size": sd.pool_size,
            "max_steps": sd.max_steps,
        },
    )
    print(f"[dagger-collect] wrote {len(records)} records -> {shard_path}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], help="dotted=value overrides")
    args = parser.parse_args()
    config = load_config(args.config)
    for item in args.overrides:
        key, _, val = item.partition("=")
        set_dotted(config, key, parse_scalar(val))
    collect(config)


if __name__ == "__main__":
    main()
