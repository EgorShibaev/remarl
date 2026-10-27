"""GNN-failure diagnostics for set-level decoding (CGSD reframe).

Separates two questions the confirmatory sweep cannot answer on its own:

  (C1) *Scorer quality* — on the SAME pool-generated states with the SAME candidate
       pool, does ``scorer_gnn`` pick worse sets than ``scorer_pool``? If gnn ~= pool
       here, the gnn scorer ranks fine; the gain/loss is not a ranking problem.

  (B / C2) *Policy dynamics* — roll out each policy on its OWN states and track
       ``pool_min_oracle_cost`` (the best achievable oracle cost in the step's
       candidate pool) over time. If gnn's local regret ~= 0 but its trajectory's
       pool_min rises faster than pool's, gnn's early choices degrade the FUTURE
       candidate pool: a trajectory/dynamics failure, not local ranking.

Plus (A) the first step where pool and gnn diverge (same seeded mask), and a causal
**first-divergence outcome delta**: immediate oracle cost + pool_min 1-2 steps later.

Both scorers share the SAME (optionally edge-enriched) candidate pool, matching the
CGSD decode condition. Run via run_diagnose.sh; outputs a JSON + console summary.
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


class _StateStepper:
    """Shared per-step computation: pool, candidates, oracle matrix. Independent of
    which scorer advances the trajectory, so two policies see identical state logic."""

    def __init__(self, backbone, feature_config, edge_model, edge_feature_config, sd, generator):
        self.backbone = backbone
        self.feature_config = feature_config
        self.edge_model = edge_model
        self.edge_feature_config = edge_feature_config
        self.sd = sd
        self.generator = generator

    @torch.no_grad()
    def step(self, current, amask, clean):
        bb = self.backbone
        positions = torch.nonzero(current[0].eq(bb.mask_token_id), as_tuple=False).flatten()
        if positions.numel() == 0:
            return None
        out = bb.forward(current, amask, timestep=0)
        logits = out["logits"]
        n_all = int(positions.numel())
        graph_full = build_full_graph(
            hidden_states=out["hidden_states"][0], logits=logits[0],
            masked_positions=positions, timestep=0, config=self.feature_config,
        )
        conf_all = graph_full.node_aux["confidence"]
        pool_size_eff = min(self.sd.pool_size, n_all)
        pool_local = torch.topk(conf_all, k=pool_size_eff).indices.sort().values
        pool_positions = positions[pool_local]
        graph = build_full_graph(
            hidden_states=out["hidden_states"][0], logits=logits[0],
            masked_positions=pool_positions, timestep=0, config=self.feature_config,
        )
        conf = graph.node_aux["confidence"]
        scores = node_scores(conf, graph.node_aux["margin"], graph.node_aux["entropy"])
        conflict_matrices = None
        if self.edge_model is not None:
            eg = graph
            if self.edge_feature_config is not self.feature_config:
                eg = build_full_graph(
                    hidden_states=out["hidden_states"][0], logits=logits[0],
                    masked_positions=pool_positions, timestep=0, config=self.edge_feature_config,
                )
            edge_logits = self.edge_model(eg.node_features, eg.edge_index, eg.edge_features)
            conflict_matrices = {
                "edge": edge_vector_to_matrix(
                    eg.edge_index, torch.sigmoid(edge_logits), num_nodes=int(pool_positions.numel())
                )
            }
        candidates = generate_inference_candidates(
            confidence=conf, scores=scores, config=self.sd,
            conflict_matrices=conflict_matrices, generator=self.generator,
        )
        if not candidates:
            candidates = [("top1", torch.topk(conf, k=1).indices)]
        oracle_mat = oracle_pool_matrix(
            bb, row_input_ids=current, attention_mask_row=amask, clean_row=clean,
            pool_positions=pool_positions, base_logits_full=logits[0],
            tau=self.sd.tau, topk_for_js=self.sd.topk_for_js, timestep=0,
        )
        return {
            "logits": logits, "graph": graph, "conf": conf,
            "pool_positions": pool_positions, "candidates": candidates, "oracle_mat": oracle_mat,
        }

    def choose(self, scorer, graph, candidates):
        preds = []
        for _name, idx_l in candidates:
            nf, ei, ef = _induce_subgraph(graph.node_features, graph.edge_index, graph.edge_features, idx_l)
            preds.append(float(scorer(nf, ei, ef)))
        return candidates[int(min(range(len(preds)), key=lambda k: preds[k]))][1]


@torch.no_grad()
def _rollout(stepper, scorer, masked, other_scorer=None) -> list[dict]:
    """Roll out ``scorer`` on its own states. If ``other_scorer`` is given, also
    record what it would pick on the SAME candidate pool (C1 same-state scoring)."""
    current = masked.input_ids_masked.clone()
    clean = masked.clean_input_ids
    amask = masked.attention_mask
    trace: list[dict] = []
    for step in range(stepper.sd.max_steps):
        st = stepper.step(current, amask, clean)
        if st is None:
            break
        cands, omat, graph = st["candidates"], st["oracle_mat"], st["graph"]
        pool_pos, conf, logits = st["pool_positions"], st["conf"], st["logits"]
        if stepper.sd.force_complete and step == stepper.sd.max_steps - 1:
            positions = torch.nonzero(current[0].eq(stepper.backbone.mask_token_id), as_tuple=False).flatten()
            current[0, positions] = logits[0, positions].argmax(dim=-1)
            break

        chosen = stepper.choose(scorer, graph, cands)
        if chosen.numel() == 0:
            chosen = torch.topk(conf, k=1).indices
        pool_min = min(set_cost(idx_l, omat) for _n, idx_l in cands) if len(cands) else 0.0
        rec = {
            "step": step,
            "selected_positions": sorted(int(p) for p in pool_pos[chosen].detach().cpu()),
            "chosen_cost": float(set_cost(chosen, omat)),
            "pool_min_oracle_cost": float(pool_min),
            "pool_size": int(pool_pos.numel()),
            "n_candidates": len(cands),
        }
        if other_scorer is not None:
            other_chosen = stepper.choose(other_scorer, graph, cands)
            if other_chosen.numel() == 0:
                other_chosen = torch.topk(conf, k=1).indices
            rec["other_chosen_cost"] = float(set_cost(other_chosen, omat))
            rec["other_selected_positions"] = sorted(int(p) for p in pool_pos[other_chosen].detach().cpu())
        trace.append(rec)

        sel_positions = pool_pos[chosen]
        current[0, sel_positions] = logits[0, sel_positions].argmax(dim=-1)
    return trace


def _first_divergence(pool_trace: list[dict], gnn_trace: list[dict]) -> int | None:
    for k in range(min(len(pool_trace), len(gnn_trace))):
        if pool_trace[k]["selected_positions"] != gnn_trace[k]["selected_positions"]:
            return k
    return None


@torch.no_grad()
def diagnose(config: dict[str, Any]) -> Path:
    device = resolve_device(config.get("device", "auto"))
    seed = int(config.get("seed", 13))
    set_seed(seed)
    out_dir = Path(config.get("output_dir", "outputs/diagnose")) / f"diag-{config.get('data',{}).get('dataset_name','x').replace('/','_')}-s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    backbone = build_backbone(config["backbone"]).to(device)
    ckpt = config.get("checkpoint", {}) or {}
    pool_scorer, feature_config = load_set_scorer_from_checkpoint(ckpt["set_scorer_pool"], device)
    gnn_scorer, _ = load_set_scorer_from_checkpoint(ckpt["set_scorer_gnn"], device)
    edge_model = edge_feature_config = None
    if ckpt.get("edge_model"):
        edge_model, edge_feature_config = build_edge_model_from_checkpoint(ckpt["edge_model"], device)
        print(f"[diagnose] edge-enriched pool from {ckpt['edge_model']}")

    sd = _decoder_config(config.get("decoding", {}), config.get("metrics", {}).get("oracle", {}) or {})
    stepper = _StateStepper(backbone, feature_config, edge_model, edge_feature_config, sd, torch.Generator(device=device).manual_seed(seed))

    data_cfg = config["data"]
    dataset = load_text_dataset(
        tokenizer=backbone.tokenizer, data_cfg=data_cfg,
        split=data_cfg.get("split", "validation"),
        max_examples=int(data_cfg.get("max_examples", 32)),
        max_length=int(data_cfg.get("max_length", 128)),
    )
    strategy, restrict_token_ids = resolve_masking(data_cfg, backbone.tokenizer)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_tokenized)
    special_ids = set(backbone.tokenizer.all_special_ids)
    mask_ratio = float(data_cfg.get("mask_ratio", 0.45))

    per_example: list[dict] = []
    for idx, batch in enumerate(tqdm(loader, desc=f"diagnose:{out_dir.name}")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        masked = apply_mask(
            input_ids=input_ids, attention_mask=attention_mask,
            mask_token_id=backbone.mask_token_id, mask_ratio=mask_ratio,
            special_token_ids=special_ids, strategy=strategy, restrict_token_ids=restrict_token_ids,
        )
        # C1 lives on the pool policy's states; B/C2 + A use each policy's own trajectory.
        pool_trace = _rollout(stepper, pool_scorer, masked, other_scorer=gnn_scorer)
        gnn_trace = _rollout(stepper, gnn_scorer, masked)
        div = _first_divergence(pool_trace, gnn_trace)
        rec = {"example_id": idx, "div_step": div, "pool_trace": pool_trace, "gnn_trace": gnn_trace}
        if div is not None and div < min(len(pool_trace), len(gnn_trace)):
            rec["immediate_delta"] = gnn_trace[div]["chosen_cost"] - pool_trace[div]["chosen_cost"]
            for ahead in (1, 2):
                k = div + ahead
                if k < len(pool_trace) and k < len(gnn_trace):
                    rec[f"future_poolmin_delta_+{ahead}"] = (
                        gnn_trace[k]["pool_min_oracle_cost"] - pool_trace[k]["pool_min_oracle_cost"]
                    )
        per_example.append(rec)

    summary = _summarize(per_example)
    write_json(out_dir / "diagnose.json", {"config_data": data_cfg, "seed": seed, "summary": summary, "per_example": per_example})
    _print_summary(out_dir, summary)
    return out_dir


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if isinstance(x, (int, float))]
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _summarize(per_example: list[dict]) -> dict:
    # C1: same-state scoring on pool-states (chosen by pool vs counterfactual gnn).
    c1_pool, c1_gnn = [], []
    # B/C2: pool_min_oracle_cost by step index for each policy.
    by_step_pool: dict[int, list[float]] = {}
    by_step_gnn: dict[int, list[float]] = {}
    div_steps, imm, fut1, fut2 = [], [], [], []
    for e in per_example:
        for r in e["pool_trace"]:
            if "other_chosen_cost" in r:
                c1_pool.append(r["chosen_cost"])
                c1_gnn.append(r["other_chosen_cost"])
            by_step_pool.setdefault(r["step"], []).append(r["pool_min_oracle_cost"])
        for r in e["gnn_trace"]:
            by_step_gnn.setdefault(r["step"], []).append(r["pool_min_oracle_cost"])
        if e.get("div_step") is not None:
            div_steps.append(e["div_step"])
        if "immediate_delta" in e:
            imm.append(e["immediate_delta"])
        if "future_poolmin_delta_+1" in e:
            fut1.append(e["future_poolmin_delta_+1"])
        if "future_poolmin_delta_+2" in e:
            fut2.append(e["future_poolmin_delta_+2"])
    steps = sorted(set(by_step_pool) | set(by_step_gnn))
    return {
        "n_examples": len(per_example),
        "C1_same_state": {
            "mean_chosen_cost_pool": _mean(c1_pool),
            "mean_chosen_cost_gnn_on_pool_states": _mean(c1_gnn),
            "mean_delta_gnn_minus_pool": _mean([g - p for g, p in zip(c1_gnn, c1_pool)]),
            "n_steps": len(c1_pool),
        },
        "BC2_trajectory_pool_min_by_step": {
            "pool": {str(s): _mean(by_step_pool.get(s, [])) for s in steps},
            "gnn": {str(s): _mean(by_step_gnn.get(s, [])) for s in steps},
        },
        "A_divergence": {
            "mean_first_divergence_step": _mean([float(d) for d in div_steps]),
            "n_diverged": len(div_steps),
        },
        "first_divergence_delta": {
            "immediate_oracle_cost_gnn_minus_pool": _mean(imm),
            "future_poolmin_delta_+1": _mean(fut1),
            "future_poolmin_delta_+2": _mean(fut2),
        },
    }


def _print_summary(out_dir: Path, s: dict) -> None:
    c1 = s["C1_same_state"]
    print(f"\n=== diagnose summary ({out_dir.name}, n={s['n_examples']}) ===")
    print(f"[C1 same-state] chosen oracle cost  pool={c1['mean_chosen_cost_pool']:.4f}  "
          f"gnn-on-pool-states={c1['mean_chosen_cost_gnn_on_pool_states']:.4f}  "
          f"delta(gnn-pool)={c1['mean_delta_gnn_minus_pool']:.4f}  (n_steps={c1['n_steps']})")
    print("  -> if delta~=0: gnn scorer ranks as well as pool on identical states (NOT a scorer problem)")
    bc = s["BC2_trajectory_pool_min_by_step"]
    print("[B/C2 trajectory] pool_min_oracle_cost by step (own trajectories):")
    print("  step:   " + " ".join(f"{k:>7s}" for k in bc["pool"]))
    print("  pool:   " + " ".join(f"{bc['pool'][k]:7.4f}" for k in bc["pool"]))
    print("  gnn :   " + " ".join(f"{bc['gnn'].get(k, float('nan')):7.4f}" for k in bc["pool"]))
    print("  -> if gnn rises above pool over time: gnn drives into harder future pools (dynamics failure)")
    a = s["A_divergence"]
    print(f"[A divergence] mean first-divergence step={a['mean_first_divergence_step']:.2f} (n_diverged={a['n_diverged']})")
    fd = s["first_divergence_delta"]
    print(f"[first-div delta] immediate(gnn-pool)={fd['immediate_oracle_cost_gnn_minus_pool']:.4f}  "
          f"future_poolmin +1={fd['future_poolmin_delta_+1']:.4f}  +2={fd['future_poolmin_delta_+2']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], help="dotted=value overrides")
    args = parser.parse_args()
    config = load_config(args.config)
    for item in args.overrides:
        key, _, val = item.partition("=")
        set_dotted(config, key, parse_scalar(val))
    diagnose(config)


if __name__ == "__main__":
    main()
