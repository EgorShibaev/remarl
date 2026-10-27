"""In-loop set-level selector decoder (Pack S2).

The set-level scorer trained offline (``SetScorer``, pool vs gnn) is lifted into a
real decode loop on a masked-diffusion backbone:

    masked seq -> backbone forward -> build a small candidate pool of unmask sets
    -> score each candidate set -> unmask the best set -> repeat.

The scorer never searches all subsets; it ranks a small inference-only candidate
pool (confidence top-k at several budgets, conservative small sets, stochastic
perturbations, and -- when available -- edge/attention conflict-greedy sets). This
matches the offline training setup and is cheap.

Two scoring backends share the exact same candidate pool so they are directly
comparable:

- ``model``  : a trained ``SetScorer`` (``pool`` or ``gnn``) predicts set cost;
- ``oracle`` : the TRUE counterfactual oracle set cost (upper bound over the pool).

When ``clean_input_ids`` is available we also log **candidate oracle regret** =
oracle_cost(chosen) - min over the candidate pool, which separates a bad scorer
from a bad candidate generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from src.backbones.base import DiffusionBackbone
from src.decoding.attention_graph_decoder import attention_conflict_matrix
from src.decoding.confidence_decoder import DecodeResult, DecodeStepTrace
from src.decoding.selection import (
    greedy_conflict_aware_selection,
    node_scores,
    stochastic_conflict_aware_selection,
)
from src.eval.metrics import MetricsHook, StepContext, run_metrics_hooks
from src.graph.features import FeatureConfig, FeatureSpec
from src.graph.graph_builder import build_full_graph, edge_vector_to_matrix
from src.labels.dependency_scores import js_divergence_from_logits
from src.models.conflict_gnn import ConflictGNN, ConflictGNNConfig
from src.models.pairwise_mlp import PairwiseMLP, PairwiseMLPConfig
from src.models.set_scorer import SetScorer, SetScorerConfig
from src.utils.checkpointing import load_checkpoint


@dataclass
class SetSelectorDecoderConfig:
    max_steps: int = 16
    force_complete: bool = True
    fallback_top1: bool = True
    scoring: str = "model"  # "model" (use set scorer) | "oracle" (true set cost)
    pool_size: int = 16  # nodes considered for candidate generation / scoring per step
    budgets: tuple[float, ...] = (0.25, 0.4, 0.6, 0.8, 1.0)
    conservative_sizes: tuple[int, ...] = (1, 2)
    num_stochastic: int = 3
    stochastic_temperature: float = 0.5
    edge_thetas: tuple[float, ...] = (0.3, 0.5)
    # oracle / regret
    tau: float = 0.05
    topk_for_js: int | None = 128
    compute_regret: bool = False
    seed: int = 0


def load_set_scorer_from_checkpoint(
    path: str, device: torch.device
) -> tuple[SetScorer, FeatureConfig]:
    ckpt = load_checkpoint(path, map_location=device)
    feature_config = FeatureConfig.from_raw(ckpt.get("feature_config", {}))
    cfg_raw = dict(ckpt["set_scorer_config"])
    model = SetScorer(SetScorerConfig(**cfg_raw)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, feature_config


def build_edge_model_from_checkpoint(
    path: str, device: torch.device
) -> tuple[nn.Module, FeatureConfig]:
    """Load a trained edge conflict model (pairwise_mlp or gnn) for candidate gen."""
    ckpt = load_checkpoint(path, map_location=device)
    model_cfg = ckpt.get("model_config", {})
    feature_config = FeatureConfig.from_raw(ckpt.get("feature_config", {}))
    spec_raw = ckpt.get("feature_spec")
    if spec_raw is not None:
        FeatureSpec.from_feature_config(int(spec_raw["hidden_dim"]), feature_config)
    common = {
        "input_node_dim": int(ckpt["input_node_dim"]),
        "input_edge_dim": int(ckpt["input_edge_dim"]),
        "node_dim": int(model_cfg.get("node_dim", 256)),
        "edge_dim": int(model_cfg.get("edge_dim", 64)),
        "hidden_dim": int(model_cfg.get("hidden_dim", 256)),
        "dropout": float(model_cfg.get("dropout", 0.1)),
    }
    if model_cfg.get("type", "gnn") == "pairwise_mlp":
        model: nn.Module = PairwiseMLP(PairwiseMLPConfig(**common))
    else:
        model = ConflictGNN(ConflictGNNConfig(**common, layers=int(model_cfg.get("layers", 2))))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, feature_config


def _induce_subgraph(node_features, edge_index, edge_features, idx):
    """Subgraph induced by local indices ``idx`` (remapped to 0..m-1)."""
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


@torch.no_grad()
def oracle_pool_matrix(
    backbone: DiffusionBackbone,
    row_input_ids: torch.Tensor,  # [1, seq]
    attention_mask_row: torch.Tensor,  # [1, seq]
    clean_row: torch.Tensor,  # [1, seq]
    pool_positions: torch.Tensor,  # absolute positions [m]
    base_logits_full: torch.Tensor,  # [seq, vocab]
    tau: float,
    topk_for_js: int | None,
    timestep: int = 0,
) -> torch.Tensor:
    """[m, m] symmetric oracle conflict (1.0 if counterfactual JS > tau).

    Reveal the clean token at each pool position and measure how much every other
    pool position's distribution shifts (JS divergence). Shared by the in-loop
    decoder (oracle scoring / regret) and the DAgger rollout relabeling.
    """
    m = int(pool_positions.numel())
    matrix = base_logits_full.new_zeros((m, m))
    for i in range(m):
        anchor = int(pool_positions[i].item())
        cf_ids = row_input_ids.clone()
        cf_ids[0, anchor] = clean_row[0, anchor]
        cf_out = backbone.forward(cf_ids, attention_mask_row, timestep=timestep)
        cf_logits = cf_out["logits"][0]
        for j in range(m):
            if j == i:
                continue
            tpos = int(pool_positions[j].item())
            js = js_divergence_from_logits(
                base_logits_full[tpos].unsqueeze(0),
                cf_logits[tpos].unsqueeze(0),
                topk=topk_for_js,
            )
            if float(js.item()) > tau:
                matrix[i, j] = 1.0
                matrix[j, i] = 1.0
    return matrix


def set_cost(idx_local: torch.Tensor, oracle_matrix: torch.Tensor) -> float:
    """Mean oracle conflict over intra-set pairs (0 for singletons)."""
    m = int(idx_local.numel())
    if m < 2:
        return 0.0
    sub = oracle_matrix.index_select(0, idx_local).index_select(1, idx_local)
    iu = torch.triu_indices(m, m, offset=1, device=sub.device)
    pairs = sub[iu[0], iu[1]]
    return float(pairs.mean().item()) if pairs.numel() else 0.0


def _dedup(cands: list[tuple[str, torch.Tensor]]) -> list[tuple[str, torch.Tensor]]:
    seen: set[frozenset[int]] = set()
    out: list[tuple[str, torch.Tensor]] = []
    for name, idx in cands:
        if idx.numel() < 1:
            continue
        key = frozenset(int(x) for x in idx.detach().cpu())
        if key in seen:
            continue
        seen.add(key)
        out.append((name, idx.sort().values))
    return out


def generate_inference_candidates(
    confidence: torch.Tensor,
    scores: torch.Tensor,
    config: SetSelectorDecoderConfig,
    conflict_matrices: dict[str, torch.Tensor] | None,
    generator: torch.Generator | None,
) -> list[tuple[str, torch.Tensor]]:
    """Candidate unmask sets over a pool of ``n`` nodes, inference signals only."""
    n = int(confidence.numel())
    device = confidence.device
    cands: list[tuple[str, torch.Tensor]] = []
    if n == 0:
        return cands
    order = torch.argsort(scores, descending=True)
    # confidence/score top-k at several budgets
    for b in config.budgets:
        k = max(1, min(n, round(b * n)))
        cands.append((f"conf_b{b:.2f}", order[:k]))
    # conservative small sets
    for s in config.conservative_sizes:
        k = max(1, min(n, s))
        cands.append((f"cons_{s}", order[:k]))
    # edge/attention conflict-greedy at several thetas
    for mname, matrix in (conflict_matrices or {}).items():
        if matrix is None or matrix.numel() == 0:
            continue
        for theta in config.edge_thetas:
            sel = greedy_conflict_aware_selection(
                scores=scores, conflict_prob=matrix, budget=n, conflict_theta=theta
            ).selected_local
            if sel.numel() >= 1:
                cands.append((f"{mname}_greedy_t{theta:.2f}", sel))
    # stochastic perturbations around the top (perturb visit order via temperature)
    zero = torch.zeros((n, n), device=device)
    base_budget = max(2, min(n, round(0.6 * n)))
    for r in range(config.num_stochastic):
        sel = stochastic_conflict_aware_selection(
            scores=scores,
            conflict_prob=zero,
            budget=base_budget,
            conflict_theta=1.0,
            temperature=config.stochastic_temperature,
            generator=generator,
        ).selected_local
        if sel.numel() >= 1:
            cands.append((f"stoch_{r}", sel))
    return _dedup(cands)


class SetSelectorDecoder:
    def __init__(
        self,
        backbone: DiffusionBackbone,
        config: SetSelectorDecoderConfig,
        set_scorer: nn.Module | None = None,
        feature_config: FeatureConfig | None = None,
        edge_model: nn.Module | None = None,
        edge_feature_config: FeatureConfig | None = None,
        use_attention: bool = False,
        metrics_hooks: list[MetricsHook] | None = None,
    ) -> None:
        self.backbone = backbone
        self.config = config
        self.set_scorer = set_scorer
        self.feature_config = feature_config or FeatureConfig()
        self.edge_model = edge_model
        self.edge_feature_config = edge_feature_config or self.feature_config
        self.use_attention = use_attention
        self.metrics_hooks: list[MetricsHook] = list(metrics_hooks or [])
        if config.scoring == "model" and set_scorer is None:
            raise ValueError("scoring='model' requires a set_scorer")

    @torch.no_grad()
    def _oracle_pool_matrix(
        self,
        row_input_ids: torch.Tensor,
        attention_mask_row: torch.Tensor,
        clean_row: torch.Tensor,
        pool_positions: torch.Tensor,
        base_logits_full: torch.Tensor,
        timestep: int,
    ) -> torch.Tensor:
        return oracle_pool_matrix(
            self.backbone,
            row_input_ids=row_input_ids,
            attention_mask_row=attention_mask_row,
            clean_row=clean_row,
            pool_positions=pool_positions,
            base_logits_full=base_logits_full,
            tau=self.config.tau,
            topk_for_js=self.config.topk_for_js,
            timestep=timestep,
        )

    _set_cost = staticmethod(set_cost)

    @torch.no_grad()
    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        timestep: int = 0,
        clean_input_ids: torch.Tensor | None = None,
    ) -> DecodeResult:
        cfg = self.config
        current = input_ids.clone()
        total_initial = int(current.eq(self.backbone.mask_token_id).sum().item())
        generator = torch.Generator(device=current.device)
        generator.manual_seed(int(cfg.seed))
        traces: list[DecodeStepTrace] = []
        need_attn = self.use_attention
        for step in range(cfg.max_steps):
            masked_by_row = [
                torch.nonzero(row.eq(self.backbone.mask_token_id), as_tuple=False).flatten()
                for row in current
            ]
            remaining = sum(int(pos.numel()) for pos in masked_by_row)
            if remaining == 0:
                return DecodeResult(current, step, traces)
            out = self.backbone.forward(
                current, attention_mask, timestep=timestep, output_attentions=need_attn
            )
            logits = out["logits"]
            attentions = out.get("attentions") if need_attn else None
            input_ids_before = current.clone()
            last_step = cfg.force_complete and step == cfg.max_steps - 1

            selected_by_row: list[list[int]] = []
            selected_positions_per_row: list[torch.Tensor] = []
            selected_local_per_row: list[torch.Tensor] = []
            conflict_matrix_per_row: list[torch.Tensor | None] = []
            selected_confidences: list[torch.Tensor] = []
            regrets: list[float] = []
            chosen_costs: list[float] = []
            for row, positions in enumerate(masked_by_row):
                if positions.numel() == 0:
                    selected_by_row.append([])
                    selected_positions_per_row.append(positions)
                    selected_local_per_row.append(positions)
                    conflict_matrix_per_row.append(None)
                    continue
                if last_step:
                    # finish everything that is left
                    row_logits = logits[row, positions]
                    top_tokens = row_logits.argmax(dim=-1)
                    current[row, positions] = top_tokens
                    selected_by_row.append([int(x) for x in positions.detach().cpu()])
                    selected_positions_per_row.append(positions)
                    selected_local_per_row.append(torch.arange(positions.numel(), device=positions.device))
                    conflict_matrix_per_row.append(None)
                    continue

                n_all = int(positions.numel())
                graph_full = build_full_graph(
                    hidden_states=out["hidden_states"][row],
                    logits=logits[row],
                    masked_positions=positions,
                    timestep=timestep,
                    config=self.feature_config,
                )
                confidence_all = graph_full.node_aux["confidence"]
                pool_size = min(cfg.pool_size, n_all)
                pool_local = torch.topk(confidence_all, k=pool_size).indices.sort().values
                pool_positions = positions[pool_local]

                graph = build_full_graph(
                    hidden_states=out["hidden_states"][row],
                    logits=logits[row],
                    masked_positions=pool_positions,
                    timestep=timestep,
                    config=self.feature_config,
                )
                conf = graph.node_aux["confidence"]
                margin = graph.node_aux["margin"]
                entropy = graph.node_aux["entropy"]
                scores = node_scores(conf, margin, entropy)

                conflict_matrices: dict[str, torch.Tensor] = {}
                if self.edge_model is not None:
                    eg = graph
                    if self.edge_feature_config is not self.feature_config:
                        eg = build_full_graph(
                            hidden_states=out["hidden_states"][row],
                            logits=logits[row],
                            masked_positions=pool_positions,
                            timestep=timestep,
                            config=self.edge_feature_config,
                        )
                    edge_logits = self.edge_model(eg.node_features, eg.edge_index, eg.edge_features)
                    conflict_matrices["edge"] = edge_vector_to_matrix(
                        eg.edge_index, torch.sigmoid(edge_logits), num_nodes=pool_size
                    )
                if attentions is not None:
                    row_attn = tuple(layer[row : row + 1] for layer in attentions)
                    conflict_matrices["attn"] = attention_conflict_matrix(
                        row_attn, pool_positions, "all"
                    )

                candidates = generate_inference_candidates(
                    confidence=conf,
                    scores=scores,
                    config=cfg,
                    conflict_matrices=conflict_matrices,
                    generator=generator,
                )
                if not candidates:
                    candidates = [("top1", torch.topk(conf, k=1).indices)]

                # oracle matrix over the pool when needed (oracle scoring or regret)
                need_oracle = cfg.scoring == "oracle" or (
                    cfg.compute_regret and clean_input_ids is not None
                )
                oracle_matrix = None
                if need_oracle and clean_input_ids is not None:
                    oracle_matrix = self._oracle_pool_matrix(
                        row_input_ids=current[row : row + 1],
                        attention_mask_row=attention_mask[row : row + 1],
                        clean_row=clean_input_ids[row : row + 1],
                        pool_positions=pool_positions,
                        base_logits_full=logits[row],
                        timestep=timestep,
                    )

                if cfg.scoring == "oracle":
                    costs = [self._set_cost(idx, oracle_matrix) for _n, idx in candidates]
                    chosen_i = int(min(range(len(costs)), key=lambda k: costs[k]))
                else:
                    preds = []
                    for _name, idx in candidates:
                        nf, ei, ef = _induce_subgraph(
                            graph.node_features, graph.edge_index, graph.edge_features, idx
                        )
                        preds.append(float(self.set_scorer(nf, ei, ef)))
                    chosen_i = int(min(range(len(preds)), key=lambda k: preds[k]))

                chosen_pool_local = candidates[chosen_i][1]
                if chosen_pool_local.numel() == 0 and cfg.fallback_top1:
                    chosen_pool_local = torch.topk(conf, k=1).indices

                if oracle_matrix is not None:
                    all_costs = [self._set_cost(idx, oracle_matrix) for _n, idx in candidates]
                    chosen_cost = self._set_cost(chosen_pool_local, oracle_matrix)
                    chosen_costs.append(chosen_cost)
                    regrets.append(chosen_cost - min(all_costs))

                selected_positions = pool_positions[chosen_pool_local]
                top_tokens = logits[row, selected_positions].argmax(dim=-1)
                current[row, selected_positions] = top_tokens
                # map chosen pool-local -> full-masked-local for hooks / metrics
                chosen_full_local = pool_local[chosen_pool_local]
                selected_by_row.append([int(x) for x in selected_positions.detach().cpu()])
                selected_positions_per_row.append(selected_positions)
                selected_local_per_row.append(chosen_full_local)
                conflict_matrix_per_row.append(None)
                if chosen_pool_local.numel() > 0:
                    selected_confidences.append(conf[chosen_pool_local])

            mean_conf = float(torch.cat(selected_confidences).mean().item()) if selected_confidences else 0.0
            step_metrics: dict[str, float] = {}
            if self.metrics_hooks:
                ctx = StepContext(
                    step=step,
                    input_ids_before=input_ids_before,
                    attention_mask=attention_mask,
                    pre_step_logits=logits,
                    masked_positions_per_row=masked_by_row,
                    selected_positions_per_row=selected_positions_per_row,
                    selected_local_per_row=selected_local_per_row,
                    conflict_matrix_per_row=conflict_matrix_per_row,
                    clean_input_ids=clean_input_ids,
                    backbone=self.backbone,
                )
                step_metrics = run_metrics_hooks(self.metrics_hooks, ctx)
            if regrets:
                step_metrics["candidate_oracle_regret"] = sum(regrets) / len(regrets)
            if chosen_costs:
                step_metrics["selected_set_oracle_cost"] = sum(chosen_costs) / len(chosen_costs)
            # Derived state-difficulty signal for diagnostics: min achievable oracle
            # cost in this step's candidate pool. Logged RAW (no clamping) so a
            # negative regret bug stays visible; aggregation clamps to >=0.
            if regrets and chosen_costs:
                step_metrics["pool_min_oracle_cost"] = (
                    step_metrics["selected_set_oracle_cost"] - step_metrics["candidate_oracle_regret"]
                )
            traces.append(
                DecodeStepTrace(
                    step=step,
                    masked_remaining=remaining,
                    selected_positions=selected_by_row,
                    mean_confidence=mean_conf,
                    metrics=step_metrics,
                )
            )
        return DecodeResult(current, cfg.max_steps, traces, {"stopped_by": "max_steps"})
