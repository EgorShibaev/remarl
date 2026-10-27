"""Candidate unmask sets + oracle set cost for the set-level experiment.

Everything is offline: a label record already stores per-pair oracle conflict
(``dependency_scores``) over local pairs (``pairs``), plus ``base_confidence``
per masked position. So for any candidate set S (subset of local masked-position
indices) we can score how unsafe it is to unmask jointly, with no new
counterfactual forward passes.

``oracle_set_cost(S)`` = aggregate (mean or max) of the oracle pairwise conflict
over the observed intra-S pairs. Candidate sets span the cost spectrum so a set
scorer has something to discriminate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.decoding.selection import greedy_conflict_aware_selection, node_scores
from src.graph.graph_builder import edge_vector_to_matrix


@dataclass(frozen=True)
class CandidateSet:
    name: str
    indices: torch.Tensor  # local indices into masked_positions
    cost: float            # oracle_set_cost (mean over observed intra-S pairs)
    cost_max: float        # oracle_set_cost (max)
    n_observed: int        # number of observed intra-S pairs


def build_conflict_matrices(
    n: int, pairs: torch.Tensor, dependency_scores: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (conflict_value_matrix [n,n], observed_mask [n,n]) from local pairs."""
    if pairs.numel() == 0:
        z = torch.zeros((n, n), dtype=torch.float32, device=dependency_scores.device)
        return z, z.clone()
    edge_index = pairs.t().contiguous()
    conf = edge_vector_to_matrix(edge_index, dependency_scores.float(), num_nodes=n)
    obs = edge_vector_to_matrix(edge_index, torch.ones_like(dependency_scores.float()), num_nodes=n)
    return conf, obs


def oracle_set_cost(
    indices: torch.Tensor, conf_mat: torch.Tensor, obs_mat: torch.Tensor
) -> tuple[float, float, int]:
    """(mean_cost, max_cost, n_observed_pairs) over observed intra-S pairs."""
    m = int(indices.numel())
    if m < 2:
        return 0.0, 0.0, 0
    idx = indices.long()
    sub_c = conf_mat.index_select(0, idx).index_select(1, idx)
    sub_o = obs_mat.index_select(0, idx).index_select(1, idx)
    iu = torch.triu_indices(m, m, offset=1, device=conf_mat.device)
    vals = sub_c[iu[0], iu[1]]
    obs = sub_o[iu[0], iu[1]] > 0
    n_obs = int(obs.sum())
    if n_obs == 0:
        return 0.0, 0.0, 0
    observed_vals = vals[obs]
    return float(observed_vals.mean()), float(observed_vals.max()), n_obs


def _high_conf_unsafe(confidence: torch.Tensor, conf_mat: torch.Tensor, budget: int) -> torch.Tensor:
    """Hard negative: a dense-conflict cluster of fairly confident nodes."""
    n = int(confidence.numel())
    selected = [int(torch.argmax(confidence).item())]
    while len(selected) < budget and len(selected) < n:
        sel = torch.tensor(selected, device=conf_mat.device)
        # score = conflict-to-selected (encourage unsafe) + small confidence pull
        conflict_to_sel = conf_mat.index_select(1, sel).sum(dim=1)
        score = conflict_to_sel + 0.1 * confidence
        for s in selected:
            score[s] = -float("inf")
        selected.append(int(torch.argmax(score).item()))
    return torch.tensor(sorted(selected), dtype=torch.long, device=conf_mat.device)


def generate_candidate_sets(
    confidence: torch.Tensor,
    pairs: torch.Tensor,
    dependency_scores: torch.Tensor,
    budget_fraction: float = 0.4,
    num_random: int = 2,
    conflict_theta: float = 0.05,
    generator: torch.Generator | None = None,
) -> list[CandidateSet]:
    """Candidate unmask sets spanning the safety spectrum (local indices)."""
    n = int(confidence.numel())
    if n < 2:
        return []
    budget = max(2, min(n, round(budget_fraction * n)))
    conf_mat, obs_mat = build_conflict_matrices(n, pairs, dependency_scores)
    device = confidence.device

    sets: dict[str, torch.Tensor] = {}
    sets["confidence_topk"] = torch.topk(confidence, k=budget).indices.sort().values
    oracle_sel = greedy_conflict_aware_selection(
        scores=node_scores(confidence), conflict_prob=conf_mat, budget=budget, conflict_theta=conflict_theta
    ).selected_local
    if oracle_sel.numel() >= 2:
        sets["oracle_greedy"] = oracle_sel
    sets["high_conf_unsafe"] = _high_conf_unsafe(confidence, conf_mat, budget)
    for r in range(num_random):
        perm = torch.randperm(n, generator=generator, device=device)[:budget]
        sets[f"random_{r}"] = perm.sort().values

    out: list[CandidateSet] = []
    for name, idx in sets.items():
        if idx.numel() < 2:
            continue
        mean_c, max_c, n_obs = oracle_set_cost(idx, conf_mat, obs_mat)
        out.append(CandidateSet(name=name, indices=idx, cost=mean_c, cost_max=max_c, n_observed=n_obs))
    return out
