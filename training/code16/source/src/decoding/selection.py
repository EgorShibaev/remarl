from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def cap_remask(remask_local: torch.Tensor, z: torch.Tensor, n_editable: int, max_frac: float) -> torch.Tensor:
    """Keep at most ``ceil(max_frac * n_editable)`` revisions this step, lowest ``z`` first.

    A *structural* brake on churn, complementing the reward-side ``gamma`` penalty: the
    planner may still want to revise everything, but only the least-confident positions
    actually get remasked. ``max_frac <= 0`` disables the cap (default behaviour).
    """
    if max_frac <= 0.0 or remask_local.numel() == 0:
        return remask_local
    budget = math.ceil(max_frac * max(n_editable, 1))
    if remask_local.numel() <= budget:
        return remask_local
    order = torch.argsort(z.index_select(0, remask_local))  # ascending: revise weakest first
    return remask_local.index_select(0, order[:budget]).sort().values


@dataclass(frozen=True)
class SelectionResult:
    selected_local: torch.Tensor
    rejected_local: torch.Tensor
    reject_reasons: dict[int, list[int]]


def node_scores(
    confidence: torch.Tensor,
    margin: torch.Tensor | None = None,
    entropy: torch.Tensor | None = None,
) -> torch.Tensor:
    score = confidence.clone()
    if margin is not None:
        score = score + 0.5 * margin
    if entropy is not None:
        score = score - 0.1 * entropy
    return score


def confidence_topk_selection(
    confidence: torch.Tensor,
    budget: int,
    min_confidence: float = 0.0,
) -> SelectionResult:
    if confidence.numel() == 0 or budget <= 0:
        empty = torch.empty(0, dtype=torch.long, device=confidence.device)
        return SelectionResult(empty, empty, {})
    eligible = torch.nonzero(confidence >= min_confidence, as_tuple=False).flatten()
    if eligible.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=confidence.device)
        return SelectionResult(empty, torch.arange(confidence.numel(), device=confidence.device), {})
    order = torch.argsort(confidence[eligible], descending=True)
    selected = eligible[order[:budget]]
    selected_set = set(int(x) for x in selected.detach().cpu())
    rejected = torch.tensor(
        [i for i in range(confidence.numel()) if i not in selected_set],
        dtype=torch.long,
        device=confidence.device,
    )
    return SelectionResult(selected.sort().values, rejected, {})


# Large finite negative used to mask already-drawn items in the Plackett-Luce
# log-softmax; -inf would risk NaN gradients, and score/temperature logits stay well
# above this magnitude for any sane temperature.
_NEG_INF = -1e9


def topk_deterministic(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Sorted local indices of the ``k`` highest scores (P2 eval keep-set)."""
    n = int(scores.numel())
    k = max(0, min(int(k), n))
    if k == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    top = torch.topk(scores, k=k, dim=0).indices
    return top.sort().values


def stochastic_topk(
    scores: torch.Tensor,
    k: int,
    temperature: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a size-``k`` keep-set WITHOUT replacement from ``softmax(scores / T)``.

    Uses the **Gumbel-top-k** trick: perturb the logits with i.i.d. Gumbel noise and take
    the top ``k``. Sorting the perturbed logits descending yields exactly a sequential
    (Plackett-Luce) draw order, so ``topk_set_logprob`` scores it consistently -- but this
    is a single vectorized op instead of ``k`` sequential ``multinomial`` calls (one
    host<->device sync instead of ``k``, which matters in the rollout hot loop).

    Returns ``(keep_sorted, order)``: the selected local indices (sorted) and the draw
    order (highest-perturbed first).
    """
    n = int(scores.numel())
    k = max(0, min(int(k), n))
    if k == 0:
        empty = torch.empty(0, dtype=torch.long, device=scores.device)
        return empty, empty
    logits = (scores / max(temperature, 1e-6)).detach()
    # Gumbel noise drawn on CPU with the reproducibility generator, then moved once.
    u = torch.rand(n, generator=generator).clamp_(1e-10, 1.0).to(logits.device)
    gumbel = -torch.log(-torch.log(u))
    order = torch.argsort(logits + gumbel, descending=True)[:k].contiguous()
    return order.sort().values, order


def topk_set_logprob(scores: torch.Tensor, order: torch.Tensor, temperature: float) -> torch.Tensor:
    """Plackett-Luce log-probability of drawing ``order`` (sequential, no replacement)
    from ``softmax(scores / T)``. Differentiable in ``scores``; recomputes what
    ``stochastic_topk`` sampled so GRPO importance ratios are consistent.

    Put drawn items first, in draw order, followed by the remaining items.
    The denominator at each draw is then its suffix logsumexp. This is the
    same ordered PL probability, not an approximation to an unordered set.
    """
    if order.numel() == 0:
        return scores.new_zeros(())
    logits = scores / max(temperature, 1e-6)
    chosen = torch.zeros_like(scores, dtype=torch.bool)
    chosen[order] = True
    ordered = torch.cat((logits[order], logits[~chosen]))
    denominators = torch.logcumsumexp(ordered.flip(0), dim=0).flip(0)
    return (ordered[:order.numel()] - denominators[:order.numel()]).sum()


def threshold_selection(
    scores: torch.Tensor,
    tau: float,
    fallback_argmax: bool = True,
) -> torch.Tensor:
    """Fast-dLLM rule: local indices of positions with ``score > tau``.

    When nothing clears the threshold and ``fallback_argmax`` is set, fall back to
    the single highest-scoring position so decoding never stalls. Returns sorted
    local indices.
    """
    if scores.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    selected = torch.nonzero(scores > tau, as_tuple=False).flatten()
    if selected.numel() == 0 and fallback_argmax:
        selected = torch.argmax(scores).reshape(1)
    return selected.sort().values


def greedy_conflict_aware_selection(
    scores: torch.Tensor,
    conflict_prob: torch.Tensor,
    budget: int,
    min_score: float = -float("inf"),
    conflict_theta: float = 0.5,
) -> SelectionResult:
    n = int(scores.numel())
    if n == 0 or budget <= 0:
        empty = torch.empty(0, dtype=torch.long, device=scores.device)
        return SelectionResult(empty, empty, {})
    order = torch.argsort(scores, descending=True).tolist()
    scores_list = scores.detach().cpu().tolist()
    conflict_list = conflict_prob.detach().cpu().tolist()
    selected: list[int] = []
    rejected: list[int] = []
    reasons: dict[int, list[int]] = {}
    for candidate in order:
        if scores_list[candidate] < min_score:
            rejected.append(candidate)
            continue
        if selected:
            row = conflict_list[candidate]
            blockers = [c for c in selected if row[c] > conflict_theta]
            if blockers:
                rejected.append(candidate)
                reasons[candidate] = blockers
                continue
        selected.append(candidate)
        if len(selected) >= budget:
            break
    selected_set = set(selected)
    for idx in range(n):
        if idx not in selected_set and idx not in rejected:
            rejected.append(idx)
    return SelectionResult(
        torch.tensor(sorted(selected), dtype=torch.long, device=scores.device),
        torch.tensor(sorted(rejected), dtype=torch.long, device=scores.device),
        reasons,
    )


def budget_only_selection(
    scores: torch.Tensor,
    conflict_prob: torch.Tensor,
    budget: int,
    min_score: float = -float("inf"),
) -> SelectionResult:
    """Top-score selection ignoring conflicts (ablation: no conflict awareness)."""
    del conflict_prob
    n = int(scores.numel())
    if n == 0 or budget <= 0:
        empty = torch.empty(0, dtype=torch.long, device=scores.device)
        return SelectionResult(empty, empty, {})
    eligible = torch.nonzero(scores >= min_score, as_tuple=False).flatten()
    if eligible.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=scores.device)
        return SelectionResult(empty, torch.arange(n, device=scores.device), {})
    order = torch.argsort(scores[eligible], descending=True)
    selected = eligible[order[:budget]]
    selected_set = set(int(x) for x in selected.detach().cpu())
    rejected = torch.tensor(
        [i for i in range(n) if i not in selected_set], dtype=torch.long, device=scores.device
    )
    return SelectionResult(selected.sort().values, rejected, {})


def threshold_only_selection(
    scores: torch.Tensor,
    conflict_prob: torch.Tensor,
    budget: int,
    min_score: float = -float("inf"),
    conflict_theta: float = 0.5,
) -> SelectionResult:
    """Conflict-aware greedy with NO budget cap (threshold is the only limiter)."""
    return greedy_conflict_aware_selection(
        scores=scores,
        conflict_prob=conflict_prob,
        budget=int(scores.numel()),
        min_score=min_score,
        conflict_theta=conflict_theta,
    )


def stochastic_conflict_aware_selection(
    scores: torch.Tensor,
    conflict_prob: torch.Tensor,
    budget: int,
    min_score: float = -float("inf"),
    conflict_theta: float = 0.5,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> SelectionResult:
    """Like greedy IS but candidates are visited in a softmax-sampled order.

    Useful for exploring beyond the deterministic greedy frontier; reduces to
    greedy as ``temperature -> 0``.
    """
    n = int(scores.numel())
    if n == 0 or budget <= 0:
        empty = torch.empty(0, dtype=torch.long, device=scores.device)
        return SelectionResult(empty, empty, {})
    finite = torch.isfinite(scores)
    logits = torch.where(finite, scores / max(temperature, 1e-6), torch.full_like(scores, -float("inf")))
    probs = torch.softmax(logits, dim=0)
    if not torch.isfinite(probs).all() or float(probs.sum()) <= 0:
        order = torch.argsort(scores, descending=True).tolist()
    else:
        order = torch.multinomial(probs, num_samples=n, replacement=False, generator=generator).tolist()
    scores_list = scores.detach().cpu().tolist()
    conflict_list = conflict_prob.detach().cpu().tolist()
    selected: list[int] = []
    rejected: list[int] = []
    reasons: dict[int, list[int]] = {}
    for candidate in order:
        if scores_list[candidate] < min_score:
            rejected.append(candidate)
            continue
        if selected:
            row = conflict_list[candidate]
            blockers = [c for c in selected if row[c] > conflict_theta]
            if blockers:
                rejected.append(candidate)
                reasons[candidate] = blockers
                continue
        selected.append(candidate)
        if len(selected) >= budget:
            break
    selected_set = set(selected)
    for idx in range(n):
        if idx not in selected_set and idx not in rejected:
            rejected.append(idx)
    return SelectionResult(
        torch.tensor(sorted(selected), dtype=torch.long, device=scores.device),
        torch.tensor(sorted(rejected), dtype=torch.long, device=scores.device),
        reasons,
    )


def max_weight_independent_set_selection(
    scores: torch.Tensor,
    conflict_prob: torch.Tensor,
    budget: int,
    min_score: float = -float("inf"),
    conflict_theta: float = 0.5,
    local_search_iters: int = 32,
) -> SelectionResult:
    """Greedy IS warm start + local-search swaps to approximate max-weight IS.

    Treats ``conflict_prob > conflict_theta`` as a hard conflict edge, then tries
    1-for-1 / 1-for-2 swaps that keep the set conflict-free and increase total
    score. Cheap approximation; exact MWIS is NP-hard.
    """
    n = int(scores.numel())
    if n == 0 or budget <= 0:
        empty = torch.empty(0, dtype=torch.long, device=scores.device)
        return SelectionResult(empty, empty, {})
    base = greedy_conflict_aware_selection(
        scores=scores, conflict_prob=conflict_prob, budget=budget,
        min_score=min_score, conflict_theta=conflict_theta,
    )
    selected = [int(x) for x in base.selected_local.detach().cpu()]
    scores_list = scores.detach().cpu().tolist()
    conflict_list = conflict_prob.detach().cpu().tolist()

    def conflicts(a: int, group: list[int]) -> bool:
        return any(conflict_list[a][b] > conflict_theta for b in group)

    eligible = [i for i in range(n) if scores_list[i] >= min_score]
    selected_set = set(selected)
    for _ in range(local_search_iters):
        improved = False
        # try adding any free node that fits and keeps under budget
        if len(selected) < budget:
            for cand in sorted(eligible, key=lambda i: scores_list[i], reverse=True):
                if cand in selected_set:
                    continue
                if not conflicts(cand, selected):
                    selected.append(cand)
                    selected_set.add(cand)
                    improved = True
                    if len(selected) >= budget:
                        break
        # try 1-for-1 swaps that increase score
        for cand in eligible:
            if cand in selected_set:
                continue
            for member in list(selected):
                if scores_list[cand] <= scores_list[member]:
                    continue
                rest = [s for s in selected if s != member]
                if not conflicts(cand, rest):
                    selected = rest + [cand]
                    selected_set = set(selected)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    selected_sorted = sorted(selected)
    selected_set = set(selected_sorted)
    rejected = [i for i in range(n) if i not in selected_set]
    return SelectionResult(
        torch.tensor(selected_sorted, dtype=torch.long, device=scores.device),
        torch.tensor(rejected, dtype=torch.long, device=scores.device),
        {},
    )


def penalty_conflict_aware_selection(
    scores: torch.Tensor,
    conflict_prob: torch.Tensor,
    budget: int,
    min_score: float = -float("inf"),
    gamma: float = 0.5,
) -> SelectionResult:
    n = int(scores.numel())
    if n == 0 or budget <= 0:
        empty = torch.empty(0, dtype=torch.long, device=scores.device)
        return SelectionResult(empty, empty, {})
    order = torch.argsort(scores, descending=True).tolist()
    scores_list = scores.detach().cpu().tolist()
    conflict_list = conflict_prob.detach().cpu().tolist()
    selected: list[int] = []
    rejected: list[int] = []
    reasons: dict[int, list[int]] = {}
    for candidate in order:
        if scores_list[candidate] < min_score:
            rejected.append(candidate)
            continue
        row = conflict_list[candidate]
        penalty = sum(row[c] for c in selected)
        if penalty > gamma:
            rejected.append(candidate)
            reasons[candidate] = selected.copy()
            continue
        selected.append(candidate)
        if len(selected) >= budget:
            break
    selected_set = set(selected)
    for idx in range(n):
        if idx not in selected_set and idx not in rejected:
            rejected.append(idx)
    return SelectionResult(
        torch.tensor(sorted(selected), dtype=torch.long, device=scores.device),
        torch.tensor(sorted(rejected), dtype=torch.long, device=scores.device),
        reasons,
    )
