from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

import torch

from src.labels.dependency_scores import js_divergence_from_logits


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


def confidence_margin_entropy(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits, dim=-1)
    top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
    confidence = top2[..., 0]
    if top2.shape[-1] == 1:
        margin = confidence
    else:
        margin = top2[..., 0] - top2[..., 1]
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    return confidence, margin, entropy


def reconstruction_accuracy(
    prediction_ids: torch.Tensor,
    clean_input_ids: torch.Tensor,
    masked_positions: list[torch.Tensor],
) -> float:
    correct = 0
    total = 0
    for row, positions in enumerate(masked_positions):
        if positions.numel() == 0:
            continue
        pred = prediction_ids[row, positions]
        gold = clean_input_ids[row, positions]
        correct += int(pred.eq(gold).sum().item())
        total += int(positions.numel())
    return correct / total if total else 0.0


@dataclass
class Timer:
    elapsed: float = 0.0


@contextmanager
def timed() -> Iterator[Timer]:
    timer = Timer()
    start = time.perf_counter()
    try:
        yield timer
    finally:
        timer.elapsed = time.perf_counter() - start


@dataclass
class StepContext:
    """Per-step bundle handed to metrics hooks.

    Decoders fill what they have. Hooks gracefully degrade when fields are missing
    (e.g. confidence decoder leaves ``conflict_matrix_per_row`` empty/None).
    """

    step: int
    input_ids_before: torch.Tensor
    attention_mask: torch.Tensor
    pre_step_logits: torch.Tensor
    masked_positions_per_row: list[torch.Tensor]
    selected_positions_per_row: list[torch.Tensor]
    selected_local_per_row: list[torch.Tensor]
    conflict_matrix_per_row: list[Optional[torch.Tensor]] = field(default_factory=list)
    clean_input_ids: Optional[torch.Tensor] = None
    backbone: Any = None


MetricsHook = Callable[[StepContext], dict[str, float]]


def selected_conflict_rate_predicted(
    conflict_matrix: torch.Tensor,
    selected_local: torch.Tensor,
    theta: float,
) -> float:
    """Fraction of selected pairs with predicted conflict probability above theta.

    For greedy_conflict_aware_selection this is trivially ~0 by construction; the
    metric is informative for penalty-based or non-conflict-aware decoders.
    """
    n = int(selected_local.numel())
    if n < 2:
        return 0.0
    sub = conflict_matrix.index_select(0, selected_local).index_select(1, selected_local)
    iu = torch.triu_indices(n, n, offset=1, device=sub.device)
    pairs = sub[iu[0], iu[1]]
    if pairs.numel() == 0:
        return 0.0
    return float((pairs > theta).float().mean().item())


@torch.no_grad()
def selected_conflict_counts_oracle(
    backbone: Any,
    input_ids_before: torch.Tensor,
    attention_mask: torch.Tensor,
    clean_input_ids: torch.Tensor,
    selected_positions: torch.Tensor,
    base_logits: torch.Tensor,
    tau: float,
    topk_for_js: int | None,
    timestep: int = 0,
) -> tuple[int, int]:
    """Raw counterfactual-JS oracle conflict counts on simultaneously-selected positions.

    Returns ``(conflict_count, pair_count)`` so callers can build a *weighted*
    rate (summing numerators / denominators across steps), avoiding the bias of
    averaging per-step rates where late steps select few positions.

    For each pair (i, j) in ``selected_positions`` we reveal the true token at
    ``i`` and measure how much the distribution at ``j`` shifts (JS divergence
    vs. ``base_logits[j]``). Pairs with JS > tau are deemed conflicting.
    """
    n = int(selected_positions.numel())
    if n < 2:
        return 0, 0
    pair_count = 0
    conflict_count = 0
    for i_idx in range(n):
        anchor_pos = int(selected_positions[i_idx].item())
        cf_ids = input_ids_before.clone()
        cf_ids[0, anchor_pos] = clean_input_ids[0, anchor_pos]
        cf_out = backbone.forward(cf_ids, attention_mask, timestep=timestep)
        cf_logits = cf_out["logits"][0]
        for j_idx in range(i_idx + 1, n):
            target_pos = int(selected_positions[j_idx].item())
            js = js_divergence_from_logits(
                base_logits[target_pos].unsqueeze(0),
                cf_logits[target_pos].unsqueeze(0),
                topk=topk_for_js,
            )
            pair_count += 1
            if float(js.item()) > tau:
                conflict_count += 1
    return conflict_count, pair_count


@torch.no_grad()
def selected_conflict_rate_oracle(
    backbone: Any,
    input_ids_before: torch.Tensor,
    attention_mask: torch.Tensor,
    clean_input_ids: torch.Tensor,
    selected_positions: torch.Tensor,
    base_logits: torch.Tensor,
    tau: float,
    topk_for_js: int | None,
    timestep: int = 0,
) -> float:
    """Counterfactual-JS oracle conflict *rate* (conflict_count / pair_count)."""
    conflict_count, pair_count = selected_conflict_counts_oracle(
        backbone=backbone,
        input_ids_before=input_ids_before,
        attention_mask=attention_mask,
        clean_input_ids=clean_input_ids,
        selected_positions=selected_positions,
        base_logits=base_logits,
        tau=tau,
        topk_for_js=topk_for_js,
        timestep=timestep,
    )
    if pair_count == 0:
        return 0.0
    return conflict_count / pair_count


def predicted_conflict_rate_hook(theta: float) -> MetricsHook:
    def hook(ctx: StepContext) -> dict[str, float]:
        rates: list[float] = []
        for selected_local, matrix in zip(ctx.selected_local_per_row, ctx.conflict_matrix_per_row):
            if matrix is None or selected_local.numel() < 2:
                continue
            rates.append(selected_conflict_rate_predicted(matrix, selected_local, theta))
        if not rates:
            return {}
        return {"selected_conflict_rate_predicted": sum(rates) / len(rates)}
    return hook


def selected_set_size_hook() -> MetricsHook:
    """Mean number of positions unmasked together per step (parallelism proxy)."""
    def hook(ctx: StepContext) -> dict[str, float]:
        sizes = [int(s.numel()) for s in ctx.selected_local_per_row if s is not None]
        sizes = [s for s in sizes if s > 0]
        if not sizes:
            return {}
        return {"selected_set_size": sum(sizes) / len(sizes)}
    return hook


def conflict_graph_density_hook(theta: float) -> MetricsHook:
    """Predicted conflict-graph density: fraction of masked pairs with prob > theta.

    Reflects how dense the model's conflict graph is (a too-dense graph makes the
    decoder over-conservative). Skipped for decoders without a conflict matrix.
    """
    def hook(ctx: StepContext) -> dict[str, float]:
        densities: list[float] = []
        for matrix in ctx.conflict_matrix_per_row:
            if matrix is None or matrix.numel() == 0 or matrix.shape[0] < 2:
                continue
            n = matrix.shape[0]
            iu = torch.triu_indices(n, n, offset=1, device=matrix.device)
            pairs = matrix[iu[0], iu[1]]
            if pairs.numel() == 0:
                continue
            densities.append(float((pairs > theta).float().mean().item()))
        if not densities:
            return {}
        return {"conflict_graph_density": sum(densities) / len(densities)}
    return hook


def oracle_conflict_rate_hook(
    tau: float,
    topk_for_js: int | None,
    timestep: int = 0,
) -> MetricsHook:
    def hook(ctx: StepContext) -> dict[str, float]:
        if ctx.clean_input_ids is None or ctx.backbone is None:
            return {}
        rates: list[float] = []
        total_conflict = 0
        total_pairs = 0
        for row, selected_positions in enumerate(ctx.selected_positions_per_row):
            if selected_positions.numel() < 2:
                continue
            conflict_count, pair_count = selected_conflict_counts_oracle(
                backbone=ctx.backbone,
                input_ids_before=ctx.input_ids_before[row : row + 1],
                attention_mask=ctx.attention_mask[row : row + 1],
                clean_input_ids=ctx.clean_input_ids[row : row + 1],
                selected_positions=selected_positions,
                base_logits=ctx.pre_step_logits[row],
                tau=tau,
                topk_for_js=topk_for_js,
                timestep=timestep,
            )
            if pair_count == 0:
                continue
            rates.append(conflict_count / pair_count)
            total_conflict += conflict_count
            total_pairs += pair_count
        if not rates:
            return {}
        # rate kept for backward-compat (mean of per-step rates); the raw
        # numerator/denominator let the aggregator build an unbiased weighted rate.
        return {
            "selected_conflict_rate_oracle": sum(rates) / len(rates),
            "selected_conflict_pairs": float(total_conflict),
            "selected_pair_count_oracle": float(total_pairs),
        }
    return hook


def run_metrics_hooks(hooks: list[MetricsHook], ctx: StepContext) -> dict[str, float]:
    merged: dict[str, float] = {}
    for hook in hooks:
        merged.update(hook(ctx))
    return merged

