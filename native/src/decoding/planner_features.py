"""Planner features for P2-style decoding.

The learned P2 planner (``setSelector_phi``) and the confidence baseline both act over
the **full editable region** ``E`` (masked positions ``M_t`` plus already-denoised
positions ``U_t``). For every editable position we build a feature row; which columns are
present is driven by a **feature spec** (a list of names) so richer inputs can be ablated
without touching the decoder or the rollout.

The residual base is always the ``score`` column -- the signal the Apple P2 self-planner
uses -- so ``Z_E = score + Delta_phi(features)`` and at residual init (``Delta_phi == 0``)
the planner is exactly confidence/keep-confidence self-planning, whatever the spec.

Available columns:

- ``score``:       confidence ``max_v p_v`` for a masked position, **keep-confidence**
                   ``p_{current token}`` for an already-denoised one. The P2 signal.
- ``conf``:        ``max_v p_v`` for every position (also for denoised ones).
- ``keep_conf``:   ``p_{current token}``, 0 for masked positions (no token there yet).
- ``is_masked``:   1.0 if currently masked.
- ``logit_max``:   the raw (unnormalized) max logit -- carries scale ``conf`` throws away.
- ``top2``..``top5``: the 2nd..5th highest probabilities (shape of the distribution, not
                   just its peak).
- ``remask_count``: how many times this position was already revised, normalized by the
                   step budget -- lets the planner notice positions it keeps churning.
- ``was_remasked``: 1.0 if this position was revised at least once.
- ``pos``:         normalized left-to-right rank in ``E``. Breaks the permutation
                   invariance of the set-attention planner, so a position-dependent
                   policy (e.g. left-to-right ordering) becomes expressible at all.

``DEFAULT_SPEC`` reproduces the historical confidence-only input.
"""

from __future__ import annotations

import torch

DEFAULT_SPEC: tuple[str, ...] = ("score", "is_masked")
POSITION_SPEC: tuple[str, ...] = ("score", "is_masked", "pos")
HORIZON_SPEC: tuple[str, ...] = (*POSITION_SPEC, "budget_fraction", "progress", "keep_fraction")

# top-k probability columns, in order
_TOPK_COLUMNS = {"top2": 1, "top3": 2, "top4": 3, "top5": 4}
FEATURE_NAMES: tuple[str, ...] = (
    "score", "conf", "keep_conf", "is_masked", "logit_max",
    *sorted(_TOPK_COLUMNS), "remask_count", "was_remasked", "pos",
    "budget_fraction", "progress", "keep_fraction", "position_id",
    "revision_id", "elapsed_steps", "total_steps", "remaining_steps", "scheduled_keep_count",
)


def normalize_spec(spec: object, include_position: bool = False) -> tuple[str, ...]:
    """Validate a feature spec; ``None`` means the default confidence-only input.

    ``include_position`` is the legacy switch that appends ``pos`` to the default spec.
    """
    if spec is None:
        return POSITION_SPEC if include_position else DEFAULT_SPEC
    names = tuple(str(s) for s in spec)
    if not names:
        raise ValueError("planner feature spec must not be empty")
    unknown = [n for n in names if n not in FEATURE_NAMES]
    if unknown:
        raise ValueError(f"Unknown planner features {unknown}. Available: {list(FEATURE_NAMES)}")
    if "score" not in names:
        raise ValueError("planner feature spec must contain 'score' (the residual base)")
    if include_position and "pos" not in names:
        names = names + ("pos",)
    return names


def build_planner_features(
    logits_row: torch.Tensor,  # [seq, vocab]
    current_ids_row: torch.Tensor,  # [seq]
    editable_positions: torch.Tensor,  # [|E|] absolute positions (sorted -> left-to-right)
    mask_token_id: int,
    include_position: bool = False,
    spec: object = None,
    remask_counts: torch.Tensor | None = None,  # [|E|] revisions so far per position
    remask_norm: float = 1.0,  # divisor for remask_count (e.g. the step budget)
    step: int | None = None,
    step_budget: int | None = None,
    keep_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(features [|E|, len(spec)], scores [|E|])`` for the editable region.

    ``scores`` is always the ``score`` column, i.e. the residual base of the planner.
    """
    names = normalize_spec(spec, include_position)
    if set(names) & {"budget_fraction", "progress", "keep_fraction", "elapsed_steps",
                     "total_steps", "remaining_steps", "scheduled_keep_count"}:
        if step is None or step_budget is None or keep_count is None:
            raise ValueError("Horizon features require explicit step, step_budget and keep_count")
        if not 1 <= step <= step_budget or not 0 <= keep_count <= editable_positions.numel():
            raise ValueError("Invalid planner horizon")
    if editable_positions.numel() == 0:
        return logits_row.new_zeros((0, len(names))), logits_row.new_zeros((0,))

    row_logits = logits_row.index_select(0, editable_positions)
    probs = torch.softmax(row_logits, dim=-1)
    current = current_ids_row.index_select(0, editable_positions)
    is_masked = current.eq(mask_token_id)
    # keep-confidence = prob assigned to the token currently occupying the position. For a
    # masked position the "current id" is the mask id (not a vocab token), so that gather
    # value is meaningless -- it is masked out below.
    safe_current = current.clamp(min=0, max=row_logits.shape[-1] - 1)
    keep_conf = probs.gather(-1, safe_current.unsqueeze(-1)).squeeze(-1)
    keep_conf = torch.where(is_masked, torch.zeros_like(keep_conf), keep_conf)
    need_topk = any(n in _TOPK_COLUMNS for n in names)
    top_probs = probs.topk(min(5, probs.shape[-1]), dim=-1).values if need_topk else None
    max_probs = probs.max(dim=-1).values
    score = torch.where(is_masked, max_probs, keep_conf)

    cols: list[torch.Tensor] = []
    for name in names:
        if name == "score":
            cols.append(score)
        elif name == "conf":
            cols.append(max_probs)
        elif name == "keep_conf":
            cols.append(keep_conf)
        elif name == "is_masked":
            cols.append(is_masked.to(score.dtype))
        elif name == "logit_max":
            cols.append(row_logits.max(dim=-1).values)
        elif name in _TOPK_COLUMNS:
            idx = _TOPK_COLUMNS[name]
            if top_probs is not None and idx < top_probs.shape[-1]:
                cols.append(top_probs[:, idx])
            else:  # vocab smaller than the requested rank
                cols.append(torch.zeros_like(score))
        elif name == "remask_count":
            if remask_counts is None:
                cols.append(torch.zeros_like(score))
            else:
                cols.append(remask_counts.to(score.dtype) / max(remask_norm, 1.0))
        elif name == "was_remasked":
            if remask_counts is None:
                cols.append(torch.zeros_like(score))
            else:
                cols.append((remask_counts > 0).to(score.dtype))
        elif name == "position_id":
            # Full editable answer region: exact 0..L-1 IDs, not prompt indices.
            cols.append(torch.arange(editable_positions.numel(), device=score.device,
                                     dtype=score.dtype))
        elif name == "revision_id":
            if remask_counts is None:
                raise ValueError('Explicit history required for revision_id')
            cols.append(remask_counts.to(score.dtype))
        elif name == "elapsed_steps":
            cols.append(torch.full_like(score, step - 1))
        elif name == "total_steps":
            cols.append(torch.full_like(score, step_budget))
        elif name == "remaining_steps":
            cols.append(torch.full_like(score, step_budget - step + 1))
        elif name == "scheduled_keep_count":
            cols.append(torch.full_like(score, keep_count))
        elif name == "pos":
            n = editable_positions.numel()
            cols.append(torch.arange(n, device=score.device, dtype=score.dtype) / max(n - 1, 1))
        elif name == "budget_fraction":
            cols.append(torch.full_like(score, step_budget / max(editable_positions.numel(), 1)))
        elif name == "progress":
            cols.append(torch.full_like(score, (step - 1) / step_budget))
        elif name == "keep_fraction":
            cols.append(torch.full_like(score, keep_count / max(editable_positions.numel(), 1)))
        else:  # pragma: no cover - normalize_spec already rejects unknown names
            raise ValueError(f"Unhandled planner feature {name!r}")
    return torch.stack(cols, dim=-1), score
