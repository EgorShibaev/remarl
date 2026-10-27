"""P2-style stochastic rollouts for GRPO training of ``setSelector_phi``.

Each group of ``G`` rollouts starts from the identical masked prompt but is assigned a
different **step count** ``S_g`` (the speed knob). At every step the frozen backbone is
run once for the whole group; for each editable position the selector produces a refined
keep-confidence ``z_i`` (confidence-only input ``[score, is_masked]``). A top-k scheduler
fixes how many positions to KEEP this step

    K_s = ceil(kappa(s / S_g) * |E|),   K_{S_g} = |E|,

and the keep-set is sampled without replacement from ``softmax(Z_E / T)`` (Plackett-Luce
``stochastic_topk``). The step then:

- unmasks the newly-kept masked positions ``D_t = KEEPS ∩ M_t`` (argmax token), and
- **remasks** the dropped denoised positions ``R_t = U_t \\ KEEPS`` (back to ``[MASK]``),

so, unlike the old threshold decoder, already-decoded tokens can be revised. Sampling
runs under ``no_grad``; each step stores the detached features + scores + draw order so
the GRPO update recomputes the Plackett-Luce log-prob with gradients into ``phi`` only.
A step whose keep-set is forced (``K_s == |E|``, i.e. the final full-unmask step) is
excluded from the policy loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.backbones.base import DiffusionBackbone
from src.decoding.planner_features import (
    DEFAULT_SPEC,
    POSITION_SPEC,
    build_planner_features,
    normalize_spec,
)
from src.decoding.schedules import make_blocks, steps_per_block, topk_schedule
from src.decoding.selection import cap_remask, stochastic_topk, topk_deterministic, topk_set_logprob
from src.models.set_selector import SetSelector


@dataclass
class StepRecord:
    node_features: torch.Tensor  # [|E|, D] detached
    scores: torch.Tensor  # [|E|] C_E detached
    order: torch.Tensor  # [K_s] Plackett-Luce draw order (local indices), detached
    logp_old: float  # log pi at sample time (detached)
    included: bool  # False for forced full-unmask steps (K_s == |E|)
    is_masked: torch.Tensor  # [|E|] bool, state BEFORE the step (for the remask-loss term)
    k: int = 0  # K_s, the keep-set size this step (for the remask-loss term)
    # Denoiser-training extras, only filled when the backbone is trained jointly: the state
    # the step started from, which positions it wrote, and the tokens it chose there.
    input_ids_before: torch.Tensor | None = None
    token_positions: torch.Tensor | None = None
    token_ids: torch.Tensor | None = None
    token_logp_old: float = 0.0  # sum log p_theta of those tokens at sample time


@dataclass
class Rollout:
    step_count: int  # S_g (target number of decode steps)
    records: list[StepRecord] = field(default_factory=list)
    nfe: int = 0
    remask_count: int = 0
    # Revisions of positions that had already been revised before, i.e.
    # ``sum_i max(0, c_i - 1)`` where ``c_i`` counts revisions of position ``i``. Separates
    # "fixed a position once" from "keeps churning the same position", which the flat
    # remask_count cannot tell apart.
    repeat_remask_count: int = 0
    final_input_ids: torch.Tensor | None = None


def _choose_tokens(
    logits_at_pos: torch.Tensor, temperature: float, generator: torch.Generator | None
) -> torch.Tensor:
    """Token for each newly-unmasked position: argmax (``temperature <= 0``) or a sample.

    Argmax is the deterministic P2/Apple setting (the dLLM is a fixed environment). When the
    denoiser is trained jointly the token choice must be a real stochastic *action* so it
    has a log-probability to differentiate, hence the sampling branch.
    """
    if temperature <= 0.0:
        return logits_at_pos.argmax(dim=-1)
    probs = torch.softmax(logits_at_pos / temperature, dim=-1)
    # sample on CPU with the reproducibility generator, then move back
    idx = torch.multinomial(probs.detach().float().cpu(), num_samples=1, generator=generator)
    return idx.squeeze(-1).to(logits_at_pos.device)


def _resolve_spec(selector: SetSelector, feature_spec: object) -> tuple[str, ...]:
    """Feature spec for a rollout, defaulting to the layout the selector was built for."""
    if feature_spec is None:
        feature_spec = POSITION_SPEC if selector.config.input_node_dim >= 3 else DEFAULT_SPEC
    spec = normalize_spec(feature_spec)
    if len(spec) != selector.config.input_node_dim:
        raise ValueError(
            f"feature spec {spec} has {len(spec)} columns but the selector expects "
            f"input_node_dim={selector.config.input_node_dim}"
        )
    return spec


@torch.no_grad()
def rollout_group(
    backbone: DiffusionBackbone,
    selector: SetSelector,
    input_ids: torch.Tensor,  # [seq] a single masked problem
    attention_mask: torch.Tensor,  # [seq]
    editable_positions: torch.Tensor,  # [|E|] absolute answer-region positions
    step_counts: list[int],  # S_g per group member (the speed knob)
    temperature: float,
    *,
    kappa: str = "linear",
    block_length: int = 0,
    max_remask_frac: float = 0.0,
    feature_spec: object = None,
    token_temperature: float = 0.0,
    store_states: bool = False,
    generator: torch.Generator | None = None,
    timestep: int = 0,
    compact_active: bool = True,
) -> list[Rollout]:
    """Roll out ``G = len(step_counts)`` group samples of one problem; return trajectories.

    ``block_length`` selects the decoding regime: ``<= 0`` (default) is full-diffusion
    P2 over the whole editable region; ``BL > 0`` is Apple-style **semi-AR** decoding,
    where blocks of ``BL`` contiguous positions are planned left to right (later blocks
    stay masked) and each member's ``S_g`` step budget is split across the blocks
    (``steps_per_block``). P2's top-k + remasking run within each block, so ``nfe`` still
    counts one forward per plan/denoise step.
    """
    g = len(step_counts)
    device = input_ids.device
    editable = editable_positions.to(device)
    n_editable = int(editable.numel())
    current = input_ids.unsqueeze(0).repeat(g, 1).clone()
    attn = attention_mask.unsqueeze(0).repeat(g, 1)
    rollouts = [Rollout(step_count=int(s)) for s in step_counts]
    blocks = make_blocks(n_editable, block_length)
    n_blocks = max(1, len(blocks))
    spb = [steps_per_block(int(s), n_blocks) for s in step_counts]  # steps per block, per member
    spec = _resolve_spec(selector, feature_spec)
    # Per-position revision history per group member (fed to the planner when the spec asks
    # for remask_count / was_remasked).
    remask_counts = torch.zeros((g, n_editable), device=device)

    for bstart, bend in blocks:
        block_editable = editable[bstart:bend]
        n_block = int(block_editable.numel())
        max_bsteps = max(spb) if spb else 0
        for step in range(1, max_bsteps + 1):
            active = [row for row in range(g) if step <= spb[row]]
            if not active:
                break
            # Completed trajectories must not continue consuming denoiser row-forwards.
            # Iterate in original row order so each action keeps its RNG assignment.
            active_index = torch.tensor(active, device=device)
            out = backbone.forward(
                current.index_select(0, active_index) if compact_active else current,
                attn.index_select(0, active_index) if compact_active else attn,
                timestep=timestep,
            )
            logits = out["logits"]
            state_before = current.clone() if store_states else None
            for active_row, row in enumerate(active):
                logits_row = active_row if compact_active else row
                s_block = spb[row]
                k = topk_schedule(kappa, step, s_block, n_block)
                node_features, scores = build_planner_features(
                    logits[logits_row], current[row], block_editable, backbone.mask_token_id,
                    spec=spec, remask_counts=remask_counts[row, bstart:bend],
                    remask_norm=float(s_block),
                    step=step, step_budget=s_block, keep_count=k,
                )
                if selector.config.backbone_hidden_dim:
                    node_features=selector.attach_hidden(node_features,out['hidden_states'][logits_row],block_editable)
                z = selector(node_features, scores)
                keep_local, order = stochastic_topk(z, k, temperature, generator=generator)
                included = 0 < k < n_block
                logp = float(topk_set_logprob(z, order, temperature).item()) if order.numel() else 0.0

                keep_mask = torch.zeros(n_block, dtype=torch.bool, device=device)
                keep_mask[keep_local] = True
                is_masked = current[row, block_editable].eq(backbone.mask_token_id)
                # D_t: kept AND currently masked -> unmask via argmax; R_t: dropped AND
                # currently denoised -> remask (both restricted to the current block).
                denoise_local = torch.nonzero(keep_mask & is_masked, as_tuple=False).flatten()
                remask_local = torch.nonzero((~keep_mask) & (~is_masked), as_tuple=False).flatten()
                remask_local = cap_remask(remask_local, z, n_block, max_remask_frac)
                pos = block_editable[denoise_local] if denoise_local.numel() > 0 else None
                chosen: torch.Tensor | None = None
                token_logp = 0.0
                if pos is not None:
                    row_logits_at_pos = logits[logits_row, pos]
                    chosen = _choose_tokens(row_logits_at_pos, token_temperature, generator)
                    current[row, pos] = chosen
                    if store_states:
                        lp = torch.log_softmax(
                            row_logits_at_pos / max(token_temperature, 1e-6), dim=-1
                        )
                        token_logp = float(lp.gather(-1, chosen.unsqueeze(-1)).sum().item())
                if remask_local.numel() > 0:
                    current[row, block_editable[remask_local]] = backbone.mask_token_id
                    rollouts[row].remask_count += int(remask_local.numel())
                    absolute = bstart + remask_local
                    rollouts[row].repeat_remask_count += int(
                        (remask_counts[row, absolute] > 0).sum().item()
                    )
                    remask_counts[row, absolute] += 1.0

                rollouts[row].nfe += 1
                rollouts[row].records.append(
                    StepRecord(
                        node_features=node_features.detach(),
                        scores=scores.detach(),
                        order=order.detach(),
                        logp_old=logp,
                        included=included,
                        is_masked=is_masked.detach().clone(),
                        k=int(k),
                        input_ids_before=state_before[row].clone() if store_states else None,
                        token_positions=pos.clone() if (store_states and pos is not None) else None,
                        token_ids=chosen.clone() if (store_states and chosen is not None) else None,
                        token_logp_old=token_logp,
                    )
                )

    for row in range(g):
        rollouts[row].final_input_ids = current[row].clone()
    return rollouts


@torch.no_grad()
def rollout_group_chunked(*args, step_counts, microbatch_size: int, **kwargs):
    """Collect one policy group in memory-bounded chunks, without any updates.

    The caller MUST normalize rewards across the combined result, not each chunk.
    RNG ordering is chunk-major and is part of the experiment protocol.
    """
    if microbatch_size < 1:
        raise ValueError('Positive rollout microbatch_size required')
    result = []
    for start in range(0, len(step_counts), microbatch_size):
        result.extend(rollout_group(*args,
            step_counts=step_counts[start:start + microbatch_size], **kwargs))
    return result


@torch.no_grad()
def expert_rollout(
    backbone: DiffusionBackbone,
    selector: SetSelector,
    input_ids: torch.Tensor,  # [seq] a single masked problem
    attention_mask: torch.Tensor,  # [seq]
    editable_positions: torch.Tensor,  # [|E|] absolute answer-region positions
    step_count: int,  # S for the expert trajectory
    temperature: float,
    *,
    kappa: str = "linear",
    block_length: int = 32,
    feature_spec: object = None,
    timestep: int = 0,
) -> Rollout:
    """One DETERMINISTIC expert trajectory for Apple-style **expert steering** (App. F).

    The expert is the confidence self-planner run in the *good* regime (semi-AR,
    ``block_length``): at each step it keeps the top-``K_s`` positions by **confidence
    ``C``** (not the learned ``Z``), i.e. exactly what P2 self-planning would do. Its
    keep-set (ordered by ``C`` descending -- the canonical Plackett-Luce order for a
    top-k-by-score pick) is recorded together with the **policy's** log-prob of that
    keep-set (``logp_old = log pi_phi(order)``), so the GRPO update can pull the
    full-diffusion policy toward this expert when the expert earns positive advantage.

    Returns a single ``Rollout`` (same shape as a policy rollout) so the trainer can mix
    it into the group for advantage normalization and the clipped surrogate.
    """
    device = input_ids.device
    editable = editable_positions.to(device)
    n_editable = int(editable.numel())
    current = input_ids.clone()
    attn = attention_mask.unsqueeze(0)
    rollout = Rollout(step_count=int(step_count))
    blocks = make_blocks(n_editable, block_length)
    n_blocks = max(1, len(blocks))
    spb = steps_per_block(int(step_count), n_blocks)
    spec = _resolve_spec(selector, feature_spec)
    remask_counts = torch.zeros(n_editable, device=device)

    for bstart, bend in blocks:
        block_editable = editable[bstart:bend]
        n_block = int(block_editable.numel())
        for step in range(1, spb + 1):
            out = backbone.forward(current.unsqueeze(0), attn, timestep=timestep)
            logits = out["logits"][0]
            k = topk_schedule(kappa, step, spb, n_block)
            node_features, scores = build_planner_features(
                logits, current, block_editable, backbone.mask_token_id,
                spec=spec, remask_counts=remask_counts[bstart:bend], remask_norm=float(spb),
                step=step, step_budget=spb, keep_count=k,
            )
            if selector.config.backbone_hidden_dim:
                raise ValueError('Hidden-feature ablation does not use expert rollouts')
            z = selector(node_features, scores)  # policy Z (for the log-prob only)
            keep_local = topk_deterministic(scores, k)  # expert picks by CONFIDENCE C
            # canonical Plackett-Luce order for a top-k-by-score pick: C descending
            order = keep_local[torch.argsort(scores[keep_local], descending=True)].contiguous()
            included = 0 < k < n_block
            logp = float(topk_set_logprob(z, order, temperature).item()) if order.numel() else 0.0

            keep_mask = torch.zeros(n_block, dtype=torch.bool, device=device)
            keep_mask[keep_local] = True
            is_masked = current[block_editable].eq(backbone.mask_token_id)
            denoise_local = torch.nonzero(keep_mask & is_masked, as_tuple=False).flatten()
            remask_local = torch.nonzero((~keep_mask) & (~is_masked), as_tuple=False).flatten()
            if denoise_local.numel() > 0:
                pos = block_editable[denoise_local]
                current[pos] = logits[pos].argmax(dim=-1)
            if remask_local.numel() > 0:
                current[block_editable[remask_local]] = backbone.mask_token_id
                rollout.remask_count += int(remask_local.numel())
                absolute = bstart + remask_local
                rollout.repeat_remask_count += int((remask_counts[absolute] > 0).sum().item())
                remask_counts[absolute] += 1.0

            rollout.nfe += 1
            rollout.records.append(
                StepRecord(
                    node_features=node_features.detach(),
                    scores=scores.detach(),
                    order=order.detach(),
                    logp_old=logp,
                    included=included,
                    is_masked=is_masked.detach().clone(),
                    k=int(k),
                )
            )
    rollout.final_input_ids = current.clone()
    return rollout


def recompute_step_logprob(selector: SetSelector, record: StepRecord, temperature: float) -> torch.Tensor:
    """Recompute the Plackett-Luce ``log pi`` for a stored step WITH gradient into phi."""
    z = selector(record.node_features, record.scores)
    return topk_set_logprob(z, record.order, temperature)


def denoiser_token_logprob(
    backbone: DiffusionBackbone,
    record: StepRecord,
    attention_mask: torch.Tensor,
    token_temperature: float,
    timestep: int = 0,
) -> torch.Tensor:
    """``sum log p_theta(chosen tokens)`` for a stored step, WITH gradient into the denoiser.

    Replays the recorded state through ``forward_train`` (one extra forward, with graph) and
    scores the tokens the rollout actually wrote. This is the second policy term that lets
    GRPO train the denoiser: the planner decides *where* to write, the denoiser *what*, and
    both actions share the group advantage. Returns 0 when the step wrote nothing.
    """
    if record.input_ids_before is None or record.token_positions is None or record.token_ids is None:
        raise ValueError("step was not recorded with store_states=True; cannot train the denoiser")
    if record.token_positions.numel() == 0:
        return torch.zeros((), device=record.input_ids_before.device)
    out = backbone.forward_train(
        record.input_ids_before.unsqueeze(0), attention_mask.unsqueeze(0), timestep=timestep
    )
    logits = out["logits"][0].index_select(0, record.token_positions)
    log_probs = torch.log_softmax(logits / max(token_temperature, 1e-6), dim=-1)
    return log_probs.gather(-1, record.token_ids.unsqueeze(-1)).squeeze(-1).sum()
