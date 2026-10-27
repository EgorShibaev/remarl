"""Deterministic P2 decoder over a frozen diffusion backbone.

Each of the ``S`` steps decomposes into *planning* and *denoising* (Apple Path Planning,
arXiv:2502.03540). Over the full editable region ``E`` the planner produces a refined
keep-confidence ``z_i``; a top-k scheduler keeps the ``K_s = ceil(kappa(s/S)*|E|)``
highest-``z`` positions and remasks the rest, so already-decoded tokens can be revised.
``K_S = |E|`` guarantees a fully unmasked answer after exactly ``S`` steps, so ``S`` is
the speed knob (NFE == S).

``z_i`` comes from a pluggable ``ZSource`` (confidence-only input, built by
``src.decoding.planner_features``):

- ``ConfidenceZSource``: ``Z_E = C_E`` (P2 self-planning baseline / Method 1).
- ``LearnedZSource``   : ``Z_E = setSelector_phi([score, is_masked])`` (learned / Method 2).

At residual init the learned source reproduces the confidence trajectory exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from src.backbones.base import DiffusionBackbone
from src.decoding.confidence_decoder import DecodeResult, DecodeStepTrace
from src.decoding.planner_features import (
    DEFAULT_SPEC,
    POSITION_SPEC,
    build_planner_features,
    normalize_spec,
)
from src.decoding.schedules import make_blocks, steps_per_block, topk_schedule
from src.decoding.selection import cap_remask, topk_deterministic, stochastic_topk
from src.models.set_selector import SetSelector


class ZSource(Protocol):
    def refined(
        self,
        logits_row: torch.Tensor,  # [seq, vocab]
        current_ids_row: torch.Tensor,  # [seq]
        editable_positions: torch.Tensor,  # [|E|] absolute positions
        mask_token_id: int,
        remask_counts: torch.Tensor | None = None,  # [|E|] revisions so far
        remask_norm: float = 1.0,
        **horizon,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(Z_E, C_E)`` over the editable region (both [|E|])."""
        ...


class ConfidenceZSource:
    """``Z_E = C_E`` (confidence / keep-confidence self-planning)."""

    def refined(self, logits_row, current_ids_row, editable_positions, mask_token_id,
                remask_counts=None, remask_norm=1.0, hidden_row=None, **horizon):
        _feats, scores = build_planner_features(logits_row, current_ids_row, editable_positions, mask_token_id)
        return scores, scores


def _logit(p: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return torch.logit(p.clamp(eps, 1.0 - eps))


class LearnedZSource:
    """``Z_E = setSelector_phi(features)``.

    Two residual parameterisations (both zero-init to *exactly* confidence, so the initial
    policy is Fast-dLLM / P2 self-planning and RL only departs as ``phi`` learns):

    - probability space (default): ``Z = C + Delta_phi``. Fine for top-k *ranking* (P2), but
      ``Z`` is not a calibrated probability, so a fixed threshold on it is meaningless -- this
      is why ``learned_threshold`` degenerates in full-diffusion.
    - logit space (``logit_space=True``, E3): ``Z = sigma(logit(C) + Delta_phi)``. ``Z`` stays a
      calibrated probability in [0,1], the residual is a correction to the odds, and a fixed
      threshold stays meaningful -- the right space for the adaptive-cardinality regime where
      Fast-dLLM wins (the planner can learn to commit more/fewer tokens, not just reorder them).

    The feature spec comes from the checkpoint (``planner_features``) so a trained selector
    always sees exactly the columns it was trained on. Falls back to inferring the legacy
    layout from ``input_node_dim`` for checkpoints saved before specs were recorded.
    """

    def __init__(self, selector: SetSelector, spec: object = None, logit_space: bool = False) -> None:
        self.selector = selector
        self.logit_space = logit_space
        if spec is None:
            spec = POSITION_SPEC if selector.config.input_node_dim >= 3 else DEFAULT_SPEC
        self.spec = normalize_spec(spec)
        if len(self.spec) != selector.config.input_node_dim:
            raise ValueError(
                f"feature spec {self.spec} has {len(self.spec)} columns but the selector "
                f"expects input_node_dim={selector.config.input_node_dim}"
            )

    def refined(self, logits_row, current_ids_row, editable_positions, mask_token_id,
                remask_counts=None, remask_norm=1.0, hidden_row=None, **horizon):
        feats, scores = build_planner_features(
            logits_row, current_ids_row, editable_positions, mask_token_id,
            spec=self.spec, remask_counts=remask_counts, remask_norm=remask_norm,
            **horizon,
        )
        feats=self.selector.attach_hidden(feats,hidden_row,editable_positions)
        if self.logit_space:
            z = torch.sigmoid(_logit(scores) + self.selector.delta(feats))
        else:
            z = self.selector(feats, scores)
        return z, scores


@dataclass
class P2DecoderConfig:
    steps: int = 8  # S: total decode iterations (~ NFE); the speed knob
    kappa: str = "linear"
    block_length: int = 0  # 0 -> full-diffusion (P2 default); >0 -> semi-AR block size (Apple)
    # True (P2 default): already-decoded tokens may be remasked (revision).
    # False: plain monotone top-k unmasking -- the standard confidence decoder. Each step
    # unmasks the highest-scoring masked positions until the denoised count reaches K_s,
    # and nothing is ever revised. Same NFE grid, so it is a like-for-like baseline that
    # isolates how much the revision mechanism itself contributes.
    allow_remask: bool = True
    max_remask_frac: float = 0.0  # >0 caps revisions per step (see cap_remask); 0 = uncapped
    policy_temperature: float = 0.0  # 0 preserves greedy; >0 samples the trained PL policy
    policy_seed: int = 0


def _monotone_denoise(z: torch.Tensor, is_masked: torch.Tensor, k: int) -> torch.Tensor:
    """Local indices to unmask so the denoised count reaches ``k``, highest ``z`` first.

    The classic monotone confidence schedule: only currently-masked positions compete, and
    the number unmasked this step is ``k - already_denoised`` (>= 0). Since the scheduler
    ends at ``k == |E|``, the answer is always fully decoded after ``S`` steps.
    """
    masked_idx = torch.nonzero(is_masked, as_tuple=False).flatten()
    n_new = int(k) - int((~is_masked).sum().item())
    if n_new <= 0 or masked_idx.numel() == 0:
        return masked_idx.new_empty((0,))
    take = min(n_new, int(masked_idx.numel()))
    order = torch.argsort(z.index_select(0, masked_idx), descending=True)[:take]
    return masked_idx.index_select(0, order)


class P2Decoder:
    def __init__(self, backbone: DiffusionBackbone, config: P2DecoderConfig, z_source: ZSource) -> None:
        self.backbone = backbone
        self.config = config
        self.z_source = z_source

    def _editable_per_row(self, input_ids: torch.Tensor) -> list[torch.Tensor]:
        return [torch.nonzero(row.eq(self.backbone.mask_token_id), as_tuple=False).flatten() for row in input_ids]

    @torch.no_grad()
    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        timestep: int = 0,
        editable_positions: list[torch.Tensor] | None = None,
    ) -> DecodeResult:
        cfg = self.config
        if cfg.policy_temperature < 0 or (cfg.policy_temperature > 0 and not cfg.allow_remask):
            raise ValueError("Stochastic evaluation requires positive temperature and remasking")
        generator = torch.Generator().manual_seed(cfg.policy_seed)
        current = input_ids.clone()
        mask_id = self.backbone.mask_token_id
        # Editable region is FIXED across steps (remasking can re-mask within it). Default
        # to the initially-masked answer region per row.
        editable = editable_positions if editable_positions is not None else self._editable_per_row(input_ids)
        # Per-row semi-AR blocks (single full block == full-diffusion, P2 default).
        blocks_per_row = [make_blocks(int(ed.numel()), cfg.block_length) for ed in editable]
        n_blocks_row = [max(1, len(b)) for b in blocks_per_row]
        sblock_row = [steps_per_block(int(cfg.steps), nb) for nb in n_blocks_row]
        max_blocks = max((len(b) for b in blocks_per_row), default=0)

        traces: list[DecodeStepTrace] = []
        total_remask = 0
        forwards = 0
        # Per-position revision history, exposed to the planner via the remask_count /
        # was_remasked features (zeros when those columns are not in the spec).
        remask_counts = [torch.zeros(int(ed.numel()), device=current.device) for ed in editable]
        for b in range(max_blocks):
            rows_in_block = [r for r in range(len(editable)) if b < len(blocks_per_row[r])]
            if not rows_in_block:
                break
            max_bsteps = max(sblock_row[r] for r in rows_in_block)
            for step in range(1, max_bsteps + 1):
                active = [r for r in rows_in_block if step <= sblock_row[r]]
                if not active:
                    break
                out = self.backbone.forward(current, attention_mask, timestep=timestep)
                logits = out["logits"]
                forwards += 1
                selected_by_row: list[list[int]] = [[] for _ in range(len(editable))]
                selected_z: list[torch.Tensor] = []
                for row in active:
                    bs, be = blocks_per_row[row][b]
                    block_editable = editable[row][bs:be]
                    n = int(block_editable.numel())
                    if n == 0:
                        continue
                    k = topk_schedule(cfg.kappa, step, sblock_row[row], n)
                    horizon = (dict(step=step, step_budget=sblock_row[row], keep_count=k,
                                    hidden_row=out.get('hidden_states',None)[row] if out.get('hidden_states') is not None else None)
                               if isinstance(self.z_source, LearnedZSource) else {})
                    z, _c = self.z_source.refined(
                        logits[row], current[row], block_editable, mask_id,
                        remask_counts=remask_counts[row][bs:be], remask_norm=float(cfg.steps),
                        **horizon,
                    )
                    keep_local = (stochastic_topk(z, k, cfg.policy_temperature, generator)[0]
                                  if cfg.policy_temperature > 0 else topk_deterministic(z, k))
                    keep_mask = torch.zeros(n, dtype=torch.bool, device=current.device)
                    keep_mask[keep_local] = True
                    is_masked = current[row, block_editable].eq(mask_id)
                    if cfg.allow_remask:
                        denoise_local = torch.nonzero(keep_mask & is_masked, as_tuple=False).flatten()
                        remask_local = torch.nonzero((~keep_mask) & (~is_masked), as_tuple=False).flatten()
                        remask_local = cap_remask(remask_local, z, n, cfg.max_remask_frac)
                    else:
                        denoise_local = _monotone_denoise(z, is_masked, k)
                        remask_local = denoise_local.new_empty((0,))
                    if denoise_local.numel() > 0:
                        pos = block_editable[denoise_local]
                        current[row, pos] = logits[row, pos].argmax(dim=-1)
                    if remask_local.numel() > 0:
                        current[row, block_editable[remask_local]] = mask_id
                        total_remask += int(remask_local.numel())
                        remask_counts[row][bs + remask_local] += 1.0
                    selected_by_row[row] = [int(x) for x in block_editable[keep_local].detach().cpu()]
                    if keep_local.numel() > 0:
                        selected_z.append(z[keep_local])
                remaining = int(current.eq(mask_id).sum().item())
                mean_z = float(torch.cat(selected_z).mean().item()) if selected_z else 0.0
                traces.append(
                    DecodeStepTrace(
                        step=forwards - 1,
                        masked_remaining=remaining,
                        selected_positions=selected_by_row,
                        mean_confidence=mean_z,
                    )
                )
        return DecodeResult(current, forwards, traces, {"remask_count": total_remask})


@dataclass
class FastDLLMConfig:
    threshold: float = 0.9  # lambda: unmask every masked position with confidence >= this
    block_length: int = 0  # 0 -> single full block (full diffusion); >0 -> semi-AR
    max_steps: int | None = None  # safety cap on forwards per block; defaults to block size


class FastDLLMDecoder:
    """Training-free confidence-threshold parallel decoding (Fast-dLLM, Wu et al. 2025).

    Monotone -- **no remasking**. Each forward, every still-masked position in the current
    semi-AR block whose SCORE ``>= threshold`` is unmasked in parallel with its argmax token;
    if none clear the bar, the single highest-scoring position is forced (so at least one
    token is committed per forward and decoding always terminates). ``NFE`` = number of
    forwards, which is **variable and set by threshold**: a lower threshold unmasks more per
    step -> fewer forwards. Sweeping the threshold traces an accuracy-vs-NFE curve directly
    comparable to the P2 step-count grid.

    The score is the model confidence ``max_v softmax(logits)_v`` (reproducing
    ``get_transfer_index(..., threshold)`` from NVlabs/Fast-dLLM) when ``z_source`` is a
    ``ConfidenceZSource``. Passing a ``LearnedZSource`` instead swaps in the learned
    ``Z = C + Delta_phi`` while keeping the same threshold decoding -- this is the
    ``learned_threshold`` method, which asks whether the learned signal beats raw confidence
    under the *adaptive* schedule that wins in semi-AR (where P2's fixed top-k schedule loses
    to Fast-dLLM). The token written is always the backbone argmax; ``z`` only ranks/gates.
    """

    def __init__(
        self, backbone: DiffusionBackbone, config: FastDLLMConfig, z_source: ZSource | None = None
    ) -> None:
        self.backbone = backbone
        self.config = config
        self.z_source = z_source or ConfidenceZSource()

    def _editable_per_row(self, input_ids: torch.Tensor) -> list[torch.Tensor]:
        mid = self.backbone.mask_token_id
        return [torch.nonzero(row.eq(mid), as_tuple=False).flatten() for row in input_ids]

    @torch.no_grad()
    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        timestep: int = 0,
        editable_positions: list[torch.Tensor] | None = None,
    ) -> DecodeResult:
        cfg = self.config
        current = input_ids.clone()
        mask_id = self.backbone.mask_token_id
        lam = float(cfg.threshold)
        editable = editable_positions if editable_positions is not None else self._editable_per_row(input_ids)
        blocks_per_row = [make_blocks(int(ed.numel()), cfg.block_length) for ed in editable]
        max_blocks = max((len(b) for b in blocks_per_row), default=0)
        forwards = 0
        traces: list[DecodeStepTrace] = []

        for b in range(max_blocks):
            # positions of block b per row (absolute); rows without a block b sit idle
            block_pos = {
                r: editable[r][blocks_per_row[r][b][0]:blocks_per_row[r][b][1]]
                for r in range(len(editable)) if b < len(blocks_per_row[r])
            }
            cap = cfg.max_steps or max((int(p.numel()) for p in block_pos.values()), default=0)
            step = 0
            while step < cap:
                active = {r: p for r, p in block_pos.items() if current[r, p].eq(mask_id).any()}
                if not active:
                    break
                out = self.backbone.forward(current, attention_mask, timestep=timestep)
                logits = out["logits"]
                forwards += 1
                last = step == cap - 1  # final allowed forward: force-complete
                for row, pos in active.items():
                    row_logits = logits[row, pos]
                    argmax_tok = row_logits.argmax(dim=-1)
                    # score gating the threshold: raw confidence, or the learned Z=C+Delta_phi
                    score, _c = self.z_source.refined(logits[row], current[row], pos, mask_id)
                    masked = current[row, pos].eq(mask_id)
                    # candidates: masked AND (score >= lambda); force the top-scoring masked
                    # position if none clear the bar (or unmask all remaining on the last step)
                    keep = masked & (score >= lam)
                    if last:
                        keep = masked
                    elif not bool(keep.any()):
                        masked_score = torch.where(masked, score, score.new_full((), -1e30))
                        keep = torch.zeros_like(masked)
                        keep[int(masked_score.argmax())] = True
                    sel = torch.nonzero(keep, as_tuple=False).flatten()
                    current[row, pos[sel]] = argmax_tok[sel]
                remaining = int(current.eq(mask_id).sum().item())
                traces.append(DecodeStepTrace(step=forwards - 1, masked_remaining=remaining,
                                              selected_positions=[], mean_confidence=lam))
                step += 1
        return DecodeResult(current, forwards, traces, {"remask_count": 0, "threshold": lam})


def frechet_cardinality(conf_masked: torch.Tensor, delta: float) -> int:
    """Fast-dLLM++ Frechet profile certificate (arXiv:2606.02955).

    Uses the FULL sorted confidence profile of the masked positions, not just the weakest
    selected one. For ``c_(1) >= ... >= c_(m)`` (masked confidences, descending):

        L_k = max(0, sum_{j<=k} c_(j) - (k-1))   # Frechet lower bound P(all top-k correct)
        U_k = 1 - c_(k)
        G_k = L_k - U_k

    Commit the LARGEST prefix ``k`` whose certificate clears ``G_k > delta`` (``0`` if none).
    ``G_k`` is decreasing in ``k`` so the feasible set is a prefix; a larger ``delta`` is
    stricter (fewer commits per step -> more forwards). The marginal-only Frechet bound is
    the strongest distribution-free certificate given only marginal confidences, so beating
    it systematically requires non-marginal (dependence / stability / state) information.
    """
    m = int(conf_masked.numel())
    if m == 0:
        return 0
    c_sorted, _ = torch.sort(conf_masked, descending=True)
    csum = torch.cumsum(c_sorted, dim=0)
    ks = torch.arange(1, m + 1, device=c_sorted.device, dtype=c_sorted.dtype)
    lower = torch.clamp(csum - (ks - 1.0), min=0.0)
    upper = 1.0 - c_sorted
    gap = lower - upper
    ok = gap > float(delta)
    if not bool(ok.any()):
        return 0
    return int(torch.nonzero(ok, as_tuple=False).flatten().max().item()) + 1


def adaptive_cardinality(count_scores_masked: torch.Tensor, rule: str, knob: float) -> int:
    """How many masked positions to commit this step, per the adaptive-count ``rule``.

    - ``threshold``: ``|{i : score_i >= knob}|`` (Fast-dLLM confidence threshold, knob = lambda).
    - ``frechet``  : Fast-dLLM++ profile certificate (knob = delta), see ``frechet_cardinality``.
    """
    if rule == "threshold":
        return int((count_scores_masked >= float(knob)).sum().item())
    if rule == "frechet":
        return frechet_cardinality(count_scores_masked, knob)
    raise ValueError(f"unknown adaptive-count rule {rule!r} (expected 'threshold' or 'frechet')")


def select_commit_set(
    count_scores: torch.Tensor,   # [n] score whose profile sets the cardinality
    member_scores: torch.Tensor,  # [n] score that ranks membership
    masked: torch.Tensor,         # [n] bool: only masked positions compete
    rule: str,
    knob: float,
    last: bool = False,
) -> torch.Tensor:
    """Local indices (into the block) to unmask this step: the ``rule`` picks HOW MANY
    (from ``count_scores``), then the top-``k`` ``member_scores`` pick WHICH. Commits >=1 so
    decoding terminates; ``last`` unmasks all remaining (force-complete the block)."""
    masked_idx = torch.nonzero(masked, as_tuple=False).flatten()
    if masked_idx.numel() == 0 or last:
        return masked_idx
    k = adaptive_cardinality(count_scores[masked_idx], rule, knob)
    k = max(1, min(int(k), int(masked_idx.numel())))
    order = torch.argsort(member_scores[masked_idx], descending=True)[:k]
    return masked_idx[order]


@dataclass
class AdaptiveConfig:
    rule: str = "threshold"       # count rule: "threshold" (knob=lambda) | "frechet" (knob=delta)
    knob: float = 0.9
    count_score: str = "C"        # score whose profile sets the CARDINALITY k: "C" (raw) | "Z" (learned)
    member_score: str = "C"       # score that ranks WHICH k positions enter the set: "C" | "Z"
    block_length: int = 0
    max_steps: int | None = None


class AdaptiveDecoder:
    """Factorised adaptive-cardinality set decoder: the count rule (Fast-dLLM / Fast-dLLM++)
    decides HOW MANY masked positions to commit, and a pluggable score decides WHICH.

    This separates the two decisions Fast-dLLM couples, so a learned planner can keep the
    strong adaptive cardinality while re-ranking membership. It subsumes the baselines and
    the residual ablations by choosing where the count and the membership come from:

    | method              | rule      | count_score | member_score |
    |---------------------|-----------|-------------|--------------|
    | fastdllm            | threshold | C           | C            |
    | learned_threshold   | threshold | Z           | Z            |
    | fastdllm_pp (++)    | frechet   | C           | C            |
    | membership_only(E1) | threshold | C           | Z            |
    | frechet_membership  | frechet   | C           | Z            |

    Monotone -- **no remasking** -- so it stays compatible with Fast-dLLM KV-caching and its
    latency profile. ``NFE`` is measured (variable), set by the knob; sweeping the knob traces
    an accuracy-vs-NFE curve. The token written is always the backbone argmax; the scores only
    size and populate the commit set.
    """

    def __init__(
        self, backbone: DiffusionBackbone, config: AdaptiveConfig, z_source: ZSource | None = None
    ) -> None:
        self.backbone = backbone
        self.config = config
        self.z_source = z_source or ConfidenceZSource()

    def _editable_per_row(self, input_ids: torch.Tensor) -> list[torch.Tensor]:
        mid = self.backbone.mask_token_id
        return [torch.nonzero(row.eq(mid), as_tuple=False).flatten() for row in input_ids]

    @torch.no_grad()
    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        timestep: int = 0,
        editable_positions: list[torch.Tensor] | None = None,
    ) -> DecodeResult:
        cfg = self.config
        current = input_ids.clone()
        mask_id = self.backbone.mask_token_id
        editable = editable_positions if editable_positions is not None else self._editable_per_row(input_ids)
        blocks_per_row = [make_blocks(int(ed.numel()), cfg.block_length) for ed in editable]
        max_blocks = max((len(b) for b in blocks_per_row), default=0)
        forwards = 0
        traces: list[DecodeStepTrace] = []

        for b in range(max_blocks):
            block_pos = {
                r: editable[r][blocks_per_row[r][b][0]:blocks_per_row[r][b][1]]
                for r in range(len(editable)) if b < len(blocks_per_row[r])
            }
            cap = cfg.max_steps or max((int(p.numel()) for p in block_pos.values()), default=0)
            step = 0
            while step < cap:
                active = {r: p for r, p in block_pos.items() if current[r, p].eq(mask_id).any()}
                if not active:
                    break
                out = self.backbone.forward(current, attention_mask, timestep=timestep)
                logits = out["logits"]
                forwards += 1
                last = step == cap - 1  # final allowed forward: force-complete this block
                for row, pos in active.items():
                    z, c = self.z_source.refined(logits[row], current[row], pos, mask_id)
                    count_scores = c if cfg.count_score == "C" else z
                    member_scores = c if cfg.member_score == "C" else z
                    masked = current[row, pos].eq(mask_id)
                    if not bool(masked.any()):
                        continue
                    sel = select_commit_set(count_scores, member_scores, masked, cfg.rule, cfg.knob, last)
                    current[row, pos[sel]] = logits[row, pos[sel]].argmax(dim=-1)
                remaining = int(current.eq(mask_id).sum().item())
                traces.append(DecodeStepTrace(step=forwards - 1, masked_remaining=remaining,
                                              selected_positions=[], mean_confidence=float(cfg.knob)))
                step += 1
        return DecodeResult(current, forwards, traces, {"remask_count": 0, "knob": float(cfg.knob)})
