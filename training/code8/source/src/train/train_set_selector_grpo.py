"""GRPO training for the learned P2 ``setSelector_phi``.

Group rollouts under sampled **step counts** ``S_g`` (the speed knob), score with the
environment verifier, compute the reward ``R = A - alpha*NFE/NFE_max - gamma*remask/len``,
group-normalize advantages, and update the selector with a clipped policy surrogate on
the Plackett-Luce keep-set log-prob plus a categorical KL anchor to the confidence
self-planner. The diffusion backbone stays frozen; only ``phi`` updates.

``run_grpo_training`` is the backbone/env-agnostic core (tested with a fake backbone +
toy env); ``train`` wires a real backbone, tokenizer, and environments from a config.
"""

from __future__ import annotations

import argparse
import copy
import math
import json
import time
from functools import partial
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from src.decoding.planner_features import normalize_spec
from src.decoding.selection import topk_set_logprob
from src.graph.features import FeatureConfig
from src.models.set_selector import SetSelector, SetSelectorConfig
from src.rl.environments import Environment, get_environment
from src.rl.grpo import (
    anchor_penalty,
    clipped_surrogate,
    compute_reward,
    expected_remask_fraction,
    group_advantages,
    revision_opportunities,
    separated_revision_advantages,
)
from src.rl.policy_reduction import reduce_policy_terms
from src.rl.lr_schedule import cosine_group_lr
from src.rl.rollout import denoiser_token_logprob, expert_rollout, rollout_group, rollout_group_chunked
from src.rl.training_state import capture_rng, environment_fingerprint, grad_norm, restore_rng
from src.utils.checkpointing import save_checkpoint
from src.utils.config import load_config, parse_scalar, set_dotted
from src.utils.device import resolve_device
from src.utils.logging import make_run_dir, write_json
from src.utils.seed import set_seed
from src.utils.tracking import init_wandb, log_artifact

# Base confidence-only planner input: [score, is_masked] (see src.decoding.planner_features).
# With model.include_position the input gains a normalized left-to-right position column
# ([score, is_masked, pos], dim 3) so the permutation-invariant planner can learn a
# position-dependent policy (needed to beat confidence in the full-diffusion regime).
PLANNER_INPUT_DIM = 2


def sample_steps(g: int, p_s: dict[str, Any], max_steps: int, generator: torch.Generator | None) -> list[int]:
    """Sample ``g`` decode step-counts from ``p(S)``: ``{values:[...]}`` or int[low,high].

    Every count is clamped to ``[1, max_steps]`` so a rollout always completes.
    """
    lo, hi = 1, max(1, int(max_steps))
    values = p_s.get("values")
    if values:
        idx = torch.randint(0, len(values), (g,), generator=generator)
        return [int(min(hi, max(lo, int(values[int(i)])))) for i in idx]
    low = int(p_s.get("low", 4))
    high = int(p_s.get("high", max_steps))
    high = max(low, high)
    draws = torch.randint(low, high + 1, (g,), generator=generator)
    return [int(min(hi, max(lo, int(x)))) for x in draws]


def _subsample(items: list, n: int, generator: torch.Generator | None) -> list:
    """Up to ``n`` items drawn without replacement (all of them when ``n <= 0``)."""
    if n <= 0 or len(items) <= n:
        return items
    idx = torch.randperm(len(items), generator=generator)[:n]
    return [items[int(i)] for i in idx]


class _EpochSampler:
    """Without-replacement problem batches, reshuffled at every epoch boundary.

    Replaces with-replacement ``randint`` draws, so the training exposure is a real pass over
    the data and is auditable: ``epochs_completed`` / ``unique_seen`` are logged per iteration.
    """

    def __init__(self, n: int, batch_size: int, generator: torch.Generator | None) -> None:
        self.n = int(n)
        self.batch_size = max(1, min(int(batch_size), self.n))
        self.generator = generator
        self._perm = torch.empty(0, dtype=torch.long)
        self._pos = 0
        self.epochs_completed = 0
        self.unique_seen: set[int] = set()

    def next_batch(self) -> list[int]:
        if self._pos >= self._perm.numel():
            if self._perm.numel() > 0:
                self.epochs_completed += 1
            self._perm = torch.randperm(self.n, generator=self.generator)
            self._pos = 0
        idx = [int(i) for i in self._perm[self._pos:self._pos + self.batch_size]]
        self._pos += len(idx)
        self.unique_seen.update(idx)
        return idx

    def state_dict(self) -> dict:
        return {"n": self.n, "batch_size": self.batch_size, "perm": self._perm.clone(),
                "pos": self._pos, "epochs_completed": self.epochs_completed,
                "unique_seen": sorted(self.unique_seen)}

    def load_state_dict(self, state: dict) -> None:
        if (state["n"], state["batch_size"]) != (self.n, self.batch_size):
            raise ValueError("Sampler geometry changed on resume")
        perm = state["perm"].cpu()
        if perm.numel() not in (0, self.n) or not 0 <= state["pos"] <= perm.numel():
            raise ValueError("Invalid sampler position/permutation")
        if perm.numel() and not torch.equal(perm.sort().values, torch.arange(self.n)):
            raise ValueError("Invalid sampler permutation")
        self._perm, self._pos = perm.clone(), int(state["pos"])
        self.epochs_completed = int(state["epochs_completed"])
        self.unique_seen = set(state["unique_seen"])


def _safe_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation, 0.0 when either side is constant (degenerate group)."""
    if a.numel() < 2:
        return 0.0
    a = a.float() - a.float().mean()
    b = b.float() - b.float().mean()
    denom = a.norm() * b.norm()
    if float(denom) <= 1e-8:
        return 0.0
    return float((a @ b) / denom)


def _build_selector(model_cfg: dict[str, Any]) -> tuple[SetSelector, SetSelectorConfig, tuple[str, ...]]:
    """Build the selector plus the planner feature spec its input dim is derived from."""
    spec = normalize_spec(
        model_cfg.get("planner_features"), include_position=bool(model_cfg.get("include_position", False))
    )
    cfg = SetSelectorConfig(
        input_node_dim=len(spec),
        node_dim=int(model_cfg.get("node_dim", 128)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        st_heads=int(model_cfg.get("st_heads", 4)),
        st_blocks=int(model_cfg.get("st_blocks", 2)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        architecture=str(model_cfg.get("architecture", "attention")),
        linear_skip=bool(model_cfg.get("linear_skip", False)),
        position_encoding=str(model_cfg.get("position_encoding", "none")),
        position_dim=int(model_cfg.get("position_dim", 16)),
        position_count=int(model_cfg.get("position_count", 256)),
        position_column=spec.index('position_id') if 'position_id' in spec else -1,
        feature_encoding=str(model_cfg.get('feature_encoding', 'none')),
        feature_columns=tuple(model_cfg.get('feature_columns', ())),
        feature_counts=tuple(model_cfg.get('feature_counts', ())),
        feature_dim=int(model_cfg.get('feature_dim', 16)),
        backbone_hidden_dim=int(model_cfg.get('backbone_hidden_dim',0)),
        hidden_projection_dim=int(model_cfg.get('hidden_projection_dim',64)),
    )
    return SetSelector(cfg), cfg, spec


def run_grpo_training(
    backbone,
    selector: SetSelector,
    env: Environment,
    *,
    params: dict[str, Any],
    device: torch.device,
    generator: torch.Generator,
    wandb_run=None,
    checkpoint_fn: Callable[[int], None] | None = None,
    save_every: int = 0,
    feature_spec: object = None,
    denoiser_params: list[torch.nn.Parameter] | None = None,
    exposure_generator: torch.Generator | None = None,
    resume_state: dict | None = None,
    state_callback: Callable[[dict], None] | None = None,
    rollout_observer: Callable[[dict], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    max_iterations_this_call: int | None = None,
    exposure_plan: list[dict] | None = None,
    rollout_collector: Callable | None = None,
    fast_policy_replay: bool = False,
) -> list[dict[str, float]]:
    """Core P2 GRPO loop. Returns per-iteration metric dicts.

    If ``save_every > 0`` and ``checkpoint_fn`` is given, ``checkpoint_fn(it+1)`` is
    called every ``save_every`` iterations so a job killed at its walltime keeps the
    latest selector (the loop otherwise only checkpoints at the end).
    """
    g = int(params.get("group_size", 8))
    microbatch = int(params.get('rollout_microbatch_size', 0))
    if microbatch < 0:
        raise ValueError('rollout_microbatch_size must be nonnegative')
    collect_group = (partial(rollout_group_chunked, microbatch_size=microbatch)
                     if microbatch else rollout_group)
    if rollout_collector is not None:
        collect_group = rollout_collector
    policy_reduction = str(params.get("policy_loss_reduction", "pooled"))
    if policy_reduction not in ("pooled", "per_rollout", "trajectory_sum"):
        raise ValueError(f"Unknown policy_loss_reduction: {policy_reduction}")
    exposure_generator = generator if exposure_generator is None else exposure_generator
    temperature = float(params.get("temperature", 0.1))
    alpha = float(params.get("alpha", 0.1))
    gamma = float(params.get("gamma", 0.0))
    epsilon = float(params.get("epsilon", 0.2))
    beta = float(params.get("beta", 0.01))
    max_steps = int(params.get("max_steps", 32))
    kappa = str(params.get("kappa", "linear"))
    block_length = int(params.get("block_length", 0))
    reward_type = str(params.get("reward_type", "additive"))
    remask_form = str(params.get("remask_form", "linear"))
    gamma_repeat = float(params.get("gamma_repeat", 0.0))
    anchor_kind = str(params.get("anchor", "kl"))
    max_remask_frac = float(params.get("max_remask_frac", 0.0))
    if remask_form == "exp_opportunity" and (
        block_length != 0 or kappa != "linear" or max_remask_frac != 0
        or (bool(params.get("expert_steering", False))
            and not bool(params.get("opportunity_reward_allow_expert", False)))
    ):
        raise ValueError("exp_opportunity requires uncapped linear full diffusion and explicit expert opt-in")
    lambda_remask = float(params.get("lambda_remask", 0.0))
    # Joint denoiser training (Э6): the token written at each unmasked position becomes a
    # second stochastic action, scored by log p_theta and sharing the group advantage.
    train_denoiser = bool(params.get("train_denoiser", False)) and bool(denoiser_params)
    if fast_policy_replay and (selector.config.architecture != 'mlp' or train_denoiser
        or beta != 0 or lambda_remask != 0 or policy_reduction != 'pooled'
        or not params.get('omit_zero_anchor', False)):
        raise ValueError('Packed replay requires frozen pointwise MLP, pooled loss and no auxiliary losses')
    if train_denoiser and (resume_state is not None or state_callback is not None):
        raise ValueError("Full resume currently supports frozen-planner training only")
    token_temperature = float(params.get("token_temperature", 0.0))
    if train_denoiser and token_temperature <= 0.0:
        # argmax has no log-probability to differentiate, so joint training needs sampling
        token_temperature = 1.0
    denoiser_steps = int(params.get("denoiser_steps", 1))
    expert_steering = bool(params.get("expert_steering", False))
    n_experts = int(params.get("n_experts", 1))
    expert_block_length = int(params.get("expert_block_length", 32))
    iterations = int(params.get("iterations", 100))
    batch_size = int(params.get("batch_size", 4))
    ppo_epochs = int(params.get("ppo_epochs", 1))
    skip_zero_variance = bool(params.get("skip_zero_variance_groups", False))
    variance_epsilon = float(params.get("group_variance_epsilon", 1e-8))
    if variance_epsilon < 0:
        raise ValueError("group_variance_epsilon must be nonnegative")
    schedule_axis = str(params.get("lr_schedule_axis", "optimizer_updates"))
    if schedule_axis not in ("optimizer_updates", "attempted_groups"):
        raise ValueError("Unknown lr_schedule_axis")
    anchor_reduction = str(params.get("anchor_loss_reduction", "pooled"))
    if anchor_reduction not in ("pooled", "per_rollout"):
        raise ValueError("Unknown anchor_loss_reduction")
    grad_clip = float(params.get("grad_clip", 1.0))
    p_s = params.get("p_S", params.get("p_s", {})) or {}
    n_problems = len(env.problems)
    if n_problems == 0:
        raise ValueError("Training environment is empty")
    per_iter = max(1, min(batch_size, n_problems))  # problems (= optimizer steps / epoch) per iteration
    # Exposure may be given in EPOCHS (real without-replacement passes over the data) instead of
    # raw iterations; `epochs` wins when set. With-replacement draws made "600 iterations" only
    # ~2400 draws with repeats -- nowhere near the stated one epoch.
    epochs = params.get("epochs")
    if epochs is not None:
        iterations = max(1, math.ceil(float(epochs) * n_problems / per_iter))
    sampler = _EpochSampler(n_problems, per_iter, exposure_generator)
    separated_weight = float(params.get("separated_revision_weight", 0.0))
    if not math.isfinite(separated_weight) or not 0 <= separated_weight <= .1:
        raise ValueError("separated_revision_weight must be in [0,.1]")
    if separated_weight and (alpha != 0 or gamma != 0 or gamma_repeat != 0
            or block_length != 0 or kappa != "linear" or max_remask_frac != 0
            or (expert_steering and not params.get("separated_revision_allow_expert", False))
            or train_denoiser or lambda_remask != 0
            or reward_type != "multiplicative"):
        raise ValueError("Separated revision signal requires unshaped binary full-diffusion training")
    if exposure_plan is not None:
        if per_iter != 1 or len(exposure_plan) != iterations:
            raise ValueError("Exposure plan requires batch one and one entry per iteration")
        allowed = set(p_s.get("values", range(1, max_steps + 1)))
        for entry in exposure_plan:
            if (not 0 <= entry["problem_idx"] < n_problems or len(entry["steps"]) != g
                    or any(s not in allowed or not 1 <= s <= max_steps for s in entry["steps"])):
                raise ValueError("Invalid exposure plan")
            expert_steps = entry.get("expert_steps", [])
            if expert_steering and n_experts > 0:
                if (len(set(entry["steps"])) != 1
                        or expert_steps != [entry["steps"][0]] * n_experts):
                    raise ValueError("Planned experts must share the policy group's NFE")
            elif expert_steps:
                raise ValueError("Unexpected planned expert trajectories")

    lr = float(params.get("lr", 1e-4))
    param_groups: list[dict[str, Any]] = [{"params": list(selector.parameters()), "lr": lr}]
    if train_denoiser:
        param_groups.append(
            {"params": list(denoiser_params or []), "lr": float(params.get("denoiser_lr", 1e-5))}
        )
    opt = torch.optim.AdamW(
        param_groups, lr=lr, weight_decay=float(params.get("weight_decay", 0.0))
    )
    # Optional cosine-with-warmup LR schedule (Apple recipe: cosine, 100 warmup steps).
    scheduler = None
    if str(params.get('lr_scheduler', 'none')) == 'cosine_groups':
        if schedule_axis != 'attempted_groups' or int(params.get('lr_warmup', 0)) != 0:
            raise ValueError('cosine_groups requires attempted_groups and no warmup')
        cosine_group_lr(lr, float(params['lr_min_ratio']), 0,
                        int(params.get('lr_schedule_groups', iterations * per_iter)))
    if str(params.get("lr_scheduler", "none")) == "cosine":
        warmup = int(params.get("lr_warmup", 0))
        # One optimizer step per (problem, ppo_epoch) and the scheduler advances per step, so
        # the schedule must be sized to the ACTUAL number of updates. Sizing it per outer
        # iteration (iterations*ppo_epochs) made a 600-iteration run reach lr=0 after ~150
        # iterations with batch_size=4 -- the remaining 75% trained at zero learning rate.
        total_updates = max(1, iterations * per_iter * ppo_epochs)

        def _lr_lambda(step: int) -> float:
            if warmup > 0 and step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total_updates - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

        if schedule_axis == "optimizer_updates":
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    history: list[dict[str, float]] = []
    signature = {"params": copy.deepcopy(params), "features": feature_spec,
                 "environment": environment_fingerprint(env), "iterations": iterations}
    if exposure_plan is not None:
        signature["exposure_plan"] = copy.deepcopy(exposure_plan)
    start, updates, attempted_groups = 0, 0, 0
    if resume_state is not None:
        if resume_state.get("version") != 1 or resume_state["signature"] != signature:
            raise ValueError("Resume configuration, features or dataset changed")
        selector.load_state_dict(resume_state["model_state"])
        opt.load_state_dict(resume_state["optimizer"])
        if (scheduler is None) != (resume_state["scheduler"] is None):
            raise ValueError("Resume scheduler changed")
        if scheduler is not None:
            scheduler.load_state_dict(resume_state["scheduler"])
        sampler.load_state_dict(resume_state["sampler"])
        generator.set_state(resume_state["generator"].cpu())
        exposure_generator.set_state(resume_state["exposure_generator"].cpu())
        restore_rng(resume_state["rng"], device)
        start, updates = int(resume_state["iteration"]), int(resume_state["updates"])
        history = copy.deepcopy(resume_state["history"])
        attempted_groups = int(resume_state.get("attempted_groups", start * per_iter))
        if len(history) != start or not 0 <= start <= iterations:
            raise ValueError("Invalid resume iteration/history")

    def snapshot(completed: int) -> dict:
        return {"version": 1, "signature": signature, "iteration": completed,
                "updates": updates, "attempted_groups": attempted_groups,
                "model_state": selector.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "sampler": sampler.state_dict(), "generator": generator.get_state(),
                "exposure_generator": exposure_generator.get_state(),
                "rng": capture_rng(device), "history": history}

    end = iterations if max_iterations_this_call is None else min(
        iterations, start + max(0, int(max_iterations_this_call)))
    for it in range(start, end):
        timing_start = time.perf_counter()
        timing_rollout = 0.0
        if should_stop is not None and should_stop():
            break
        idx = sampler.next_batch()  # without replacement within an epoch
        if exposure_plan is not None and idx != [exposure_plan[it]["problem_idx"]]:
            raise ValueError("Exposure plan disagrees with the independently sampled question order")
        batch = [env.problems[i] for i in idx]
        iter_reward = iter_acc = iter_nfe = iter_remask = iter_loss = iter_kl = 0.0
        iter_l_remask = iter_zero_var = iter_reward_of_wrong = 0.0
        iter_accuracy_adv = iter_revision_adv = iter_signal = 0.0
        iter_corr_adv_remask = iter_degenerate = iter_denoiser_loss = iter_repeat_remask = 0.0
        n_groups = n_wrong_groups = skipped_groups = 0
        updates_before = updates
        replay_error = 0.0
        last_grad_norm = last_policy_norm = last_anchor_norm = 0.0
        post_update_delta = post_update_outside = post_update_measured = 0.0
        last_lr_used, measured_grad = float(opt.param_groups[0]["lr"]), 0.0
        for group_index, (problem_idx, problem) in enumerate(zip(idx, batch)):
            if str(params.get('lr_scheduler', 'none')) == 'cosine_groups':
                rate = cosine_group_lr(lr, float(params['lr_min_ratio']),
                                       attempted_groups + group_index,
                                       int(params.get('lr_schedule_groups', iterations * per_iter)))
                opt.param_groups[0]['lr'] = rate
                if train_denoiser:
                    opt.param_groups[1]['lr'] = float(params.get('denoiser_lr', 1e-5)) * rate / lr
                last_lr_used = rate
            step_counts = (list(exposure_plan[it]["steps"]) if exposure_plan is not None
                           else sample_steps(g, p_s, max_steps, exposure_generator))
            answer_len = int(problem.answer_span.numel())
            selector.eval()
            rollout_start = time.perf_counter()
            rollouts = collect_group(
                backbone, selector, problem.input_ids.to(device), problem.attention_mask.to(device),
                editable_positions=problem.answer_span.to(device), step_counts=step_counts,
                temperature=temperature, kappa=kappa, block_length=block_length,
                max_remask_frac=max_remask_frac, feature_spec=feature_spec,
                token_temperature=token_temperature, store_states=train_denoiser,
                generator=generator,
            )
            timing_rollout += time.perf_counter() - rollout_start
            n_policy = len(rollouts)  # policy rollouts occupy [0, n_policy); experts appended after
            # Expert steering (Apple App. F): mix in E deterministic semi-AR confidence
            # rollouts. If the expert beats the (full-diffusion) policy it earns positive
            # advantage and pulls the policy toward the good unmasking order.
            if expert_steering and n_experts > 0:
                expert_steps = (exposure_plan[it]["expert_steps"] if exposure_plan is not None
                                else sample_steps(n_experts, p_s, max_steps, exposure_generator))
                for s_e in expert_steps:
                    rollouts.append(expert_rollout(
                        backbone, selector, problem.input_ids.to(device), problem.attention_mask.to(device),
                        editable_positions=problem.answer_span.to(device), step_count=s_e,
                        temperature=temperature, kappa=kappa, block_length=expert_block_length,
                        feature_spec=feature_spec,
                    ))
            accuracies, rewards, answers = [], [], []
            for r in rollouts:
                answer = env.decode_answer(r.final_input_ids, problem)
                answers.append(answer)
                acc = float(env.verifier(answer, problem.y_star))
                accuracies.append(acc)
                rewards.append(
                    compute_reward(acc, r.nfe, problem.nfe_max, alpha, r.remask_count,
                                   answer_len, gamma, reward_type=reward_type,
                                   remask_form=remask_form,
                                   repeat_remask_count=r.repeat_remask_count,
                                   gamma_repeat=gamma_repeat)
                )
            reward_t = torch.tensor(rewards, device=device)
            if not torch.isfinite(reward_t).all():
                raise FloatingPointError("Nonfinite group rewards")
            advantages = group_advantages(reward_t)
            accuracy_advantages = advantages
            revision_advantages = torch.zeros_like(advantages)
            if separated_weight:
                fractions = torch.tensor([
                    r.remask_count / max(1, revision_opportunities(answer_len, r.nfe))
                    for r in rollouts], device=device)
                advantages, accuracy_advantages, revision_advantages = separated_revision_advantages(
                    torch.tensor(accuracies, device=device), fractions, separated_weight)
            # Range avoids reduction-roundoff producing a tiny nonzero std for
            # identical non-binary shaped rewards. The tolerance is in reward units.
            zero_variance = float(reward_t.max() - reward_t.min()) <= variance_epsilon
            no_signal = (float(advantages.abs().max()) <= variance_epsilon
                         if separated_weight else zero_variance)
            skip_group = skip_zero_variance and no_signal
            iter_accuracy_adv += float(accuracy_advantages.abs().mean())
            iter_revision_adv += float(revision_advantages.abs().mean())
            iter_signal += float(not no_signal)
            skipped_groups += int(skip_group)
            if rollout_observer is not None:
                rollout_observer({"iteration": it, "problem_idx": problem_idx,
                                  "rollouts": rollouts, "answers": answers,
                                  "accuracies": accuracies, "rewards": rewards,
                                  "advantages": advantages.detach().cpu().tolist(),
                                  "accuracy_advantages": accuracy_advantages.detach().cpu().tolist(),
                                  "revision_advantages": revision_advantages.detach().cpu().tolist(),
                                  "skipped_zero_variance": skip_group})

            # Score the recorded actions in the SAME mode they were rolled out in (eval: dropout
            # off; gradients still flow). Switching to train() here made the PPO ratio
            # pi_new/pi_old absorb dropout noise rather than only the parameter change, so the
            # clipped surrogate reacted to sampling noise, not learning.
            selector.eval()
            packed = None
            if fast_policy_replay and not skip_group:
                from src.rl.packed_replay import PackedReplay
                packed = PackedReplay(rollouts, advantages, device)
            for _epoch in range(0 if skip_group else ppo_epochs):
                if schedule_axis == "attempted_groups" and str(params.get("lr_scheduler")) == "cosine":
                    # Fixed exposure budget: no synthetic optimizer/scheduler steps for
                    # rejected groups, but the next accepted group's LR follows exposure.
                    factor = _lr_lambda((attempted_groups + group_index) * ppo_epochs + _epoch)
                    opt.param_groups[0]["lr"] = lr * factor
                    if train_denoiser:
                        opt.param_groups[1]["lr"] = float(params.get("denoiser_lr", 1e-5)) * factor
                surrogate_terms: list[torch.Tensor] = []
                rollout_terms: list[list[torch.Tensor]] = []
                kl_terms: list[torch.Tensor] = []
                rollout_kl: list[list[torch.Tensor]] = []
                replay_differences: list[torch.Tensor] = []
                remask_terms: list[torch.Tensor] = []
                denoiser_terms: list[torch.Tensor] = []
                if packed is not None:
                    logp = packed.log_probs(selector, temperature)
                    if _epoch == 0:
                        replay_differences.append((logp.detach() - packed.old).abs().max())
                    surrogate_terms = list(clipped_surrogate(
                        logp, packed.old, packed.advantages, epsilon).unbind())
                    rollout_terms = [surrogate_terms]  # exact global pooled denominator
                for r, adv in ([] if packed is not None else zip(rollouts, advantages)):
                    own_terms: list[torch.Tensor] = []
                    own_kl: list[torch.Tensor] = []
                    for rec in r.records:
                        if anchor_reduction == "per_rollout" and not rec.included:
                            continue
                        z = selector(rec.node_features.to(device), rec.scores.to(device))
                        if beta != 0 or not params.get('omit_zero_anchor', False):
                            anchor = anchor_penalty(z, rec.scores.to(device), temperature, kind=anchor_kind)
                            kl_terms.append(anchor)
                            own_kl.append(anchor)
                        if lambda_remask > 0.0:
                            remask_terms.append(
                                expected_remask_fraction(z, rec.is_masked.to(device), rec.k, temperature)
                            )
                        if not rec.included:
                            continue
                        logp = topk_set_logprob(z, rec.order.to(device), temperature)
                        logp_old = torch.tensor(rec.logp_old, device=device)
                        if _epoch == 0:
                            replay_differences.append((logp.detach() - logp_old).abs())
                        term = clipped_surrogate(logp, logp_old, adv, epsilon)
                        surrogate_terms.append(term)
                        own_terms.append(term)
                    rollout_terms.append(own_terms)
                    rollout_kl.append(own_kl)
                    if train_denoiser:
                        # One extra grad forward per sampled step is expensive on an 8B
                        # backbone, so score only a few steps per rollout.
                        with_tokens = [
                            rec for rec in r.records
                            if rec.token_positions is not None and rec.token_positions.numel() > 0
                        ]
                        for rec in _subsample(with_tokens, denoiser_steps, generator):
                            tlogp = denoiser_token_logprob(
                                backbone, rec, problem.attention_mask.to(device), token_temperature
                            )
                            tlogp_old = torch.tensor(rec.token_logp_old, device=device)
                            denoiser_terms.append(clipped_surrogate(tlogp, tlogp_old, adv, epsilon))
                if not surrogate_terms:
                    continue
                if replay_differences:
                    replay_error = max(replay_error, float(torch.stack(replay_differences).max()))
                    if (params.get("fail_on_nonfinite", False) or fast_policy_replay) and replay_error > 1e-3:
                        raise RuntimeError(f"Gradient-enabled on-policy replay mismatch: {replay_error}")
                surrogate = reduce_policy_terms(
                    rollout_terms, policy_reduction,
                    float(params.get("policy_reference_decisions", 1.0)),
                )
                kl = (reduce_policy_terms(rollout_kl, anchor_reduction)
                      if kl_terms else torch.zeros((), device=device))
                l_remask = (
                    torch.stack(remask_terms).mean() if remask_terms else torch.zeros((), device=device)
                )
                denoiser_loss = (
                    torch.stack(denoiser_terms).mean() if denoiser_terms else torch.zeros((), device=device)
                )
                loss = surrogate + beta * kl + lambda_remask * l_remask + denoiser_loss
                if not torch.isfinite(loss):
                    if params.get("fail_on_nonfinite", False):
                        raise FloatingPointError("Nonfinite GRPO loss")
                    # skip a non-finite update rather than poisoning the weights with NaN
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.zero_grad(set_to_none=True)
                diag_every = int(params.get("gradient_diagnostics_every", 0))
                if diag_every > 0 and updates % diag_every == 0:
                    trainable = list(selector.parameters())
                    last_policy_norm = grad_norm(torch.autograd.grad(
                        surrogate, trainable, retain_graph=True, allow_unused=True))
                    last_anchor_norm = grad_norm(torch.autograd.grad(
                        beta * kl, trainable, retain_graph=True, allow_unused=True))
                    measured_grad = 1.0
                loss.backward()
                last_grad_norm = grad_norm(p.grad for p in selector.parameters())
                if params.get("fail_on_nonfinite", False) and not math.isfinite(last_grad_norm):
                    raise FloatingPointError("Nonfinite planner gradients")
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(selector.parameters(), grad_clip)
                    if train_denoiser:
                        torch.nn.utils.clip_grad_norm_(denoiser_params or [], grad_clip)
                last_lr_used = float(opt.param_groups[0]["lr"])
                opt.step()
                policy_diag_every=int(params.get('policy_diagnostics_every',0))
                measure_policy=(policy_diag_every>0 and (it+1)%policy_diag_every==0)
                if (params.get("post_update_diagnostics", False) and measured_grad) or measure_policy:
                    with torch.no_grad():
                        deltas = (packed.log_probs(selector, temperature) - packed.old
                                  if packed is not None else torch.stack([
                            topk_set_logprob(selector(rec.node_features.to(device),
                                                     rec.scores.to(device)),
                                             rec.order.to(device), temperature) - rec.logp_old
                            for r in rollouts for rec in r.records if rec.included]))
                        if not bool(torch.isfinite(deltas).all()):
                            raise FloatingPointError("Nonfinite post-update log probabilities")
                        post_update_delta = float(deltas.abs().mean())
                        post_update_outside = float(((deltas < math.log(1 - epsilon))
                                                   | (deltas > math.log(1 + epsilon))).float().mean())
                        post_update_measured = 1.0
                updates += 1
                if params.get('update_diagnostics_path'):
                    with open(params['update_diagnostics_path'], 'a') as stream:
                        stream.write(json.dumps(dict(group=it+1, reuse_epoch=_epoch,
                            optimizer_update=updates, reward_mean=float(reward_t.mean()),
                            advantages=advantages.detach().cpu().tolist(),
                            loss=float(loss.detach()), policy_loss=float(surrogate.detach()),
                            grad_norm=last_grad_norm, lr=last_lr_used,
                            post_update_diagnostics_measured=bool(post_update_measured),
                            post_update_abs_log_ratio=post_update_delta if post_update_measured else None,
                            post_update_outside_clip=post_update_outside if post_update_measured else None),allow_nan=False)+'\n')
                if scheduler is not None:
                    scheduler.step()
                iter_loss += float(loss.item())
                iter_kl += float(kl.item())
                iter_l_remask += float(l_remask.item())
                iter_denoiser_loss += float(denoiser_loss.item())
            # Report metrics over the POLICY rollouts only (exclude injected experts) so
            # acc/nfe/remask track the learned policy, not the mixed-in expert.
            pol = max(1, n_policy)
            iter_reward += sum(rewards[:pol]) / pol
            iter_acc += sum(accuracies[:pol]) / pol
            iter_nfe += sum(r.nfe for r in rollouts[:pol]) / pol
            iter_remask += sum(r.remask_count for r in rollouts[:pol]) / pol
            iter_repeat_remask += sum(r.repeat_remask_count for r in rollouts[:pol]) / pol
            # Reward-hacking diagnostics (the failure mode Apple documents for additive
            # rewards): a group where every rollout is wrong must carry NO signal, and the
            # advantage must not correlate positively with churn or with unmasking-everything.
            iter_zero_var += float(zero_variance)
            wrong = [rw for rw, ac in zip(rewards, accuracies) if ac <= 0.0]
            if wrong:
                iter_reward_of_wrong += sum(wrong) / len(wrong)
                n_wrong_groups += 1
            remask_t = torch.tensor([float(r.remask_count) for r in rollouts], device=device)
            iter_corr_adv_remask += _safe_corr(advantages, remask_t)
            iter_degenerate += sum(1 for r in rollouts if r.nfe <= 1) / max(1, len(rollouts))
            n_groups += 1
        attempted_groups += len(idx)
        n_groups = max(n_groups, 1)
        metrics = {
            "iter": float(it),
            "reward": iter_reward / n_groups,
            "accuracy": iter_acc / n_groups,
            "nfe": iter_nfe / n_groups,
            "remask": iter_remask / n_groups,
            "repeat_remask": iter_repeat_remask / n_groups,
            "loss": iter_loss / n_groups,
            "kl": iter_kl / n_groups,
            "l_remask": iter_l_remask / n_groups,
            "frac_zero_var_groups": iter_zero_var / n_groups,
            "accuracy_advantage_abs_mean": iter_accuracy_adv / n_groups,
            "revision_advantage_abs_mean": iter_revision_adv / n_groups,
            "frac_signal_groups": iter_signal / n_groups,
            "reward_of_wrong": iter_reward_of_wrong / max(1, n_wrong_groups),
            "corr_adv_remask": iter_corr_adv_remask / n_groups,
            "frac_degenerate": iter_degenerate / n_groups,
            "denoiser_loss": iter_denoiser_loss / n_groups,
            # exposure audit: real passes over the data + distinct problems seen so far
            "epochs_completed": float(sampler.epochs_completed),
            "unique_seen": float(len(sampler.unique_seen)),
            "lr": float(opt.param_groups[0]["lr"]),
            "lr_used": last_lr_used, "optimizer_updates": float(updates),
            "gradient_norm_before_clip": last_grad_norm,
            "policy_gradient_norm": last_policy_norm, "anchor_gradient_norm": last_anchor_norm,
            "gradient_diagnostics_measured": measured_grad,
            "groups_skipped": float(skipped_groups),
            "optimizer_steps_this_iteration": float(updates - updates_before),
            "attempted_groups": float(attempted_groups),
            "replay_logprob_max_error": replay_error,
            "post_update_diagnostics_measured": post_update_measured,
            "post_update_logratio_abs_mean": post_update_delta,
            "post_update_ratio_outside_clip_fraction": post_update_outside,
        }
        history.append(metrics)
        if fast_policy_replay:
            print('EXECUTION_TIMING',json.dumps(dict(group=it+1,
                total_seconds=time.perf_counter()-timing_start,
                rollout_seconds=timing_rollout)),flush=True)
        if wandb_run is not None:
            wandb_run.log(metrics)
        print(f"[grpo] it={it} reward={metrics['reward']:.3f} acc={metrics['accuracy']:.3f} "
              f"nfe={metrics['nfe']:.2f} remask={metrics['remask']:.2f} "
              f"rep={metrics['repeat_remask']:.2f} kl={metrics['kl']:.4f} "
              f"loss={metrics['loss']:.4f} lrm={metrics['l_remask']:.4f} "
              f"zvar={metrics['frac_zero_var_groups']:.2f} rwrong={metrics['reward_of_wrong']:.3f} "
              f"corr={metrics['corr_adv_remask']:+.2f} degen={metrics['frac_degenerate']:.2f}")
        if checkpoint_fn is not None and save_every > 0 and (it + 1) % save_every == 0:
            checkpoint_fn(it + 1)
        if state_callback is not None and save_every > 0 and (it + 1) % save_every == 0:
            state_callback(snapshot(it + 1))
    if state_callback is not None:
        state_callback(snapshot(len(history)))
    return history


def train(config: dict[str, Any]) -> Path:
    from src.backbones.registry import build_backbone

    device = resolve_device(config.get("device", "auto"))
    set_seed(int(config.get("seed", 13)))
    run_dir = make_run_dir(config.get("output_dir", "outputs/runs"), "grpo")
    write_json(run_dir / "resolved_config.json", config)

    feature_config = FeatureConfig.from_raw(config.get("features", {}))
    backbone = build_backbone(config.get("backbone", {})).to(device)
    selector, selector_cfg, feature_spec = _build_selector(config.get("model", {}))
    selector.to(device)

    # Joint denoiser training: attach LoRA adapters (base weights stay frozen).
    denoiser_params: list[torch.nn.Parameter] | None = None
    if bool(config.get("grpo", {}).get("train_denoiser", False)):
        denoiser_params = backbone.apply_lora(config.get("backbone", {}).get("lora", {}) or {})
        n_trainable = sum(int(p.numel()) for p in denoiser_params)
        print(f"[grpo] denoiser LoRA attached: {len(denoiser_params)} tensors, {n_trainable/1e6:.2f}M params")

    env_cfg = config.get("env", {}) or {}
    env_kwargs = dict(env_cfg.get("kwargs", {}))
    if env_cfg.get("name", "toy") != "toy":
        env_kwargs.setdefault("tokenizer", backbone.tokenizer)
        env_kwargs.setdefault("mask_token_id", backbone.mask_token_id)
        env_kwargs.setdefault("pad_token_id", backbone.pad_token_id)
        env_kwargs.setdefault("max_length", feature_config.max_length)
    env = get_environment(env_cfg.get("name", "toy"), **env_kwargs)

    wandb_run = init_wandb(config, run_dir, job_type="grpo", extra_tags={"env": env.name})
    generator = torch.Generator(device="cpu").manual_seed(int(config.get("seed", 13)))
    ckpt_path = run_dir / "set-selector-grpo.pt"

    def _payload() -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model_state": selector.state_dict(),
            "set_selector_config": vars(selector_cfg),
            "feature_config": config.get("features", {}),
            "input_node_dim": selector_cfg.input_node_dim,
            # Explicit column layout so eval feeds the selector exactly what it was
            # trained on (inferring it from input_node_dim breaks with richer specs).
            "planner_features": list(feature_spec),
            "env": env.name,
            "seed": int(config.get("seed", 13)),
        }
        if denoiser_params:
            # A jointly-trained planner is only reproducible together with the denoiser it
            # was trained against, so the LoRA adapters (and the config needed to rebuild
            # the same adapter layout at eval time) travel in the same checkpoint.
            payload["denoiser_lora_state"] = {
                k: v.detach().cpu() for k, v in backbone.model.state_dict().items() if "lora_" in k
            }
            payload["denoiser_lora_config"] = dict(config.get("backbone", {}).get("lora", {}) or {})
        return payload

    def _checkpoint(step: int) -> None:
        payload = _payload()
        payload["iteration"] = int(step)
        save_checkpoint(ckpt_path, payload)
        print(f"[grpo] checkpoint saved at it={step} -> {ckpt_path}")
        # Off-cluster backup: the workspace is scratch storage.
        log_artifact(
            wandb_run, ckpt_path, name=f"set-selector-{run_dir.name}", artifact_type="model",
            metadata={"iteration": int(step), "env": env.name,
                      "planner_features": list(feature_spec)},
            aliases=[f"it{int(step)}"],
        )

    history = run_grpo_training(
        backbone, selector, env,
        params=config.get("grpo", {}), device=device, generator=generator, wandb_run=wandb_run,
        checkpoint_fn=_checkpoint, save_every=int(config.get("grpo", {}).get("save_every", 100)),
        feature_spec=feature_spec, denoiser_params=denoiser_params,
    )

    save_checkpoint(ckpt_path, _payload())
    write_json(run_dir / "history.json", {"history": history})
    log_artifact(
        wandb_run, ckpt_path, name=f"set-selector-{run_dir.name}", artifact_type="model",
        metadata={"iteration": len(history), "env": env.name,
                  "planner_features": list(feature_spec), "final": True},
        aliases=["latest", "final"],
    )
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
