"""GRPO objective pieces: reward, group-normalized advantage, clipped surrogate, and
the categorical KL anchor to the confidence self-planner.

These are pure tensor functions (no backbone / no environment) so they unit-test in
isolation and are composed by the trainer.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch

from src.decoding.schedules import topk_schedule

_EPS = 1e-8


@lru_cache(maxsize=256)
def revision_opportunities(answer_len: int, steps: int) -> int:
    """Revision-event upper bound for uncapped, linear full diffusion, not FLOPs."""
    if answer_len < 1 or steps < 1:
        raise ValueError("Positive length and step count required")
    previous, bound = 0, 0
    for step in range(1, steps + 1):
        current = topk_schedule("linear", step, steps, answer_len)
        bound += min(previous, answer_len - current)
        previous = current
    return bound


def compute_reward(
    accuracy: float,
    nfe: int,
    nfe_max: int,
    alpha: float,
    remask_count: int = 0,
    answer_len: int = 1,
    gamma: float = 0.0,
    reward_type: str = "additive",
    remask_form: str = "linear",
    repeat_remask_count: int = 0,
    gamma_repeat: float = 0.0,
) -> float:
    """Reward combining task accuracy with a compute (and revision) penalty.

    ``reward_type``:

    - ``"multiplicative"`` (Apple recipe, Eq. 3.1):
      ``R = A * (1 - NFE/NFE_max)^alpha * revision(remask)``.
      Correctness **gates** everything: a wrong sample (``A == 0``) always scores 0
      regardless of how fast/clean it is, so an all-wrong group has zero variance and
      cannot hand a positive advantage to a "fast-but-wrong" rollout. This is the fix
      for the additive reward-hacking Apple documents (all-wrong group -> fast wrong
      sample gets positive advantage -> policy collapses to unmask-everything).
    - ``"additive"`` (legacy): ``R = A - alpha*NFE/NFE_max - gamma*remask/answer_len``.

    ``remask_form`` picks how the revision factor is shaped (multiplicative mode only):

    - ``"linear"``: ``max(0, 1 - gamma * remask_norm)``.
    - ``"power"``:  ``max(0, 1 - remask_norm) ** gamma`` -- the *same* functional form as
      the compute penalty, so revisions and steps are priced consistently.
    - ``"exp_opportunity"``: ``exp(-gamma * remask_count / B(L,S))``, where B is
      the linear full-diffusion schedule's maximum revision opportunity count.
      At gamma=.1 the multiplier is in [exp(-.1), 1]; no hard-zero cutoff.

    ``gamma == 0`` drops the revision term in both modes (``power`` then yields 1.0).

    ``gamma_repeat`` prices **repeat** revisions on top of that. ``repeat_remask_count`` counts
    only revisions of positions that had already been revised before, so fixing a position once
    is free while oscillating on the same position keeps costing. Shaped as
    ``(1 + repeat_norm) ** -gamma_repeat``: exactly 1 at zero repeats, decaying smoothly, and --
    unlike the ``linear`` revision term, which clamps to 0 once ``gamma * remask_norm >= 1`` and
    from there provides no gradient at all -- it never saturates.
    """
    nfe_norm = nfe / max(nfe_max, 1)
    remask_norm = remask_count / max(answer_len, 1)
    repeat_norm = repeat_remask_count / max(answer_len, 1)
    if reward_type == "multiplicative":
        speed = max(0.0, 1.0 - nfe_norm) ** float(alpha)
        if remask_form == "linear":
            revision = max(0.0, 1.0 - float(gamma) * remask_norm)
        elif remask_form == "power":
            revision = max(0.0, 1.0 - remask_norm) ** float(gamma)
        elif remask_form == "exp_opportunity":
            bound = revision_opportunities(int(answer_len), int(nfe))
            if gamma < 0 or not 0 <= remask_count <= bound:
                raise ValueError("Invalid revision count/weight for the full-diffusion bound")
            revision = math.exp(-float(gamma) * remask_count / max(bound, 1))
        else:
            raise ValueError(f"Unknown remask_form: {remask_form!r}")
        repeat = (1.0 + repeat_norm) ** (-float(gamma_repeat))
        return float(accuracy) * speed * revision * repeat
    if reward_type != "additive":
        raise ValueError(f"Unknown reward_type: {reward_type!r} (use 'additive'|'multiplicative')")
    return (float(accuracy) - float(alpha) * nfe_norm - float(gamma) * remask_norm
            - float(gamma_repeat) * repeat_norm)


def group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """``(R - mean) / (std + eps)`` over a group; all-equal rewards -> zeros."""
    rewards = rewards.float()
    # A float32 mean of identical decimal rewards can round away from each
    # element, yielding spurious nonzero advantages. Exact constants carry no signal.
    if rewards.numel() == 0 or bool((rewards == rewards.flatten()[0]).all()):
        return torch.zeros_like(rewards)
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    return (rewards - mean) / (std + _EPS)


def separated_revision_advantages(
    correctness: torch.Tensor, revision_fraction: torch.Tensor, weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Correctness GRPO plus a bounded, non-standardized revision tie-breaker.

    Center revision fractions among correct answers only. Wrong answers never
    receive a revision bonus; all-wrong groups carry exactly zero signal. This
    is an explicit alternative advantage estimator, not the old shaped reward.
    """
    correctness, revision_fraction = correctness.float(), revision_fraction.float()
    if correctness.shape != revision_fraction.shape or correctness.numel() == 0:
        raise ValueError("Nonempty matching tensors required")
    if (not math.isfinite(weight) or not 0 <= weight <= .1
            or not bool(((correctness == 0) | (correctness == 1)).all())
            or not bool(((revision_fraction >= 0) & (revision_fraction <= 1)).all())):
        raise ValueError("Binary correctness, bounded revision fractions and weight [0,.1] required")
    accuracy_advantage = group_advantages(correctness)
    revision_advantage = torch.zeros_like(correctness)
    correct = correctness.bool()
    if bool(correct.any()):
        fractions = revision_fraction[correct]
        # Exact constants must produce exact zeros, including decimal constants.
        if not bool((fractions == fractions[0]).all()):
            revision_advantage[correct] = -weight * (fractions - fractions.mean())
    return accuracy_advantage + revision_advantage, accuracy_advantage, revision_advantage


def clipped_surrogate(
    logp: torch.Tensor,
    logp_old: torch.Tensor,
    advantage: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """PPO/GRPO clipped surrogate **loss** (to minimize):

        rho = exp(logp - logp_old)
        L = -min(rho * A, clip(rho, 1-eps, 1+eps) * A)
    """
    rho = torch.exp(logp - logp_old)
    clipped = torch.clamp(rho, 1.0 - epsilon, 1.0 + epsilon)
    return -torch.min(rho * advantage, clipped * advantage)


def anchor_penalty(z: torch.Tensor, c: torch.Tensor, temperature: float, kind: str = "kl") -> torch.Tensor:
    """Penalty pulling the learned scores back toward confidence self-planning.

    - ``"kl"``:  ``KL(softmax(Z/T) || softmax(C/T))`` -- compares the induced keep
      *distributions*, so it is scale-free but depends on ``T``.
    - ``"mse"``: ``mean((Z - C)^2)`` -- pulls the scores themselves, independent of ``T``.

    Both are zero at residual init (``Z == C``) and non-negative.
    """
    if kind == "kl":
        return categorical_kl(z, c, temperature)
    if kind == "mse":
        if z.numel() == 0:
            return z.new_zeros(())
        return ((z - c) ** 2).mean()
    raise ValueError(f"Unknown anchor kind: {kind!r} (use 'kl'|'mse')")


def expected_remask_fraction(
    z: torch.Tensor, is_masked: torch.Tensor, k: int, temperature: float
) -> torch.Tensor:
    """Differentiable expected fraction of currently-denoised positions that get remasked.

    The keep-set is a size-``k`` sample without replacement from ``softmax(Z/T)``; its exact
    marginal inclusion probabilities are intractable, so use the standard approximation
    ``p_keep_i ~= min(1, k * softmax(Z/T)_i)``. A denoised position is revised when it is
    *not* kept, hence the penalty ``mean(1 - p_keep)`` over denoised positions.

    Unlike the reward term (which only reaches ``phi`` through a noisy group advantage),
    this gives a direct gradient telling the planner to keep what it already decoded.
    Returns 0 when nothing is denoised yet.

    Written without ``.item()`` or boolean indexing on purpose: both force a host sync, and
    this runs once per recorded step (hundreds of times per iteration), which made the loop
    pathologically slow on GPU.
    """
    if z.numel() == 0 or k <= 0:
        return z.new_zeros(())
    denoised = (~is_masked.bool()).to(z.dtype)
    p_keep = (int(k) * torch.softmax(z / max(temperature, _EPS), dim=0)).clamp(max=1.0)
    # mean over denoised positions, expressed as a masked sum so nothing syncs
    return ((1.0 - p_keep) * denoised).sum() / denoised.sum().clamp(min=1.0)


def categorical_kl(z: torch.Tensor, c: torch.Tensor, temperature: float) -> torch.Tensor:
    """``KL(softmax(Z_E / T) || softmax(C_E / T))`` over the editable region (scalar).

    Anchors the learned keep-distribution to the confidence self-planner. Zero when
    ``Z_E == C_E`` (residual init), non-negative otherwise.
    """
    if z.numel() == 0:
        return z.new_zeros(())
    t = max(temperature, _EPS)
    log_p = torch.log_softmax(z / t, dim=0)
    log_q = torch.log_softmax(c / t, dim=0)
    p = log_p.exp()
    return (p * (log_p - log_q)).sum()
