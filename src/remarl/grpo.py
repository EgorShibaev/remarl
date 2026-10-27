"""Group-normalized advantages and clipped GRPO loss."""

import torch

_EPS = 1e-8


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
