"""Policy-only reduction choices; anchor/auxiliary reductions stay unchanged."""

import torch


def reduce_policy_terms(groups: list[list[torch.Tensor]], mode: str,
                        reference_decisions: float = 1.0) -> torch.Tensor:
    if mode not in ("pooled", "per_rollout", "trajectory_sum"):
        raise ValueError(f"Unknown policy loss reduction: {mode}")
    if reference_decisions <= 0:
        raise ValueError("reference_decisions must be positive")
    flat = [term for group in groups for term in group]
    if not flat:
        raise ValueError("At least one eligible decision is required")
    if mode == "pooled":
        return torch.stack(flat).mean()
    zero = flat[0] * 0
    if mode == "trajectory_sum":
        return torch.stack([torch.stack(g).sum() if g else zero for g in groups]).mean() / reference_decisions
    return torch.stack([torch.stack(g).mean() if g else zero for g in groups]).mean()
