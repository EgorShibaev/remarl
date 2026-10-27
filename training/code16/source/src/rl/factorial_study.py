"""Fixed exposure contract for the 2026-09-14 full-diffusion factorial pilot."""
import torch

CELLS = {
    "mixed-correctness": ("mixed", 0.0),
    "shared-correctness": ("shared", 0.0),
    "mixed-separated": ("mixed", .1),
    "shared-separated": ("shared", .1),
}


def make_plan(size: int, seed: int, mode: str, budgets=(8, 16, 32, 64), group=8):
    """Same question permutation in every cell; exactly balanced marginal S.

    Budget randomness never consumes the question or action RNG. In shared mode
    all eight trajectories answer the same question with the same assigned S.
    """
    if mode not in ("mixed", "shared") or size % len(budgets):
        raise ValueError("Unknown grouping or unbalanced exposure length")
    questions = torch.randperm(size, generator=torch.Generator().manual_seed(seed + 100003))
    count = size * (group if mode == "mixed" else 1)
    deck = torch.tensor(list(budgets) * (count // len(budgets)))
    deck = deck[torch.randperm(count, generator=torch.Generator().manual_seed(seed + 200003))]
    steps = deck.reshape(size, group) if mode == "mixed" else deck[:, None].expand(size, group)
    return [{"problem_idx": int(q), "steps": s.tolist()} for q, s in zip(questions, steps)]
