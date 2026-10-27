"""Multi-question GRPO primitives, independent of the frozen rollout backend.

Configurable complete passes over whole-question minibatches. Raw hidden states and saved
ordered-PL behavior probabilities are immutable CPU replay, not a new behavior
policy reconstructed after the first optimizer step.
"""
import hashlib
import math

import torch

from src.rl.grpo import clipped_surrogate, group_advantages
from worker import ordered_logprob


def state_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def informative(rewards):
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    if rewards.shape != (8,) or not ((rewards == 0) | (rewards == 1)).all():
        raise ValueError("A group must contain exactly eight binary rewards")
    return 0 < rewards.sum().item() < 8


class QuestionSampler:
    """Without-replacement epochs; additionally unique questions within a batch."""
    def __init__(self, count, seed):
        self.count = count
        self.rng = torch.Generator().manual_seed(seed)
        self.remaining = []

    def next(self, excluded):
        if len(excluded) >= self.count:
            raise RuntimeError("Exhausted distinct questions in the current batch")
        while True:
            if not self.remaining:
                self.remaining = torch.randperm(self.count, generator=self.rng).tolist()
            index = self.remaining.pop()
            if index not in excluded:
                return index

    def state_dict(self):
        return dict(count=self.count, rng=self.rng.get_state(), remaining=self.remaining[:])

    def load_state_dict(self, state):
        assert self.count == state["count"]
        self.rng.set_state(state["rng"])
        self.remaining = state["remaining"][:]


def minibatches(groups, rng, expected_groups=16, groups_per_minibatch=8):
    if expected_groups not in (8, 16):
        raise ValueError("Only eight or sixteen groups per collected batch supported")
    if len(groups) != expected_groups or len({g["question_id"] for g in groups}) != expected_groups:
        raise ValueError(f"Require {expected_groups} distinct informative question-groups")
    if not all(informative(g["rewards"]) for g in groups):
        raise ValueError("Uninformative group entered optimizer batch")
    if groups_per_minibatch not in (8, 16) or expected_groups % groups_per_minibatch:
        raise ValueError("Require complete 8- or 16-question minibatches")
    order = torch.randperm(expected_groups, generator=rng).tolist()
    return [[groups[i] for i in order[start:start + groups_per_minibatch]]
            for start in range(0, expected_groups, groups_per_minibatch)]


def cosine_lr(completed_batches, total_batches, maximum=3e-5, minimum=3e-6):
    if not 0 <= completed_batches < total_batches:
        raise ValueError("Invalid schedule position")
    fraction = completed_batches / max(1, total_batches - 1)
    return minimum + (maximum - minimum) * .5 * (1 + math.cos(math.pi * fraction))


def optimize_collection(selector, optimizer, groups, device, rng, expected_groups=16, passes=1,
                        groups_per_minibatch=8):
    """Reshuffle each pass; never refresh behavior probabilities or rewards."""
    if type(passes) is not int or passes not in (1, 2, 4):
        raise ValueError("Only one, two or four passes supported")
    original = [(g["replay"].old_logp.clone(), list(g["rewards"])) for g in groups]
    updates = []
    for epoch in range(passes):
        parts = minibatches(groups, rng, expected_groups, groups_per_minibatch)
        used = []
        for index, part in enumerate(parts):
            result = optimize_minibatch(selector, optimizer, part, device)
            result.update(pass_index=epoch + 1, minibatch_index=index + 1)
            updates.append(result)
            used.extend(result["question_ids"])
        assert len(used) == len(set(used)) == expected_groups
    assert all(torch.equal(old, g["replay"].old_logp) and rewards == g["rewards"]
               for (old, rewards), g in zip(original, groups))
    return updates


def replay_chunks(groups, device, groups_per_chunk=4):
    """Stage at most 4 groups × 8 trajectories × one decision on GPU at once."""
    if groups_per_chunk < 1:
        raise ValueError("Positive chunk size required")
    counts = groups[0]["replay"].counts
    assert all(g["replay"].counts == counts for g in groups)
    advantages = [group_advantages(torch.as_tensor(g["rewards"], dtype=torch.float32))
                  for g in groups]
    for step, _ in enumerate(counts):
        for start in range(0, len(groups), groups_per_chunk):
            part = groups[start:start + groups_per_chunk]
            def cat(name, part=part, step=step):
                return torch.cat([getattr(g["replay"], name)[step] for g in part]).to(device)
            yield (cat("features"), cat("scores"), cat("orders"), cat("old_logp"),
                   torch.cat(advantages[start:start + groups_per_chunk]).to(device))


def replay_logp(selector, features, scores, orders):
    z = selector(features.flatten(0, 1), scores.flatten()).reshape_as(scores)
    return ordered_logprob(z, orders)


@torch.no_grad()
def replay_error(selector, groups, device):
    errors = []
    for features, scores, orders, old, _ in replay_chunks(groups, device):
        errors.append(float((replay_logp(selector, features, scores, orders) - old).abs().max()))
    return max(errors, default=0.)


def optimize_minibatch(selector, optimizer, groups, device, groups_per_chunk=4):
    """ONE step after accumulating all decisions from 8 or 16 questions."""
    if len(groups) not in (8, 16) or len({g["question_id"] for g in groups}) != len(groups):
        raise ValueError("An optimizer minibatch must contain 8 or 16 distinct questions")
    if not all(informative(g["rewards"]) for g in groups):
        raise ValueError("Only informative groups enter the optimizer")
    records = sum(g["replay"].old_logp.numel() for g in groups)
    before = torch.cat([p.detach().flatten() for p in selector.parameters()])
    optimizer.zero_grad(set_to_none=True)
    totals = dict(loss=0., approx_kl=0., clip_fraction=0., logratio_abs=0.)
    for features, scores, orders, old, advantages in replay_chunks(groups, device, groups_per_chunk):
        new = replay_logp(selector, features, scores, orders)
        logratio = new - old
        if not torch.isfinite(new).all() or logratio.abs().max() > 80:
            raise FloatingPointError("Invalid PPO likelihood ratio; aborting before step")
        ratio = logratio.exp()
        loss = clipped_surrogate(new, old, advantages, .2).sum() / records
        loss.backward()
        with torch.no_grad():
            totals["loss"] += float(loss)
            totals["approx_kl"] += float((ratio - 1 - logratio).sum()) / records
            totals["clip_fraction"] += float(((ratio < .8) | (ratio > 1.2)).sum()) / records
            totals["logratio_abs"] += float(logratio.abs().sum()) / records
    norm = torch.nn.utils.clip_grad_norm_(selector.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    after = torch.cat([p.detach().flatten() for p in selector.parameters()])
    if not torch.isfinite(after).all():
        raise FloatingPointError("Nonfinite parameters after optimizer step")
    totals.update(gradient_norm=float(norm), parameter_delta_norm=float((after - before).norm()),
                  question_ids=[g["question_id"] for g in groups], trajectories=8 * len(groups),
                  decisions=records, optimizer_steps=1)
    return totals
