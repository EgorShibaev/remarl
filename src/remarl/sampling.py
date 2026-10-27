"""Recorded per-question random streams and confidence-only control."""

import hashlib
import math
from types import SimpleNamespace
import torch

COUNT = 1319


def question_seed(index, nfe, seed):
    key = f"gsm8k-test:{index}:{nfe}:{seed}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:15], 16)


def draws(index, nfe, seed, length=256):
    generator = torch.Generator().manual_seed(question_seed(index, nfe, seed))
    # Explicit per-step calls reproduce the historical CPU sampling stream.
    return torch.stack([torch.rand(length, generator=generator) for _ in range(nfe)])


def batch_indices(start, count=COUNT, batch=8):
    real = list(range(start, min(start + batch, count)))
    if not real:
        raise ValueError("Empty batch")
    return real + [real[-1]] * (batch - len(real)), len(real)


class P2Scores(torch.nn.Module):
    """Exactly confidence; no learned parameters or hidden-state conditioning."""

    def __init__(self, temperature=1.0, retention_bonus=0.0):
        super().__init__()
        if not all(math.isfinite(x) and x >= 0 for x in (temperature, retention_bonus)):
            raise ValueError("Temperature and retention bonus must be finite and nonnegative")
        self.temperature, self.retention_bonus = temperature, retention_bonus
        self.config = SimpleNamespace(
            architecture="mlp",
            input_node_dim=3,
            position_encoding="learned",
            feature_encoding="none",
            backbone_hidden_dim=0,
        )
        self.eval()

    def forward(self, features, scores):
        return scores + self.retention_bonus * (1 - features[:, 1])


def validate_records(records, count=COUNT):
    assert [r["index"] for r in records] == list(range(count)), "Missing/duplicate test IDs"
    for r in records:
        assert r["source_id"] == f"gsm8k-test:{r['index']}"
        assert len(r["answer_token_ids"]) == 256
        assert 126336 not in r["answer_token_ids"]
        assert all(isinstance(t, int) and 0 <= t < 126464 for t in r["answer_token_ids"])
