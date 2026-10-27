"""Explicit, checkpointed controls for the two native GRPO experiment recipes."""
from dataclasses import asdict, dataclass, replace

SOTA_SHA = "36b2938e70b2d82d8447f2035c6e864d61fcf42a990965ac577aef61cc87d3bc"


def validate_full_training_records(training, raw):
    """Exact one-to-one official-train membership and original prompt/gold text."""
    assert len(raw) == len(training) == 7473
    assert {r["source_id"] for r in training} == {f"gsm8k_train:{i}" for i in range(7473)}
    for entry in training:
        original = raw[int(entry["source_id"].split(":")[-1])]
        assert entry["prompt"] == f"Question: {original['question']}\nAnswer:"
        assert entry["gold"] == original["answer"].split("####")[-1].strip()


@dataclass(frozen=True)
class Recipe:
    initialization: str = "fresh"
    training_questions: int = 1024
    accepted_groups_per_batch: int = 16
    groups_per_minibatch: int = 8
    passes: int = 1
    target_accepted_groups: int = 32768
    lr_schedule: str = "cosine"
    learning_rate: float = 3e-5
    minimum_lr: float = 3e-6
    policy_width: int | None = None
    nfe: int = 8

    def __post_init__(self):
        if self.initialization not in ("fresh", "sota"):
            raise ValueError("Unknown initialization")
        if self.policy_width not in (None, 256) or (self.policy_width is not None and self.initialization != "fresh"):
            raise ValueError("Width override supports fresh 256-wide MLP only")
        if self.nfe not in (4, 6, 8, 16, 32, 64):
            raise ValueError("Unsupported fixed paper NFE")
        if self.training_questions not in (1024, 7473, 14973):
            raise ValueError("Unknown training set")
        if (self.accepted_groups_per_batch not in (8, 16)
                or self.groups_per_minibatch not in (8, 16)
                or self.accepted_groups_per_batch % self.groups_per_minibatch):
            raise ValueError("Require complete 64- or 128-trajectory optimizer minibatches")
        if type(self.passes) is not int or self.passes not in (1, 2, 4):
            raise ValueError("Only one, two or four complete passes supported")
        if self.target_accepted_groups <= 0 or self.target_accepted_groups % 512:
            raise ValueError("Target must be a positive multiple of 512 groups")
        if self.lr_schedule not in ("constant", "cosine"):
            raise ValueError("Unknown LR schedule")
        if not 0 < self.minimum_lr <= self.learning_rate < 1:
            raise ValueError("Invalid learning rate")

    @property
    def total_batches(self):
        return self.target_accepted_groups // self.accepted_groups_per_batch

    @property
    def updates_per_batch(self):
        return self.passes * (self.accepted_groups_per_batch // self.groups_per_minibatch)

    @property
    def probe_split(self):
        return "train_probe" if self.training_questions in (7473, 14973) else "dev"

    def learning_rate_at(self, completed):
        from batch_grpo_core import cosine_lr
        if not 0 <= completed < self.total_batches:
            raise ValueError("Invalid batch number")
        if self.lr_schedule == "constant":
            return self.learning_rate
        return cosine_lr(completed, self.total_batches, self.learning_rate, self.minimum_lr)

    def to_dict(self):
        return asdict(self)

    def selector_config(self, config):
        if self.policy_width is None:
            return config
        assert config.architecture == "mlp"
        return replace(config, hidden_dim=self.policy_width)


def initialize_selector(selector, saved, initialization):
    """Load the exact trained policy, not only the saved architecture config."""
    from batch_grpo_core import state_digest
    if initialization == "sota":
        selector.load_state_dict(saved["model_state"], strict=True)
        assert state_digest(selector.state_dict()) == state_digest(saved["model_state"])
    elif initialization != "fresh":
        raise ValueError("Unknown initialization")
    return state_digest(selector.state_dict())
