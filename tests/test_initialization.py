"""Initialization-ablation scores, public training command and checkpoint roundtrips."""

from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest
import torch

from remarl import training
from remarl.cli import main
from remarl.optimization import replay_error, state_digest
from remarl.policy import ReMaRLPolicy, PolicyConfig
from remarl.runtime import load_policy
from test_lifecycle import ToyGenerator, ToyScorer


ROOT = Path(__file__).resolve().parents[1]


def small_config(**kwargs):
    return PolicyConfig(hidden_dim=8, backbone_hidden_dim=3, hidden_projection_dim=4, **kwargs)


def test_random_policy_uses_confidence_as_input_without_fixed_addition():
    torch.manual_seed(7)
    model = ReMaRLPolicy(small_config(confidence_residual=False, zero_initialize_output=False))
    features = torch.randn(16, 6)
    features[:, 2] = torch.arange(16)
    features.requires_grad_()
    confidence = features[:, 0].detach()
    actual = model(features, confidence)
    assert torch.equal(actual, model.delta(features))
    assert torch.equal(actual, model(features, confidence + 10))
    assert actual.std() > 0
    for head in (model.pointwise[-1].weight, model.pointwise[-1].bias, model.linear_skip.weight):
        assert torch.count_nonzero(head) == head.numel()
    actual.square().sum().backward()
    assert features.grad[:, 0].norm() > 0
    assert model.pointwise[0].weight.grad.norm() > 0
    assert model.hidden_projection[1].weight.grad.norm() > 0


def test_old_checkpoints_keep_p2_prime_initialization(tmp_path):
    model = ReMaRLPolicy(small_config())
    features = torch.randn(16, 6)
    features[:, 2] = torch.arange(16)
    confidence = features[:, 0]
    assert torch.equal(model(features, confidence), confidence)
    legacy = asdict(model.config)
    del legacy["confidence_residual"], legacy["zero_initialize_output"]
    path = tmp_path / "legacy.pt"
    torch.save(
        dict(
            model=model.state_dict(),
            selector_config=legacy,
            policy_sha256=state_digest(model.state_dict()),
        ),
        path,
    )
    loaded, _ = load_policy(path)
    assert loaded.config == model.config
    assert torch.equal(loaded(features, confidence), confidence)
    assert state_digest(loaded.state_dict()) == state_digest(model.state_dict())


def test_initialization_recipe_keeps_other_training_settings():
    cfg = json.loads((ROOT / "configs/ablations/random_mlp_nfe8.json").read_text())
    full = json.loads((ROOT / "configs/ablations/full.json").read_text())
    assert cfg.pop("policy_initialization") == "random_mlp_no_confidence_residual"
    full.pop("variant")
    assert cfg == full


def test_unknown_initialization_is_not_silently_ignored():
    cfg = json.loads((ROOT / "configs/ablations/full.json").read_text())
    cfg["policy_initialization"] = "misspelled"
    with pytest.raises(ValueError, match="Unknown policy initialization"):
        training.validate_config(cfg)


def test_random_training_cli_resume_and_evaluation_loading(tmp_path, monkeypatch):
    cfg = json.loads((ROOT / "configs/ablations/random_mlp_nfe8.json").read_text())
    cfg.update(training_questions=64, target_accepted_groups=32, save_every_groups=16)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    rows = [dict(prompt_ids=[2, i % 8 + 1], gold={}) for i in range(64)]
    manifest = dict(domain="math", required_canvas=4096, rows_sha256="toy-data")
    original_config = training.policy_config_for_training
    constructed = []

    def fixture_config(config):
        return replace(
            original_config(config), hidden_dim=8, backbone_hidden_dim=3, hidden_projection_dim=4
        )

    def generator(backbone, policy, device, backend, max_length):
        constructed.append(policy)
        result = ToyGenerator(policy.eval())
        result.tokenizer = None
        return result

    monkeypatch.setattr(training, "configure_compute", lambda: None)
    monkeypatch.setattr(training, "load_prepared", lambda _: ({"train": rows}, manifest))
    monkeypatch.setattr(training, "policy_config_for_training", fixture_config)
    monkeypatch.setattr(training, "Generator", generator)
    monkeypatch.setattr(training, "Scorer", lambda *_: ToyScorer())
    args = [
        "train",
        "--config",
        str(path),
        "--model",
        "toy-model",
        "--data",
        "toy-data",
        "--devices",
        "cpu",
        "--backend",
        "torch",
    ]
    first = tmp_path / "first"
    main(args + ["--output", str(first)])
    initial, _ = load_policy(first / "initial.pt")
    loaded, saved = load_policy(first / "latest.pt")
    assert saved["accepted_groups"] == 32
    assert state_digest(initial.state_dict()) != saved["policy_sha256"]
    for model in [initial, loaded, *constructed]:
        assert not model.config.confidence_residual
        assert not model.config.zero_initialize_output
    noise = torch.rand(8, 8, 256, generator=torch.Generator().manual_seed(17))
    rollout = ToyGenerator(loaded).generate(rows[:8], noise, True)
    groups = [dict(question_id=0, rewards=[0, 1] * 4, replay=rollout["replay"])]
    assert replay_error(loaded, groups, "cpu") < 0.003
    resumed = tmp_path / "resumed"
    main(args + ["--output", str(resumed), "--resume", str(first / "policy-000016.pt")])
    recovered, recovered_state = load_policy(resumed / "latest.pt")
    assert recovered_state["policy_sha256"] == saved["policy_sha256"]
    assert recovered.config == loaded.config
    assert ToyGenerator(recovered).generate(rows[:8], noise)["tokens"] == rollout["tokens"]
