import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from remarl.evaluation import evaluate
from remarl.policy import ReMaRLPolicy, PolicyConfig
from remarl.rollout import collect
from remarl.runtime import canvas, load_policy
from remarl.training import Trainer
from remarl.optimization import replay_error
from remarl.ablations import selector_class
from remarl.sampling import P2Scores


class ToyGenerator:
    def __init__(self, policy):
        self.policy = policy

    def forward(self, ids, attention, positions):
        values = ids.gather(1, positions).float()
        hidden = torch.stack((values / 10, positions / 256, attention.gather(1, positions)), -1)
        vocab = torch.arange(12).float()
        logits = torch.sin(values[..., None] / 7 + vocab)
        logits[..., 11] = -100
        return logits, hidden

    def generate(self, rows, uniforms, record=False):
        ids, attention, positions = canvas(rows, 0, uniforms.shape[-1], 11)
        result = collect(
            SimpleNamespace(mask_token_id=11),
            self.policy,
            ids,
            attention,
            positions,
            uniforms,
            answer_backbone=self.forward,
            record=record,
        )
        return dict(
            tokens=result.final_ids.gather(1, positions).tolist(),
            replay=result,
            revisions=result.revisions.tolist(),
        )


class ToyScorer:
    def begin_collection(self):
        pass

    def close(self):
        pass

    def __call__(self, tokens, golds):
        # A controlled mixed-reward group tests filtering, replay, and checkpoint state.
        return [dict(reward=float(i % 2), text=str(t[:3])) for i, t in enumerate(tokens)]


def policy(variant="full"):
    cls = ReMaRLPolicy if variant == "full" else selector_class(ReMaRLPolicy, variant)
    return cls(PolicyConfig(hidden_dim=8, backbone_hidden_dim=3, hidden_projection_dim=4)).eval()


def test_training_save_resume_and_inference(tmp_path):
    torch.set_num_threads(1)
    cfg = json.loads((Path(__file__).parents[1] / "configs/math/nfe4.json").read_text())
    cfg.update(target_accepted_groups=32, save_every_groups=16)
    rows = [dict(prompt_ids=[2, i % 8 + 1], gold={}) for i in range(64)]
    model = policy()
    trainer = Trainer(
        cfg,
        rows,
        {"fixture": True},
        tmp_path / "first",
        [ToyGenerator(copy.deepcopy(model))],
        ToyScorer(),
        model,
        "cpu",
    )
    trainer.step()
    saved = torch.load(tmp_path / "first/latest.pt", weights_only=True)
    loaded, _ = load_policy(tmp_path / "first/latest.pt")
    assert loaded.config == model.config
    trainer.step()
    expected = trainer.policy_hash
    trainer.close()
    resumed_model = policy()
    resumed = Trainer(
        cfg,
        rows,
        {"fixture": True},
        tmp_path / "resumed",
        [ToyGenerator(copy.deepcopy(resumed_model))],
        ToyScorer(),
        resumed_model,
        "cpu",
    )
    try:
        resumed.restore(saved)
        resumed.run()
        assert resumed.policy_hash == expected
        assert resumed.completed == 2
        assert (tmp_path / "resumed/policy-000032.pt").exists()
    finally:
        resumed.close()


@pytest.mark.parametrize(
    "variant", ["full", "no_hidden", "no_position", "no_revision", "bias_only"]
)
def test_ablation_rollout_replay(variant):
    model = policy(variant)
    engine = ToyGenerator(model)
    rows = [dict(prompt_ids=[2, 3], gold={})] * 8
    result = engine.generate(rows, torch.rand(8, 8, 256), True)
    groups = [dict(question_id=0, rewards=[0, 1] * 4, replay=result["replay"])]
    assert replay_error(model, groups, "cpu") < 0.003
    if variant == "no_revision":
        assert result["revisions"] == [0] * 8
    assert all(11 not in tokens for tokens in result["tokens"])


def test_evaluation_resumes_partial_batch(tmp_path):
    model = policy()
    rows = [dict(prompt_ids=[2, 3], gold={})] * 11
    args = dict(nfe=4, seeds=[101], identity={"fixture": True}, batch_size=8)
    expected = evaluate(ToyGenerator(model), ToyScorer(), rows, tmp_path, **args)
    path = tmp_path / "answers-seed101.jsonl"
    complete = path.read_text()
    path.write_text("\n".join(complete.splitlines()[:3]) + "\n")
    actual = evaluate(ToyGenerator(model), ToyScorer(), rows, tmp_path, **args)
    assert actual == expected
    assert path.read_text() == complete


def test_greedy_control_ignores_position_noise():
    generator = ToyGenerator(P2Scores(temperature=0))
    rows = [dict(prompt_ids=[2, 3], gold={})] * 8
    first = generator.generate(rows, torch.rand(8, 8, 256), True)
    second = generator.generate(rows, torch.rand(8, 8, 256), True)
    assert first["tokens"] == second["tokens"]
    assert all(torch.equal(a, b) for a, b in zip(first["replay"].orders, second["replay"].orders))
