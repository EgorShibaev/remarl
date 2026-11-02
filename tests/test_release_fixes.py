"""Regressions for released recipes, temperature sweeps and mixed-policy loading."""

from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from remarl import evaluation
from remarl.cli import main
from remarl.optimization import replay_logp, state_digest
from remarl.runtime import load_policy
from test_lifecycle import ToyGenerator, ToyScorer, policy


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def checkpoint(tmp_path):
    torch.set_num_threads(1)

    def save(variant="full"):
        torch.manual_seed(81)
        model = policy(variant)
        # Exercise temperature scaling of learned scores, not only zero-init P2-prime.
        with torch.no_grad():
            model.linear_skip.weight[0, 0] = 2.0
            model.pointwise[-1].weight.fill_(0.1)
        path = tmp_path / f"{variant}.pt"
        torch.save(
            dict(
                model=model.state_dict(),
                selector_config=asdict(model.config),
                policy_sha256=state_digest(model.state_dict()),
                config={"variant": variant},
            ),
            path,
        )
        return path

    return save


@pytest.mark.parametrize("total", [8, 64, 512, 4096])
def test_every_recorded_data_size_seed_has_a_recipe(total):
    costs = json.loads((ROOT / "results/training-costs.json").read_text())["records"]
    runs = [r for r in costs if r["run"].startswith(f"n{total}-seed")]
    assert len(runs) == 3
    reference = json.loads((ROOT / f"configs/data_size/n{total}-seed7.json").read_text())
    for run in runs:
        cfg = json.loads((ROOT / "configs/data_size" / (run["run"] + ".json")).read_text())
        seed = int(run["run"].split("seed")[1])
        assert cfg == dict(reference, seed=seed)
        assert cfg["training_questions"] == 2 * cfg["per_dataset"] == total
        assert cfg["target_accepted_groups"] == run["accepted_groups"]
        assert cfg["nfe"] == run["nfe"]


@pytest.fixture
def toy_evaluation(monkeypatch):
    rows = [dict(prompt_ids=[2, 3], gold={})] * 3
    manifest = dict(rows_sha256="toy-data", domain="math", required_canvas=4096)
    generators = []

    def generator(model, loaded_policy, device, backend, max_length):
        result = ToyGenerator(loaded_policy)
        generators.append(result)
        return result

    monkeypatch.setattr(evaluation, "configure_compute", lambda: None)
    monkeypatch.setattr(evaluation, "load_prepared", lambda _: ({"test_gsm8k": rows}, manifest))
    monkeypatch.setattr(evaluation, "Generator", generator)
    monkeypatch.setattr(ToyGenerator, "tokenizer", None, raising=False)
    monkeypatch.setattr(evaluation, "Scorer", lambda *_: ToyScorer())
    return generators


@pytest.mark.parametrize("temperature", [0.0, 0.25, 1.0, 2.0])
@pytest.mark.parametrize("variant", ["full", "no_revision"])
def test_cli_evaluates_trained_policy_at_requested_temperature(
    tmp_path, checkpoint, toy_evaluation, temperature, variant
):
    path = checkpoint(variant)
    original_hash = torch.load(path, weights_only=True)["policy_sha256"]
    out = tmp_path / "evaluation"
    main(
        [
            "evaluate",
            "--policy",
            str(path),
            "--nfe",
            "8",
            "--temperature",
            str(temperature),
            "--model",
            "toy-model",
            "--data",
            "toy-data",
            "--output",
            str(out),
            "--device",
            "cpu",
            "--backend",
            "torch",
            "--seeds",
            "101",
            "--batch-size",
            "2",
        ]
    )
    loaded = toy_evaluation[-1].policy
    assert loaded.temperature == temperature
    assert state_digest(loaded.state_dict()) == original_hash
    for name in ["evaluation.json", "summary.json"]:
        metadata = json.loads((out / "test_gsm8k" / name).read_text())
        assert metadata["position_temperature"] == temperature
        assert metadata["policy_sha256"] == original_hash


@pytest.mark.parametrize("temperature", [-1, float("nan"), float("inf")])
def test_invalid_policy_temperature_is_rejected_before_model_load(
    tmp_path, checkpoint, toy_evaluation, temperature
):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        evaluation.run_evaluation(
            "data",
            "model",
            tmp_path,
            checkpoint(),
            8,
            [101],
            "cpu",
            "torch",
            temperature=temperature,
        )
    assert not toy_evaluation


def test_temperature_evaluation_resume_identity(tmp_path, checkpoint, toy_evaluation):
    args = ("data", "model", tmp_path / "eval", checkpoint(), 8, [101], "cpu", "torch")
    first = evaluation.run_evaluation(*args, temperature=1.0, batch_size=2)
    directory = tmp_path / "eval/test_gsm8k"
    answers = (directory / "answers-seed101.jsonl").read_bytes()
    # Before temperature sweeps were supported, trained-policy metadata omitted tau=1.
    path = directory / "evaluation.json"
    legacy = json.loads(path.read_text())
    legacy.pop("position_temperature", None)
    legacy.pop("confidence_residual", None)
    path.write_text(json.dumps(legacy))
    repeated = evaluation.run_evaluation(*args, temperature=1.0, batch_size=2)
    assert repeated == first
    with pytest.raises(ValueError, match="another evaluation"):
        evaluation.run_evaluation(*args, temperature=0.25, batch_size=2)
    assert (directory / "answers-seed101.jsonl").read_bytes() == answers


def test_trained_policy_does_not_accept_retention_control(tmp_path, checkpoint, toy_evaluation):
    with pytest.raises(ValueError, match="only to --p2-prime"):
        evaluation.run_evaluation(
            "data", "model", tmp_path, checkpoint(), 8, [101], "cpu", "torch", retention_bonus=1.0
        )
    assert not toy_evaluation


def test_evaluation_does_not_reuse_answers_from_another_scoring_rule(
    tmp_path, checkpoint, toy_evaluation
):
    path = checkpoint()
    args = ("data", "model", tmp_path / "eval", path, 8, [101], "cpu", "torch")
    evaluation.run_evaluation(*args, batch_size=2)
    saved = torch.load(path, weights_only=True)
    saved["selector_config"]["confidence_residual"] = False
    torch.save(saved, path)
    with pytest.raises(ValueError, match="another evaluation"):
        evaluation.run_evaluation(*args, batch_size=2)


def test_loaded_policy_temperatures_are_independent(checkpoint):
    path = checkpoint("no_revision")
    first, _ = load_policy(path)
    first.temperature = 0.25
    rows = [dict(prompt_ids=[2, 3], gold={})] * 2
    noise = torch.rand(8, 2, 256, generator=torch.Generator().manual_seed(46))
    before = ToyGenerator(first).generate(rows, noise, True)
    second, _ = load_policy(path)
    second.temperature = 2.0
    ToyGenerator(second).generate(rows, noise, True)
    after = ToyGenerator(first).generate(rows, noise, True)
    assert first.temperature == 0.25
    assert before["tokens"] == after["tokens"]
    assert torch.equal(before["replay"].old_logp, after["replay"].old_logp)


@pytest.mark.parametrize("other_variant", ["full", "no_hidden", "no_position", "bias_only"])
def test_loading_another_checkpoint_preserves_no_revision_sampling_and_replay(
    checkpoint, other_variant
):
    first_path = checkpoint("no_revision")
    other_path = checkpoint(other_variant)
    first, _ = load_policy(first_path)
    engine = ToyGenerator(first)
    rows = [dict(prompt_ids=[2, 3], gold={})] * 2
    noise = torch.rand(8, 2, 256, generator=torch.Generator().manual_seed(44))
    before = engine.generate(rows, noise, True)
    replay = before["replay"]
    expected_logp = [
        replay_logp(first, f, s, a).detach().clone()
        for f, s, a in zip(replay.features, replay.scores, replay.orders)
    ]
    second, _ = load_policy(other_path)
    assert second is not first
    after = engine.generate(rows, noise, True)
    assert before["revisions"] == after["revisions"] == [0, 0]
    assert before["tokens"] == after["tokens"]
    for a, b in zip(replay.orders, after["replay"].orders):
        assert torch.equal(a, b)
    for f, s, a, expected in zip(replay.features, replay.scores, replay.orders, expected_logp):
        torch.testing.assert_close(replay_logp(first, f, s, a), expected, atol=0, rtol=0)


@pytest.mark.parametrize("variant", ["full", "no_revision"])
@pytest.mark.parametrize("temperature", [0.0, 0.25, 1.0, 2.0])
def test_policy_temperature_matches_sampling_and_replay(checkpoint, variant, temperature):
    first, _ = load_policy(checkpoint(variant))
    first.temperature = temperature
    engine = ToyGenerator(first)
    rows = [dict(prompt_ids=[2, 3], gold={})] * 2
    noise = torch.rand(8, 2, 256, generator=torch.Generator().manual_seed(45))
    result = engine.generate(rows, noise, True)["replay"]
    for f, scores, order, old in zip(
        result.features, result.scores, result.orders, result.old_logp
    ):
        z = first(f.flatten(0, 1), scores.flatten()).reshape_as(scores)
        masked = f[..., 1].bool()
        # Evaluate the action probability independently over its candidate set.
        expected = []
        for row in range(len(z)):
            pool = masked[row].clone() if variant == "no_revision" else torch.ones_like(masked[row])
            chosen = order[row][masked[row, order[row]]] if variant == "no_revision" else order[row]
            value = z[row].sum() * 0
            if temperature == 0:
                assert torch.equal(
                    chosen,
                    torch.where(pool)[0][
                        z[row, pool].argsort(descending=True, stable=True)[: len(chosen)]
                    ],
                )
            else:
                for index in chosen:
                    value = (
                        value
                        + z[row, index] / temperature
                        - torch.logsumexp(z[row, pool] / temperature, 0)
                    )
                    pool[index] = False
            expected.append(value)
        torch.testing.assert_close(old, torch.stack(expected), atol=3e-4, rtol=1e-5)
        torch.testing.assert_close(replay_logp(first, f, scores, order), old, atol=3e-4, rtol=1e-5)
