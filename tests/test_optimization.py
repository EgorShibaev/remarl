import copy
from types import SimpleNamespace
import pytest
import torch
from remarl.optimization import (
    QuestionSampler,
    cosine_lr,
    informative,
    minibatches,
    optimize_minibatch,
    optimize_collection,
    replay_chunks,
    replay_error,
    replay_logp,
    state_digest,
)
from remarl.policy import ReMaRLPolicy as SetSelector, PolicyConfig as SetSelectorConfig
from remarl.grpo import clipped_surrogate, group_advantages
from remarl.rollout import sample_order


@pytest.fixture
def example():
    torch.set_num_threads(1)
    torch.manual_seed(71)
    model = SetSelector(
        SetSelectorConfig(
            input_node_dim=3,
            architecture="mlp",
            hidden_dim=12,
            position_encoding="learned",
            position_column=2,
            position_count=6,
            backbone_hidden_dim=4,
            hidden_projection_dim=4,
        )
    ).eval()
    groups = []
    for i in range(16):
        features = torch.randn(3, 8, 6, 7)
        features[..., 2] = torch.arange(6)
        scores = torch.rand(3, 8, 6)
        orders, old = [], []
        with torch.no_grad():
            for step, count in enumerate((2, 3, 5)):
                z = model(features[step].flatten(0, 1), scores[step].flatten()).reshape(8, 6)
                order = sample_order(z, torch.rand(8, 6), count)
                orders.append(order)
                old.append(replay_logp(model, features[step], scores[step], order))
        groups.append(
            dict(
                question_id=i,
                rewards=[float(j < (i % 7 + 1)) for j in range(8)],
                replay=SimpleNamespace(
                    features=features,
                    scores=scores,
                    orders=orders,
                    old_logp=torch.stack(old),
                    counts=[2, 3, 5],
                ),
            )
        )
    return model, groups


@pytest.mark.parametrize(
    "rewards,result",
    [([0] * 8, False), ([1] * 8, False), ([0] * 7 + [1], True), ([1] * 7 + [0], True)],
)
def test_acceptance(rewards, result):
    assert informative(rewards) == result


@pytest.mark.parametrize("rewards", [[0] * 7, [0] * 9, [0] * 7 + [0.5], [0] * 7 + [float("nan")]])
def test_invalid_rewards(rewards):
    with pytest.raises(ValueError):
        informative(rewards)


def test_minibatches_keep_groups_and_use_each_once(example):
    _, groups = example
    parts = minibatches(groups, torch.Generator().manual_seed(3))
    assert [len(p) for p in parts] == [8, 8]
    assert sorted(g["question_id"] for p in parts for g in p) == list(range(16))
    assert len({id(g) for p in parts for g in p}) == 16


def test_no_partial_batches(example):
    _, groups = example
    with pytest.raises(ValueError):
        minibatches(groups[:15], torch.Generator())
    groups[0]["rewards"] = [0] * 8
    with pytest.raises(ValueError):
        minibatches(groups, torch.Generator())


def test_sampler_resume_and_distinctness():
    a = QuestionSampler(25, 7)
    seen = set()
    for _ in range(20):
        index = a.next(seen)
        assert index not in seen
        seen.add(index)
    b = QuestionSampler(25, 999)
    b.load_state_dict(a.state_dict())
    assert [a.next(set()) for _ in range(100)] == [b.next(set()) for _ in range(100)]


def test_group_normalization_is_not_global(example):
    _, groups = example
    adv = [group_advantages(torch.tensor(g["rewards"])) for g in groups]
    for value in adv:
        torch.testing.assert_close(value.mean(), torch.tensor(0.0), atol=1e-7, rtol=0)
        torch.testing.assert_close(value.std(unbiased=False), torch.tensor(1.0))
    assert not torch.equal(
        torch.cat(adv), group_advantages(torch.tensor([r for g in groups for r in g["rewards"]]))
    )


@pytest.mark.parametrize("question_count", [8, 16])
def test_streaming_gradient_matches_complete_multiquestion_objective(example, question_count):
    model, groups = example
    groups = groups[:question_count]
    a, b = copy.deepcopy(model), copy.deepcopy(model)
    opt_a = torch.optim.AdamW(a.parameters(), lr=3e-5, weight_decay=0.0)
    opt_b = torch.optim.AdamW(b.parameters(), lr=3e-5, weight_decay=0.0)
    optimize_minibatch(a, opt_a, groups, "cpu", groups_per_chunk=2)
    losses = []
    for features, scores, orders, old, advantages in replay_chunks(groups, "cpu", 8):
        losses.append(
            clipped_surrogate(replay_logp(b, features, scores, orders), old, advantages, 0.2)
        )
    torch.cat(losses).mean().backward()
    torch.nn.utils.clip_grad_norm_(b.parameters(), 1.0, error_if_nonfinite=True)
    opt_b.step()
    for (name, p), (other_name, q) in zip(a.named_parameters(), b.named_parameters()):
        assert name == other_name
        # Ordered-PL is invariant to a common score bias. Tiny FP32 reduction
        # differences in its near-zero gradient can be amplified by Adam's eps.
        torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=1e-5)
        if name == "pointwise.4.bias":
            # This single scalar is an unidentifiable common-logit offset.
            # Adam can amplify rounding of its mathematically-zero gradient.
            assert p.numel() == 1 and p.grad.abs().max() < 1e-7 and q.grad.abs().max() < 1e-7
        else:
            torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-5)
    with torch.no_grad():
        for features, scores, orders, _, _ in replay_chunks(groups, "cpu", 8):
            torch.testing.assert_close(
                replay_logp(a, features, scores, orders),
                replay_logp(b, features, scores, orders),
                atol=1e-5,
                rtol=0,
            )


def test_common_output_bias_does_not_change_policy(example):
    model, groups = example
    shifted = copy.deepcopy(model)
    with torch.no_grad():
        shifted.pointwise[-1].bias.add_(0.25)
        for features, scores, orders, _, _ in replay_chunks(groups, "cpu", 8):
            torch.testing.assert_close(
                replay_logp(model, features, scores, orders),
                replay_logp(shifted, features, scores, orders),
                atol=1e-5,
                rtol=0,
            )


def test_behavior_probs_stay_frozen_and_second_step_is_offpolicy(example):
    model, groups = example
    old = [g["replay"].old_logp.clone() for g in groups]
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    assert replay_error(model, groups, "cpu") < 1e-5
    parts = minibatches(groups, torch.Generator().manual_seed(42))
    first = optimize_minibatch(model, opt, parts[0], "cpu")
    second = optimize_minibatch(model, opt, parts[1], "cpu")
    assert first["optimizer_steps"] == second["optimizer_steps"] == 1
    assert second["logratio_abs"] > 0.001
    assert second["parameter_delta_norm"] > 0
    assert all(torch.equal(a, g["replay"].old_logp) for a, g in zip(old, groups))


def test_optimizer_resume_exact(example, tmp_path):
    model, groups = example
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.0)
    optimize_minibatch(model, opt, groups[:8], "cpu")
    path = tmp_path / "state.pt"
    torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict()), path)
    optimize_minibatch(model, opt, groups[8:], "cpu")
    expected = state_digest(model.state_dict())
    saved = torch.load(path, weights_only=True)
    model.load_state_dict(saved["model"])
    opt.load_state_dict(saved["optimizer"])
    optimize_minibatch(model, opt, groups[8:], "cpu")
    assert state_digest(model.state_dict()) == expected


def test_cosine_axis_is_collected_batches():
    assert cosine_lr(0, 2048) == pytest.approx(3e-5)
    assert cosine_lr(2047, 2048) == pytest.approx(3e-6)
    assert cosine_lr(1000, 2048) > cosine_lr(1001, 2048)


def test_single_batch64_update_uses_all_groups_once(example):
    model, groups = example
    groups = groups[:8]
    parts = minibatches(groups, torch.Generator().manual_seed(9), expected_groups=8)
    assert len(parts) == 1 and len(parts[0]) == 8
    assert {g["question_id"] for g in parts[0]} == set(range(8))
    old = [g["replay"].old_logp.clone() for g in groups]
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.0)
    result = optimize_minibatch(model, optimizer, parts[0], "cpu")
    assert result["optimizer_steps"] == 1 and result["trajectories"] == 64
    assert result["decisions"] == 64 * 3
    assert all(torch.equal(a, b["replay"].old_logp) for a, b in zip(old, groups))
    with pytest.raises(ValueError):
        minibatches(groups[:7], torch.Generator(), expected_groups=8)


def test_four_updates_preserve_behavior_and_exact_replay(example):
    model, groups = example
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    rng = torch.Generator().manual_seed(7002)
    before = copy.deepcopy(model.state_dict())
    opt_before = copy.deepcopy(optimizer.state_dict())
    rng_before = rng.get_state().clone()
    old = [g["replay"].old_logp.clone() for g in groups]
    rewards = [list(g["rewards"]) for g in groups]
    updates = optimize_collection(model, optimizer, groups, "cpu", rng, 16, 4, 16)
    assert len(updates) == 4
    assert [u["pass_index"] for u in updates] == [1, 2, 3, 4]
    assert [u["minibatch_index"] for u in updates] == [1, 1, 1, 1]
    for epoch in (1, 2, 3, 4):
        ids = [q for u in updates if u["pass_index"] == epoch for q in u["question_ids"]]
        assert sorted(ids) == list(range(16))
    assert all(u["trajectories"] == 128 and u["optimizer_steps"] == 1 for u in updates)
    assert all(u["decisions"] == 128 * 3 for u in updates)
    assert all(u["parameter_delta_norm"] > 0 for u in updates)
    assert updates[1]["logratio_abs"] > updates[0]["logratio_abs"]
    assert all(
        torch.equal(a, g["replay"].old_logp) and b == g["rewards"]
        for a, b, g in zip(old, rewards, groups)
    )
    after = copy.deepcopy(model.state_dict())
    after_optimizer = copy.deepcopy(optimizer.state_dict())
    model.load_state_dict(before)
    optimizer.load_state_dict(opt_before)
    rng.set_state(rng_before)
    repeated = optimize_collection(model, optimizer, groups, "cpu", rng, 16, 4, 16)
    assert updates == repeated and state_digest(model.state_dict()) == state_digest(after)
    for k, fields in optimizer.state_dict()["state"].items():
        assert all(torch.equal(v, after_optimizer["state"][k][name]) for name, v in fields.items())
