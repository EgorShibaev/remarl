"""Cheap behavioral gates before allocating time to real LLaDA trajectories."""

from unittest.mock import patch

import pytest
import torch
from src.rl.packed_replay import PackedReplay
from worker import (
    CapturedForward,
    collect,
    initial_representatives,
    ordered_logprob,
    sample_order,
    transition,
)

from src.decoding.selection import topk_set_logprob
from src.models.set_selector import SetSelector, SetSelectorConfig
from src.rl.rollout import Rollout, StepRecord, rollout_group


class FakeBackbone:
    mask_token_id = 19

    def forward(self, ids, attention_mask, timestep=0):
        # Context within each request; no cross-request reductions.
        context = (ids * attention_mask).sum(-1)[:, None, None] / 100
        pos = torch.arange(ids.shape[1]).float()[None, :, None]
        vocab = torch.arange(20).float()[None, None, :]
        logits = torch.cos(pos * 0.7 + vocab + context + ids[..., None] * 0.13)
        hidden = torch.sin(pos + torch.arange(4).float()[None, None, :] + context)
        return {"logits": logits, "hidden_states": hidden}


def inputs(nfe=4):
    torch.manual_seed(31)
    policy = SetSelector(
        SetSelectorConfig(
            input_node_dim=3,
            architecture="mlp",
            hidden_dim=12,
            position_encoding="learned",
            position_column=2,
            position_count=8,
            backbone_hidden_dim=4,
            hidden_projection_dim=4,
            dropout=0.0,
        )
    ).eval()
    with torch.no_grad():
        policy.pointwise[-1].weight.normal_(0, 0.2)
    ids = torch.tensor([[1, 19, 19, 19, 19, 19, 19, 0], [2, 3, 19, 19, 19, 19, 19, 19]])
    mask = torch.ones_like(ids)
    mask[0, -1] = 0
    positions = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])
    uniforms = torch.rand(nfe, 2, 6)
    return FakeBackbone(), policy, ids, mask, positions, uniforms


def reference_row(bb, policy, ids, mask, positions, uniforms):
    draws = iter(uniforms)

    def sampler(z, k, temperature, generator=None):
        order = sample_order(z[None], next(draws)[None], k, temperature)[0]
        return order.sort().values, order

    with patch("src.rl.rollout.stochastic_topk", sampler):
        return rollout_group(
            bb,
            policy,
            ids,
            mask,
            positions,
            [len(uniforms)],
            1.0,
            feature_spec=("score", "is_masked", "position_id"),
        )[0]


@pytest.mark.parametrize("nfe", [1, 4, 6, 8])
def test_vectorized_collection_matches_per_request_reference(nfe):
    bb, policy, ids, mask, positions, u = inputs(nfe)
    result = collect(bb, policy, ids, mask, positions, u)
    for row in range(2):
        ref = reference_row(bb, policy, ids[row], mask[row], positions[row], u[:, row])
        assert torch.equal(result.final_ids[row], ref.final_input_ids)
        assert result.revisions[row] == ref.remask_count
        assert result.repeat_revisions[row] == ref.repeat_remask_count
        assert result.nfe == ref.nfe == nfe
        included = [r for r in ref.records if r.included]
        assert len(result.counts) == len(included)
        for step, record in enumerate(included):
            torch.testing.assert_close(result.features[step, row], record.node_features)
            torch.testing.assert_close(result.scores[step, row], record.scores)
            torch.testing.assert_close(result.old_logp[step, row], torch.tensor(record.logp_old))
            assert torch.equal(result.orders[step][row], record.order)
    # Padding and the prompt must remain unchanged.
    editable = torch.zeros_like(ids, dtype=torch.bool).scatter_(1, positions, True)
    assert torch.equal(result.final_ids[~editable], ids[~editable])


def test_ordered_probability_and_gradients_match_reference():
    torch.manual_seed(18)
    scores = torch.randn(4, 11, requires_grad=True)
    order = sample_order(scores.detach(), torch.rand_like(scores), 7)
    actual = ordered_logprob(scores, order, 0.7)
    expected = torch.stack([topk_set_logprob(s, o, 0.7) for s, o in zip(scores, order)])
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual.sum(), scores, retain_graph=True)[0]
    ge = torch.autograd.grad(expected.sum(), scores)[0]
    torch.testing.assert_close(ga, ge)


def test_replay_features_preserve_trainable_hidden_projection_gradient():
    bb, policy, ids, mask, positions, u = inputs()
    result = collect(bb, policy, ids, mask, positions, u)
    refs = [reference_row(bb, policy, ids[r], mask[r], positions[r], u[:, r]) for r in range(2)]
    converted = []
    for row in range(2):
        records = [
            StepRecord(
                result.features[s, row],
                result.scores[s, row],
                result.orders[s][row],
                float(result.old_logp[s, row]),
                True,
                result.masks[s, row],
                k,
            )
            for s, k in enumerate(result.counts)
        ]
        converted.append(Rollout(4, records=records))

    def gradients(rollouts):
        policy.zero_grad(set_to_none=True)
        replay = PackedReplay(rollouts, torch.tensor([1.0, -1.0]), "cpu", chunk_size=2)
        logp = replay.log_probs(policy, 1.0)
        (logp * replay.advantages).mean().backward()
        return logp.detach(), {n: p.grad.clone() for n, p in policy.named_parameters()}

    values, grads = gradients(converted)
    refvalues, refgrads = gradients(refs)
    torch.testing.assert_close(values, refvalues)
    for name in grads:
        torch.testing.assert_close(grads[name], refgrads[name], atol=2e-6, rtol=2e-5)
    assert grads["hidden_projection.1.weight"].abs().max() > 0


def test_frozen_question_batch_is_independent_of_request_order():
    bb, policy, ids, mask, positions, u = inputs()
    a = collect(bb, policy, ids, mask, positions, u)
    b = collect(bb, policy, ids.flip(0), mask.flip(0), positions.flip(0), u.flip(1))
    assert torch.equal(a.final_ids, b.final_ids.flip(0))


def test_kept_visible_tokens_are_not_overwritten():
    ids, predicted = torch.tensor([[5, 19, 7]]), torch.tensor([[1, 2, 3]])
    following, revisions = transition(ids, predicted, torch.tensor([[0, 1]]), 19)
    assert following.tolist() == [[5, 2, 19]]
    assert revisions.tolist() == [[False, False, True]]


def test_host_budget_rejected_before_collection():
    with pytest.raises(MemoryError):
        collect(*inputs(), host_limit_bytes=1)


def test_rejects_out_of_range_or_repeated_answer_positions():
    bb, policy, ids, mask, positions, u = inputs()
    positions[0, 0] = positions[0, 1]
    with pytest.raises(ValueError):
        collect(bb, policy, ids, mask, positions, u)


@pytest.mark.parametrize("nfe", [1, 4, 8])
def test_initial_reuse_preserves_records_with_multiple_questions(nfe):
    bb, policy, ids, mask, positions, u = inputs(nfe)
    ids, mask, positions = (x.repeat_interleave(2, 0) for x in (ids, mask, positions))
    u = torch.rand(nfe, 4, 6)
    expected = collect(bb, policy, ids, mask, positions, u)
    with patch.object(bb, "forward", wraps=bb.forward) as forward:
        actual = collect(bb, policy, ids, mask, positions, u, reuse_initial=True)
    assert [len(call.args[0]) for call in forward.call_args_list] == [2] + [4] * (nfe - 1)
    for field in ("final_ids", "revisions", "repeat_revisions", "features", "scores", "old_logp"):
        torch.testing.assert_close(getattr(actual, field), getattr(expected, field), atol=0, rtol=0)
    for a, b in zip(actual.orders, expected.orders):
        assert torch.equal(a, b)


def test_initial_reuse_key_includes_padding_and_answer_indices():
    _, _, ids, mask, positions, _ = inputs()
    ids, mask, positions = (x[:1].repeat(3, 1) for x in (ids, mask, positions))
    mask[1, -1] = 1
    positions[2] += 1
    representatives, inverse = initial_representatives(ids, mask, positions)
    assert representatives.tolist() == [0, 1, 2]
    assert inverse.tolist() == [0, 1, 2]


def test_captured_forward_retains_bias_and_updates_all_mutable_inputs():
    import gc
    import weakref

    bias = torch.tensor([3.0])
    bias_ref = weakref.ref(bias)
    ids, positions = torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, 1, dtype=torch.long)
    logits, hidden = torch.zeros(1), torch.zeros(1)

    class FakeGraph:
        def replay(self):
            assert bias_ref() is not None
            logits.copy_(ids.sum() + bias_ref())
            hidden.copy_(positions.sum())

    captured = CapturedForward(FakeGraph(), ids, positions, bias, logits, hidden)
    del bias
    gc.collect()
    for value in (1, 2, 1):
        actual, h = captured(torch.full_like(ids, value), None, torch.full_like(positions, value))
        assert actual.item() == 3 * value + 3
        assert h.item() == value
    with pytest.raises(ValueError):
        captured(torch.zeros(2, 3), None, positions)
    del captured
    gc.collect()
    assert bias_ref() is None
