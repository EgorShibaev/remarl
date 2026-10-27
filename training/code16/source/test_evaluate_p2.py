"""CPU-only evaluation contract gates (no model download, no GPU)."""

import pytest
import torch

from evaluate_p2 import P2Scores, batch_indices, draws, question_seed, validate_records
from test_worker import FakeBackbone
from worker import collect
from src.decoding.threshold_decoder import ConfidenceZSource, P2Decoder, P2DecoderConfig


@pytest.mark.parametrize("nfe", [8, 16])
@pytest.mark.parametrize("seed", [101, 202])
def test_p2_collect_matches_legacy_decoder(nfe, seed):
    bb = FakeBackbone()
    ids = torch.tensor([[1] + [19]*16 + [0], [2, 3] + [19]*16])
    mask = torch.ones_like(ids)
    mask[0, -1] = 0
    pos = torch.stack([torch.arange(1, 17), torch.arange(2, 18)])
    u = torch.stack([draws(i, nfe, seed, 16) for i in range(2)], 1)
    result = collect(bb, P2Scores(), ids, mask, pos, u, record=False)
    for i in range(2):
        decoder = P2Decoder(bb, P2DecoderConfig(steps=nfe, policy_temperature=1.,
                            policy_seed=question_seed(i, nfe, seed)), ConfidenceZSource())
        old = decoder.decode(ids[i:i+1], mask[i:i+1], editable_positions=[pos[i]])
        assert torch.equal(result.final_ids[i], old.input_ids[0])
        assert int(result.revisions[i]) == old.metadata["remask_count"]


def test_draws_preserve_original_per_question_cpu_rng():
    g = torch.Generator().manual_seed(question_seed(100, 16, 202))
    expected = torch.stack([torch.rand(256, generator=g) for _ in range(16)])
    assert torch.equal(draws(100, 16, 202), expected)
    assert not torch.equal(draws(100, 16, 101), expected)


def test_fixed_tail_coverage_excludes_padding_duplicates():
    indices, real = batch_indices(1312)
    assert indices == [1312, 1313, 1314, 1315, 1316, 1317, 1318, 1318]
    assert real == 7
    full = [i for start in range(0, 1319, 8)
            for i in batch_indices(start)[0][:batch_indices(start)[1]]]
    assert full == list(range(1319))


def test_coverage_audit_rejects_missing_and_duplicate_questions():
    rows = [dict(index=i, source_id=f"gsm8k-test:{i}", answer_token_ids=[1]*256)
            for i in range(2)]
    validate_records(rows, count=2)
    with pytest.raises(AssertionError):
        validate_records(rows[:1], count=2)
    with pytest.raises(AssertionError):
        validate_records([rows[0], rows[0]], count=2)
