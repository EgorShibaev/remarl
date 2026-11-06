import json
from types import SimpleNamespace

import pytest
import torch

from remarl.dream import DreamAnswerModel
from remarl.policy import ReMaRLPolicy, PolicyConfig
from remarl.runtime import backbone_spec, canvas, check_prepared_backbone
from remarl.rollout import collect
from remarl.optimization import replay_error


class FakeDream(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = torch.nn.Linear(3, 12, bias=False)
        self.last_call = None

    def model(self, input_ids, attention_mask, position_ids, **kwargs):
        self.last_call = (attention_mask, position_ids, kwargs)
        h = torch.stack((input_ids.float() / 10, position_ids.float(), input_ids.float() % 3), -1)
        return SimpleNamespace(last_hidden_state=h)


def test_dream_shift_matches_native_logits_and_features():
    native = FakeDream()
    adapter = DreamAnswerModel(native)
    ids = torch.tensor([[2, 4, 11, 11, 0], [3, 5, 6, 11, 11]])
    valid = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    positions = torch.tensor([[2, 3], [3, 4]])
    logits, hidden = adapter(ids, valid, positions)
    mask, token_positions, kwargs = native.last_call
    expected = native.model(ids, mask, token_positions).last_hidden_state
    full_logits = native.lm_head(expected)
    shifted = torch.cat((full_logits[:, :1], full_logits[:, :-1]), 1)
    assert torch.equal(logits, shifted.gather(1, positions[..., None].expand(-1, -1, 12)))
    assert torch.equal(hidden, expected.gather(1, (positions - 1)[..., None].expand(-1, -1, 3)))
    assert not mask[0, :, 4, :].any() and not mask[0, :, :, 4].any()
    assert mask[1, 0, 0, -1]  # Future answer tokens remain visible to attention.
    assert kwargs["use_cache"] is False
    assert all(not p.requires_grad for p in native.parameters())


def test_dream_rollout_uses_native_mask_and_replays():
    torch.set_num_threads(1)
    policy = ReMaRLPolicy(PolicyConfig(backbone_hidden_dim=3)).eval()
    native = FakeDream()
    with torch.no_grad():
        native.lm_head.weight[11].fill_(-100)
    backend = DreamAnswerModel(native)
    ids, attention, positions = canvas([dict(prompt_ids=[2, 3])] * 8, 0, 256, 11)
    result = collect(
        SimpleNamespace(mask_token_id=11),
        policy,
        ids,
        attention,
        positions,
        torch.rand(8, 8, 256),
        answer_backbone=backend,
        record=True,
    )
    assert result.nfe == 8
    assert (result.final_ids[:, :2] == ids[:, :2]).all()
    assert not result.final_ids.gather(1, positions).eq(11).any()
    groups = [dict(question_id=0, rewards=[0, 1] * 4, replay=result)]
    assert replay_error(policy, groups, "cpu") < 0.003


def test_backbone_specs_and_prompt_identity(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(dict(model_type="Dream", hidden_size=3584, mask_token_id=151666)))
    dream = backbone_spec(tmp_path)
    assert dream["hidden_size"] == 3584 and dream["mask_id"] == 151666
    p.write_text(json.dumps(dict(model_type="llada", d_model=4096)))
    llada = backbone_spec(tmp_path)
    assert llada["hidden_size"] == 4096 and llada["mask_id"] == 126336
    with pytest.raises(ValueError, match="another generator"):
        check_prepared_backbone(dict(backbone=llada), dream)
    check_prepared_backbone(dict(backbone=dream), dream)
