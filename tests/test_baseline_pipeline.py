import torch
import pytest
from remarl.baselines.confidence import NativeAdapter, decode
from remarl.baselines.standard import decode as standard


class ToyAnswerEngine:
    def prepare(self, ids, attention, positions):
        def forward(current, mask, answer_positions):
            scores = torch.arange(17, dtype=torch.float32).expand(1, 256, 17)
            return scores.to(torch.bfloat16), None

        return forward


@pytest.mark.parametrize(
    "mode,block", [(m, b) for m in ("low_confidence", "random", "fastdllm") for b in (32, 256)]
)
def test_apple_pipeline(mode, block):
    prompt = torch.tensor([[2, 3]])
    adapter = NativeAdapter(ToyAnswerEngine(), prompt)
    cfg = dict(mode=mode, block=block, steps=8, threshold=0.9)
    result = decode(adapter, prompt, cfg)
    assert result.sequences.shape == (1, 258)
    assert 126336 not in result.sequences[0, 2:]
    assert adapter.calls == int(result.steps_taken[0])


@pytest.mark.parametrize("mode", ["p2_original", "margin", "pc_sampler", "klass", "eb_sampler"])
def test_standard_pipeline(mode):
    prompt = torch.tensor([[2, 3]])
    adapter = NativeAdapter(ToyAnswerEngine(), prompt)
    cfg = dict(mode=mode, block=256, steps=8, kl=0.015, confidence=0.6, gamma=0.1)
    result = standard(adapter, prompt, cfg, frequency=torch.ones(17) / 17)
    assert result.sequences.shape == (1, 258)
    assert 126336 not in result.sequences[0, 2:]
    assert adapter.calls == int(result.steps_taken[0])
