"""Dream alignment, block scheduling, and native adaptive call accounting."""

from types import SimpleNamespace

import pytest
import torch

from remarl.baselines.dream import DreamBaseline
from remarl.dream import DreamAnswerModel
from remarl.protocols.official.apple_generation import generate_unified


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trace = []

    def forward(self, input_ids, *args, **kwargs):
        self.trace.append(input_ids.clone())
        position = torch.arange(input_ids.shape[1])[None, :, None]
        vocabulary = torch.arange(17)[None, None, :]
        phase = input_ids.remainder(17).sum(-1)[:, None, None] * .017
        logits = torch.sin(position * .41 + vocabulary * 1.31 + phase)
        logits = logits * (1 + position * .011)
        logits[..., 16] = -100  # Mask is not a valid proposed token.
        return SimpleNamespace(last_hidden_state=logits)


class Model(torch.nn.Module):
    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.model = Decoder()
        self.lm_head = torch.nn.Linear(17, 17, bias=False)
        self.lm_head.weight.data.copy_(torch.eye(17))
        self.config = SimpleNamespace(vocab_size=17, mask_token_id=16)

    def forward(self, input_ids, *args, **kwargs):
        hidden = self.model(input_ids, *args, **kwargs).last_hidden_state
        return SimpleNamespace(logits=self.lm_head(hidden))


def adapter():
    torch.set_num_threads(1)
    result = object.__new__(DreamBaseline)
    result.model = Model()
    result.engine = DreamAnswerModel(result.model)
    result.spec = dict(kind="dream", mask_id=16)
    result.tokenizer = SimpleNamespace(mask_token_id=16)
    result.calls = 0
    result.model.model.register_forward_pre_hook(result.on_forward)
    return result


@pytest.mark.parametrize("block", [32, 256])
@pytest.mark.parametrize("nfe", [8, 16, 32])
def test_confidence_matches_apple_dream_trace(block, nfe):
    prompt = torch.tensor([[2, 3, 4]])
    actual, reference = adapter(), Model()
    result = actual.decode(prompt, dict(mode="low_confidence", block=block, steps=nfe))
    expected = generate_unified(reference, prompt, "low_confidence", steps=nfe,
        gen_length=256, block_length=block, mask_id=16, model_type="Dream")
    assert torch.equal(result.sequences, expected.sequences)
    assert actual.calls == len(reference.model.trace) == nfe
    assert all(torch.equal(a, b) for a, b in zip(actual.model.model.trace, reference.model.trace))


@pytest.mark.parametrize("block", [32, 256])
@pytest.mark.parametrize("mode", ["fastdllm", "eb", "klass"])
def test_native_adaptive_completion_and_block_boundaries(upstream_sources, block, mode):
    model = adapter()
    prompt = torch.tensor([[2, 3, 4]])
    result = model.decode(prompt, dict(mode=mode, block=block, parameter=.5))
    assert result.sequences.shape == (1, 259)
    assert 16 not in result.sequences[0, 3:]
    assert result.steps_taken.item() == len(model.model.model.trace)
    for state in model.model.model.trace:
        answer = state[0, 3:]
        masked = answer.eq(16).nonzero().flatten()
        if masked.numel():
            current_block = int(masked[0]) // block
            assert answer[(current_block + 1) * block:].eq(16).all()


def test_bad_native_call_count_is_rejected(monkeypatch):
    model = adapter()
    prompt = torch.tensor([[2, 3]])
    monkeypatch.setattr("remarl.baselines.dream.generate_unified",
        lambda *a, **kw: SimpleNamespace(sequences=torch.ones(1, 258, dtype=torch.long),
                                        steps_taken=torch.tensor([1])))
    with pytest.raises(RuntimeError, match="model-call counts"):
        model.decode(prompt, dict(mode="fastdllm", block=256, parameter=.5))
