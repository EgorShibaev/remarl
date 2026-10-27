"""APPLe confidence, random, and Fast-dLLM decoding."""

from types import SimpleNamespace
import torch
from ..protocols.official import apple_generation as apple


class AnswerLogits:
    """Expose answer logits without a dynamically created class/closure cycle."""

    def __init__(self, answer, prompt_length):
        self.answer = answer
        self.prompt_length = prompt_length

    def __getitem__(self, key):
        assert key == (slice(None), slice(self.prompt_length, None))
        return self.answer


class NativeAdapter:
    dtype = torch.bfloat16

    def __init__(self, engine, prompt):
        self.engine = engine
        self.prompt_length = prompt.shape[1]
        self.positions = torch.arange(
            self.prompt_length, self.prompt_length + 256, device=prompt.device
        )[None]
        self.mask = torch.ones(
            (1, self.prompt_length + 256), dtype=torch.bool, device=prompt.device
        )
        self.forward = None
        self.calls = 0

    def __call__(self, x, attention_mask=None, output_hidden_states=False):
        assert not output_hidden_states and torch.equal(attention_mask.bool(), self.mask)
        if self.forward is None:
            self.forward = self.engine.prepare(x, self.mask, self.positions)
        answer, _ = self.forward(x, self.mask, self.positions)
        assert answer.dtype == torch.bfloat16
        self.calls += 1
        # Public generation only reads the answer suffix. Avoid a prompt-sized LM head.
        return SimpleNamespace(logits=AnswerLogits(answer, self.prompt_length))


def decode(model, prompt, cfg):
    return apple.generate_unified(
        model,
        prompt,
        cfg["mode"],
        steps=cfg.get("steps"),
        thres=cfg.get("threshold"),
        gen_length=256,
        block_length=cfg["block"],
        temperature=0.0,
        mask_id=126336,
        model_type="LLaDA",
        attention_mask=torch.ones_like(prompt, dtype=torch.bool),
    )
