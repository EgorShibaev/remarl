"""Protocol-only change to the pinned SOTA trainer; no optimization changes."""
import json
import os
from pathlib import Path

from src.rl.apple_protocol import load_official, score
from src.rl.datasets import _make_env


class AppleVerifier:
    def __init__(self, official):
        self.official = official

    def __call__(self, text, gold):
        return float(score(self.official, text, gold)['apple_correctness'])


def make_environment(bb, rows):
    root = Path(os.environ['REMASK_ROOT'])
    official = load_official(root/'official')
    tokenized = [bb.tokenizer(r['prompt'], truncation=False)['input_ids'] for r in rows]
    length = max(map(len, tokenized)) + 256
    assert length <= 2048
    env = _make_env('apple-gsm8k', rows, bb.tokenizer,
        lambda r: (r['prompt'], r['apple_gold']), AppleVerifier(official),
        answer_len=256, max_length=length, max_examples=len(rows),
        mask_token_id=bb.mask_token_id, pad_token_id=bb.pad_token_id, nfe_max=256)
    assert len(env.problems) == len(rows)
    for r, ids, p in zip(rows, tokenized, env.problems):
        assert p.input_ids[:len(ids)].tolist() == ids
        assert p.answer_span.tolist() == list(range(len(ids),len(ids)+256))
        assert int(p.attention_mask.sum()) == len(ids)+256
        assert p.y_star == r['apple_gold']
    return env
