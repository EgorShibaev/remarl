"""Evaluation-only adapters; the frozen generator and scoring pipeline are unchanged."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

BASE = Path(__file__).resolve().parent
UPSTREAM = BASE / 'sources'


def definitions(path, names, extra=None):
    """Load only reviewed function/class definitions, without upstream entrypoints."""
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name in names]
    assert {n.name for n in nodes} == set(names)
    import numpy as np
    scope = dict(torch=torch, F=F, nn=torch.nn, np=np, BASE_LINE=None)
    scope.update(extra or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope


def frequencies(device, vocabulary=126464):
    path = UPSTREAM / 'Uncode/data/baseline/reference_corpus_llada.json'
    data = json.loads(path.read_text())
    freq = torch.full((vocabulary,), 1 / data['num_token'], device=device)
    keys = torch.tensor([int(k) for k in data['p_baseline_dict']], device=device)
    values = torch.tensor([v / data['num_token'] for v in data['p_baseline_dict'].values()],
                          device=device)
    freq.scatter_(0, keys, values)
    return freq


_FREQUENCIES = {}


@torch.inference_mode()
def decode(model, prompt, cfg, mask_id=126336, length=256, frequency=None):
    """Answer-only equivalent of the cited implementations; batch one, greedy tokens."""
    assert prompt.shape[0] == 1
    mode = cfg['mode']
    block = cfg['block']
    assert length % block == 0
    x = torch.full((1, prompt.shape[1] + length), mask_id, device=prompt.device,
                   dtype=torch.long)
    x[:, :prompt.shape[1]] = prompt
    answer = x[:, prompt.shape[1]:]
    attention = torch.ones_like(x, dtype=torch.bool)
    calls = 0

    def forward():
        nonlocal calls
        logits = model(x, attention_mask=attention).logits[:, prompt.shape[1]:]
        calls += 1
        return logits

    if mode == 'p2_original':
        # Peng et al., Appendix E: fresh proposal log-probabilities at EVERY
        # position, eta=1, tau=0, threshold ties retained as in the reference.
        for step in range(1, cfg['steps'] + 1):
            previous_mask = answer.eq(mask_id)
            scores, proposals = forward().double().log_softmax(-1).max(-1)
            count = int(length * (1 - step / cfg['steps']))
            threshold = scores.sort(-1).values[:, count:count + 1]
            remask = scores < threshold
            answer[remask] = mask_id
            reveal = previous_mask & ~remask
            answer[reveal] = proposals[reveal]
        remaining = answer.eq(mask_id)
        answer[remaining] = proposals[remaining]
    else:
        if mode == 'pc_sampler' and frequency is None:
            key = str(prompt.device)
            if key not in _FREQUENCIES:
                _FREQUENCIES[key] = frequencies(prompt.device)
            frequency = _FREQUENCIES[key]
        previous = None
        history = torch.zeros((1, length, 2), device=prompt.device, dtype=torch.float64)
        for start in range(0, length, block):
            stop = start + block
            fixed = mode in ('margin', 'pc_sampler')
            per_block = cfg['steps'] // (length // block) if fixed else block
            if fixed:
                assert cfg['steps'] % (length // block) == 0
            for step in range(per_block):
                masked = answer[:, start:stop].eq(mask_id)
                if not masked.any():
                    break
                logits = forward()
                if mode == 'klass':
                    probs = logits.double().softmax(-1)
                    conf, proposal = probs.max(-1)
                    if previous is None:
                        previous = torch.zeros_like(probs)
                    kl = (probs * (torch.log(probs + 1e-12)
                                   - torch.log(previous + 1e-12))).sum(-1)
                    history = torch.roll(history, shifts=-1, dims=-1)
                    history[..., -1] = kl
                    previous = probs
                    stable = history[:, start:stop].lt(cfg['kl']).all(-1)
                    ready = stable & conf[:, start:stop].gt(cfg['confidence']) & masked
                    if step < 1:
                        ready.zero_()
                    if ready.any():
                        chosen = ready[0].nonzero().flatten()
                    else:
                        # Preserve upstream top-k tie handling on the full canvas.
                        full = torch.full(x.shape, -torch.inf, dtype=conf.dtype, device=x.device)
                        offset = prompt.shape[1] + start
                        full[:, offset:offset + block] = conf[:, start:stop].masked_fill(~masked, -torch.inf)
                        chosen = full[0].topk(1).indices - offset
                elif mode == 'eb_sampler':
                    proposal = logits.argmax(-1)
                    error = torch.distributions.Categorical(
                        logits=logits[:, start:stop][masked]).entropy()
                    order = error.argsort()
                    entropy = error[order]
                    k = (entropy.cumsum(0) - entropy.cummax(0).values <= cfg['gamma']).sum()
                    k = k.clamp(min=1, max=order.numel())
                    chosen = masked[0].nonzero().flatten()[order[:k]]
                else:
                    probs = logits.softmax(-1)
                    proposal = logits.argmax(-1)
                    local = probs[:, start:stop]
                    if mode == 'margin':
                        top = local.topk(2, dim=-1).values
                        score = top[..., 0] - top[..., 1]
                    elif mode == 'pc_sampler':
                        token = torch.where(masked, proposal[:, start:stop], answer[:, start:stop])
                        confidence = local.gather(-1, proposal[:, start:stop, None]).squeeze(-1)
                        cross_entropy = (-confidence * torch.log(frequency[token] + 1e-9)).clamp(max=10.)
                        position = torch.arange(block, device=x.device, dtype=torch.float32)
                        score = torch.exp(-.25 * position) * cross_entropy
                    else:
                        raise ValueError(mode)
                    quota = block // per_block + int(step < block % per_block)
                    chosen = score.masked_fill(~masked, -torch.inf)[0].topk(quota).indices
                answer[0, chosen + start] = proposal[0, chosen + start]
    assert not answer.eq(mask_id).any(), cfg
    assert torch.equal(x[:, :prompt.shape[1]], prompt)
    return SimpleNamespace(sequences=x, steps_taken=torch.tensor([calls], device=x.device))


def grid(kind):
    budgets = [4, 8, 16, 32, 64, 128, 256] if kind == 'math' else [8, 16, 32, 64, 128, 256]
    configs = []
    # Complete the low-NFE cells first, then the higher budgets.
    for steps in budgets:
        for mode in ('p2_original', 'pc_sampler'):
            configs.append(dict(mode=mode, block=256, steps=steps, seed=101))
        if kind == 'math':
            for block in (256, 32):
                if steps >= 256 // block:
                    configs.append(dict(mode='margin', block=block, steps=steps, seed=101))
        for block in ((256, 32) if kind == 'math' else (32,)):
            if steps >= 256 // block:
                for seed in (101, 202, 303):
                    configs.append(dict(mode='random', block=block, steps=steps, seed=seed))
    if kind == 'math':
        for block in (256, 32):
            for gamma in (.001, .01, .1, 1., 10.):
                configs.append(dict(mode='eb_sampler', block=block, gamma=gamma, seed=101))
        for block in (256, 64):
            for kl in (.001, .01, .015, .1):
                configs.append(dict(mode='klass', block=block, kl=kl, confidence=.6, seed=101))
    splits = [('test_gsm8k', 1319), ('test_math', 500)] if kind == 'math' else [('test_humaneval', 164), ('test_mbpp', 500)]
    cells = []
    for cfg in configs:
        setting = (f"nfe{cfg['steps']}" if 'steps' in cfg else
                   f"gamma{cfg['gamma']:g}" if 'gamma' in cfg else f"conf{cfg['confidence']:g}-kl{cfg['kl']:g}")
        for split, count in splits:
            name = f"{cfg['mode']}-bl{cfg['block']}-{setting}-{split}-s{cfg['seed']}"
            cells.append(dict(**cfg, split=split, count=count, name=name))
    return cells
