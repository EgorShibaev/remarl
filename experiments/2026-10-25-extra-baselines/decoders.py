"""Pinned official RCR, CoRe and DAPD decoders; no training."""
import ast
from functools import lru_cache
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

BASE = Path(__file__).resolve().parent


def definitions(path, scope):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope


@lru_cache(None)
def implementations():
    scope = dict(torch=torch, np=np, F=F, dists=torch.distributions)
    definitions(BASE/'sources/mdpo/src/llada/generate.py', scope)
    rcr = definitions(BASE/'sources/mdpo/src/mdlm_generation_utils.py', scope)['diffusion_generate']
    spec = importlib.util.spec_from_file_location('official_core', BASE/'sources/CoRe/generate.py')
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    sys.path.insert(0, str(BASE/'sources/DAPD'))
    from dapd.generation import generate_dapd_with_blocks
    return rcr, core.generate, generate_dapd_with_blocks


def core_steps(budget):
    # Reserve every possible verification call, even if a particular answer
    # has no eligible candidates. Do not charge only the base denoising steps.
    return max(s for s in range(1, budget + 1)
               if s + sum(i % 8 == 0 and int(.25*s) <= i < int(.75*s)
                          for i in range(s)) <= budget)


def decode(model, prompt, cfg, length=256, mask_id=126336):
    rcr, core, dapd = implementations()
    before = model.calls
    if cfg['mode'] == 'rcr':
        answer, *_ = rcr(model, prompt, mask_id=mask_id, steps=cfg['steps'],
                        gen_length=length, block_length=length, temperature=0.,
                        conf_alg='low_confidence', mode='linear', rcr=True)
        output = torch.cat((prompt, answer), -1)
    elif cfg['mode'] == 'core':
        required = dict(REVISE_EVERY='8', CANDIDATE_M='32', BASE_MASKING='confidence',
                        JOINT_REEVAL='0', CORE_DEBUG='0')
        for name, value in required.items():
            assert os.environ.get(name, value) == value, (name, os.environ.get(name))
        output, _, _ = core(model, prompt, steps=core_steps(cfg['budget']),
                            gen_length=length, block_length=length, temperature=0.,
                            remasking='core', mask_id=mask_id)
    elif cfg['mode'] == 'dapd':
        output, stats = dapd(model, prompt, gen_length=length, block_length=cfg['block'],
                            mask_id=mask_id, alg=cfg['variant'], layer_ratio=.3,
                            tau_min=.01, tau_max=.15, collect_step_history=False)
        assert stats['total_forward_passes'] == model.calls - before
    else:
        raise ValueError(cfg)
    calls = model.calls - before
    assert torch.equal(output[:, :prompt.shape[1]], prompt)
    assert not output[:, prompt.shape[1]:].eq(mask_id).any()
    assert calls <= cfg.get('budget', cfg.get('steps', length))
    if cfg['mode'] == 'rcr':
        assert calls == cfg['steps']
    return SimpleNamespace(sequences=output, steps_taken=torch.tensor([calls]))


def grid(kind):
    budgets = [4, 8, 16, 32, 64, 128, 256] if kind == 'math' else [8, 16, 32, 64, 128, 256]
    configs = [dict(mode='rcr', block=256, steps=n, seed=101) for n in budgets]
    configs += [dict(mode='core', block=256, budget=n, seed=101) for n in budgets]
    configs += [dict(mode='dapd', block=b, variant=v, seed=101)
                for b in (256, 64) for v in ('dapd_direct', 'dapd_staged')]
    splits = [('test_gsm8k', 1319), ('test_math', 500)] if kind == 'math' else [('test_humaneval', 164), ('test_mbpp', 500)]
    cells = []
    for cfg in configs:
        setting = ('nfe'+str(cfg['steps']) if 'steps' in cfg else
                   'cap'+str(cfg['budget']) if 'budget' in cfg else cfg['variant'])
        for split, count in splits:
            cells.append(dict(**cfg, split=split, count=count,
                              name=f"{cfg['mode']}-bl{cfg['block']}-{setting}-{split}-s101"))
    return cells
