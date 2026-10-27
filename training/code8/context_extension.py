"""Private nonpersistent RoPE-cache extension; exact pinned HF CPU math."""
import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import torch
from torch import nn


def extend(engine, minimum, checkpoint):
    old=engine.config.max_sequence_length
    if minimum<=old:return dict(old=old,new=old,changed=False)
    tree=ast.parse((Path(checkpoint)/'modeling_llada.py').read_text())
    definitions=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))
                 and n.name in ('_non_meta_init_device','RotaryEmbedding')]
    assert len(definitions)==2
    namespace=dict(torch=torch,nn=nn,ModelConfig=SimpleNamespace,BufferCache=dict,
                   Tuple=Tuple,einsum=torch.einsum)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),'pinned_HF_RoPE','exec'),namespace)
    cfg=SimpleNamespace(d_model=engine.config.d_model,n_heads=engine.config.n_heads,
        max_sequence_length=old,rope_theta=engine.config.rope_theta,init_device='cpu',rope_full_precision=True)
    reference=namespace['RotaryEmbedding'](cfg,{})
    sin,cos=reference.get_rotary_embedding(minimum,torch.device('cpu'))
    sin,cos=sin.permute(0,2,1,3),cos.permute(0,2,1,3)
    assert torch.equal(sin[:,:old],engine.sin.cpu()) and torch.equal(cos[:,:old],engine.cos.cpu())
    parameters=[p.data_ptr() for p in engine.parameters()]
    engine.sin=sin.to(engine.embedding.device);engine.cos=cos.to(engine.embedding.device)
    engine.config=replace(engine.config,max_sequence_length=minimum)
    assert parameters==[p.data_ptr() for p in engine.parameters()]
    return dict(old=old,new=minimum,changed=True,original_prefix_bit_exact=True,
                pinned_HF_CPU_cache_math='exact',parameter_tensors_unchanged=True,rope_scaling=False)
