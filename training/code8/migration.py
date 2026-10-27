"""Authorized cohort migration with order-preserving sampler remapping."""
import copy
import torch

METADATA_KEYS = frozenset(('code_training_manifest_sha256','cohort_manifest_sha256',
    'runtime_validation_sha256','data_audit_sha256','source_manifest_sha256'))
CHANGED_KEYS = METADATA_KEYS | {'recipe','sampler'}

def exact(a,b):
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict): return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return type(a) is type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b

def migrate(parent,old_metadata,new_metadata,kept,recipe):
    assert old_metadata.keys()==new_metadata.keys()==METADATA_KEYS
    assert all(parent[k]==v for k,v in old_metadata.items())
    assert parent['completed']==95 and parent['optimizer_updates']==380 and parent['attempted']==6345
    assert kept==sorted(set(kept)) and 0<len(kept)<parent['sampler']['count']
    assert min(kept)>=0 and max(kept)<parent['sampler']['count']
    expected=dict(parent['recipe'],training_questions=len(kept))
    assert recipe==expected and parent['recipe']['nfe']==8 and parent['recipe']['passes']==4
    result=copy.deepcopy(parent);result.update(new_metadata);result['recipe']=copy.deepcopy(recipe)
    mapping={old:new for new,old in enumerate(kept)}
    result['sampler']['count']=len(kept)
    result['sampler']['remaining']=[mapping[i] for i in parent['sampler']['remaining'] if i in mapping]
    assert torch.equal(result['sampler']['rng'],parent['sampler']['rng'])
    assert all(exact(parent[k],result[k]) for k in parent if k not in CHANGED_KEYS)
    return result
