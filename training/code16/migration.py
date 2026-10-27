"""Remove approved rows without redrawing sampler order or changing learned state."""
import copy
import torch

META=frozenset(('code_training_manifest_sha256','cohort_manifest_sha256',
    'runtime_validation_sha256','data_audit_sha256','source_manifest_sha256'))
CHANGED=META|{'recipe','sampler'}

def exact(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return type(a) is type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b

def migrate(parent,recipe,kept,metadata):
    count=parent['sampler']['count']
    assert metadata.keys()==META
    assert kept==sorted(set(kept)) and 0<len(kept)<count
    assert min(kept)>=0 and max(kept)<count
    assert all(0<=i<count for i in parent['sampler']['remaining'])
    assert parent['recipe']['training_questions']==count
    assert recipe==dict(parent['recipe'],training_questions=len(kept))
    result=copy.deepcopy(parent);result.update(metadata);result['recipe']=copy.deepcopy(recipe)
    mapping={old:new for new,old in enumerate(kept)}
    result['sampler']['count']=len(kept)
    result['sampler']['remaining']=[mapping[i] for i in parent['sampler']['remaining'] if i in mapping]
    assert torch.equal(result['sampler']['rng'],parent['sampler']['rng'])
    assert all(exact(parent[k],result[k]) for k in parent if k not in CHANGED)
    return result
