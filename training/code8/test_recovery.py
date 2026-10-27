import copy
import pytest
import torch
from migration import migrate,exact,METADATA_KEYS,CHANGED_KEYS

def state():
    return dict(completed=95,optimizer_updates=380,attempted=6345,passes=4,
        recipe=dict(nfe=8,passes=4,learning_rate=1e-4,training_questions=10),
        sampler=dict(count=10,rng=torch.Generator().manual_seed(7).get_state(),remaining=[8,3,5,1,6]),
        model=dict(w=torch.tensor([1.,2.])),optimizer=dict(step=torch.tensor(380.)),
        noise=torch.Generator().manual_seed(2).get_state(),shuffle=torch.Generator().manual_seed(3).get_state(),
        **{k:'old-'+k for k in METADATA_KEYS})

def arguments(original,kept):
    return ({k:original[k] for k in METADATA_KEYS},{k:'new-'+k for k in METADATA_KEYS},
            kept,dict(original['recipe'],training_questions=len(kept)))

def test_order_preserving_remap_and_rng():
    parent=state();pristine=copy.deepcopy(parent);kept=[0,1,2,3,4,6,7,9]
    migrated=migrate(parent,*arguments(parent,kept))
    assert migrated['sampler']['remaining']==[3,1,5]
    assert torch.equal(migrated['sampler']['rng'],parent['sampler']['rng'])
    assert all(exact(parent[k],migrated[k]) for k in parent if k not in CHANGED_KEYS)
    migrated['optimizer']['step'].add_(4)
    assert exact(parent,pristine)

def test_refuse_recipe_drift_or_invalid_mapping():
    parent=state();args=list(arguments(parent,list(range(9))))
    args[3]['learning_rate']=3e-4
    with pytest.raises(AssertionError):migrate(parent,*args)
    with pytest.raises(AssertionError):migrate(parent,*arguments(parent,[0,0,1]))
    with pytest.raises(AssertionError):migrate(parent,*arguments(parent,[2,1]))
    with pytest.raises(AssertionError):migrate(parent,*arguments(parent,[0,10]))

def test_reject_wrong_parent_metadata():
    parent=state();args=list(arguments(parent,list(range(9))))
    args[0][next(iter(METADATA_KEYS))]='wrong'
    with pytest.raises(AssertionError):migrate(parent,*args)
