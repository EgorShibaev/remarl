import copy
import pytest
import torch
from migration import META,exact,migrate

def parent():
    return dict(model={'weight':torch.tensor([1.,2.])},optimizer={'moment':torch.tensor([3.,4.])},
        sampler=dict(count=4,remaining=[3,1,2,0],rng=torch.Generator().manual_seed(7).get_state()),
        noise=torch.Generator().manual_seed(11).get_state(),shuffle=torch.tensor([5,6]),
        completed=1348,optimizer_updates=5392,attempted=55200,
        recipe=dict(training_questions=4,learning_rate=1e-4,nfe=16,passes=4),
        **{k:'old' for k in META})

def test_exact_state_and_order_preserving_sampler_filter():
    p=parent();original=copy.deepcopy(p)
    q=migrate(p,dict(p['recipe'],training_questions=3),[0,2,3],{k:'new' for k in META})
    assert q['sampler']['remaining']==[2,1,0]
    assert q['sampler']['count']==3 and torch.equal(q['sampler']['rng'],p['sampler']['rng'])
    assert exact(p,original)
    for key in ('model','optimizer','noise','shuffle','completed','optimizer_updates','attempted'):
        assert exact(q[key],p[key])

@pytest.mark.parametrize('kept',[[2,0],[0,0],[0,4],[]])
def test_invalid_kept_indices_rejected(kept):
    p=parent()
    with pytest.raises(AssertionError):migrate(p,dict(p['recipe'],training_questions=len(kept)),kept,{k:'new' for k in META})

def test_learning_rate_change_rejected():
    p=parent()
    with pytest.raises(AssertionError):migrate(p,dict(p['recipe'],training_questions=3,learning_rate=3e-4),[0,2,3],{k:'new' for k in META})

def test_incomplete_provenance_rejected():
    p=parent()
    with pytest.raises(AssertionError):migrate(p,dict(p['recipe'],training_questions=3),[0,2,3],{})
