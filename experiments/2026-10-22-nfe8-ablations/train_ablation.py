"""Fresh NFE8 training with one intervention and separate real-GPU gates."""
import copy
import json
import os
from pathlib import Path
import time
import torch
import ablation_ops as ops
import train_batch_grpo as train
from logging_repair import append, save
from src.rl.training_state import atomic_torch_save

BASE = Path(os.environ['ABLATION_ROOT'])
MODE = os.environ['ABLATION_VARIANT']
ops.configure(MODE)
train.SetSelector = ops.selector_class(train.SetSelector)
train.append, train.save = append, save


def exact(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b


def validate(t):
    initial = t.snapshot()
    uniforms = torch.rand((8,8,256),generator=torch.Generator().manual_seed(777))
    replica = t.replicas[0]
    first = replica.generate(t.inputs['train'],[0]*8,uniforms)
    repeat = replica.generate(t.inputs['train'],[0]*8,uniforms)
    p2 = replica.generate(t.inputs['train'],[0]*8,uniforms,False,True)
    assert first['tokens']==repeat['tokens']==p2['tokens']
    assert torch.equal(first['replay'].old_logp,repeat['replay'].old_logp)
    if MODE=='no_revision':assert not first['replay'].revisions.any()
    del first,repeat,p2
    groups,logs,_ = t.gather()
    before = t.snapshot()
    one = t.update(groups);after = t.snapshot()
    assert after['policy_sha256']!=before['policy_sha256']
    trainable = {n for n,p in t.selector.named_parameters() if p.requires_grad}
    changed = [n for n,v in before['model'].items() if not torch.equal(v,after['model'][n])]
    assert changed and all(n in trainable for n in changed)
    if MODE in ('no_revision','no_position'):
        assert any(n.startswith('hidden_projection.') for n in changed)
    atomic_torch_save(before,t.out/'resume-gate.pt')
    t.restore(torch.load(t.out/'resume-gate.pt',map_location='cpu',weights_only=True))
    two = t.update(groups);again = t.snapshot()
    assert one==two and exact(after,again), 'Full-state replay/resume differs'
    save(t.out/'VALIDATED.json',dict(passed=True,variant=MODE,time=time.time(),
        initialization_sha256=initial['policy_sha256'],
        source_manifest_sha256=train.sha(t.root/'source.sha256'),
        data_audit_sha256=train.sha(t.root/'data-audit.json'),
        exact_repeatability=True,initial_p2_parity=True,full_resume_state='exact',
        changed_parameters=changed,trainable_parameters=sum(p.numel() for p in t.selector.parameters() if p.requires_grad),
        total_parameters=sum(p.numel() for p in t.selector.parameters()),
        no_revision_action_support=(MODE=='no_revision'),replay_error=one['replay_error']))
    del groups,logs,before,after,again
    t.restore(initial)


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.device_count()==2 and all('A100' in torch.cuda.get_device_name(i) for i in (0,1))
    root=BASE/'variants'/MODE/'source';out=root/'validation'
    assert not out.exists() and not (root/'production').exists()
    save(BASE/'variants'/MODE/'STARTED.json',dict(time=time.time(),pid=os.getpid(),gpus=os.environ['CUDA_VISIBLE_DEVICES'],variant=MODE))
    t=train.Trainer(root,out,BASE/'runtime',[0,1])
    try:
        assert t.recipe.nfe==8 and t.total_batches==2048 and t.recipe.updates_per_batch==4
        validate(t)
        t.out=root/'production';t.out.mkdir()
        # Validation restores the untouched initialization, optimizer and all random streams.
        train.production(t)
        receipt=json.loads((t.out/'COMPLETE.json').read_text())
        for split,count in (('test_gsm8k',1319),('test_math',500)):
            for seed in (101,202,303):
                cell=json.loads((t.out/f'evaluations/{split}-032768-seed{seed}/summary.json').read_text())
                assert cell['complete'] and cell['count']==count
        save(BASE/'variants'/MODE/'COMPLETE.json',dict(**receipt,variant=MODE,final_tests_complete=True))
    finally:
        t.close()


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save(BASE/'variants'/MODE/'FAILED.json',dict(time=time.time(),type=type(exc).__name__,message=str(exc)))
        raise
