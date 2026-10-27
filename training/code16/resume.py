"""Five provenance hashes migrate; model/optimizer/sampler/RNG and recipe stay exact."""
import copy
import json
import os
from pathlib import Path
import time

import torch
import train_code as code
import train_batch_grpo as core
from evaluate_p2 import draws,save,sha
from src.rl.training_state import atomic_torch_save
from migration import migrate, CHANGED

BASE=Path(__file__).resolve().parent
META=frozenset(('code_training_manifest_sha256','cohort_manifest_sha256',
    'runtime_validation_sha256','data_audit_sha256','source_manifest_sha256'))

def exact(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return type(a) is type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b

def main():
    assert os.uname().nodename.split('.')[0]=='i8027' and os.environ['SLURM_JOB_ID']=='2357570'
    assert torch.cuda.device_count()==8
    assert all('A100' in torch.cuda.get_device_name(i) for i in range(8))
    assert json.loads((BASE/'PRODUCTION_PREPARED.json').read_text())['passed']
    assert not (BASE/'PRODUCTION_RESUMED.json').exists()
    parent=torch.load(BASE/'parent-latest-021568.pt',map_location='cpu',weights_only=True)
    assert parent['completed']==1348 and parent['optimizer_updates']==5392 and parent['attempted']==55200
    assert parent['recipe']['nfe']==16 and parent['recipe']['target_accepted_groups']==32768
    current=dict(code_training_manifest_sha256=sha(BASE/'training-source.sha256'),
        cohort_manifest_sha256=sha(BASE/'cohort/MANIFEST.json'),
        runtime_validation_sha256=sha(BASE/'RUNTIME_VALIDATED.json'),
        data_audit_sha256=sha(BASE/'source/data-audit.json'),source_manifest_sha256=sha(BASE/'source/source.sha256'))
    assert current.keys()==META
    quarantine=json.loads((BASE/'QUARANTINE.json').read_text())
    assert quarantine['removed_sources']==[8633]
    recipe=json.loads((BASE/'source/recipe.json').read_text())
    state=migrate(parent,recipe,quarantine['kept_old_indices'],current)
    assert all(exact(parent[k],state[k]) for k in parent if k not in CHANGED)
    prod=BASE/'source/production'
    metrics=[json.loads(s) for s in (prod/'metrics.jsonl').read_text().splitlines()]
    assert len(metrics)==1348 and metrics[-1]['new_policy_hash']==parent['policy_sha256']
    assert sha(prod/'latest.pt')==sha(BASE/'parent-latest-021568.pt')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);core.Replica=code.CodeReplica
    gate=BASE/'source/validation';assert not gate.exists()
    trainer=code.Trainer(BASE/'source',gate,BASE/'runtime',list(range(8)))
    try:
        for split in ('train',)+code.TESTS:
            longest=int(trainer.inputs[split]['mask'].sum(-1).argmax())
            uniforms=torch.stack([draws(longest,16,101)]*8,1)
            a=trainer.replicas[0].generate(trainer.inputs[split],[longest]*8,uniforms,False)
            b=trainer.replicas[1].generate(trainer.inputs[split],[longest]*8,uniforms,False)
            assert a['tokens']==b['tokens'],split
            save(gate/(split+'-longest-parity.json'),dict(passed=True,index=longest))
            del a,b
        core.validate(trainer)
        trainer.restore(copy.deepcopy(state));assert exact(trainer.snapshot(),state)
        groups,logs,_=trainer.gather();before=trainer.snapshot()
        first=trainer.update(groups);after=trainer.snapshot()
        trainer.restore(copy.deepcopy(before));second=trainer.update(groups)
        assert first==second and exact(after,trainer.snapshot()),'Warm four-pass replay mismatch'
        assert after['policy_sha256']!=state['policy_sha256']
        trainer.restore(copy.deepcopy(state));assert exact(trainer.snapshot(),state)
        del groups,logs,before,after
        assert all(exact(parent[k],state[k]) for k in parent if k not in CHANGED)
        receipt=dict(passed=True,time=time.time(),node='i8027',holder='2357570',
            step='2357570.'+os.environ['SLURM_STEP_ID'],name='code16',accepted_groups=21568,
            optimizer_updates=5392,target_groups=32768,parent_checkpoint_sha256=sha(BASE/'parent-latest-021568.pt'),
            full_scientific_state_restore='exact',warm_four_update_replay='exact',replicas=8,
            model_optimizer_rollout_shuffle_rng_counters_unchanged=True,
            sampler_rng_unchanged=True,sampler_migration='Remove old index7883 and remap remaining order without RNG draws',
            removed_training_sources=[8633],training_questions=9148,
            metadata_changes={k:dict(parent=parent[k],current=state[k]) for k in sorted(META)},
            security_sha256=sha(BASE/'SECURITY_VALIDATED.json'),reference_sha256=sha(BASE/'REFERENCE_VALIDATED.json'))
        save(BASE/'RESUME_VALIDATED.json',receipt)
        atomic_torch_save(state,prod/'latest.pt')
        resolved=json.loads((prod/'resolved-recipe.json').read_text())
        resolved.update(**recipe,fresh_optimizer=False,restored_optimizer=True,recovery=receipt)
        save(prod/'resolved-recipe.json',resolved)
        trainer.out=prod;save(BASE/'PRODUCTION_RESUMED.json',receipt)
        print('CODE16_SECCOMP_PRODUCTION_RESUMED',21568,5392,flush=True)
        code.production(trainer,resume=True)
    except BaseException as exc:
        save(BASE/('recovery-failure-'+str(int(time.time()))+'.json'),
            dict(type=type(exc).__name__,message=str(exc),time=time.time()))
        raise
    finally:trainer.close()

if __name__=='__main__':main()
