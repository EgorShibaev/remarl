"""GPU validation, sampler-aware cohort migration, then real code continuation."""
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
from migration import exact,migrate,CHANGED_KEYS,METADATA_KEYS

BASE=Path(__file__).resolve().parent
def main():
    assert os.uname().nodename.split('.')[0]=='i8027' and os.environ['SLURM_JOB_ID']=='2357570'
    assert torch.cuda.device_count()==8 and all('A100' in torch.cuda.get_device_name(i) for i in range(8))
    assert json.loads((BASE/'RUNTIME_VALIDATED.json').read_text())['passed']
    assert not (BASE/'RESUME_VALIDATED.json').exists()
    staged=json.loads((BASE/'RECOVERY_STAGED.json').read_text());parent_path=BASE/'recovery-parent.pt'
    assert sha(parent_path)==staged['checkpoint_sha256']==sha(BASE/'source/production/latest.pt')
    parent=torch.load(parent_path,map_location='cpu',weights_only=True)
    current=dict(code_training_manifest_sha256=sha(BASE/'training-source.sha256'),
        cohort_manifest_sha256=sha(BASE/'cohort/MANIFEST.json'),
        runtime_validation_sha256=sha(BASE/'RUNTIME_VALIDATED.json'),
        data_audit_sha256=sha(BASE/'source/data-audit.json'),source_manifest_sha256=sha(BASE/'source/source.sha256'))
    recipe=json.loads((BASE/'source/recipe.json').read_text())
    state=migrate(parent,staged['parent_metadata'],current,staged['kept_old_indices'],recipe)
    metrics=[json.loads(s) for s in (BASE/'source/production/metrics.jsonl').read_text().splitlines()]
    assert len(metrics)==95 and metrics[-1]['new_policy_hash']==state['policy_sha256']
    assert metrics[-1]['attempted_groups']==state['attempted']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);core.Replica=code.CodeReplica
    gate_dir=BASE/'source/recovery-validation';assert not gate_dir.exists()
    trainer=code.Trainer(BASE/'source',gate_dir,BASE/'runtime',list(range(8)))
    try:
        for split in ('train',)+code.TESTS:
            longest=max(range(len(trainer.rows[split])),key=lambda i:len(trainer.rows[split][i]['prompt_ids']))
            uniforms=torch.stack([draws(longest,8,101)]*8,1)
            a=trainer.replicas[0].generate(trainer.inputs[split],[longest]*8,uniforms,False)
            b=trainer.replicas[1].generate(trainer.inputs[split],[longest]*8,uniforms,False)
            assert a['tokens']==b['tokens'],split
            save(gate_dir/(split+'-longest-parity.json'),dict(passed=True,index=longest,tokens_equal=True))
            del a,b
        core.validate(trainer)
        trainer.restore(copy.deepcopy(state));assert exact(trainer.snapshot(),state)
        groups,logs,folder=trainer.gather();before=trainer.snapshot()
        first=trainer.update(groups);after=trainer.snapshot()
        trainer.restore(copy.deepcopy(before));second=trainer.update(groups)
        assert first==second and exact(after,trainer.snapshot()),'Warm update replay mismatch'
        assert after['policy_sha256']!=state['policy_sha256']
        trainer.restore(copy.deepcopy(state));assert exact(trainer.snapshot(),state)
        del groups,logs,before,after
        assert all(exact(parent[k],state[k]) for k in parent if k not in CHANGED_KEYS)
        assert sha(parent_path)==staged['checkpoint_sha256']
        receipt=dict(passed=True,parent_step='2356156.13',parent_checkpoint_sha256=staged['checkpoint_sha256'],
            migrated_full_state_restore='exact',warm_update_replay='exact',
            model_optimizer_rollout_shuffle_rng_counters_unchanged=True,sampler_rng_unchanged=True,
            sampler_migration='Filter quarantined old indices; remap remaining order without RNG draws',
            removed_sources=staged['quarantine']['additional_excluded_sources'],
            old_training_questions=parent['recipe']['training_questions'],training_questions=len(staged['kept_old_indices']),
            metadata_changes={k:dict(parent=parent[k],current=state[k]) for k in METADATA_KEYS},
            gpu_count=8,node='i8027',step=os.environ['SLURM_STEP_ID'],time=time.time(),
            accepted_groups=1520,optimizer_updates=380,trajectories=128,passes=4,lr=1e-4,
            benchmark_prompts_tokens_tests_unchanged=True)
        save(BASE/'RESUME_VALIDATED.json',receipt)
        archive=BASE/'recovery-parent-outputs';archive.mkdir()
        partial=BASE/'source/production/batches/000096'
        if partial.exists():
            assert not (partial/'complete.json').exists();partial.rename(archive/'partial-batch-000096')
        for p in (BASE/'source/production').glob('failure-*.json'):p.rename(archive/p.name)
        gate=BASE/'source/validation/VALIDATED.json';gate.rename(archive/'VALIDATED-parent.json')
        save(gate,json.loads((gate_dir/'VALIDATED.json').read_text()))
        atomic_torch_save(state,BASE/'source/production/latest.pt')
        resolved_path=BASE/'source/production/resolved-recipe.json'
        resolved=json.loads(resolved_path.read_text());save(archive/'resolved-recipe-parent.json',resolved)
        resolved.update(fresh_optimizer=False,restored_optimizer=True,recovery=receipt,recipe=recipe)
        save(resolved_path,resolved)
        trainer.out=BASE/'source/production';save(BASE/'PRODUCTION_RESUMED.json',receipt)
        print('CODE_NFE8_PRODUCTION_RESUMED',1520,380,flush=True)
        code.production(trainer,resume=True)
    except BaseException as exc:
        save(BASE/('recovery-failure-'+str(int(time.time()))+'.json'),
             dict(type=type(exc).__name__,message=str(exc),time=time.time()))
        raise
    finally:trainer.close()

if __name__=='__main__':main()
