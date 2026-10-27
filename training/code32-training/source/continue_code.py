"""Resume the unchanged code32 scientific state with the tested collector."""
import copy
import json
import os
from pathlib import Path
import time

import torch
import train_code as code
import train_batch_grpo as core
from evaluate_p2 import save, sha
from pipelined_gather import install as install_pipeline
from reliable_sandbox import install as install_sandbox

BASE = Path(__file__).resolve().parent


def exact(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def main():
    c = json.loads((BASE/'STAGED.json').read_text())
    original = Path(c['original']); prod = original/'source/production'
    assert os.getuid() == c['uid'] and os.uname().nodename.split('.')[0] == c['node']
    assert os.environ['SLURM_JOB_ID'] == c['holder'] and not (BASE/'STARTED.json').exists()
    assert sha(prod/'latest.pt') == sha(BASE/'parent-latest.pt') == c['parent_checkpoint_sha256']
    parent = torch.load(BASE/'parent-latest.pt', map_location='cpu', weights_only=True)
    assert parent['completed']*16 == c['groups'] and parent['optimizer_updates'] == c['updates']
    step = c['holder']+'.'+os.environ['SLURM_STEP_ID']
    save(BASE/'STARTED.json', dict(time=time.time(), step=step, node=c['node'], groups=c['groups']))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    install_pipeline(core.Trainer)
    install_sandbox(BASE/'execution-retries')
    core.Replica = code.CodeReplica
    trainer = None
    try:
        trainer = code.Trainer(original/'source', prod, BASE/'runtime', list(range(8)))
        trainer.restore(copy.deepcopy(parent))
        assert exact(trainer.snapshot(), parent), 'Full-state restoration mismatch'
        orphan = prod/'batches'/f'{parent["completed"]+1:06d}'
        if orphan.exists():
            assert not (orphan/'complete.json').exists()
            preserved = BASE/('interrupted-production-batch-'+orphan.name)
            assert not preserved.exists()
            orphan.rename(preserved)
            save(BASE/'PARTIAL_BATCH_PRESERVED.json', dict(original=str(orphan), preserved=str(preserved),
                reason='Interrupted at user-requested pause; replay from exact last completed checkpoint'))
        receipt = dict(passed=True, time=time.time(), step=step, node=c['node'], name=c['name'],
            gpus=8, original=str(original), groups=c['groups'], updates=c['updates'],
            parent_checkpoint_sha256=c['parent_checkpoint_sha256'], full_state_restore='exact',
            scientific_recipe_unchanged=True, metadata_unchanged=True,
            collector='ordered-pipeline-v1', speculative_groups=0,
            tests_root=c['tests_root'], tested_source_hashes=c['tested_source_hashes'],
            source_commit=c['source_commit'], training_source_sha256=c['training_source_sha256'],
            validation_mode='Reuse completed89tests+69subtests, three exact GPU pairs and full sandbox suite')
        save(BASE/'PRODUCTION_RESUMED.json', receipt)
        save(original/'ACTIVE_RECOVERY.json', dict(root=str(BASE), step=step, production_resumed=True, time=time.time()))
        print('PRODUCTION_RESUMED_PIPELINE', c['groups'], c['updates'], flush=True)
        code.production(trainer, resume=True)
    except BaseException as exc:
        save(BASE/('failure-'+str(int(time.time()))+'.json'),
            dict(type=type(exc).__name__, message=str(exc), time=time.time()))
        raise
    finally:
        if trainer is not None:
            trainer.close()


if __name__ == '__main__':
    main()
