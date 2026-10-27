"""Code-domain adapter around the validated synchronous GRPO learner."""
import argparse
import json
import os
from pathlib import Path
import time

import torch

import train_batch_grpo as core
from batch_grpo_core import state_digest
from context_extension import extend
from evaluate_p2 import draws, save, sha
from src.rl.training_state import atomic_torch_save

BASE = Path(__file__).resolve().parent
TESTS = ('test_humaneval','test_mbpp')


class CodeReplica(core.Replica):
    def __init__(self, gpu, port, checkpoint, config):
        super().__init__(gpu,port,checkpoint,config)
        required = json.loads((BASE/'source/data-audit.json').read_text())['required_canvas']
        self.context_gate = extend(self.engine,required,checkpoint)


class Trainer(core.Trainer):
    def gather(self, max_attempts=1024):
        # Every collection starts with independent correct/wrong execution probes.
        self.scorer.begin_collection()
        return super().gather(max_attempts)

    def snapshot(self):
        state = super().snapshot()
        state.update(code_training_manifest_sha256=sha(BASE/'training-source.sha256'),
                     cohort_manifest_sha256=sha(BASE/'cohort/MANIFEST.json'),
                     runtime_validation_sha256=sha(BASE/'RUNTIME_VALIDATED.json'))
        return state

    def restore(self, state):
        assert state['code_training_manifest_sha256']==sha(BASE/'training-source.sha256')
        assert state['cohort_manifest_sha256']==sha(BASE/'cohort/MANIFEST.json')
        assert state['runtime_validation_sha256']==sha(BASE/'RUNTIME_VALIDATED.json')
        super().restore(state)

    def evaluate(self, split, seed):
        folder = self.out/'evaluations'/f'{split}-{self.completed*self.batch_groups:06d}-seed{seed}'
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/'summary.json').exists():
            old = json.loads((folder/'summary.json').read_text())
            assert old['complete'] and old['policy_sha256']==self.behavior_hash
            assert old['count']==len(self.rows[split])
            return
        assert not (folder/'answers.jsonl').exists(), 'Partial evaluation must be inspected before recovery'
        self.scorer.begin_collection()
        total = len(self.rows[split]); correct = 0; count = 0; started = time.monotonic()
        for start in range(0,total,len(self.replicas)*8):
            futures = []
            for j, replica in enumerate(self.replicas):
                ids = list(range(start+j*8,min(start+(j+1)*8,total)))
                if not ids: break
                real = len(ids); ids += [ids[-1]]*(8-real)
                uniforms = torch.stack([draws(i,self.recipe.nfe,seed) for i in ids],1)
                futures.append((ids,real,self.pool.submit(replica.generate,self.inputs[split],ids,uniforms,False)))
            for ids,real,future in futures:
                group = future.result(timeout=300)
                answers = self.scorer(group['tokens'][:real],[self.rows[split][i]['gold'] for i in ids[:real]])
                for index,tokens,answer in zip(ids[:real],group['tokens'][:real],answers):
                    assert index==count
                    core.append(folder/'answers.jsonl',dict(index=index,task_id=self.rows[split][index]['task_id'],
                                tokens=tokens,**answer))
                    correct += int(answer['reward']); count += 1
        assert count==total and state_digest(self.selector.state_dict())==self.behavior_hash
        assert all(state_digest(r.selector.state_dict())==self.behavior_hash for r in self.replicas)
        result = dict(complete=True,split=split,count=count,correct=correct,accuracy=correct/count,
            accepted_groups=self.completed*self.batch_groups,optimizer_updates=self.completed*4,
            attempted_groups=self.attempted,seed=seed,nfe=self.recipe.nfe,policy_sha256=self.behavior_hash,
            held_out=split in TESTS,seconds=time.monotonic()-started,
            prompt='APPLe released code-specific template',scorer='APPLe sanitizer + official HF execution',
            position_temperature=1.,token_temperature=0.,answer_length=256,
            no_benchmark_exclusions=True,training_source_sha256=sha(BASE/'training-source.sha256'),
            answers_sha256=sha(folder/'answers.jsonl'),backend='native-vllm-fa2')
        save(folder/'summary.json',result)
        print('EVALUATED',folder.name,correct,count,flush=True)


def production(trainer, resume=False):
    gate = json.loads((trainer.root/'validation/VALIDATED.json').read_text())
    assert gate['passed'] and gate['source_manifest_sha256']==sha(trainer.root/'source.sha256')
    assert gate['data_audit_sha256']==sha(trainer.root/'data-audit.json')
    if resume:
        trainer.restore(torch.load(trainer.out/'latest.pt',map_location='cpu',weights_only=True))
    else:
        assert gate['initialization_sha256']==trainer.behavior_hash
        assert not (trainer.out/'latest.pt').exists()
        atomic_torch_save(trainer.snapshot(),trainer.out/'latest.pt')
    # Iteration-zero is fresh stochastic P2, evaluated under the same backend/protocol.
    if trainer.completed==0:
        for seed in (101,202,303):
            for split in TESTS: trainer.evaluate(split,seed)
    while trainer.completed<trainer.total_batches:
        started = time.monotonic()
        groups,logs,folder = trainer.gather()
        diagnostics = trainer.update(groups)
        rewards = [r for log in logs for r in log['rewards']]
        trainer.completed += 1
        state = trainer.snapshot()
        atomic_torch_save(state,trainer.out/'latest.pt')
        summary = dict(batch=trainer.completed,accepted_groups=trainer.completed*16,
            optimizer_updates=trainer.completed*4,attempted_groups=trainer.attempted,
            attempted_this_batch=len(logs),all_attempt_accuracy=sum(rewards)/len(rewards),
            accepted_accuracy=sum(sum(g['rewards']) for g in groups)/128,
            acceptance_rate=16/len(logs),all_wrong=sum(sum(l['rewards'])==0 for l in logs),
            all_correct=sum(sum(l['rewards'])==8 for l in logs),
            mean_revisions=sum(sum(l['revisions']) for l in logs)/len(rewards),
            seconds=time.monotonic()-started,time=time.time(),
            behavior_hash=trainer.behavior_hash,new_policy_hash=state['policy_sha256'],
            task_metrics={'kodcode':dict(attempted=len(logs),accepted=16,correct=sum(rewards),trajectories=len(rewards))},
            **diagnostics)
        save(folder/'complete.json',summary); core.append(trainer.out/'metrics.jsonl',summary)
        trainer.sync()
        print('BATCH',trainer.completed,'accepted',trainer.completed*16,'attempted',trainer.attempted,
              'accuracy',round(summary['all_attempt_accuracy'],4),'seconds',round(summary['seconds'],2),flush=True)
        del groups,logs,state
        accepted = trainer.completed*16
        if accepted%512==0:
            atomic_torch_save(trainer.snapshot(),trainer.out/f'selector-{accepted:06d}.pt')
            trainer.evaluate('train_probe',101)
        if accepted in (4096,8192,16384,32768):
            for seed in (101,202,303):
                for split in TESTS: trainer.evaluate(split,seed)
    for seed in (101,202,303):
        for split in TESTS: trainer.evaluate(split,seed)
    save(trainer.out/'COMPLETE.json',dict(completed=time.time(),accepted_groups=32768,
         optimizer_updates=8192,attempted_groups=trainer.attempted,
         final_test_counts={f'{s}-{seed}':len(trainer.rows[s]) for s in TESTS for seed in (101,202,303)}))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('mode',choices=('validate','train'))
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    assert os.uname().nodename.split('.')[0]=='i8027' and torch.cuda.device_count()==8
    assert os.environ['SLURM_JOB_ID']=='2357570'
    assert json.loads((BASE/'source/recipe.json').read_text())['nfe']==16
    assert json.loads((BASE/'RUNTIME_VALIDATED.json').read_text())['passed']
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    core.Replica = CodeReplica
    folder = BASE/'source'/('validation' if args.mode=='validate' else 'production')
    assert args.resume or not folder.exists(), 'Never overwrite previous workflow output'
    assert not args.resume or args.mode=='train'
    trainer = Trainer(BASE/'source',folder,BASE/'runtime',list(range(8)))
    try:
        if args.mode=='validate':
            for split in ('train',)+TESTS:
                longest = int(trainer.inputs[split]['mask'].sum(-1).argmax())
                uniforms = torch.stack([draws(longest,trainer.recipe.nfe,101)]*8,1)
                a = trainer.replicas[0].generate(trainer.inputs[split],[longest]*8,uniforms,False)
                b = trainer.replicas[1].generate(trainer.inputs[split],[longest]*8,uniforms,False)
                assert a['tokens']==b['tokens'], ('Longest-context cross-GPU mismatch',split)
                save(folder/f'{split}-longest-parity.json',dict(passed=True,index=longest,
                     tokens_equal=True,nfe=trainer.recipe.nfe,prompt_tokens=len(trainer.rows[split][longest]['prompt_ids'])))
            # Same repeatability, replay, hidden-projection update and exact full-state resume gates.
            core.validate(trainer)
            gate = json.loads((folder/'VALIDATED.json').read_text())
            assert gate['optimizer_steps']==4 and gate['trajectories']==128
            save(folder/'context-gates.json',[r.context_gate for r in trainer.replicas])
        else:
            assert not args.resume, 'This workflow is a fresh NFE16 run, not a continuation'
            save(BASE/'PRODUCTION_STARTED.json',dict(time=time.time(),node='i8027',
                 holder=os.environ['SLURM_JOB_ID'],step=os.environ['SLURM_STEP_ID'],
                 initialization='fresh',recipe=trainer.recipe.to_dict(),
                 initial_policy_sha256=trainer.behavior_hash,
                 source_manifest_sha256=sha(BASE/'training-source.sha256')))
            production(trainer,args.resume)
    except BaseException as exc:
        save(folder/f'failure-{int(time.time())}.json',dict(type=type(exc).__name__,message=str(exc),time=time.time()))
        raise
    finally: trainer.close()


if __name__=='__main__': main()
