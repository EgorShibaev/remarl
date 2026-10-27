"""Training-calibrated P2 temperature and revision-rate controls; frozen model."""
import concurrent.futures
import json
import os
from pathlib import Path
import time
import torch
import ablation_ops as ops
import train_batch_grpo as train
from logging_repair import append, save

BASE=Path(os.environ['ABLATION_ROOT'])
TEMPERATURES=(0.,.05,.1,.25,.5,1.,2.)
SEEDS=(101,202,303)


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.device_count()==2
    source=BASE/'shared';out=BASE/'controls';out.mkdir()
    save(out/'STARTED.json',dict(time=time.time(),pid=os.getpid(),gpus=os.environ['CUDA_VISIBLE_DEVICES']))
    state=torch.load(BASE/'reference-nfe8-032768.pt',map_location='cpu',weights_only=True)
    assert state['recipe']['nfe']==8 and state['completed']*state['accepted_groups_per_batch']==32768
    assert train.state_digest(state['model'])==state['policy_sha256']
    inputs=torch.load(source/'inputs.pt',map_location='cpu',weights_only=True)
    rows=json.loads((source/'rows.json').read_text())
    rng=torch.Generator().manual_seed(802208)
    chosen=[]
    for task in ('gsm8k','math'):
        candidates=[i for i,r in enumerate(rows['train']) if r['task']==task]
        chosen += [candidates[i] for i in torch.randperm(len(candidates),generator=rng)[:128].tolist()]
    assert len(set(chosen))==256
    rows['calibration']=[rows['train'][i] for i in chosen]
    inputs['calibration']={k:v[chosen] for k,v in inputs['train'].items()}
    save(out/'CALIBRATION_PROTOCOL.json',dict(training_indices=chosen,training_only=True,
        held_out=False,questions_per_task=128,seed=701,temperature_grid=TEMPERATURES,
        select_temperature_by='training-calibration macro task accuracy; ties prefer closer to T=1',
        revision_target_policy_sha256=state['policy_sha256'],revision_target_groups=32768,
        test_results_never_select_hyperparameters=True))
    scorer=train.__dict__.get('Scorer')
    from paper_protocol import Scorer
    scorer=Scorer(source,BASE/'runtime',out/'scorer.log')
    local=Path((BASE/'runtime/local_root.txt').read_text().strip())
    port=train.load_port(source);config=train.SetSelectorConfig(**state['selector_config'])
    replicas=[train.Replica(i,port,local/'models/LLaDA-8B-Instruct',config) for i in (0,1)]
    for replica in replicas:replica.synchronize(state['model'],state['policy_sha256'])
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=2)
    completed=[]

    def evaluate(name,split,seed,*,temperature=1.,bonus=0.,learned=False):
        ops.configure('full' if learned else 'p2',temperature,bonus)
        folder=out/'results'/f'{name}-{split}-seed{seed}';folder.mkdir(parents=True)
        assert not (folder/'answers.jsonl').exists()
        save(folder/'identity.json',dict(name=name,split=split,seed=seed,nfe=8,
            temperature=temperature,retention_bonus=bonus,learned=learned,answer_length=256,
            token_temperature=0.,policy_sha256=state['policy_sha256'] if learned else None))
        records=[];count=len(rows[split]);started=time.monotonic()
        for start in range(0,count,16):
            futures=[]
            for gpu,replica in enumerate(replicas):
                ids=list(range(start+gpu*8,min(start+(gpu+1)*8,count)))
                if not ids:break
                real=len(ids);ids += [ids[-1]]*(8-real)
                uniforms=torch.stack([train.draws(i,8,seed) for i in ids],1)
                futures.append((ids,real,pool.submit(replica.generate,inputs[split],ids,uniforms,False,not learned)))
            for ids,real,future in futures:
                result=future.result(timeout=600);assert result['nfe_calls']==8
                answers=scorer(result['tokens'][:real],[rows[split][i]['gold'] for i in ids[:real]])
                for i,tokens,answer,revisions in zip(ids[:real],result['tokens'][:real],answers,result['replay'].revisions[:real]):
                    assert i==len(records)
                    record=dict(index=i,tokens=tokens,revisions=int(revisions),**answer)
                    assert record['task']==rows[split][i]['task']
                    records.append(record);append(folder/'answers.jsonl',record)
            save(folder/'progress.json',dict(count=len(records),expected=count,time=time.time()))
            print('CONTROL_PROGRESS',name,split,seed,len(records),count,flush=True)
        summary=dict(name=name,split=split,seed=seed,nfe=8,count=count,complete=True,
            correct=sum(r['reward'] for r in records),mean_revisions=sum(r['revisions'] for r in records)/count,
            macro_accuracy=sum(sum(r['reward'] for r in records if r['task']==task)/sum(r['task']==task for r in records)
                               for task in ('gsm8k','math') if any(r['task']==task for r in records))
                           /len({r['task'] for r in records}),
            seconds=time.monotonic()-started,answers_sha256=train.sha(folder/'answers.jsonl'),
            temperature=temperature,retention_bonus=bonus,held_out=split.startswith('test_'))
        save(folder/'summary.json',summary);completed.append(summary)
        save(out/'progress.json',dict(time=time.time(),cells_complete=len(completed),cells=completed,complete=False))
        return summary

    try:
        target=evaluate('learned-reference','calibration',701,learned=True)
        temperatures=[evaluate(f'p2-temp-{v:g}','calibration',701,temperature=v) for v in TEMPERATURES]
        best=max(temperatures,key=lambda r:(r['macro_accuracy'],-abs(r['temperature']-1.)))
        save(out/'TEMPERATURE_SELECTED.json',dict(temperature=best['temperature'],selection_split='training-calibration',
            calibration=temperatures,selected_before_any_test=True))
        candidates=[]
        for bonus in (0.,.25,.5,1.,2.,4.,8.,16.):
            candidates.append(evaluate(f'p2-retain-{bonus:g}','calibration',701,bonus=bonus))
        goal=target['mean_revisions']
        for j in range(6):
            above=[r for r in candidates if r['mean_revisions']>=goal]
            below=[r for r in candidates if r['mean_revisions']<=goal]
            if not above or not below:break
            low=min(above,key=lambda r:r['mean_revisions']-goal)['retention_bonus']
            high=min(below,key=lambda r:goal-r['mean_revisions'])['retention_bonus']
            bonus=(low+high)/2
            if any(abs(r['retention_bonus']-bonus)<1e-8 for r in candidates):break
            candidates.append(evaluate(f'p2-retain-refine{j}','calibration',701,bonus=bonus))
        matched=min(candidates,key=lambda r:(abs(r['mean_revisions']-goal),r['retention_bonus']))
        save(out/'REVISION_MATCH_SELECTED.json',dict(retention_bonus=matched['retention_bonus'],
            target_mean_revisions=goal,control_mean_revisions=matched['mean_revisions'],
            absolute_match_error=abs(matched['mean_revisions']-goal),selection_split='training-calibration',
            selected_before_any_test=True,candidates=candidates))
        for temperature in TEMPERATURES:
            for seed in ((101,) if temperature==0 else SEEDS):
                for split in ('test_gsm8k','test_math'):
                    evaluate(f'p2-temp-{temperature:g}',split,seed,temperature=temperature)
        for seed in SEEDS:
            for split in ('test_gsm8k','test_math'):
                evaluate('p2-revision-matched',split,seed,bonus=matched['retention_bonus'])
        tests=[r for r in completed if r['held_out']]
        assert len(tests)==44 and all(r['count']==(1319 if r['split']=='test_gsm8k' else 500) for r in tests)
        save(out/'COMPLETE.json',dict(time=time.time(),complete=True,test_cells=44,calibration_cells=len(completed)-44))
    finally:
        pool.shutdown(wait=True,cancel_futures=True);scorer.close()


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save(BASE/'CONTROLS_FAILED.json',dict(time=time.time(),type=type(exc).__name__,message=str(exc)))
        raise
