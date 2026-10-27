"""Run only this suite; controls release their GPU pair to no-revision training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent
GPU_IDS=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
assert len(GPU_IDS)==8


def save(name,obj):
    p=BASE/name;tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(p)


def environment(variant,gpus):
    env=os.environ.copy();env['ABLATION_ROOT']=str(BASE);env['ABLATION_VARIANT']=variant
    env['CUDA_VISIBLE_DEVICES']=','.join(GPU_IDS[i] for i in gpus)
    source=BASE/'shared' if variant=='controls' else BASE/'variants'/variant/'source'
    env['PYTHONPATH']=str(source)+':'+str(BASE)
    return env


def main():
    assert not (BASE/'STARTED.json').exists()
    save('STARTED.json',dict(time=time.time(),step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
        node=os.uname().nodename,gpus=GPU_IDS,all_six_ablations_authorized=True))
    with (BASE/'cpu-tests.log').open('xb') as log:
        result=subprocess.run([sys.executable,str(BASE/'test_ablation.py')],env=environment('no_revision',(0,1)),
                              stdout=log,stderr=subprocess.STDOUT,timeout=180)
    assert result.returncode==0,'CPU ablation tests failed; no scientific job launched'
    save('CPU_VALIDATED.json',dict(time=time.time(),passed=True,tests=7))
    active={};finished={};launched={}

    def launch(variant,gpus):
        assert variant not in launched
        script='evaluate_controls.py' if variant=='controls' else 'train_ablation.py'
        with (BASE/(variant+'.log')).open('xb') as log:
            process=subprocess.Popen([sys.executable,str(BASE/script)],env=environment(variant,gpus),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        active[variant]=process
        launched[variant]=dict(time=time.time(),pid=process.pid,gpu_indices=list(gpus))
        save('RUNS.json',dict(launched=launched,finished=finished))

    launch('controls',(0,1))
    launch('no_hidden',(2,3))
    launch('no_position',(4,5))
    launch('bias_only',(6,7))
    while active:
        for variant,p in list(active.items()):
            result=p.poll()
            if result is None:continue
            finished[variant]=dict(time=time.time(),exit_code=result)
            del active[variant]
            save('RUNS.json',dict(launched=launched,finished=finished))
            if variant=='controls':launch('no_revision',(0,1))
        time.sleep(10)
    assert set(finished)=={'controls','no_hidden','no_position','bias_only','no_revision'}
    assert all(r['exit_code']==0 for r in finished.values()),finished
    save('COMPLETE.json',dict(time=time.time(),runs=finished,all_six_ablations_complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save('SUITE_FAILED.json',dict(time=time.time(),type=type(exc).__name__,message=str(exc)))
        raise
