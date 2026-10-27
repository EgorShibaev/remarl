"""Wait for evaluations to finish; the filename must not shadow stdlib queue."""
import json
import os
from pathlib import Path
import subprocess
import time

BASE=Path(__file__).resolve().parent
PREVIOUS=Path('/data/horse/ws/egsh587h-ldlm/remasking/baseline-reevaluation-i8022-20261024')


def state(**data):
    p=BASE/'QUEUE_STATUS.json';t=p.with_suffix('.tmp')
    t.write_text(json.dumps(dict(time=time.time(),**data),indent=2)+'\n');t.replace(p)


def main():
    assert os.getuid()==2224008
    with (BASE/'QUEUE_STARTED.json').open('x') as f:
        json.dump(dict(time=time.time(),pid=os.getpid(),previous=str(PREVIOUS)),f)
    for name in ('CPU_TESTS.json','GPU_TINY_TESTS.json','GPU_REAL_TESTS.json'):
        assert json.loads((BASE/name).read_text())['passed'],name
    while True:
        holder=subprocess.check_output(['scontrol','show','job','2362070','-o'],text=True)
        if not all(s in holder for s in ('JobState=RUNNING','NodeList=i8022 ','UserId=egsh587h(2224008)')):
            raise RuntimeError('Verified holder is no longer available')
        steps=subprocess.check_output(['squeue','--steps','-j','2362070','-h','-o','%i'],text=True).splitlines()
        active=[s for s in steps if not s.endswith(('.batch','.extern'))]
        if (PREVIOUS/'COMPLETE.json').exists() and not active:
            break
        state(state='WAITING',previous_complete=(PREVIOUS/'COMPLETE.json').exists(),active_steps=active)
        time.sleep(60)
    command=['srun','--jobid=2362070','--overlap','--exact','-N1','-n1','-c32',
             '--mem=320G','--gres=gpu:8','--nodelist=i8022','--time=2-00:00:00',
             '--job-name=revision','--chdir='+str(BASE),'bash',str(BASE/'run.sh'),str(BASE)]
    state(state='STARTING',command=command)
    result=subprocess.run(command)
    if result.returncode:
        raise RuntimeError(f'Evaluation exited with status {result.returncode}')
    assert (BASE/'COMPLETE.json').exists()
    state(state='COMPLETE')


if __name__=='__main__':
    try:main()
    except BaseException as e:
        state(state='FAILED',error=repr(e));raise
