"""Launch after WINO/ReMDM releases the owned holder; avoid stdlib module names."""
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime

BASE=Path(__file__).resolve().parent
PREVIOUS=BASE.parent/'revision-adapters-20261024'


def state(**data):
    p=BASE/'QUEUE_STATUS.json';t=p.with_suffix('.tmp')
    t.write_text(json.dumps(dict(time=time.time(),**data),indent=2)+'\n');t.replace(p)


def main():
    assert os.getuid()==2224008
    assert json.loads((BASE/'CPU_TESTS.json').read_text())['passed']
    with (BASE/'QUEUE_STARTED.json').open('x') as f:
        json.dump(dict(time=time.time(),pid=os.getpid(),previous=str(PREVIOUS)),f)
    while True:
        holder=subprocess.check_output(['scontrol','show','job','2362070','-o'],text=True)
        assert all(s in holder for s in ('JobState=RUNNING','NodeList=i8022 ','UserId=egsh587h(2224008)')),holder
        steps=subprocess.check_output(['squeue','--steps','-j','2362070','-h','-o','%i'],text=True).splitlines()
        active=[s for s in steps if not s.endswith(('.batch','.extern'))]
        previous_complete=(PREVIOUS/'COMPLETE.json').exists()
        if previous_complete and not active:
            break
        state(state='WAITING',previous=str(PREVIOUS),previous_complete=previous_complete,
              active_steps=active,validation='CPU passed; CUDA and full-pipeline probes run before evaluation')
        time.sleep(60)
    fields=dict(s.split('=',1) for s in holder.split() if '=' in s)
    end=datetime.fromisoformat(fields['EndTime']).timestamp()
    seconds=min(48*3600,int(end-time.time())-300)
    assert seconds>3600,'Holder has insufficient time remaining'
    minutes=seconds//60
    command=['srun','--jobid=2362070','--overlap','--exact','-N1','-n1','-c32',
             '--mem=320G','--gres=gpu:8','--nodelist=i8022','--time='+str(minutes),
             '--job-name=extras','--chdir='+str(BASE),'bash',str(BASE/'run.sh'),str(BASE)]
    state(state='VALIDATING_AND_STARTING',command=command)
    result=subprocess.run(command)
    assert result.returncode==0,f'Validation/evaluation exited {result.returncode}'
    assert (BASE/'COMPLETE.json').exists()
    state(state='COMPLETE')


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        state(state='FAILED',error=repr(exc));raise
