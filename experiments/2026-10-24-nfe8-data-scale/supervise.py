"""Four GPU pairs, each executing three seeds of one data size."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
GPU_IDS = os.environ['CUDA_VISIBLE_DEVICES'].split(',')


def run_size(slot, size):
    outcomes = []
    for seed in (7,17,27):
        name = f'n{size*2}-seed{seed}'
        env = os.environ.copy()
        env.update(SCALE_RUN=name, SCALE_SEED=str(seed),
                   CUDA_VISIBLE_DEVICES=','.join(GPU_IDS[slot*2:slot*2+2]),
                   PYTHONPATH=str(BASE/'runs'/name/'source')+':'+str(BASE))
        with (BASE / f'{name}.log').open('xb') as log:
            p = subprocess.run([sys.executable, str(BASE/'train.py')], env=env,
                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        outcomes.append(dict(name=name, exit_code=p.returncode, time=time.time()))
        (BASE / f'size-{size}-STATUS.json').write_text(json.dumps(outcomes, indent=2)+'\n')
        # A failed seed must not prevent independent seeds from running.
    return outcomes


if __name__ == '__main__':
    assert len(GPU_IDS) == 8 and (BASE/'SUBSETS_READY.json').exists()
    (BASE/'STARTED.json').write_text(json.dumps(dict(time=time.time(), node=os.uname().nodename,
        step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'], gpus=GPU_IDS))+'\n')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda item: run_size(*item), enumerate((4,32,256,2048))))
    (BASE/'FINISHED.json').write_text(json.dumps(dict(time=time.time(), outcomes=results), indent=2)+'\n')
