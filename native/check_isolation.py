"""Fail closed unless the authorized physical GPUs/CPU cores are isolated."""
import json
import os
from pathlib import Path
import subprocess
import torch

assert os.uname().nodename.split('.')[0]=='i8020'
assert os.environ['SLURM_JOB_ID']=='2357849'
assert os.environ['CUDA_VISIBLE_DEVICES']=='0,1,2,3,4,5,6'
assert os.sched_getaffinity(0)==set(range(23)) | set(range(24,47))
assert Path.cwd().is_relative_to('/data/horse/ws/alpo658h-ldlm_eval_20260921')
expected=['ad4f286f-5264-eab2-ee4b-75c688daf82e','0c2a0cbf-4a99-33d2-1279-ed864d881802',
          'ad0aee88-08b4-f8a3-9ad5-f2acb0b1e1cb','c09ddaca-4bfa-62f3-5966-c3b4c6220ac8',
          'd5418933-4974-3172-880a-0e60d265f67c','403581cb-e628-eab8-967c-7759886e7ef0',
          '9fe995e8-f257-d3b7-adb5-be21fb0179ee']
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
assert not any(u in active for u in expected), 'Authorized GPUs are no longer empty'
assert torch.cuda.device_count()==7
actual=[str(torch.cuda.get_device_properties(i).uuid).lower().removeprefix('gpu-') for i in range(7)]
assert actual==expected,(actual,expected)
for i in range(7):
 t=torch.ones(8,device=f'cuda:{i}'); assert t.sum().item()==8
record=dict(physical_gpus=list(range(7)),uuids=actual,cpus=sorted(os.sched_getaffinity(0)),
            prior_gpu_processes=active,passed=True)
Path('resource-isolation.json').write_text(json.dumps(record,indent=2))
print('RESOURCE_ISOLATION_PASS',json.dumps(record),flush=True)
