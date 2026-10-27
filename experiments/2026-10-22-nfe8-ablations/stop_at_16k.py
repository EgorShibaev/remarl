"""User-authorized end of the ablation sweep; preserve holders and all artifacts."""
import json
import subprocess
from pathlib import Path

SCRIPT = r'''
import datetime, hashlib, json, subprocess, time
from pathlib import Path
base=Path('/data/horse/ws/egsh587h-ldlm/remasking/nfe8-ablations-i8022-20261022')
delegate=Path('/data/horse/ws/egsh587h-ldlm/remasking/nfe8-no-revision-i8008-20261022')
targets={'2362070.7':('a8abl','i8022'),'2362071.6':('a8norev','i8008')}
receipt=dict(time=datetime.datetime.now().astimezone().isoformat(),reason='User requested stopping ablations and using matched 16384-group checkpoints',selected_groups=16384,steps=list(targets),checkpoints={},holders_preserved=True)
for step,(name,node) in targets.items():
 text=subprocess.check_output(['scontrol','show','step',step],text=True)
 assert 'UserId=2224008' in text and 'Name='+name in text and 'NodeList='+node in text and 'State=RUNNING' in text,text
for variant in ('no_hidden','no_position','bias_only','no_revision'):
 p=base/'variants'/variant/'source/production'
 checkpoint=p/'selector-016384.pt'
 assert checkpoint.is_file() and checkpoint.stat().st_size>100000
 tests=[json.loads(s.read_text()) for s in (p/'evaluations').glob('*/summary.json')]
 chosen=[s for s in tests if s.get('accepted_groups')==16384 and s.get('split') in ('test_gsm8k','test_math')]
 assert len(chosen)==6 and all(s['complete'] for s in chosen)
 assert sorted((s['split'],s['seed'],s['count']) for s in chosen)==sorted((split,seed,count) for split,count in [('test_gsm8k',1319),('test_math',500)] for seed in (101,202,303))
 last=json.loads((p/'metrics.jsonl').read_text().splitlines()[-1])
 receipt['checkpoints'][variant]=dict(selected_path=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),latest_groups_before_stop=last['accepted_groups'],latest_updates_before_stop=last['optimizer_updates'],full_state_path=str(p/'latest.pt'))
for root in (base,delegate):
 path=root/'USER_STOP_REQUESTED_16K.json'
 with path.open('x') as f:json.dump(receipt,f,indent=2)
subprocess.run(['scancel',*targets],check=True)
for attempt in range(20):
 active=subprocess.check_output(['squeue','--steps','-j','2362070,2362071','-h','-o','%i|%j'],text=True)
 if not any(s in active for s in targets):break
 time.sleep(2)
assert not any(s in active for s in targets),active
receipt['verified_at']=datetime.datetime.now().astimezone().isoformat()
receipt['remaining_steps']=active
receipt['accounting']=subprocess.check_output(['sacct','-j',','.join(targets),'-n','-P','-o','JobID,JobName,State,ExitCode,End'],text=True)
for holder in ('2362070','2362071'):
 state=subprocess.check_output(['scontrol','show','job',holder],text=True)
 assert 'JobState=RUNNING' in state
for root in (base,delegate):
 with (root/'USER_STOPPED_16K.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
'''

if __name__ == '__main__':
    command=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no',
             '-o','ConnectTimeout=12','-o','UserKnownHostsFile=/Users/egor.shibaev/.ssh/known_hosts.before-alpha-update-20261022T094532Z',
             '-o','HostName=login2.alpha.hpc.tu-dresden.de','1.alpha','python3 -']
    result=subprocess.run(command,input=SCRIPT,text=True,capture_output=True,timeout=90,check=True)
    data=json.loads(result.stdout)
    Path(__file__).with_name('USER_STOPPED_16K.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))
