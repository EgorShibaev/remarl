"""Audit saved answers and metrics without touching any cluster experiment."""
import importlib.util
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('common_deploy',BASE.parent/'2026-10-22-code32-i8022/deploy.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
REMOTE=r'''
import datetime,hashlib,json,sys
from pathlib import Path
b=Path(sys.argv[1]);config=json.loads((b/'STAGED.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text()) if p.exists() else None
rows=read(b/'shared/rows.json');calibration=read(b/'controls/CALIBRATION_PROTOCOL.json')
assert sha(b/'shared/inputs.pt')==config['inputs_sha256']
assert sha(b/'shared/rows.json')==config['rows_sha256']
assert calibration['training_only'] and not calibration['held_out']
assert len(set(calibration['training_indices']))==256
rows['calibration']=[rows['train'][i] for i in calibration['training_indices']]
assert all(sum(r['task']==task for r in rows['calibration'])==128 for task in ('gsm8k','math'))
def audit(summary,folder,split):
 raw=(folder/'answers.jsonl').read_bytes();records=[json.loads(x) for x in raw.splitlines()]
 count=len(rows[split]);assert count in (256,500,1319)
 assert summary['complete'] and summary['count']==count==len(records)
 assert [r['index'] for r in records]==list(range(count))
 assert all(r['reward'] in (0,1) and len(r['tokens'])==256 and 126336 not in r['tokens'] for r in records)
 assert all(r['task']==rows[split][r['index']]['task'] for r in records)
 assert sum(r['reward'] for r in records)==summary['correct']
 digest=hashlib.sha256(raw).hexdigest()
 if 'answers_sha256' in summary:assert summary['answers_sha256']==digest
 if 'mean_revisions' in summary:
  assert abs(sum(r['revisions'] for r in records)/count-summary['mean_revisions'])<1e-10
 return dict(summary,answers_sha256=digest,origin=str(folder))
out=dict(checked_at=datetime.datetime.now().astimezone().isoformat(),config=config,
         calibration_protocol=calibration,variants={},controls=[])
fields=('batch','accepted_groups','optimizer_updates','attempted_groups','attempted_this_batch',
        'all_attempt_accuracy','accepted_accuracy','acceptance_rate','mean_revisions','task_metrics',
        'seconds','time','behavior_hash','new_policy_hash','replay_error','lr')
for variant in config['variants']:
 v=b/'variants'/variant;s=v/'source';p=s/'production/metrics.jsonl'
 metric_bytes=p.read_bytes() if p.exists() else b''
 metrics=[json.loads(line) for line in metric_bytes.splitlines()]
 for i,m in enumerate(metrics,1):
  assert m['batch']==i and m['accepted_groups']==16*i and m['optimizer_updates']==4*i
  assert len(m['updates'])==4 and all(u['trajectories']==128 for u in m['updates'])
  if i>1:assert m['behavior_hash']==metrics[i-2]['new_policy_hash']
 tests=[]
 for p in sorted((s/'production/evaluations').glob('*/summary.json')):
  summary=read(p)
  if not summary['split'].startswith('test_'):continue
  assert summary['nfe']==8 and summary['position_temperature']==1. and summary['token_temperature']==0.
  tests.append(audit(summary,p.parent,summary['split']))
 out['variants'][variant]=dict(metrics=[{k:m[k] for k in fields} for m in metrics],
    metrics_sha256=hashlib.sha256(metric_bytes).hexdigest() if metrics else None,tests=tests,
    gate=read(s/'validation/VALIDATED.json'),failed=read(v/'FAILED.json'),complete=read(v/'COMPLETE.json'))
for p in sorted((b/'controls/results').glob('*/summary.json')):
 summary=read(p);assert summary['nfe']==8
 out['controls'].append(audit(summary,p.parent,summary['split']))
out['temperature_selected']=read(b/'controls/TEMPERATURE_SELECTED.json')
out['revision_match_selected']=read(b/'controls/REVISION_MATCH_SELECTED.json')
out['runs']=read(b/'RUNS.json');out['controls_complete']=read(b/'controls/COMPLETE.json')
out['controls_failed']=read(b/'CONTROLS_FAILED.json');out['suite_failed']=read(b/'SUITE_FAILED.json')
print(json.dumps(out))
'''

if __name__=='__main__':
    config=json.loads((BASE/'STAGED.json').read_text())
    result=json.loads(common.remote(REMOTE,config['root']))
    (BASE/'RESULTS_AUDITED.json').write_text(json.dumps(result,indent=2)+'\n')
    for variant,row in result['variants'].items():
        last=row['metrics'][-1] if row['metrics'] else {}
        print(variant,{k:last.get(k) for k in ('accepted_groups','optimizer_updates','seconds')},
              'tests',len(row['tests']),'failed',row['failed'])
    print('controls',len(result['controls']),'held-out cells',sum(r['held_out'] for r in result['controls']))
