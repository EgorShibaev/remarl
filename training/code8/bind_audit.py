"""Bind complete repeated reference evidence and unchanged benchmark runtime."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'protocol'))
from sandbox import run,validate
from prepare import sanitizer
from validated_training import load_validated
from evaluate_p2 import save,sha

def main():
    assert not (BASE/'RUNTIME_VALIDATED.json').exists()
    started=time.monotonic();save(BASE/'sandbox-validation-repaired.json',validate())
    rows,allow,manifest=load_validated(BASE/'cohort',BASE/'protocol')
    audit=json.loads((BASE/'recovery-evidence/quarantine-audit/COMPLETE.json').read_text())
    assert sys.version==audit['python']==manifest['runtime_audit_python']
    assert sha(BASE/'runtime/packages.txt')==audit['packages_sha256']
    source=BASE/'recovery-evidence/quarantine-audit'
    eligible={a['source_index'] for a in allow}
    for receipt in audit['repeats']:
        p=source/f"repeat-{receipt['repeat']}.jsonl";assert sha(p)==receipt['sha256']
        results=[json.loads(s) for s in p.read_text().splitlines()]
        retained=[r for r in results if r['source_index'] in eligible]
        assert len(retained)==len(rows) and all(r['passed'] for r in retained)
    risk=source/'set-order-32-repeats.jsonl';assert sha(risk)==audit['risk_log_sha256']
    assert all(json.loads(s)['passed'] for s in risk.read_text().splitlines()
               if json.loads(s)['source_index'] in eligible)
    data=json.loads((BASE/'source/rows.json').read_text());sanitize=sanitizer();benchmarks={}
    for split in ('test_humaneval','test_mbpp'):
        def check(row):
            gold=row['gold']
            code=(sanitize.sanitize_humaneval(gold['reference'],gold['entry_point'])
                  if gold['task']=='humaneval' else sanitize.sanitize_mbpp(gold['reference']))
            return dict(task_id=gold['task_id'],**run(code+'\n'+gold['tests']))
        with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(check,data[split]))
        assert len(results)==(164 if split=='test_humaneval' else 500)
        assert sum(r['passed'] for r in results)==(163 if split=='test_humaneval' else 498)
        benchmarks[split]=dict(count=len(results),passed=sum(r['passed'] for r in results),
            failed=[r for r in results if not r['passed']],no_exclusions=True)
    save(BASE/'RUNTIME_VALIDATED.json',dict(passed=True,retained_references=len(rows),reference_repeats=3,
        reference_audit_sha256=sha(source/'COMPLETE.json'),set_risk_repeats=32,
        benchmark_reference_audits=benchmarks,dataset_manifest_sha256=sha(BASE/'cohort/MANIFEST.json'),
        source_manifest_sha256=sha(BASE/'training-source.sha256'),packages_sha256=sha(BASE/'runtime/packages.txt'),
        sandbox_validation_sha256=sha(BASE/'sandbox-validation-repaired.json'),
        python=sys.version,node='i8027',seconds=time.monotonic()-started))
    print('RUNTIME_VALIDATED',len(rows),flush=True)

if __name__=='__main__':main()
