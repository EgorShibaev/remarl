"""Bind completed three-pass reference audit to this unchanged node/runtime."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

from cohort_hashes import sha
from validated_training import load_validated
BASE = Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'protocol'))
from prepare import preprocess, sanitizer, save
from sandbox import run, validate
from code_reward import checked_verdict


def main():
    assert not (BASE/'RUNTIME_VALIDATED.json').exists()
    save(BASE/'sandbox-validation.json',validate())
    rows,allow,manifest = load_validated(BASE/'cohort',BASE/'protocol')
    sanit = sanitizer(); process = preprocess()
    started = time.monotonic()
    evidence = BASE/'compatibility-evidence'
    audit = json.loads((evidence/'COMPLETE.json').read_text())
    assert audit['completed'] and len(audit['repeats'])==3 and audit['reference_count']==9153
    assert audit['python']==sys.version==manifest['runtime_audit_python']
    assert audit['original_manifest_sha256']==manifest['parent_manifest_sha256']
    assert sha(evidence/'COMPLETE.json')==manifest['runtime_reaudit_sha256']
    assert sha(BASE/'runtime/packages.txt')==manifest['runtime_packages_sha256']
    expected={a['source_index']:a['task_id'] for a in allow}
    for repeat in audit['repeats']:
        path=evidence/f"repeat-{repeat['repeat']}.jsonl"
        assert sha(path)==repeat['sha256']
        records=[json.loads(s) for s in path.read_text().splitlines()]
        assert len(records)==9153 and len({r['source_index'] for r in records})==9153
        retained=[r for r in records if r['source_index'] in expected]
        assert len(retained)==len(allow)
        assert all(r['task_id']==expected[r['source_index']] and r['passed']
                   and r['result']=='passed' for r in retained)
    total=len(allow)
    save(BASE/'reference-progress.json',dict(count=total,expected=len(rows),passed=total,
         repeats=3,audit_receipt_sha256=sha(evidence/'COMPLETE.json')))
    print('REFERENCE_REVALIDATED',total,'THREE_REPEAT_RECEIPTS_BOUND',flush=True)
    data = json.loads((BASE/'source/rows.json').read_text())
    benchmarks = {}
    for split in ('test_humaneval','test_mbpp'):
        def check_bench(row):
            g = row['gold']
            code = (sanit.sanitize_humaneval(g['reference'],g['entry_point'])
                    if g['task']=='humaneval' else sanit.sanitize_mbpp(g['reference']))
            return dict(task_id=g['task_id'],**checked_verdict(run(code+'\n'+g['tests'])))
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(check_bench,data[split]))
        expected_passed = 163 if split=='test_humaneval' else 498
        assert sum(r['passed'] for r in results)==expected_passed, ('Changed benchmark runtime',split,results)
        benchmarks[split] = dict(count=len(results),passed=expected_passed,
                                 failed=[r for r in results if not r['passed']],no_exclusions=True)
    save(BASE/'RUNTIME_VALIDATED.json',dict(passed=True,retained_references=total,reference_repeats=3,
         repeated_reference_receipt_sha256=sha(evidence/'COMPLETE.json'),
         benchmark_reference_audits=benchmarks,seconds=time.monotonic()-started,
         dataset_manifest_sha256=sha(BASE/'cohort/MANIFEST.json'),
         source_manifest_sha256=sha(BASE/'training-source.sha256'),
         sandbox_validation_sha256=sha(BASE/'sandbox-validation.json'),
         packages_sha256=sha(BASE/'runtime/packages.txt'),python=sys.version,node='i8026'))
    print('RUNTIME_VALIDATED',total,flush=True)


if __name__=='__main__': main()
