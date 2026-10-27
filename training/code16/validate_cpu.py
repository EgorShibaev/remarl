"""Compute-only security, watchdog and unchanged reward-reference compatibility gates."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

BASE = Path(__file__).resolve().parent
PARENT = Path('/data/horse/ws/egsh587h-ldlm/remasking/code-nfe16-i8027-20261011')
sys.path.insert(0, str(BASE/'protocol'))
import sandbox
from prepare import preprocess, sanitizer
import pyarrow.parquet as pq

def save(name, value):
    (BASE/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def main():
    assert os.uname().nodename.split('.')[0] == 'i8027'
    assert os.environ['SLURM_JOB_ID'] == '2357570' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    started = time.monotonic()
    health = sandbox.validate()
    with tempfile.TemporaryDirectory(dir=BASE/'protocol/scratch') as name:
        work = Path(name)
        (work/'probe.py').write_bytes((BASE/'protocol/security_probe.py').read_bytes())
        with sandbox.filter_path().open('rb') as stream:
            p = subprocess.run(sandbox.command(work, stream.fileno(), '/work/probe.py'),
                stdin=subprocess.DEVNULL, capture_output=True, timeout=10,
                pass_fds=(stream.fileno(),), close_fds=True, env={'PATH':'/usr/bin:/bin'})
        assert p.returncode == 0, p.stderr.decode(errors='replace')
        security = json.loads(p.stdout)
        assert security['passed']
    regressions = {}
    for name, program in {
        'c_loop':'sum(range(10**11))',
        'ignored_alarm':'import signal\nsignal.signal(signal.SIGALRM, signal.SIG_IGN)\nwhile True: pass',
    }.items():
        result = sandbox.run(program)
        assert not result['passed'] and result['result'] == 'timed out' and result['supervisor_timeout']
        regressions[name] = result
    try:
        sandbox.run('import posix\nposix._exit(7)')
    except RuntimeError:
        pass
    else:
        raise AssertionError('Infrastructure failure was silently scored')
    with ThreadPoolExecutor(max_workers=8) as pool:
        repeated = list(pool.map(sandbox.run, ['assert 1 + 1 == 2']*128))
    assert all(p['passed'] for p in repeated)
    save('SECURITY_VALIDATED.json',dict(passed=True,health=health,security=security,
        regressions=regressions,concurrent_probes=len(repeated),unexpected_exit_fatal=True))
    print('SECURITY_GATES_PASS', flush=True)
    clean = sanitizer(); prep = preprocess()
    train = pq.read_table(BASE/'cohort/train.parquet').to_pylist()
    assert len(train) == 9149
    def check(pair):
        i, row = pair
        program = clean.sanitize_humaneval(row['solution'],row['test_info'][0]['function_name'])+'\n'+prep(row['test'])
        return dict(index=i,task_id=row['question_id'],**sandbox.run(program))
    records = []
    with (BASE/'references.jsonl').open('x') as handle, ThreadPoolExecutor(max_workers=8) as pool:
        for row in pool.map(check, enumerate(train)):
            handle.write(json.dumps(row)+'\n');handle.flush();records.append(row)
            if len(records)%500 == 0: print('REFERENCES',len(records),flush=True)
    assert all(r['passed'] for r in records), [(r['index'],r['result']) for r in records if not r['passed']]
    data = json.loads((BASE/'source/rows.json').read_text())
    benchmarks = {}
    expected = json.loads((PARENT/'RUNTIME_VALIDATED.json').read_text())['benchmark_reference_audits']
    for split in ('test_humaneval','test_mbpp'):
        def bench(row):
            gold=row['gold']
            program=(clean.sanitize_humaneval(gold['reference'],gold['entry_point'])
                if gold['task']=='humaneval' else clean.sanitize_mbpp(gold['reference']))
            return dict(task_id=gold['task_id'],**sandbox.run(program+'\n'+gold['tests']))
        with ThreadPoolExecutor(max_workers=8) as pool:
            rows = list(pool.map(bench,data[split]))
        failures = [r for r in rows if not r['passed']]
        assert len(rows) == expected[split]['count']
        assert sum(r['passed'] for r in rows) == expected[split]['passed']
        assert {r['task_id'] for r in failures} == {r['task_id'] for r in expected[split]['failed']}
        benchmarks[split]=dict(count=len(rows),passed=sum(r['passed'] for r in rows),failed=failures,no_exclusions=True)
    official = {}
    for p in (BASE/'protocol/official').iterdir():
        if p.is_file():
            assert p.read_bytes() == (PARENT/'protocol/official'/p.name).read_bytes()
            official[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    receipt=dict(passed=True,training_count=9149,training_passed=9149,
        benchmarks=benchmarks,official_hashes=official,
        source_cohort_bytes_unchanged=True,no_benchmark_exclusions=True,
        seconds=time.monotonic()-started,node='i8027',step=os.environ['SLURM_STEP_ID'],
        reference_sha256=hashlib.sha256((BASE/'references.jsonl').read_bytes()).hexdigest())
    save('REFERENCE_VALIDATED.json',receipt)
    print('FULL_REFERENCE_COMPATIBILITY_PASS',flush=True)

if __name__ == '__main__':
    main()
