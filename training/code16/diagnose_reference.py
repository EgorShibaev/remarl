"""Bounded, predeclared repeat count; never retry until a reference passes."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parent
PARENT=Path('/data/horse/ws/egsh587h-ldlm/remasking/code-nfe16-i8027-20261011')
sys.path.insert(0,str(BASE/'protocol'))
from prepare import preprocess,sanitizer
from sandbox import run
import pyarrow.parquet as pq

def main():
    assert os.uname().nodename.split('.')[0]=='i8027' and os.environ['CUDA_VISIBLE_DEVICES']==''
    clean=sanitizer();prep=preprocess()
    table=pq.read_table(BASE/'cohort/train.parquet');parent=pq.read_table(PARENT/'cohort/train.parquet')
    row=table.slice(7883,1).to_pylist()[0]
    assert row==parent.slice(7883,1).to_pylist()[0]
    program=clean.sanitize_humaneval(row['solution'],row['test_info'][0]['function_name'])+'\n'+prep(row['test'])
    count=64
    def repeat(i):return dict(repeat=i,**run(program))
    with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(repeat,range(count)))
    allow=[json.loads(x) for x in (BASE/'cohort/allowlist.jsonl').read_text().splitlines()]
    diagnostic=dict(index=7883,source_index=allow[7883]['source_index'],task_id=row['question_id'],
        repeats=count,passed=sum(r['passed'] for r in results),results=results,
        program_sha256=hashlib.sha256(program.encode()).hexdigest(),
        row_identical_to_parent=True,solution=row['solution'],tests=row['test'])
    (BASE/'REFERENCE_DIAGNOSTIC.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
    print('REFERENCE_DIAGNOSTIC',diagnostic['source_index'],diagnostic['passed'],count,flush=True)
    data=json.loads((BASE/'source/rows.json').read_text())
    expected=json.loads((PARENT/'RUNTIME_VALIDATED.json').read_text())['benchmark_reference_audits']
    benchmarks={}
    for split in ('test_humaneval','test_mbpp'):
        def bench(row):
            g=row['gold'];code=(clean.sanitize_humaneval(g['reference'],g['entry_point'])
                if g['task']=='humaneval' else clean.sanitize_mbpp(g['reference']))
            return dict(task_id=g['task_id'],**run(code+'\n'+g['tests']))
        with ThreadPoolExecutor(max_workers=8) as pool:out=list(pool.map(bench,data[split]))
        failures=[x for x in out if not x['passed']]
        assert len(out)==expected[split]['count']
        assert sum(x['passed'] for x in out)==expected[split]['passed']
        assert {x['task_id'] for x in failures}=={x['task_id'] for x in expected[split]['failed']}
        benchmarks[split]=dict(count=len(out),passed=sum(x['passed'] for x in out),failed=failures)
    (BASE/'BENCHMARK_REFERENCE_VALIDATED.json').write_text(json.dumps(dict(passed=True,benchmarks=benchmarks,no_exclusions=True),indent=2)+'\n')
    print('BENCHMARK_REFERENCE_COMPATIBILITY_PASS',flush=True)

if __name__=='__main__':main()
