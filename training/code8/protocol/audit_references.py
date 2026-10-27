"""Full reference execution audit; never quietly repair or filter official tests."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from prepare import load_raw, preprocess, sanitizer, save
from sandbox import run, validate

BASE=Path(__file__).resolve().parent
def append(path,row):
    with path.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
def main():
    save(BASE/'sandbox-validation.json',validate())
    raw=load_raw();sanitize=sanitizer();process=preprocess()
    rows=json.loads((BASE/'rows.json').read_text())
    with ThreadPoolExecutor(max_workers=8) as pool:
        for split in ('humaneval','mbpp'):
            path=BASE/f'reference-{split}.jsonl';assert not path.exists()
            def check(row):
                code=(sanitize.sanitize_humaneval(row['reference'],row['entry_point'])
                      if split=='humaneval' else sanitize.sanitize_mbpp(row['reference']))
                return dict(index=row['index'],task_id=row['task_id'],**run(code+'\n'+row['tests']))
            results=list(pool.map(check,rows[split]))
            for result in results:append(path,result)
            save(BASE/f'reference-{split}-summary.json',dict(count=len(results),
                 passed=sum(r['passed'] for r in results),failed=[r for r in results if not r['passed']],
                 unchanged_tests=True,no_exclusions=True))
            print('REFERENCE_BENCHMARK',split,sum(r['passed'] for r in results),len(results),flush=True)
        path=BASE/'reference-kodcode.jsonl';assert not path.exists()
        def check_train(pair):
            i,row=pair;info=row['test_info'][0]
            # Stored solution already contains the declaration, unlike continuation tokens.
            code=sanitize.sanitize_humaneval(row['solution'],info['function_name'])
            return dict(index=i,task_id=row.get('question_id',row.get('id',i)),
                        **run(code+'\n'+process(row['test'])))
        total=passed=0;failed=[];started=time.monotonic()
        for result in pool.map(check_train,enumerate(raw['kodcode'])):
            append(path,result);total+=1;passed+=result['passed']
            if not result['passed']:failed.append(result)
            if total%128==0 or total==10000:
                save(BASE/'reference-kodcode-progress.json',dict(count=total,expected=10000,passed=passed,
                     seconds=time.monotonic()-started))
                print('REFERENCE_KODCODE',total,passed,flush=True)
        save(BASE/'reference-kodcode-summary.json',dict(count=total,passed=passed,failed=failed,
             unchanged_official_direct_call_harness=True,no_exclusions=True,seconds=time.monotonic()-started))
    save(BASE/'REFERENCE_AUDIT_COMPLETE.json',dict(time=time.time(),benchmarks=664,train=10000))
if __name__=='__main__':main()
