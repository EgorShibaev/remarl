"""One explicitly approved row removed; frozen benchmarks and training probe retained."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import torch
import pyarrow.parquet as pq

BASE=Path(__file__).resolve().parent
PARENT=Path('/data/horse/ws/egsh587h-ldlm/remasking/code-nfe16-i8027-20261011')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def main():
    assert os.uname().nodename.split('.')[0]=='i8027' and os.environ['CUDA_VISIBLE_DEVICES']==''
    assert not (BASE/'QUARANTINE.json').exists() and not (BASE/'source/production').exists()
    diagnostic=json.loads((BASE/'REFERENCE_DIAGNOSTIC.json').read_text())
    assert diagnostic['source_index']==8633 and diagnostic['index']==7883 and diagnostic['passed']==62
    benchmark=json.loads((BASE/'BENCHMARK_REFERENCE_VALIDATED.json').read_text());assert benchmark['passed']
    evidence=BASE/'parent-provenance/pre-quarantine';evidence.mkdir()
    for name in ('cohort/MANIFEST.json','cohort/train.parquet','cohort/allowlist.jsonl','cohort/exclusions.jsonl',
        'cohort/runtime-compatibility-report.json','source/inputs.pt','source/rows.json','source/data-audit.json','source/recipe.json'):
        target=evidence/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(BASE/name,target)
        assert sha(target)==sha(PARENT/name),name
    allow=[json.loads(x) for x in (BASE/'cohort/allowlist.jsonl').read_text().splitlines()]
    assert len(allow)==9149 and allow[7883]['source_index']==8633
    kept=[i for i,a in enumerate(allow) if a['source_index']!=8633]
    assert kept==[i for i in range(9149) if i!=7883]
    records=[json.loads(x) for x in (BASE/'references.jsonl').read_text().splitlines()]
    assert len(records)==9149 and [r['index'] for r in records]==list(range(9149))
    assert all(records[i]['passed'] for i in kept)
    table=pq.read_table(BASE/'cohort/train.parquet');assert table.num_rows==9149
    pq.write_table(table.take(kept),BASE/'cohort/train.parquet',compression='zstd')
    retained=[allow[i] for i in kept]
    excluded=[json.loads(x) for x in (BASE/'cohort/exclusions.jsonl').read_text().splitlines()]
    excluded.append(dict(allow[7883],retained=False,reasons=['runtime_reference_unstable'],continuation_quarantine=True))
    for name,rows in [('allowlist.jsonl',retained),('exclusions.jsonl',sorted(excluded,key=lambda x:x['source_index']))]:
        (BASE/'cohort'/name).write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    cohort=json.loads((BASE/'cohort/MANIFEST.json').read_text())
    cohort.update(data_revision='validated-kodcode-v4',continuation_parent_manifest_sha256=sha(evidence/'cohort/MANIFEST.json'),
        continuation_additional_quarantine=[8633],runtime_additional_quarantine=sorted(set(cohort['runtime_additional_quarantine'])|{8633}))
    cohort['counts'].update(retained=9148,excluded=852)
    for name in ('reason_counts_nonexclusive','primary_reason_counts'):
        cohort['counts'][name]['runtime_reference_unstable']=len(cohort['runtime_additional_quarantine'])
    for name in ('train.parquet','allowlist.jsonl','exclusions.jsonl'):cohort['artifact_hashes'][name]=sha(BASE/'cohort'/name)
    save(BASE/'cohort/MANIFEST.json',cohort)
    compatibility=json.loads((BASE/'cohort/runtime-compatibility-report.json').read_text())
    compatibility.update(retained=9148,additional_excluded_sources=[8633],seccomp=True)
    save(BASE/'cohort/runtime-compatibility-report.json',compatibility)
    data=json.loads((BASE/'source/rows.json').read_text())
    inputs=torch.load(BASE/'source/inputs.pt',map_location='cpu',weights_only=True)
    assert 8633 not in {r['source_index'] for r in data['train_probe']},'Training probe must remain unchanged'
    old_data=json.loads((evidence/'source/rows.json').read_text())
    old_inputs=torch.load(evidence/'source/inputs.pt',map_location='cpu',weights_only=True)
    data['train']=[dict(data['train'][i],index=j) for j,i in enumerate(kept)]
    inputs['train']={k:v[kept].clone() for k,v in inputs['train'].items()}
    assert [r['source_index'] for r in data['train']]==[a['source_index'] for a in retained]
    for split in ('train_probe','test_humaneval','test_mbpp'):
        assert data[split]==old_data[split]
        assert all(torch.equal(v,old_inputs[split][k]) for k,v in inputs[split].items())
    torch.save(inputs,BASE/'source/inputs.pt');save(BASE/'source/rows.json',data)
    recipe=json.loads((BASE/'source/recipe.json').read_text());assert recipe['training_questions']==9149
    recipe['training_questions']=9148;save(BASE/'source/recipe.json',recipe)
    path=BASE/'source/batch_grpo_recipe.py';text=path.read_text();assert text.count('14973, 9151, 9149)')==2
    path.write_text(text.replace('14973, 9151, 9149)','14973, 9151, 9149, 9148)'))
    path=BASE/'test_training.py';text=path.read_text();assert text.count("'validated-kodcode-v3'")==1
    path.write_text(text.replace("'validated-kodcode-v3'","'validated-kodcode-v4'").replace('count==9149','count==9148'))
    audit=json.loads((BASE/'source/data-audit.json').read_text())
    audit.update(train=9148,inputs_sha256=sha(BASE/'source/inputs.pt'),rows_sha256=sha(BASE/'source/rows.json'),
        continuation_quarantine_sources=sorted(set(audit['continuation_quarantine_sources'])|{8633}))
    save(BASE/'source/data-audit.json',audit)
    save(BASE/'QUARANTINE.json',dict(user_authorized=True,removed_sources=[8633],removed_old_indices=[7883],
        kept_old_indices=kept,parent_count=9149,current_count=9148,train_probe_unchanged=True,benchmark_exclusions=0))
    elapsed=(BASE/'references.jsonl').stat().st_mtime-json.loads((BASE/'CPU_LAUNCHED-v2.json').read_text())['time']
    save(BASE/'REFERENCE_VALIDATED.json',dict(passed=True,training_count=9148,training_passed=9148,
        benchmarks=benchmark['benchmarks'],seconds=elapsed,
        reference_sha256=sha(BASE/'references.jsonl'),no_benchmark_exclusions=True,
        note='All9148 retained rows passed preserved full9149 audit; approved flaky row removed',excluded_sources=[8633]))
    print('QUARANTINE_8633_PASS_9148',flush=True)

if __name__=='__main__':main()
