"""Versioned quarantine; no reference/generated code is executed by this builder."""
import hashlib
import json
from pathlib import Path
import shutil
import torch
import pyarrow.parquet as pq

BASE=Path(__file__).resolve().parent
AUDIT_ROOT=Path('/data/horse/ws/egsh587h-ldlm/remasking/code-nfe8-i8027-20261010')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def main():
    assert BASE.name=='code-nfe8-i8027-20261010-v2' and not (BASE/'RECOVERY_STAGED.json').exists()
    audit=json.loads((AUDIT_ROOT/'quarantine-audit/COMPLETE.json').read_text())
    assert audit['completed'] and audit['reference_count']==9151 and len(audit['repeats'])==3
    assert audit['original_manifest_sha256']==sha(AUDIT_ROOT/'cohort/MANIFEST.json')
    old_metadata=json.loads((AUDIT_ROOT/'RECOVERY_STAGED.json').read_text())['parent_metadata']
    assert sha(BASE/'cohort/MANIFEST.json')==old_metadata['cohort_manifest_sha256']
    evidence=BASE/'recovery-evidence';evidence.mkdir()
    shutil.copytree(AUDIT_ROOT/'quarantine-audit',evidence/'quarantine-audit')
    shutil.copyfile(AUDIT_ROOT/'REFERENCE_DIAGNOSTIC.json',evidence/'REFERENCE_DIAGNOSTIC.json')
    for name in ('training-source.sha256','RUNTIME_VALIDATED.json','CPU_PREFLIGHT.json'):
        shutil.copyfile(BASE/name,evidence/('parent-'+name))
    shutil.copyfile(BASE/'source/production/latest.pt',BASE/'recovery-parent.pt')
    assert sha(BASE/'recovery-parent.pt')=='74aa36c2f38a3026af6352af6e2717f3096f57c7eed748e453c1d9305d28f2c5'
    for name in ('sandbox.py','sandbox_runner.py'):
        shutil.copyfile(AUDIT_ROOT/'protocol'/name,BASE/'protocol'/name)
    old_cohort=BASE/'cohort';old_cohort.rename(evidence/'parent-cohort')
    old_cohort=evidence/'parent-cohort';cohort=BASE/'cohort';cohort.mkdir()
    manifest=json.loads((old_cohort/'MANIFEST.json').read_text())
    for name,value in manifest['artifact_hashes'].items():assert sha(old_cohort/name)==value
    allow=[json.loads(s) for s in (old_cohort/'allowlist.jsonl').read_text().splitlines()]
    failed=set(audit['failed_source_indices']);assert 6395 in failed and 0<len(failed)<100
    kept=[i for i,a in enumerate(allow) if a['source_index'] not in failed]
    expected=[a['source_index'] for a in allow]
    for receipt in audit['repeats']:
        path=evidence/'quarantine-audit'/f"repeat-{receipt['repeat']}.jsonl"
        assert sha(path)==receipt['sha256']
        results=[json.loads(s) for s in path.read_text().splitlines()]
        assert [r['source_index'] for r in results]==expected
        assert all(results[i]['passed'] for i in kept)
    risk_path=evidence/'quarantine-audit/set-order-32-repeats.jsonl'
    assert sha(risk_path)==audit['risk_log_sha256']
    risk_results=[json.loads(s) for s in risk_path.read_text().splitlines()]
    assert len(risk_results)==audit['risk_count']*32
    assert all(r['passed'] for r in risk_results if r['source_index'] not in failed)
    table=pq.read_table(old_cohort/'train.parquet');assert table.num_rows==len(allow)==9151
    pq.write_table(table.take(kept),cohort/'train.parquet',compression='zstd')
    retained=[allow[i] for i in kept]
    exclusions=[json.loads(s) for s in (old_cohort/'exclusions.jsonl').read_text().splitlines()]
    exclusions += [dict(a,retained=False,reasons=['runtime_reference_unstable'],
        continuation_quarantine=True) for a in allow if a['source_index'] in failed]
    for name,rows in [('allowlist.jsonl',retained),('exclusions.jsonl',sorted(exclusions,key=lambda x:x['source_index']))]:
        (cohort/name).write_text(''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows))
    report=dict(retained=len(kept),additional_excluded_sources=sorted(failed),
        full_repeats=3,set_risk_count=audit['risk_count'],set_risk_repeats=32,
        benchmark_exclusions=0,scorer_changes=False,original_cohort_preserved=True)
    save(cohort/'runtime-compatibility-report.json',report)
    manifest.update(data_revision='validated-kodcode-v3',
        continuation_parent_manifest_sha256=old_metadata['cohort_manifest_sha256'],
        continuation_audit_sha256=sha(evidence/'quarantine-audit/COMPLETE.json'),
        continuation_additional_quarantine=sorted(failed),
        runtime_additional_quarantine=sorted(set(manifest['runtime_additional_quarantine'])|failed),
        counts=dict(manifest['counts'],retained=len(kept),excluded=10000-len(kept)),
        artifact_hashes={n:sha(cohort/n) for n in ('train.parquet','allowlist.jsonl','exclusions.jsonl')})
    for k in ('reason_counts_nonexclusive','primary_reason_counts'):
        manifest['counts'][k]=dict(manifest['counts'][k])
        manifest['counts'][k]['runtime_reference_unstable']=len(manifest['runtime_additional_quarantine'])
    for name in ('sandbox.py','sandbox_runner.py'):manifest['protocol_hashes'][name]=sha(BASE/'protocol'/name)
    save(cohort/'MANIFEST.json',manifest)
    source=BASE/'source'
    data=json.loads((source/'rows.json').read_text());inputs=torch.load(source/'inputs.pt',weights_only=True,map_location='cpu')
    for name in ('rows.json','inputs.pt','data-audit.json','recipe.json'):
        shutil.copyfile(source/name,evidence/('parent-'+name))
    old_train=data['train'];data['train']=[dict(old_train[i],index=j) for j,i in enumerate(kept)]
    inputs['train']={k:v[kept].clone() for k,v in inputs['train'].items()}
    probe_sources={r['source_index'] for r in data['train_probe']}
    probe_changed=bool(probe_sources & failed)
    if probe_changed:
        data['train_probe']=data['train'][:128]
        inputs['train_probe']={k:v[:128].clone() for k,v in inputs['train'].items()}
    assert all(a['source_index']==b['source_index'] for a,b in zip(data['train'],retained))
    original=json.loads((evidence/'parent-rows.json').read_text())
    old_inputs=torch.load(evidence/'parent-inputs.pt',weights_only=True,map_location='cpu')
    for split in ('test_humaneval','test_mbpp'):
        assert data[split]==original[split]
        assert all(torch.equal(v,old_inputs[split][k]) for k,v in inputs[split].items())
    torch.save(inputs,source/'inputs.pt');save(source/'rows.json',data)
    recipe=json.loads((source/'recipe.json').read_text());recipe['training_questions']=len(kept)
    save(source/'recipe.json',recipe)
    # Add only the exact validated size, retaining all existing controls.
    path=source/'batch_grpo_recipe.py';text=path.read_text()
    assert text.count('14973, 9151)')==2
    path.write_text(text.replace('14973, 9151)',f'14973, 9151, {len(kept)})'))
    path=BASE/'test_training.py';text=path.read_text()
    assert text.count("'validated-kodcode-v2'")==1
    path.write_text(text.replace("'validated-kodcode-v2'","'validated-kodcode-v3'"))
    data_audit=json.loads((source/'data-audit.json').read_text())
    data_audit.update(train=len(kept),inputs_sha256=sha(source/'inputs.pt'),rows_sha256=sha(source/'rows.json'),
        cohort_manifest_sha256=sha(cohort/'MANIFEST.json'),continuation_quarantine_sources=sorted(failed),
        train_probe_changed=probe_changed,benchmark_prompts_tokens_tests_unchanged=True)
    save(source/'data-audit.json',data_audit)
    for name in ('RUNTIME_VALIDATED.json','CPU_PREFLIGHT.json'):(BASE/name).rename(evidence/('copied-'+name))
    for name in ('local_root.txt','READY','packages.txt'):(BASE/'runtime'/name).rename(evidence/('parent-runtime-'+name))
    for name in ('local_root.txt','READY','packages.txt'):shutil.copyfile(AUDIT_ROOT/'runtime'/name,BASE/'runtime'/name)
    assert sha(BASE/'runtime/packages.txt')==audit['packages_sha256']
    source_files=[s.split('  ',1)[1] for s in (source/'source.sha256').read_text().splitlines()]
    (source/'source.sha256').write_text(''.join(sha(source/n)+'  '+n+'\n' for n in source_files))
    files=[s.split('  ',1)[1] for s in (BASE/'training-source.sha256').read_text().splitlines()]
    files+=['build_successor.py','migration.py','recover.py','run_recovery.sh','bind_audit.py','test_recovery.py']
    (BASE/'training-source.sha256').write_text(''.join(sha(BASE/n)+'  '+n+'\n' for n in sorted(set(files))))
    save(BASE/'RECOVERY_STAGED.json',dict(parent_metadata=old_metadata,kept_old_indices=kept,
        checkpoint_sha256=sha(BASE/'recovery-parent.pt'),parent_step='2356156.13',
        training_manifest_sha256=sha(BASE/'training-source.sha256'),quarantine=report,
        benchmark_prompts_tokens_tests_unchanged=True,train_probe_changed=probe_changed))
    print('QUARANTINE_STAGED',len(kept),sorted(failed),flush=True)

if __name__=='__main__':main()
