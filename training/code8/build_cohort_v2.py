"""Versioned training-only quarantine based on complete repeated reference audits.

Static dataset transformation only: no dataset code is executed by this script.
Original rows, official scorers and full benchmarks are never edited.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

BASE = Path(__file__).resolve().parent
PARENT = Path('/data/horse/ws/arma647i-ldlm/egor-code-nfe8-i8026-20261009')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    audit = json.loads((PARENT/'diagnostics/COMPLETE.json').read_text())
    assert audit['completed'] and len(audit['repeats'])==3 and audit['reference_count']==9153
    parent = json.loads((PARENT/'cohort/MANIFEST.json').read_text())
    assert audit['original_manifest_sha256']==sha(PARENT/'cohort/MANIFEST.json')
    for name,h in parent['artifact_hashes'].items(): assert sha(PARENT/'cohort'/name)==h
    allow = [json.loads(s) for s in (PARENT/'cohort/allowlist.jsonl').read_text().splitlines()]
    table = pq.read_table(PARENT/'cohort/train.parquet')
    assert table.num_rows==len(allow)==9153
    failed = set(audit['failed_source_indices'])
    all_results = {}
    for repeat in audit['repeats']:
        path = PARENT/'diagnostics'/f"repeat-{repeat['repeat']}.jsonl"
        assert sha(path)==repeat['sha256']
        results = [json.loads(s) for s in path.read_text().splitlines()]
        assert len(results)==9153 and [r['source_index'] for r in results]==[a['source_index'] for a in allow]
        for a,r in zip(allow,results):
            assert a['task_id']==r['task_id']
            all_results.setdefault(r['source_index'],[]).append(r)
    # Include the demonstrated 8/24-failure task even if all three later repeats pass.
    row_repeats = json.loads((PARENT/'diagnostics/row-4886-repeats.json').read_text())
    assert len(row_repeats)==24 and any(not r['passed'] for r in row_repeats)
    failed.add(4886)
    assert all(i in all_results for i in failed) and 0<len(failed)<50
    out = BASE/'cohort'; assert not out.exists()
    out.mkdir()
    kept = [i for i,a in enumerate(allow) if a['source_index'] not in failed]
    pq.write_table(table.take(kept),out/'train.parquet',compression='zstd')
    retained = [allow[i] for i in kept]
    old_excluded = [json.loads(s) for s in (PARENT/'cohort/exclusions.jsonl').read_text().splitlines()]
    quarantined=[]
    for a in allow:
        if a['source_index'] in failed:
            b=dict(a,retained=False,reasons=['runtime_reference_unstable'],
                   runtime_repeats=all_results[a['source_index']])
            if a['source_index']==4886: b['additional_24_repeats']=row_repeats
            quarantined.append(b)
    for name,records in [('allowlist.jsonl',retained),('exclusions.jsonl',sorted(old_excluded+quarantined,key=lambda r:r['source_index']))]:
        (out/name).write_text(''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in records))
    manifest=dict(parent)
    counts=dict(parent['counts'],retained=len(retained),excluded=10000-len(retained))
    for key in ('reason_counts_nonexclusive','primary_reason_counts'):
        counts[key]=dict(parent['counts'][key],runtime_reference_unstable=len(failed))
    manifest.update(created_utc=datetime.now(timezone.utc).isoformat(),data_revision='validated-kodcode-v2',
        counts=counts,parent_manifest_sha256=sha(PARENT/'cohort/MANIFEST.json'),
        runtime_reaudit_sha256=sha(PARENT/'diagnostics/COMPLETE.json'),
        runtime_additional_quarantine=sorted(failed),
        runtime_packages_sha256=sha(PARENT/'runtime/packages.txt'),
        runtime_audit_python=audit['python'],
        policy=parent['policy']+' Additionally quarantine any previously retained reference failing any of three isolated runtime repeats or the 24-run instability probe. No reference/test repairs.',
        limitation=parent['limitation']+' Three passing repeats are a practical stability screen, not a proof of deterministic tests.',
        artifact_hashes={name:sha(out/name) for name in ('train.parquet','allowlist.jsonl','exclusions.jsonl')})
    (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    original_rows=table.to_pylist()
    excluded_sources=[dict(source_index=a['source_index'],task_id=a['task_id'],
        **{k:row[k] for k in ('question','solution','test')})
        for row,a in zip(original_rows,allow) if a['source_index'] in failed]
    (out/'runtime-compatibility-report.json').write_text(json.dumps(dict(retained=len(retained),
        additional_excluded=len(failed),source_indices=sorted(failed),rows=quarantined,
        static_excluded_sources=excluded_sources,
        benchmark_exclusions=0,scorer_changes=False,original_cohort_retained=True),indent=2))
    evidence=BASE/'compatibility-evidence'; evidence.mkdir()
    import shutil
    for p in (PARENT/'diagnostics').glob('*'):
        if p.is_file(): shutil.copy2(p,evidence/p.name)
    print('NEW_COHORT',len(retained),'QUARANTINE',sorted(failed),flush=True)

if __name__=='__main__': main()
