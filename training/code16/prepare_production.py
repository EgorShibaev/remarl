"""Bind proven replacement protocol and fork output history; never edit the parent."""
import hashlib
import json
from pathlib import Path
import shutil

BASE=Path(__file__).resolve().parent
PARENT=Path('/data/horse/ws/egsh587h-ldlm/remasking/code-nfe16-i8027-20261011')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def main():
    assert BASE.name=='code16-seccomp-i8027-20261013'
    assert json.loads((BASE/'REFERENCE_VALIDATED.json').read_text())['passed']
    assert json.loads((BASE/'SECURITY_VALIDATED.json').read_text())['passed']
    assert not (BASE/'PRODUCTION_PREPARED.json').exists()
    assert sha(PARENT/'source/production/latest.pt')==sha(BASE/'parent-latest-021568.pt')
    assert not (BASE/'source/production').exists()
    shutil.copyfile(PARENT/'PRODUCTION_STARTED.json',BASE/'PRODUCTION_STARTED.json')
    quarantine=json.loads((BASE/'QUARANTINE.json').read_text())
    assert quarantine['removed_sources']==[8633] and quarantine['current_count']==9148
    # The one approved training-row removal is preserved separately from protocol binding.
    cohort=json.loads((BASE/'cohort/MANIFEST.json').read_text())
    assert cohort['counts']['retained']==9148
    for name in ('sandbox.py','build_seccomp.py','security_probe.py','network-deny.bpf','network-deny.json'):
        cohort['protocol_hashes'][name]=sha(BASE/'protocol'/name)
    save(BASE/'cohort/MANIFEST.json',cohort)
    audit=json.loads((BASE/'source/data-audit.json').read_text())
    audit['cohort_manifest_sha256']=sha(BASE/'cohort/MANIFEST.json')
    save(BASE/'source/data-audit.json',audit)
    names=[line.split('  ',1)[1] for line in (BASE/'source/source.sha256').read_text().splitlines()]
    (BASE/'source/source.sha256').write_text(''.join(sha(BASE/'source'/name)+'  '+name+'\n' for name in names))
    files=[line.split('  ',1)[1] for line in (PARENT/'training-source.sha256').read_text().splitlines()]
    files+=['protocol/build_seccomp.py','protocol/security_probe.py','protocol/network-deny.bpf',
        'protocol/network-deny.json','protocol/network-deny.pfc',
        'validate_cpu.py','prepare_production.py','resume.py','run_production.sh','run_cpu.sh',
        'quarantine.py','migration.py','test_migration.py','test_sandbox.py','QUARANTINE.json']
    (BASE/'training-source.sha256').write_text(''.join(sha(BASE/name)+'  '+name+'\n' for name in sorted(set(files))))
    reference=json.loads((BASE/'REFERENCE_VALIDATED.json').read_text())
    old_runtime=json.loads((PARENT/'RUNTIME_VALIDATED.json').read_text())
    old_runtime.update(dataset_manifest_sha256=sha(BASE/'cohort/MANIFEST.json'),
        retained_references=9148,quarantined_source=8633,
        reference_repeats=1, reference_audit_sha256=sha(BASE/'REFERENCE_VALIDATED.json'),
        benchmark_reference_audits=reference['benchmarks'],seconds=reference['seconds'],
        source_manifest_sha256=sha(BASE/'training-source.sha256'),
        sandbox_validation_sha256=sha(BASE/'SECURITY_VALIDATED.json'),
        seccomp_reference_audit_sha256=sha(BASE/'REFERENCE_VALIDATED.json'),
        network_namespace=False,network_syscall_denial=True,
        parent_runtime_validation_sha256=sha(PARENT/'RUNTIME_VALIDATED.json'))
    save(BASE/'RUNTIME_VALIDATED.json',old_runtime)
    # Full independent copies: no hardlinks that could alter historical results.
    shutil.copytree(PARENT/'source/production',BASE/'source/production')
    archive=BASE/'parent-provenance/interrupted-production';archive.mkdir()
    orphan=BASE/'source/production/batches/001349'
    if orphan.exists():
        assert not (orphan/'complete.json').exists();orphan.rename(archive/'batch-001349')
    for p in (BASE/'source/production').glob('failure-*.json'):p.rename(archive/p.name)
    for name in ('protocol/sandbox_runner.py','runtime/packages.txt'):
        assert sha(BASE/name)==sha(PARENT/name),name
    for p in (BASE/'protocol/official').iterdir():
        if p.is_file():assert sha(p)==sha(PARENT/'protocol/official'/p.name)
    save(BASE/'PRODUCTION_PREPARED.json',dict(passed=True,parent_checkpoint_sha256=sha(BASE/'parent-latest-021568.pt'),
        source_manifest_sha256=sha(BASE/'training-source.sha256'),
        original_outputs_unchanged=True,independent_history_copy=True,
        optimization_prompts_scorers_benchmark_tests_unchanged=True,quarantined_sources=[8633],training_count=9148,
        functional_changes=['Seccomp network syscall denial replaces network namespace','Approved one-row quarantine']))
    print('PRODUCTION_PREPARED',flush=True)

if __name__=='__main__':main()
