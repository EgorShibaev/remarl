"""Fail-closed loader and reward adapter for the future code trainer.

This is NOT connected to any live training/evaluation. Candidate programs must
only be executed through the previously validated, unchanged isolated sandbox.
"""
import json
from pathlib import Path

import pyarrow.parquet as pq

from cohort_hashes import digest_row, sha


class InfrastructureFailure(RuntimeError):
    pass


def load_validated(root, protocol_root):
    root, protocol_root = Path(root), Path(protocol_root)
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    if manifest.get('version') != 1:
        raise InfrastructureFailure('Unknown manifest version')
    for name, expected in manifest['artifact_hashes'].items():
        if sha(root / name) != expected:
            raise InfrastructureFailure(f'Dataset artifact changed: {name}')
    for name, expected in manifest['protocol_hashes'].items():
        if sha(protocol_root / name) != expected:
            raise InfrastructureFailure(f'Audited reward protocol changed: {name}')
    allow = [json.loads(line) for line in (root / 'allowlist.jsonl').read_text().splitlines()]
    rows = pq.read_table(root / 'train.parquet').to_pylist()
    if len(rows) != len(allow) or len(rows) != manifest['counts']['retained']:
        raise InfrastructureFailure('Dataset/allowlist count mismatch')
    if len({a['source_index'] for a in allow}) != len(allow):
        raise InfrastructureFailure('Duplicate allowlist source index')
    for row, record in zip(rows, allow):
        if (record['task_id'] != row['question_id'] or record['row_sha256'] != digest_row(row)
                or not record['reference_passed'] or not record['retained'] or record['reasons']):
            raise InfrastructureFailure('Unvalidated row in training dataset')
    return rows, allow, manifest


class ReferenceCheckedReward:
    """Caller supplies exact audited sanitize/preprocess/sandbox functions.

    Begin every collection with begin_collection(). Any infrastructure exception
    propagates: callers must discard/abort the collection, not substitute zeros.
    score_program expects complete candidate source, including its declaration.
    Future training integration must verify the dependency/runtime/sandbox gates
    on the actual compute node before constructing this adapter.
    """
    def __init__(self, rows, allowlist, run_sandbox, sanitize, preprocess):
        if len(rows) != len(allowlist):
            raise InfrastructureFailure('Row/allowlist length mismatch')
        self.rows = {}
        for row, record in zip(rows, allowlist):
            if (not record['reference_passed'] or not record['retained'] or record['reasons']
                    or record['row_sha256'] != digest_row(row)):
                raise InfrastructureFailure('Unvalidated reward row')
            self.rows[record['source_index']] = row
        self.run = run_sandbox
        self.sanitize = sanitize
        self.preprocess = preprocess
        self.healthy = False

    def begin_collection(self):
        self.healthy = False
        correct = self.run('assert 2 + 2 == 4')
        wrong = self.run('assert False')
        if correct.get('passed') is not True or correct.get('result') != 'passed':
            raise InfrastructureFailure('Execution health probe failed')
        if wrong.get('passed') is not False or not wrong.get('result', '').startswith('failed:'):
            raise InfrastructureFailure('Incorrect-program health probe failed')
        self.healthy = True

    def score_program(self, source_index, program):
        if not self.healthy:
            raise InfrastructureFailure('No successful collection health gate')
        if source_index not in self.rows:
            raise InfrastructureFailure('Task not present in validated allowlist')
        row = self.rows[source_index]
        code = self.sanitize(program, row['test_info'][0]['function_name'])
        try:
            result = self.run(code + '\n' + self.preprocess(row['test']))
        except Exception as exc:
            self.healthy = False
            raise InfrastructureFailure('Sandbox execution failed; abort collection') from exc
        # An outer watchdog firing is not the official program timeout verdict.
        if type(result.get('passed')) is not bool or not isinstance(result.get('result'), str):
            self.healthy = False
            raise InfrastructureFailure('Malformed sandbox verdict')
        verdict = result['result']
        if not (verdict == 'passed' or verdict == 'timed out' or verdict.startswith('failed:')):
            self.healthy = False
            raise InfrastructureFailure(f'Non-program sandbox verdict: {verdict}')
        if result['passed'] != (verdict == 'passed'):
            self.healthy = False
            raise InfrastructureFailure('Conflicting sandbox verdict')
        return dict(reward=float(result['passed']), source_index=source_index, **result)
