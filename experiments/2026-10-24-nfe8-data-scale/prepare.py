"""Build nested, balanced train-only subsets; preserve test tensors exactly."""
import hashlib
import json
from pathlib import Path
import random
import torch

BASE = Path(__file__).resolve().parent
SEEDS = (7, 17, 27)
SIZES = (4, 32, 256, 2048)


def indices_for(rows, seed, size):
    selected = []
    for offset, task in enumerate(('gsm8k', 'math')):
        indices = [i for i, row in enumerate(rows) if row['task'] == task]
        random.Random(seed * 100 + offset).shuffle(indices)
        selected.extend(indices[:size])
    return selected


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = BASE / 'shared'
    rows = json.loads((source / 'rows.json').read_text())
    inputs = torch.load(source / 'inputs.pt', map_location='cpu', weights_only=True)
    original_audit = json.loads((source / 'data-audit.json').read_text())
    assert len(rows['train']) == 14973
    assert sha(source / 'inputs.pt') == original_audit['inputs_sha256']
    assert sha(source / 'rows.json') == original_audit['rows_sha256']
    manifest = {}
    for seed in SEEDS:
        previous = set()
        for size in SIZES:
            name = f'n{size*2}-seed{seed}'
            root = BASE / 'runs' / name / 'source'
            assert root.is_dir() and not (root / 'inputs.pt').exists()
            indices = indices_for(rows['train'], seed, size)
            assert len(indices) == len(set(indices)) == size*2
            assert previous <= set(indices)
            previous = set(indices)
            chosen_rows = [rows['train'][i] for i in indices]
            assert all(sum(r['task'] == task for r in chosen_rows) == size for task in ('gsm8k', 'math'))
            subset_rows = dict(rows, train=chosen_rows)
            probe = list(range(min(size, 32))) + list(range(size, size + min(size, 32)))
            subset_rows['train_probe'] = [chosen_rows[i] for i in probe]
            subset_inputs = dict(inputs)
            subset_inputs['train'] = {k: v[indices].clone() for k,v in inputs['train'].items()}
            subset_inputs['train_probe'] = {k: v[probe].clone() for k,v in subset_inputs['train'].items()}
            for split in ('test_gsm8k', 'test_math'):
                assert subset_rows[split] == rows[split]
                assert all(torch.equal(subset_inputs[split][k], v) for k,v in inputs[split].items())
            torch.save(subset_inputs, root / 'inputs.pt')
            (root / 'rows.json').write_text(json.dumps(subset_rows) + '\n')
            audit = dict(original_audit, train=size*2, train_task_counts={'gsm8k':size, 'math':size},
                         inputs_sha256=sha(root / 'inputs.pt'), rows_sha256=sha(root / 'rows.json'),
                         parent_inputs_sha256=original_audit['inputs_sha256'],
                         parent_rows_sha256=original_audit['rows_sha256'])
            (root / 'data-audit.json').write_text(json.dumps(audit, indent=2)+'\n')
            membership = dict(seed=seed, per_dataset=size, total=size*2, parent_indices=indices,
                              source_ids=[r['source_id'] for r in chosen_rows],
                              selection='seeded permutation per dataset, nested prefixes, no reward filtering')
            (root / 'subset.json').write_text(json.dumps(membership, indent=2)+'\n')
            manifest[name] = membership
            files = [p for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'source.sha256']
            (root / 'source.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(root)}\n' for p in files))
    (BASE / 'SUBSETS_READY.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('PREPARED', len(manifest), 'nested balanced subsets', flush=True)


if __name__ == '__main__':
    main()
