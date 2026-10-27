"""One evaluation-only queue across math and code, using eight frozen replicas."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from decoders import decode, grid

BASE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod


def evaluators():
    modules = {}
    for kind in ('math', 'code'):
        mod = load(kind + '_audit', BASE / kind / 'base_evaluate.py')
        mod.grid = lambda kind=kind: grid(kind)
        modules[kind] = mod
    return modules


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def aggregate(modules):
    states = {kind: mod.aggregate() for kind, mod in modules.items()}
    report = dict(time=time.time(), complete=all(s['complete'] for s in states.values()),
                  cells=sum(len(s['cells']) for s in states.values()),
                  expected_cells=sum(s['expected_cells'] for s in states.values()),
                  answers=sum(sum(c['count'] for c in s['cells']) +
                              sum(c['count'] for c in s['partial']) for s in states.values()))
    save(BASE / 'STATUS.json', report)
    return report


def worker(rank, probe=False):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.device_count() == 1
    modules = evaluators()
    rows = {k: json.loads((BASE / k / 'shared/rows.json').read_text()) for k in modules}
    modules['code'].install_sandbox(BASE / 'code' / f'execution-retries-{rank}')
    scorers = {k: m.Scorer(BASE / k / 'shared', BASE / k / 'runtime',
                          BASE / k / f'scorer{rank}.log') for k, m in modules.items()}
    local = Path((BASE / 'math/runtime/local_root.txt').read_text().strip())
    port = load('llada_native_port', BASE / 'math/shared/model.py')
    engine = port.LLaDAOfflineModel.from_local_checkpoint(local / 'models/LLaDA-8B-Instruct',
        device=torch.device('cuda:0'), backend='vllm', fuse_projections=True)
    required = max(len(r['prompt_ids']) + 256 for split in rows['code'].values() for r in split)
    context = modules['code'].extend(engine, required, local / 'models/LLaDA-8B-Instruct')
    assert not any(p.requires_grad for p in engine.parameters())
    save(BASE / f'worker{rank}-ready.json', dict(time=time.time(), pid=os.getpid(),
        gpu=os.environ['CUDA_VISIBLE_DEVICES'], context=context, training=False))
    tasks = json.loads((BASE / 'TASKS.json').read_text())
    if probe:
        probe_configs = [dict(mode='p2_original', block=256, steps=4),
                         dict(mode='pc_sampler', block=256, steps=4),
                         dict(mode='margin', block=32, steps=8),
                         dict(mode='eb_sampler', block=32, gamma=.1),
                         dict(mode='klass', block=64, confidence=.6, kl=.01),
                         dict(mode='random', block=32, steps=8)]
        tasks = [dict(kind='math', cell=dict(c, split='test_gsm8k', seed=101,
                    name='probe-' + c['mode']), start=0, stop=1) for c in probe_configs]
        # Longest code prompt exercises the existing context extension and scorer.
        longest = max(range(len(rows['code']['test_mbpp'])),
                      key=lambda i: len(rows['code']['test_mbpp'][i]['prompt_ids']))
        tasks.append(dict(kind='code', cell=dict(mode='pc_sampler', block=256, steps=8,
                     split='test_mbpp', seed=101, name='probe-code'), start=longest, stop=longest + 1))
    started = time.monotonic()
    total = 0
    receipts = []
    try:
        for task_number in (range(len(tasks)) if probe else range(rank, len(tasks), 8)):
            task = tasks[task_number]
            kind, cell = task['kind'], task['cell']
            folder = BASE / ('probe' if probe else kind + '/results') / cell['name']
            path = folder / f"shard-{task['start']:04}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            records = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
            indices = list(range(task['start'], task['stop']))
            assert [r['index'] for r in records] == indices[:len(records)]
            if probe:
                receipts.extend(dict(kind=kind, cell=cell, index=r['index'], nfe=r['nfe'],
                                     seconds=r.get('generation_and_scoring_seconds', r['seconds']),
                                     reused_probe=True) for r in records)
            with path.open('a') as f:
                for index in indices[len(records):]:
                    row = rows[kind][cell['split']][index]
                    ids = row['prompt_ids']
                    scorer = scorers[kind]
                    assert scorer.tokenizer(row['prompt'], truncation=False)['input_ids'] == ids
                    prompt = torch.tensor([ids], device='cuda')
                    key = f"{cell['split']}:{index}:{cell['seed']}:{cell.get('steps', cell.get('threshold'))}"
                    torch.manual_seed(int(hashlib.sha256(key.encode()).hexdigest()[:15], 16))
                    adapter = modules[kind].NativeAdapter(engine, prompt)
                    begin = time.monotonic()
                    if cell['mode'] == 'random':
                        result = modules[kind].generate(adapter, prompt, cell)
                    else:
                        result = decode(adapter, prompt, cell)
                    tokens = result.sequences[0, -256:].tolist()
                    nfe = int(result.steps_taken[0])
                    assert nfe == adapter.calls and 126336 not in tokens and nfe <= 256
                    assert torch.equal(result.sequences[:, :len(ids)], prompt)
                    if 'steps' in cell:
                        assert nfe == cell['steps']
                    answer = scorer([tokens], [row['gold']])[0]
                    assert answer['task'] == row['task']
                    identity = dict(source_id=row['source_id']) if kind == 'math' else dict(task_id=row['task_id'])
                    elapsed = time.monotonic() - begin
                    record = dict(index=index, **identity, tokens=tokens,
                        nfe=nfe, **answer)
                    record['seconds' if kind == 'math' else 'generation_and_scoring_seconds'] = elapsed
                    f.write(json.dumps(record, allow_nan=False) + '\n')
                    f.flush()
                    os.fsync(f.fileno())
                    total += 1
                    save(BASE / f'worker{rank}-progress.json', dict(time=time.time(),
                         answers=total, task=task_number, kind=kind, cell=cell['name'], index=index,
                         elapsed=time.monotonic() - started, nfe=nfe,
                         seconds=elapsed, allocated_bytes=torch.cuda.memory_allocated(),
                         reserved_bytes=torch.cuda.memory_reserved()))
                    if probe:
                        receipts.append(dict(kind=kind, cell=cell, index=index, nfe=nfe,
                            seconds=elapsed, allocated_bytes=torch.cuda.memory_allocated(),
                            peak_bytes=torch.cuda.max_memory_allocated()))
                    del result, adapter, prompt
            print('SHARD_COMPLETE', task_number, kind, cell['name'], task['start'], task['stop'], flush=True)
    finally:
        for scorer in scorers.values():
            scorer.close()
    if probe:
        assert len(receipts) == 7
        save(BASE / 'GPU_PROBE.json', dict(time=time.time(), passed=True, tests=receipts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--aggregate', action='store_true')
    args = parser.parse_args()
    if args.worker is not None or args.probe:
        worker(args.worker or 0, args.probe)
        return
    modules = evaluators()
    if args.aggregate:
        aggregate(modules)
        return
    from test_decoders import run
    save(BASE / 'DECODER_TESTS.json', dict(time=time.time(), **run()))
    assert json.loads((BASE / 'GPU_PROBE.json').read_text())['passed']
    assert torch.cuda.device_count() == 8
    assert not (BASE / 'STARTED.json').exists()
    save(BASE / 'STARTED.json', dict(time=time.time(), step=os.environ['SLURM_STEP_ID'], training=False))
    uuids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    children = []
    try:
        for rank in range(8):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuids[rank])
            with (BASE / f'worker{rank}.log').open('a') as log:
                children.append(subprocess.Popen([sys.executable, '-B', __file__, '--worker', str(rank)],
                    stdout=log, stderr=subprocess.STDOUT, env=env))
        while any(p.poll() is None for p in children):
            failures = [(i, p.returncode) for i, p in enumerate(children) if p.poll() not in (None, 0)]
            if failures:
                raise RuntimeError(f'Worker failure: {failures}')
            aggregate(modules)
            time.sleep(30)
        assert all(p.returncode == 0 for p in children)
        assert aggregate(modules)['complete']
        save(BASE / 'COMPLETE.json', dict(time=time.time(), training=False))
    finally:
        for p in children:
            if p.poll() is None:
                p.terminate()
        for p in children:
            p.wait()


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save(BASE / f'FAILED-{os.getpid()}.json', dict(time=time.time(), error=repr(exc)))
        raise
