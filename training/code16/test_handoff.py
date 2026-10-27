"""CPU-only handoff and variable-NFE evaluation regressions."""
import ast
import datetime as dt
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('code16_handoff', BASE / 'wait_and_launch.py')
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


@pytest.mark.parametrize('state', sorted(handoff.FAILED))
def test_failed_parent_never_launches(state):
    with pytest.raises(RuntimeError):
        handoff.parent_state(f'{handoff.STEP}|{state}|1:0')


def test_only_successful_slurm_exit_is_accepted():
    assert handoff.parent_state(f'{handoff.STEP}|COMPLETED|0:0')
    assert not handoff.parent_state(f'{handoff.STEP}|RUNNING|0:0')
    for row in ('', f'{handoff.STEP}|COMPLETED|1:0', 'another.step|COMPLETED|0:0'):
        with pytest.raises(AssertionError):
            handoff.parent_state(row)


def test_unrelated_payload_blocks_handoff():
    assert handoff.active_payloads('2357570.batch\n2357570.extern') == []
    assert handoff.active_payloads('2357570.31\n2357570.batch') == ['2357570.31']
    assert handoff.active_payloads('2357570.99') == ['2357570.99']


def test_holder_identity_node_expiry():
    end = dt.datetime(2026, 10, 16, 10, 59, 51)
    detail = f'JobState=RUNNING NodeList=i8027 UserId=egsh587h(2224008) EndTime={end.isoformat()} '
    assert handoff.holder_remaining(detail, end.timestamp() - 3600) == 3000
    for old, new in [('i8027', 'i8005'), ('egsh587h', 'arma647i'), ('RUNNING', 'CANCELLED')]:
        with pytest.raises(AssertionError):
            handoff.holder_remaining(detail.replace(old, new))


def test_missing_or_short_completion_is_rejected(tmp_path):
    root = tmp_path / 'source/production'
    root.mkdir(parents=True)
    (root / 'COMPLETE.json').write_text(json.dumps(dict(accepted_groups=16384, optimizer_updates=4096)))
    with pytest.raises(AssertionError):
        handoff.validate_parent(tmp_path)


def test_eval_uses_recipe_nfe_and_preserves_padding(tmp_path):
    # Execute only the AST-extracted evaluate method with fake replicas/scorer;
    # no model imports, CUDA, external Python programs or sandbox execution.
    path = BASE / 'train_code.py'
    if not path.exists():
        path = BASE / 'template/train_code.py'
    tree = ast.parse(path.read_text())
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Trainer')
    method = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == 'evaluate')
    called = []
    summaries = []
    records = []

    class Replica:
        selector = SimpleNamespace(state_dict=lambda: {})

        def generate(self, inputs, ids, uniforms, record):
            assert len(ids) == 8 and len(uniforms) == 8
            assert all(nfe == 16 for _, nfe, _ in uniforms)
            return {'tokens': [[1] * 256 for _ in ids]}

    class Scorer:
        def begin_collection(self):
            pass

        def __call__(self, tokens, gold):
            assert len(tokens) == len(gold) == 2
            return [{'reward': 1}, {'reward': 0}]

    def draws(index, nfe, seed):
        called.append((index, nfe, seed))
        return index, nfe, seed

    def submit(fn, *args):
        result = fn(*args)
        return SimpleNamespace(result=lambda **kw: result)

    namespace = dict(json=json, torch=SimpleNamespace(stack=lambda values, dim: values),
        draws=draws, time=SimpleNamespace(monotonic=lambda: 1), state_digest=lambda _: 'policy',
        core=SimpleNamespace(append=lambda path, value: records.append(value)),
        save=lambda path, value: summaries.append(value), sha=lambda _: 'hash', BASE=BASE,
        TESTS=('test_humaneval', 'test_mbpp'))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    trainer = SimpleNamespace(out=tmp_path, completed=0, batch_groups=16, attempted=0,
        recipe=SimpleNamespace(nfe=16), behavior_hash='policy', scorer=Scorer(),
        replicas=[Replica()], selector=Replica.selector, inputs={'test_humaneval': {}},
        rows={'test_humaneval': [dict(task_id='a', gold={}), dict(task_id='b', gold={})]},
        pool=SimpleNamespace(submit=submit))
    namespace['evaluate'](trainer, 'test_humaneval', 101)
    assert len(called) == 8 and all(nfe == 16 for _, nfe, _ in called)
    assert [row['index'] for row in records] == [0, 1]
    assert summaries[0]['nfe'] == 16 and summaries[0]['correct'] == 1
    assert summaries[0]['count'] == 2


def test_validation_and_metadata_have_no_fixed_nfe8():
    path = BASE / 'train_code.py'
    if not path.exists():
        path = BASE / 'template/train_code.py'
    source = path.read_text()
    assert 'draws(i,self.recipe.nfe,seed)' in source
    assert 'draws(longest,trainer.recipe.nfe,101)' in source
    assert 'nfe=8' not in source and 'draws(i,8,seed)' not in source
    assert "assert not args.resume" in source
