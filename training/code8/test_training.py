import ast
import json
from pathlib import Path

import pytest

from cohort_hashes import digest_row, sha
from validated_training import InfrastructureFailure, ReferenceCheckedReward, load_validated
from prepare_code import train_prompt_builder
from batch_grpo_recipe import Recipe

BASE = Path(__file__).resolve().parent


def test_cohort_and_protocol_immutable():
    rows,allow,manifest = load_validated(BASE/'cohort',BASE/'protocol')
    assert len(rows)==manifest['counts']['retained'] and all(r['source_index']!=2894 for r in allow)
    assert manifest['parent_manifest_sha256']=='719019c6989727d6f90cce83f4a7060c6045e93536bf0d253bbfa5ba1f7297cf'
    assert manifest['data_revision']=='validated-kodcode-v3'
    assert all(r['source_index'] not in manifest['runtime_additional_quarantine'] for r in allow)


def test_recipe_matches_math_optimization():
    recipe = Recipe(**json.loads((BASE/'source/recipe.json').read_text()))
    count=json.loads((BASE/'cohort/MANIFEST.json').read_text())['counts']['retained']
    assert recipe.initialization=='fresh' and recipe.nfe==8 and recipe.training_questions==count
    assert recipe.accepted_groups_per_batch==recipe.groups_per_minibatch==16
    assert recipe.total_batches==2048 and recipe.updates_per_batch==4 and recipe.passes==4
    assert recipe.learning_rate_at(0)==recipe.learning_rate_at(2047)==1e-4


def test_exact_official_train_prompt():
    class Tokenizer:
        def apply_chat_template(self,messages,add_generation_prompt,tokenize):
            assert add_generation_prompt and not tokenize
            return '<CHAT>'+messages[0]['content']+'<ASSISTANT>'
    row = dict(question='Double a value.',solution='unused',test='unused',
               test_info=[dict(function_declaration='def double(x):',function_name='double')])
    item = train_prompt_builder(Tokenizer())([row])[0]
    assert item['text']=='<CHAT>You are an expert Python programmer, and here is your task:\n\nDouble a value.\n\nImplement the function exactly as specified below.<ASSISTANT>Here is the completed function:\n```python\ndef double(x):\n'


def guard(run):
    row = dict(question_id='t',question='q',solution='s',test='assert f()==1',
               test_info=[dict(function_name='f')])
    a = dict(source_index=1,reference_passed=True,retained=True,reasons=[],row_sha256=digest_row(row))
    return ReferenceCheckedReward([row],[a],run,lambda p,n:p,lambda t:t)


def verdict(program):
    return dict(passed='assert False' not in program,
                result='failed: AssertionError' if 'assert False' in program else 'passed')


def test_health_required():
    g=guard(verdict)
    with pytest.raises(InfrastructureFailure): g.score_program(1,'code')
    g.begin_collection(); assert g.score_program(1,'code')['reward']==1.


def test_outer_timeout_never_becomes_zero_reward():
    def run(p): return dict(passed=False,result='outer timeout') if p=='code\nassert f()==1' else verdict(p)
    g=guard(run);g.begin_collection()
    with pytest.raises(InfrastructureFailure): g.score_program(1,'code')


def test_runtime_exception_aborts_collection():
    def run(p):
        if p.startswith('code'): raise RuntimeError('sandbox failed')
        return verdict(p)
    g=guard(run);g.begin_collection()
    with pytest.raises(InfrastructureFailure): g.score_program(1,'code')
    assert not g.healthy


def test_production_has_no_math_tests_or_warm_start():
    source=(BASE/'train_code.py').read_text()
    assert "TESTS = ('test_humaneval','test_mbpp')" in source
    assert 'test_gsm8k' not in source and 'test_math' not in source
    assert 'self.scorer.begin_collection()' in source
    assert 'return super().gather(max_attempts)' in source # No partial-batch replacement.
    ast.parse(source)
