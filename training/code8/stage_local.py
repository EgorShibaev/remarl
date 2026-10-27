"""Prepare isolated code training; never mutate live or immutable parents."""
import hashlib
import json
from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parent
PARENT = BASE.parent/'2026-10-09-code-domain'
CLEAN = BASE.parent/'2026-10-09-code-cleanup'
RUNTIME = BASE.parent/'2026-10-07-apple-mixture-native'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert not (BASE/'protocol').exists()
assert sha(CLEAN/'validated-kodcode-v1/MANIFEST.json') == '719019c6989727d6f90cce83f4a7060c6045e93536bf0d253bbfa5ba1f7297cf'
(BASE/'protocol').mkdir()
for name in ('prepare.py','audit_references.py','sandbox.py','sandbox_runner.py'):
    shutil.copy2(PARENT/name, BASE/'protocol'/name)
shutil.copytree(PARENT/'official', BASE/'protocol/official')
shutil.copytree(PARENT/'data', BASE/'protocol/data')
shutil.copytree(CLEAN/'validated-kodcode-v1', BASE/'cohort')
shutil.copy2(CLEAN/'validated_training.py', BASE/'validated_training.py')
(BASE/'runtime').mkdir()
for name in ('uv','checkpoint_manifest.json','sota-selector.pt','setup_assets.py'):
    shutil.copy2(RUNTIME/name, BASE/'runtime'/name)
shutil.copy2(BASE/'source/source.sha256', BASE/'parent-source.sha256')
recipe = dict(initialization='fresh',training_questions=9151,accepted_groups_per_batch=16,
    groups_per_minibatch=16,passes=4,target_accepted_groups=32768,lr_schedule='constant',
    learning_rate=1e-4,minimum_lr=1e-4,policy_width=None,nfe=8)
(BASE/'source/recipe.json').write_text(json.dumps(recipe,indent=2))
(BASE/'LAUNCH_PLAN.json').write_text(json.dumps(dict(node='i8026',owner='arma647i',
    holder=2356156,name='kill_it_when_node_needed',gpus=8,cpus=32,
    remote='/data/horse/ws/arma647i-ldlm/egor-code-nfe8-i8026-20261009-v2',
    recipe=recipe,benchmarks={'humaneval':164,'mbpp':500},inference_seeds=[101,202,303],
    dataset_manifest_sha256=sha(BASE/'cohort/MANIFEST.json'),
    parent_source_manifest_sha256=sha(BASE/'parent-source.sha256'),
    no_home_artifacts=True,fresh_code_policy=True),indent=2))
