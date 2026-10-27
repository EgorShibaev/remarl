"""Exact released code prompts, immutable training allowlist, complete benchmarks."""
import ast
import json
from pathlib import Path
import re
import runpy
import sys
import types

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

from cohort_hashes import sha
from validated_training import load_validated

BASE = Path(__file__).resolve().parent
PROTOCOL = BASE/'protocol'


def save(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False))


def train_prompt_builder(tokenizer):
    tree = ast.parse((PROTOCOL/'official/data_utils.py').read_text())
    nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef)
             and n.name in ('kodcode_preprocess_test','get_kodcode_questions')]
    class Rows:
        def __init__(self, rows): self.rows = rows
        def __contains__(self, key): return False
        def map(self, fn): return [fn(r) for r in self.rows]
    ns = {'re':re,'Dataset':object}
    def build(rows):
        ns['_load_dataset_with_fallback'] = lambda *a, **kw: Rows(rows)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'pinned_APPLe_kodcode','exec'),ns)
        entries = ns['get_kodcode_questions']()
        for e in entries:
            e['text'] = tokenizer.apply_chat_template(e['prompt'],add_generation_prompt=True,
                                                      tokenize=False)+e['gen_prefix']
        return entries
    return build


def benchmark_builders(tok, raw):
    fake = types.ModuleType('data.loaders.gsm8k'); fake.DATASETS_PATH = 'unused'
    sys.modules['data.loaders.gsm8k'] = fake
    fake_ds = types.ModuleType('datasets')
    fake_ds.load_from_disk = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('Mutable load forbidden'))
    sys.modules['datasets'] = fake_ds
    api = {n:runpy.run_path(str(PROTOCOL/'official'/f'{n}.py')) for n in ('humaneval','mbpp')}
    he = object.__new__(api['humaneval']['HumanEvalDataset']); he.tokenizer = tok
    mb = object.__new__(api['mbpp']['MBPPDataset']); mb.tokenizer = tok; mb.num_examples = 3
    mb.load_few_shot_examples = lambda: raw['mbpp-prompt'][:3]
    mb.create_few_shot_prompt()
    return he, mb


def main():
    out = BASE/'source'
    assert not (out/'inputs.pt').exists()
    training, allow, manifest = load_validated(BASE/'cohort',PROTOCOL)
    local = Path((BASE/'runtime/local_root.txt').read_text().strip())
    tok = AutoTokenizer.from_pretrained(local/'models/LLaDA-8B-Instruct',trust_remote_code=True,
                                       local_files_only=True)
    raw = {n:pq.read_table(PROTOCOL/'data'/f'{n}.parquet').to_pylist()
           for n in ('humaneval','mbpp-test','mbpp-prompt')}
    for name, digest in manifest['raw_hashes'].items():
        assert sha(PROTOCOL/'data'/name) == digest
    rows = {'train':[]}
    for i, (row, a, item) in enumerate(zip(training,allow,train_prompt_builder(tok)(training))):
        rows['train'].append(dict(index=i,source_index=a['source_index'],task='kodcode',
            task_id=row['question_id'],prompt=item['text'],prompt_ids=tok(item['text'],truncation=False)['input_ids'],
            gold=dict(task='kodcode',source_index=a['source_index'],raw_prompt=item['raw_prompt']+'\n'),
            function_name=item['function_name']))
    he, mb = benchmark_builders(tok, raw)
    for task in ('humaneval','mbpp'):
        split = 'test_'+task; rows[split] = []
        for i, item in enumerate(raw['humaneval' if task=='humaneval' else 'mbpp-test']):
            if task=='humaneval':
                prompt = he.create_prompt(item['prompt'])
                tests = item['test']+'\ncheck('+item['entry_point']+')\n'
                reference = item['prompt']+item['canonical_solution']
            else:
                prompt = mb.create_prompt(item['text'],item['test_list'])
                tests = '\n'.join(item['test_list']); reference = item['code']
            gold = dict(task=task,task_id=item['task_id'],index=i,tests=tests,reference=reference,
                        raw_prompt=item.get('prompt',''),entry_point=item.get('entry_point'))
            rows[split].append(dict(index=i,task=task,task_id=item['task_id'],prompt=prompt,
                prompt_ids=tok(prompt,truncation=False)['input_ids'],gold=gold))
    rows['train_probe'] = rows['train'][:128] # Fixed training-only probe, not validation/test.
    count = manifest['counts']['retained']
    assert [len(rows[s]) for s in ('train','test_humaneval','test_mbpp')] == [count,164,500]
    required = max(len(r['prompt_ids'])+256 for split in rows.values() for r in split)
    assert required <= 8192, ('Unusually long prompt: do not truncate or silently exclude',required)
    tensors = {}
    for split, entries in rows.items():
        maximum = max(len(r['prompt_ids'])+256 for r in entries)
        ids = torch.full((len(entries),maximum),tok.pad_token_id,dtype=torch.long)
        mask = torch.zeros_like(ids); positions = torch.empty((len(entries),256),dtype=torch.long)
        for i,r in enumerate(entries):
            n = len(r['prompt_ids']); ids[i,:n] = torch.tensor(r['prompt_ids'])
            ids[i,n:n+256] = 126336; mask[i,:n+256] = 1; positions[i] = torch.arange(n,n+256)
        tensors[split] = dict(ids=ids,mask=mask,positions=positions)
    torch.save(tensors,out/'inputs.pt');save(out/'rows.json',rows)
    save(out/'data-audit.json',dict(train=count,inputs_sha256=sha(out/'inputs.pt'),
         rows_sha256=sha(out/'rows.json'),required_canvas=required,
         cohort_manifest_sha256=sha(BASE/'cohort/MANIFEST.json'),raw_hashes=manifest['raw_hashes'],
         exact_APPLe_train_and_test_prompts=True,no_prompt_truncation=True,
         benchmark_counts={'humaneval':164,'mbpp':500},probe_is_training_only=True))
    print('CODE_DATA_READY',count,'canvas',required,flush=True)


if __name__=='__main__': main()
