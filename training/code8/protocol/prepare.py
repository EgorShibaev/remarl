"""Pin full code splits; prove prompts against released APPLe loader functions."""
import ast
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import runpy
import sys
import types

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False))
def sanitizer():
    spec=importlib.util.spec_from_file_location('apple_sanitize',BASE/'official/sanitize.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m
def preprocess():
    tree=ast.parse((BASE/'official/data_utils.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='kodcode_preprocess_test')
    namespace={'re':re}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'official_preprocess','exec'),namespace)
    return namespace[node.name]
def load_raw():
    return {name:pq.read_table(BASE/'data'/f'{name}.parquet').to_pylist()
            for name in ('kodcode','humaneval','mbpp-test','mbpp-prompt')}
def main():
    assert not (BASE/'inputs.pt').exists()
    raw=load_raw()
    assert [len(raw[n]) for n in raw]==[10000,164,500,10]
    local=Path((BASE/'runtime/local_root.txt').read_text().strip())
    manifest=json.loads((BASE/'runtime/checkpoint_manifest.json').read_text())
    for item in manifest['files']:
        assert sha(local/item['local_path'])==item['sha256']
    tok=AutoTokenizer.from_pretrained(local/'models/LLaDA-8B-Instruct',trust_remote_code=True,
                                      local_files_only=True)
    # Import exact upstream classes without loading any remote or mutable disk dataset.
    fake=types.ModuleType('data.loaders.gsm8k');fake.DATASETS_PATH='unused'
    sys.modules['data.loaders.gsm8k']=fake
    # The pinned runtime intentionally has no HF datasets package. Only the
    # released prompt functions are used: parquet rows were loaded above.
    # Fail if upstream ever tries to load a mutable dataset instead.
    fake_datasets=types.ModuleType('datasets')
    def forbidden_load(*args,**kwargs):raise AssertionError('Unpinned dataset load forbidden')
    fake_datasets.load_from_disk=forbidden_load
    sys.modules['datasets']=fake_datasets
    api={n:runpy.run_path(str(BASE/'official'/f'{n}.py')) for n in ('humaneval','mbpp')}
    he=object.__new__(api['humaneval']['HumanEvalDataset']);he.tokenizer=tok
    mb=object.__new__(api['mbpp']['MBPPDataset']);mb.tokenizer=tok;mb.num_examples=3
    mb.load_few_shot_examples=lambda:raw['mbpp-prompt'][:3]
    mb.create_few_shot_prompt()
    rows={}
    for split in ('humaneval','mbpp'):
        entries=raw['humaneval' if split=='humaneval' else 'mbpp-test']
        rows[split]=[]
        for i,item in enumerate(entries):
            if split=='humaneval':
                text=he.create_prompt(item['prompt'])
                tests=item['test']+'\ncheck('+item['entry_point']+')\n'
                reference=item['prompt']+item['canonical_solution']
            else:
                text=mb.create_prompt(item['text'],item['test_list'])
                tests='\n'.join(item['test_list']);reference=item['code']
            ids=tok(text,truncation=False)['input_ids']
            assert len(ids)+256<=8192,(split,i,len(ids),'unreasonably long prompt; stop rather than truncate')
            rows[split].append(dict(index=i,task_id=item['task_id'],prompt=text,prompt_ids=ids,
                tests=tests,reference=reference,raw_prompt=item.get('prompt',''),
                entry_point=item.get('entry_point'),task=split))
        assert len({r['task_id'] for r in rows[split]})==len(entries)
    tensors={}
    for split,entries in rows.items():
        maximum=max(len(r['prompt_ids'])+256 for r in entries)
        ids=torch.full((len(entries),maximum),tok.pad_token_id,dtype=torch.long)
        mask=torch.zeros_like(ids);positions=torch.empty((len(entries),256),dtype=torch.long)
        for i,row in enumerate(entries):
            p=row['prompt_ids'];size=len(p);ids[i,:size]=torch.tensor(p)
            ids[i,size:size+256]=126336;mask[i,:size+256]=1;positions[i]=torch.arange(size,size+256)
        tensors[split]=dict(ids=ids,mask=mask,positions=positions)
    torch.save(tensors,BASE/'inputs.pt');save(BASE/'rows.json',rows)
    # Static audit full training set, never execute tests or solution here.
    processed=preprocess();issues=[];test_nodes=0;syntax_errors=[]
    for i,row in enumerate(raw['kodcode']):
        try:tree=ast.parse(row['test'])
        except SyntaxError as exc:
            syntax_errors.append(dict(index=i,error=str(exc)));continue
        fs=[n for n in tree.body if isinstance(n,ast.FunctionDef)
                                      and n.name.startswith('test_')]
        test_nodes+=len(fs)
        required=[n.name for n in fs if len(n.args.posonlyargs+n.args.args)>len(n.args.defaults)
                  or any(d is None for d in n.args.kw_defaults)]
        if required:issues.append(dict(index=i,task_id=row.get('question_id',row.get('id',i)),functions=required))
        assert isinstance(processed(row['test']),str)
    bench_text={re.sub(r'\s+',' ',r['prompt']).strip() for r in raw['humaneval']}
    bench_text|={re.sub(r'\s+',' ',r['text']).strip() for r in raw['mbpp-test']}
    overlaps=[i for i,r in enumerate(raw['kodcode']) if re.sub(r'\s+',' ',r['question']).strip() in bench_text]
    benchmark_codes={re.sub(r'\s+',' ',r['reference']).strip(): (s,r['task_id'])
                     for s,entries in rows.items() for r in entries}
    code_overlaps=[dict(index=i,benchmark=benchmark_codes[re.sub(r'\s+',' ',r['solution']).strip()])
                   for i,r in enumerate(raw['kodcode'])
                   if re.sub(r'\s+',' ',r['solution']).strip() in benchmark_codes]
    # Literal near-duplicate candidate screening, not a claim of semantic decontamination.
    def grams(text):
        words=re.findall(r'\w+',text.lower())
        return {tuple(words[j:j+5]) for j in range(max(0,len(words)-4))}
    benchmarks=[('humaneval',r['task_id'],r['prompt']) for r in raw['humaneval']]
    benchmarks += [('mbpp',r['task_id'],r['text']) for r in raw['mbpp-test']]
    sets=[grams(r[2]) for r in benchmarks];index=defaultdict(set)
    for j,g in enumerate(sets):
        for token in g:index[token].add(j)
    near=[]
    for i,row in enumerate(raw['kodcode']):
        g=grams(row['question']);counts=Counter(j for token in g for j in index.get(token,()))
        for j,common in counts.items():
            denominator=min(len(g),len(sets[j]))
            if common>=5 and denominator and common/denominator>=0.8:
                near.append(dict(train_index=i,benchmark=benchmarks[j][:2],
                                 shared_5grams=common,containment=common/denominator))
    save(BASE/'data-audit.json',dict(counts={n:len(v) for n,v in raw.items()},
        raw_hashes={p.name:sha(p) for p in (BASE/'data').glob('*.parquet')},
        input_sha256=sha(BASE/'inputs.pt'),rows_sha256=sha(BASE/'rows.json'),
        apple_commit='1309c06470deeea529645efcb79bd0744a16c445',
        kodcode_revision='dcf78a8bbba9a613b596ce993c4921a38687dfcc',
        mbpp_revision='4bb6404fdc6cacfda99d4ac4205087b89d32030c',
        humaneval_revision='7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544',
        metric_revision='b9bbd07a81faa38c020f938c01bfdcbf035b7f9b',
        prompt_parity='exact upstream loader functions; HumanEval0shot, MBPP3shot full500',
        mbpp_caveat='paper-time pre-saved dataset bytes unpinned', no_truncation=True,
        prompt_max={s:max(len(r['prompt_ids']) for r in v) for s,v in rows.items()},
        required_canvas=max(t['ids'].shape[1] for t in tensors.values()),
        beyond_default_canvas=[dict(split=s,index=r['index'],task_id=r['task_id'],tokens=len(r['prompt_ids'])+256)
                               for s,v in rows.items() for r in v if len(r['prompt_ids'])+256>4096],
        train_test_exact_normalized_text_overlap=overlaps,exact_reference_code_overlap=code_overlaps,
        near_duplicate_candidates=near,
        near_duplicate_audit='literal 5gram containment >=0.8 and >=5shared; manual review required; not semantic decontamination',
        kodcode_test_functions=test_nodes,required_fixture_rows=issues,
        test_syntax_errors=syntax_errors,
        sanitizer_hash=sha(BASE/'official/sanitize.py'), execution_hash=sha(BASE/'official/execute.py')))
    print('PREPARED_CODE_DATA', {s:len(v) for s,v in rows.items()},'fixture_rows',len(issues),flush=True)
if __name__=='__main__':main()
