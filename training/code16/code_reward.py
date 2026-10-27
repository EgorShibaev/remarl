"""Exact pinned APPLe sanitizer/tests; only isolated namespace program execution."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer
from validated_training import InfrastructureFailure, ReferenceCheckedReward, load_validated

BASE = Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'protocol'))
from prepare import preprocess, sanitizer
from sandbox import run


def checked_verdict(result):
    if (type(result.get('passed')) is not bool or not isinstance(result.get('result'),str)
            or not (result['result']=='passed' or result['result']=='timed out'
                    or result['result'].startswith('failed:'))
            or result['passed'] != (result['result']=='passed')):
        raise InfrastructureFailure('Malformed or infrastructure sandbox verdict')
    return result


class Scorer:
    def __init__(self, root, assets, log):
        self.root = root
        local = Path((assets/'local_root.txt').read_text().strip())
        self.tokenizer = AutoTokenizer.from_pretrained(local/'models/LLaDA-8B-Instruct',
                              trust_remote_code=True,local_files_only=True)
        rows, allow, manifest = load_validated(BASE/'cohort',BASE/'protocol')
        self.sanitize = sanitizer()
        self.guard = ReferenceCheckedReward(rows,allow,run,self.sanitize.sanitize_humaneval,preprocess())
        self.pool = ThreadPoolExecutor(max_workers=8)
        self.begin_collection()

    def begin_collection(self): self.guard.begin_collection()

    def score(self, pair):
        tokens, gold = pair
        text = self.tokenizer.decode(tokens,skip_special_tokens=True)
        if gold['task']=='kodcode':
            result = self.guard.score_program(gold['source_index'],gold['raw_prompt']+text)
        else:
            code = (self.sanitize.sanitize_humaneval(gold['raw_prompt']+text,gold['entry_point'])
                    if gold['task']=='humaneval' else self.sanitize.sanitize_mbpp(text))
            result = dict(reward=float((outcome:=checked_verdict(run(code+'\n'+gold['tests'])))['passed']),
                          **outcome)
        return dict(text=text,task=gold['task'],**result)

    def __call__(self, tokens, golds):
        assert len(tokens)==len(golds)
        return list(self.pool.map(self.score,zip(tokens,golds)))

    def close(self): self.pool.shutdown(wait=True,cancel_futures=True)
