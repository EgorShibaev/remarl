"""Execute exact functions from pinned APPLe source, without importing its stack."""
from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace


def load_official(root):
    root = Path(root)
    namespace = {"re": re}
    loader = ast.parse((root / "gsm8k.py").read_text())
    prompt = next(n for n in loader.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "GSM_SYSTEM_PROMPT" for t in n.targets))
    cls = next(n for n in loader.body if isinstance(n, ast.ClassDef) and n.name == "GSM8KDataset")
    create = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "create_prompt")
    parser = ast.parse((root / "parse_and_get_acc.py").read_text())
    functions = [n for n in parser.body if isinstance(n, ast.FunctionDef)
                 and n.name in {"extract_gsm_answer", "check_gsm_correct"}]
    goldtree = ast.parse((root / "parsers.py").read_text())
    goldcls = next(n for n in goldtree.body if isinstance(n, ast.ClassDef) and n.name == "Parser")
    gold = next(n for n in goldcls.body if isinstance(n, ast.FunctionDef) and n.name == "extract_answer_gsm8k")
    gold.decorator_list = []
    assert len(functions) == 2
    module = ast.Module(body=[prompt, create, *functions, gold], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "pinned_APPLe_functions", "exec"), namespace)
    return namespace


def make_prompt(official, tokenizer, question):
    context = SimpleNamespace(tokenizer=tokenizer, num_examples=0, add_reasoning=True,
                              system_prompt=official["GSM_SYSTEM_PROMPT"])
    return official["create_prompt"](context, question)


def score(official, text, gold):
    parsed = official["extract_gsm_answer"](text)
    return dict(apple_extracted=parsed, apple_parseable=parsed is not None,
                apple_correctness=int(official["check_gsm_correct"](parsed, gold)))
