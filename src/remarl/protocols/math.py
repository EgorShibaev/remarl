"""Exact APPLe evaluation templates/scorers, used consistently in train and test.

The selected AST function bodies are unmodified; imports of unrelated code
execution evaluators are deliberately avoided. Vendored sources retain license.
"""

import ast
import re
from pathlib import Path
from types import SimpleNamespace


def official(root):
    root = Path(root)
    ns = {"re": re}
    nodes = []
    for filename, constant in (
        ("gsm8k.py", "GSM_SYSTEM_PROMPT"),
        ("math500.py", "MATH500_SYSTEM_PROMPT"),
    ):
        tree = ast.parse((root / filename).read_text())
        nodes += [
            n
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == constant for t in n.targets)
        ]
        if filename == "gsm8k.py":
            cls = next(
                n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GSM8KDataset"
            )
            nodes += [
                n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "create_prompt"
            ]
    utils = ast.parse((root / "parser_utils.py").read_text())
    nodes += [n for n in utils.body if isinstance(n, ast.FunctionDef)]
    parser = ast.parse((root / "parse_and_get_acc.py").read_text())
    nodes += [
        n
        for n in parser.body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in {"extract_gsm_answer", "check_gsm_correct", "extract_math_answer", "check_math_correct"}
    ]
    goldcls = next(
        n
        for n in ast.parse((root / "parsers.py").read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "Parser"
    )
    gold = next(
        n
        for n in goldcls.body
        if isinstance(n, ast.FunctionDef) and n.name == "extract_answer_gsm8k"
    )
    gold.decorator_list = []
    nodes.append(gold)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
            "pinned_APPLe_evaluation_protocol",
            "exec",
        ),
        ns,
    )
    return ns


def prompt(api, tokenizer, question, task):
    context = SimpleNamespace(
        tokenizer=tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=api["GSM_SYSTEM_PROMPT" if task == "gsm8k" else "MATH500_SYSTEM_PROMPT"],
    )
    return api["create_prompt"](context, question)


def score(api, text, gold):
    task = gold["task"]
    assert task in ("gsm8k", "math")
    key = "gsm" if task == "gsm8k" else "math"
    extracted = api[f"extract_{key}_answer"](text)
    return dict(
        text=text,
        extracted=extracted,
        reward=float(api[f"check_{key}_correct"](extracted, gold["answer"])),
        task=task,
    )
