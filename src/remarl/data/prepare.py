"""Prepare the paper's prompts, training cohorts, and complete test sets."""

import ast
import json
from pathlib import Path
import random
import re
from types import SimpleNamespace
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from ..io import save_json, sha256
from ..protocols.math import official, prompt
from ..protocols.hashes import digest_row
from ..runtime import load_tokenizer

OFFICIAL = Path(__file__).resolve().parents[1] / "protocols/official"


def read_rows(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pq.read_table(path).to_pylist()
    if path.suffix == ".arrow":
        return ipc.open_stream(str(path)).read_all().to_pylist()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines()]
    raise ValueError(f"Expected Arrow, Parquet, or JSONL: {path}")


def load_named(directory, name):
    matches = [Path(directory) / (name + suffix) for suffix in (".arrow", ".parquet", ".jsonl")]
    present = [p for p in matches if p.exists()]
    if len(present) != 1:
        raise ValueError(f"Provide exactly one file for {name}: {matches}")
    return read_rows(present[0]), present[0]


def code_api():
    # Compile reviewed prompt methods only; upstream dataset loaders are unused.
    scope = dict(re=re, Path=Path)
    for name, cls_name in (("humaneval", "HumanEvalDataset"), ("mbpp", "MBPPDataset")):
        tree = ast.parse((OFFICIAL / f"{name}.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
        nodes = [n for n in tree.body if isinstance(n, ast.Assign)]
        methods = [
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name in ("create_prompt", "create_few_shot_prompt")
        ]
        # Keep only prompt constants, excluding filesystem-dependent assignments.
        nodes = [n for n in nodes if isinstance(n.value, (ast.Constant, ast.JoinedStr))]
        for method in methods:
            method.name = f"{name}_{method.name}"
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=nodes + methods, type_ignores=[])),
                f"APPLe_{name}_prompts",
                "exec",
            ),
            scope,
        )
    tree = ast.parse((OFFICIAL / "data_utils.py").read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("kodcode_preprocess_test", "get_kodcode_questions")
    ]
    scope["Dataset"] = object
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "APPLe_kodcode", "exec"), scope)
    return scope


def kodcode_prompts(api, rows, tokenizer):
    class Rows:
        def __contains__(self, key):
            return False

        def map(self, fn):
            return [fn(row) for row in rows]

    api["_load_dataset_with_fallback"] = lambda *args, **kwargs: Rows()
    entries = api["get_kodcode_questions"]()
    for entry in entries:
        entry["text"] = (
            tokenizer.apply_chat_template(
                entry["prompt"], add_generation_prompt=True, tokenize=False
            )
            + entry["gen_prefix"]
        )
    return entries


def prepare_math(raw, tokenizer):
    api = official(OFFICIAL)
    rows, hashes = {"train": [], "test_gsm8k": [], "test_math": []}, {}
    sources = [
        ("gsm8k-train", "train", "gsm8k", 7473),
        ("math-500-train", "train", "math", 7500),
        ("gsm8k-test", "test_gsm8k", "gsm8k", 1319),
        ("math-500-test", "test_math", "math", 500),
    ]
    for name, split, task, count in sources:
        values, path = load_named(raw, name)
        if len(values) != count:
            raise ValueError(f"{name}: expected {count}, found {len(values)}")
        hashes[path.name] = sha256(path)
        for i, item in enumerate(values):
            question = item["question" if task == "gsm8k" else "problem"]
            if task == "gsm8k":
                answer = api["extract_answer_gsm8k"](None, item["answer"])
            elif split == "train":
                answer = api["remove_boxed"](api["last_boxed_only_string"](item["solution"]))
            else:
                answer = item["answer"]
            text = prompt(api, tokenizer, question, task)
            rows[split].append(
                dict(
                    index=len(rows[split]),
                    source_id=f"{name}.arrow:{i}",
                    task=task,
                    question=question,
                    prompt=text,
                    prompt_ids=tokenizer(text, truncation=False)["input_ids"],
                    gold=dict(task=task, answer=answer),
                )
            )
    train_questions = {r["question"].strip() for r in rows["train"]}
    if any(
        r["question"].strip() in train_questions
        for s in ("test_gsm8k", "test_math")
        for r in rows[s]
    ):
        raise ValueError("Exact train/test question overlap")
    return rows, hashes


def prepare_code(raw, tokenizer, cohort):
    api = code_api()
    sources = {}
    hashes = {}
    for name in ("kodcode", "humaneval", "mbpp-test", "mbpp-prompt"):
        sources[name], path = load_named(raw, name)
        hashes[path.name] = sha256(path)
    folder = Path(__file__).parent / "cohorts" / cohort
    allow = [json.loads(line) for line in (folder / "allowlist.jsonl").read_text().splitlines()]
    selected = []
    for record in allow:
        item = sources["kodcode"][record["source_index"]]
        if digest_row(item) != record["row_sha256"] or not record["reference_passed"]:
            raise ValueError(f"KodCode row changed: {record['source_index']}")
        selected.append(item)
    entries = kodcode_prompts(api, selected, tokenizer)
    rows = {"train": [], "test_humaneval": [], "test_mbpp": []}
    for i, (item, source, record) in enumerate(zip(entries, selected, allow)):
        gold = dict(
            task="kodcode",
            source_index=record["source_index"],
            raw_prompt=item["raw_prompt"] + "\n",
            entry_point=item["function_name"],
            tests=api["kodcode_preprocess_test"](source["test"]),
            reference=source["solution"],
        )
        rows["train"].append(
            dict(
                index=i,
                task="kodcode",
                task_id=source["question_id"],
                prompt=item["text"],
                gold=gold,
            )
        )
    he = SimpleNamespace(tokenizer=tokenizer)
    mb = SimpleNamespace(
        tokenizer=tokenizer,
        num_examples=3,
        load_few_shot_examples=lambda: sources["mbpp-prompt"][:3],
    )
    api["mbpp_create_few_shot_prompt"](mb)
    for task in ("humaneval", "mbpp"):
        for i, item in enumerate(sources["humaneval" if task == "humaneval" else "mbpp-test"]):
            if task == "humaneval":
                text = api["humaneval_create_prompt"](he, item["prompt"])
                tests = item["test"] + "\ncheck(" + item["entry_point"] + ")\n"
                reference = item["prompt"] + item["canonical_solution"]
            else:
                text = api["mbpp_create_prompt"](mb, item["text"], item["test_list"])
                tests, reference = "\n".join(item["test_list"]), item["code"]
            gold = dict(
                task=task,
                tests=tests,
                reference=reference,
                raw_prompt=item.get("prompt", ""),
                entry_point=item.get("entry_point"),
            )
            rows["test_" + task].append(
                dict(index=i, task=task, task_id=item["task_id"], prompt=text, gold=gold)
            )
    for split in rows.values():
        for row in split:
            row["prompt_ids"] = tokenizer(row["prompt"], truncation=False)["input_ids"]
    if len(rows["test_humaneval"]) != 164 or len(rows["test_mbpp"]) != 500:
        raise ValueError("Incomplete code benchmarks")
    return rows, hashes


def prepare(domain, raw, model, output, cohort="code16"):
    output = Path(output)
    if (output / "rows.json").exists():
        raise FileExistsError(f"Prepared data already exists: {output}")
    tokenizer = load_tokenizer(model)
    rows, hashes = (
        prepare_math(raw, tokenizer) if domain == "math" else prepare_code(raw, tokenizer, cohort)
    )
    required = max(len(r["prompt_ids"]) + 256 for split in rows.values() for r in split)
    if required > (4096 if domain == "math" else 8192):
        raise ValueError(f"Canvas too long ({required}); prompts are never truncated")
    save_json(output / "rows.json", rows)
    save_json(
        output / "manifest.json",
        dict(
            domain=domain,
            cohort=cohort if domain == "code" else None,
            counts={s: len(v) for s, v in rows.items()},
            required_canvas=required,
            answer_length=256,
            raw_sha256=hashes,
            rows_sha256=sha256(output / "rows.json"),
        ),
    )


def load_prepared(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if sha256(path / "rows.json") != manifest["rows_sha256"]:
        raise ValueError("Prepared rows changed; prepare a new dataset directory")
    return json.loads((path / "rows.json").read_text()), manifest


def subset(rows, seed, per_dataset):
    selected = []
    for offset, task in enumerate(("gsm8k", "math")):
        indices = [i for i, r in enumerate(rows) if r["task"] == task]
        random.Random(seed * 100 + offset).shuffle(indices)
        selected.extend(indices[:per_dataset])
    if len(selected) != 2 * per_dataset:
        raise ValueError("Insufficient questions for requested balanced subset")
    return [rows[i] for i in selected], selected
