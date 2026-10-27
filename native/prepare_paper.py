"""Audited full training mixture, no prompt truncation and no test leakage."""
import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.ipc as ipc
import torch
from transformers import AutoTokenizer

from evaluate_p2 import save, sha
from paper_protocol import official, prompt, score


def prepare(root, assets):
    assert not (root / "inputs.pt").exists()
    local = Path((assets / "local_root.txt").read_text().strip())
    manifest = json.loads((assets / "checkpoint_manifest.json").read_text())
    for entry in manifest["files"]:
        path = local / entry["local_path"]
        assert path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"]
    tok = AutoTokenizer.from_pretrained(local / "models/LLaDA-8B-Instruct",
                                       trust_remote_code=True, local_files_only=True)
    api = official(root / "official")
    specs = [("gsm8k-train.arrow", "train", "gsm8k", 7473),
             ("math-500-train.arrow", "train", "math", 7500),
             ("gsm8k-test.arrow", "test_gsm8k", "gsm8k", 1319),
             ("math-500-test.arrow", "test_math", "math", 500)]
    rows = {"train": [], "test_gsm8k": [], "test_math": []}
    raw_hashes, unparseable, empty_gold = {}, [], []
    for filename, split, task, count in specs:
        path = root / "data" / filename
        raw_hashes[filename] = sha(path)
        raw = ipc.open_stream(str(path)).read_all().to_pylist()
        assert len(raw) == count
        for i, item in enumerate(raw):
            question = item["question" if task == "gsm8k" else "problem"]
            if task == "gsm8k":
                answer = api["extract_answer_gsm8k"](None, item["answer"])
            elif split == "train":
                answer = api["remove_boxed"](api["last_boxed_only_string"](item["solution"]))
            else:
                answer = item["answer"]
            if answer is None:
                assert task == "gsm8k", (filename, i, "unexpected unparseable MATH gold")
                unparseable.append(dict(file=filename, index=i, raw_answer=item["answer"]))
            elif not str(answer).strip():
                assert task == "math" and split == "train"
                empty_gold.append(dict(file=filename, index=i, raw_solution=item["solution"]))
            text = prompt(api, tok, question, task)
            token_ids = tok(text, truncation=False)["input_ids"]
            assert len(token_ids) + 256 <= 4096, (filename, i, len(token_ids))
            gold = dict(task=task, answer=answer)
            # Gold sanity uses the task scorer, not a substitute numeric parser.
            if answer is not None:
                assert score(api, "\\boxed{" + str(answer) + "}", gold)["reward"] == 1, (filename, i, gold)
            else:
                assert score(api, "\\boxed{0}", gold)["reward"] == 0
            rows[split].append(dict(index=len(rows[split]), source_id=f"{filename}:{i}",
                question=question, prompt=text, prompt_ids=token_ids, gold=gold, task=task))
    training = {r["question"].strip() for r in rows["train"]}
    for split in ("test_gsm8k", "test_math"):
        overlap = training & {r["question"].strip() for r in rows[split]}
        assert not overlap, (split, "train/test exact-question overlap", len(overlap))
    # Non-held-out stratified probe for monitoring only, not model selection.
    rows["train_probe"] = [dict(r) for r in rows["train"][:32] + rows["train"][7473:7505]]
    tensors, lengths = {}, {}
    for split, entries in rows.items():
        maximum = max(len(r["prompt_ids"]) + 256 for r in entries)
        ids = torch.full((len(entries), maximum), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        positions = torch.empty((len(entries), 256), dtype=torch.long)
        for j, r in enumerate(entries):
            tokens = r["prompt_ids"]
            size = len(tokens)
            ids[j, :size] = torch.tensor(tokens)
            ids[j, size:size+256] = 126336
            mask[j, :size+256] = 1
            positions[j] = torch.arange(size, size+256)
            assert ids[j, :size].tolist() == tokens
        tensors[split] = dict(ids=ids, mask=mask, positions=positions)
        lengths[split] = dict(min=min(len(r["prompt_ids"]) for r in entries), max=maximum-256)
    torch.save(tensors, root / "inputs.pt")
    save(root / "rows.json", rows)
    save(root / "data-audit.json", dict(train=14973, train_task_counts=dict(gsm8k=7473, math=7500),
        tests=dict(test_gsm8k=1319, test_math=500), inputs_sha256=sha(root / "inputs.pt"),
        rows_sha256=sha(root / "rows.json"), raw_sha256=raw_hashes, no_truncation=True,
        exact_question_overlap=0, train_probe_held_out=False, prompt_lengths=lengths,
        official_unparseable_gold_rows=unparseable, unparseable_policy="retained; official reward always zero",
        official_empty_math_gold_rows=empty_gold, empty_math_gold_policy="retained; unmodified official scorer",
        official_source_hashes={p.name: sha(p) for p in (root / "official").glob("*.py")},
        prompt="APPLe task-specific evaluation template, zero shot, reasoning suffix; same train/test",
        scorer="APPLe official task-specific evaluator", answer_length=256,
        selection="fixed final horizon; tests never used for checkpoint choice",
        tokenization_version=__import__("transformers").__version__, model_assets=manifest))
    print("DATA_PREPARED", lengths, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    a = p.parse_args()
    prepare(a.root, a.assets)
