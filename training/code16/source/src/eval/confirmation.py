"""Small paired frozen-checkpoint audit; pure identity/resume helpers."""
import hashlib
import json
import random
import unicodedata

BUDGETS = (8, 16, 32, 64)
STUDY = "20260915-confirmation"
SEEDS = (7, 13, 42, 101)
CELLS = ("mixed-correctness", "shared-correctness", "mixed-separated", "shared-separated")
JULY = {
    "july-base": ("grpo-20260720-073701", "b7864d2e15006717fdbc873fc4c58daa62d5fc4e618fc654074929db2299edd1"),
    "july-alpha14": ("grpo-20260722-071830", "3454b1784cf5dce27c9b924b4eb962292eb1e4931956dc9564a463d1c9e2b9e5"),
    "july-power": ("grpo-20260726-183939-1989083-1", "367ff79fd57bb915ddce6954f4692f1c8ba92ebbffebbb4456638b83ea4d4928"),
    "july-gamma1": ("grpo-20260726-183941", "9d1cf00d318449ae92fd1236bde7c3b9bb4ae5df75b5f3c1f92db66ef9e0b690"),
    "july-history": ("grpo-20260726-183942-597540-1", "5007d1c01d40005abcb646797d2332dd757fec6431bfca0edbb282ebe7672915"),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def question_hash(question):
    return hashlib.sha256(" ".join(unicodedata.normalize("NFKC", question).split()).encode()).hexdigest()


def select_rows(rows, tokenizer, forbidden=(), count=128):
    """Filter by provenance/length, then sample without looking at model outcomes."""
    candidates, excluded, seen = [], [], set(forbidden)
    for index, row in enumerate(rows):
        q = row["question"]
        qhash = question_hash(q)
        prompt = f"Question: {q}\nAnswer:"
        ids = tokenizer(prompt, truncation=False)["input_ids"]
        if qhash in seen or len(ids) > 256:
            excluded.append({"index": index, "reason": "duplicate_or_train_overlap" if qhash in seen else "prompt_over_256"})
            continue
        seen.add(qhash)
        assert "####" in row["answer"]
        gold = row["answer"].split("####")[-1].strip()
        assert gold
        candidates.append(dict(source_id=f"gsm8k-test:{index}", question=q,
                               question_sha256=qhash, prompt=prompt, prompt_ids=ids,
                               gold=gold, reference=row["answer"], domain="gsm8k"))
    selected = random.Random(20260914).sample(candidates, count)
    return dict(rows=selected, eligible=len(candidates), excluded=excluded,
                selection_seed=20260914, selection="uniform_without_replacement_after_filters")


def validate_prefix(records, rows, nfe):
    assert len(records) <= len(rows)
    for record, row in zip(records, rows):
        assert record["source_id"] == row["source_id"]
        assert record["gold"] == row["gold"]
        assert record["requested_nfe"] == record["actual_nfe"] == nfe
        assert record["complete"] and len(record["answer_token_ids"]) == 256
        assert record["correctness"] in (0., 1.)


def assignments(methods, worker):
    assert worker in range(8) and len(methods) == 8
    return [methods[worker]]
