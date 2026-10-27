"""Small, repeated-question diagnostic; not a benchmark or an old-recipe reproduction."""

import copy
import hashlib
import random
import unicodedata

CELLS = {"gsm-t01": .1, "gsm-t10": 1.0}
LABELS = {"gsm-t01": "GSM8K · temperature 0.1", "gsm-t10": "GSM8K · temperature 1.0"}
TRAIN_SIZE = 32
DEV_SIZE = 64
PASSES = 16
NFE = 32
GROUP = 8
EVALUATIONS = (0, 128, 256, 512)


def select_data(source, source_sha):
    """Select without looking at model outcomes; retain earlier split provenance."""
    rng = random.Random(20260914)
    selected = {}
    for split, size in (("training", TRAIN_SIZE), ("development", DEV_SIZE)):
        rows = sorted((r for r in source[split] if r["domain"] == "gsm8k"),
                      key=lambda r: r["source_id"])
        if len(rows) < size:
            raise ValueError("Insufficient GSM8K questions")
        selected[split] = copy.deepcopy(rng.sample(rows, size))
    train_ids = {r["question_sha256"] for r in selected["training"]}
    dev_ids = {r["question_sha256"] for r in selected["development"]}
    if len(train_ids) != TRAIN_SIZE or len(dev_ids) != DEV_SIZE or train_ids & dev_ids:
        raise ValueError("Duplicate or overlapping questions")
    for row in selected["training"] + selected["development"]:
        key = " ".join(unicodedata.normalize("NFKC", row["question"]).split())
        if hashlib.sha256(key.encode()).hexdigest() != row["question_sha256"]:
            raise ValueError("Question hash mismatch")
    return {
        **selected, "version": 1, "parent_data_sha256": source_sha,
        "parent_protocol": source.get("protocol"), "selection_seed": 20260914,
        "protocol": "32 GSM8K training questions repeated 16 times; 64 disjoint dev; fixed32 NFE",
        "selection_uses_model_outcomes": False,
        "development_is_official_test": False,
    }


def make_overfit_plan(seed, *, size=TRAIN_SIZE, passes=PASSES, steps=NFE, group=GROUP):
    """Match the trainer's independent epoch sampler, including repeated epochs."""
    import torch

    rng = torch.Generator().manual_seed(seed + 100003)
    return [{"problem_idx": int(i), "steps": [steps] * group}
            for _ in range(passes) for i in torch.randperm(size, generator=rng)]


def evaluation_metrics(rows, update):
    output = {"progress/attempted_questions": update}
    for split in ("train_fit", "heldout"):
        for method in ("learned_remasking", "confidence_remasking"):
            subset = [r for r in rows if r["split"] == split and r["method"] == method]
            if subset:
                prefix = f"overfit/{split}/{method}"
                output[prefix + "/accuracy"] = sum(r["correctness"] for r in subset) / len(subset)
                output[prefix + "/revisions"] = sum(r["revisions"] for r in subset) / len(subset)
                output[prefix + "/questions"] = len(subset)
    return output
