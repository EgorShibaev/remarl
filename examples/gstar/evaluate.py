"""Calibrate positive loop counts on GSM8K training prompts and score both full tests."""

import argparse
from pathlib import Path

import torch
import torch.distributed as dist

from common import BUDGETS, LENGTH, batches, bind, normalized, question_splits, setup
from remarl.io import append_json, save_json, sha256
from remarl.protocols.scoring import Scorer
from remarl.runtime import MASK_ID


@torch.inference_mode()
def generate(answer, forward, head, budget, loops):
    draft_calls = budget - 2 * loops
    if draft_calls < 1 or loops < 1:
        raise ValueError("Allow at least one draft call and one two-call refinement loop")
    for step in range(draft_calls):
        logits, _ = forward(answer)
        logits[..., MASK_ID] = -torch.inf
        top, proposals = logits.max(-1)
        confidence = top.float() - logits.float().logsumexp(-1)
        count = (step + 1) * LENGTH // draft_calls - step * LENGTH // draft_calls
        positions = confidence.masked_fill(answer != MASK_ID, -torch.inf).topk(count).indices
        answer.scatter_(1, positions, proposals.gather(1, positions))
        del logits
    for _ in range(loops):
        logits, hidden = forward(answer)
        del logits
        positions = (normalized(hidden) @ head[0] + head[1]).topk(15).indices
        answer.scatter_(1, positions, MASK_ID)
        logits, _ = forward(answer)
        logits[..., MASK_ID] = -torch.inf
        answer.scatter_(1, positions, logits.argmax(-1).gather(1, positions))
        del logits
    return answer.cpu().tolist()


def score(rows, split, budget, loops, scorer, model, tokenizer, head, device, output):
    counts = torch.zeros(2, device=device, dtype=torch.long)
    rank = dist.get_rank()
    path = output / f"{split}-n{budget}-loops{loops}-rank{rank}.jsonl"
    for indices in batches(range(rank, len(rows), dist.get_world_size())):
        batch = [rows[i] for i in indices]
        answer, forward = bind(model, tokenizer, batch, device)
        tokens = generate(answer, forward, head, budget, loops)
        outcomes = scorer(tokens, [row["gold"] for row in batch])
        for row, outcome in zip(batch, outcomes, strict=True):
            append_json(path, {"qid": row["source_id"], **outcome})
        counts[0] += sum(int(o["reward"]) for o in outcomes)
        counts[1] += len(outcomes)
    dist.all_reduce(counts)
    correct, count = counts.tolist()
    assert count == len(rows)
    return dict(
        loops=loops,
        draft_calls=budget - 2 * loops,
        refinement_calls=2 * loops,
        correct=correct,
        count=count,
        accuracy=correct / count,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", type=Path, default=Path(__file__).with_name("head.pt"))
    args, device, rows, tokenizer, model = setup(parser)
    head = torch.load(args.head, map_location=device, weights_only=True)
    _, _, selected = question_splits(rows["train"])
    validation = [rows["train"][i] for i in selected]
    scorer = Scorer(tokenizer, "math")
    results = {}
    for budget in BUDGETS:
        candidates = sorted({max(1, round(budget * f / 2)) for f in (0.10, 0.25, 0.50)})

        def evaluate(batch, name, loops):
            return score(
                batch, name, budget, loops, scorer, model, tokenizer, head, device, args.output
            )

        scores = [evaluate(validation, "calibration", loops) for loops in candidates]
        best = max(scores, key=lambda s: (s["correct"], -s["loops"]))
        results[budget] = {"calibration": scores}
        for benchmark, split in (("gsm8k", "test_gsm8k"), ("math500", "test_math")):
            results[budget][benchmark] = evaluate(rows[split], benchmark, best["loops"])
            if dist.get_rank() == 0:
                save_json(
                    args.output / "results.json",
                    {"head_sha256": sha256(args.head), "budgets": results},
                )
    scorer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
