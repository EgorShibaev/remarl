"""Complete-benchmark evaluation with per-question seeds and resumable output."""

import json
from pathlib import Path
import statistics
import torch
from .data.prepare import load_prepared
from .io import append_json, save_json
from .protocols.scoring import Scorer
from .runtime import Generator, load_policy, configure_compute
from .sampling import draws, P2Scores


def evaluate(generator, scorer, rows, output, *, nfe, seeds, identity, batch_size=8):
    if not rows or batch_size < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide nonempty rows, unique seeds, and a positive batch size")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    request = dict(identity, nfe=nfe, seeds=list(seeds), count=len(rows), batch_size=batch_size)
    request_path = output / "evaluation.json"
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise ValueError("Output belongs to another evaluation; choose a new directory")
    save_json(request_path, request)
    summaries = []
    for seed in seeds:
        path = output / f"answers-seed{seed}.jsonl"
        records = (
            [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        )
        if [r["index"] for r in records] != list(range(len(records))) or len(records) > len(rows):
            raise ValueError("Saved answers are not a complete, ordered prefix")
        scorer.begin_collection()
        # Resume an interrupted batch with its original padding and tensor shape.
        first = (
            len(records) if len(records) == len(rows) else len(records) // batch_size * batch_size
        )
        for start in range(first, len(rows), batch_size):
            indices = list(range(start, min(start + batch_size, len(rows))))
            real = len(indices)
            indices += [indices[-1]] * (batch_size - real)
            uniforms = torch.stack([draws(i, nfe, seed) for i in indices], 1)
            result = generator.generate([rows[i] for i in indices], uniforms, False)
            answers = scorer(result["tokens"][:real], [rows[i]["gold"] for i in indices[:real]])
            for i, tokens, answer in zip(indices[:real], result["tokens"][:real], answers):
                record = dict(index=i, tokens=tokens, **answer)
                if i < len(records):
                    if records[i] != record:
                        raise RuntimeError("Resumed batch differs from saved answers")
                    continue
                records.append(record)
                append_json(path, record)
        if len(records) != len(rows):
            raise RuntimeError("Incomplete benchmark coverage")
        correct = sum(r["reward"] for r in records)
        summary = dict(seed=seed, count=len(rows), correct=correct, accuracy=correct / len(rows))
        summaries.append(summary)
        save_json(output / f"summary-seed{seed}.json", summary)
    accuracy = [s["accuracy"] for s in summaries]
    summary = dict(
        request,
        complete=True,
        seeds=summaries,
        accuracy=statistics.mean(accuracy),
        accuracy_percent=100 * statistics.mean(accuracy),
        seed_std=statistics.stdev(accuracy) if len(accuracy) > 1 else None,
    )
    save_json(output / "summary.json", summary)
    return summary


def run_evaluation(
    data,
    model,
    output,
    policy_path,
    nfe,
    seeds,
    device,
    backend,
    split=None,
    batch_size=8,
    temperature=1.0,
    retention_bonus=0.0,
):
    if policy_path and (temperature != 1.0 or retention_bonus != 0.0):
        raise ValueError("Temperature and retention controls apply only to --p2-prime")
    configure_compute()
    rows, manifest = load_prepared(data)
    if policy_path:
        policy, saved = load_policy(policy_path, device)
        identity = dict(method="remarl", policy_sha256=saved["policy_sha256"])
    else:
        policy = P2Scores(temperature, retention_bonus)
        identity = dict(
            method="p2-prime", position_temperature=temperature, retention_bonus=retention_bonus
        )
    identity.update(data_sha256=manifest["rows_sha256"], backend=backend)
    generator = Generator(model, policy, device, backend, manifest["required_canvas"])
    scorer = Scorer(generator.tokenizer, manifest["domain"])
    splits = [split] if split else [s for s in rows if s.startswith("test_")]
    try:
        results = {}
        for name in splits:
            if not name.startswith("test_") or name not in rows:
                raise ValueError(f"Unknown test split: {name}")
            results[name] = evaluate(
                generator,
                scorer,
                rows[name],
                Path(output) / name,
                nfe=nfe,
                seeds=seeds,
                identity=dict(identity, split=name),
                batch_size=batch_size,
            )
        return results
    finally:
        scorer.close()
