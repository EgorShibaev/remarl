"""Matched full-test P2 quality check; never import the candidate tokenizer/scorer.

Prepare inputs and random draws once in the reference environment. Generate with
each frozen backend, then decode and score BOTH in that same reference environment.
Test plan: CPU legacy-decoder parity, per-question RNG, fixed-batch padding and
strict coverage tests; real-GPU repeatability and HF legacy parity before full test.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch

DATA_SHA = "45965b000311d1550e5619b60b5bf31cf76edebfd8b8eddc62a876fbf8c9be95"
PORT_SHA = "73cac7db80542a3c25e1448332175864b5a9e53cd7c943fe76e107b736b3c0bd"
COUNT = 1319


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("x") as stream:
        json.dump(value, stream, allow_nan=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def question_seed(index, nfe, seed):
    key = f"gsm8k-test:{index}:{nfe}:{seed}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:15], 16)


def draws(index, nfe, seed, length=256):
    generator = torch.Generator().manual_seed(question_seed(index, nfe, seed))
    # Explicit per-step calls reproduce the historical CPU sampling stream.
    return torch.stack([torch.rand(length, generator=generator) for _ in range(nfe)])


def batch_indices(start, count=COUNT, batch=8):
    real = list(range(start, min(start + batch, count)))
    if not real:
        raise ValueError("Empty batch")
    return real + [real[-1]] * (batch - len(real)), len(real)


class P2Scores(torch.nn.Module):
    """Exactly confidence; no learned parameters or hidden-state conditioning."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(architecture="mlp", input_node_dim=3,
                                      position_encoding="learned", feature_encoding="none",
                                      backbone_hidden_dim=0)
        self.eval()

    def forward(self, features, scores):
        return scores


def validate_records(records, count=COUNT):
    assert [r["index"] for r in records] == list(range(count)), "Missing/duplicate test IDs"
    for r in records:
        assert r["source_id"] == f"gsm8k-test:{r['index']}"
        assert len(r["answer_token_ids"]) == 256
        assert 126336 not in r["answer_token_ids"]
        assert all(isinstance(t, int) and 0 <= t < 126464 for t in r["answer_token_ids"])


def prepare(root, assets):
    import pyarrow.ipc as ipc
    from transformers import AutoTokenizer
    from src.rl.environments import build_masked_problem

    assert not (root / "inputs.pt").exists()
    assert sha(root / "gsm8k-test.arrow") == DATA_SHA
    local = Path((assets / "local_root.txt").read_text().strip())
    manifest = json.loads((assets / "checkpoint_manifest.json").read_text())
    for entry in manifest["files"]:
        path = local / entry["local_path"]
        assert path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"]
    assert len(manifest["files"]) == 14
    assert sha(assets / "reference.tar.gz") == "9dd4199731e398cb9cf6e2b654adda7a67192364aa193f454b505526e5e45245"
    reference_sources = {str(path.relative_to(assets / "reference")): sha(path)
                         for path in sorted((assets / "reference/src").rglob("*.py"))}
    tokenizer = AutoTokenizer.from_pretrained(local / "models/LLaDA-8B-Instruct",
                                              trust_remote_code=True, local_files_only=True)
    raw = ipc.open_stream(str(root / "gsm8k-test.arrow")).read_all().to_pylist()
    assert len(raw) == COUNT
    rows, problems = [], []
    for i, row in enumerate(raw):
        prompt = f"Question: {row['question']}\nAnswer:"
        ids = tokenizer(prompt, truncation=False)["input_ids"]
        gold = row["answer"].split("####")[-1].strip()
        assert len(ids) <= 193 and gold
        problem = build_masked_problem(torch.tensor(ids), 256, 126336, 449,
                                       tokenizer.pad_token_id, gold)
        assert problem.input_ids[:len(ids)].tolist() == ids
        rows.append(dict(index=i, source_id=f"gsm8k-test:{i}", prompt=prompt,
                         gold=gold, prompt_ids=ids))
        problems.append(problem)
    assert max(len(r["prompt_ids"]) for r in rows) == 193
    tensors = {
        "ids": torch.stack([p.input_ids for p in problems]),
        "mask": torch.stack([p.attention_mask for p in problems]),
        "positions": torch.stack([p.answer_span for p in problems]),
        "uniforms": {f"{nfe}-{seed}": torch.stack([draws(i, nfe, seed) for i in range(COUNT)], 1)
                     for nfe in (8, 16) for seed in (101, 202)},
    }
    torch.save(tensors, root / "inputs.pt")
    save(root / "rows.json", rows)
    save(root / "protocol.json", {
        "count": COUNT, "nfe": [8, 16], "inference_seeds": [101, 202],
        "data_sha256": DATA_SHA, "inputs_sha256": sha(root / "inputs.pt"),
        "rows_sha256": sha(root / "rows.json"), "port_sha256": PORT_SHA,
        "worker_sha256": sha(root / "worker.py"), "evaluator_sha256": sha(root / "evaluate_p2.py"),
        "prompt": "Question: {question}\\nAnswer:", "scorer": "unchanged reference MathVerifier",
        "answer_length": 256, "padded_length": 449, "batch_size": 8,
        "tail": "Duplicate final row to batch8; exclude duplicates from coverage and scoring",
        "policy": "P2 confidence, PL position sampling T=1, argmax tokens, linear keep, remasking",
        "checkpoint_revision": "08b83a6feb34df1a6011b80c3c00c7563e963b07",
        "backends": ["hf", "vllm"], "hardware": "A100-SXM4-40GB",
        "verified_checkpoint_assets": manifest, "reference_sources": reference_sources,
        "caveat": "Different PyTorch/CUDA builds; same frozen weights, inputs, draws, batching and scorer. Fresh batched HF comparator, not a claim of reproducing historical batch1 trajectories.",
    })
    print("PREPARED", COUNT, sha(root / "inputs.pt"), flush=True)


class CountedEngine:
    def __init__(self, engine):
        self.engine, self.calls = engine, 0

    def prepare(self, ids, mask, positions):
        forward = (self.engine.prepare(ids, mask, positions)
                   if hasattr(self.engine, "prepare") else self.engine)

        def call(*args):
            self.calls += 1
            return forward(*args)

        return call


@torch.inference_mode()
def generate(root, assets, backend, nfe, seed):
    from benchmark_native import cuda_smoke, load_port
    from worker import AnswerBackbone, collect

    protocol = json.loads((root / "protocol.json").read_text())
    for name, key in (("inputs.pt", "inputs_sha256"), ("rows.json", "rows_sha256"),
                      ("worker.py", "worker_sha256"), ("evaluate_p2.py", "evaluator_sha256")):
        assert sha(root / name) == protocol[key]
    assert sha(root / "model.py") == PORT_SHA
    assert all(sha(assets / "reference" / name) == value
               for name, value in protocol["reference_sources"].items())
    assert torch.cuda.device_count() == 1
    assert "A100" in torch.cuda.get_device_name()
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    local = Path((assets / "local_root.txt").read_text().strip())
    checkpoint = local / "models/LLaDA-8B-Instruct"
    out = root / "results" / f"{backend}-nfe{nfe}-seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "started.json").exists(), "Existing attempt: inspect rather than overwrite"
    save(out / "started.json", dict(started=time.time(), backend=backend, nfe=nfe, seed=seed,
         protocol_sha256=sha(root / "protocol.json"), torch=torch.__version__,
         cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
         transformers=importlib.metadata.version("transformers"),
         node=os.uname().nodename, job=os.environ.get("SLURM_JOB_ID"),
         step=os.environ.get("SLURM_STEP_ID"), visible=os.environ.get("CUDA_VISIBLE_DEVICES")))
    tensors = torch.load(root / "inputs.pt", map_location="cpu", weights_only=True)
    if backend == "hf":
        from src.backbones.registry import build_backbone
        bb = build_backbone(dict(type="llada", name=str(checkpoint), dtype="bfloat16",
                                 trust_remote_code=True, mask_token_id=126336)).to("cuda")
        bb.freeze()
        engine = AnswerBackbone(bb)
        engine.__enter__()
    else:
        port = load_port(root)
        smoke = cuda_smoke(port)
        save(out / "cuda-smoke.json", smoke)
        torch.cuda.empty_cache()
        engine = port.LLaDAOfflineModel.from_local_checkpoint(checkpoint, backend="vllm",
                                                             fuse_projections=True)
        assert sum(p.numel() for p in engine.parameters()) == 8015581184
        bb = SimpleNamespace(mask_token_id=126336)
    counted = CountedEngine(engine)
    selector = P2Scores()

    def run(indices):
        ids, mask, pos = [tensors[k][indices].cuda() for k in ("ids", "mask", "positions")]
        uniforms = tensors["uniforms"][f"{nfe}-{seed}"][:, indices]
        before = counted.calls
        result = collect(bb, selector, ids, mask, pos, uniforms, answer_backbone=counted,
                         record=False)
        assert counted.calls - before == nfe
        editable = torch.zeros_like(ids, dtype=torch.bool).scatter_(1, pos, True)
        assert torch.equal(result.final_ids[~editable], ids[~editable])
        return result.final_ids.gather(1, pos).cpu(), result.revisions.cpu()

    # Same-device exact repeatability; legacy evaluator compatibility on HF B1.
    first, revisions = run(list(range(8)))
    repeated, repeated_rev = run(list(range(8)))
    assert torch.equal(first, repeated) and torch.equal(revisions, repeated_rev)
    gate = dict(repeatability=True, real_forward_count=nfe, prompt_and_padding_unchanged=True)
    if backend == "hf":
        from src.decoding.threshold_decoder import P2Decoder, P2DecoderConfig, ConfidenceZSource
        single, single_rev = run([0])
        decoder = P2Decoder(bb, P2DecoderConfig(steps=nfe, policy_temperature=1.,
                            policy_seed=question_seed(0, nfe, seed)), ConfidenceZSource())
        ids, mask, pos = [tensors[k][:1].cuda() for k in ("ids", "mask", "positions")]
        with engine.unhooked():
            legacy = decoder.decode(ids, mask, editable_positions=[pos[0]])
        assert torch.equal(single, legacy.input_ids.gather(1, pos).cpu())
        assert int(single_rev[0]) == legacy.metadata["remask_count"]
        gate["legacy_batch1_P2_exact"] = True
    save(out / "gate.json", gate)
    print("GATE_PASS", out.name, flush=True)
    started = time.perf_counter()
    records, batch_seconds = [], []
    torch.cuda.reset_peak_memory_stats()
    for start in range(0, COUNT, 8):
        indices, real = batch_indices(start)
        begin = time.perf_counter()
        tokens, revisions = run(indices)
        elapsed = time.perf_counter() - begin
        batch_seconds.append(elapsed)
        chunk = [dict(index=i, source_id=f"gsm8k-test:{i}", actual_nfe=nfe,
                      answer_token_ids=tokens[j].tolist(), revisions=int(revisions[j]))
                 for j, i in enumerate(indices[:real])]
        records.extend(chunk)
        save(out / f"batch-{start:04d}.json", dict(seconds=elapsed, rows=chunk))
        save(out / "progress.json", dict(count=len(records), seconds=time.perf_counter()-started))
        if len(records) % 128 == 0 or len(records) == COUNT:
            print("PROGRESS", out.name, len(records), round(time.perf_counter()-started, 1), flush=True)
    validate_records(records)
    save(out / "tokens.json", records)
    save(out / "generation.json", dict(complete=True, count=COUNT, backend=backend, nfe=nfe,
         seed=seed, seconds=time.perf_counter()-started, batch_seconds=batch_seconds,
         tokens_sha256=sha(out / "tokens.json"), peak_memory=torch.cuda.max_memory_allocated()))
    print("GENERATED", out.name, flush=True)


def score(root, assets, backend, nfe, seed):
    from transformers import AutoTokenizer
    from src.rl.answer_extraction import extract_math_answer
    from src.rl.verifiers import MathVerifier

    out = root / "results" / f"{backend}-nfe{nfe}-seed{seed}"
    gen = json.loads((out / "generation.json").read_text())
    assert gen["complete"] and gen["count"] == COUNT
    assert sha(out / "tokens.json") == gen["tokens_sha256"]
    records = json.loads((out / "tokens.json").read_text())
    validate_records(records)
    rows = json.loads((root / "rows.json").read_text())
    local = Path((assets / "local_root.txt").read_text().strip())
    tokenizer = AutoTokenizer.from_pretrained(local / "models/LLaDA-8B-Instruct",
                                              trust_remote_code=True, local_files_only=True)
    verifier = MathVerifier()
    for r, question in zip(records, rows):
        assert r["source_id"] == question["source_id"] and r["actual_nfe"] == nfe
        text = tokenizer.decode(r["answer_token_ids"], skip_special_tokens=True)
        r.update(generated_text=text, gold=question["gold"],
                 extracted_answer=extract_math_answer(text), correctness=verifier(text, question["gold"]))
        assert r["correctness"] in (0., 1.)
    correct = int(sum(r["correctness"] for r in records))
    save(out / "answers.json", records)
    save(out / "summary.json", dict(complete=True, count=COUNT, correct=correct,
         accuracy=correct/COUNT, backend=backend, nfe=nfe, seed=seed,
         answers_sha256=sha(out / "answers.json"), generation=gen,
         scorer_sha256=sha(assets / "reference/src/rl/verifiers.py"),
         extraction_sha256=sha(assets / "reference/src/rl/answer_extraction.py"),
         packages={k: importlib.metadata.version(k) for k in
                   ("transformers", "math-verify", "latex2sympy2_extended")}))
    print("SCORED", out.name, correct, COUNT, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("prepare", "generate", "score"))
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--assets", required=True, type=Path)
    p.add_argument("--backend", choices=("hf", "vllm"))
    p.add_argument("--nfe", type=int, choices=(8, 16))
    p.add_argument("--seed", type=int, choices=(101, 202))
    a = p.parse_args()
    if a.mode == "prepare":
        prepare(a.root, a.assets)
    else:
        assert a.backend and a.nfe and a.seed
        globals()[a.mode](a.root, a.assets, a.backend, a.nfe, a.seed)
