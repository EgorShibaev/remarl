"""Native fork parity pilot on saved fixed states and unchanged policy rollouts.

Two processes/environments: export with the pinned HF control; check with the
exact vLLM wheel plus our new source module. No training or optimizer changes.
Numerical differences are reported, not silently excused as sampling noise.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch


def save_json(path, data):
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(data, indent=2, allow_nan=False))
    partial.replace(path)


def load_port(root):
    spec = importlib.util.spec_from_file_location("llada_native_port", root / "model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(logits, hidden, current):
    probs = logits.float().softmax(-1)
    return {
        "hidden": hidden.cpu(),
        "argmax": logits.argmax(-1).cpu(),
        "confidence": probs.max(-1).values.cpu(),
        "current_probability": probs.gather(-1, current[..., None])[..., 0].cpu(),
        "logit_probe": logits[..., ::251].float().cpu(),
    }


def difference(reference, actual):
    out = {"argmax_mismatches": int((reference["argmax"] != actual["argmax"]).sum())}
    for name in ("hidden", "confidence", "current_probability", "logit_probe"):
        delta = (reference[name].float() - actual[name].float()).abs()
        out[name + "_max_abs"] = float(delta.max())
        out[name + "_mean_abs"] = float(delta.mean())
    out["finite"] = all(torch.isfinite(t).all().item() for t in actual.values())
    out["bit_exact"] = all(torch.equal(reference[k], actual[k]) for k in reference)
    return out


def measure(forward, ids, mask, pos):
    forward(ids, mask, pos)
    torch.cuda.synchronize()
    times = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        start = time.perf_counter()
        logits, hidden = forward(ids, mask, pos)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        del logits, hidden
    logits, hidden = forward(ids, mask, pos)
    summary = summarize(logits, hidden, ids.gather(1, pos))
    return summary, {
        "forward_seconds_median": statistics.median(times),
        "forward_seconds": times,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


def selector_for(assets):
    from src.models.set_selector import SetSelector, SetSelectorConfig

    path = assets / "sota-selector.pt"
    assert digest(path) == "36b2938e70b2d82d8447f2035c6e864d61fcf42a990965ac577aef61cc87d3bc"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    selector = SetSelector(SetSelectorConfig(**saved["set_selector_config"])).cuda().eval()
    selector.load_state_dict(saved["model_state"])
    return selector, saved


def rollout_summary(result):
    # Retain replay for 8 NFE only; bounded ~224 MiB per G8 trajectory bundle.
    return {
        "final_ids": result.final_ids.cpu(), "old_logp": result.old_logp,
        "orders": result.orders, "features": result.features,
        "revisions": result.revisions.cpu(), "nfe": result.nfe,
    }


def cuda_smoke(port):
    """Real FA2 boundary/noncausal smoke before allocating the 8B checkpoint."""
    with torch.device("cuda"):
        config = port.LLaDAConfig(128, 2, 1, 2, 192, 301, 32)
        control = port.LLaDAOfflineModel(config, backend="torch")
        candidate = port.LLaDAOfflineModel(config, backend="vllm")
        weights = [(k, torch.randn_like(v) * 0.1) for k, v in control.weight_mapping().items()]
        control.load_weights(weights)
        candidate.load_weights(weights)
        ids = torch.randint(0, 301, (2, 16))
        mask = torch.ones_like(ids)
        mask[0, 14:] = 0
        mask[1, [0, 13, 14, 15]] = 0
        positions = torch.tensor([[2, 4, 7], [2, 6, 8]])
        layout = port.CanvasLayout.build(mask, positions)
        expected = control(ids, layout)
        actual = candidate(ids, layout)
        errors = [float((a.float() - b.float()).abs().max()) for a, b in zip(actual, expected)]
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=0.03, rtol=0.1)
        changed = ids.clone()
        changed[mask == 0] = (changed[mask == 0] + 1) % 301
        padded = candidate(changed, layout)
        for a, b in zip(actual, padded):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        changed[1, 12] = (changed[1, 12] + 1) % 301
        other = candidate(changed, layout)[1]
        torch.testing.assert_close(other[0], actual[1][0], atol=0, rtol=0)
        assert not torch.equal(other[1], actual[1][1]), "Future token not visible"
    return {"passed": True, "max_abs_logits_hidden": errors,
            "padding_and_request_isolation": "exact", "noncausal": True}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("export", "check"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--backend", choices=("torch", "vllm"), default="vllm")
    parser.add_argument("--separate-projections", action="store_true")
    args = parser.parse_args()
    root, assets = args.root, args.assets
    suffix = "-separate" if args.separate_projections else ""
    output = root / ("reference.json" if args.mode == "export" else f"{args.backend}{suffix}.json")
    assert not output.exists(), f"Refuse to overwrite {output}"
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    local = Path((assets / "local_root.txt").read_text().strip())
    checkpoint = local / "models/LLaDA-8B-Instruct"
    report = {
        "status": "running", "mode": args.mode, "backend": args.backend,
        "fuse_projections": not args.separate_projections,
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "source_sha": digest(root / "model.py"),
        "started_unix": time.time(), "states": [], "rollouts": [],
        "scope": "G8, answer256, padded512, two training questions; NOT GSM8K test accuracy",
    }
    save_json(output, report)
    from worker import AnswerBackbone, collect

    if args.mode == "export":
        from src.backbones.registry import build_backbone
        from src.rl.environments import build_masked_problem

        bb = build_backbone({"type": "llada", "name": str(checkpoint),
                             "dtype": "bfloat16", "trust_remote_code": True,
                             "mask_token_id": 126336}).to("cuda")
        bb.freeze()
        engine = AnswerBackbone(bb)
        engine.__enter__()
        rows = json.loads((assets / "original-questions.json").read_text())["training"][:2]
        states = []
        for row in rows:
            tokens = bb.tokenizer(row["prompt"], truncation=False)["input_ids"]
            assert len(tokens) == row["prompt_token_count"]
            problem = build_masked_problem(torch.tensor(tokens), 256, bb.mask_token_id,
                                           512, bb.pad_token_id, row["gold"])
            states.append({"ids": problem.input_ids[None].repeat(8, 1),
                           "mask": problem.attention_mask[None].repeat(8, 1),
                           "positions": problem.answer_span[None].repeat(8, 1)})
        # Changed state exercises visible answer tokens, not just all-mask input.
        changed = {k: v.clone() for k, v in states[0].items()}
        generator = torch.Generator().manual_seed(217)
        chosen = changed["positions"][:, ::3]
        changed["ids"].scatter_(1, chosen, torch.randint(0, 120000, chosen.shape,
                                                       generator=generator))
        states.append(changed)
        torch.save(states, root / "states.pt")
    else:
        port = load_port(root)
        if args.backend == "vllm":
            report["cuda_smoke"] = cuda_smoke(port)
            print("CUDA_SMOKE", report["cuda_smoke"], flush=True)
            torch.cuda.empty_cache()
            save_json(output, report)
        model = port.LLaDAOfflineModel.from_local_checkpoint(
            checkpoint, backend=args.backend, fuse_projections=not args.separate_projections
        )
        report["parameter_count"] = sum(p.numel() for p in model.parameters())
        report["checkpoint_tensor_count"] = len(model.weight_mapping())
        report["vllm_base_commit"] = "ab5266769e702434a0968d47319d0731d8ccac35"
        bb = SimpleNamespace(mask_token_id=126336)
        states = torch.load(root / "states.pt", weights_only=True)
        engine = model
    for i, state in enumerate(states):
        ids, mask, pos = [state[k].cuda() for k in ("ids", "mask", "positions")]
        forward = engine.prepare(ids, mask, pos) if args.mode == "check" else engine
        summary, metrics = measure(forward, ids, mask, pos)
        path = root / f"reference-state-{i}.pt"
        if args.mode == "export":
            torch.save(summary, path)
        else:
            metrics["vs_pinned_hf"] = difference(torch.load(path, weights_only=True), summary)
        report["states"].append(metrics)
        print("STATE", i, metrics, flush=True)
        save_json(output, report)
        del summary
    selector, saved = selector_for(assets)
    ids, mask, pos = [states[0][k].cuda() for k in ("ids", "mask", "positions")]
    uniforms = torch.rand(8, 8, 256, generator=torch.Generator().manual_seed(991))
    for name in ("p2", "sota"):
        selector.load_state_dict(saved["model_state"])
        if name == "p2":
            selector.pointwise[-1].weight.zero_()
            selector.pointwise[-1].bias.zero_()
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = collect(bb, selector, ids, mask, pos, uniforms, answer_backbone=engine)
        elapsed = time.perf_counter() - start
        summary = rollout_summary(result)
        metrics = {"policy": name, "seconds": elapsed, "nfe": result.nfe}
        path = root / f"reference-rollout-{name}.pt"
        if args.mode == "export":
            torch.save(summary, path)
        else:
            ref = torch.load(path, weights_only=True)
            metrics.update(
                final_token_mismatches=int((summary["final_ids"] != ref["final_ids"]).sum()),
                ordered_action_mismatches=sum(int((a != b).sum()) for a, b in
                                             zip(summary["orders"], ref["orders"])),
                max_old_logp_error=float((summary["old_logp"] - ref["old_logp"]).abs().max()),
                max_feature_error=float((summary["features"] - ref["features"]).abs().max()),
                revision_match=torch.equal(summary["revisions"], ref["revisions"]),
            )
            metrics["strict_trajectory_parity"] = (
                metrics["final_token_mismatches"] == metrics["ordered_action_mismatches"] == 0
                and metrics["max_feature_error"] < 1e-4
                and metrics["max_old_logp_error"] < 0.002
                and metrics["revision_match"]
            )
            del ref
        report["rollouts"].append(metrics)
        print("ROLLOUT", metrics, flush=True)
        save_json(output, report)
        del summary, result
        gc.collect()
    report["status"] = "complete"
    report["finished_unix"] = time.time()
    save_json(output, report)


if __name__ == "__main__":
    main()
