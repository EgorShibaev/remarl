"""Command-line interface for data, training, inference, and evaluation."""

import argparse
import json
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(prog="remarl", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("download", help="Download pinned model, data, or released policies")
    fetch.add_argument(
        "asset", choices=("backbone", "checkpoints", "math-data", "code-data", "baseline-sources")
    )
    fetch.add_argument("--output", type=Path, required=True)
    prep = sub.add_parser("prepare", help="Build the exact task prompts and training cohort")
    prep.add_argument("domain", choices=("math", "code"))
    prep.add_argument("--raw-data", required=True, type=Path)
    prep.add_argument("--model", required=True, type=Path)
    prep.add_argument("--output", required=True, type=Path)
    prep.add_argument("--cohort", choices=("code8", "code16"), default="code16")
    train = sub.add_parser("train", help="Train a policy from scratch with GRPO")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--data", required=True, type=Path)
    train.add_argument("--model", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--devices", default="cuda:0", help="Comma-separated device names")
    train.add_argument("--backend", choices=("vllm", "torch"), default="vllm")
    train.add_argument("--resume", type=Path)
    ev = sub.add_parser("evaluate", help="Evaluate a policy on complete benchmarks")
    ev.add_argument("--data", required=True, type=Path)
    ev.add_argument("--model", required=True, type=Path)
    method = ev.add_mutually_exclusive_group(required=True)
    method.add_argument("--policy", type=Path)
    method.add_argument("--p2-prime", action="store_true")
    ev.add_argument("--nfe", required=True, type=int, choices=(4, 8, 16, 32, 64, 128, 256))
    ev.add_argument("--seeds", default="101,202,303")
    ev.add_argument("--split")
    ev.add_argument("--output", required=True, type=Path)
    ev.add_argument("--device", default="cuda:0")
    ev.add_argument("--backend", choices=("vllm", "torch"), default="vllm")
    ev.add_argument("--batch-size", type=int, default=8)
    ev.add_argument(
        "--temperature", type=float, default=1.0, help="P2-prime position temperature; 0 is greedy"
    )
    ev.add_argument(
        "--retention-bonus", type=float, default=0.0, help="P2-prime visible-token bonus"
    )
    gen = sub.add_parser("generate", help="Generate one answer with a released policy")
    gen.add_argument("--model", required=True, type=Path)
    gen.add_argument("--policy", required=True, type=Path)
    gen.add_argument("--nfe", required=True, type=int)
    gen.add_argument("--task", choices=("gsm8k", "math", "raw"), default="gsm8k")
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--seed", type=int, default=101)
    gen.add_argument("--device", default="cuda:0")
    gen.add_argument("--backend", choices=("vllm", "torch"), default="vllm")
    verify = sub.add_parser("verify", help="Verify all released checkpoint hashes")
    verify.add_argument("directory", type=Path)
    baseline = sub.add_parser(
        "baseline", help="Evaluate a baseline with measured model-call counts"
    )
    for name in ("config", "data", "model", "output"):
        baseline.add_argument("--" + name, required=True, type=Path)
    baseline.add_argument("--device", default="cuda:0")
    baseline.add_argument("--backend", choices=("vllm", "torch"), default="vllm")
    baseline.add_argument("--sources", type=Path, default=Path("assets/baselines"))
    sub.add_parser("check-sandbox", help="Build and test the Linux code-execution sandbox")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "download":
        from .downloads import download

        download(args.asset, args.output)
    elif args.command == "verify":
        from .downloads import verify

        print(json.dumps(verify(args.directory), indent=2))
    elif args.command == "prepare":
        from .data.prepare import prepare

        prepare(args.domain, args.raw_data, args.model, args.output, args.cohort)
    elif args.command == "baseline":
        import os

        os.environ["REMARL_BASELINE_SOURCES"] = str(args.sources.resolve())
        from .baselines.evaluate import run

        print(
            json.dumps(
                run(args.config, args.data, args.model, args.output, args.device, args.backend),
                indent=2,
            )
        )
    elif args.command == "train":
        from .training import train

        train(
            args.config,
            args.data,
            args.model,
            args.output,
            args.devices.split(","),
            args.backend,
            args.resume,
        )
    elif args.command == "evaluate":
        from .evaluation import run_evaluation

        result = run_evaluation(
            args.data,
            args.model,
            args.output,
            args.policy,
            args.nfe,
            [int(s) for s in args.seeds.split(",")],
            args.device,
            args.backend,
            args.split,
            args.batch_size,
            args.temperature,
            args.retention_bonus,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "check-sandbox":
        from .protocols.sandbox import setup

        print(json.dumps(setup(), indent=2))
    elif args.command == "generate":
        from .inference import generate

        print(json.dumps(generate(args), indent=2))


if __name__ == "__main__":
    main()
