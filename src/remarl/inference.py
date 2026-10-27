"""Single-prompt generation through the same rollout used in training."""

from pathlib import Path
import torch
from .protocols.math import official, prompt
from .runtime import Generator, load_policy, load_tokenizer, configure_compute


def generate(args):
    configure_compute()
    policy, saved = load_policy(args.policy, args.device)
    tokenizer = load_tokenizer(args.model)
    text = args.prompt
    if args.task != "raw":
        api = official(Path(__file__).parent / "protocols/official")
        text = prompt(api, tokenizer, text, args.task)
    row = dict(prompt=text, prompt_ids=tokenizer(text, truncation=False)["input_ids"])
    length = policy.config.position_count
    if len(row["prompt_ids"]) + length > 8192:
        raise ValueError("Prompt exceeds supported context; it will not be truncated")
    generator = Generator(
        args.model, policy, args.device, args.backend, max(4096, len(row["prompt_ids"]) + length)
    )
    rng = torch.Generator().manual_seed(args.seed)
    uniforms = torch.stack([torch.rand((1, length), generator=rng) for _ in range(args.nfe)])
    result = generator.generate([row], uniforms, False)
    return dict(
        answer=tokenizer.decode(result["tokens"][0], skip_special_tokens=True),
        tokens=result["tokens"][0],
        nfe=args.nfe,
        seed=args.seed,
        policy_sha256=saved["policy_sha256"],
    )
