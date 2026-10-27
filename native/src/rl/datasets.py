"""HuggingFace-backed RL environments: GSM8K, MATH, Big-Math, Reasoning Gym (math,
exact-answer reward) and MBPP, APPS (code, unit-test pass-rate reward).

Each builder tokenizes a prompt, appends a masked answer region, and binds the matching
verifier. ``datasets`` is imported lazily inside each builder so this module imports
cleanly without HuggingFace installed; a missing dataset raises a clear error at build
time rather than at import. Builders register themselves into the environment registry.
"""

from __future__ import annotations

import json
import random
from typing import Any, Callable

import torch

from src.rl.environments import (
    Environment,
    build_masked_problem,
    decode_with_tokenizer,
    register_environment,
)
from src.rl.verifiers import CodeVerifier, MathVerifier


def _load_hf(name: str, config: str | None, split: str, revision: str | None = None):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only without datasets
        raise ImportError("Install the `datasets` package to use real RL environments") from exc
    return load_dataset(name, config, split=split, revision=revision)


def _make_env(
    name: str,
    rows: Any,
    tokenizer,
    row_to_prompt_answer: Callable[[dict], tuple[str, object] | None],
    verifier,
    *,
    answer_len: int,
    max_length: int,
    max_examples: int,
    mask_token_id: int,
    pad_token_id: int,
    nfe_max: int | None,
) -> Environment:
    problems = []
    for row in rows:
        pa = row_to_prompt_answer(row)
        if pa is None:
            continue
        prompt_text, y_star = pa
        enc = tokenizer(prompt_text, truncation=True, max_length=max_length - answer_len, return_tensors="pt")
        problems.append(
            build_masked_problem(
                prompt_ids=enc["input_ids"].squeeze(0),
                answer_len=answer_len,
                mask_token_id=mask_token_id,
                max_length=max_length,
                pad_token_id=pad_token_id,
                y_star=y_star,
                nfe_max=nfe_max,
            )
        )
        if len(problems) >= max_examples:
            break
    return Environment(name=name, verifier=verifier, problems=problems, decode_answer=decode_with_tokenizer(tokenizer))


def _split_name(split: str, val: str = "test") -> str:
    return val if split in ("val", "validation", "dev") else split


# --------------------------------------------------------------------------------------
# Math environments (exact-answer reward).
# --------------------------------------------------------------------------------------
def build_gsm8k(tokenizer, split="train", max_examples=1000, answer_len=48, max_length=256,
                mask_token_id=126336, pad_token_id=0, nfe_max=None, **_) -> Environment:
    rows = _load_hf("openai/gsm8k", "main", _split_name(split))

    def row_to(row: dict):
        return f"Question: {row['question']}\nAnswer:", row["answer"]

    return _make_env("gsm8k", rows, tokenizer, row_to, MathVerifier(), answer_len=answer_len,
                     max_length=max_length, max_examples=max_examples, mask_token_id=mask_token_id,
                     pad_token_id=pad_token_id, nfe_max=nfe_max)


def build_math(tokenizer, split="train", max_examples=1000, answer_len=64, max_length=512,
               mask_token_id=126336, pad_token_id=0, nfe_max=None, **_) -> Environment:
    # The original hendrycks/competition_math is script-based (unsupported by
    # datasets>=4). Use parquet mirrors: full MATH train for training, MATH-500 for eval.
    if split in ("val", "validation", "dev", "test"):
        rows = _load_hf("HuggingFaceH4/MATH-500", None, "test")

        def row_to(row: dict):
            return f"Problem: {row['problem']}\nSolution:", row["answer"]
    else:
        rows = _load_hf("qwedsacf/competition_math", None, "train")

        def row_to(row: dict):
            return f"Problem: {row['problem']}\nSolution:", row["solution"]

    return _make_env("math", rows, tokenizer, row_to, MathVerifier(), answer_len=answer_len,
                     max_length=max_length, max_examples=max_examples, mask_token_id=mask_token_id,
                     pad_token_id=pad_token_id, nfe_max=nfe_max)


def build_big_math(tokenizer, split="train", max_examples=1000, answer_len=48, max_length=512,
                   mask_token_id=126336, pad_token_id=0, nfe_max=None, **_) -> Environment:
    # The canonical ``SynthLabsAI/Big-Math-RL-Verified`` is gated (needs an auth token).
    # ``ZhuofengLi/Big-Math-RL-Verified`` is a faithful, ungated full mirror with the same
    # 251k rows and schema (``problem`` / ``answer`` / ``source`` / ``llama8b_solve_rate``),
    # so it loads offline once cached.
    rows = _load_hf("ZhuofengLi/Big-Math-RL-Verified", None, _split_name(split, "train"))

    def row_to(row: dict):
        return f"Problem: {row['problem']}\nAnswer:", row.get("answer", "")

    return _make_env("big_math", rows, tokenizer, row_to, MathVerifier(), answer_len=answer_len,
                     max_length=max_length, max_examples=max_examples, mask_token_id=mask_token_id,
                     pad_token_id=pad_token_id, nfe_max=nfe_max)


def build_reasoning_gym(tokenizer, split="train", max_examples=1000, answer_len=48, max_length=512,
                        mask_token_id=126336, pad_token_id=0, nfe_max=None, dataset="chain_sum", **_) -> Environment:
    # Reasoning Gym is a generator library; fall back to a HF mirror if present.
    rows = _load_hf("reasoning-gym/reasoning-gym", dataset, _split_name(split, "train"))

    def row_to(row: dict):
        return f"{row.get('question', row.get('prompt', ''))}\nAnswer:", row.get("answer", "")

    return _make_env("reasoning_gym", rows, tokenizer, row_to, MathVerifier(), answer_len=answer_len,
                     max_length=max_length, max_examples=max_examples, mask_token_id=mask_token_id,
                     pad_token_id=pad_token_id, nfe_max=nfe_max)


# --------------------------------------------------------------------------------------
# Code environments (unit-test pass-rate reward).
# --------------------------------------------------------------------------------------
def build_mbpp(tokenizer, split="train", max_examples=500, answer_len=96, max_length=512,
               mask_token_id=126336, pad_token_id=0, nfe_max=None, timeout=5.0, **_) -> Environment:
    rows = _load_hf("google-research-datasets/mbpp", None, _split_name(split))

    def row_to(row: dict):
        prompt = f"# {row['text']}\n# {row['test_list'][0]}\n"
        return prompt, {"asserts": list(row["test_list"])}

    return _make_env("mbpp", rows, tokenizer, row_to, CodeVerifier(timeout=timeout), answer_len=answer_len,
                     max_length=max_length, max_examples=max_examples, mask_token_id=mask_token_id,
                     pad_token_id=pad_token_id, nfe_max=nfe_max)


def build_apps(tokenizer, split="train", max_examples=300, answer_len=128, max_length=1024,
               mask_token_id=126336, pad_token_id=0, nfe_max=None, timeout=10.0, **_) -> Environment:
    # ``codeparrot/apps`` ships a loading script, which ``datasets>=4`` refuses to run.
    # The Hub auto-converts every dataset to parquet under the ``refs/convert/parquet``
    # ref; load that revision (default config = all difficulties) so it works offline
    # once cached. Columns match the script output (question / solutions / input_output).
    rows = _load_hf("codeparrot/apps", None, _split_name(split), revision="refs/convert/parquet")

    def row_to(row: dict):
        try:
            io = json.loads(row["input_output"]) if row.get("input_output") else {}
        except (json.JSONDecodeError, TypeError):
            return None
        inputs, outputs = io.get("inputs", []), io.get("outputs", [])
        cases = [(i, o) for i, o in zip(inputs, outputs)]
        if not cases:
            return None
        return f"{row['question']}\n", {"io": cases}

    return _make_env("apps", rows, tokenizer, row_to, CodeVerifier(timeout=timeout), answer_len=answer_len,
                     max_length=max_length, max_examples=max_examples, mask_token_id=mask_token_id,
                     pad_token_id=pad_token_id, nfe_max=nfe_max)


# --------------------------------------------------------------------------------------
# Synthetic arithmetic (no dataset download). Difficulty (digits / op) is tunable so the
# frozen base model can be put in the high-signal ~50%-solve regime that GRPO needs.
# --------------------------------------------------------------------------------------
def build_arith(tokenizer, split="train", num_problems=2000, max_examples=None,
                digits=2, op="+", answer_len=8, max_length=64,
                mask_token_id=126336, pad_token_id=0, nfe_max=None, seed=0, **_) -> Environment:
    n = int(max_examples) if max_examples else int(num_problems)
    # distinct stream for train vs eval so eval problems are unseen
    rng = random.Random(int(seed) + (0 if split == "train" else 10_000))
    lo, hi = 10 ** (int(digits) - 1), 10 ** int(digits) - 1
    problems = []
    for _ in range(n):
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        if op == "-":
            a, b = max(a, b), min(a, b)
            res = a - b
        elif op == "*":
            res = a * b
        else:
            op = "+"
            res = a + b
        enc = tokenizer(f"{a} {op} {b} = ", truncation=True, max_length=max_length - answer_len, return_tensors="pt")
        problems.append(
            build_masked_problem(
                prompt_ids=enc["input_ids"].squeeze(0), answer_len=answer_len, mask_token_id=mask_token_id,
                max_length=max_length, pad_token_id=pad_token_id, y_star=str(res), nfe_max=nfe_max,
            )
        )
    return Environment("arith", MathVerifier(), problems, decode_with_tokenizer(tokenizer))


register_environment("arith", build_arith)


# --------------------------------------------------------------------------------------
# Dataset mixture (Apple recipe: one policy trained on a GSM8K + MATH mixture, ~15k
# samples, sampled proportionally, then transferred). Concatenates the sub-environments'
# problems (each already tokenized to the same answer_len / max_length) and shuffles.
# --------------------------------------------------------------------------------------
_MIX_BUILDERS: dict[str, Callable[..., Environment]] = {
    "gsm8k": build_gsm8k, "math": build_math, "big_math": build_big_math,
    "reasoning_gym": build_reasoning_gym, "mbpp": build_mbpp, "apps": build_apps,
}


def build_mixture(tokenizer, split="train", components=("gsm8k", "math"), weights=None,
                  max_examples=15000, answer_len=256, max_length=512, mask_token_id=126336,
                  pad_token_id=0, nfe_max=None, seed=0, **_) -> Environment:
    comps = list(components)
    if not comps:
        raise ValueError("mixture needs at least one component dataset")
    weights = list(weights) if weights else [1.0] * len(comps)
    if len(weights) != len(comps):
        raise ValueError("mixture `weights` must match `components`")
    total_w = float(sum(weights)) or 1.0
    problems: list[RLProblem] = []
    for name, w in zip(comps, weights):
        if name not in _MIX_BUILDERS:
            raise KeyError(f"mixture component '{name}' is not a known dataset")
        share = max(1, int(round(int(max_examples) * (float(w) / total_w))))
        sub = _MIX_BUILDERS[name](
            tokenizer=tokenizer, split=split, max_examples=share, answer_len=answer_len,
            max_length=max_length, mask_token_id=mask_token_id, pad_token_id=pad_token_id,
            nfe_max=nfe_max,
        )
        problems.extend(sub.problems)
    random.Random(int(seed)).shuffle(problems)
    # GSM8K/MATH/Big-Math all use exact-answer matching, so a single MathVerifier fits.
    return Environment("mixture", MathVerifier(), problems, decode_with_tokenizer(tokenizer))


for _name, _builder in {**_MIX_BUILDERS, "mixture": build_mixture}.items():
    register_environment(_name, _builder)
