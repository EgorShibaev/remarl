"""RL environments: problems with a masked answer region + a verifier.

An ``Environment`` yields ``RLProblem``s. Each problem is a tokenized prompt followed
by a fixed-length **masked answer region** (those positions hold ``mask_token_id``);
decoding fills the region and the environment's verifier scores the decoded answer.

Environments are resolved by name through a registry so configs can list them. A tiny
in-memory ``toy`` environment (no tokenizer / no HuggingFace) is provided for tests and
CPU smoke runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from src.rl.verifiers import MathVerifier, Verifier


@dataclass
class RLProblem:
    input_ids: torch.Tensor  # [seq], answer region == mask_token_id
    attention_mask: torch.Tensor  # [seq]
    answer_span: torch.Tensor  # [answer_len] absolute positions
    y_star: object
    nfe_max: int


@dataclass
class Environment:
    name: str
    verifier: Verifier
    problems: list[RLProblem]
    # final_input_ids [seq] + problem -> decoded answer string for the verifier
    decode_answer: Callable[[torch.Tensor, RLProblem], str]


def build_masked_problem(
    prompt_ids: torch.Tensor,
    answer_len: int,
    mask_token_id: int,
    max_length: int,
    pad_token_id: int,
    y_star: object,
    nfe_max: int | None = None,
) -> RLProblem:
    """Prompt tokens + ``answer_len`` masked positions, padded to ``max_length``."""
    prompt_ids = prompt_ids.flatten().long()
    room = max_length - answer_len
    if room < 1:
        raise ValueError(f"answer_len {answer_len} leaves no room within max_length {max_length}")
    prompt_ids = prompt_ids[:room]
    answer = torch.full((answer_len,), mask_token_id, dtype=torch.long)
    body = torch.cat([prompt_ids, answer])
    span = torch.arange(prompt_ids.numel(), prompt_ids.numel() + answer_len, dtype=torch.long)
    attn = torch.ones(body.numel(), dtype=torch.long)
    if body.numel() < max_length:
        pad = torch.full((max_length - body.numel(),), pad_token_id, dtype=torch.long)
        body = torch.cat([body, pad])
        attn = torch.cat([attn, torch.zeros(pad.numel(), dtype=torch.long)])
    return RLProblem(
        input_ids=body,
        attention_mask=attn,
        answer_span=span,
        y_star=y_star,
        nfe_max=int(nfe_max if nfe_max is not None else answer_len),
    )


def decode_with_tokenizer(tokenizer) -> Callable[[torch.Tensor, RLProblem], str]:
    def _decode(final_input_ids: torch.Tensor, problem: RLProblem) -> str:
        ids = final_input_ids[problem.answer_span]
        return tokenizer.decode(ids.tolist(), skip_special_tokens=True)

    return _decode


# --------------------------------------------------------------------------------------
# Toy environment (no tokenizer / no datasets) for tests and CPU smoke runs.
# --------------------------------------------------------------------------------------
def build_toy_env(
    mask_token_id: int = 99,
    pad_token_id: int = 0,
    max_length: int = 16,
    answer_len: int = 4,
    num_problems: int = 4,
    **_: object,
) -> Environment:
    problems: list[RLProblem] = []
    for i in range(num_problems):
        prompt = torch.tensor([1, 2 + i, 3, 4])
        problems.append(
            build_masked_problem(
                prompt_ids=prompt,
                answer_len=answer_len,
                mask_token_id=mask_token_id,
                max_length=max_length,
                pad_token_id=pad_token_id,
                y_star=str(i),
            )
        )

    def _decode(final_input_ids: torch.Tensor, problem: RLProblem) -> str:
        # toy "answer" is the space-joined decoded ids of the answer region
        return " ".join(str(int(x)) for x in final_input_ids[problem.answer_span])

    return Environment(name="toy", verifier=MathVerifier(), problems=problems, decode_answer=_decode)


_BUILDERS: dict[str, Callable[..., Environment]] = {"toy": build_toy_env}


def register_environment(name: str, builder: Callable[..., Environment]) -> None:
    _BUILDERS[name] = builder


def available_environments() -> list[str]:
    return sorted(_BUILDERS)


def get_environment(name: str, **kwargs) -> Environment:
    if name not in _BUILDERS:
        raise KeyError(f"Unknown environment '{name}'. Available: {available_environments()}")
    return _BUILDERS[name](**kwargs)


# Real dataset environments register themselves on import (import-guarded inside).
from src.rl import datasets as _datasets  # noqa: E402,F401
