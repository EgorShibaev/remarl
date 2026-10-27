from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TokenizedExample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    text: str


class TokenizedTextDataset(Dataset[TokenizedExample]):
    def __init__(self, examples: list[TokenizedExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TokenizedExample:
        return self.examples[index]


def load_hf_text_dataset(
    tokenizer: Any,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    max_examples: int,
    max_length: int,
) -> TokenizedTextDataset:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets to load HuggingFace datasets") from exc

    dataset = load_dataset(dataset_name, dataset_config, split=split)
    examples: list[TokenizedExample] = []
    for row in dataset:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        encoded = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        attention_count = int(encoded["attention_mask"].sum().item())
        if attention_count < 8:
            continue
        examples.append(
            TokenizedExample(
                input_ids=encoded["input_ids"].squeeze(0),
                attention_mask=encoded["attention_mask"].squeeze(0),
                text=text,
            )
        )
        if len(examples) >= max_examples:
            break
    return TokenizedTextDataset(examples)


def _tokenize_texts(
    tokenizer: Any, texts: list[str], max_examples: int, max_length: int
) -> TokenizedTextDataset:
    examples: list[TokenizedExample] = []
    for text in texts:
        text = str(text).strip()
        if not text:
            continue
        encoded = tokenizer(
            text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        if int(encoded["attention_mask"].sum().item()) < 8:
            continue
        examples.append(
            TokenizedExample(
                input_ids=encoded["input_ids"].squeeze(0),
                attention_mask=encoded["attention_mask"].squeeze(0),
                text=text,
            )
        )
        if len(examples) >= max_examples:
            break
    return TokenizedTextDataset(examples)


# default HF text field per dataset family
_TEXT_FIELDS = {
    "Salesforce/wikitext": "text",
    "wikitext": "text",
    "mbpp": "code",
    "google-research-datasets/mbpp": "code",
    "code_search_net": "whole_func_string",
    "spider": "query",
    "xlangai/spider": "query",
    "code_search_net": "func_code_string",
    "code-search-net/code_search_net": "func_code_string",
}


def load_text_dataset(
    tokenizer: Any,
    data_cfg: dict[str, Any],
    split: str,
    max_examples: int,
    max_length: int,
) -> TokenizedTextDataset:
    """Unified loader: synthetic generators + arbitrary HF text datasets.

    ``data_cfg`` keys: ``dataset_name``, ``dataset_config``, ``text_field``,
    and for synthetic sets ``synthetic_depth`` / ``synthetic_coupling``.
    """
    name = str(data_cfg.get("dataset_name", "Salesforce/wikitext"))
    split_seed = {"train": 0, "validation": 1, "dev": 1, "test": 2}.get(split, abs(hash(split)) % 1000)

    if name == "synthetic_json":
        from src.data.synthetic import generate_json_examples

        texts = generate_json_examples(
            n=max_examples * 2, depth=int(data_cfg.get("synthetic_depth", 2)), seed=split_seed
        )
        return _tokenize_texts(tokenizer, texts, max_examples, max_length)
    if name == "synthetic_equations":
        from src.data.synthetic import generate_equation_examples

        texts = generate_equation_examples(
            n=max_examples * 2, coupling=int(data_cfg.get("synthetic_coupling", 2)), seed=split_seed
        )
        return _tokenize_texts(tokenizer, texts, max_examples, max_length)
    if name == "synthetic_constrained_json":
        from src.data.synthetic import generate_constrained_json_examples

        texts = generate_constrained_json_examples(
            n=max_examples * 2,
            depth=int(data_cfg.get("synthetic_depth", 4)),
            constraint=str(data_cfg.get("synthetic_constraint", "cross_field")),
            seed=split_seed,
        )
        return _tokenize_texts(tokenizer, texts, max_examples, max_length)

    # HuggingFace dataset path
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets to load HuggingFace datasets") from exc
    dataset = load_dataset(name, data_cfg.get("dataset_config"), split=split)
    text_field = data_cfg.get("text_field") or _TEXT_FIELDS.get(name, "text")
    texts = [row.get(text_field, "") for row in dataset]
    return _tokenize_texts(tokenizer, texts, max_examples, max_length)


def char_class_token_ids(tokenizer: Any, kind: str) -> set[int]:
    """Token ids whose decoded text is alphabetic ("alpha") or punctuation ("punct").

    Used for code masking regimes: identifier-only ~ "alpha", syntax-critical ~
    "punct". Heuristic over the tokenizer vocabulary (BERT word-pieces).
    """
    ids: set[int] = set()
    vocab = tokenizer.get_vocab()
    for token, tid in vocab.items():
        piece = token.lstrip("Ġ▁##").strip()
        if not piece:
            continue
        is_alpha = any(c.isalpha() for c in piece) and all(c.isalnum() or c == "_" for c in piece)
        is_punct = all(not c.isalnum() for c in piece)
        if kind == "alpha" and is_alpha:
            ids.add(int(tid))
        elif kind == "punct" and is_punct:
            ids.add(int(tid))
    return ids


def resolve_masking(data_cfg: dict[str, Any], tokenizer: Any) -> tuple[str, set[int] | None]:
    """Map a ``data.masking`` config block to (strategy, restrict_token_ids).

    strategy options: random (default), span, identifier, syntax, mixed.
    identifier/syntax restrict masking to alpha/punct tokens (code regimes).
    """
    masking = (data_cfg or {}).get("masking", {}) or {}
    name = str(masking.get("strategy", "random"))
    if name == "span":
        return "span", None
    if name == "identifier":
        return "random", char_class_token_ids(tokenizer, "alpha")
    if name == "syntax":
        return "random", char_class_token_ids(tokenizer, "punct")
    return "random", None  # random / mixed


def collate_tokenized(examples: list[TokenizedExample]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "input_ids": torch.stack([example.input_ids for example in examples], dim=0),
        "attention_mask": torch.stack([example.attention_mask for example in examples], dim=0),
        "text": [example.text for example in examples],
    }

