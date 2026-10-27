from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def deep_get(config: dict[str, Any], key: str, default: Any = None) -> Any:
    current: Any = config
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def set_dotted(config: dict[str, Any], dotted: str, value: Any) -> None:
    """Set a nested key by dotted path, creating intermediate dicts."""
    parts = dotted.split(".")
    cur = config
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def parse_scalar(text: str) -> Any:
    """Parse a CLI scalar like an int/float/bool/null/str via YAML rules."""
    return yaml.safe_load(text)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``; returns a new dict.

    Nested dicts are merged key by key. Non-dict values in ``override`` always
    replace the value in ``base``. Lists are treated as scalars (replaced, not
    extended) to keep semantics predictable for configs.
    """
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

