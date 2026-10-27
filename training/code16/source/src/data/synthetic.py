"""Synthetic structured-text generators with controlled dependency strength.

These give the cleanest causal test for the conflict-graph hypothesis (H2): as
nesting depth / coupling grows, an ideal conflict graph should help more. No
network or external data needed.

- ``generate_json_examples``: nested JSON with repeated keys, arrays, and
  cross-field constraints (e.g. a ``count`` field equal to an array length).
- ``generate_equation_examples``: arithmetic expressions with parentheses,
  repeated variables, and equality constraints.
"""

from __future__ import annotations

import random


_KEYS = ["id", "name", "type", "count", "value", "items", "meta", "flag", "size", "tag"]
_WORDS = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "theta", "kappa"]


def _rand_scalar(rng: random.Random) -> str:
    choice = rng.random()
    if choice < 0.4:
        return str(rng.randint(0, 99))
    if choice < 0.7:
        return f'"{rng.choice(_WORDS)}"'
    return rng.choice(["true", "false", "null"])


def _gen_object(rng: random.Random, depth: int, max_depth: int) -> str:
    n_keys = rng.randint(2, 4)
    keys = rng.sample(_KEYS, k=min(n_keys, len(_KEYS)))
    parts: list[str] = []
    array_len: int | None = None
    for key in keys:
        if key == "items" and depth < max_depth:
            length = rng.randint(1, 3)
            array_len = length
            elems = [_gen_value(rng, depth + 1, max_depth) for _ in range(length)]
            parts.append(f'"items": [{", ".join(elems)}]')
        elif key == "count" and array_len is not None:
            # cross-field constraint: count == len(items)
            parts.append(f'"count": {array_len}')
        elif key in ("meta",) and depth < max_depth:
            parts.append(f'"meta": {_gen_object(rng, depth + 1, max_depth)}')
        else:
            parts.append(f'"{key}": {_rand_scalar(rng)}')
    return "{" + ", ".join(parts) + "}"


def _gen_value(rng: random.Random, depth: int, max_depth: int) -> str:
    if depth < max_depth and rng.random() < 0.6:
        return _gen_object(rng, depth, max_depth)
    return _rand_scalar(rng)


def generate_json_examples(n: int, depth: int = 2, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    return [_gen_object(rng, 0, depth) for _ in range(n)]


def _cross_field(rng: random.Random, depth: int) -> str:
    """type value determines which keys must appear (long-range agreement)."""
    schema = {"user": ["name", "email"], "item": ["sku", "price"], "event": ["ts", "kind"]}
    t = rng.choice(list(schema))
    parts = [f'"type": "{t}"']
    for k in schema[t]:
        parts.append(f'"{k}": {_rand_scalar(rng)}')
    if depth > 1 and rng.random() < 0.7:
        parts.append(f'"child": {_cross_field(rng, depth - 1)}')
    return "{" + ", ".join(parts) + "}"


def _repeated_ref(rng: random.Random, depth: int) -> str:
    """a value defined once must reappear deeper (long-range copy)."""
    ref = rng.randint(100, 999)
    inner = f'"ref": {ref}'
    node = "{" + inner + "}"
    for _ in range(max(1, depth - 1)):
        node = '{"id": ' + str(ref) + ', "data": ' + node + "}"
    return '{"id": ' + str(ref) + ', "body": ' + node + "}"


def _schema(rng: random.Random, depth: int) -> str:
    """top-level schema key conditions the nested grammar."""
    mode = rng.choice(["list", "map", "leaf"])
    if mode == "list" and depth > 1:
        elems = [_schema(rng, depth - 1) for _ in range(rng.randint(1, 3))]
        body = "[" + ", ".join(elems) + "]"
    elif mode == "map" and depth > 1:
        body = "{" + ", ".join(f'"k{i}": {_schema(rng, depth - 1)}' for i in range(rng.randint(1, 2))) + "}"
    else:
        body = _rand_scalar(rng)
    return '{"schema": "' + mode + '", "value": ' + body + "}"


def _distractor(rng: random.Random, depth: int) -> str:
    """locally similar fields; only the same-subtree pair truly couples."""
    val = rng.randint(0, 9)
    subtree = '{"x": ' + str(val) + ', "y": ' + str(val) + "}"  # x==y inside subtree
    distractors = ", ".join(f'"x": {rng.randint(0, 9)}' for _ in range(max(1, depth)))
    return "{" + distractors + ', "sub": ' + subtree + "}"


def generate_constrained_json_examples(
    n: int, depth: int = 4, constraint: str = "cross_field", seed: int = 0
) -> list[str]:
    """Higher-order constrained JSON for the H2 stress test: conflicts that are
    not visible from a token pair alone (cross-field / repeated-ref / schema /
    distractor)."""
    rng = random.Random(seed)
    gens = {
        "cross_field": _cross_field,
        "repeated_ref": _repeated_ref,
        "schema_conditioned": _schema,
        "distractor": _distractor,
    }
    if constraint not in gens:
        raise ValueError(f"Unknown constraint: {constraint}")
    fn = gens[constraint]
    return [fn(rng, depth) for _ in range(n)]


def generate_equation_examples(n: int, coupling: int = 2, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    out: list[str] = []
    variables = ["x", "y", "z", "w", "u", "v"]
    for _ in range(n):
        used = variables[: max(1, min(coupling, len(variables)))]
        terms = []
        for var in used:
            coef = rng.randint(1, 9)
            terms.append(f"{coef} * {var}")
        lhs = " + ".join(terms)
        # repeated-variable + parenthesized coupling
        extra = rng.choice(used)
        rhs = rng.randint(1, 50)
        out.append(f"( {lhs} ) + {extra} = {rhs}")
    return out
