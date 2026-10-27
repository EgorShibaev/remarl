"""Visibility and block schedules."""

import math


def kappa_fraction(kind: str, u: float) -> float:
    """P2 unmasking scheduler ``kappa``: monotone non-decreasing on ``u in [0,1]`` with
    ``kappa(0) == 0`` and ``kappa(1) == 1``.

    - ``linear``: ``u`` (unmask a constant fraction each step).
    - ``cosine``: ``sin(pi*u/2)`` (concave; unmask faster early, then slow down).
    - ``sqrt``:   ``sqrt(u)`` (concave; also front-loaded).
    """
    u = min(1.0, max(0.0, float(u)))
    if kind == "linear":
        return u
    if kind == "cosine":
        return math.sin(math.pi * u / 2.0)
    if kind == "sqrt":
        return math.sqrt(u)
    raise ValueError(f"Unknown kappa schedule kind: {kind}")


def topk_schedule(kind: str, step: int, total_steps: int, n_editable: int) -> int:
    """Number of positions to KEEP unmasked after 1-indexed ``step`` of ``total_steps``.

    ``K_s = ceil(kappa(s / total_steps) * n_editable)``, clamped to ``[1, n_editable]``
    and forced to ``n_editable`` on the final step (``kappa(1) == 1`` -> full unmask), so
    decoding always completes after ``total_steps`` iterations.
    """
    if n_editable <= 0:
        return 0
    total_steps = max(1, int(total_steps))
    step = max(1, min(int(step), total_steps))
    if step >= total_steps:
        return n_editable
    frac = kappa_fraction(kind, step / total_steps)
    k = math.ceil(frac * n_editable)
    return max(1, min(n_editable, k))


def make_blocks(n_editable: int, block_length: int) -> list[tuple[int, int]]:
    """Contiguous ``[start, end)`` index ranges over the editable region for semi-AR
    (block) decoding, left to right.

    ``block_length <= 0`` or ``>= n_editable`` yields a single full-region block, i.e.
    **full-diffusion** -- the P2 default (no behaviour change). Apple's recipe uses
    ``block_length == 32``: the planner then only acts on the current block while later
    blocks stay masked, and P2's top-k + remasking run *within* each block.
    """
    if n_editable <= 0:
        return []
    if block_length <= 0 or block_length >= n_editable:
        return [(0, n_editable)]
    return [(i, min(i + block_length, n_editable)) for i in range(0, n_editable, block_length)]


def steps_per_block(total_steps: int, n_blocks: int) -> int:
    """Split a total step budget ``S`` across ``n_blocks`` semi-AR blocks.

    Keeps ``S`` meaning ~total NFE regardless of block size: total forwards ``==
    steps_per_block * n_blocks ~= S``. Always ``>= 1`` so every block runs at least one
    plan/denoise step (``kappa(1) == 1`` still forces a full unmask of the block).
    """
    return max(1, round(int(total_steps) / max(1, int(n_blocks))))
