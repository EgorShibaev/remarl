"""Speed-quality curve helpers for the learned-vs-confidence comparison.

A *point* is ``(speed, accuracy)``; a *curve* for fixed-budget queries is a list of
``(nfe, accuracy)``. ``speed`` normalizes NFE against the conservative/aggressive
extremes so both methods live on the same axis.
"""

from __future__ import annotations

_EPS = 1e-8


def speed_from_nfe(nfe: float, nfe_slow: float, nfe_fast: float) -> float:
    """``(NFE_slow - NFE) / (NFE_slow - NFE_fast)`` -> ~0 (slow) .. ~1 (fast)."""
    return (nfe_slow - nfe) / (nfe_slow - nfe_fast + _EPS)


def pareto_filter(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Keep non-dominated ``(speed, acc)`` points (higher speed AND higher acc is better)."""
    kept: list[tuple[float, float]] = []
    for i, (s_a, a_a) in enumerate(points):
        dominated = False
        for j, (s_b, a_b) in enumerate(points):
            if i == j:
                continue
            if s_b >= s_a and a_b >= a_a and (s_b > s_a or a_b > a_a):
                dominated = True
                break
        if not dominated:
            kept.append((s_a, a_a))
    return kept


def _best_acc_at_speed(points: list[tuple[float, float]], s: float) -> float:
    """Best accuracy achievable at speed >= s (0 if the method can't reach speed s)."""
    vals = [a for sp, a in points if sp >= s - 1e-9]
    return max(vals) if vals else 0.0


def auc_over_speed(points: list[tuple[float, float]], num: int = 201) -> float:
    """Area under accuracy-vs-speed on the COMMON [0,1] speed axis.

    For each speed level s in [0,1], the achievable accuracy is the best accuracy
    among points with speed >= s (a faster method covers more of the axis). This is
    the fair speed-quality AUC: a method that is uniformly faster AND more accurate
    scores higher. (Plain ``auc`` below integrates only a method's own speed span and
    so under-credits a uniformly-fast method -- kept for the raw-curve helper/tests.)
    """
    if not points:
        return 0.0
    grid = [i / (num - 1) for i in range(num)]
    area = 0.0
    prev_s, prev_a = grid[0], _best_acc_at_speed(points, grid[0])
    for s in grid[1:]:
        a = _best_acc_at_speed(points, s)
        area += (s - prev_s) * (a + prev_a) / 2.0
        prev_s, prev_a = s, a
    return area


def auc(points: list[tuple[float, float]]) -> float:
    """Trapezoidal area under the speed-quality curve (sorted by speed)."""
    pts = sorted(set(points))
    if len(pts) < 2:
        return 0.0
    area = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0
    return area


def best_acc_at_nfe(curve: list[tuple[float, float]], budget: float) -> float | None:
    """Best accuracy achievable within ``NFE <= budget`` (None if nothing qualifies)."""
    eligible = [acc for nfe, acc in curve if nfe <= budget]
    return max(eligible) if eligible else None


def min_nfe_at_acc(curve: list[tuple[float, float]], target: float) -> float | None:
    """Smallest NFE reaching ``accuracy >= target`` (None if unreachable)."""
    eligible = [nfe for nfe, acc in curve if acc >= target]
    return min(eligible) if eligible else None


def gain_at_nfe(
    learned: list[tuple[float, float]],
    confidence: list[tuple[float, float]],
    budgets: list[float],
) -> dict[float, float | None]:
    out: dict[float, float | None] = {}
    for b in budgets:
        la, ca = best_acc_at_nfe(learned, b), best_acc_at_nfe(confidence, b)
        out[b] = (la - ca) if (la is not None and ca is not None) else None
    return out


def speedup_at_acc(
    confidence: list[tuple[float, float]],
    learned: list[tuple[float, float]],
    targets: list[float],
) -> dict[float, float | None]:
    out: dict[float, float | None] = {}
    for q in targets:
        cn, ln = min_nfe_at_acc(confidence, q), min_nfe_at_acc(learned, q)
        out[q] = (cn / ln) if (cn is not None and ln not in (None, 0)) else None
    return out
