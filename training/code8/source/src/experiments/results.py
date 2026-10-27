"""Load sweep results into tidy pandas DataFrames for notebooks / analysis.

Thin layer over ``aggregate.collect_cell`` that flattens every eval run across a
sweep's cells into one DataFrame (one row per cell x decoder x mask_ratio), plus
helpers to read training curves. Keeps notebooks short.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.experiments.aggregate import collect_cell

SWEEPS_ROOT = Path("outputs/sweeps")


def list_sweeps(root: str | Path = SWEEPS_ROOT) -> list[str]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p.is_dir() or p.is_symlink()))


def load_sweep_df(sweep: str, root: str | Path = SWEEPS_ROOT) -> pd.DataFrame:
    """One row per (cell, decoder/method, mask_ratio). Sweep params are expanded
    into their own columns (e.g. ``synthetic_depth``, ``masking.strategy``).
    """
    sweep_dir = Path(root) / sweep
    rows: list[dict] = []
    if not sweep_dir.exists():
        return pd.DataFrame()
    for cell_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir() or p.is_symlink()):
        for row in collect_cell(cell_dir):
            params = json.loads(row.get("params", "{}") or "{}")
            for key, value in params.items():
                row[key.split(".")[-1]] = value
            row["sweep"] = sweep
            rows.append(row)
    df = pd.DataFrame(rows)
    return df


def load_all_sweeps_df(root: str | Path = SWEEPS_ROOT) -> pd.DataFrame:
    frames = [load_sweep_df(s, root) for s in list_sweeps(root)]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_train_curves(sweep: str, root: str | Path = SWEEPS_ROOT) -> pd.DataFrame:
    """Per-epoch training curves (train/dev loss, dev AUC) for each cell."""
    sweep_dir = Path(root) / sweep
    rows: list[dict] = []
    if not sweep_dir.exists():
        return pd.DataFrame()
    for cell_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir() or p.is_symlink()):
        for metrics in (cell_dir / "runs").glob("train-conflict-*/metrics.jsonl"):
            for line in metrics.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                rec["cell"] = cell_dir.name
                rows.append(rec)
    return pd.DataFrame(rows)


def method_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std per method across the sweep (the decision table)."""
    if df.empty:
        return df
    agg = (
        df.groupby("method")[["accuracy", "steps", "latency", "oracle_conflict", "conflict_auc"]]
        .agg(["mean", "std"])
        .round(4)
    )
    return agg
