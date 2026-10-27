"""Aggregate a sweep into a decision table, frontier, and Phase 11 verdict.

Scans ``outputs/sweeps/<name>/<cell>/`` for:
- ``runs/eval-<decoder>-*/summary.json`` (recon accuracy, steps, latency, oracle
  & predicted conflict rate; single- or multi-ratio form)
- ``runs/eval-conflict-labels-*/conflict_label_metrics.json`` (AUC/AP/F1)
- ``cell_config.json`` (cell params incl. model.type + seed)

"method" unifies decoder + trained model: decoder ``gnn`` is reported as the
trained model type (``gnn`` or ``pairwise_mlp``); ``confidence`` / ``attention``
/ ``oracle`` map to themselves. Outputs ``report.json`` + ``report.csv`` +
``frontier.csv`` and prints the decision table (mean +/- std across seeds),
per-seed GNN-minus-pairwise deltas, and matched-accuracy comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path
from typing import Any

DECODER_DTYPES = ("confidence", "gnn", "attention", "oracle")
METHOD_ORDER = ("confidence", "attention", "pairwise_mlp", "gnn", "oracle")


def _latest(paths: list[Path]) -> Path | None:
    paths = sorted(paths)
    return paths[-1] if paths else None


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return summary["sweep"] if "sweep" in summary else [summary]


def collect_cell(cell_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cfg_path = cell_dir / "cell_config.json"
    params: dict[str, Any] = {}
    model_type = None
    if cfg_path.exists():
        cfg = _load_json(cfg_path)
        params = (cfg.get("cell", {}) or {}).get("params", {})
        model_type = (cfg.get("model", {}) or {}).get("type")
    runs = cell_dir / "runs"
    if not runs.exists():
        return rows
    label_metrics: dict[str, Any] = {}
    cl = _latest(list(runs.glob("eval-conflict-labels-*/conflict_label_metrics.json")))
    if cl is not None:
        label_metrics = _load_json(cl)
    seed = params.get("seed")
    for decoder in DECODER_DTYPES:
        summ = _latest(list(runs.glob(f"eval-{decoder}-*/summary.json")))
        if summ is None:
            continue
        method = model_type if decoder == "gnn" else decoder
        for entry in _summary_rows(_load_json(summ)):
            rows.append(
                {
                    "cell": cell_dir.name,
                    "method": method,
                    "decoder": decoder,
                    "model_type": model_type,
                    "seed": seed,
                    "params": json.dumps(params, sort_keys=True),
                    "mask_ratio": entry.get("mask_ratio"),
                    "accuracy": entry.get("mean_accuracy"),
                    "steps": entry.get("mean_steps"),
                    "latency": entry.get("mean_latency"),
                    "oracle_conflict": entry.get("mean_selected_conflict_rate_oracle"),
                    "predicted_conflict": entry.get("mean_selected_conflict_rate_predicted"),
                    "conflict_auc": label_metrics.get("roc_auc"),
                }
            )
    return rows


def _mean_std(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return float("nan"), float("nan")
    return st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)


def decision_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = {r["method"] for r in rows if r["method"]}
    table: list[dict[str, Any]] = []
    means: dict[str, dict[str, float]] = {}
    for method in methods:
        mrows = [r for r in rows if r["method"] == method]
        rec = {
            "method": method,
            "n": len(mrows),
            "accuracy": _mean_std([r["accuracy"] for r in mrows]),
            "steps": _mean_std([r["steps"] for r in mrows]),
            "latency": _mean_std([r["latency"] for r in mrows]),
            "oracle_conflict": _mean_std([r["oracle_conflict"] for r in mrows]),
            "conflict_auc": _mean_std([r["conflict_auc"] for r in mrows]),
        }
        means[method] = {k: rec[k][0] for k in ("accuracy", "steps", "oracle_conflict")}
        table.append(rec)
    conf = means.get("confidence", {}).get("oracle_conflict")
    pair = means.get("pairwise_mlp", {}).get("oracle_conflict")
    for rec in table:
        oc = rec["oracle_conflict"][0]
        rec["delta_oracle_vs_confidence"] = (oc - conf) if conf is not None else None
        rec["delta_oracle_vs_pairwise"] = (oc - pair) if pair is not None else None
    table.sort(key=lambda r: METHOD_ORDER.index(r["method"]) if r["method"] in METHOD_ORDER else 99)
    return table


def paired_gnn_pairwise_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per (seed, mask_ratio): GNN minus pairwise on oracle/acc/steps."""
    def key(r):
        return (r["seed"], r["mask_ratio"])

    gnn = {key(r): r for r in rows if r["method"] == "gnn"}
    pair = {key(r): r for r in rows if r["method"] == "pairwise_mlp"}
    out: list[dict[str, Any]] = []
    for k in sorted(set(gnn) & set(pair), key=lambda x: (str(x[0]), str(x[1]))):
        g, p = gnn[k], pair[k]
        def d(field):
            if isinstance(g.get(field), (int, float)) and isinstance(p.get(field), (int, float)):
                return g[field] - p[field]
            return None
        out.append(
            {
                "seed": k[0], "mask_ratio": k[1],
                "d_oracle": d("oracle_conflict"), "d_accuracy": d("accuracy"), "d_steps": d("steps"),
            }
        )
    return out


def matched_accuracy_compare(rows: list[dict[str, Any]], a: str, b: str) -> dict[str, Any]:
    """Compare oracle conflict at matched accuracy by interpolating each method's
    (accuracy -> oracle_conflict) curve over the overlapping accuracy range.

    Useful when a method has multiple operating points (e.g. Batch 6 theta sweep).
    """
    def curve(method):
        pts = sorted(
            (r["accuracy"], r["oracle_conflict"]) for r in rows
            if r["method"] == method and isinstance(r["accuracy"], (int, float))
            and isinstance(r["oracle_conflict"], (int, float))
        )
        return pts

    ca, cb = curve(a), curve(b)
    if len(ca) < 2 or len(cb) < 2:
        return {"note": f"need >=2 operating points per method (have {len(ca)}/{len(cb)})"}
    lo = max(ca[0][0], cb[0][0])
    hi = min(ca[-1][0], cb[-1][0])
    if hi <= lo:
        return {"note": "no overlapping accuracy range"}

    def interp(curve_pts, x):
        for (x0, y0), (x1, y1) in zip(curve_pts, curve_pts[1:]):
            if x0 <= x <= x1 and x1 != x0:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return curve_pts[-1][1]

    grid = [lo + (hi - lo) * i / 8 for i in range(9)]
    diffs = [interp(ca, x) - interp(cb, x) for x in grid]
    return {
        "accuracy_range": [lo, hi],
        f"mean_oracle_{a}_minus_{b}_at_matched_acc": st.mean(diffs),
    }


def aggregate_sweep(sweep_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for cell_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        rows.extend(collect_cell(cell_dir))
    table = decision_table(rows)
    deltas = paired_gnn_pairwise_deltas(rows)
    matched = matched_accuracy_compare(rows, "gnn", "pairwise_mlp")
    report = {
        "sweep": sweep_dir.name,
        "n_rows": len(rows),
        "decision_table": table,
        "gnn_minus_pairwise_per_seed": deltas,
        "gnn_vs_pairwise_matched_accuracy": matched,
        "rows": rows,
    }
    if rows:
        with (sweep_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with (sweep_dir / "frontier.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["method", "seed", "mask_ratio", "accuracy", "steps", "latency", "oracle_conflict"])
            for r in rows:
                writer.writerow([r["method"], r["seed"], r["mask_ratio"], r["accuracy"], r["steps"], r["latency"], r["oracle_conflict"]])
    with (sweep_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)
    return report


def _fmt(ms: tuple[float, float]) -> str:
    m, s = ms
    return f"{m:.4f}+/-{s:.4f}" if s == s else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True, help="sweep name or path under outputs/sweeps")
    args = parser.parse_args()
    sweep_dir = Path(args.sweep)
    if not sweep_dir.exists():
        sweep_dir = Path("outputs/sweeps") / args.sweep
    report = aggregate_sweep(sweep_dir)
    print(f"=== {report['sweep']}: {report['n_rows']} rows ===")
    print(f"{'method':14s} {'acc':>16s} {'steps':>14s} {'oracle_conf':>16s} {'dOrc_vs_conf':>13s} {'dOrc_vs_pair':>13s}")
    for r in report["decision_table"]:
        dc = r["delta_oracle_vs_confidence"]
        dp = r["delta_oracle_vs_pairwise"]
        print(f"{r['method']:14s} {_fmt(r['accuracy']):>16s} {_fmt(r['steps']):>14s} "
              f"{_fmt(r['oracle_conflict']):>16s} "
              f"{(f'{dc:+.4f}' if isinstance(dc,(int,float)) else '   -'):>13s} "
              f"{(f'{dp:+.4f}' if isinstance(dp,(int,float)) else '   -'):>13s}")
    deltas = report["gnn_minus_pairwise_per_seed"]
    if deltas:
        valid = [d["d_oracle"] for d in deltas if isinstance(d.get("d_oracle"), (int, float))]
        if valid:
            print(f"\nGNN-minus-pairwise oracle conflict (per seed x ratio): "
                  f"mean={st.mean(valid):+.4f} (n={len(valid)}; negative => GNN better)")
    print("matched-accuracy:", json.dumps(report["gnn_vs_pairwise_matched_accuracy"], default=str))


if __name__ == "__main__":
    main()
