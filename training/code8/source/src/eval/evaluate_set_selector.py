"""Validation: learned P2 planner vs the confidence self-planning baseline across the
accuracy / NFE speed-quality curve.

Speed is controlled by the **number of decode steps ``S``** (P2 top-k scheduler): for
each method (confidence, learned) and each ``S`` in the grid, decode the whole
validation set deterministically, aggregate mean accuracy / NFE (== S) / remask count,
build the speed-quality curve, Pareto-filter it, and report AUC / Delta-AUC plus accuracy
gain at fixed NFE and speedup at fixed accuracy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.decoding.apple_dit_policy import AppleDiTConfig, AppleDiTDecoder
from src.decoding.threshold_decoder import (
    AdaptiveConfig,
    AdaptiveDecoder,
    ConfidenceZSource,
    LearnedZSource,
    P2Decoder,
    P2DecoderConfig,
)
from src.eval.pareto import auc_over_speed, gain_at_nfe, pareto_filter, speed_from_nfe, speedup_at_acc
from src.graph.features import FeatureConfig
from src.models.set_selector import SetSelector, SetSelectorConfig
from src.rl.environments import Environment, get_environment
from src.utils.checkpointing import load_checkpoint
from src.utils.config import load_config, parse_scalar, set_dotted
from src.utils.device import resolve_device
from src.utils.logging import make_run_dir, write_json


def load_selector_from_checkpoint(
    path: str, device: torch.device
) -> tuple[SetSelector, FeatureConfig, object, dict[str, Any]]:
    """Load a trained selector, its planner feature spec, and any denoiser LoRA payload.

    The last element is empty for the usual frozen-backbone checkpoints.
    """
    ckpt = load_checkpoint(path, map_location=device)
    feature_config = FeatureConfig.from_raw(ckpt.get("feature_config", {}))
    selector = SetSelector(SetSelectorConfig(**dict(ckpt["set_selector_config"]))).to(device)
    selector.load_state_dict(ckpt["model_state"])
    selector.eval()
    lora = {
        "state": ckpt.get("denoiser_lora_state") or {},
        "config": ckpt.get("denoiser_lora_config") or {},
    }
    return selector, feature_config, ckpt.get("planner_features"), lora


def restore_denoiser_lora(backbone, lora: dict[str, Any], fallback_config: dict[str, Any]) -> int:
    """Re-attach and load jointly-trained LoRA adapters; returns the number of tensors loaded.

    A planner trained jointly with the denoiser is only valid against *that* denoiser, so
    skipping this would silently evaluate it against the un-adapted backbone.
    """
    state = lora.get("state") or {}
    if not state:
        return 0
    backbone.apply_lora(lora.get("config") or fallback_config or {})
    missing, unexpected = backbone.model.load_state_dict(state, strict=False)
    loaded = len(state) - len(unexpected)
    if unexpected:
        raise ValueError(
            f"{len(unexpected)} LoRA tensors did not match the rebuilt adapters "
            f"(e.g. {unexpected[:3]}); check backbone.lora matches the training config"
        )
    del missing  # base weights are intentionally absent from the saved adapter state
    print(f"[eval-selector] restored {loaded} jointly-trained LoRA tensors into the denoiser")
    return loaded


def _decode_all(
    decoder, env: Environment, device: torch.device
) -> tuple[list[float], list[float], list[float]]:
    """Per-example (accuracy, NFE, remask) over the whole validation set, in problem order.

    Examples are in the same order for every method/knob, so example ``i`` is comparable
    across methods -> the arrays support a *paired* bootstrap over examples.
    """
    accs, nfes, remasks = [], [], []
    for problem in env.problems:
        result = decoder.decode(
            problem.input_ids.unsqueeze(0).to(device),
            problem.attention_mask.unsqueeze(0).to(device),
            editable_positions=[problem.answer_span.to(device)],
        )
        answer = env.decode_answer(result.input_ids.squeeze(0), problem)
        accs.append(float(env.verifier(answer, problem.y_star)))
        nfes.append(float(result.steps))
        remasks.append(float(result.metadata.get("remask_count", 0)))
    return accs, nfes, remasks


def _mean(xs: list[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _acc_at(curve: list[tuple[float, float]], nfe: float) -> float:
    """Linear interpolation of a method's accuracy at a given NFE (clamped to the curve's ends)."""
    pts = sorted(curve)
    if not pts:
        return float("nan")
    if nfe <= pts[0][0]:
        return pts[0][1]
    if nfe >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= nfe <= x1:
            if x1 == x0:
                return max(y0, y1)
            return y0 + (y1 - y0) * (nfe - x0) / (x1 - x0)
    return pts[-1][1]


# method -> (Z source factory, allow_remask). ``confidence`` is P2 self-planning (revision
# allowed); ``confidence_monotone`` is the plain confidence decoder with no revision, which
# isolates what the remasking mechanism itself buys.
METHODS: dict[str, tuple[str, bool]] = {
    "confidence": ("confidence", True),
    "confidence_monotone": ("confidence", False),
    "learned": ("learned", True),
    "learned_monotone": ("learned", False),
}

# Knob-driven methods produce a point per knob value with a MEASURED (variable) NFE, unlike
# the P2 methods above which are driven by the step-count grid (NFE == S). All go through the
# factorised AdaptiveDecoder, which separates the two decisions Fast-dLLM couples: an adaptive
# COUNT rule (threshold on lambda, or the Fast-dLLM++ Frechet profile on delta) sets how many
# to commit, and a score sets which. This lets us attribute any gain to cardinality vs
# membership vs the learned signal:
#   fastdllm            -- Fast-dLLM v1 (threshold, count=member=C).
#   learned_threshold   -- learned Z under the same threshold rule (count=member=Z).
#   fastdllm_pp         -- Fast-dLLM++ Frechet profile decoding (E0 baseline).
#   membership_only     -- E1: Fast-dLLM cardinality (count=C) but learned membership (top-k Z).
#   frechet_membership  -- E1 with the ++ cardinality (frechet count=C) + learned membership.
@dataclass(frozen=True)
class KnobSpec:
    z_source: str      # "confidence" | "learned"
    rule: str          # "threshold" (knob=lambda) | "frechet" (knob=delta)
    count_score: str   # "C" | "Z": which score sets the cardinality
    member_score: str  # "C" | "Z": which score picks the members


KNOB_METHODS: dict[str, KnobSpec] = {
    "fastdllm":           KnobSpec("confidence", "threshold", "C", "C"),
    "learned_threshold":  KnobSpec("learned",    "threshold", "Z", "Z"),
    "fastdllm_pp":        KnobSpec("confidence", "frechet",   "C", "C"),
    "membership_only":    KnobSpec("learned",    "threshold", "C", "Z"),
    "frechet_membership": KnobSpec("learned",    "frechet",   "C", "Z"),
}


def _pareto_auc(rows: list[tuple[float, float]], nfe_slow: float, nfe_fast: float) -> float:
    """AUC over the common [0,1] speed axis for (nfe, acc) points (speed clamped to [0,1])."""
    pts = [(min(1.0, max(0.0, speed_from_nfe(nfe, nfe_slow, nfe_fast))), acc) for nfe, acc in rows]
    return auc_over_speed(pareto_filter(pts))


def bootstrap_auc_deltas(
    raw_ex: dict[str, list[tuple[float, list[float], list[float]]]],
    methods: list[str],
    baseline: str,
    nfe_slow: float,
    nfe_fast: float,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Paired bootstrap (over examples) of each method's Delta-AUC vs the baseline.

    ``raw_ex[method]`` is ``[(knob, per-example accs, per-example nfes)]`` with examples in a
    fixed shared order, so a single resample of example indices is applied to *every* method
    (paired). For each resample we recompute each method's mean curve -> AUC -> Delta vs the
    baseline; the 2.5/97.5 percentiles give a 95% CI and ``p_gt0`` the fraction of resamples
    with a positive delta. Speed normalisation uses the fixed full-sample nfe range so the axis
    does not jitter across resamples.
    """
    others = [m for m in methods if m != baseline]
    if not others:
        return {}
    n = len(raw_ex[baseline][0][1]) if raw_ex[baseline] else 0
    if n == 0:
        return {}
    acc_t = {m: torch.tensor([a for _k, a, _n in raw_ex[m]]) for m in methods}  # [K, n]
    nfe_t = {m: torch.tensor([nn for _k, _a, nn in raw_ex[m]]) for m in methods}
    gen = torch.Generator().manual_seed(seed)
    deltas: dict[str, list[float]] = {m: [] for m in others}
    for _b in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=gen)
        aucs = {}
        for m in methods:
            ma = acc_t[m].index_select(1, idx).mean(dim=1)
            mn = nfe_t[m].index_select(1, idx).mean(dim=1)
            aucs[m] = _pareto_auc(list(zip(mn.tolist(), ma.tolist())), nfe_slow, nfe_fast)
        for m in others:
            deltas[m].append(aucs[m] - aucs[baseline])
    out: dict[str, dict[str, float]] = {}
    for m, ds in deltas.items():
        t = torch.tensor(ds)
        out[m] = {
            "delta_mean": float(t.mean()),
            "ci_lo": float(torch.quantile(t, 0.025)),
            "ci_hi": float(torch.quantile(t, 0.975)),
            "p_gt0": float((t > 0).float().mean()),
            "n_examples": int(n),
            "n_boot": int(n_boot),
        }
    return out


def evaluate_curves(
    backbone,
    selector: SetSelector,
    env: Environment,
    *,
    step_grid: list[int],
    kappa: str = "linear",
    block_length: int = 0,
    budgets: list[float] | None = None,
    acc_targets: list[float] | None = None,
    device: torch.device | None = None,
    methods: list[str] | None = None,
    baseline: str = "confidence",
    feature_spec: object = None,
    fastdllm_thresholds: list[float] | None = None,
    frechet_deltas: list[float] | None = None,
    apple_dit_policy: object = None,
    apple_dit_stop_logits: list[float] | None = None,
    apple_dit_seeds: list[int] | None = None,
    apple_dit_temperature: float = 1.0,
    n_boot: int = 0,
    speed_axis: str = "fixed",
    axis_nfe_fast: float = 1.0,
    axis_nfe_slow: float = 256.0,
) -> dict[str, Any]:
    device = device or torch.device("cpu")
    methods = list(methods or ["confidence", "confidence_monotone", "learned"])
    APPLE = {"apple_dit", "apple_dit_threshold"}
    known = set(METHODS) | set(KNOB_METHODS) | APPLE
    unknown = [m for m in methods if m not in known]
    if unknown:
        raise ValueError(f"Unknown eval methods {unknown}. Available: {sorted(known)}")
    if baseline not in methods:
        raise ValueError(f"baseline {baseline!r} must be one of the evaluated methods {methods}")
    if baseline == "apple_dit":
        raise ValueError("Native Apple is one stochastic operating point, not a curve baseline")
    if (APPLE & set(methods)) and apple_dit_policy is None:
        raise ValueError("apple_dit* methods require apple_dit_policy (set apple_dit_checkpoint)")
    # threshold rule sweeps lambda; frechet rule (Fast-dLLM++) sweeps delta over its own grid.
    lambdas = fastdllm_thresholds or [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    deltas = frechet_deltas or [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
    stop_logits = apple_dit_stop_logits or [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
    seeds = [int(s) for s in (apple_dit_seeds or [0, 1, 2])]
    z_sources = {
        "confidence": ConfidenceZSource(),
        "learned": LearnedZSource(selector, spec=feature_spec),
    }
    # method -> [(knob, acc, nfe, remask)]; knob is the step count S for P2 methods and the
    # threshold lambda (or frechet delta / apple stop_logit) for knob methods. NFE is what
    # pareto/AUC consumes. raw_ex keeps the PER-EXAMPLE (accs, nfes) per knob for the paired
    # bootstrap (examples share a fixed order across methods).
    raw: dict[str, list[tuple[float, float, float, float]]] = {}
    raw_ex: dict[str, list[tuple[float, list[float], list[float]]]] = {}
    native_seed_rows = []
    for method in methods:
        rows: list[tuple[float, float, float, float]] = []
        ex_rows: list[tuple[float, list[float], list[float]]] = []

        def _run(knob: float, decoder) -> None:
            accs, nfes, remasks = _decode_all(decoder, env, device)
            rows.append((float(knob), _mean(accs), _mean(nfes), _mean(remasks)))
            ex_rows.append((float(knob), accs, nfes))

        if method == "apple_dit":  # NATIVE Apple inference: Bernoulli sampling, one point per seed
            for seed in seeds:
                _run(seed, AppleDiTDecoder(
                    backbone, apple_dit_policy,
                    AppleDiTConfig(sampling_mode="bernoulli", seed=seed, block_length=block_length,
                                   full_context=True, temperature_policy=float(apple_dit_temperature)),
                ))
        elif method == "apple_dit_threshold":  # MODIFIED decoder (deterministic logit>stop_logit sweep)
            for sl in stop_logits:
                _run(sl, AppleDiTDecoder(
                    backbone, apple_dit_policy,
                    AppleDiTConfig(sampling_mode="threshold", stop_logit=float(sl),
                                   block_length=block_length, full_context=True),
                ))
        elif method in KNOB_METHODS:  # adaptive-cardinality decoders: sweep knob, NFE measured
            spec = KNOB_METHODS[method]
            src = z_sources[spec.z_source]
            for knob in (deltas if spec.rule == "frechet" else lambdas):
                _run(knob, AdaptiveDecoder(
                    backbone,
                    AdaptiveConfig(
                        rule=spec.rule, knob=float(knob), count_score=spec.count_score,
                        member_score=spec.member_score, block_length=block_length,
                    ),
                    z_source=src,
                ))
        else:  # P2 method: sweep the step-count grid, NFE == S
            src_key, allow_remask = METHODS[method]
            src = z_sources[src_key]
            for s in step_grid:
                _run(s, P2Decoder(
                    backbone,
                    P2DecoderConfig(
                        steps=int(s), kappa=kappa, block_length=block_length, allow_remask=allow_remask
                    ),
                    src,
                ))
        if method == "apple_dit":
            native_seed_rows = list(rows)
            rows = [(0., _mean([r[1] for r in rows]), _mean([r[2] for r in rows]),
                     _mean([r[3] for r in rows]))]
            ex_rows = [(0., [_mean(values) for values in zip(*(r[1] for r in ex_rows))],
                        [_mean(values) for values in zip(*(r[2] for r in ex_rows))])]
        raw[method] = rows
        raw_ex[method] = ex_rows

    # Speed axis. "fixed" (default) normalises NFE with FIXED bounds so AUCs are comparable
    # across evaluations. "run_relative" (legacy) uses the min/max NFE of the methods in THIS
    # run -- adding a method with a different NFE range then rescales every other method's AUC,
    # which silently made numbers from separately produced tables incomparable.
    if speed_axis == "run_relative":
        all_nfe = [nfe for rows in raw.values() for _s, _a, nfe, _r in rows]
        nfe_slow, nfe_fast = (max(all_nfe), min(all_nfe)) if all_nfe else (1.0, 0.0)
    elif speed_axis == "fixed":
        nfe_fast, nfe_slow = float(axis_nfe_fast), float(axis_nfe_slow)
    else:
        raise ValueError(f"speed_axis must be 'fixed' or 'run_relative', got {speed_axis!r}")

    def _speed(nfe: float) -> float:  # clamp so NFE outside the fixed bounds stays in [0,1]
        return min(1.0, max(0.0, speed_from_nfe(nfe, nfe_slow, nfe_fast)))

    report: dict[str, Any] = {
        "nfe_slow": nfe_slow, "nfe_fast": nfe_fast,
        "speed_axis": {"mode": speed_axis, "nfe_fast": nfe_fast, "nfe_slow": nfe_slow},
        "tables": {}, "auc": {},
    }
    curves_nfe = {}
    for method, rows in raw.items():
        points = [(_speed(nfe), acc) for _s, acc, nfe, _r in rows]
        # "steps" carries the swept knob: an int step count for P2 methods, the (float)
        # threshold lambda for fastdllm.
        report["tables"][method] = [
            {"steps": None if method == "apple_dit" else int(s) if float(s).is_integer() else s,
             "acc": a, "nfe": n, "remask": r}
            for s, a, n, r in rows
        ]
        front = pareto_filter(points)
        # AUC on the common [0,1] speed axis so a uniformly-faster+more-accurate
        # method scores higher (plain per-method-span AUC under-credits fast methods).
        if method != "apple_dit":
            report["auc"][method] = auc_over_speed(front)
        curves_nfe[method] = [(nfe, acc) for _s, acc, nfe, _r in rows]

    # Native Apple summary: the policy has ONE operating point (per seed), so the fair read is
    # its mean +/- std over seeds and every other method's accuracy interpolated at that NFE.
    if "apple_dit" in raw and raw["apple_dit"]:
        accs = [a for _s, a, _n, _r in native_seed_rows]
        nfes = [n for _s, _a, n, _r in native_seed_rows]
        m_nfe = _mean(nfes)
        report["apple_dit_native"] = {
            "n_seeds": len(accs),
            "acc_mean": _mean(accs), "acc_std": _std(accs),
            "nfe_mean": m_nfe, "nfe_std": _std(nfes),
            "seed_results": [{"seed": int(s), "acc": a, "nfe": n, "remask": r}
                             for s, a, n, r in native_seed_rows],
            "auc_excluded_reason": "Seeds are repetitions of one policy, not speed controls",
            "others_acc_at_native_nfe": {
                m: _acc_at(curves_nfe[m], m_nfe) for m in methods if m != "apple_dit"
            },
        }

    budgets = budgets or sorted({n for _s, _a, n, _r in raw[baseline]})
    acc_targets = acc_targets or sorted({a for _s, a, _n, _r in raw[baseline]})
    report["baseline"] = baseline
    report["delta_auc_by_method"] = {
        m: report["auc"][m] - report["auc"][baseline]
        for m in methods if m not in (baseline, "apple_dit")
    }
    # Paired bootstrap over examples: 95% CI + P(delta>0) on each Delta-AUC vs baseline, so a
    # gain can be called significant instead of eyeballing noisy means.
    if n_boot > 0:
        report["auc_delta_ci"] = bootstrap_auc_deltas(
            raw_ex, [m for m in methods if m != "apple_dit"], baseline,
            nfe_slow, nfe_fast, n_boot=n_boot
        )
    report["gain_at_nfe_by_method"] = {
        m: gain_at_nfe(curves_nfe[m], curves_nfe[baseline], budgets)
        for m in methods if m not in (baseline, "apple_dit")
    }
    report["speedup_at_acc_by_method"] = {
        m: speedup_at_acc(curves_nfe[baseline], curves_nfe[m], acc_targets)
        for m in methods if m not in (baseline, "apple_dit")
    }
    # Back-compat headline keys: learned vs the P2 confidence self-planner.
    if "learned" in methods and "confidence" in methods:
        report["delta_auc"] = report["auc"]["learned"] - report["auc"]["confidence"]
        report["learned_beats_confidence"] = report["delta_auc"] > 0.0
        report["gain_at_nfe"] = gain_at_nfe(curves_nfe["learned"], curves_nfe["confidence"], budgets)
        report["speedup_at_acc"] = speedup_at_acc(
            curves_nfe["confidence"], curves_nfe["learned"], acc_targets
        )
    return report


def evaluate(config: dict[str, Any]) -> Path:
    from src.backbones.registry import build_backbone

    device = resolve_device(config.get("device", "auto"))
    run_dir = make_run_dir(config.get("output_dir", "outputs/runs"), "eval-selector")
    write_json(run_dir / "resolved_config.json", config)

    backbone = build_backbone(config.get("backbone", {})).to(device)
    ckpt_path = config["checkpoint"]
    selector, feature_config, planner_spec, lora = load_selector_from_checkpoint(ckpt_path, device)
    restore_denoiser_lora(backbone, lora, (config.get("backbone", {}) or {}).get("lora", {}))
    file_fc = FeatureConfig.from_raw(config.get("features", {}))
    feature_config = file_fc if config.get("features") else feature_config

    env_cfg = config.get("env", {}) or {}
    env_kwargs = dict(env_cfg.get("kwargs", {}))
    if env_cfg.get("name", "toy") != "toy":
        env_kwargs.setdefault("tokenizer", backbone.tokenizer)
        env_kwargs.setdefault("mask_token_id", backbone.mask_token_id)
        env_kwargs.setdefault("pad_token_id", backbone.pad_token_id)
        env_kwargs.setdefault("max_length", feature_config.max_length)
        env_kwargs.setdefault("split", "val")
    env = get_environment(env_cfg.get("name", "toy"), **env_kwargs)

    # Optionally load Apple's trained dit_confidence policy for a head-to-head on the same axes.
    apple_policy = None
    if {"apple_dit", "apple_dit_threshold"} & set(config.get("methods") or []):
        from src.decoding.apple_dit_policy import load_apple_dit_policy

        apple_ckpt = config.get("apple_dit_checkpoint")
        if not apple_ckpt:
            raise ValueError("methods include 'apple_dit' but apple_dit_checkpoint is unset")
        apple_policy = load_apple_dit_policy(apple_ckpt, arch=config.get("apple_dit_arch"), device=device)

    report = evaluate_curves(
        backbone, selector.to(device), env,
        step_grid=[int(s) for s in config.get("step_grid", [4, 8, 16, 32])],
        kappa=str(config.get("kappa", "linear")),
        block_length=int(config.get("block_length", 0)),
        budgets=config.get("budgets"), acc_targets=config.get("acc_targets"),
        device=device,
        methods=config.get("methods"), baseline=str(config.get("baseline", "confidence")),
        feature_spec=planner_spec, fastdllm_thresholds=config.get("fastdllm_thresholds"),
        frechet_deltas=config.get("frechet_deltas"),
        apple_dit_policy=apple_policy, apple_dit_stop_logits=config.get("apple_dit_stop_logits"),
        apple_dit_seeds=config.get("apple_dit_seeds"),
        apple_dit_temperature=float(config.get("apple_dit_temperature", 1.0)),
        n_boot=int(config.get("n_boot", 1000)),
        speed_axis=str(config.get("speed_axis", "fixed")),
        axis_nfe_fast=float(config.get("axis_nfe_fast", 1.0)),
        axis_nfe_slow=float(config.get("axis_nfe_slow", 256.0)),
    )
    write_json(run_dir / "report.json", report)
    aucs = " ".join(f"{m}={v:.4f}" for m, v in report["auc"].items())
    print(f"[eval-selector] AUC {aucs} | baseline={report['baseline']} "
          + " ".join(f"d[{m}]={v:+.4f}" for m, v in report["delta_auc_by_method"].items()))
    if report.get("auc_delta_ci"):
        ci = " ".join(
            f"{m}={v['delta_mean']:+.4f}[{v['ci_lo']:+.3f},{v['ci_hi']:+.3f}]p>0={v['p_gt0']:.2f}"
            for m, v in report["auc_delta_ci"].items()
        )
        print(f"[eval-selector] Delta-AUC vs {report['baseline']} (paired bootstrap 95% CI): {ci}")
    if report.get("apple_dit_native"):
        n = report["apple_dit_native"]
        others = " ".join(f"{m}@nfe={v:.3f}" for m, v in n["others_acc_at_native_nfe"].items())
        print(f"[eval-selector] apple_dit NATIVE ({n['n_seeds']} seeds): acc={n['acc_mean']:.3f}"
              f"+-{n['acc_std']:.3f} nfe={n['nfe_mean']:.1f}+-{n['nfe_std']:.1f} | others at that NFE: {others}")
    ax = report.get("speed_axis", {})
    print(f"[eval-selector] speed axis: {ax.get('mode')} nfe_fast={ax.get('nfe_fast')} nfe_slow={ax.get('nfe_slow')}")
    if "delta_auc" in report:
        print(f"[eval-selector] AUC conf={report['auc']['confidence']:.4f} "
              f"learned={report['auc']['learned']:.4f} delta={report['delta_auc']:+.4f} "
              f"(learned_better={report['learned_beats_confidence']})")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], help="dotted=value overrides")
    args = parser.parse_args()
    config = load_config(args.config)
    for item in args.overrides:
        key, _, val = item.partition("=")
        set_dotted(config, key, parse_scalar(val))
    evaluate(config)


if __name__ == "__main__":
    main()
