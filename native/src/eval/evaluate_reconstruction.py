from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.backbones.registry import build_backbone
from src.data.datasets import collate_tokenized, load_text_dataset, resolve_masking
from src.data.masking import apply_mask
from src.decoding.attention_graph_decoder import AttentionDecoderConfig, AttentionGraphDecoder
from src.decoding.confidence_decoder import ConfidenceDecoder, ConfidenceDecoderConfig
from src.decoding.gnn_conflict_decoder import (
    GNNConflictDecoder,
    GNNDecoderConfig,
    build_conflict_model_from_checkpoint,
    schedule_from_config,
)
from src.decoding.oracle_conflict_decoder import OracleConflictDecoder, OracleDecoderConfig
from src.decoding.set_selector_decoder import (
    SetSelectorDecoder,
    SetSelectorDecoderConfig,
    build_edge_model_from_checkpoint,
    load_set_scorer_from_checkpoint,
)
from src.eval.metrics import (
    MetricsHook,
    conflict_graph_density_hook,
    oracle_conflict_rate_hook,
    predicted_conflict_rate_hook,
    reconstruction_accuracy,
    selected_set_size_hook,
    timed,
)
from src.utils.config import deep_merge, load_config, parse_scalar, set_dotted
from src.utils.device import resolve_device
from src.utils.logging import JsonlLogger, make_run_dir, write_json
from src.utils.seed import set_seed
from src.utils.tracking import init_wandb


def _build_metrics_hooks(config: dict) -> list[MetricsHook]:
    metrics_cfg = config.get("metrics", {})
    hooks: list[MetricsHook] = []
    if bool(metrics_cfg.get("compute_selected_conflict_rate", False)):
        theta = float(metrics_cfg.get("predicted_conflict_theta", 0.5))
        hooks.append(predicted_conflict_rate_hook(theta))
    if bool(metrics_cfg.get("compute_selected_set_size", True)):
        hooks.append(selected_set_size_hook())
    if bool(metrics_cfg.get("compute_conflict_density", True)):
        hooks.append(conflict_graph_density_hook(float(metrics_cfg.get("predicted_conflict_theta", 0.5))))
    if bool(metrics_cfg.get("compute_selected_conflict_rate_oracle", False)):
        oracle_cfg = metrics_cfg.get("oracle", {}) or {}
        tau = float(oracle_cfg.get("tau", 0.05))
        topk_for_js = oracle_cfg.get("topk_for_js")
        hooks.append(
            oracle_conflict_rate_hook(
                tau=tau,
                topk_for_js=int(topk_for_js) if topk_for_js is not None else None,
            )
        )
    return hooks


def _build_decoder(name: str, backbone, config: dict, device: torch.device, hooks: list[MetricsHook]):
    if name == "confidence":
        decode_cfg = config.get("decoding", {})
        return ConfidenceDecoder(
            backbone,
            ConfidenceDecoderConfig(
                max_steps=int(decode_cfg.get("max_steps", 16)),
                min_confidence=float(decode_cfg.get("min_confidence", 0.0)),
                budget_fraction=float(decode_cfg.get("budget_fraction", 0.30)),
                force_complete=bool(decode_cfg.get("force_complete", True)),
            ),
            metrics_hooks=hooks,
        )
    if name == "gnn":
        checkpoint_path = config.get("checkpoint", {}).get("path")
        if not checkpoint_path:
            raise ValueError("GNN evaluation requires checkpoint.path")
        model, feature_config = build_conflict_model_from_checkpoint(checkpoint_path, device)
        decode_cfg = config.get("decoding", {})
        return GNNConflictDecoder(
            backbone,
            model,
            feature_config,
            GNNDecoderConfig(
                max_steps=int(decode_cfg.get("max_steps", 16)),
                force_complete=bool(decode_cfg.get("force_complete", True)),
                fallback_top1=bool(decode_cfg.get("fallback_top1", True)),
                selection=str(decode_cfg.get("selection", "greedy")),
                gamma=float(decode_cfg.get("gamma", 0.5)),
                temperature=float(decode_cfg.get("temperature", 1.0)),
                schedule=schedule_from_config(decode_cfg.get("schedule")),
            ),
            metrics_hooks=hooks,
        )
    if name == "attention":
        decode_cfg = config.get("decoding", {})
        return AttentionGraphDecoder(
            backbone,
            AttentionDecoderConfig(
                max_steps=int(decode_cfg.get("max_steps", 16)),
                force_complete=bool(decode_cfg.get("force_complete", True)),
                fallback_top1=bool(decode_cfg.get("fallback_top1", True)),
                layers=decode_cfg.get("attention_layers", "all"),
                schedule=schedule_from_config(decode_cfg.get("schedule")),
            ),
            metrics_hooks=hooks,
        )
    if name == "oracle":
        decode_cfg = config.get("decoding", {})
        oracle_cfg = config.get("metrics", {}).get("oracle", {}) or {}
        topk = oracle_cfg.get("topk_for_js")
        return OracleConflictDecoder(
            backbone,
            OracleDecoderConfig(
                max_steps=int(decode_cfg.get("max_steps", 16)),
                force_complete=bool(decode_cfg.get("force_complete", True)),
                fallback_top1=bool(decode_cfg.get("fallback_top1", True)),
                tau=float(oracle_cfg.get("tau", 0.05)),
                topk_for_js=int(topk) if topk is not None else None,
                max_candidates=int(decode_cfg.get("oracle_max_candidates", 24)),
                schedule=schedule_from_config(decode_cfg.get("schedule")),
            ),
            metrics_hooks=hooks,
        )
    if name in ("set_pool", "set_gnn", "set_oracle"):
        decode_cfg = config.get("decoding", {})
        ckpt_cfg = config.get("checkpoint", {})
        oracle_cfg = config.get("metrics", {}).get("oracle", {}) or {}
        topk = oracle_cfg.get("topk_for_js")
        sd_cfg = SetSelectorDecoderConfig(
            max_steps=int(decode_cfg.get("max_steps", 16)),
            force_complete=bool(decode_cfg.get("force_complete", True)),
            fallback_top1=bool(decode_cfg.get("fallback_top1", True)),
            scoring="oracle" if name == "set_oracle" else "model",
            pool_size=int(decode_cfg.get("pool_size", 16)),
            budgets=tuple(float(b) for b in decode_cfg.get("set_budgets", (0.25, 0.4, 0.6, 0.8, 1.0))),
            conservative_sizes=tuple(int(s) for s in decode_cfg.get("conservative_sizes", (1, 2))),
            num_stochastic=int(decode_cfg.get("num_stochastic", 3)),
            stochastic_temperature=float(decode_cfg.get("stochastic_temperature", 0.5)),
            edge_thetas=tuple(float(t) for t in decode_cfg.get("edge_thetas", (0.3, 0.5))),
            tau=float(oracle_cfg.get("tau", 0.05)),
            topk_for_js=int(topk) if topk is not None else None,
            compute_regret=bool(decode_cfg.get("compute_regret", name != "set_oracle")),
            seed=int(config.get("seed", 13)),
        )
        set_scorer = None
        feature_config = None
        if name in ("set_pool", "set_gnn"):
            key = "set_scorer_pool" if name == "set_pool" else "set_scorer_gnn"
            path = ckpt_cfg.get(key)
            if not path:
                raise ValueError(f"{name} requires checkpoint.{key}")
            set_scorer, feature_config = load_set_scorer_from_checkpoint(path, device)
        edge_model = None
        edge_feature_config = None
        edge_path = ckpt_cfg.get("edge_model")
        if edge_path:
            edge_model, edge_feature_config = build_edge_model_from_checkpoint(edge_path, device)
        return SetSelectorDecoder(
            backbone,
            sd_cfg,
            set_scorer=set_scorer,
            feature_config=feature_config,
            edge_model=edge_model,
            edge_feature_config=edge_feature_config,
            use_attention=bool(decode_cfg.get("set_use_attention", False)),
            metrics_hooks=hooks,
        )
    raise ValueError(f"Unknown decoder: {name}")


def _bootstrap_mean_ci(values: list[float], confidence: float = 0.95, n_resamples: int = 1000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = torch.tensor(values, dtype=torch.float32)
    n = arr.numel()
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(0, n, (n_resamples, n), generator=generator)
    means = arr[indices].mean(dim=-1)
    lower_q = (1.0 - confidence) / 2.0
    upper_q = 1.0 - lower_q
    lower = float(torch.quantile(means, lower_q).item())
    upper = float(torch.quantile(means, upper_q).item())
    return lower, upper


def _aggregate_step_metrics(traces_per_example: list[list[dict]]) -> dict[str, float]:
    bucket: dict[str, list[float]] = {}
    for traces in traces_per_example:
        for trace in traces:
            for key, value in trace.get("metrics", {}).items():
                bucket.setdefault(key, []).append(float(value))
    return {f"mean_{key}": sum(values) / len(values) for key, values in bucket.items() if values}


def _clean_fingerprint(clean_input_ids: torch.Tensor) -> str:
    """Stable content hash of an example's clean token ids (order-independent id).

    Used by the paired-bootstrap aggregator to fail-fast if two runs do not share
    the same example at a given (domain, seed, example_id), rather than blindly
    aligning by row index.
    """
    ids = clean_input_ids.detach().to(torch.int64).flatten().cpu().numpy().tobytes()
    return hashlib.sha1(ids).hexdigest()[:16]


def _reduce_example_metrics(trace_dicts: list[dict]) -> dict[str, float]:
    """Reduce one example's per-step trace metrics into raw aggregable fields.

    Oracle conflict is kept as numerator/denominator (``oracle_conflict_sum`` /
    ``selected_pair_count``) so the aggregator can form an unbiased weighted rate;
    regret is summarised by sum/num_steps/mean/max.
    """
    conflict_sum = 0.0
    pair_count = 0.0
    regrets: list[float] = []
    for trace in trace_dicts:
        m = trace.get("metrics", {})
        if "selected_conflict_pairs" in m and "selected_pair_count_oracle" in m:
            conflict_sum += float(m["selected_conflict_pairs"])
            pair_count += float(m["selected_pair_count_oracle"])
        if "candidate_oracle_regret" in m:
            regrets.append(float(m["candidate_oracle_regret"]))
    out: dict[str, float] = {
        "oracle_conflict_sum": conflict_sum,
        "selected_pair_count": pair_count,
    }
    if regrets:
        out["sum_regret"] = float(sum(regrets))
        out["num_steps"] = float(len(regrets))
        out["mean_regret"] = float(sum(regrets) / len(regrets))
        out["max_regret"] = float(max(regrets))
    return out


def _evaluate_single_ratio(
    config: dict,
    decoder_name: str,
    decoder: Any,
    backbone: Any,
    device: torch.device,
    mask_ratio: float,
    special_ids: set[int],
    run_dir: Path,
    suffix: str = "",
) -> dict[str, Any]:
    data_cfg = config["data"]
    dataset = load_text_dataset(
        tokenizer=backbone.tokenizer,
        data_cfg=data_cfg,
        split=data_cfg.get("split", "test"),
        max_examples=int(data_cfg.get("max_examples", 64)),
        max_length=int(data_cfg.get("max_length", 128)),
    )
    strategy, restrict_token_ids = resolve_masking(data_cfg, backbone.tokenizer)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_tokenized)
    metrics_path = run_dir / f"metrics{suffix}.jsonl"
    logger = JsonlLogger(metrics_path)
    accuracies: list[float] = []
    steps: list[int] = []
    latencies: list[float] = []
    traces_per_example: list[list[dict]] = []
    per_example: list[dict[str, Any]] = []
    seed = int(config.get("seed", 13))
    domain = str(data_cfg.get("dataset_name", "?"))
    desc = f"eval:{decoder_name}:mr={mask_ratio:.2f}"
    for idx, batch in enumerate(tqdm(loader, desc=desc)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        masked = apply_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_token_id=backbone.mask_token_id,
            mask_ratio=mask_ratio,
            special_token_ids=special_ids,
            strategy=strategy,
            restrict_token_ids=restrict_token_ids,
        )
        with timed() as timer:
            result = decoder.decode(
                masked.input_ids_masked,
                masked.attention_mask,
                clean_input_ids=masked.clean_input_ids,
            )
        acc = reconstruction_accuracy(result.input_ids, masked.clean_input_ids, masked.masked_positions)
        accuracies.append(acc)
        steps.append(result.steps)
        latencies.append(timer.elapsed)
        trace_dicts = [trace.__dict__ for trace in result.traces]
        traces_per_example.append(trace_dicts)
        record = {
            "example_id": idx,
            "fingerprint": _clean_fingerprint(masked.clean_input_ids),
            "seed": seed,
            "domain": domain,
            "accuracy": acc,
            "steps": result.steps,
            "latency": timer.elapsed,
        }
        record.update(_reduce_example_metrics(trace_dicts))
        per_example.append(record)
        logger.write({**record, "example": idx, "trace": trace_dicts})
    mean_acc = sum(accuracies) / max(len(accuracies), 1)
    ci_low, ci_high = _bootstrap_mean_ci(accuracies)
    summary = {
        "decoder": decoder_name,
        "mask_ratio": mask_ratio,
        "examples": len(accuracies),
        "mean_accuracy": mean_acc,
        "accuracy_ci95": [ci_low, ci_high],
        "mean_steps": sum(steps) / max(len(steps), 1),
        "mean_latency": sum(latencies) / max(len(latencies), 1),
    }
    summary.update(_aggregate_step_metrics(traces_per_example))
    summary["per_example"] = per_example
    return summary


def evaluate(config: dict, decoder_name: str) -> Path:
    set_seed(int(config.get("seed", 13)))
    device = resolve_device(config.get("device", "auto"))
    run_dir = make_run_dir(config.get("output_dir", "outputs/runs"), f"eval-{decoder_name}")
    write_json(run_dir / "resolved_config.json", config)
    wandb_run = init_wandb(config, run_dir, job_type=f"eval-{decoder_name}", extra_tags={"decoder": decoder_name})
    backbone = build_backbone(config.get("backbone", {})).to(device)
    hooks = _build_metrics_hooks(config)
    decoder = _build_decoder(decoder_name, backbone, config, device, hooks)
    data_cfg = config["data"]
    special_ids = set(backbone.tokenizer.all_special_ids)

    mask_ratios: list[float]
    if "mask_ratios" in data_cfg:
        mask_ratios = [float(r) for r in data_cfg["mask_ratios"]]
    else:
        mask_ratios = [float(data_cfg.get("mask_ratio", 0.30))]

    summaries: list[dict[str, Any]] = []
    for ratio in mask_ratios:
        suffix = "" if len(mask_ratios) == 1 else f"-mr{ratio:.2f}"
        summary = _evaluate_single_ratio(
            config=config,
            decoder_name=decoder_name,
            decoder=decoder,
            backbone=backbone,
            device=device,
            mask_ratio=ratio,
            special_ids=special_ids,
            run_dir=run_dir,
            suffix=suffix,
        )
        summaries.append(summary)
        if wandb_run is not None:
            wandb_run.log({f"eval/{k}": v for k, v in summary.items() if isinstance(v, (int, float))})
    if len(summaries) == 1:
        write_json(run_dir / "summary.json", summaries[0])
    else:
        write_json(run_dir / "summary.json", {"decoder": decoder_name, "sweep": summaries})
    if wandb_run is not None:
        # frontier-style summary: best accuracy and its cost
        best = max(summaries, key=lambda s: s.get("mean_accuracy", 0.0))
        wandb_run.summary.update(
            {
                "best/mean_accuracy": best.get("mean_accuracy", 0.0),
                "best/mean_steps": best.get("mean_steps", 0.0),
                "best/mean_latency": best.get("mean_latency", 0.0),
                "best/oracle_conflict_rate": best.get("mean_selected_conflict_rate_oracle", float("nan")),
            }
        )
        wandb_run.finish()
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--decoder",
        choices=["confidence", "gnn", "attention", "oracle", "set_pool", "set_gnn", "set_oracle"],
        default="confidence",
    )
    parser.add_argument("--decoder-config", default=None)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], help="dotted=value overrides")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.decoder_config:
        config = deep_merge(config, load_config(args.decoder_config))
    for item in args.overrides:
        key, _, val = item.partition("=")
        set_dotted(config, key, parse_scalar(val))
    evaluate(config, args.decoder)


if __name__ == "__main__":
    main()

