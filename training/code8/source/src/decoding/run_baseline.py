from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.backbones.hf_mlm_adapter import build_hf_mlm_backbone
from src.data.datasets import collate_tokenized, load_hf_text_dataset
from src.data.masking import apply_mask
from src.decoding.confidence_decoder import ConfidenceDecoder, ConfidenceDecoderConfig
from src.eval.metrics import reconstruction_accuracy, timed
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.logging import JsonlLogger, make_run_dir, write_json
from src.utils.seed import set_seed


def run(config: dict) -> None:
    set_seed(int(config.get("seed", 13)))
    device = resolve_device(config.get("device", "auto"))
    run_dir = make_run_dir(config.get("output_dir", "outputs/runs"), "baseline")
    write_json(run_dir / "resolved_config.json", config)
    logger = JsonlLogger(run_dir / "metrics.jsonl")
    backbone = build_hf_mlm_backbone(config.get("backbone", {})).to(device)
    data_cfg = config["data"]
    dataset = load_hf_text_dataset(
        tokenizer=backbone.tokenizer,
        dataset_name=data_cfg.get("dataset_name", "wikitext"),
        dataset_config=data_cfg.get("dataset_config", "wikitext-2-raw-v1"),
        split=data_cfg.get("split", "validation"),
        max_examples=int(data_cfg.get("max_examples", 32)),
        max_length=int(data_cfg.get("max_length", 128)),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_tokenized)
    decode_cfg = config.get("decoding", {})
    decoder = ConfidenceDecoder(
        backbone,
        ConfidenceDecoderConfig(
            max_steps=int(decode_cfg.get("max_steps", 16)),
            min_confidence=float(decode_cfg.get("min_confidence", 0.0)),
            budget_fraction=float(decode_cfg.get("budget_fraction", 0.30)),
            force_complete=bool(decode_cfg.get("force_complete", True)),
        ),
    )
    special_ids = set(backbone.tokenizer.all_special_ids)
    accuracies: list[float] = []
    steps: list[int] = []
    latencies: list[float] = []
    for idx, batch in enumerate(tqdm(loader, desc="baseline")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        masked = apply_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_token_id=backbone.mask_token_id,
            mask_ratio=float(data_cfg.get("mask_ratio", 0.30)),
            special_token_ids=special_ids,
        )
        with timed() as timer:
            result = decoder.decode(masked.input_ids_masked, masked.attention_mask)
        acc = reconstruction_accuracy(result.input_ids, masked.clean_input_ids, masked.masked_positions)
        accuracies.append(acc)
        steps.append(result.steps)
        latencies.append(timer.elapsed)
        logger.write({"example": idx, "accuracy": acc, "steps": result.steps, "latency": timer.elapsed})
    summary = {
        "mean_accuracy": sum(accuracies) / max(len(accuracies), 1),
        "mean_steps": sum(steps) / max(len(steps), 1),
        "mean_latency": sum(latencies) / max(len(latencies), 1),
        "examples": len(accuracies),
    }
    write_json(run_dir / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()

