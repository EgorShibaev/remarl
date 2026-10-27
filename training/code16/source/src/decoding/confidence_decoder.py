from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from src.backbones.base import DiffusionBackbone
from src.data.masking import get_masked_positions
from src.decoding.schedules import budget_from_fraction
from src.decoding.selection import confidence_topk_selection
from src.eval.metrics import MetricsHook, StepContext, confidence_margin_entropy, run_metrics_hooks


@dataclass
class DecodeStepTrace:
    step: int
    masked_remaining: int
    selected_positions: list[list[int]]
    mean_confidence: float
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class DecodeResult:
    input_ids: torch.Tensor
    steps: int
    traces: list[DecodeStepTrace] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfidenceDecoderConfig:
    max_steps: int = 16
    min_confidence: float = 0.0
    budget_fraction: float = 0.30
    force_complete: bool = True


class ConfidenceDecoder:
    def __init__(
        self,
        backbone: DiffusionBackbone,
        config: ConfidenceDecoderConfig,
        metrics_hooks: list[MetricsHook] | None = None,
    ) -> None:
        self.backbone = backbone
        self.config = config
        self.metrics_hooks: list[MetricsHook] = list(metrics_hooks or [])

    @torch.no_grad()
    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        timestep: int = 0,
        clean_input_ids: torch.Tensor | None = None,
    ) -> DecodeResult:
        current = input_ids.clone()
        traces: list[DecodeStepTrace] = []
        for step in range(self.config.max_steps):
            masked_positions = get_masked_positions(current, self.backbone.mask_token_id)
            total_remaining = sum(int(pos.numel()) for pos in masked_positions)
            if total_remaining == 0:
                return DecodeResult(current, step, traces)
            out = self.backbone.forward(current, attention_mask, timestep=timestep)
            logits = out["logits"]
            input_ids_before = current.clone()
            selected_by_row: list[list[int]] = []
            selected_positions_per_row: list[torch.Tensor] = []
            selected_local_per_row: list[torch.Tensor] = []
            selected_confidences: list[torch.Tensor] = []
            for row, positions in enumerate(masked_positions):
                if positions.numel() == 0:
                    selected_by_row.append([])
                    selected_positions_per_row.append(positions)
                    selected_local_per_row.append(positions)
                    continue
                row_logits = logits[row, positions]
                confidence, _, _ = confidence_margin_entropy(row_logits)
                budget = budget_from_fraction(int(positions.numel()), self.config.budget_fraction)
                if self.config.force_complete and step == self.config.max_steps - 1:
                    budget = int(positions.numel())
                result = confidence_topk_selection(
                    confidence=confidence,
                    budget=budget,
                    min_confidence=self.config.min_confidence,
                )
                if result.selected_local.numel() == 0 and self.config.force_complete:
                    result = confidence_topk_selection(confidence=confidence, budget=1)
                selected_positions = positions[result.selected_local]
                top_tokens = row_logits.argmax(dim=-1)[result.selected_local]
                current[row, selected_positions] = top_tokens
                selected_by_row.append([int(x) for x in selected_positions.detach().cpu()])
                selected_positions_per_row.append(selected_positions)
                selected_local_per_row.append(result.selected_local)
                if result.selected_local.numel() > 0:
                    selected_confidences.append(confidence[result.selected_local])
            if selected_confidences:
                mean_conf = float(torch.cat(selected_confidences).mean().item())
            else:
                mean_conf = 0.0
            step_metrics: dict[str, float] = {}
            if self.metrics_hooks:
                ctx = StepContext(
                    step=step,
                    input_ids_before=input_ids_before,
                    attention_mask=attention_mask,
                    pre_step_logits=logits,
                    masked_positions_per_row=masked_positions,
                    selected_positions_per_row=selected_positions_per_row,
                    selected_local_per_row=selected_local_per_row,
                    conflict_matrix_per_row=[None] * len(masked_positions),
                    clean_input_ids=clean_input_ids,
                    backbone=self.backbone,
                )
                step_metrics = run_metrics_hooks(self.metrics_hooks, ctx)
            traces.append(
                DecodeStepTrace(
                    step=step,
                    masked_remaining=total_remaining,
                    selected_positions=selected_by_row,
                    mean_confidence=mean_conf,
                    metrics=step_metrics,
                )
            )
        return DecodeResult(current, self.config.max_steps, traces, {"stopped_by": "max_steps"})

