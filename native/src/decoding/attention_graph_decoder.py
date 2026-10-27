"""Attention-graph decoder - training-free dependency baseline (DAPD/DAWN-style).

The conflict matrix between masked positions is read directly from the backbone's
self-attention (averaged over the configured layers and all heads, symmetrized,
normalized to [0, 1]) instead of a learned head. Strong mutual attention => likely
dependency => should not be unmasked together. Same greedy selection as the GNN
decoder, so it isolates "learned conflict graph vs attention proxy".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.backbones.base import DiffusionBackbone
from src.decoding.confidence_decoder import DecodeResult, DecodeStepTrace
from src.decoding.schedules import DecodeScheduleState, ThreeStageSchedule, budget_from_fraction
from src.decoding.selection import greedy_conflict_aware_selection, node_scores
from src.eval.metrics import MetricsHook, StepContext, confidence_margin_entropy, run_metrics_hooks


@dataclass
class AttentionDecoderConfig:
    max_steps: int = 16
    force_complete: bool = True
    fallback_top1: bool = True
    layers: str | list[int] = "all"  # "all", "last", or explicit list of layer indices
    schedule: ThreeStageSchedule = field(default_factory=ThreeStageSchedule)


def _select_layers(attentions: tuple, spec: str | list[int]) -> list[int]:
    n = len(attentions)
    if isinstance(spec, list):
        return [i for i in spec if -n <= i < n]
    if spec == "last":
        return [n - 1]
    return list(range(n))


def attention_conflict_matrix(
    attentions: tuple,
    positions: torch.Tensor,
    layers: str | list[int],
) -> torch.Tensor:
    """Symmetric, [0,1]-normalized attention affinity between masked positions."""
    n = int(positions.numel())
    if n == 0:
        device = attentions[0].device if attentions else torch.device("cpu")
        return torch.zeros((0, 0), device=device)
    layer_ids = [
        li for li in _select_layers(attentions, layers) if attentions[li] is not None
    ]
    device = attentions[0].device if attentions and attentions[0] is not None else positions.device
    if not layer_ids:
        # backbone returned no attentions (e.g. sdpa); no conflict info available
        return torch.zeros((n, n), device=device)
    acc = None
    for li in layer_ids:
        # attentions[li]: [batch, heads, seq, seq]; row already selected upstream
        attn = attentions[li].mean(dim=1)[0]  # [seq, seq], mean over heads
        sub = attn.index_select(0, positions).index_select(1, positions)  # [n, n]
        acc = sub if acc is None else acc + sub
    acc = acc / max(len(layer_ids), 1)
    matrix = 0.5 * (acc + acc.t())
    matrix.fill_diagonal_(0.0)
    max_val = matrix.max()
    if float(max_val) > 0:
        matrix = matrix / max_val
    return matrix


class AttentionGraphDecoder:
    def __init__(
        self,
        backbone: DiffusionBackbone,
        config: AttentionDecoderConfig,
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
        total_initial = int(current.eq(self.backbone.mask_token_id).sum().item())
        traces: list[DecodeStepTrace] = []
        for step in range(self.config.max_steps):
            masked_by_row = [
                torch.nonzero(row.eq(self.backbone.mask_token_id), as_tuple=False).flatten()
                for row in current
            ]
            remaining = sum(int(pos.numel()) for pos in masked_by_row)
            if remaining == 0:
                return DecodeResult(current, step, traces)
            decoded = total_initial - remaining
            schedule_state = self.config.schedule.state(decoded, total_initial)
            out = self.backbone.forward(current, attention_mask, timestep=timestep, output_attentions=True)
            logits = out["logits"]
            attentions = out.get("attentions")
            if attentions is None:
                raise ValueError("AttentionGraphDecoder requires a backbone that returns attentions")
            input_ids_before = current.clone()
            selected_by_row: list[list[int]] = []
            selected_positions_per_row: list[torch.Tensor] = []
            selected_local_per_row: list[torch.Tensor] = []
            conflict_matrix_per_row: list[torch.Tensor | None] = []
            selected_confidences: list[torch.Tensor] = []
            for row, positions in enumerate(masked_by_row):
                if positions.numel() == 0:
                    selected_by_row.append([])
                    selected_positions_per_row.append(positions)
                    selected_local_per_row.append(positions)
                    conflict_matrix_per_row.append(None)
                    continue
                row_logits = logits[row, positions]
                confidence, margin, entropy = confidence_margin_entropy(row_logits)
                row_attn = tuple(layer[row : row + 1] for layer in attentions)
                conflict_matrix = attention_conflict_matrix(row_attn, positions, self.config.layers)
                scores = node_scores(confidence, margin, entropy)
                scores = torch.where(
                    confidence >= schedule_state.min_confidence,
                    scores,
                    torch.full_like(scores, -float("inf")),
                )
                budget = budget_from_fraction(int(positions.numel()), schedule_state.budget_fraction)
                if self.config.force_complete and step == self.config.max_steps - 1:
                    budget = int(positions.numel())
                    schedule_state = DecodeScheduleState(0.0, 1.0, 1.0)
                    scores = node_scores(confidence, margin, entropy)
                selection = greedy_conflict_aware_selection(
                    scores=scores,
                    conflict_prob=conflict_matrix,
                    budget=budget,
                    conflict_theta=schedule_state.conflict_theta,
                )
                if selection.selected_local.numel() == 0 and self.config.fallback_top1:
                    selection = greedy_conflict_aware_selection(
                        scores=node_scores(confidence, margin, entropy),
                        conflict_prob=conflict_matrix,
                        budget=1,
                        conflict_theta=1.0,
                    )
                selected_positions = positions[selection.selected_local]
                top_tokens = row_logits.argmax(dim=-1)[selection.selected_local]
                current[row, selected_positions] = top_tokens
                selected_by_row.append([int(x) for x in selected_positions.detach().cpu()])
                selected_positions_per_row.append(selected_positions)
                selected_local_per_row.append(selection.selected_local)
                conflict_matrix_per_row.append(conflict_matrix)
                if selection.selected_local.numel() > 0:
                    selected_confidences.append(confidence[selection.selected_local])
            mean_conf = float(torch.cat(selected_confidences).mean().item()) if selected_confidences else 0.0
            step_metrics: dict[str, float] = {}
            if self.metrics_hooks:
                ctx = StepContext(
                    step=step,
                    input_ids_before=input_ids_before,
                    attention_mask=attention_mask,
                    pre_step_logits=logits,
                    masked_positions_per_row=masked_by_row,
                    selected_positions_per_row=selected_positions_per_row,
                    selected_local_per_row=selected_local_per_row,
                    conflict_matrix_per_row=conflict_matrix_per_row,
                    clean_input_ids=clean_input_ids,
                    backbone=self.backbone,
                )
                step_metrics = run_metrics_hooks(self.metrics_hooks, ctx)
            traces.append(
                DecodeStepTrace(
                    step=step,
                    masked_remaining=remaining,
                    selected_positions=selected_by_row,
                    mean_confidence=mean_conf,
                    metrics=step_metrics,
                )
            )
        return DecodeResult(current, self.config.max_steps, traces, {"stopped_by": "max_steps"})
