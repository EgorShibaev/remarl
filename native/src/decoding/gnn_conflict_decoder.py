from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from src.backbones.base import DiffusionBackbone
from src.decoding.confidence_decoder import DecodeResult, DecodeStepTrace
from src.decoding.schedules import DecodeScheduleState, ThreeStageSchedule, budget_from_fraction
from src.decoding.selection import (
    SelectionResult,
    budget_only_selection,
    greedy_conflict_aware_selection,
    max_weight_independent_set_selection,
    node_scores,
    penalty_conflict_aware_selection,
    stochastic_conflict_aware_selection,
    threshold_only_selection,
)
from src.eval.metrics import MetricsHook, StepContext, run_metrics_hooks
from src.graph.features import FeatureConfig, FeatureSpec
from src.graph.graph_builder import build_full_graph, edge_vector_to_matrix
from src.models.conflict_gnn import ConflictGNN, ConflictGNNConfig
from src.models.pairwise_mlp import PairwiseMLP, PairwiseMLPConfig
from src.utils.checkpointing import load_checkpoint


@dataclass
class GNNDecoderConfig:
    max_steps: int = 16
    force_complete: bool = True
    fallback_top1: bool = True
    selection: str = "greedy"  # greedy | penalty | mwis | stochastic | budget_only | threshold_only
    gamma: float = 0.5  # penalty budget for the "penalty" selection
    temperature: float = 1.0  # for "stochastic" selection
    schedule: ThreeStageSchedule = field(default_factory=ThreeStageSchedule)


def select_with_strategy(
    strategy: str,
    scores: torch.Tensor,
    conflict_prob: torch.Tensor,
    budget: int,
    conflict_theta: float,
    gamma: float = 0.5,
    temperature: float = 1.0,
) -> SelectionResult:
    if strategy == "greedy":
        return greedy_conflict_aware_selection(scores, conflict_prob, budget, conflict_theta=conflict_theta)
    if strategy == "penalty":
        return penalty_conflict_aware_selection(scores, conflict_prob, budget, gamma=gamma)
    if strategy == "mwis":
        return max_weight_independent_set_selection(scores, conflict_prob, budget, conflict_theta=conflict_theta)
    if strategy == "stochastic":
        return stochastic_conflict_aware_selection(
            scores, conflict_prob, budget, conflict_theta=conflict_theta, temperature=temperature
        )
    if strategy == "budget_only":
        return budget_only_selection(scores, conflict_prob, budget)
    if strategy == "threshold_only":
        return threshold_only_selection(scores, conflict_prob, budget, conflict_theta=conflict_theta)
    raise ValueError(f"Unknown selection strategy: {strategy}")


def build_conflict_model_from_checkpoint(path: str, device: torch.device) -> tuple[nn.Module, FeatureConfig]:
    checkpoint = load_checkpoint(path, map_location=device)
    model_cfg = checkpoint.get("model_config", {})
    feature_config = FeatureConfig.from_raw(checkpoint.get("feature_config", {}))
    input_node_dim = int(checkpoint["input_node_dim"])
    input_edge_dim = int(checkpoint["input_edge_dim"])
    spec_raw = checkpoint.get("feature_spec")
    if spec_raw is not None:
        spec = FeatureSpec.from_feature_config(int(spec_raw["hidden_dim"]), feature_config)
        if spec.node_dim != input_node_dim or spec.edge_dim != input_edge_dim:
            raise ValueError(
                "Checkpoint feature_spec disagrees with saved input dims: "
                f"spec=({spec.node_dim},{spec.edge_dim}) "
                f"saved=({input_node_dim},{input_edge_dim})"
            )
    common = {
        "input_node_dim": input_node_dim,
        "input_edge_dim": input_edge_dim,
        "node_dim": int(model_cfg.get("node_dim", 256)),
        "edge_dim": int(model_cfg.get("edge_dim", 64)),
        "hidden_dim": int(model_cfg.get("hidden_dim", 256)),
        "dropout": float(model_cfg.get("dropout", 0.1)),
    }
    if model_cfg.get("type", "gnn") == "pairwise_mlp":
        model: nn.Module = PairwiseMLP(PairwiseMLPConfig(**common))
    else:
        model = ConflictGNN(
            ConflictGNNConfig(**common, layers=int(model_cfg.get("layers", 2)))
        )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, feature_config


class GNNConflictDecoder:
    def __init__(
        self,
        backbone: DiffusionBackbone,
        conflict_model: nn.Module,
        feature_config: FeatureConfig,
        config: GNNDecoderConfig,
        metrics_hooks: list[MetricsHook] | None = None,
    ) -> None:
        self.backbone = backbone
        self.conflict_model = conflict_model
        self.feature_config = feature_config
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
            out = self.backbone.forward(current, attention_mask, timestep=timestep)
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
                graph = build_full_graph(
                    hidden_states=out["hidden_states"][row],
                    logits=out["logits"][row],
                    masked_positions=positions,
                    timestep=timestep,
                    config=self.feature_config,
                )
                confidence = graph.node_aux["confidence"]
                margin = graph.node_aux["margin"]
                entropy = graph.node_aux["entropy"]
                scores = node_scores(confidence, margin, entropy)
                scores = torch.where(
                    confidence >= schedule_state.min_confidence,
                    scores,
                    torch.full_like(scores, -float("inf")),
                )
                edge_logits = self.conflict_model(
                    graph.node_features,
                    graph.edge_index,
                    graph.edge_features,
                )
                conflict_matrix = edge_vector_to_matrix(
                    graph.edge_index,
                    torch.sigmoid(edge_logits),
                    num_nodes=int(positions.numel()),
                )
                budget = budget_from_fraction(int(positions.numel()), schedule_state.budget_fraction)
                if self.config.force_complete and step == self.config.max_steps - 1:
                    budget = int(positions.numel())
                    schedule_state = DecodeScheduleState(0.0, 1.0, 1.0)
                    scores = node_scores(confidence, margin, entropy)
                selection = select_with_strategy(
                    self.config.selection,
                    scores=scores,
                    conflict_prob=conflict_matrix,
                    budget=budget,
                    conflict_theta=schedule_state.conflict_theta,
                    gamma=self.config.gamma,
                    temperature=self.config.temperature,
                )
                if selection.selected_local.numel() == 0 and self.config.fallback_top1:
                    selection = greedy_conflict_aware_selection(
                        scores=node_scores(confidence, margin, entropy),
                        conflict_prob=conflict_matrix,
                        budget=1,
                        conflict_theta=1.0,
                    )
                row_logits = out["logits"][row, positions]
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
                    pre_step_logits=out["logits"],
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


def schedule_from_config(raw: dict[str, Any] | None) -> ThreeStageSchedule:
    if not raw:
        return ThreeStageSchedule()
    # flat override (single operating point) for calibration sweeps: a scalar
    # decoding.conflict_theta (+ optional budget_fraction / min_confidence)
    # applies the same values to all schedule stages.
    if any(k in raw for k in ("conflict_theta", "budget_fraction", "min_confidence")) and "early" not in raw:
        state = DecodeScheduleState(
            float(raw.get("min_confidence", 0.0)),
            float(raw.get("conflict_theta", 0.5)),
            float(raw.get("budget_fraction", 1.0)),
        )
        return ThreeStageSchedule(early=state, middle=state, late=state)
    early = raw.get("early", {})
    middle = raw.get("middle", {})
    late = raw.get("late", {})
    return ThreeStageSchedule(
        early=DecodeScheduleState(
            float(early.get("min_confidence", 0.80)),
            float(early.get("conflict_theta", 0.30)),
            float(early.get("budget_fraction", 0.20)),
        ),
        middle=DecodeScheduleState(
            float(middle.get("min_confidence", 0.65)),
            float(middle.get("conflict_theta", 0.45)),
            float(middle.get("budget_fraction", 0.40)),
        ),
        late=DecodeScheduleState(
            float(late.get("min_confidence", 0.40)),
            float(late.get("conflict_theta", 0.60)),
            float(late.get("budget_fraction", 1.00)),
        ),
        early_progress_lt=float(early.get("progress_lt", 0.30)),
        middle_progress_lt=float(middle.get("progress_lt", 0.80)),
    )

