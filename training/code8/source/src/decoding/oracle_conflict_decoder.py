"""Oracle conflict decoder - analysis-only upper bound.

Instead of a learned conflict head, the conflict matrix is computed from the
TRUE counterfactual signal: reveal the ground-truth token at position i and check
whether the distribution at position j moves (JS > tau). This answers "how much
could a perfect conflict graph buy us?". If the oracle barely beats confidence,
the domain/setting has no headroom; if the oracle wins big but the GNN does not,
the gap is a model/training problem.

Cost is bounded by only scoring conflicts among the top-``max_candidates``
highest-confidence masked positions each step (one backbone forward per
candidate). Requires ``clean_input_ids`` (the ground truth).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.backbones.base import DiffusionBackbone
from src.decoding.confidence_decoder import DecodeResult, DecodeStepTrace
from src.decoding.schedules import DecodeScheduleState, ThreeStageSchedule, budget_from_fraction
from src.decoding.selection import greedy_conflict_aware_selection, node_scores
from src.eval.metrics import MetricsHook, StepContext, confidence_margin_entropy, run_metrics_hooks
from src.labels.dependency_scores import js_divergence_from_logits


@dataclass
class OracleDecoderConfig:
    max_steps: int = 16
    force_complete: bool = True
    fallback_top1: bool = True
    tau: float = 0.05
    topk_for_js: int | None = 128
    max_candidates: int = 24
    schedule: ThreeStageSchedule = field(default_factory=ThreeStageSchedule)


class OracleConflictDecoder:
    def __init__(
        self,
        backbone: DiffusionBackbone,
        config: OracleDecoderConfig,
        metrics_hooks: list[MetricsHook] | None = None,
    ) -> None:
        self.backbone = backbone
        self.config = config
        self.metrics_hooks: list[MetricsHook] = list(metrics_hooks or [])

    @torch.no_grad()
    def _oracle_matrix(
        self,
        row_input_ids: torch.Tensor,
        attention_mask_row: torch.Tensor,
        clean_row: torch.Tensor,
        positions: torch.Tensor,
        base_logits_full: torch.Tensor,
        candidate_local: torch.Tensor,
        timestep: int,
    ) -> torch.Tensor:
        n = int(positions.numel())
        matrix = base_logits_full.new_zeros((n, n))
        cand = candidate_local.tolist()
        for i_local in cand:
            anchor_pos = int(positions[i_local].item())
            cf_ids = row_input_ids.clone()
            cf_ids[0, anchor_pos] = clean_row[0, anchor_pos]
            cf_out = self.backbone.forward(cf_ids, attention_mask_row, timestep=timestep)
            cf_logits = cf_out["logits"][0]
            for j_local in cand:
                if j_local == i_local:
                    continue
                target_pos = int(positions[j_local].item())
                js = js_divergence_from_logits(
                    base_logits_full[target_pos].unsqueeze(0),
                    cf_logits[target_pos].unsqueeze(0),
                    topk=self.config.topk_for_js,
                )
                if float(js.item()) > self.config.tau:
                    matrix[i_local, j_local] = 1.0
                    matrix[j_local, i_local] = 1.0
        return matrix

    @torch.no_grad()
    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        timestep: int = 0,
        clean_input_ids: torch.Tensor | None = None,
    ) -> DecodeResult:
        if clean_input_ids is None:
            raise ValueError("OracleConflictDecoder requires clean_input_ids")
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
            logits = out["logits"]
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
                n = int(positions.numel())
                m = min(self.config.max_candidates, n)
                candidate_local = torch.topk(confidence, k=m).indices
                conflict_matrix = self._oracle_matrix(
                    row_input_ids=current[row : row + 1],
                    attention_mask_row=attention_mask[row : row + 1],
                    clean_row=clean_input_ids[row : row + 1],
                    positions=positions,
                    base_logits_full=logits[row],
                    candidate_local=candidate_local,
                    timestep=timestep,
                )
                scores = node_scores(confidence, margin, entropy)
                # restrict selection to scored candidates that pass min_confidence
                candidate_mask = torch.zeros(n, dtype=torch.bool, device=positions.device)
                candidate_mask[candidate_local] = True
                eligible = candidate_mask & (confidence >= schedule_state.min_confidence)
                scores = torch.where(eligible, scores, torch.full_like(scores, -float("inf")))
                budget = budget_from_fraction(n, schedule_state.budget_fraction)
                if self.config.force_complete and step == self.config.max_steps - 1:
                    budget = n
                    schedule_state = DecodeScheduleState(0.0, 1.0, 1.0)
                    scores = node_scores(confidence, margin, entropy)
                    scores = torch.where(candidate_mask, scores, torch.full_like(scores, -float("inf")))
                selection = greedy_conflict_aware_selection(
                    scores=scores,
                    conflict_prob=conflict_matrix,
                    budget=budget,
                    conflict_theta=schedule_state.conflict_theta,
                )
                if selection.selected_local.numel() == 0 and self.config.fallback_top1:
                    fallback_scores = node_scores(confidence, margin, entropy)
                    fallback_scores = torch.where(
                        candidate_mask, fallback_scores, torch.full_like(fallback_scores, -float("inf"))
                    )
                    selection = greedy_conflict_aware_selection(
                        scores=fallback_scores,
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
