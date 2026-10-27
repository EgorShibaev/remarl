from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.data.masking import mask_run_features
from src.eval.metrics import confidence_margin_entropy
from src.labels.pair_sampling import same_span_mask, topk_overlap_at_pairs


@dataclass(frozen=True)
class FeatureConfig:
    max_length: int = 128
    top_k: int = 8
    distance_buckets: int = 16
    # feature-ablation toggles (all default True => current behavior). top_k=0
    # disables the top-k probability block.
    include_hidden: bool = True
    include_scalar: bool = True  # confidence / margin / entropy
    include_timestep: bool = True
    include_position: bool = True
    include_runs: bool = True  # mask-run local-context features
    include_edge_features: bool = True

    @classmethod
    def from_raw(cls, raw: dict | None) -> "FeatureConfig":
        raw = dict(raw or {})
        # named ablation presets (used by the feature-ablation sweep)
        presets: dict[str, dict[str, Any]] = {
            "full": {},
            "no_edge": {"include_edge_features": False},
            "no_topk": {"top_k": 0},
            "hidden_only": {
                "include_scalar": False,
                "include_timestep": False,
                "include_position": False,
                "include_runs": False,
                "top_k": 0,
            },
            "hidden_plus_scalar": {
                "include_timestep": False,
                "include_position": False,
                "include_runs": False,
                "top_k": 0,
            },
        }
        ablation = raw.pop("ablation", None)
        if ablation is not None:
            if ablation not in presets:
                raise ValueError(f"Unknown feature ablation preset: {ablation}")
            raw.update(presets[ablation])
        return cls(
            max_length=int(raw.get("max_length", 128)),
            top_k=int(raw.get("top_k", 8)),
            distance_buckets=int(raw.get("distance_buckets", 16)),
            include_hidden=bool(raw.get("include_hidden", True)),
            include_scalar=bool(raw.get("include_scalar", True)),
            include_timestep=bool(raw.get("include_timestep", True)),
            include_position=bool(raw.get("include_position", True)),
            include_runs=bool(raw.get("include_runs", True)),
            include_edge_features=bool(raw.get("include_edge_features", True)),
        )


@dataclass(frozen=True)
class FeatureSpec:
    """Single source of truth for graph-feature dimensions.

    Keeping this beside the feature builders means callers (training,
    decoding, checkpoint validation) never have to probe a real batch to
    discover input dims.
    """

    hidden_dim: int
    top_k: int
    distance_buckets: int
    include_hidden: bool = True
    include_scalar: bool = True
    include_timestep: bool = True
    include_position: bool = True
    include_runs: bool = True
    include_edge_features: bool = True

    @property
    def node_dim(self) -> int:
        dim = self.top_k
        if self.include_hidden:
            dim += self.hidden_dim
        if self.include_scalar:
            dim += 3  # confidence, margin, entropy
        if self.include_timestep:
            dim += 1
        if self.include_position:
            dim += 1
        if self.include_runs:
            dim += 4
        return dim

    @property
    def edge_dim(self) -> int:
        if not self.include_edge_features:
            return 1  # single constant column placeholder
        # distance buckets one-hot + signed_norm + abs_norm + same_span + conf_pair (2) + entropy_pair (2) + pair_conf + topk overlap
        return self.distance_buckets + 1 + 1 + 1 + 2 + 2 + 1 + 1

    @classmethod
    def from_feature_config(cls, hidden_dim: int, config: FeatureConfig) -> "FeatureSpec":
        return cls(
            hidden_dim=hidden_dim,
            top_k=config.top_k,
            distance_buckets=config.distance_buckets,
            include_hidden=config.include_hidden,
            include_scalar=config.include_scalar,
            include_timestep=config.include_timestep,
            include_position=config.include_position,
            include_runs=config.include_runs,
            include_edge_features=config.include_edge_features,
        )


def _assemble_node_features(
    row_hidden: torch.Tensor,
    confidence: torch.Tensor,
    margin: torch.Tensor,
    entropy: torch.Tensor,
    timestep_feature: torch.Tensor,
    position_feature: torch.Tensor,
    run_features: torch.Tensor,
    topk_probs: torch.Tensor,
    config: FeatureConfig,
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    if config.include_hidden:
        pieces.append(row_hidden)
    if config.include_scalar:
        pieces.append(torch.stack([confidence, margin, entropy], dim=-1))
    if config.include_timestep:
        pieces.append(timestep_feature)
    if config.include_position:
        pieces.append(position_feature)
    if config.include_runs:
        pieces.append(run_features)
    pieces.append(topk_probs)  # width 0 when top_k == 0
    return torch.cat(pieces, dim=-1)


def distance_bucket(abs_distance: torch.Tensor, bucket_count: int) -> torch.Tensor:
    clipped = abs_distance.clamp(min=0)
    bucket = torch.floor(torch.log2(clipped.float().clamp_min(1.0))).long()
    return bucket.clamp(max=bucket_count - 1)


def build_node_features(
    hidden_states: torch.Tensor,
    logits: torch.Tensor,
    masked_positions: torch.Tensor,
    timestep: int | float | torch.Tensor,
    config: FeatureConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if masked_positions.numel() == 0:
        spec = FeatureSpec.from_feature_config(hidden_states.shape[-1], config)
        empty = hidden_states.new_empty((0, spec.node_dim))
        return empty, {}
    row_hidden = hidden_states.index_select(0, masked_positions)
    row_logits = logits.index_select(0, masked_positions)
    confidence, margin, entropy = confidence_margin_entropy(row_logits)
    topk = torch.topk(torch.softmax(row_logits, dim=-1), k=min(config.top_k, row_logits.shape[-1]), dim=-1)
    topk_probs = topk.values
    if topk_probs.shape[-1] < config.top_k:
        pad = topk_probs.new_zeros((topk_probs.shape[0], config.top_k - topk_probs.shape[-1]))
        topk_probs = torch.cat([topk_probs, pad], dim=-1)
    if isinstance(timestep, torch.Tensor):
        timestep_value = float(timestep.flatten()[0].item())
    else:
        timestep_value = float(timestep)
    timestep_feature = row_hidden.new_full((masked_positions.numel(), 1), timestep_value)
    position_feature = masked_positions.float().unsqueeze(-1) / max(config.max_length - 1, 1)
    run_features = mask_run_features(masked_positions, hidden_states.shape[0])
    run_scale = row_hidden.new_tensor(
        [config.max_length, config.max_length, config.max_length, config.max_length]
    )
    run_features = run_features / run_scale.clamp_min(1.0)
    node_features = _assemble_node_features(
        row_hidden, confidence, margin, entropy, timestep_feature, position_feature,
        run_features, topk_probs, config,
    )
    aux = {
        "confidence": confidence,
        "margin": margin,
        "entropy": entropy,
        "topk_ids": topk.indices,
        "topk_probs": topk_probs,
    }
    return node_features, aux


def build_node_features_from_cached(
    cached: dict,
    masked_positions: torch.Tensor,
    timestep: int | float | torch.Tensor,
    config: FeatureConfig,
    seq_len: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Same outputs as :func:`build_node_features` but reads pre-computed
    backbone slices instead of running the backbone here.
    """
    device = masked_positions.device
    hidden = cached["hidden_at_masked"].to(device)
    if masked_positions.numel() == 0:
        spec = FeatureSpec.from_feature_config(int(hidden.shape[-1]), config)
        empty = torch.empty((0, spec.node_dim), dtype=torch.float32, device=device)
        return empty, {}
    confidence = cached["confidence"].to(device)
    margin = cached["margin"].to(device)
    entropy = cached["entropy"].to(device)
    topk_ids = cached["topk_ids"].to(device)
    topk_probs = cached["topk_probs"].to(device)
    if topk_probs.shape[-1] < config.top_k:
        pad = topk_probs.new_zeros((topk_probs.shape[0], config.top_k - topk_probs.shape[-1]))
        topk_probs = torch.cat([topk_probs, pad], dim=-1)
    if isinstance(timestep, torch.Tensor):
        timestep_value = float(timestep.flatten()[0].item())
    else:
        timestep_value = float(timestep)
    timestep_feature = hidden.new_full((masked_positions.numel(), 1), timestep_value)
    position_feature = masked_positions.float().unsqueeze(-1) / max(config.max_length - 1, 1)
    run_features = mask_run_features(masked_positions, seq_len)
    run_scale = hidden.new_tensor(
        [config.max_length, config.max_length, config.max_length, config.max_length]
    )
    run_features = run_features / run_scale.clamp_min(1.0)
    node_features = _assemble_node_features(
        hidden, confidence, margin, entropy, timestep_feature, position_feature,
        run_features, topk_probs, config,
    )
    aux = {
        "confidence": confidence,
        "margin": margin,
        "entropy": entropy,
        "topk_ids": topk_ids,
        "topk_probs": topk_probs,
    }
    return node_features, aux


def build_edge_features(
    masked_positions: torch.Tensor,
    pairs: torch.Tensor,
    node_aux: dict[str, torch.Tensor],
    config: FeatureConfig,
) -> torch.Tensor:
    if pairs.numel() == 0:
        spec = FeatureSpec.from_feature_config(0, config)
        return torch.empty((0, spec.edge_dim), dtype=torch.float32, device=masked_positions.device)
    if not config.include_edge_features:
        return torch.zeros((pairs.shape[0], 1), dtype=torch.float32, device=masked_positions.device)
    i = pairs[:, 0]
    j = pairs[:, 1]
    signed = masked_positions[j] - masked_positions[i]
    abs_distance = signed.abs()
    bucket = distance_bucket(abs_distance, config.distance_buckets)
    bucket_one_hot = torch.nn.functional.one_hot(bucket, num_classes=config.distance_buckets).float()
    same_span = same_span_mask(masked_positions, pairs).float().unsqueeze(-1)
    confidence = node_aux["confidence"]
    entropy = node_aux["entropy"]
    pair_conf = (confidence[i] * confidence[j]).unsqueeze(-1)
    conf_pair = torch.stack([confidence[i], confidence[j]], dim=-1)
    entropy_pair = torch.stack([entropy[i], entropy[j]], dim=-1)
    signed_norm = signed.float().unsqueeze(-1) / max(config.max_length - 1, 1)
    abs_norm = abs_distance.float().unsqueeze(-1) / max(config.max_length - 1, 1)
    overlap = topk_overlap_at_pairs(node_aux["topk_ids"], pairs).unsqueeze(-1)
    return torch.cat(
        [bucket_one_hot, signed_norm, abs_norm, same_span, conf_pair, entropy_pair, pair_conf, overlap],
        dim=-1,
    )
