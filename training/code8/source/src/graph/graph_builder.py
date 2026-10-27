from __future__ import annotations

from dataclasses import dataclass

import torch

from src.graph.features import (
    FeatureConfig,
    build_edge_features,
    build_node_features,
    build_node_features_from_cached,
)
from src.labels.pair_sampling import all_undirected_pairs


@dataclass
class ConflictGraph:
    node_features: torch.Tensor
    edge_features: torch.Tensor
    edge_index: torch.Tensor
    masked_positions: torch.Tensor
    node_aux: dict[str, torch.Tensor]


def build_full_graph(
    hidden_states: torch.Tensor,
    logits: torch.Tensor,
    masked_positions: torch.Tensor,
    timestep: int | float | torch.Tensor,
    config: FeatureConfig = FeatureConfig(),
    pairs: torch.Tensor | None = None,
) -> ConflictGraph:
    node_features, node_aux = build_node_features(
        hidden_states=hidden_states,
        logits=logits,
        masked_positions=masked_positions,
        timestep=timestep,
        config=config,
    )
    if pairs is None:
        pairs = all_undirected_pairs(int(masked_positions.numel()), masked_positions.device)
    else:
        pairs = pairs.to(device=masked_positions.device, dtype=torch.long)
    edge_features = build_edge_features(masked_positions, pairs, node_aux, config)
    return ConflictGraph(
        node_features=node_features,
        edge_features=edge_features,
        edge_index=pairs.t().contiguous(),
        masked_positions=masked_positions,
        node_aux=node_aux,
    )


def build_full_graph_from_cached(
    cached: dict,
    masked_positions: torch.Tensor,
    seq_len: int,
    timestep: int | float | torch.Tensor,
    config: FeatureConfig = FeatureConfig(),
    pairs: torch.Tensor | None = None,
) -> ConflictGraph:
    node_features, node_aux = build_node_features_from_cached(
        cached=cached,
        masked_positions=masked_positions,
        timestep=timestep,
        config=config,
        seq_len=seq_len,
    )
    if pairs is None:
        pairs = all_undirected_pairs(int(masked_positions.numel()), masked_positions.device)
    else:
        pairs = pairs.to(device=masked_positions.device, dtype=torch.long)
    edge_features = build_edge_features(masked_positions, pairs, node_aux, config)
    return ConflictGraph(
        node_features=node_features,
        edge_features=edge_features,
        edge_index=pairs.t().contiguous(),
        masked_positions=masked_positions,
        node_aux=node_aux,
    )


def edge_vector_to_matrix(
    edge_index: torch.Tensor,
    edge_values: torch.Tensor,
    num_nodes: int,
    fill_value: float = 0.0,
) -> torch.Tensor:
    matrix = edge_values.new_full((num_nodes, num_nodes), fill_value)
    if edge_index.numel() == 0:
        return matrix
    src = edge_index[0]
    dst = edge_index[1]
    matrix[src, dst] = edge_values
    matrix[dst, src] = edge_values
    return matrix
