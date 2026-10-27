from __future__ import annotations

from dataclasses import dataclass

import torch

from src.graph.graph_builder import ConflictGraph


@dataclass
class BatchedGraph:
    node_features: torch.Tensor
    edge_features: torch.Tensor
    edge_index: torch.Tensor
    graph_id_for_node: torch.Tensor
    graph_id_for_edge: torch.Tensor
    node_offsets: torch.Tensor


def batch_graphs(graphs: list[ConflictGraph]) -> BatchedGraph:
    node_features = []
    edge_features = []
    edge_indices = []
    graph_id_for_node = []
    graph_id_for_edge = []
    offsets = []
    node_offset = 0
    for graph_id, graph in enumerate(graphs):
        offsets.append(node_offset)
        node_count = graph.node_features.shape[0]
        edge_count = graph.edge_features.shape[0]
        node_features.append(graph.node_features)
        edge_features.append(graph.edge_features)
        if edge_count:
            edge_indices.append(graph.edge_index + node_offset)
        graph_id_for_node.append(torch.full((node_count,), graph_id, dtype=torch.long, device=graph.node_features.device))
        graph_id_for_edge.append(torch.full((edge_count,), graph_id, dtype=torch.long, device=graph.edge_features.device))
        node_offset += node_count
    device = graphs[0].node_features.device if graphs else torch.device("cpu")
    return BatchedGraph(
        node_features=torch.cat(node_features, dim=0) if node_features else torch.empty((0, 0), device=device),
        edge_features=torch.cat(edge_features, dim=0) if edge_features else torch.empty((0, 0), device=device),
        edge_index=torch.cat(edge_indices, dim=1) if edge_indices else torch.empty((2, 0), dtype=torch.long, device=device),
        graph_id_for_node=torch.cat(graph_id_for_node, dim=0) if graph_id_for_node else torch.empty(0, dtype=torch.long, device=device),
        graph_id_for_edge=torch.cat(graph_id_for_edge, dim=0) if graph_id_for_edge else torch.empty(0, dtype=torch.long, device=device),
        node_offsets=torch.tensor(offsets, dtype=torch.long, device=device),
    )

