from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ConflictGNNConfig:
    input_node_dim: int
    input_edge_dim: int
    node_dim: int = 256
    edge_dim: int = 64
    hidden_dim: int = 256
    layers: int = 2
    dropout: float = 0.1


class GatedMessagePassingLayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        pair_dim = node_dim * 2 + edge_dim
        self.message = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim),
        )
        self.gate = nn.Sequential(nn.Linear(pair_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.update = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, node_states: torch.Tensor, edge_index: torch.Tensor, edge_states: torch.Tensor) -> torch.Tensor:
        if edge_states.numel() == 0:
            return node_states
        src, dst = edge_index
        messages = node_states.new_zeros(node_states.shape)
        for sender, receiver in ((src, dst), (dst, src)):
            pair = torch.cat([node_states[sender], node_states[receiver], edge_states], dim=-1)
            gated = torch.sigmoid(self.gate(pair)) * self.message(pair)
            messages.index_add_(0, receiver, gated)
        update = self.update(torch.cat([node_states, messages], dim=-1))
        return self.norm(node_states + update)


class ConflictGNN(nn.Module):
    def __init__(self, config: ConflictGNNConfig) -> None:
        super().__init__()
        self.config = config
        self.node_proj = nn.Sequential(
            nn.Linear(config.input_node_dim, config.node_dim),
            nn.LayerNorm(config.node_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(config.input_edge_dim, config.edge_dim),
            nn.LayerNorm(config.edge_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.layers = nn.ModuleList(
            [
                GatedMessagePassingLayer(
                    node_dim=config.node_dim,
                    edge_dim=config.edge_dim,
                    hidden_dim=config.hidden_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.layers)
            ]
        )
        pair_dim = config.node_dim * 4 + config.edge_dim
        self.classifier = nn.Sequential(
            nn.Linear(pair_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        if edge_features.numel() == 0:
            return edge_features.new_empty((0,))
        nodes = self.node_proj(node_features)
        edges = self.edge_proj(edge_features)
        for layer in self.layers:
            nodes = layer(nodes, edge_index, edges)
        src, dst = edge_index
        h_src = nodes[src]
        h_dst = nodes[dst]
        pair = torch.cat([h_src, h_dst, (h_src - h_dst).abs(), h_src * h_dst, edges], dim=-1)
        return self.classifier(pair).squeeze(-1)

