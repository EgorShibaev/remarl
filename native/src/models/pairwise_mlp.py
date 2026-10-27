from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PairwiseMLPConfig:
    input_node_dim: int
    input_edge_dim: int
    node_dim: int = 256
    edge_dim: int = 64
    hidden_dim: int = 256
    dropout: float = 0.1


class PairwiseMLP(nn.Module):
    def __init__(self, config: PairwiseMLPConfig) -> None:
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
        src, dst = edge_index
        h_src = nodes[src]
        h_dst = nodes[dst]
        pair = torch.cat([h_src, h_dst, (h_src - h_dst).abs(), h_src * h_dst, edges], dim=-1)
        return self.classifier(pair).squeeze(-1)

