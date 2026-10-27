"""Set scorer for set-level oracle imitation.

Scores how unsafe a candidate unmask set S is to decode jointly, from the
subgraph induced by S (node features of S + intra-S edges/edge features).

Architectures (``SetScorerConfig.mode``), all sharing the exact same
``(node_features, edge_index, edge_features) -> scalar`` signature so the in-loop
decoder and DAgger collector are architecture-agnostic:

- ``pool``           : project node features, mean+max pool, MLP -> scalar (no graph).
- ``gnn``            : L gated message-passing layers over the S-subgraph, then pool.
- ``pna``            : multi-aggregator + degree-scaler message passing (PNA), then pool.
- ``gine``           : edge-conditioned (GINE) message passing, then pool.
- ``gat``            : multi-head graph attention, then pool.
- ``set_transformer``: permutation-invariant SAB+PMA set encoder (ignores edges).
- ``pairwise_agg``   : linear over pooled node + intra-S edge-feature stats (no
  message passing) - a non-graph capacity floor / control.

``pool``/``gnn``/``pna``/``gine``/``gat`` share the same readout, so only the
message-passing layer differs (a clean architecture ablation). The ``pool`` and
``gnn`` module layouts are unchanged for backward compatibility with existing
checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from src.models.conflict_gnn import GatedMessagePassingLayer
from src.models.gnn_layers import PNA_AGGREGATORS, SetTransformerHead, make_message_layer

GNN_FAMILY = ("gnn", "pna", "gine", "gat")


@dataclass(frozen=True)
class SetScorerConfig:
    input_node_dim: int
    input_edge_dim: int
    node_dim: int = 128
    edge_dim: int = 64
    hidden_dim: int = 128
    layers: int = 2
    dropout: float = 0.1
    mode: str = "gnn"  # pool | gnn | pna | gine | gat | set_transformer | pairwise_agg
    # architecture-specific (defaulted so existing checkpoints round-trip)
    pna_aggregators: tuple[str, ...] = field(default_factory=lambda: PNA_AGGREGATORS)
    gat_heads: int = 4
    gat_topk: int = 0
    st_heads: int = 4
    st_blocks: int = 1


class SetScorer(nn.Module):
    def __init__(self, config: SetScorerConfig) -> None:
        super().__init__()
        self.config = config
        mode = config.mode
        self.node_proj = nn.Sequential(
            nn.Linear(config.input_node_dim, config.node_dim),
            nn.LayerNorm(config.node_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        if mode in ("pool", "gnn"):
            # Preserve the original layout (node_proj / edge_proj / layers / head)
            # so previously trained pool & gnn checkpoints load unchanged.
            self.edge_proj = self._make_edge_proj()
            self.layers = nn.ModuleList(
                [
                    GatedMessagePassingLayer(config.node_dim, config.edge_dim, config.hidden_dim, config.dropout)
                    for _ in range(config.layers if mode == "gnn" else 0)
                ]
            )
            self.head = self._make_head()
        elif mode in GNN_FAMILY:
            self.edge_proj = self._make_edge_proj()
            self.layers = nn.ModuleList(
                [
                    make_message_layer(mode, config.node_dim, config.edge_dim, config.hidden_dim, config.dropout, config)
                    for _ in range(config.layers)
                ]
            )
            self.head = self._make_head()
        elif mode == "set_transformer":
            self.set_transformer = SetTransformerHead(
                dim=config.node_dim,
                hidden_dim=config.hidden_dim,
                heads=config.st_heads,
                dropout=config.dropout,
                blocks=config.st_blocks,
            )
        elif mode == "pairwise_agg":
            self.edge_proj = self._make_edge_proj()
            self.pair_head = nn.Linear(config.node_dim * 3 + config.edge_dim * 3, 1)
        else:
            raise ValueError(f"Unknown SetScorer mode: {mode}")

    def _make_edge_proj(self) -> nn.Module:
        c = self.config
        return nn.Sequential(
            nn.Linear(c.input_edge_dim, c.edge_dim),
            nn.LayerNorm(c.edge_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
        )

    def _make_head(self) -> nn.Module:
        c = self.config
        return nn.Sequential(
            nn.Linear(c.node_dim * 2, c.hidden_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """node_features [m, Dn] for the set S; edges are local to S (0..m-1).
        Returns a scalar predicted set cost (0-dim tensor).
        """
        nodes = self.node_proj(node_features)
        mode = self.config.mode

        if mode == "set_transformer":
            return self.set_transformer(nodes)

        if mode == "pairwise_agg":
            node_stat = torch.cat([nodes.mean(dim=0), nodes.amax(dim=0), nodes.std(dim=0, unbiased=False)], dim=-1)
            if edge_features.numel() > 0:
                edges = self.edge_proj(edge_features)
                edge_stat = torch.cat(
                    [edges.mean(dim=0), edges.amax(dim=0), edges.std(dim=0, unbiased=False)], dim=-1
                )
            else:
                edge_stat = nodes.new_zeros(self.config.edge_dim * 3)
            return self.pair_head(torch.cat([node_stat, edge_stat], dim=-1)).reshape(())

        if mode in GNN_FAMILY and edge_features.numel() > 0 and len(self.layers) > 0:
            edges = self.edge_proj(edge_features)
            for layer in self.layers:
                nodes = layer(nodes, edge_index, edges)
        pooled = torch.cat([nodes.mean(dim=0), nodes.amax(dim=0)], dim=-1)
        return self.head(pooled).squeeze(-1)
