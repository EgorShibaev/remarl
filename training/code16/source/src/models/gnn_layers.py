"""Alternative message-passing layers and set encoders for the set-level selector.

All graph layers share the convention used by ``GatedMessagePassingLayer`` in
[conflict_gnn.py]: ``edge_index`` is stored once ([2, E]) and symmetrized inside
the layer (messages flow both ways). They take projected node states
``[n, node_dim]``, an edge index ``[2, E]``, and projected edge states
``[E, edge_dim]`` and return updated node states ``[n, node_dim]``.

Implemented here (Pack G1 architecture sweep):
- ``PNALayer``  : Principal Neighbourhood Aggregation (multi-aggregator + degree
  scalers) - exposes graph statistics (mean/max/min/std) explicitly.
- ``GINELayer`` : edge-conditioned message passing (GINE-style), messages depend
  on edge features and a learnable ``eps`` self-weight.
- ``GATLayer``  : multi-head masked attention over neighbours, optional top-k
  neighbour sparsification.
- ``SetTransformerHead`` : permutation-invariant set encoder (SAB + PMA) used as a
  non-GNN set baseline (ignores edges).
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.conflict_gnn import GatedMessagePassingLayer

PNA_AGGREGATORS = ("mean", "max", "min", "std")


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = src.new_zeros((n,) + tuple(src.shape[1:]))
    out.index_add_(0, index, src)
    return out


def _scatter_reduce(src: torch.Tensor, index: torch.Tensor, n: int, reduce: str) -> torch.Tensor:
    out = src.new_zeros((n,) + tuple(src.shape[1:]))
    idx = index.view((-1,) + (1,) * (src.dim() - 1)).expand_as(src)
    out.scatter_reduce_(0, idx, src, reduce=reduce, include_self=False)
    return out


def _both_directions(edge_index: torch.Tensor, edge_states: torch.Tensor):
    src, dst = edge_index
    send = torch.cat([src, dst], dim=0)
    recv = torch.cat([dst, src], dim=0)
    edges = torch.cat([edge_states, edge_states], dim=0)
    return send, recv, edges


class PNALayer(nn.Module):
    """Principal Neighbourhood Aggregation layer.

    Per-edge messages are aggregated with several aggregators (mean/max/min/std)
    and scaled by degree scalers (identity / amplification / attenuation), which
    lets the set scorer condition on graph statistics beyond the mean (e.g. one
    outlier conflict, dense conflict clusters).
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        dropout: float,
        aggregators: tuple[str, ...] = PNA_AGGREGATORS,
    ) -> None:
        super().__init__()
        self.aggregators = tuple(aggregators)
        pair_dim = node_dim * 2 + edge_dim
        self.message = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim),
        )
        n_scalers = 3  # identity, amplification, attenuation
        self.post = nn.Linear(node_dim * len(self.aggregators) * n_scalers, node_dim)
        self.update = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, node_states: torch.Tensor, edge_index: torch.Tensor, edge_states: torch.Tensor) -> torch.Tensor:
        if edge_states.numel() == 0 or edge_index.numel() == 0:
            return node_states
        n = node_states.shape[0]
        send, recv, edges = _both_directions(edge_index, edge_states)
        pair = torch.cat([node_states[send], node_states[recv], edges], dim=-1)
        m = self.message(pair)
        deg = _scatter_sum(torch.ones(send.shape[0], device=m.device, dtype=m.dtype), recv, n).clamp(min=1.0)
        aggs = []
        for name in self.aggregators:
            if name == "mean":
                aggs.append(_scatter_sum(m, recv, n) / deg.unsqueeze(-1))
            elif name == "max":
                aggs.append(_scatter_reduce(m, recv, n, "amax"))
            elif name == "min":
                aggs.append(_scatter_reduce(m, recv, n, "amin"))
            elif name == "std":
                mean = _scatter_sum(m, recv, n) / deg.unsqueeze(-1)
                sq = _scatter_sum(m * m, recv, n) / deg.unsqueeze(-1)
                aggs.append((sq - mean * mean).clamp(min=0.0).add(1e-9).sqrt())
            else:
                raise ValueError(f"Unknown PNA aggregator: {name}")
        agg = torch.cat(aggs, dim=-1)
        log_deg = torch.log1p(deg)
        delta = log_deg.mean().clamp(min=1e-6)
        amp = (log_deg / delta).unsqueeze(-1)
        att = (delta / log_deg.clamp(min=1e-6)).unsqueeze(-1)
        scaled = torch.cat([agg, agg * amp, agg * att], dim=-1)
        out = self.post(scaled)
        update = self.update(torch.cat([node_states, out], dim=-1))
        return self.norm(node_states + update)


class GINELayer(nn.Module):
    """Edge-conditioned (GINE-style) message passing.

    Messages are an MLP of ``[h_i, h_j, e_ij]`` (edge features participate in the
    message, not only in the readout), summed over neighbours, with a learnable
    self-weight ``eps``.
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        pair_dim = node_dim * 2 + edge_dim
        self.message = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim),
        )
        self.eps = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, node_states: torch.Tensor, edge_index: torch.Tensor, edge_states: torch.Tensor) -> torch.Tensor:
        if edge_states.numel() == 0 or edge_index.numel() == 0:
            return node_states
        n = node_states.shape[0]
        send, recv, edges = _both_directions(edge_index, edge_states)
        pair = torch.cat([node_states[send], node_states[recv], edges], dim=-1)
        m = self.message(pair)
        agg = _scatter_sum(m, recv, n)
        out = self.update((1.0 + self.eps) * node_states + agg)
        return self.norm(node_states + out)


class GATLayer(nn.Module):
    """Multi-head graph attention with edge features.

    Attention logits are computed from ``[Wh_i, Wh_j, We_ij]`` per head and
    softmax-normalised over each node's incoming neighbours. ``topk > 0`` keeps
    only the top-k incoming edges per node before the softmax (self-contained
    sparsification; structural sparsity from syntax is deferred to Phase 3).
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        dropout: float,
        heads: int = 4,
        topk: int = 0,
    ) -> None:
        super().__init__()
        self.heads = max(1, int(heads))
        self.head_dim = max(1, node_dim // self.heads)
        self.out_dim = self.head_dim * self.heads
        self.topk = int(topk)
        self.val = nn.Linear(node_dim, self.out_dim)
        self.edge_lin = nn.Linear(edge_dim, self.out_dim)
        self.attn = nn.Linear(self.head_dim * 3, 1)
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(self.out_dim, node_dim)
        self.norm = nn.LayerNorm(node_dim)

    def _scatter_softmax(self, logit: torch.Tensor, recv: torch.Tensor, n: int) -> torch.Tensor:
        maxes = _scatter_reduce(logit, recv, n, "amax")
        centered = logit - maxes[recv]
        exp = centered.exp()
        denom = _scatter_sum(exp, recv, n)[recv] + 1e-16
        return exp / denom

    def _topk_mask(self, score: torch.Tensor, recv: torch.Tensor, n: int) -> torch.Tensor:
        keep = torch.ones(score.shape[0], dtype=torch.bool, device=score.device)
        for node in torch.unique(recv):
            idx = (recv == node).nonzero(as_tuple=False).flatten()
            if idx.numel() <= self.topk:
                continue
            order = torch.argsort(score[idx], descending=True)
            drop = idx[order[self.topk:]]
            keep[drop] = False
        return keep

    def forward(self, node_states: torch.Tensor, edge_index: torch.Tensor, edge_states: torch.Tensor) -> torch.Tensor:
        if edge_states.numel() == 0 or edge_index.numel() == 0:
            return node_states
        n = node_states.shape[0]
        send, recv, edges = _both_directions(edge_index, edge_states)
        vh = self.val(node_states).view(n, self.heads, self.head_dim)
        ee = self.edge_lin(edges).view(-1, self.heads, self.head_dim)
        feat = torch.cat([vh[send], vh[recv], ee], dim=-1)
        logit = self.leaky(self.attn(feat).squeeze(-1))  # [2E, heads]
        if self.topk > 0:
            keep = self._topk_mask(logit.mean(dim=-1), recv, n)
            logit = logit.masked_fill(~keep.unsqueeze(-1), float("-inf"))
        alpha = self.dropout(self._scatter_softmax(logit, recv, n))  # [2E, heads]
        msg = alpha.unsqueeze(-1) * vh[send]  # [2E, heads, head_dim]
        agg = _scatter_sum(msg, recv, n).reshape(n, self.out_dim)
        out = self.proj(agg)
        return self.norm(node_states + out)


def make_message_layer(mode: str, node_dim, edge_dim, hidden_dim, dropout, config) -> nn.Module:
    if mode == "gnn":
        return GatedMessagePassingLayer(node_dim, edge_dim, hidden_dim, dropout)
    if mode == "pna":
        return PNALayer(node_dim, edge_dim, hidden_dim, dropout, aggregators=tuple(config.pna_aggregators))
    if mode == "gine":
        return GINELayer(node_dim, edge_dim, hidden_dim, dropout)
    if mode == "gat":
        return GATLayer(node_dim, edge_dim, hidden_dim, dropout, heads=config.gat_heads, topk=config.gat_topk)
    raise ValueError(f"Unknown message-passing mode: {mode}")


class _SAB(nn.Module):
    """Set Attention Block: self-attention over the set + feed-forward."""

    def __init__(self, dim: int, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _ = self.mha(x, x, x, need_weights=False)
        x = self.n1(x + a)
        return self.n2(x + self.ff(x))


class SetTransformerHead(nn.Module):
    """Permutation-invariant set encoder: SAB stack + pooling-by-attention (PMA).

    Operates on already-projected node states ``[m, dim]`` (edges ignored);
    returns a scalar set score. A strong non-GNN set baseline.
    """

    def __init__(self, dim: int, hidden_dim: int, heads: int, dropout: float, blocks: int = 1) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_SAB(dim, hidden_dim, heads, dropout) for _ in range(max(1, blocks))])
        self.seed = nn.Parameter(torch.randn(1, 1, dim))
        self.pma = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        x = nodes.unsqueeze(0)  # [1, m, dim]
        for block in self.blocks:
            x = block(x)
        seed = self.seed.expand(x.shape[0], -1, -1)
        pooled, _ = self.pma(seed, x, x, need_weights=False)
        pooled = self.norm(pooled.squeeze(1))  # [1, dim]
        return self.head(pooled).reshape(())
