"""Joint per-position confidence refiner for learned P2-style decoding.

``setSelector_phi`` takes ALL editable-position features jointly and emits a refined
keep-confidence ``z_i`` for every editable position *simultaneously*. The decode rule
is P2: a top-k scheduler keeps the ``K_s`` highest-``z`` positions each step and remasks
the rest (see ``src/decoding/threshold_decoder.py`` and ``src/rl/rollout.py``).

Input is **confidence-only** (per the Apple P2 self-planner): each row is
``[score, is_masked]`` where ``score`` is confidence for masked positions and
keep-confidence for denoised ones (built by ``src.decoding.planner_features``), so
``input_node_dim == 2``. The module itself stays generic in ``input_node_dim`` so it can
also be driven by richer features in ablations.

Architecture: project node features, run a stack of Set Attention Blocks (``_SAB``)
so every position attends to every other editable position, then a per-token head.

Residual initialization (the whole point):

    Z_E = C_E + Delta_phi(features)

with the head's final linear zero-initialized so ``Delta_phi == 0`` at init, hence
``Z_E == C_E``. The initial policy is therefore *exactly* confidence/keep-confidence
self-planning (P2) and RL can only depart from it as ``phi`` learns.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from src.models.gnn_layers import _SAB


@dataclass(frozen=True)
class SetSelectorConfig:
    input_node_dim: int
    node_dim: int = 128
    hidden_dim: int = 128
    st_heads: int = 4
    st_blocks: int = 2
    dropout: float = 0.1
    architecture: str = "attention"
    linear_skip: bool = False
    position_encoding: str = "none"
    position_dim: int = 16
    position_count: int = 256
    position_column: int = -1
    feature_encoding: str = "none"
    feature_columns: tuple[int, ...] = ()
    feature_counts: tuple[int, ...] = ()
    feature_dim: int = 16
    backbone_hidden_dim: int = 0
    hidden_projection_dim: int = 64


class SetSelector(nn.Module):
    def __init__(self, config: SetSelectorConfig) -> None:
        super().__init__()
        self.config = config
        if config.position_encoding not in ('none', 'learned', 'sinusoidal'):
            raise ValueError('Unknown position encoding')
        input_dim = config.input_node_dim
        if config.feature_encoding not in ('none', 'learned', 'sinusoidal'):
            raise ValueError('Unknown feature encoding')
        if config.feature_encoding != 'none':
            if config.architecture != 'mlp' or config.position_encoding != 'none':
                raise ValueError('Generic feature encoding requires MLP without separate position encoding')
            columns = config.feature_columns
            if (not columns or len(set(columns)) != len(columns)
                or len(columns) != len(config.feature_counts)
                or any(c < 0 or c >= input_dim for c in columns)
                or any(n < 1 for n in config.feature_counts)
                or config.feature_dim < 2 or config.feature_dim % 2):
                raise ValueError('Invalid feature encoding specification')
            self.feature_embeddings = nn.ModuleList()
            for count in config.feature_counts:
                if config.feature_encoding == 'learned':
                    embedding = nn.Embedding(count, config.feature_dim)
                    nn.init.normal_(embedding.weight, std=.02)
                else:
                    pos = torch.arange(count).float().unsqueeze(1)
                    freq = torch.exp(torch.arange(0, config.feature_dim, 2).float()
                                     * (-math.log(10000.) / config.feature_dim))
                    table = torch.empty(count, config.feature_dim)
                    table[:, 0::2], table[:, 1::2] = torch.sin(pos*freq), torch.cos(pos*freq)
                    embedding = nn.Embedding.from_pretrained(table, freeze=True)
                self.feature_embeddings.append(embedding)
            input_dim += len(columns) * (config.feature_dim - 1)
        if config.position_encoding != 'none':
            if config.architecture != 'mlp' or not 0 <= config.position_column < input_dim:
                raise ValueError('Position encodings require MLP and explicit position_id column')
            if config.position_count < 1 or config.position_dim < 2 or config.position_dim % 2:
                raise ValueError('Position count must be positive and embedding dimension positive/even')
            if config.position_encoding == 'learned':
                self.position_embedding = nn.Embedding(config.position_count, config.position_dim)
                nn.init.normal_(self.position_embedding.weight, std=0.02)
            else:
                pos = torch.arange(config.position_count).float().unsqueeze(1)
                freq = torch.exp(torch.arange(0, config.position_dim, 2).float()
                                 * (-math.log(10000.0) / config.position_dim))
                table = torch.empty(config.position_count, config.position_dim)
                table[:, 0::2], table[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
                self.position_embedding = nn.Embedding.from_pretrained(table, freeze=True)
            input_dim = input_dim - 1 + config.position_dim
        if config.architecture == "linear":
            self.pointwise = nn.Linear(config.input_node_dim, 1, bias=False)
            nn.init.zeros_(self.pointwise.weight)
            return
        if config.architecture == "mlp":
            if config.backbone_hidden_dim:
                if config.position_encoding != 'learned' or config.feature_encoding != 'none' or config.input_node_dim != 3:
                    raise ValueError('Hidden ablation requires scalar confidence/mask and learned position')
                self.hidden_projection = nn.Sequential(nn.LayerNorm(config.backbone_hidden_dim),
                    nn.Linear(config.backbone_hidden_dim,config.hidden_projection_dim))
                input_dim += config.hidden_projection_dim
            self.pointwise = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim), nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim), nn.GELU(),
                nn.Linear(config.hidden_dim, 1),
            )
            nn.init.zeros_(self.pointwise[-1].weight)
            nn.init.zeros_(self.pointwise[-1].bias)
            if config.linear_skip:
                self.linear_skip = nn.Linear(input_dim, 1, bias=False)
                nn.init.zeros_(self.linear_skip.weight)
            return
        if config.architecture != "attention":
            raise ValueError(f"Unknown selector architecture: {config.architecture}")
        self.node_proj = nn.Sequential(
            nn.Linear(config.input_node_dim, config.node_dim),
            nn.LayerNorm(config.node_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.blocks = nn.ModuleList(
            [
                _SAB(config.node_dim, config.hidden_dim, config.st_heads, config.dropout)
                for _ in range(max(1, config.st_blocks))
            ]
        )
        # Per-token delta head. The FINAL linear is zero-initialized so that
        # Delta_phi == 0 at init and Z_M == C_M (confidence Fast-dLLM).
        self.head = nn.Sequential(
            nn.Linear(config.node_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        final = self.head[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def delta(self, node_features: torch.Tensor) -> torch.Tensor:
        """Per-position refinement ``Delta_phi`` for ``node_features`` [m, Dn] -> [m]."""
        if node_features.numel() == 0:
            return node_features.new_zeros((0,))
        if self.config.architecture != "attention":
            hidden=None
            if self.config.backbone_hidden_dim:
                width=self.config.input_node_dim
                if node_features.shape[-1] != width+self.config.backbone_hidden_dim:
                    raise ValueError('Missing last-layer hidden features')
                hidden=self.hidden_projection(node_features[:,width:].detach())
                node_features=node_features[:,:width]
            if self.config.feature_encoding != 'none':
                raw_columns = [c for c in range(self.config.input_node_dim)
                               if c not in self.config.feature_columns]
                parts = [node_features[:, raw_columns]]
                for col, embedding in zip(self.config.feature_columns, self.feature_embeddings):
                    # Exact pre-action integer IDs, retained in rollout replay.
                    parts.append(embedding(node_features[:, col].long()).to(node_features.dtype))
                node_features = torch.cat(parts, dim=-1)
            if self.config.position_encoding != 'none':
                col = self.config.position_column
                # IDs are explicit answer-relative ranks from planner_features,
                # preserved in replay. Never infer IDs from a batch/subset length.
                ids = node_features[:, col].long()
                embedding = self.position_embedding(ids).to(node_features.dtype)
                node_features = torch.cat((node_features[:, :col],
                    node_features[:, col+1:], embedding), dim=-1)
            if hidden is not None:
                node_features=torch.cat([node_features,hidden],dim=-1)
            correction = self.pointwise(node_features).squeeze(-1)
            if hasattr(self, 'linear_skip'):
                correction = correction + self.linear_skip(node_features).squeeze(-1)
            return correction
        x = self.node_proj(node_features).unsqueeze(0)  # [1, m, node_dim]
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(0).squeeze(-1)  # [m]

    def forward(self, node_features: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        """Refined confidences ``Z_M = C_M + Delta_phi`` for the masked set.

        ``node_features`` [m, Dn] are the joint masked-position features (H_M etc.);
        ``confidence`` [m] is C_M. Returns Z_M [m].
        """
        if confidence.numel() == 0:
            return confidence.new_zeros((0,))
        return confidence + self.delta(node_features)

    def attach_hidden(self, features, hidden_row, editable_positions):
        if not self.config.backbone_hidden_dim:
            return features
        if hidden_row is None or hidden_row.ndim!=2 or hidden_row.shape[-1]!=self.config.backbone_hidden_dim:
            raise ValueError('Real final backbone hidden state is required; logits fallback is forbidden')
        hidden=hidden_row.index_select(0,editable_positions).detach().to(features.dtype)
        return torch.cat([features,hidden],dim=-1)
