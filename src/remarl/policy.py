"""ReMaRL's per-position residual MLP, compatible with the released weights."""

from dataclasses import dataclass, fields
import torch
from torch import nn


@dataclass(frozen=True)
class PolicyConfig:
    input_node_dim: int = 3
    hidden_dim: int = 64
    architecture: str = "mlp"
    linear_skip: bool = True
    position_encoding: str = "learned"
    position_dim: int = 16
    position_count: int = 256
    position_column: int = 2
    feature_encoding: str = "none"
    backbone_hidden_dim: int = 4096
    hidden_projection_dim: int = 64

    @classmethod
    def from_dict(cls, values):
        # Older checkpoints also record unused attention-architecture options.
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in names})


class ReMaRLPolicy(nn.Module):
    def __init__(self, config=PolicyConfig()):
        super().__init__()
        self.config = config
        if (
            config.architecture != "mlp"
            or config.input_node_dim != 3
            or config.position_column != 2
            or config.feature_encoding != "none"
            or config.position_encoding != "learned"
        ):
            raise ValueError("ReMaRL expects confidence, mask state, position, and hidden features")
        self.position_embedding = nn.Embedding(config.position_count, config.position_dim)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        width = 2 + config.position_dim
        if config.backbone_hidden_dim:
            self.hidden_projection = nn.Sequential(
                nn.LayerNorm(config.backbone_hidden_dim),
                nn.Linear(config.backbone_hidden_dim, config.hidden_projection_dim),
            )
            width += config.hidden_projection_dim
        self.pointwise = nn.Sequential(
            nn.Linear(width, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        nn.init.zeros_(self.pointwise[-1].weight)
        nn.init.zeros_(self.pointwise[-1].bias)
        if config.linear_skip:
            self.linear_skip = nn.Linear(width, 1, bias=False)
            nn.init.zeros_(self.linear_skip.weight)

    def delta(self, features):
        if features.numel() == 0:
            return features.new_zeros((0,))
        position = self.position_embedding(features[:, 2].long()).to(features.dtype)
        parts = [features[:, :2], position]
        if self.config.backbone_hidden_dim:
            if features.shape[-1] != 3 + self.config.backbone_hidden_dim:
                raise ValueError("Missing final-layer backbone features")
            parts.append(self.hidden_projection(features[:, 3:].detach()))
        encoded = torch.cat(parts, dim=-1)
        correction = self.pointwise(encoded).squeeze(-1)
        if self.config.linear_skip:
            correction = correction + self.linear_skip(encoded).squeeze(-1)
        return correction

    def forward(self, features, confidence):
        return confidence + self.delta(features)
