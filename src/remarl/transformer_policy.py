"""Jazbec-style one-block scorer with the original ReMaRL feature interface.

The RoPE DiT block is vendored unchanged from apple/ml-rl-dllm. We retain
ReMaRL's learned position/hidden projections, confidence residual and linear
skip. Mask conditioning uses the existing mask input; no timestep is added.
This is an architecture ablation within ReMaRL, not a reproduction of the
Jazbec et al. Bernoulli decoder or its confidence-only input policy.
"""
from dataclasses import replace

import torch
from torch import nn

from .apple_policy_layers import RoPEDiTBlock
from .policy import ReMaRLPolicy as MLPSelector


ARCHITECTURE = "remarl_jazbec_dit"


def transformer_config(config):
    return replace(config, architecture=ARCHITECTURE, node_dim=128,
                   st_heads=2, st_blocks=1, dropout=0.0)


class TransformerPolicy(MLPSelector):
    def __init__(self, config):
        assert config.architecture == ARCHITECTURE
        assert config.input_node_dim == 3 and config.position_column == 2
        assert config.position_encoding == "learned" and config.feature_encoding == "none"
        assert config.backbone_hidden_dim == 4096 and config.hidden_projection_dim == 64
        assert config.position_dim == 16 and config.position_count == 256
        assert config.linear_skip and config.dropout == 0
        assert config.node_dim == 128 and config.st_heads == 2 and config.st_blocks == 1
        assert config.confidence_residual and config.zero_initialize_output
        # Construct shared input modules in the same RNG order as the MLP.
        super().__init__(replace(config, architecture="mlp"))
        self.config = config
        del self.pointwise
        self.input_projection = nn.Linear(82, 128)
        self.mask_conditioning = nn.Embedding(2, 128)
        self.transformer = RoPEDiTBlock(
            d_model=128, nhead=2, dim_feedforward=512, dropout=0.0,
            batch_first=True, activation="gelu")
        self.final_norm = nn.LayerNorm(128)
        self.output_projection = nn.Linear(128, 1)
        nn.init.zeros_(self.transformer.ada_conditioning.weight)
        nn.init.zeros_(self.transformer.ada_conditioning.bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        assert torch.count_nonzero(self.linear_skip.weight) == 0

    def delta(self, features):
        if features.ndim != 3 or features.shape[1:] != (256, 4099):
            raise ValueError("Transformer requires [answers, 256 positions, 4099 features]")
        positions = features[..., 2].long()
        expected = torch.arange(256, device=features.device).expand_as(positions)
        if not torch.equal(positions, expected):
            raise ValueError("Each sequence must contain one complete, ordered answer")
        hidden = self.hidden_projection(features[..., 3:].detach())
        embedded = self.position_embedding(positions).to(features.dtype)
        inputs = torch.cat((features[..., :2], embedded, hidden), dim=-1)
        condition = self.mask_conditioning(features[..., 1].long())
        x = self.transformer(self.input_projection(inputs), condition)
        return (self.output_projection(self.final_norm(x)).squeeze(-1)
                + self.linear_skip(inputs).squeeze(-1))

    def forward(self, features, confidence):
        if confidence.ndim != 2 or features.shape[:2] != confidence.shape:
            raise ValueError("Answer boundaries must be explicit in both inputs")
        return confidence + self.delta(features)
