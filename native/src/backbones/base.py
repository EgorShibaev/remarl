from __future__ import annotations

from typing import Any, Optional, Protocol, TypedDict

import torch


class BackboneOutput(TypedDict):
    hidden_states: torch.Tensor
    logits: torch.Tensor
    attentions: Optional[Any]


class DiffusionBackbone(Protocol):
    mask_token_id: int
    pad_token_id: int
    vocab_size: int
    hidden_dim: int

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        timestep: torch.Tensor | int,
        output_attentions: bool = False,
    ) -> BackboneOutput:
        ...

