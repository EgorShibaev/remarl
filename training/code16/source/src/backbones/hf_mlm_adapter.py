from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from src.backbones.base import BackboneOutput


@dataclass(frozen=True)
class HFBackboneConfig:
    name: str = "bert-base-uncased"
    mixed_precision: bool = False
    trust_remote_code: bool = False


class HuggingFaceMLMBackbone(nn.Module):
    """Adapter exposing a masked-LM as a diffusion-style backbone proxy."""

    def __init__(self, config: HFBackboneConfig) -> None:
        super().__init__()
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install transformers to use HuggingFaceMLMBackbone") from exc

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.name,
            trust_remote_code=config.trust_remote_code,
        )
        try:
            # eager attention is required for output_attentions (the attention-graph
            # decoder baseline); the sdpa default returns empty attentions.
            self.model = AutoModelForMaskedLM.from_pretrained(
                config.name,
                trust_remote_code=config.trust_remote_code,
                attn_implementation="eager",
            )
        except (TypeError, ValueError):
            self.model = AutoModelForMaskedLM.from_pretrained(
                config.name,
                trust_remote_code=config.trust_remote_code,
            )
        if self.tokenizer.mask_token_id is None:
            raise ValueError(f"Tokenizer has no mask token: {config.name}")
        if self.tokenizer.pad_token_id is None:
            raise ValueError(f"Tokenizer has no pad token: {config.name}")
        self.mask_token_id = int(self.tokenizer.mask_token_id)
        self.pad_token_id = int(self.tokenizer.pad_token_id)
        self.vocab_size = int(self.model.config.vocab_size)
        self.hidden_dim = int(self.model.config.hidden_size)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        timestep: torch.Tensor | int,
        output_attentions: bool = False,
    ) -> BackboneOutput:
        del timestep
        autocast_enabled = bool(self.config.mixed_precision and input_ids.device.type == "cuda")
        with torch.autocast(device_type=input_ids.device.type, enabled=autocast_enabled):
            outputs: Any = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=output_attentions,
                return_dict=True,
            )
        return {
            "hidden_states": outputs.hidden_states[-1],
            "logits": outputs.logits,
            "attentions": outputs.attentions if output_attentions else None,
        }

    def freeze(self) -> None:
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)


def build_hf_mlm_backbone(config: dict[str, Any]) -> HuggingFaceMLMBackbone:
    cfg = HFBackboneConfig(
        name=str(config.get("name", "bert-base-uncased")),
        mixed_precision=bool(config.get("mixed_precision", False)),
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
    backbone = HuggingFaceMLMBackbone(cfg)
    backbone.freeze()
    return backbone

