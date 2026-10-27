"""Backbone registry: dispatch on ``backbone.type``.

All backbones implement the ``DiffusionBackbone`` Protocol (see ``base.py``), so
the graph / labels / training / decoding code is backbone-agnostic. Each backbone
has a different ``hidden_dim``, so a conflict head trained on one backbone is not
transferable to another - key checkpoints/configs/W&B runs by backbone.

Types:
- ``hf_mlm`` (default): any HF ``AutoModelForMaskedLM`` (e.g. bert-base-uncased).
- ``mdlm``: a masked-diffusion LM (kuleshov-group/MDLM family) via the MDLM adapter.
"""

from __future__ import annotations

from typing import Any


def build_backbone(config: dict[str, Any]):
    """Construct a backbone from a ``backbone`` config block."""
    backbone_type = str((config or {}).get("type", "hf_mlm"))
    if backbone_type == "hf_mlm":
        from src.backbones.hf_mlm_adapter import build_hf_mlm_backbone

        return build_hf_mlm_backbone(config)
    if backbone_type in ("mdlm", "llada", "dream", "diffusion"):
        # all real masked-diffusion LMs share the same adapter (logits + hidden
        # states at masked positions); LLaDA/Dream run via SDPA (no flash_attn).
        from src.backbones.mdlm_adapter import build_mdlm_backbone

        return build_mdlm_backbone(config)
    raise ValueError(f"Unknown backbone type: {backbone_type}")
