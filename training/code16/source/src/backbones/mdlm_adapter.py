"""MDLM (masked diffusion LM) backbone adapter.

Wraps a kuleshov-group MDLM-style masked-diffusion checkpoint behind the
``DiffusionBackbone`` Protocol: ``forward`` returns last hidden state + token
logits at every position, plus a mask token id. Unlike the BERT proxy, the
``timestep`` argument is meaningful (it indexes the diffusion noise level); we
pass it through when the underlying model accepts it and otherwise ignore it.

MDLM checkpoints are published with custom modeling code, so loading uses
``trust_remote_code=True``. Because the exact module API varies across releases,
this adapter probes for the common shapes (``logits``/``hidden_states`` from a
HF-style output, or a tensor) and raises a clear error otherwise so the failure
is obvious during the E2 smoke step rather than deep in training.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from src.backbones.base import BackboneOutput


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _shim_apply_rotary_emb_qkv_(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *args, **kwargs):
    """SDPA-compatible drop-in for flash_attn.layers.rotary.apply_rotary_emb_qkv_.

    qkv: [b, s, 3, h, d]; cos/sin: [s, d/2] (GPT-NeoX / rotate_half convention,
    matching MDLM's apply_rotary_pos_emb slicing). Rotates q (idx 0) and k (idx 1),
    leaves v (idx 2) unchanged; modifies and returns qkv.
    """
    cos_f = torch.cat([cos, cos], dim=-1)[None, :, None, :].to(qkv.dtype)  # [1,s,1,d]
    sin_f = torch.cat([sin, sin], dim=-1)[None, :, None, :].to(qkv.dtype)
    q = qkv[:, :, 0]
    k = qkv[:, :, 1]
    qkv[:, :, 0] = q * cos_f + _rotate_half(q) * sin_f
    qkv[:, :, 1] = k * cos_f + _rotate_half(k) * sin_f
    return qkv


def _shim_flash_attn_varlen_qkvpacked(qkv, cu_seqlens, max_seqlen, dropout_p=0.0, *args, **kwargs):
    """SDPA drop-in for flash_attn_varlen_qkvpacked_func with UNIFORM seqlens.

    qkv: [(b*s), 3, h, d]; returns [(b*s), h, d]. MDLM calls this with
    ``seqlens=None`` -> equal-length sequences, so we reshape to [b, s, h, d] and
    run full (non-causal) scaled-dot-product attention per sequence.
    """
    causal = bool(kwargs.get("causal", args[1] if len(args) > 1 else False))
    total, three, h, d = qkv.shape
    s = int(max_seqlen)
    b = total // s
    qkv = qkv.view(b, s, three, h, d)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [b,s,h,d]
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [b,h,s,d]
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, dropout_p=dropout_p if torch.is_grad_enabled() else 0.0, is_causal=causal
    )
    out = out.transpose(1, 2).reshape(total, h, d)
    return out


def _install_flash_attn_sdpa_shim() -> bool:
    """Register an SDPA-backed fake ``flash_attn`` in sys.modules if the real one is
    unavailable. MDLM remote code hard-imports flash_attn (no wheel for our CUDA),
    but only uses rotary + non-causal varlen attention, both expressible via SDPA.
    Returns True if a shim was installed.
    """
    try:
        import flash_attn  # noqa: F401

        return False
    except Exception:  # noqa: BLE001 - any import failure -> shim
        pass
    import importlib.machinery as _m

    def _mod(name: str, is_pkg: bool = False) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__spec__ = _m.ModuleSpec(name, loader=None, is_package=is_pkg)
        if is_pkg:
            mod.__path__ = []  # mark as a package so submodule imports resolve
        return mod

    flash_attn = _mod("flash_attn", is_pkg=True)
    iface = _mod("flash_attn.flash_attn_interface")
    iface.flash_attn_varlen_qkvpacked_func = _shim_flash_attn_varlen_qkvpacked
    layers = _mod("flash_attn.layers", is_pkg=True)
    rotary = _mod("flash_attn.layers.rotary")
    rotary.apply_rotary_emb_qkv_ = _shim_apply_rotary_emb_qkv_
    layers.rotary = rotary
    flash_attn.flash_attn_interface = iface
    flash_attn.flash_attn_varlen_qkvpacked_func = _shim_flash_attn_varlen_qkvpacked
    flash_attn.layers = layers
    sys.modules["flash_attn"] = flash_attn
    sys.modules["flash_attn.flash_attn_interface"] = iface
    sys.modules["flash_attn.layers"] = layers
    sys.modules["flash_attn.layers.rotary"] = rotary
    print("[mdlm-adapter] installed SDPA flash_attn shim (real flash_attn unavailable)")
    return True


@dataclass(frozen=True)
class MDLMBackboneConfig:
    name: str = "kuleshov-group/mdlm-owt"
    mixed_precision: bool = False
    trust_remote_code: bool = True
    mask_token: str | None = None  # explicit mask token if tokenizer lacks one
    mask_token_id: int | None = None  # explicit mask token id (e.g. LLaDA 126336)
    tokenizer_name: str | None = None  # explicit fallback tokenizer (e.g. "bert-base-uncased")
    dtype: str | None = None  # e.g. "bfloat16" for large models (LLaDA/Dream 7-8B)


class MDLMBackbone(nn.Module):
    def __init__(self, config: MDLMBackboneConfig) -> None:
        super().__init__()
        from transformers import (
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForMaskedLM,
            AutoTokenizer,
        )

        self.config = config
        # MDLM remote code hard-imports flash_attn; install an SDPA shim when the
        # real package is unavailable (it only needs rotary + non-causal attention).
        _install_flash_attn_sdpa_shim()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.name, trust_remote_code=config.trust_remote_code
            )
        except (ValueError, OSError):
            # some diffusion-LM checkpoints ship a tokenizer that cannot be
            # converted to the fast backend; fall back to the slow tokenizer,
            # and finally to an explicit base tokenizer if configured.
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    config.name, trust_remote_code=config.trust_remote_code, use_fast=False
                )
            except (ValueError, OSError):
                if not config.tokenizer_name:
                    raise
                self.tokenizer = AutoTokenizer.from_pretrained(
                    config.tokenizer_name, trust_remote_code=config.trust_remote_code
                )
        # transformers 5.x expects models to define `all_tied_weights_keys`, which
        # older custom diffusion-LM code (LLaDA/Dream, written for 4.x) lacks. For
        # untied models (LLaDA-8B-Base has weight_tying=false) adding an empty one
        # is safe: there are no tied keys to reconcile and all weights load.
        if config.trust_remote_code:
            try:
                from transformers import AutoConfig
                from transformers.dynamic_module_utils import get_class_from_dynamic_module

                auto_map = getattr(
                    AutoConfig.from_pretrained(config.name, trust_remote_code=True), "auto_map", {}
                ) or {}
                ref = (
                    auto_map.get("AutoModelForCausalLM")
                    or auto_map.get("AutoModel")
                    or auto_map.get("AutoModelForMaskedLM")
                )
                if ref:
                    klass = get_class_from_dynamic_module(ref, config.name, trust_remote_code=True)
                    if not hasattr(klass, "all_tied_weights_keys"):
                        klass.all_tied_weights_keys = {}
                    # transformers 5 calls tie_weights(missing_keys=..., recompute_mapping=...);
                    # old code's signature rejects these. For untied models tie is a no-op.
                    _orig_tie = klass.tie_weights

                    def _tie(self, *args, **kwargs):  # noqa: ANN001
                        try:
                            return _orig_tie(self)
                        except TypeError:
                            return None

                    klass.tie_weights = _tie
            except Exception as exc:  # noqa: BLE001 - best-effort compat shim
                print(f"[mdlm-adapter] tied-keys compat shim skipped: {exc}")

        load_kwargs: dict[str, Any] = {"trust_remote_code": config.trust_remote_code}
        if config.dtype is not None:
            load_kwargs["torch_dtype"] = getattr(torch, config.dtype)
        self.model = None
        load_errors: list[str] = []
        # LLaDA/Dream register AutoModelForCausalLM; MDLM uses AutoModel. Try the
        # diffusion-LM order first.
        for auto_cls in (AutoModelForCausalLM, AutoModel, AutoModelForMaskedLM):
            try:
                self.model = auto_cls.from_pretrained(config.name, **load_kwargs)
                break
            except (ValueError, OSError, KeyError) as exc:
                load_errors.append(f"{auto_cls.__name__}: {str(exc)[:120]}")
        if self.model is None:
            raise ValueError(f"Could not load diffusion model {config.name}: {load_errors}")

        mask_id = config.mask_token_id
        if mask_id is None:
            mask_id = self.tokenizer.mask_token_id
        if mask_id is None and config.mask_token is not None:
            mask_id = self.tokenizer.convert_tokens_to_ids(config.mask_token)
        if mask_id is None:
            raise ValueError(
                f"No mask token for {config.name}; set backbone.mask_token_id or backbone.mask_token"
            )
        self.mask_token_id = int(mask_id)
        # GPT-2 (MDLM-owt's tokenizer) ships no pad token; reuse eos so batched
        # tokenization can pad. Padding positions are masked out by attention_mask.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.convert_ids_to_tokens(0)
        self.pad_token_id = int(self.tokenizer.pad_token_id or 0)
        model_config = self.model.config
        # old custom configs may lack standard defaults the forward reads
        for key, default in {"use_cache": False, "output_attentions": False}.items():
            if not hasattr(model_config, key):
                setattr(model_config, key, default)
        self.vocab_size = int(
            getattr(model_config, "vocab_size", None)
            or getattr(model_config, "embedding_size", 0)
            or int(self.model.get_output_embeddings().weight.shape[0])
        )
        self.hidden_dim = int(
            getattr(model_config, "hidden_size", None)
            or getattr(model_config, "d_model", None)
            or getattr(model_config, "hidden_dim", 0)  # MDLM uses hidden_dim
        )
        if self.hidden_dim == 0:
            raise ValueError("Could not determine diffusion-model hidden size from config")

    def _call_model(self, input_ids, attention_mask, timestep, output_attentions):
        import inspect

        sig = inspect.signature(self.model.forward)
        kwargs: dict[str, Any] = {}
        if "attention_mask" in sig.parameters:
            kwargs["attention_mask"] = attention_mask
        # pass diffusion timestep under common names if the signature accepts it.
        # Some checkpoints (MDLM) embed it per-example via t[:, None], so a 0-dim
        # scalar fails; broadcast to shape [batch].
        for key in ("timesteps", "t", "sigma", "timestep"):
            if key in sig.parameters:
                ts = timestep if torch.is_tensor(timestep) else torch.tensor(
                    float(timestep), device=input_ids.device
                )
                if ts.dim() == 0:
                    ts = ts.view(1).expand(input_ids.shape[0]).contiguous()
                kwargs[key] = ts
                break
        if "output_hidden_states" in sig.parameters:
            kwargs["output_hidden_states"] = True
        if "output_attentions" in sig.parameters:
            kwargs["output_attentions"] = output_attentions
        if "return_dict" in sig.parameters:
            kwargs["return_dict"] = True
        return self.model(input_ids=input_ids, **kwargs)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        timestep: torch.Tensor | int,
        output_attentions: bool = False,
    ) -> BackboneOutput:
        return self._forward_impl(input_ids, attention_mask, timestep, output_attentions)

    def forward_train(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        timestep: torch.Tensor | int,
        output_attentions: bool = False,
    ) -> BackboneOutput:
        """Same as ``forward`` but WITH autograd, for jointly training the denoiser.

        Rollouts always use the ``no_grad`` path; only the GRPO update recomputes the steps
        it needs through this one, so the graph stays as small as possible.
        """
        return self._forward_impl(input_ids, attention_mask, timestep, output_attentions)

    def _forward_impl(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        timestep: torch.Tensor | int,
        output_attentions: bool = False,
    ) -> BackboneOutput:
        autocast = bool(self.config.mixed_precision and input_ids.device.type == "cuda")
        with torch.autocast(device_type=input_ids.device.type, enabled=autocast):
            out = self._call_model(input_ids, attention_mask, timestep, output_attentions)
        logits = getattr(out, "logits", None)
        hidden_states = getattr(out, "hidden_states", None)
        if logits is None and isinstance(out, torch.Tensor):
            logits = out
        elif logits is None and isinstance(out, (tuple, list)) and len(out) > 0:
            # non-return_dict checkpoints: (logits[, all_hidden_states])
            logits = out[0]
            if len(out) > 1 and hidden_states is None:
                hidden_states = out[1]
        if logits is None:
            raise ValueError("MDLM model output has no logits; adapter needs adjustment for this checkpoint")
        hidden = None
        if hidden_states is not None:
            hidden = hidden_states[-1]
        else:
            hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            # fall back to logits as a stand-in feature space (still functional)
            hidden = logits
        # cast to float32 so the rest of the pipeline (built for fp32) matches even
        # when the model runs in bf16 (avoids index_put dtype mismatches).
        attn = getattr(out, "attentions", None) if output_attentions else None
        return {
            "hidden_states": hidden.float(),
            "logits": logits.float(),
            "attentions": tuple(a.float() for a in attn) if attn is not None else None,
        }

    def freeze(self) -> None:
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def apply_lora(self, lora_cfg: dict[str, Any]) -> list[torch.nn.Parameter]:
        """Attach LoRA adapters so the denoiser can be trained alongside the planner.

        The base weights stay frozen; only the adapters get gradients, which is what makes
        joint training fit on a single GPU (8B frozen + a few M trainable). Returns the
        trainable parameters so the caller can hand them to an optimizer.
        """
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError("Install `peft` to train the denoiser with LoRA") from exc

        targets = lora_cfg.get("target_modules")
        if not targets:
            targets = _default_lora_targets(self.model)
        peft_config = LoraConfig(
            r=int(lora_cfg.get("r", 16)),
            lora_alpha=int(lora_cfg.get("alpha", 32)),
            lora_dropout=float(lora_cfg.get("dropout", 0.0)),
            bias="none",
            target_modules=list(targets),
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.train()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError(f"LoRA matched no modules (targets={targets}); check target_modules")
        return trainable


def _default_lora_targets(model: torch.nn.Module) -> list[str]:
    """Leaf ``nn.Linear`` names that look like attention/MLP projections.

    Checkpoints differ (LLaDA uses ``q_proj``/``ff_proj``, MDLM uses ``Wqkv``), so match on
    the common name fragments instead of hard-coding one architecture.
    """
    wanted = ("q_proj", "k_proj", "v_proj", "o_proj", "qkv", "Wqkv", "out_proj",
              "ff_proj", "up_proj", "down_proj", "gate_proj", "ff_out")
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            leaf = name.split(".")[-1]
            if any(w.lower() in leaf.lower() for w in wanted):
                names.add(leaf)
    return sorted(names)


def build_mdlm_backbone(config: dict[str, Any]) -> MDLMBackbone:
    cfg = MDLMBackboneConfig(
        name=str(config.get("name", "kuleshov-group/mdlm-owt")),
        mixed_precision=bool(config.get("mixed_precision", False)),
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        mask_token=config.get("mask_token"),
        mask_token_id=config.get("mask_token_id"),
        tokenizer_name=config.get("tokenizer_name"),
        dtype=config.get("dtype"),
    )
    backbone = MDLMBackbone(cfg)
    backbone.freeze()
    return backbone
