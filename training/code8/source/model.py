# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache-free TP1 LLaDA backbone for an external diffusion-policy worker.

This is a native weight port, not a HuggingFace model wrapper. Every call
recomputes prompt AND answer representations. Only answer rows reach the head.
The CUDA backend uses vLLM's noncausal FlashAttention-2 kernel, with independent
sequence boundaries. No serving scheduler, KV cache, or token sampler is used.

The small torch backend exists for inexpensive mapping/semantics tests. Fused
QKV and gate/up GEMMs and FlashAttention can change BF16 rounding; trajectory
parity and held-out quality are required before replacing a research backend.
"""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


@dataclass(frozen=True)
class LLaDAConfig:
    d_model: int
    n_heads: int
    n_kv_heads: int
    n_layers: int
    mlp_hidden_size: int
    embedding_size: int
    max_sequence_length: int
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-5

    def __post_init__(self):
        dimensions = (
            self.d_model,
            self.n_heads,
            self.n_kv_heads,
            self.n_layers,
            self.mlp_hidden_size,
            self.embedding_size,
            self.max_sequence_length,
        )
        if any(type(x) is not int or x <= 0 for x in dimensions):
            raise ValueError("LLaDA dimensions must be positive integers")
        if self.d_model % self.n_heads or self.n_heads % self.n_kv_heads:
            raise ValueError("Invalid attention head dimensions")
        if self.head_dim % 2 or self.rope_theta <= 0 or self.rms_norm_eps <= 0:
            raise ValueError("Invalid RoPE / normalization configuration")

    @property
    def head_dim(self):
        return self.d_model // self.n_heads

    @classmethod
    def from_hf_dict(cls, values: Mapping):
        # Fail closed on architectural differences instead of silently treating
        # another LLaDA variant as the pinned 8B-Instruct architecture.
        required = {
            "model_type": "llada",
            "block_type": "llama",
            "activation_type": "silu",
            "layer_norm_type": "rms",
            "layer_norm_with_affine": True,
            "bias_for_layer_norm": False,
            "include_bias": False,
            "include_qkv_bias": False,
            "attention_layer_norm": False,
            "input_emb_norm": False,
            "weight_tying": False,
            "scale_logits": False,
            "alibi": False,
            "rope": True,
            "rope_full_precision": True,
            "block_group_size": 1,
        }
        for key, expected in required.items():
            if key not in values or values[key] != expected:
                raise ValueError(f"Unsupported {key}={values.get(key)!r}")
        return cls(**{key: values[key] for key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class CanvasLayout:
    """Static attention/answer metadata, prepared once outside the NFE loop.

    Input mask: [B,S], true for real prompt/answer tokens, false for padding.
    Answer positions: [B,A], absolute padded-canvas indices (not RoPE resets).
    All GPU-to-CPU validation/synchronization happens here, never per layer.
    """

    attention_mask: torch.Tensor
    answer_positions: torch.Tensor
    kv_indices: torch.Tensor
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    max_k: int
    all_valid: bool

    @classmethod
    def build(cls, attention_mask: torch.Tensor, answer_positions: torch.Tensor):
        if attention_mask.ndim != 2 or answer_positions.ndim != 2:
            raise ValueError("Mask and answer positions must be rank two")
        b, s = attention_mask.shape
        if not b or not s or answer_positions.shape[0] != b:
            raise ValueError("Invalid canvas batch shape")
        if answer_positions.shape[1] == 0 or answer_positions.dtype != torch.long:
            raise ValueError("Answer positions must be nonempty int64 indices")
        if attention_mask.device != answer_positions.device:
            raise ValueError("Mask and positions must share a device")
        if not ((attention_mask == 0) | (attention_mask == 1)).all().item():
            raise ValueError("Attention mask must be binary")
        mask = attention_mask.bool().clone()
        positions = answer_positions.clone()
        if (positions < 0).any().item() or (positions >= s).any().item():
            raise ValueError("Answer position outside canvas")
        if not mask.gather(1, positions).all().item():
            raise ValueError("Answer positions may not point into padding")
        ordered = positions.sort(dim=1).values
        if (ordered[:, 1:] == ordered[:, :-1]).any().item():
            raise ValueError("Duplicate answer positions")
        lengths = mask.sum(1, dtype=torch.int32)
        if (lengths == 0).any().item():
            raise ValueError("Empty attention sequence")
        cu_k = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
        cu_q = torch.arange(b + 1, device=mask.device, dtype=torch.int32) * s
        indices = mask.flatten().nonzero().flatten()
        return cls(
            mask,
            positions,
            indices,
            cu_q,
            cu_k,
            int(lengths.max().item()),
            bool(mask.all().item()),
        )


class _Linear(nn.Module):
    def __init__(self, inputs, outputs, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(outputs, inputs, dtype=dtype))

    def forward(self, x):
        return F.linear(x, self.weight)


class _Norm(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config.d_model, dtype=dtype))
        self.eps = config.rms_norm_eps

    def forward(self, x):
        # The reference rounds normalized x to BF16 BEFORE multiplying weight.
        # Do not substitute a fused RMSNorm with a different rounding boundary.
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return value.to(x.dtype) * self.weight


class _Block(nn.Module):
    def __init__(self, config, dtype, fuse_projections):
        super().__init__()
        self.config = config
        self.fuse_projections = fuse_projections
        self.kv_width = config.n_kv_heads * config.head_dim
        self.qkv = _Linear(config.d_model, config.d_model + 2 * self.kv_width, dtype)
        self.attn_out = _Linear(config.d_model, config.d_model, dtype)
        self.gate_up = _Linear(config.d_model, 2 * config.mlp_hidden_size, dtype)
        self.ff_out = _Linear(config.mlp_hidden_size, config.d_model, dtype)
        self.attn_norm = _Norm(config, dtype)
        self.ff_norm = _Norm(config, dtype)

    def forward(self, x, layout, sin, cos, flash):
        c = self.config
        b, s, _ = x.shape
        normalized = self.attn_norm(x)
        widths = (c.d_model, self.kv_width, self.kv_width)
        if self.fuse_projections:
            q, k, v = self.qkv(normalized).split(widths, dim=-1)
        else:
            q, k, v = (
                F.linear(normalized, weight)
                for weight in self.qkv.weight.split(widths, dim=0)
            )
        q = q.view(b, s, c.n_heads, c.head_dim)
        k = k.view(b, s, c.n_kv_heads, c.head_dim)
        v = v.view(b, s, c.n_kv_heads, c.head_dim)

        def rotate(t):
            value = t.float()
            first, second = value.chunk(2, dim=-1)
            return (value * cos + torch.cat((-second, first), -1) * sin).to(t.dtype)

        q, k = rotate(q), rotate(k)
        if flash is None:
            repeat = c.n_heads // c.n_kv_heads
            k = k.repeat_interleave(repeat, dim=2)
            v = v.repeat_interleave(repeat, dim=2)
            bias = None
            if not layout.all_valid:
                bias = torch.zeros_like(layout.attention_mask, dtype=x.dtype)
                bias.masked_fill_(~layout.attention_mask, torch.finfo(x.dtype).min)
                # Match the original dense additive-bias layout in this control.
                bias = bias[:, None, None, :].expand(b, 1, s, s).contiguous()
            # The pinned HF constructor globally disables EFFICIENT_ATTENTION.
            # Express that locally: importing HF must not change this control.
            with sdpa_kernel(
                [
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.MATH,
                    SDPBackend.CUDNN_ATTENTION,
                ]
            ):
                attended = F.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    attn_mask=bias,
                    dropout_p=0.0,
                    is_causal=False,
                ).transpose(1, 2)
        else:
            # Keep all query rows/GEMM shapes, remove only padding KEYS. Each
            # request has separate Q/K cumulative lengths: no cross-row leakage.
            keys = k.reshape(-1, c.n_kv_heads, c.head_dim)
            values = v.reshape_as(keys)
            if not layout.all_valid:
                keys = keys.index_select(0, layout.kv_indices)
                values = values.index_select(0, layout.kv_indices)
            attended = flash(
                q.reshape(-1, c.n_heads, c.head_dim),
                keys,
                values,
                cu_seqlens_q=layout.cu_q,
                cu_seqlens_k=layout.cu_k,
                max_seqlen_q=s,
                max_seqlen_k=layout.max_k,
                dropout_p=0.0,
                softmax_scale=c.head_dim**-0.5,
                causal=False,
                fa_version=2,
            )
        x = x + self.attn_out(attended.reshape(b, s, c.d_model))
        normalized = self.ff_norm(x)
        if self.fuse_projections:
            gate, up = self.gate_up(normalized).chunk(2, dim=-1)
        else:
            gate, up = (
                F.linear(normalized, weight)
                for weight in self.gate_up.weight.chunk(2, dim=0)
            )
        # Preserve the reference's separate SiLU BF16 rounding before multiply.
        return x + self.ff_out(F.silu(gate) * up)


class LLaDAOfflineModel(nn.Module):
    """Frozen TP1 backbone returning (answer logits, final-normalized hidden).

    `backend="vllm"` requires CUDA and the fork's compiled FA2 extension.
    `backend="torch"` is a standalone CPU/GPU correctness control.
    The learned policy, RNG, remasking, rewards, and replay remain external.
    """

    def __init__(
        self,
        config: LLaDAConfig,
        *,
        dtype=torch.bfloat16,
        backend: Literal["torch", "vllm"] = "vllm",
        fuse_projections: bool = True,
    ):
        super().__init__()
        if backend not in ("torch", "vllm"):
            raise ValueError(f"Unknown backend: {backend}")
        self.config = config
        self.backend = backend
        self._ready = False
        self.flash = None
        if backend == "vllm":
            from vllm.vllm_flash_attn import flash_attn_varlen_func

            self.flash = flash_attn_varlen_func
        self.embedding = nn.Parameter(
            torch.empty(config.embedding_size, config.d_model, dtype=dtype)
        )
        self.blocks = nn.ModuleList(
            [_Block(config, dtype, fuse_projections) for _ in range(config.n_layers)]
        )
        self.norm = _Norm(config, dtype)
        self.head = _Linear(config.d_model, config.embedding_size, dtype)
        # The pinned HF constructor warms RoPE on CPU before moving the model.
        # CPU vs CUDA sin/cos rounding can compound through 32 BF16 layers.
        # Preserve that origin, not merely the FP32 dtype and mathematical formula.
        inv = 1.0 / config.rope_theta ** (
            torch.arange(0, config.head_dim, 2, dtype=torch.float32, device="cpu")
            / config.head_dim
        )
        freq = torch.outer(
            torch.arange(config.max_sequence_length, dtype=torch.float32, device="cpu"),
            inv,
        )
        angle = torch.cat((freq, freq), dim=-1)[None, :, None, :]
        self.register_buffer(
            "sin", angle.sin().to(self.embedding.device), persistent=False
        )
        self.register_buffer(
            "cos", angle.cos().to(self.embedding.device), persistent=False
        )
        self.requires_grad_(False)
        self.eval()

    def weight_mapping(self):
        """Map every reference tensor to a whole weight or a fused row slice."""
        prefix = "model.transformer."
        mapping = {
            prefix + "wte.weight": self.embedding,
            prefix + "ln_f.weight": self.norm.weight,
            prefix + "ff_out.weight": self.head.weight,
        }
        d, m = self.config.d_model, self.config.mlp_hidden_size
        for i, block in enumerate(self.blocks):
            base = prefix + f"blocks.{i}."
            kv = block.kv_width
            mapping.update(
                {
                    base + "q_proj.weight": block.qkv.weight[:d],
                    base + "k_proj.weight": block.qkv.weight[d : d + kv],
                    base + "v_proj.weight": block.qkv.weight[d + kv :],
                    base + "ff_proj.weight": block.gate_up.weight[:m],
                    base + "up_proj.weight": block.gate_up.weight[m:],
                    base + "attn_out.weight": block.attn_out.weight,
                    base + "ff_out.weight": block.ff_out.weight,
                    base + "attn_norm.weight": block.attn_norm.weight,
                    base + "ff_norm.weight": block.ff_norm.weight,
                }
            )
        return mapping

    @torch.no_grad()
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Strict streaming loader; incomplete/duplicate/unknown loads fail closed."""
        self._ready = False
        mapping, seen = self.weight_mapping(), set()
        for name, value in weights:
            if name not in mapping or name in seen:
                raise ValueError(f"Unexpected or duplicate checkpoint tensor: {name}")
            target = mapping[name]
            if target.shape != value.shape or not value.is_floating_point():
                raise ValueError(f"Invalid shape/dtype for {name}: {value.shape}")
            target.copy_(value)
            seen.add(name)
        if missing := mapping.keys() - seen:
            raise ValueError(f"Missing checkpoint tensors: {sorted(missing)}")
        self._ready = True
        return seen

    @classmethod
    def from_local_checkpoint(cls, directory, *, device="cuda", **kwargs):
        """Load local safetensors only; never execute checkpoint-provided Python."""
        from safetensors import safe_open

        root = Path(directory).resolve()
        config = LLaDAConfig.from_hf_dict(
            json.loads((root / "config.json").read_text())
        )
        index_path = root / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())["weight_map"]
            filenames = sorted(set(index.values()))
        else:
            filenames = ["model.safetensors"]
        paths = [(root / name).resolve() for name in filenames]
        if any(path.parent != root or not path.is_file() for path in paths):
            raise ValueError("Invalid or missing local checkpoint shard")

        def weights():
            for path in paths:
                with safe_open(path, framework="pt", device="cpu") as shard:
                    for name in shard.keys():  # noqa: SIM118 (safe_open is not iterable)
                        yield name, shard.get_tensor(name)

        with torch.device(device):
            model = cls(config, **kwargs)
        model.load_weights(weights())
        return model

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, layout: CanvasLayout):
        if not self._ready:
            raise RuntimeError("A complete validated checkpoint must be loaded first")
        if input_ids.shape != layout.attention_mask.shape:
            raise ValueError("Input shape does not match static canvas layout")
        if input_ids.dtype != torch.long or input_ids.device != self.embedding.device:
            raise ValueError("Input IDs must be int64 on the model device")
        if layout.attention_mask.device != input_ids.device:
            raise ValueError("Layout must be on the model device")
        s = input_ids.shape[1]
        if s > self.config.max_sequence_length:
            raise ValueError("Canvas exceeds maximum sequence length")
        if self.backend == "vllm" and input_ids.device.type != "cuda":
            raise ValueError("The vLLM backend requires CUDA")
        x = F.embedding(input_ids, self.embedding)
        for block in self.blocks:
            x = block(x, layout, self.sin[:, :s], self.cos[:, :s], self.flash)
        x = self.norm(x)
        hidden = x.gather(
            1, layout.answer_positions[..., None].expand(-1, -1, x.shape[-1])
        )
        return self.head(hidden), hidden

    def prepare(self, input_ids, attention_mask, answer_positions):
        """Bind static metadata to the existing answer-backbone worker contract.

        The returned callable accepts (ids, mask, positions); ids may change
        every step. Mask/position tensors must remain the original, unmodified
        objects for this collection. Re-prepare when the request batch changes.
        No policy, sampling, remasking, or replay logic is replaced here.
        """
        layout = CanvasLayout.build(attention_mask, answer_positions)
        if input_ids.shape != attention_mask.shape:
            raise ValueError("Input shape does not match mask")

        def answer_forward(ids, mask, positions):
            if mask is not attention_mask or positions is not answer_positions:
                raise ValueError("Request metadata changed; prepare a new canvas")
            return self(ids, layout)

        return answer_forward
