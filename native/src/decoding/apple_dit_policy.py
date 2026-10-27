"""Vendored Apple ``DiT-confidence`` unmasking policy (arXiv:2512.09106,
apple/ml-rl-dllm), ported so their trained planner can be evaluated head-to-head
against ours on the *same* harness, backbone, verifier and NFE definition.

Only the ``dit_confidence`` variant is vendored: a small RoPE-DiT transformer that maps
(mask, top-p confidences, diffusion timestep) -> per-position unmasking *logit*. Unlike the
``dit_hidden`` variant it needs no access to the 8B backbone internals, so it is fully
self-contained here. The forward is copied verbatim from ``common/models/policy.py`` /
``policy_layers.py`` so the trained ``model.safetensors`` loads with strict key matching.

At inference their decoder is monotone (no remasking): each step unmasks the positions whose
policy logit clears a stop threshold (deterministic ``bernoulli-argmax``: logit > stop_logit,
i.e. sigma(logit) > sigma(stop_logit)); the count is emergent, so it is an adaptive-cardinality
policy like Fast-dLLM but with a *learned* per-position score. We therefore expose it as a
``ZSource`` returning ``Z = sigma(policy_logit)`` and drive it with the same AdaptiveDecoder
threshold sweep used for ``fastdllm`` / ``learned_threshold``.

Copyright note: the model code below is derived from Apple Inc.'s ml-rl-dllm (see that repo's
LICENSE); vendored here only to load and evaluate their released-format checkpoints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from src.backbones.base import DiffusionBackbone
from src.decoding.confidence_decoder import DecodeResult, DecodeStepTrace
from src.decoding.schedules import make_blocks


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)


def sinusoidal_time_embedding(
    timesteps: torch.Tensor, dim: int, max_period: float = 10000, dtype=torch.bfloat16
) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(max_period, dtype=torch.float32, device=timesteps.device))
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps.unsqueeze(-1).float() * freqs.view(*([1] * timesteps.ndim), -1)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
    return embedding.to(dtype)


class RoPEDiTBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        batch_first: bool,
        activation: str | Callable,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.batch_first = batch_first
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)
        if isinstance(activation, str):
            assert hasattr(F, activation)
            self.activation_fn = getattr(F, activation)
        else:
            assert isinstance(activation, Callable)
            self.activation_fn = activation
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.ada_conditioning = nn.Linear(d_model, 6 * d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _get_rope_embed(self, seq_len: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        angles = positions.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        return torch.cos(angles), torch.sin(angles)

    def forward(self, src, cond, src_mask=None, src_key_padding_mask=None, is_causal=False):
        del src_key_padding_mask
        batch_size, seq_len = (src.shape[:2] if self.batch_first else src.shape[1::-1])
        ada_mod = self.ada_conditioning(cond)
        gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2 = ada_mod.chunk(6, dim=-1)

        residual = src
        src = self.norm1(src)
        src = src * (1 + gamma_1) + beta_1

        if self.batch_first:
            q = self.q_proj(src).view(batch_size, seq_len, self.nhead, self.head_dim)
            k = self.k_proj(src).view(batch_size, seq_len, self.nhead, self.head_dim)
            v = self.v_proj(src).view(batch_size, seq_len, self.nhead, self.head_dim)
        else:
            q = self.q_proj(src).view(seq_len, batch_size, self.nhead, self.head_dim)
            k = self.k_proj(src).view(seq_len, batch_size, self.nhead, self.head_dim)
            v = self.v_proj(src).view(seq_len, batch_size, self.nhead, self.head_dim)

        cos_emb, sin_emb = self._get_rope_embed(seq_len, src.device)
        if self.batch_first:
            cos_emb = cos_emb.unsqueeze(0).unsqueeze(2)
            sin_emb = sin_emb.unsqueeze(0).unsqueeze(2)
        else:
            cos_emb = cos_emb.unsqueeze(1).unsqueeze(2)
            sin_emb = sin_emb.unsqueeze(1).unsqueeze(2)
        q = apply_rotary_pos_emb(q, cos_emb, sin_emb)
        k = apply_rotary_pos_emb(k, cos_emb, sin_emb)

        if self.batch_first:
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        else:
            q, k, v = q.permute(1, 2, 0, 3), k.permute(1, 2, 0, 3), v.permute(1, 2, 0, 3)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=src_mask,
            dropout_p=self.dropout.p if self.training else 0.0, is_causal=is_causal,
        )
        if self.batch_first:
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        else:
            attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(seq_len, batch_size, self.d_model)

        attn_output = self.out_proj(attn_output)
        attn_output = attn_output * (1 + alpha_1)
        src = residual + self.dropout1(attn_output)

        residual = src
        src = self.norm2(src)
        src = src * (1 + gamma_2) + beta_2
        src = self.linear2(self.dropout(self.activation_fn(self.linear1(src))))
        src = src * (1 + alpha_2)
        src = residual + self.dropout2(src)
        return src


class DiTConfidencePolicy(nn.Module):
    """Maps (mask, top-p confidences, timestep) -> per-position unmasking logits."""

    def __init__(
        self,
        hidden_dim: int = 128,
        feedforward_dim: int = 512,
        num_heads: int = 1,
        dropout: float = 0.0,
        time_embed_dim: int = 128,
        confidences_top_p: int = 1,
        smart_init: float | None = None,
        num_blocks: int = 1,
        time_period: float = 1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.time_period = time_period
        self.num_blocks = num_blocks
        self.confidences_top_p = confidences_top_p
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.confidence_proj = nn.Linear(confidences_top_p, hidden_dim)
        self.mask_embedding = nn.Embedding(2, hidden_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                RoPEDiTBlock(
                    d_model=hidden_dim, nhead=num_heads, dim_feedforward=feedforward_dim,
                    dropout=dropout, activation="gelu", batch_first=True,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 1)
        if smart_init is not None:
            with torch.no_grad():
                for block in self.transformer_blocks:
                    block.ada_conditioning.weight.data.zero_()
                    block.ada_conditioning.bias.data.zero_()
                self.output_proj.bias.data.fill_(smart_init)

    def forward(self, m: torch.Tensor, c: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """m: (*B,L) 1=masked; c: (*B,L,top_p) confidences in [0,1]; timestep: (*B,1) in [0,1]."""
        *B, L, _ = c.shape
        cond = self.mask_embedding(m.int())
        time_embed = sinusoidal_time_embedding(timestep, self.time_embed_dim, max_period=self.time_period)
        time_embed = time_embed.to(cond.dtype).expand((*([-1] * len(B)), L, -1))
        cond = cond + self.time_mlp(time_embed)
        x = self.confidence_proj(c)
        original_shape = x.shape
        x_flat = x.view(-1, L, self.hidden_dim)
        cond_flat = cond.view(-1, L, self.hidden_dim)
        for block in self.transformer_blocks:
            x_flat = block(x_flat, cond_flat)
        x = self.final_norm(x_flat.view(original_shape))
        return self.output_proj(x).squeeze(-1)


# Arch of the checkpoints we train (configs/experiment_configs/llada_8b_instruct_dit_confidence_*).
DEFAULT_DIT_CONFIG: dict[str, object] = {
    "hidden_dim": 128,
    "feedforward_dim": 512,
    "num_heads": 2,
    "dropout": 0.0,
    "time_embed_dim": 128,
    "confidences_top_p": 1,
    "smart_init": -2.0,
    "num_blocks": 1,
    "time_period": 1,
}


def load_apple_dit_policy(
    checkpoint_dir: str, arch: dict[str, object] | None = None, device=None
) -> DiTConfidencePolicy:
    """Load a trained Apple ``dit_confidence`` policy from a HF-format checkpoint directory.

    The saved ``model.safetensors`` is a ``PolicyHFWrapper`` whose weights are prefixed
    ``base_policy.``; we strip that and load the bare ``DiTConfidencePolicy`` (strict).
    """
    from safetensors.torch import load_file

    cfg = dict(DEFAULT_DIT_CONFIG)
    if arch:
        cfg.update(arch)
    # allow an optional arch.json dropped next to the checkpoint to override defaults
    arch_path = Path(checkpoint_dir) / "dit_arch.json"
    if arch_path.exists():
        cfg.update(json.loads(arch_path.read_text()))
    policy = DiTConfidencePolicy(**cfg)  # type: ignore[arg-type]
    state = load_file(str(Path(checkpoint_dir) / "model.safetensors"))
    stripped = {k[len("base_policy."):]: v for k, v in state.items() if k.startswith("base_policy.")}
    missing, unexpected = policy.load_state_dict(stripped, strict=False)
    unexpected = [u for u in unexpected if not u.endswith("inv_freq")]  # buffer, recomputed
    if unexpected:
        raise ValueError(f"unexpected keys loading Apple DiT policy: {unexpected[:5]}")
    if missing:
        raise ValueError(f"missing keys loading Apple DiT policy: {missing[:5]}")
    policy.eval()
    return policy.to(device) if device is not None else policy


@dataclass
class AppleDiTConfig:
    # How policy logits become unmask decisions:
    #   "bernoulli"        -- NATIVE (Apple eval default): b ~ Bernoulli(sigmoid(logit)) per masked
    #                          position; a step that samples nothing burns a forward (as native).
    #   "bernoulli-argmax" -- native variant: same, but force the top-logit position if none drawn.
    #   "threshold"        -- deterministic logit > stop_logit. NOT Apple's inference rule; kept
    #                          only as a labelled variant (it changes trajectories and NFE).
    sampling_mode: str = "bernoulli"
    stop_logit: float = 0.0        # threshold mode only
    temperature_policy: float = 1.0  # native knob: logits / T before sampling
    seed: int | None = None        # RNG seed for the stochastic modes (reproducible rollouts)
    block_length: int = 0          # 0 -> full diffusion (single block == gen region)
    confidences_top_p: int = 1     # top-p confidences fed to the policy (checkpoints use 1)
    full_context: bool = True      # policy sees the whole gen region each step (checkpoints use true)
    max_steps: int | None = None   # safety cap on forwards per block; defaults to block size


class AppleDiTDecoder:
    """Port of Apple's ``generate_unified(remasking='policy')`` so their trained
    ``dit_confidence`` planner runs on OUR backbone / env / verifier / NFE definition.

    Each forward the policy scores every gen position from (mask, top-p confidence, timestep)
    and, in the NATIVE ``bernoulli`` mode, each still-masked position in the active block is
    unmasked with probability ``sigmoid(logit / T)`` (``bernoulli_sample`` in
    ``common/generation/sampling.py``); the token written is the backbone argmax. Monotone -- no
    remasking. NFE is emergent (a step that draws nothing burns a forward, as native), so a
    trained policy has ONE native operating point per seed -- not a swept curve. The
    deterministic ``threshold`` mode (logit > stop_logit, swept) is a *modified* decoder kept only
    as a labelled variant. With ``full_context`` the policy attends over the whole gen region but
    only commits inside the active block; the per-block forward cap force-completes the block.
    """

    def __init__(self, backbone: DiffusionBackbone, policy: DiTConfidencePolicy, config: AppleDiTConfig) -> None:
        self.backbone = backbone
        self.policy = policy
        self.config = config

    def _editable_per_row(self, input_ids: torch.Tensor) -> list[torch.Tensor]:
        mid = self.backbone.mask_token_id
        return [torch.nonzero(row.eq(mid), as_tuple=False).flatten() for row in input_ids]

    @torch.no_grad()
    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        timestep: int = 0,
        editable_positions: list[torch.Tensor] | None = None,
    ) -> DecodeResult:
        del timestep
        cfg = self.config
        if cfg.sampling_mode not in ("bernoulli", "bernoulli-argmax", "threshold"):
            raise ValueError(f"unknown sampling_mode {cfg.sampling_mode!r}")
        current = input_ids.clone()
        mask_id = self.backbone.mask_token_id
        p_dtype = next(self.policy.parameters()).dtype
        gen = None
        if cfg.sampling_mode != "threshold" and cfg.seed is not None:
            gen = torch.Generator(device=current.device).manual_seed(int(cfg.seed))
        editable = editable_positions if editable_positions is not None else self._editable_per_row(input_ids)
        blocks_per_row = [make_blocks(int(ed.numel()), cfg.block_length) for ed in editable]
        max_blocks = max((len(b) for b in blocks_per_row), default=0)
        forwards = 0
        traces: list[DecodeStepTrace] = []
        # per-row diffusion "steps_taken" -> timestep = steps_taken / L (L = gen-region length)
        steps_taken = [0 for _ in editable]

        for b in range(max_blocks):
            block_pos = {
                r: editable[r][blocks_per_row[r][b][0]:blocks_per_row[r][b][1]]
                for r in range(len(editable)) if b < len(blocks_per_row[r])
            }
            cap = cfg.max_steps or max((int(p.numel()) for p in block_pos.values()), default=0)
            step = 0
            while step < cap:
                active = {r: p for r, p in block_pos.items() if current[r, p].eq(mask_id).any()}
                if not active:
                    break
                out = self.backbone.forward(current, attention_mask, timestep=0)
                logits = out["logits"]
                forwards += 1
                last = step == cap - 1
                for row, blk in active.items():
                    ed = editable[row]
                    scope = ed if cfg.full_context else blk  # positions the policy scores
                    L = int(ed.numel())
                    probs = torch.softmax(logits[row, scope].float(), dim=-1)
                    c = probs.topk(cfg.confidences_top_p, dim=-1).values  # [scope, top_p]
                    m = current[row, scope].eq(mask_id)  # [scope] 1=masked
                    tstep = torch.tensor([[steps_taken[row] / max(L, 1)]], device=current.device)
                    plogits = self.policy(
                        m.to(p_dtype).unsqueeze(0), c.to(p_dtype).unsqueeze(0), tstep.to(p_dtype)
                    ).squeeze(0)  # [scope]
                    # restrict commit decisions to still-masked positions in the ACTIVE block
                    in_block = torch.zeros(scope.numel(), dtype=torch.bool, device=current.device)
                    block_local = torch.searchsorted(scope, blk) if cfg.full_context else torch.arange(blk.numel(), device=current.device)
                    in_block[block_local] = True
                    cand = in_block & current[row, scope].eq(mask_id)
                    if not bool(cand.any()):
                        continue
                    if cfg.temperature_policy != 1.0:
                        plogits = plogits / float(cfg.temperature_policy)
                    if cfg.sampling_mode == "threshold":
                        keep = cand & (plogits > cfg.stop_logit)
                        force_if_empty = True  # deterministic variant always commits >= 1
                    else:
                        # native: b ~ Bernoulli(sigmoid(logit)) over the available positions
                        probs = torch.sigmoid(plogits.float()) * cand.float()
                        draw = torch.rand(probs.shape, generator=gen, device=probs.device) < probs
                        keep = cand & draw
                        force_if_empty = cfg.sampling_mode == "bernoulli-argmax"
                    if last:
                        keep = cand  # per-block forward cap: force-complete (never leak masks)
                    elif not bool(keep.any()) and force_if_empty:
                        gated = torch.where(cand, plogits, plogits.new_full((), -1e30))
                        keep = torch.zeros_like(cand)
                        keep[int(gated.argmax())] = True
                    steps_taken[row] += 1
                    if not bool(keep.any()):
                        continue  # native "bernoulli": nothing drawn -> burn this forward
                    sel = scope[torch.nonzero(keep, as_tuple=False).flatten()]
                    current[row, sel] = logits[row, sel].argmax(dim=-1)
                remaining = int(current.eq(mask_id).sum().item())
                traces.append(DecodeStepTrace(step=forwards - 1, masked_remaining=remaining,
                                              selected_positions=[], mean_confidence=float(cfg.stop_logit)))
                step += 1
        return DecodeResult(
            current, forwards, traces,
            {"remask_count": 0, "sampling_mode": cfg.sampling_mode, "stop_logit": float(cfg.stop_logit),
             "temperature_policy": float(cfg.temperature_policy), "seed": cfg.seed},
        )
