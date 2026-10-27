"""WINO attention adapter and UNCODE ReMDM revision decoder."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

BASE = Path(__file__).resolve().parent


def definitions(path, names):
    tree = ast.parse(Path(path).read_text())
    nodes = [
        n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    assert {n.name for n in nodes} == set(names)
    scope = dict(torch=torch, F=F, np=np, nn=torch.nn)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


class ExplicitAttentionAdapter:
    """Read the existing frozen native weights with WINO's dense SDPA geometry.

    Physical canvas offsets and rotary position IDs are distinct. Repeated RoPE
    IDs are permitted; padding, unknown masks and missing positions are rejected.
    No fused-kernel or cache approximation of the verification mask is used.
    """

    def __init__(self, engine):
        if not engine._ready:
            raise ValueError("Load the complete checkpoint first")
        self.engine = engine
        self.device = engine.embedding.device
        self.dtype = engine.embedding.dtype
        self.calls = 0
        # WINO's released constructor warms RoPE on CUDA when available. Keep
        # that arithmetic here without changing the existing worker's buffers.
        c = engine.config
        inv = 1.0 / (
            c.rope_theta
            ** (
                torch.arange(0, c.head_dim, 2, device=self.device, dtype=torch.float32) / c.head_dim
            )
        )
        freq = torch.einsum(
            "i,j->ij",
            torch.arange(engine.sin.shape[1], device=self.device, dtype=torch.float32),
            inv,
        )
        angle = torch.cat((freq, freq), -1)[None, None]
        self.sin, self.cos = angle.sin(), angle.cos()

    @torch.inference_mode()
    def __call__(self, ids, attention_mask=None, position_ids=None):
        b, s = ids.shape
        assert b == 1 and ids.dtype == torch.long and ids.device == self.device
        if attention_mask is not None:
            assert attention_mask.dtype == torch.bool
            assert attention_mask.shape == (b, 1, s, s)
            assert attention_mask.any(-1).all(), "Every query must have at least one key"
        if position_ids is None:
            position_ids = torch.arange(s, device=ids.device)
        assert position_ids.shape == (s,) and position_ids.dtype == torch.long
        assert int(position_ids.min()) >= 0
        assert int(position_ids.max()) < self.sin.shape[-2]
        sin, cos = self.sin[:, :, position_ids], self.cos[:, :, position_ids]
        c = self.engine.config
        x = F.embedding(ids, self.engine.embedding)
        for block in self.engine.blocks:
            norm = block.attn_norm(x)
            widths = (c.d_model, block.kv_width, block.kv_width)
            # Separate projections preserve the official WINO forward's GEMMs.
            q, k, v = [F.linear(norm, w) for w in block.qkv.weight.split(widths)]
            q = q.view(b, s, c.n_heads, c.head_dim).transpose(1, 2)
            k = k.view(b, s, c.n_kv_heads, c.head_dim).transpose(1, 2)
            v = v.view(b, s, c.n_kv_heads, c.head_dim).transpose(1, 2)

            def rotate(t):
                y = t.float()
                first, second = y.chunk(2, -1)
                return (y * cos + torch.cat((-second, first), -1) * sin).to(t.dtype)

            q, k = rotate(q), rotate(k)
            repeat = c.n_heads // c.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            x = x + block.attn_out(out.transpose(1, 2).reshape(b, s, c.d_model))
            norm = block.ff_norm(x)
            gate, up = [F.linear(norm, w) for w in block.gate_up.weight.chunk(2)]
            x = x + block.ff_out(F.silu(gate) * up)
        self.calls += 1
        return SimpleNamespace(logits=self.engine.head(self.engine.norm(x)))


def wino(
    model,
    prompt,
    *,
    source,
    length=256,
    block=128,
    threshold=0.6,
    threshold_back=0.9,
    mask_id=126336,
):
    decoder = definitions(source, ["decoding_wino", "add_gumbel_noise"])["decoding_wino"]
    before = model.calls
    result, calls = decoder(
        model,
        prompt,
        gen_length=length,
        block_length=block,
        temperature=0.0,
        mask_id=mask_id,
        threshold=threshold,
        threshold_back=threshold_back,
    )
    assert model.calls - before == calls
    assert torch.equal(result[:, : prompt.shape[1]], prompt)
    assert not result[:, prompt.shape[1] :].eq(mask_id).any()
    return SimpleNamespace(sequences=result, steps_taken=torch.tensor([calls]))


@torch.inference_mode()
def remdm_uncode(
    model, prompt, *, length=256, initial_fraction=0.875, k=8, loops=16, mask_id=126336
):
    """Literal two-pass confidence-loop adaptation in UNCODE, without text logging.

    NFE = length/k + 2*loops. This is NOT length/k + loops. Use an explicit
    (k, loops) pair and actual calls rather than relabeling an over-budget run.
    """
    initial = length * initial_fraction
    assert initial == int(initial) and int(initial) % k == 0 and length % k == 0
    assert 0 < initial < length and k > 0 and loops >= 0
    x = torch.full((1, prompt.shape[1] + length), mask_id, device=prompt.device, dtype=torch.long)
    x[:, : prompt.shape[1]] = prompt
    prefix = torch.zeros_like(x, dtype=torch.bool)
    prefix[:, : prompt.shape[1]] = True
    calls = 0

    def probabilities():
        nonlocal calls
        logits = model(x).logits
        calls += 1
        return logits, logits.softmax(-1)

    def reveal(count):
        logits, p = probabilities()
        proposal = logits.argmax(-1)
        confidence = p.gather(-1, proposal.unsqueeze(-1)).squeeze(-1)
        indices = confidence.masked_fill(x.ne(mask_id) | prefix, -torch.inf).topk(count).indices
        x[0, indices] = proposal[0, indices]

    for _ in range(int(initial) // k):
        reveal(k)
    for _ in range(loops):
        _, p = probabilities()
        confidence = p.gather(-1, x.unsqueeze(-1)).squeeze(-1)
        visible = x.ne(mask_id) & ~prefix
        indices = confidence.masked_fill(~visible, torch.inf).topk(k, largest=False).indices
        x[0, indices] = mask_id
        reveal(k)
    while x[:, prompt.shape[1] :].eq(mask_id).any():
        reveal(min(k, int(x[:, prompt.shape[1] :].eq(mask_id).sum())))
    assert calls == length // k + 2 * loops
    assert torch.equal(x[:, : prompt.shape[1]], prompt)
    return SimpleNamespace(sequences=x, steps_taken=torch.tensor([calls]))


def remdm_probabilities(p, x, alpha_t, alpha_s, eta, mask_id, variant="rescale"):
    """Official ReMDM-cap/rescale posterior, with carry-over at visible tokens.

    Exposed for reference tests and future presets; not substituted for the
    confidence-loop LLaDA baseline without labeling that different sampler.
    """
    assert 0 <= alpha_t < alpha_s <= 1 and 0 <= eta <= 1
    sigma_max = min(1.0, (1 - alpha_s) / alpha_t) if alpha_t else 1.0
    sigma = min(eta, sigma_max) if variant == "cap" else eta * sigma_max
    assert variant in ("cap", "rescale")
    visible = x.ne(mask_id)
    p = p.clone()
    p[..., mask_id] = 0
    p /= p.sum(-1, keepdim=True)
    p[visible] = F.one_hot(x[visible], p.shape[-1]).to(p.dtype)
    reveal = (alpha_s - (1 - sigma) * alpha_t) / (1 - alpha_t)
    q_masked = p * reveal
    q_masked[..., mask_id] = 1 - reveal
    q_visible = p * (1 - sigma)
    q_visible[..., mask_id] = sigma
    return torch.where(visible.unsqueeze(-1), q_visible, q_masked)
