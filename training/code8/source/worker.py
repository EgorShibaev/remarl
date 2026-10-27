"""Experimental synchronous, cache-free rollout worker for the small hidden MLP.

No optimizer or production collector is changed. Callers supply a frozen policy,
equal-length canvases, exact answer indices and explicit CPU-generated uniforms.
The leading batch dimension may contain trajectories from different questions.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from src.decoding.schedules import topk_schedule


class Timings:
    """CUDA events avoid per-section synchronization; resolve only after a rollout."""

    def __init__(self, enabled=False):
        self.enabled = enabled and torch.cuda.is_available()
        self.events = []

    @contextmanager
    def section(self, name):
        if not self.enabled:
            yield
            return
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        self.events.append((name, start, end))

    def resolve(self):
        if not self.enabled:
            return {}
        torch.cuda.synchronize()
        result = {}
        for name, start, end in self.events:
            result[name] = result.get(name, 0.0) + start.elapsed_time(end) / 1000
        return result


def ordered_logprob(scores, order, temperature=1.0):
    """Batched ordered PL probability, retaining the reference tail ordering."""
    batch, length = scores.shape
    count = order.shape[1]
    if not count:
        return scores.sum(-1) * 0
    chosen = torch.zeros_like(scores, dtype=torch.bool).scatter(1, order, True)
    indices = torch.arange(length, device=scores.device).expand(batch, -1)
    rest = (indices + chosen * length).argsort(-1)[:, : length - count]
    permutation = torch.cat((order, rest), -1)
    values = (scores / max(temperature, 1e-6)).gather(1, permutation)
    denom = torch.logcumsumexp(values.flip(-1), dim=-1).flip(-1)
    return (values[:, :count] - denom[:, :count]).sum(-1)


def sample_order(scores, uniforms, count, temperature=1.0):
    noise = -torch.log(-torch.log(uniforms.clamp(1e-10, 1.0)))
    return torch.argsort(scores / max(temperature, 1e-6) + noise, descending=True)[:, :count]


def features_from_answer(logits, hidden, current, mask_id):
    """Full-vocabulary FP32 normalization; exact original three raw inputs."""
    probabilities = logits.float().softmax(-1)
    masked = current.eq(mask_id)
    maximum = probabilities.max(-1).values
    keep = probabilities.gather(-1, current.clamp(0, logits.shape[-1] - 1)[..., None])[..., 0]
    scores = torch.where(masked, maximum, keep)
    positions = torch.arange(current.shape[-1], device=current.device, dtype=scores.dtype)
    basic = torch.stack((scores, masked.to(scores.dtype), positions.expand_as(scores)), -1)
    features = torch.cat((basic, hidden.float()), -1) if hidden is not None else basic
    return features, scores, masked


def transition(current, predicted, order, mask_id):
    keep = torch.zeros_like(current, dtype=torch.bool).scatter(1, order, True)
    masked = current.eq(mask_id)
    revised = ~keep & ~masked
    following = torch.where(keep, torch.where(masked, predicted, current), mask_id)
    return following, revised


class AnswerBackbone:
    """Same HF model: omit intermediate hidden collection and prompt LM-head rows.

    An explicit context installs/removes one hook on this isolated model instance.
    The head receives only post-final-normalization answer hidden states. The
    transformer still recomputes every position at every denoising step.
    """

    def __init__(self, backbone):
        self.backbone = backbone
        self.core = backbone.model.model
        if self.core.config.weight_tying or self.core.config.scale_logits:
            raise ValueError("Initial prototype requires the pinned untied, unscaled LLaDA head")
        self.hidden = None
        self.positions = None

    def __enter__(self):
        def select_answer(module, args):
            x = args[0]
            self.hidden = x.gather(1, self.positions[..., None].expand(-1, -1, x.shape[-1]))
            return (self.hidden,)

        self.select_answer = select_answer
        self.hook = self.core.transformer.ff_out.register_forward_pre_hook(select_answer)
        return self

    def __exit__(self, *unused):
        self.hook.remove()
        self.hidden = None

    @contextmanager
    def unhooked(self):
        """Allow an independent reference forward during parity diagnostics."""
        self.hook.remove()
        try:
            yield
        finally:
            self.hook = self.core.transformer.ff_out.register_forward_pre_hook(self.select_answer)

    def __call__(self, ids, mask, positions):
        self.positions = positions
        out = self.core(
            input_ids=ids, attention_mask=mask, use_cache=False, output_hidden_states=False
        )
        return out.logits.float(), self.hidden


def initial_representatives(ids, attention, positions):
    """Merge only exactly identical initial inputs, including padding and spans."""
    keys = torch.cat((ids, attention, positions), dim=1).cpu().tolist()
    unique, representatives, inverse = {}, [], []
    for row, key in enumerate(keys):
        key = tuple(key)
        if key not in unique:
            unique[key] = len(representatives)
            representatives.append(row)
        inverse.append(unique[key])
    return (
        torch.tensor(representatives, dtype=torch.long, device=ids.device),
        torch.tensor(inverse, dtype=torch.long, device=ids.device),
    )


@dataclass
class CapturedForward:
    """Own EVERY external tensor whose address the captured graph reads.

    CUDA graphs retain addresses, not Python references to their input tensors.
    In particular the attention bias must outlive all graph replays, even though
    it is immutable and therefore never copied in __call__.
    """

    graph: object
    ids: torch.Tensor
    positions: torch.Tensor
    bias: torch.Tensor | None
    logits: torch.Tensor
    hidden: torch.Tensor

    def __call__(self, ids, mask, positions):
        # The owner checks mask identity once in prepare(), outside the loop.
        if ids.shape != self.ids.shape or positions.shape != self.positions.shape:
            raise ValueError("Graph input shape changed")
        self.ids.copy_(ids)
        self.positions.copy_(positions)
        self.graph.replay()
        return self.logits, self.hidden


class GraphedAnswerBackbone(AnswerBackbone):
    """Bounded fixed-shape CUDA graphs of the unchanged, frozen LLaDA forward.

    Prepare outside the denoising loop. The pinned model's Python mask branch
    synchronizes with the host, so capture the *actual* bias delivered to its
    first transformer block in an eager pass and supply that same bias directly.
    Do not infer or change its attention semantics. Each graph has independent
    buffers; outputs are borrowed and valid until that graph's next invocation.
    No activations or K/V are cached between denoising steps.
    """

    def __init__(self, backbone, max_graphs=4):
        super().__init__(backbone)
        self.max_graphs = max_graphs
        self.graphs = {}
        self.preparations = []
        self.sequence_length = None

    @torch.no_grad()
    def prepare(self, ids, mask, positions):
        import time

        if not ids.is_cuda or self.core.training:
            raise ValueError("CUDA and an eval-mode backbone are required")
        if self.core.config.block_group_size != 1:
            raise ValueError("Only the pinned ungrouped LLaDA blocks are supported")
        if self.sequence_length is not None and self.sequence_length != ids.shape[1]:
            raise ValueError(
                "Use a separate graph worker for each sequence length (rotary cache lifetime)"
            )
        self.sequence_length = ids.shape[1]
        # A different padding pattern must never accidentally reuse stale bias.
        key = (tuple(ids.shape), tuple(positions.shape), tuple(mask.cpu().flatten().tolist()))
        if key in self.graphs:
            return self.graphs[key]
        if len(self.graphs) >= self.max_graphs:
            raise MemoryError("CUDA graph cache limit reached; use a new bounded worker")
        started = time.perf_counter()
        static_ids, static_positions = ids.clone(), positions.clone()
        captured_bias = []

        def capture_bias(module, args, kwargs):
            bias = kwargs.get("attention_bias")
            captured_bias.append(None if bias is None else bias.detach().clone())

        hook = self.core.transformer.blocks[0].register_forward_pre_hook(
            capture_bias, with_kwargs=True
        )
        try:
            # This also populates the model's fixed-shape rotary buffers.
            super().__call__(ids, mask, positions)
        finally:
            hook.remove()
        if len(captured_bias) != 1:
            raise RuntimeError("Could not unambiguously capture the reference attention bias")
        bias = captured_bias[0]

        def forward():
            self.positions = static_positions
            out = self.core(
                input_ids=static_ids,
                attention_mask=None,
                attention_bias=bias,
                use_cache=False,
                output_hidden_states=False,
            )
            return out.logits.float(), self.hidden

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            logits, hidden = forward()
        torch.cuda.synchronize()

        replay = CapturedForward(graph, static_ids, static_positions, bias, logits, hidden)
        self.graphs[key] = replay
        self.preparations.append(
            {
                "batch": len(ids),
                "seconds": time.perf_counter() - started,
                "bias_shape": None if bias is None else list(bias.shape),
            }
        )
        return replay

    def __exit__(self, *unused):
        torch.cuda.synchronize()
        self.graphs.clear()
        super().__exit__(*unused)


@dataclass
class BatchResult:
    final_ids: torch.Tensor
    revisions: torch.Tensor
    repeat_revisions: torch.Tensor
    nfe: int
    features: torch.Tensor | None
    scores: torch.Tensor | None
    orders: list
    old_logp: torch.Tensor | None
    counts: list
    masks: torch.Tensor | None
    sections: dict


@torch.no_grad()
def collect(
    backbone,
    selector,
    ids,
    attention,
    positions,
    uniforms,
    *,
    answer_backbone=None,
    reuse_initial=False,
    record=True,
    profile=False,
    host_limit_bytes=4 * 1024**3,
):
    """Collect a fixed-NFE microbatch; replay storage bounded and staged to CPU.

    Uniforms have shape [NFE, batch, answer_length]. Storage stays FP32 for
    hidden features in this first version. Forced all-kept steps are not saved.
    Request IDs/group IDs are carried by the caller; collection never filters
    or reorders groups and cannot update the policy.
    """
    if selector.training or selector.config.architecture != "mlp":
        raise ValueError("Frozen eval-mode pointwise MLP required")
    cfg = selector.config
    if (
        cfg.input_node_dim != 3
        or cfg.position_encoding != "learned"
        or cfg.feature_encoding != "none"
    ):
        raise ValueError("Prototype is limited to the SOTA position(+hidden) input contract")
    if ids.ndim != 2 or positions.shape[0] != ids.shape[0] or attention.shape != ids.shape:
        raise ValueError("Incompatible request shapes")
    nfe, batch, length = uniforms.shape
    if nfe < 1 or positions.shape != (batch, length) or length < 1:
        raise ValueError("Invalid NFE or answer shapes")
    if positions.dtype != torch.long or (positions < 0).any() or (positions >= ids.shape[1]).any():
        raise ValueError("Answer indices out of range")
    if length > 1 and not (positions[:, 1:] > positions[:, :-1]).all():
        raise ValueError("Answer positions must be unique and increasing")
    ks = [topk_schedule("linear", s, nfe, length) for s in range(1, nfe + 1)]
    counts = [k for k in ks if 0 < k < length]
    width = 3 + cfg.backbone_hidden_dim
    # Account for features, scores, masks, logp and worst-case order storage.
    required = len(counts) * batch * (length * (4 * width + 4 + 1 + 8) + 4)
    if record and required > host_limit_bytes:
        raise MemoryError(f"Replay needs {required} host bytes, limit={host_limit_bytes}")
    pin = ids.is_cuda
    alloc = lambda shape, dtype: torch.empty(shape, dtype=dtype, device="cpu", pin_memory=pin)
    features = alloc((len(counts), batch, length, width), torch.float32) if record else None
    scores = alloc((len(counts), batch, length), torch.float32) if record else None
    masks = alloc((len(counts), batch, length), torch.bool) if record else None
    old = alloc((len(counts), batch), torch.float32) if record else None
    orders = [alloc((batch, k), torch.long) for k in counts] if record else []
    current = ids.clone()
    revisions = torch.zeros((batch, length), device=ids.device, dtype=torch.int32)
    repeated = torch.zeros(batch, device=ids.device, dtype=torch.int32)
    uniforms = uniforms.to(ids.device)
    representatives = inverse = None
    first_backend = answer_backbone
    if reuse_initial:
        representatives, inverse = initial_representatives(ids, attention, positions)
    if hasattr(answer_backbone, "prepare"):
        first_backend = (
            answer_backbone.prepare(
                ids[representatives], attention[representatives], positions[representatives]
            )
            if reuse_initial
            else None
        )
        answer_backbone = answer_backbone.prepare(ids, attention, positions)
    timer = Timings(profile)
    record_index = 0
    for step, count in enumerate(ks):
        with timer.section("backbone_and_head"):
            reuse = reuse_initial and step == 0
            forward_ids = current[representatives] if reuse else current
            forward_mask = attention[representatives] if reuse else attention
            forward_pos = positions[representatives] if reuse else positions
            backend = first_backend if reuse else answer_backbone
            if backend is not None:
                logits, hidden = backend(forward_ids, forward_mask, forward_pos)
            else:
                out = backbone.forward(forward_ids, forward_mask, timestep=0)
                logits = out["logits"].gather(
                    1, forward_pos[..., None].expand(-1, -1, out["logits"].shape[-1])
                )
                hidden = out["hidden_states"].gather(
                    1, forward_pos[..., None].expand(-1, -1, out["hidden_states"].shape[-1])
                )
                del out
            if reuse:
                logits, hidden = logits[inverse], hidden[inverse]
        with timer.section("features_policy_sampling"):
            answer = current.gather(1, positions)
            # The final keep set is forced; no policy or softmax is needed.
            predicted = logits.argmax(-1)
            if count == length:
                following = torch.where(answer.eq(backbone.mask_token_id), predicted, answer)
                current.scatter_(1, positions, following)
                del logits, hidden
                continue
            f, sc, masked = features_from_answer(
                logits, hidden if cfg.backbone_hidden_dim else None, answer, backbone.mask_token_id
            )
            z = selector(f.reshape(batch * length, width), sc.flatten()).reshape(batch, length)
            order = sample_order(z, uniforms[step], count)
            logp = ordered_logprob(z, order)
            following, revised = transition(answer, predicted, order, backbone.mask_token_id)
            repeated += (revised & (revisions > 0)).sum(-1).to(torch.int32)
            revisions += revised.to(torch.int32)
            current.scatter_(1, positions, following)
        with timer.section("replay_to_host"):
            if record and 0 < count < length:
                features[record_index].copy_(f, non_blocking=pin)
                scores[record_index].copy_(sc, non_blocking=pin)
                masks[record_index].copy_(masked, non_blocking=pin)
                old[record_index].copy_(logp, non_blocking=pin)
                orders[record_index].copy_(order, non_blocking=pin)
                record_index += 1
        del logits, hidden, f, sc, z
    if pin:
        torch.cuda.synchronize()
    return BatchResult(
        current,
        revisions.sum(-1),
        repeated,
        nfe,
        features,
        scores,
        orders,
        old,
        counts,
        masks,
        timer.resolve(),
    )
