"""Cache-free rollout with ordered position sampling and frozen-policy replay."""

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from .schedules import topk_schedule


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
    hidden features. Forced all-kept steps are not saved.
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
        raise ValueError("Expected confidence, mask, learned-position and hidden features")
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
            temperature = getattr(selector, "temperature", 1.0)
            if getattr(selector, "variant", "full") == "no_revision":
                from .ablations import select_order, action_logprob

                order = select_order(
                    z, uniforms[step], count, masked, mode=selector.variant, temperature=temperature
                )
                logp = action_logprob(
                    z, order, masked, mode=selector.variant, temperature=temperature
                )
            else:
                if temperature == 0:
                    order = z.argsort(dim=-1, descending=True, stable=True)[:, :count]
                    logp = z.sum(-1) * 0
                else:
                    order = sample_order(z, uniforms[step], count, temperature)
                    logp = ordered_logprob(z, order, temperature)
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
