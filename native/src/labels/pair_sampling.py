from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PairSamplingConfig:
    random: float = 0.30
    nearby: float = 0.25
    high_confidence: float = 0.25
    same_span: float = 0.10
    topk_overlap: float = 0.10
    nearby_window: int = 16
    max_pairs: int | None = None


def all_undirected_pairs(num_nodes: int, device: torch.device | None = None) -> torch.Tensor:
    if num_nodes < 2:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    rows, cols = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)
    return torch.stack([rows, cols], dim=-1)


def same_span_mask(masked_positions: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    if pairs.numel() == 0:
        return torch.empty(0, dtype=torch.bool, device=masked_positions.device)
    if masked_positions.numel() == 0:
        return torch.zeros(pairs.shape[0], dtype=torch.bool, device=masked_positions.device)
    positions = masked_positions.long()
    diffs = positions[1:] - positions[:-1]
    new_group = torch.cat(
        [
            torch.ones(1, dtype=torch.long, device=positions.device),
            (diffs != 1).long(),
        ]
    )
    span_ids = new_group.cumsum(0) - 1
    return span_ids[pairs[:, 0]].eq(span_ids[pairs[:, 1]])


def _same_span_mask_legacy(masked_positions: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """Reference Python-loop implementation, kept for parity tests."""
    if pairs.numel() == 0:
        return torch.empty(0, dtype=torch.bool, device=masked_positions.device)
    positions = masked_positions.detach().cpu().tolist()
    span_ids: list[int] = []
    current_span = -1
    previous = None
    for pos in positions:
        if previous is None or pos != previous + 1:
            current_span += 1
        span_ids.append(current_span)
        previous = pos
    span_tensor = torch.tensor(span_ids, dtype=torch.long, device=masked_positions.device)
    return span_tensor[pairs[:, 0]].eq(span_tensor[pairs[:, 1]])


def _first_occurrence_mask(values: torch.Tensor) -> torch.Tensor:
    """Mask of first-occurrence positions inside each row.

    For row ``values[i]``, ``mask[i, a]`` is ``True`` iff ``values[i, a]`` does
    not appear at any earlier position in row ``i``. Used to de-duplicate rows
    for set-based IoU computation, matching the legacy ``set()`` semantics.
    """
    if values.dim() == 1:
        values = values.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    k = values.shape[-1]
    if k == 0:
        return torch.empty_like(values, dtype=torch.bool)
    idx = torch.arange(k, device=values.device)
    earlier = idx[None, :, None] > idx[None, None, :]
    same = values[..., :, None].eq(values[..., None, :])
    not_first = (earlier & same).any(dim=-1)
    mask = ~not_first
    return mask.squeeze(0) if squeeze else mask


def topk_overlap_matrix(topk_ids: torch.Tensor) -> torch.Tensor:
    """IoU of top-k id sets for every node pair. Diagonal is forced to 0.

    Matches the legacy ``set()``-based semantics: duplicates within a row are
    collapsed before computing intersection/union sizes.
    """
    num_nodes, k = topk_ids.shape
    if num_nodes == 0:
        return torch.zeros((0, 0), dtype=torch.float32, device=topk_ids.device)
    if k == 0:
        return torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=topk_ids.device)
    is_first = _first_occurrence_mask(topk_ids)  # [N, k]
    eq = topk_ids[:, None, :, None].eq(topk_ids[None, :, None, :])  # [N, N, k, k]
    in_other = eq.any(dim=-1)  # [N, N, k]
    intersect = (in_other & is_first[:, None, :]).sum(dim=-1).float()
    unique_count = is_first.sum(dim=-1).float()
    union = (unique_count[:, None] + unique_count[None, :] - intersect).clamp_min(1.0)
    overlap = intersect / union
    overlap.fill_diagonal_(0.0)
    return overlap


def topk_overlap_at_pairs(topk_ids: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """IoU of top-k id sets evaluated only at the given pairs.

    Cheaper than building the full NxN matrix when pairs are sparse. Pairs are
    treated as undirected, so the caller should ensure ``pairs[:, 0] != pairs[:, 1]``.
    """
    if pairs.numel() == 0:
        return torch.empty(0, dtype=torch.float32, device=topk_ids.device)
    k = topk_ids.shape[1]
    if k == 0:
        return torch.zeros(pairs.shape[0], dtype=torch.float32, device=topk_ids.device)
    a = topk_ids[pairs[:, 0]]  # [P, k]
    b = topk_ids[pairs[:, 1]]
    is_first_a = _first_occurrence_mask(a)
    is_first_b = _first_occurrence_mask(b)
    eq = a[:, :, None].eq(b[:, None, :])  # [P, k, k]
    in_b = eq.any(dim=-1)  # [P, k]
    intersect = (in_b & is_first_a).sum(dim=-1).float()
    unique_a = is_first_a.sum(dim=-1).float()
    unique_b = is_first_b.sum(dim=-1).float()
    union = (unique_a + unique_b - intersect).clamp_min(1.0)
    return intersect / union


def _topk_overlap_matrix_legacy(topk_ids: torch.Tensor) -> torch.Tensor:
    """Reference Python-loop implementation, kept for parity tests."""
    num_nodes = topk_ids.shape[0]
    overlap = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=topk_ids.device)
    for i in range(num_nodes):
        ids_i = set(int(x) for x in topk_ids[i].detach().cpu())
        for j in range(i + 1, num_nodes):
            ids_j = set(int(x) for x in topk_ids[j].detach().cpu())
            value = len(ids_i & ids_j) / max(len(ids_i | ids_j), 1)
            overlap[i, j] = value
            overlap[j, i] = value
    return overlap


def sample_pairs(
    masked_positions: torch.Tensor,
    confidence: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    config: PairSamplingConfig = PairSamplingConfig(),
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    pairs = all_undirected_pairs(int(masked_positions.numel()), masked_positions.device)
    if pairs.numel() == 0 or config.max_pairs is None or pairs.shape[0] <= config.max_pairs:
        return pairs

    target = config.max_pairs
    buckets: list[torch.Tensor] = []
    distance = (masked_positions[pairs[:, 0]] - masked_positions[pairs[:, 1]]).abs()
    buckets.append(pairs[distance <= config.nearby_window])
    buckets.append(pairs[same_span_mask(masked_positions, pairs)])
    if confidence is not None:
        pair_conf = confidence[pairs[:, 0]] * confidence[pairs[:, 1]]
        k = min(max(1, int(target * config.high_confidence)), pairs.shape[0])
        buckets.append(pairs[torch.topk(pair_conf, k=k).indices])
    if topk_ids is not None:
        values = topk_overlap_at_pairs(topk_ids, pairs)
        k = min(max(1, int(target * config.topk_overlap)), pairs.shape[0])
        buckets.append(pairs[torch.topk(values, k=k).indices])

    random_k = min(max(1, int(target * config.random)), pairs.shape[0])
    random_idx = torch.randperm(pairs.shape[0], generator=generator, device=pairs.device)[:random_k]
    buckets.append(pairs[random_idx])

    merged = torch.unique(torch.cat([bucket for bucket in buckets if bucket.numel() > 0], dim=0), dim=0)
    if merged.shape[0] > target:
        idx = torch.randperm(merged.shape[0], generator=generator, device=merged.device)[:target]
        merged = merged[idx]
    order = torch.argsort(merged[:, 0] * max(int(masked_positions.numel()), 1) + merged[:, 1])
    return merged[order]
