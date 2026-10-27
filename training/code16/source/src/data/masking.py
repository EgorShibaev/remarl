from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MaskedBatch:
    input_ids_masked: torch.Tensor
    clean_input_ids: torch.Tensor
    attention_mask: torch.Tensor
    masked_positions: list[torch.Tensor]


def _span_select(valid_pos: torch.Tensor, count: int, generator: torch.Generator | None) -> torch.Tensor:
    n = int(valid_pos.numel())
    if count >= n:
        return valid_pos.sort().values
    start = int(torch.randint(0, n - count + 1, (1,), generator=generator, device=valid_pos.device).item())
    return valid_pos[start : start + count].sort().values


def sample_mask_positions(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_ratio: float,
    special_token_ids: set[int],
    max_masked_positions: int | None = None,
    generator: torch.Generator | None = None,
    strategy: str = "random",
    restrict_token_ids: set[int] | None = None,
) -> list[torch.Tensor]:
    positions: list[torch.Tensor] = []
    batch, _ = input_ids.shape
    special = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in special_token_ids:
        special |= input_ids.eq(token_id)
    valid = attention_mask.bool() & ~special
    if restrict_token_ids:
        restrict = torch.tensor(sorted(restrict_token_ids), device=input_ids.device)
        valid &= torch.isin(input_ids, restrict)
    for row in range(batch):
        valid_pos = torch.nonzero(valid[row], as_tuple=False).flatten()
        if valid_pos.numel() == 0:
            positions.append(valid_pos)
            continue
        count = max(1, int(round(float(valid_pos.numel()) * mask_ratio)))
        if max_masked_positions is not None:
            count = min(count, max_masked_positions)
        count = min(count, int(valid_pos.numel()))
        if strategy == "span":
            positions.append(_span_select(valid_pos, count, generator))
        else:
            perm = torch.randperm(valid_pos.numel(), generator=generator, device=valid_pos.device)
            positions.append(valid_pos[perm[:count]].sort().values)
    return positions


def apply_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_token_id: int,
    mask_ratio: float,
    special_token_ids: set[int],
    max_masked_positions: int | None = None,
    generator: torch.Generator | None = None,
    strategy: str = "random",
    restrict_token_ids: set[int] | None = None,
) -> MaskedBatch:
    masked = input_ids.clone()
    positions = sample_mask_positions(
        input_ids=input_ids,
        attention_mask=attention_mask,
        mask_ratio=mask_ratio,
        special_token_ids=special_token_ids,
        max_masked_positions=max_masked_positions,
        generator=generator,
        strategy=strategy,
        restrict_token_ids=restrict_token_ids,
    )
    for row, pos in enumerate(positions):
        masked[row, pos] = mask_token_id
    return MaskedBatch(
        input_ids_masked=masked,
        clean_input_ids=input_ids.clone(),
        attention_mask=attention_mask.clone(),
        masked_positions=positions,
    )


def get_masked_positions(input_ids: torch.Tensor, mask_token_id: int) -> list[torch.Tensor]:
    return [torch.nonzero(row.eq(mask_token_id), as_tuple=False).flatten() for row in input_ids]


def mask_run_features(masked_positions: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Return span length, index inside span, left gap, right gap for each masked position.

    Vectorized: detect run boundaries via boolean diff on a per-token mask
    indicator and gather per-position span bounds in one shot.
    """
    if masked_positions.numel() == 0:
        return torch.empty((0, 4), dtype=torch.float32, device=masked_positions.device)
    device = masked_positions.device
    is_masked = torch.zeros(seq_len, dtype=torch.bool, device=device)
    is_masked[masked_positions] = True
    prev = torch.cat([torch.zeros(1, dtype=torch.bool, device=device), is_masked[:-1]])
    next_ = torch.cat([is_masked[1:], torch.zeros(1, dtype=torch.bool, device=device)])
    is_start = is_masked & ~prev
    is_end = is_masked & ~next_
    group_id_seq = is_start.long().cumsum(0)  # 1-indexed at masked positions
    starts_pos = torch.nonzero(is_start, as_tuple=False).flatten()
    ends_pos = torch.nonzero(is_end, as_tuple=False).flatten()
    pos_long = masked_positions.long()
    group_ids = group_id_seq[pos_long] - 1  # 0-indexed into starts_pos/ends_pos
    left = starts_pos[group_ids]
    right = ends_pos[group_ids]
    span_len = (right - left + 1).float()
    index = (pos_long - left).float()
    left_gap = left.float()
    right_gap = (seq_len - 1 - right).float()
    return torch.stack([span_len, index, left_gap, right_gap], dim=-1)


def _mask_run_features_legacy(masked_positions: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Reference Python-loop implementation, kept for parity tests."""
    if masked_positions.numel() == 0:
        return torch.empty((0, 4), dtype=torch.float32, device=masked_positions.device)
    is_masked = torch.zeros(seq_len, dtype=torch.bool, device=masked_positions.device)
    is_masked[masked_positions] = True
    rows: list[list[float]] = []
    for pos_tensor in masked_positions:
        pos = int(pos_tensor.item())
        left = pos
        while left - 1 >= 0 and bool(is_masked[left - 1]):
            left -= 1
        right = pos
        while right + 1 < seq_len and bool(is_masked[right + 1]):
            right += 1
        span_len = right - left + 1
        index = pos - left
        left_gap = left
        right_gap = seq_len - 1 - right
        rows.append([float(span_len), float(index), float(left_gap), float(right_gap)])
    return torch.tensor(rows, dtype=torch.float32, device=masked_positions.device)

