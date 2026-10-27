from __future__ import annotations

from dataclasses import dataclass

import torch

from src.data.masking import MaskedBatch, apply_mask


@dataclass
class MaskingCollator:
    mask_token_id: int
    special_token_ids: set[int]
    mask_ratio: float
    max_masked_positions: int | None = None

    def __call__(self, batch: dict[str, torch.Tensor]) -> MaskedBatch:
        return apply_mask(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            mask_token_id=self.mask_token_id,
            mask_ratio=self.mask_ratio,
            special_token_ids=self.special_token_ids,
            max_masked_positions=self.max_masked_positions,
        )

