"""Stateless exposure-based LR: identical across pauses and skipped groups."""
import math


def cosine_group_lr(peak, floor_ratio, group_index, total_groups):
    """Zero-based group index; exact peak at first and floor at last group."""
    if not math.isfinite(peak) or peak <= 0 or not 0 <= floor_ratio <= 1:
        raise ValueError('Invalid cosine learning rate')
    if total_groups < 2 or group_index < 0:
        raise ValueError('Invalid cosine group horizon/index')
    # After the configured horizon, hold the floor; never restart the cosine.
    progress = min(group_index, total_groups - 1) / (total_groups - 1)
    return peak * (floor_ratio + (1 - floor_ratio) * .5 * (1 + math.cos(math.pi * progress)))
