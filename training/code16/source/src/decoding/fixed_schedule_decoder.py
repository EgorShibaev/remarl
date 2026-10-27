from __future__ import annotations

from src.decoding.confidence_decoder import ConfidenceDecoder, ConfidenceDecoderConfig


class FixedScheduleDecoder(ConfidenceDecoder):
    """Fixed-ratio baseline implemented as confidence ranking with no threshold."""

    def __init__(self, backbone, budget_fraction: float = 0.30, max_steps: int = 16) -> None:
        super().__init__(
            backbone,
            ConfidenceDecoderConfig(
                max_steps=max_steps,
                min_confidence=0.0,
                budget_fraction=budget_fraction,
                force_complete=True,
            ),
        )

