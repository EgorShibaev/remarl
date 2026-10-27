from __future__ import annotations

from src.decoding.confidence_decoder import ConfidenceDecoder, ConfidenceDecoderConfig


class SequentialDecoder(ConfidenceDecoder):
    """Near-sequential baseline: decode one highest-confidence token per step."""

    def __init__(self, backbone, max_steps: int = 128) -> None:
        super().__init__(
            backbone,
            ConfidenceDecoderConfig(
                max_steps=max_steps,
                min_confidence=0.0,
                budget_fraction=0.0,
                force_complete=True,
            ),
        )

