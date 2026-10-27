from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset


LabelRecord = dict[str, Any]


def save_label_shard(path: str | Path, records: list[LabelRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, path)


def load_label_shard(path: str | Path) -> list[LabelRecord]:
    records = torch.load(path, map_location="cpu")
    if not isinstance(records, list):
        raise ValueError(f"Label shard must contain a list: {path}")
    return records


def iter_label_records(label_dir: str | Path) -> Iterator[LabelRecord]:
    for shard in sorted(Path(label_dir).glob("labels-*.pt")):
        yield from load_label_shard(shard)


def iter_feature_records(feature_cache_dir: str | Path) -> Iterator[dict[str, Any]]:
    for shard in sorted(Path(feature_cache_dir).glob("features-*.pt")):
        payload = torch.load(shard, map_location="cpu")
        if not isinstance(payload, list):
            raise ValueError(f"Feature shard must contain a list: {shard}")
        yield from payload


class LabelRecordDataset(Dataset[LabelRecord]):
    """Dataset over label shards.

    When ``feature_cache_dir`` is provided, each returned record is augmented
    with a ``cached_features`` dict so the training loop can skip the backbone
    forward pass.
    """

    def __init__(
        self,
        label_dir: str | Path,
        feature_cache_dir: str | Path | None = None,
    ) -> None:
        self.records = list(iter_label_records(label_dir))
        if not self.records:
            raise ValueError(f"No label records found in {label_dir}")
        if feature_cache_dir is not None:
            self.feature_records: list[dict[str, Any]] | None = list(
                iter_feature_records(feature_cache_dir)
            )
            if len(self.feature_records) != len(self.records):
                raise ValueError(
                    f"Label/feature record count mismatch: "
                    f"{len(self.records)} labels vs {len(self.feature_records)} features"
                )
        else:
            self.feature_records = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> LabelRecord:
        record = self.records[index]
        if self.feature_records is not None:
            record = dict(record)
            record["cached_features"] = self.feature_records[index]
        return record

