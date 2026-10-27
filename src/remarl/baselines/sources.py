"""Pinned upstream decoder sources, stored outside the installed package."""

import json
import os
from pathlib import Path
from ..io import sha256


def source_root():
    return Path(os.environ.get("REMARL_BASELINE_SOURCES", "assets/baselines")).resolve()


def download_sources(output):
    from ..downloads import fetch

    manifest = json.loads((Path(__file__).parent / "upstream.json").read_text())
    for name, entry in manifest.items():
        for filename, checksum in entry["files"].items():
            destination = Path(output) / name / filename
            if not destination.exists():
                fetch(
                    f"https://raw.githubusercontent.com/{entry['repository']}/{entry['commit']}/{filename}",
                    destination,
                )
            if sha256(destination) != checksum:
                raise ValueError(f"Upstream source hash mismatch: {destination}")
    print(f"Verified upstream decoders in {Path(output).resolve()}")
