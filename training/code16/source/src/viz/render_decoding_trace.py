from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.viz.decoding_trace import save_step_count_plot


def render(run: str | Path) -> None:
    run = Path(run)
    metrics_path = run / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            record = json.loads(line)
            if "trace" not in record:
                continue
            save_step_count_plot(record["trace"], run / "figures" / f"trace-{line_index:04d}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    render(args.run)


if __name__ == "__main__":
    main()

