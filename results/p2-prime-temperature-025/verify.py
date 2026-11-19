"""Verify the published temperature-0.25 summaries without a GPU."""

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    folder = Path(__file__).resolve().parent
    results = folder.parent
    provenance = json.loads((folder / "provenance.json").read_text())
    for name, expected_hash in provenance["files"].items():
        actual_hash = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Hash mismatch: {name}")

    seeds = json.loads((folder / "seed-results.json").read_text())
    summaries = json.loads((folder / "summary.json").read_text())
    counts = {"gsm8k": 1319, "math500": 500, "humaneval": 164, "mbpp": 500}
    expected = {
        (benchmark, nfe)
        for benchmark in counts
        for nfe in ((4,) if benchmark in ("gsm8k", "math500") else ())
        + (8, 16, 32, 64, 128, 256)
    }
    if len(seeds) != 78 or len({row["cell"] for row in seeds}) != 78:
        raise ValueError("Expected 78 distinct seed evaluations")
    groups = defaultdict(list)
    for row in seeds:
        valid = (
            row["complete"]
            and row["answers_verified"]
            and row["method"] == "p2-prime"
            and row["temperature"] == 0.25
            and row["token_temperature"] == 0
            and row["answer_length"] == 256
            and row["blocks"] == 1
            and row["count"] == counts[row["benchmark"]]
            and row["correct"] == int(row["correct"])
            and 0 <= row["correct"] <= row["count"]
            and math.isclose(row["accuracy"], row["correct"] / row["count"], abs_tol=1e-14)
        )
        if not valid:
            raise ValueError(f"Invalid seed summary: {row['cell']}")
        groups[row["benchmark"], row["nfe"]].append(row)
    if set(groups) != expected or len(summaries) != len(expected):
        raise ValueError("Incomplete benchmark–budget grid")
    by_key = {(row["benchmark"], row["nfe"]): row for row in summaries}
    if set(by_key) != expected:
        raise ValueError("Summary grid differs from seed grid")
    for key, rows in groups.items():
        if sorted(row["seed"] for row in rows) != [101, 202, 303]:
            raise ValueError(f"Missing or repeated seed: {key}")
        scores = [100 * row["correct"] / row["count"] for row in rows]
        values = (statistics.mean(scores), min(scores), max(scores))
        for field, value in zip(("mean_percent", "min_percent", "max_percent"), values):
            if not math.isclose(by_key[key][field], value, abs_tol=1e-12):
                raise ValueError(f"Aggregation mismatch: {key}, {field}")

    with (folder / "summary.csv").open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    csv_keys = [(row["benchmark"], int(row["nfe"])) for row in csv_rows]
    if len(csv_keys) != len(expected) or set(csv_keys) != expected:
        raise ValueError("CSV summary grid mismatch")
    for row, key in zip(csv_rows, csv_keys):
        for field in ("mean_percent", "min_percent", "max_percent"):
            if not math.isclose(float(row[field]), by_key[key][field], abs_tol=1e-12):
                raise ValueError(f"CSV summary mismatch: {key}, {field}")

    with (results / "benchmark-observations.csv").open(newline="") as stream:
        observations = [row for row in csv.DictReader(stream) if row["method"] == "P2' (tau=0.25)"]
    observation_keys = [(row["benchmark"], float(row["mean_nfe"])) for row in observations]
    if len(observation_keys) != len(expected) or set(observation_keys) != expected:
        raise ValueError("Repository observations grid mismatch")
    for row, key in zip(observations, observation_keys):
        if (row["blocks"], row["source"], row["group"]) != ("1x256", "C", "fixed"):
            raise ValueError(f"Observation protocol mismatch: {key}")
        if not math.isclose(float(row["score_percent"]), by_key[key]["mean_percent"], abs_tol=1e-12):
            raise ValueError(f"Observation score mismatch: {key}")

    receipt = json.loads((results / "data-receipt.json").read_text())
    for name in ("seed-results.json", "summary.json"):
        key = f"results/p2-prime-temperature-025/{name}"
        if receipt["inputs"][key] != provenance["files"][name]:
            raise ValueError(f"Receipt hash mismatch: {name}")
    for benchmark in counts:
        rows = [row for row in receipt["tables"][benchmark]["fixed"]
                if row["label"] == "P2$'$ ($\\tau=0.25$)"]
        if len(rows) != 1:
            raise ValueError(f"Expected one calibrated row: {benchmark}")
        values = rows[0]["observations"]
        keys = [(benchmark, row["nfe"]) for row in values]
        if len(keys) != len(set(keys)) or set(keys) != {k for k in expected if k[0] == benchmark}:
            raise ValueError(f"Receipt grid mismatch: {benchmark}")
        for row, key in zip(values, keys):
            if not math.isclose(row["score"], by_key[key]["mean_percent"], abs_tol=1e-12):
                raise ValueError(f"Receipt score mismatch: {key}")
    print("Verified 78 seed summaries, 26 means, file hashes, and repository result files.")


if __name__ == "__main__":
    main()
