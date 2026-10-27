"""Exact immutable scientific-row hashing, without screening dependencies."""

import hashlib
import json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_row(row):
    fields = {k: row[k] for k in ("question_id", "question", "solution", "test", "test_info")}
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()
