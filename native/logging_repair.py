"""JSON-only encoding of non-finite APPLe extracted answers; no scoring changes.

Do not apply a blanket nan_to_num to diagnostics or training tensors. Only the
informational `extracted` field is tagged, after the official scorer ran.
"""
import json
import math
import os


def encode_metadata(value, key=None):
    if isinstance(value, float) and not math.isfinite(value):
        if key != "extracted":
            raise ValueError(f"Non-finite numeric value outside extracted metadata: {key}")
        return {"__nonfinite_float__": "nan" if math.isnan(value) else
                ("+inf" if value > 0 else "-inf")}
    if isinstance(value, dict):
        return {k: encode_metadata(v, k) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_metadata(v) for v in value]
    return value


def append(path, value):
    # Validate the entire record before opening the destination.
    text = json.dumps(encode_metadata(value), allow_nan=False) + "\n"
    with path.open("a") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def save(path, value):
    text = json.dumps(encode_metadata(value), allow_nan=False, indent=2)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
