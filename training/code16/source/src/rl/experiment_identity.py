"""Path-independent experiment identity and complete source/runtime resume guards."""

import copy
import hashlib
import json
from pathlib import Path


def semantic_config_digest(config, backbone_manifest):
    semantic = copy.deepcopy(config)
    semantic["backbone"]["name"] = "verified-local-backbone"
    # Only verified model bytes define backbone identity, not transfer/cache locations.
    semantic["backbone"]["content"] = sorted(
        (entry["local_path"], entry["bytes"], entry["sha256"])
        for entry in backbone_manifest["files"] if entry["local_path"].startswith("models/")
    )
    if not semantic["backbone"]["content"]:
        raise ValueError("Empty backbone manifest")
    return hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()


def source_fingerprint(repo):
    repo = Path(repo)
    paths = sorted(p for folder in ("src", "scripts") for p in (repo / folder).rglob("*.py")
                   if not p.name.startswith("._") and "__pycache__" not in p.parts)
    return {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def validate_resume_identity(saved, current):
    for key in ("config_sha256", "data_sha256", "source_sha256", "packages", "code_commit"):
        if saved.get(key) != current.get(key):
            raise ValueError(f"Refusing resume after {key} changed")


def pending_evaluations(control, completed):
    control = Path(control)
    return [step for step in (512, 1024, 3200) if step <= completed
            and not (control / f"development-{step}-complete.json").exists()]


def inference_payload(full_payload):
    """No optimizer or Python/NumPy RNG pickle objects in evaluation artifacts."""
    return {key: full_payload[key] for key in (
        "model_state", "set_selector_config", "planner_features", "feature_config", "provenance"
    ) if key in full_payload}
