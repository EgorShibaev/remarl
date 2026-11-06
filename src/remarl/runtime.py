"""Frozen generator and policy loading shared by training and evaluation."""

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import torch
from .backbone import LLaDAOfflineModel
from .context import extend
from .optimization import state_digest
from .policy import PolicyConfig, policy_class_for
from .rollout import collect

MASK_ID = 126336
BACKBONE_ID = "GSAI-ML/LLaDA-8B-Instruct"
BACKBONE_REVISION = "08b83a6feb34df1a6011b80c3c00c7563e963b07"


def configure_compute():
    # Configure cuBLAS before the first CUDA context is created.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def load_tokenizer(model):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=True)


def backbone_spec(model):
    path = Path(model) / "config.json"
    cfg = json.loads(path.read_text())
    kind = cfg.get("model_type", "").lower()
    if kind == "dream":
        hidden = cfg["hidden_size"]
        mask = cfg["mask_token_id"]
    elif kind == "llada":
        hidden = cfg.get("d_model", cfg.get("hidden_size"))
        mask = cfg.get("mask_token_id", MASK_ID)
    else:
        raise ValueError(f"Unsupported frozen generator: {kind}")
    if not isinstance(hidden, int) or hidden <= 0:
        raise ValueError("Generator configuration has no hidden size")
    result = dict(
        kind=kind,
        hidden_size=hidden,
        mask_id=mask,
        config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    metadata = Path(model) / "remarl-model.json"
    if metadata.exists():
        receipt = json.loads(metadata.read_text())
        result.update(model_id=receipt["model_id"], revision=receipt["revision"])
    return result


def check_prepared_backbone(manifest, spec):
    recorded = manifest.get("backbone")
    if recorded is not None and recorded != spec:
        raise ValueError("Prepared prompts belong to another generator/tokenizer")


def load_policy(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    state = saved["model"]
    digest = state_digest(state)
    if digest != saved["policy_sha256"]:
        raise ValueError(f"Policy hash mismatch: {path}")
    from .ablations import selector_class

    variant = saved.get("config", {}).get("variant", "full")
    config = PolicyConfig.from_dict(saved["selector_config"])
    base_class = policy_class_for(config)
    policy_class = base_class if variant == "full" else selector_class(base_class, variant)
    policy = policy_class(config).to(device).eval()
    policy.load_state_dict(state, strict=True)
    return policy, saved


def canvas(rows, pad_id, length=256, mask_id=MASK_ID):
    maximum = max(len(row["prompt_ids"]) + length for row in rows)
    ids = torch.full((len(rows), maximum), pad_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    positions = torch.empty((len(rows), length), dtype=torch.long)
    for i, row in enumerate(rows):
        n = len(row["prompt_ids"])
        ids[i, :n] = torch.tensor(row["prompt_ids"])
        ids[i, n : n + length] = mask_id
        attention[i, : n + length] = 1
        positions[i] = torch.arange(n, n + length)
    return ids, attention, positions


class Generator:
    def __init__(self, model, policy, device="cuda:0", backend="vllm", max_length=4096):
        configure_compute()
        self.device = torch.device(device)
        self.policy = policy.to(self.device).eval()
        self.tokenizer = load_tokenizer(model)
        self.spec = backbone_spec(model)
        self.mask_id = self.spec["mask_id"]
        if self.policy.config.backbone_hidden_dim not in (0, self.spec["hidden_size"]):
            raise ValueError("Policy hidden size differs from the frozen generator")
        if self.spec["kind"] == "dream":
            if backend != "torch":
                raise ValueError("Dream currently requires --backend torch")
            from .dream import DreamAnswerModel

            self.engine = DreamAnswerModel.from_local_checkpoint(model, self.device)
        else:
            self.engine = LLaDAOfflineModel.from_local_checkpoint(
                Path(model), device=self.device, backend=backend, fuse_projections=True
            )
            extend(self.engine, max_length, model)

    def generate(self, rows, uniforms, record=False):
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        ids, attention, positions = [
            t.to(self.device)
            for t in canvas(rows, self.tokenizer.pad_token_id, uniforms.shape[-1], self.mask_id)
        ]
        result = collect(
            SimpleNamespace(mask_token_id=self.mask_id),
            self.policy,
            ids,
            attention,
            positions,
            uniforms,
            answer_backbone=self.engine,
            record=record,
        )
        answer_tokens = result.final_ids.gather(1, positions).cpu().tolist()
        # Keep only the replay on the host; no completed canvas needs GPU storage.
        result.final_ids = result.final_ids.cpu()
        result.revisions = result.revisions.cpu()
        result.repeat_revisions = result.repeat_revisions.cpu()
        return {
            "tokens": answer_tokens,
            "replay": result,
            "revisions": result.revisions.tolist(),
            "nfe": result.nfe,
        }
