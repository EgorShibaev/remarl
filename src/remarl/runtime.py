"""Frozen generator and policy loading shared by training and evaluation."""

from pathlib import Path
from types import SimpleNamespace
import os
import torch
from .backbone import LLaDAOfflineModel
from .context import extend
from .optimization import state_digest
from .policy import ReMaRLPolicy, PolicyConfig
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


def load_policy(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    state = saved["model"]
    digest = state_digest(state)
    if digest != saved["policy_sha256"]:
        raise ValueError(f"Policy hash mismatch: {path}")
    from .ablations import configure, selector_class

    variant = saved.get("config", {}).get("variant", "full")
    configure(variant)
    policy_class = ReMaRLPolicy if variant == "full" else selector_class(ReMaRLPolicy)
    policy = policy_class(PolicyConfig.from_dict(saved["selector_config"])).to(device).eval()
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
        self.engine = LLaDAOfflineModel.from_local_checkpoint(
            Path(model), device=self.device, backend=backend, fuse_projections=True
        )
        extend(self.engine, max_length, model)

    def generate(self, rows, uniforms, record=False):
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        ids, attention, positions = [
            t.to(self.device) for t in canvas(rows, self.tokenizer.pad_token_id, uniforms.shape[-1])
        ]
        result = collect(
            SimpleNamespace(mask_token_id=MASK_ID),
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
