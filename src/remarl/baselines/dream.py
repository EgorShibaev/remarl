"""Dream baseline adapters used for the released full-benchmark observations."""

import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from ..dream import DreamAnswerModel
from ..io import sha256
from ..protocols.official.apple_generation import (
    _fixed_step_unmask_decisions,
    generate_unified,
)
from ..runtime import backbone_spec, check_prepared_backbone, load_tokenizer
from .sources import source_root


def upstream(name, filename):
    manifest = json.loads((Path(__file__).parent / "upstream.json").read_text())
    path = source_root() / name / filename
    if not path.exists() or sha256(path) != manifest[name]["files"][filename]:
        raise ValueError("Download verified sources with remarl download baseline-sources")
    key = "remarl_upstream_" + name + "_" + path.stem
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


class DreamBaseline:
    def __init__(self, path, manifest, device, backend):
        if backend != "torch":
            raise ValueError("Dream baselines require --backend torch")
        self.spec = backbone_spec(path)
        if self.spec["kind"] != "dream":
            raise ValueError("This baseline configuration requires Dream")
        check_prepared_backbone(manifest, self.spec)
        self.tokenizer = load_tokenizer(path)
        self.engine = DreamAnswerModel.from_local_checkpoint(path, device)
        self.model = self.engine.model
        self.calls = 0
        self.model.model.register_forward_pre_hook(self.on_forward)

    def on_forward(self, *_):
        self.calls += 1

    def confidence(self, prompt, cfg):
        nfe, block = cfg["steps"], cfg["block"]
        quota = 256 // nfe
        if 256 % nfe or 256 % block or block % quota:
            raise ValueError("Incompatible NFE and block length")
        ids = torch.cat((prompt, torch.full((1, 256), self.spec["mask_id"],
                                           device=prompt.device, dtype=prompt.dtype)), 1)
        pos = torch.arange(prompt.shape[1], ids.shape[1], device=ids.device)[None]
        attention = torch.ones_like(ids)
        answer = ids.gather(1, pos)
        for t in range(nfe):
            start = (t * quota // block) * block
            region = slice(start, start + block)
            logits, _ = self.engine(ids, attention, pos)
            proposals = logits.argmax(-1)
            selected = _fixed_step_unmask_decisions(
                answer[:, region].eq(self.spec["mask_id"]), logits.softmax(-1),
                proposals, region, quota, "low_confidence")
            if not selected.sum(-1).eq(quota).all():
                raise RuntimeError("Confidence unmask count differs from schedule")
            answer = torch.where(selected, proposals, answer)
            ids.scatter_(1, pos, answer)
        return ids, nfe

    @torch.inference_mode()
    def decode(self, prompt, cfg):
        self.calls = 0
        mode, block, parameter = cfg["mode"], cfg["block"], cfg.get("parameter")
        mask = self.spec["mask_id"]
        if mode == "low_confidence":
            seq, nfe = self.confidence(prompt, cfg)
        elif mode == "fastdllm":
            result = generate_unified(self.model, prompt, "fastdllm", thres=parameter,
                gen_length=256, block_length=block, temperature=0., mask_id=mask,
                model_type="Dream")
            seq, nfe = result.sequences, result.steps_taken.item()
        elif mode == "eb":
            module = upstream("DAPD", "baselines/KLASS/src/model/dream_eb_wrapper.py")
            seq, stats = module.generate_eb_dream(self.model, prompt, gen_length=256,
                block_length=block, eb_gamma=parameter, mask_id=mask,
                remasking="low_confidence")
            nfe = stats["total_forward_passes"]
        elif mode == "klass":
            module = upstream("DAPD", "baselines/KLASS/src/model/dream_klass.py")
            settings = module.DreamGenerationConfig(max_length=prompt.shape[1] + 256,
                alg="klass", steps=256, conf_threshold=parameter, kl_threshold=.01,
                kl_history_length=2, unmask_strategy="all", block_length=block,
                mask_token_id=mask, temperature=0.)
            seq, nfe = module.DreamGenerationMixin._sample(self.model, prompt, None,
                settings, lambda step, x, logits: x, lambda step, x, logits: logits)
        elif mode in ("dapd_direct", "dapd_staged"):
            # Verify the full package before importing its relative dependencies.
            manifest = json.loads((Path(__file__).parent / "upstream.json").read_text())
            for filename, checksum in manifest["DAPD"]["files"].items():
                if filename.startswith("dapd/"):
                    path = source_root() / "DAPD" / filename
                    if not path.exists() or sha256(path) != checksum:
                        raise ValueError("Download verified DAPD sources first")
            sys.path.insert(0, str(source_root() / "DAPD"))
            try:
                module = importlib.import_module("dapd.dream_generation")
            finally:
                sys.path.pop(0)
            seq, stats = module.dream_generate_dapd_with_blocks(
                self.model, self.tokenizer, prompt, gen_length=256, block_length=block,
                dapd_alg=mode, layer_ratio=.3, tau_min=.005,
                tau_max=.01 if mode == "dapd_direct" else .05)
            nfe = stats["total_forward_passes"]
        else:
            raise ValueError(f"Unsupported Dream baseline: {mode}")
        if int(nfe) != self.calls or not 0 < nfe <= 256:
            raise RuntimeError("Reported and observed model-call counts differ")
        if seq.shape != (1, prompt.shape[1] + 256) or not torch.equal(seq[:, :prompt.shape[1]], prompt):
            raise RuntimeError("Decoder changed the prompt or answer length")
        if seq[:, -256:].eq(mask).any():
            raise RuntimeError("Decoder returned unfilled masks")
        return SimpleNamespace(sequences=seq, steps_taken=torch.tensor([int(nfe)]))
