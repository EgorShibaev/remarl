"""Evaluate frozen-backbone baselines and retain measured model-call counts."""

import hashlib
import json
from pathlib import Path
import statistics
import torch
from ..backbone import LLaDAOfflineModel
from ..context import extend
from ..data.prepare import load_prepared
from ..io import append_json, save_json
from ..protocols.scoring import Scorer
from ..runtime import load_tokenizer, configure_compute
from .confidence import NativeAdapter


def run(config, data, model, output, device="cuda:0", backend="vllm"):
    configure_compute()
    from . import confidence, standard, revision, extra
    from .adapters import ExplicitAttentionAdapter
    from .model_adapter import Adapter, reference

    cfg = json.loads(Path(config).read_text())
    rows, manifest = load_prepared(data)
    engine = LLaDAOfflineModel.from_local_checkpoint(
        model, device=device, backend=backend, fuse_projections=True
    )
    extend(engine, manifest["required_canvas"], model)
    scorer = Scorer(load_tokenizer(model), manifest["domain"])
    method = cfg["mode"]
    ref = reference(engine) if method in ("rcr", "core", "dapd") else None
    summaries = []
    try:
        for split in [s for s in rows if s.startswith("test_")]:
            folder = Path(output) / split
            folder.mkdir(parents=True, exist_ok=True)
            identity = dict(
                config=cfg, split=split, data_sha256=manifest["rows_sha256"], backend=backend
            )
            spec = folder / "evaluation.json"
            if spec.exists() and json.loads(spec.read_text()) != identity:
                raise ValueError("Output belongs to another baseline evaluation")
            save_json(spec, identity)
            records_path = folder / "answers.jsonl"
            records = (
                [json.loads(s) for s in records_path.read_text().splitlines()]
                if records_path.exists()
                else []
            )
            if len(records) > len(rows[split]) or [r["index"] for r in records] != list(
                range(len(records))
            ):
                raise ValueError("Invalid saved evaluation prefix")
            scorer.begin_collection()
            for index in range(len(records), len(rows[split])):
                row = rows[split][index]
                prompt = torch.tensor([row["prompt_ids"]], device=device)
                key = f"{split}:{index}:{cfg.get('seed', 101)}:{cfg.get('steps', cfg.get('threshold'))}"
                torch.manual_seed(int(hashlib.sha256(key.encode()).hexdigest()[:15], 16))
                if method in ("low_confidence", "random", "fastdllm"):
                    adapter, decoder = NativeAdapter(engine, prompt), confidence.decode
                elif method in ("p2_original", "margin", "pc_sampler", "klass", "eb_sampler"):
                    adapter, decoder = NativeAdapter(engine, prompt), standard.decode
                elif method in ("wino", "remdm_uncode"):
                    adapter, decoder = ExplicitAttentionAdapter(engine), revision.decode
                elif method in ("rcr", "core", "dapd"):
                    adapter, decoder = Adapter(ref), extra.decode
                else:
                    raise ValueError(f"Unknown decoder: {method}")
                result = decoder(adapter, prompt, cfg)
                tokens = result.sequences[0, -256:].tolist()
                calls = int(result.steps_taken[0])
                if calls != adapter.calls or 126336 in tokens:
                    raise RuntimeError("Incorrect call count or incomplete answer")
                answer = scorer([tokens], [row["gold"]])[0]
                record = dict(index=index, tokens=tokens, nfe=calls, **answer)
                append_json(records_path, record)
                records.append(record)
            summary = dict(
                identity,
                complete=True,
                count=len(records),
                accuracy=statistics.mean(r["reward"] for r in records),
                mean_nfe=statistics.mean(r["nfe"] for r in records),
            )
            save_json(folder / "summary.json", summary)
            summaries.append(summary)
        return summaries
    finally:
        scorer.close()
