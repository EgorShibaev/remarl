"""Shared LLaDA setup and the paper's held-out question splits."""

import datetime
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from remarl.backbone import LLaDAOfflineModel
from remarl.context import extend
from remarl.data.prepare import load_prepared
from remarl.runtime import canvas, configure_compute, load_tokenizer

LENGTH = 256
BUDGETS = (4, 8, 16, 32, 64, 128, 256)


def question_splits(rows):
    rng = random.Random(19571)
    gsm, math = (
        [i for i, row in enumerate(rows) if row["task"] == task] for task in ("gsm8k", "math")
    )
    rng.shuffle(gsm)
    rng.shuffle(math)
    schedule = gsm[:128]
    validation = [i for pair in zip(gsm[128:384], math[:256]) for i in pair]
    # Preserve the balanced 8,192-question subset at the start of the nested training sets.
    train = [i for pair in zip(gsm[384:], math[256:]) for i in pair][:8192]
    used = set(schedule + validation + train)
    remaining = [i for i in range(len(rows)) if i not in used]
    random.Random(411).shuffle(remaining)
    train.extend(remaining)
    if (len(train), len(validation), len(schedule)) != (14333, 512, 128):
        raise ValueError("Expected the repository's prepared GSM8K and MATH splits")
    return train, validation, schedule


def setup(parser):
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    configure_compute()
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=4))
    # Each invocation owns a fresh directory, including its rank-local outputs.
    if dist.get_rank() == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    rows, manifest = load_prepared(args.data)
    tokenizer = load_tokenizer(args.model)
    model = (
        LLaDAOfflineModel.from_local_checkpoint(
            args.model, device=device, backend="torch", fuse_projections=True
        )
        .eval()
        .requires_grad_(False)
    )
    extend(model, manifest["required_canvas"], args.model)
    return args, device, rows, tokenizer, model


def batches(items):
    for start in range(0, len(items), 4):
        yield items[start : start + 4]


def normalized(hidden):
    return F.normalize(hidden.float(), dim=-1) * hidden.shape[-1] ** 0.5


def bind(model, tokenizer, rows, device):
    ids, attention, positions = [t.to(device) for t in canvas(rows, tokenizer.pad_token_id, LENGTH)]
    forward = model.prepare(ids, attention, positions)

    def call(answer):
        ids.scatter_(1, positions, answer)
        return forward(ids, attention, positions)

    return ids.gather(1, positions), call
