"""Fit the published linear error predictor, or repeat the validation sweep."""

import argparse
import datetime
import gc
import re
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from common import LENGTH, batches, bind, normalized, question_splits, setup
from remarl.data.prepare import load_named
from remarl.io import save_checkpoint, save_json
from remarl.runtime import MASK_ID


def reference(row, raw, tokenizer):
    task = row["task"]
    item = raw[task][int(row["source_id"].rsplit(":", 1)[1])]
    if row["question"] != item["question" if task == "gsm8k" else "problem"]:
        raise ValueError("Prepared question does not match raw solution")
    reasoning = (
        item["solution"]
        if task == "math"
        else re.sub(r"<<[^>]*>>", "", item["answer"].rsplit("####", 1)[0]).strip()
    )
    text = (
        "\n"
        + reasoning
        + "\n</reasoning>\n<answer>\n\\boxed{"
        + str(row["gold"]["answer"])
        + "}\n</answer>"
    )
    return (tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id])[:LENGTH]


@torch.inference_mode()
def collect(rows, indices, view, raw, tokenizer, model, device, path):
    features, labels, counts = [], [], []
    for batch in batches(indices):
        questions = [rows[i] for i in batch]
        refs = [reference(q, raw, tokenizer) for q in questions]
        clean = torch.tensor(
            [r + [tokenizer.eos_token_id] * (LENGTH - len(r)) for r in refs], device=device
        )
        partial = clean.clone()
        reveal = torch.zeros_like(clean, dtype=torch.bool)
        for j, i in enumerate(batch):
            gen = torch.Generator(device=device).manual_seed(19571 + i + 10**6 * view)
            rate = 0.1 + 0.9 * torch.rand((), generator=gen, device=device).item()
            mask = torch.rand(LENGTH, generator=gen, device=device) < rate
            partial[j, mask] = MASK_ID
            fraction = 1.0 if (i + view) % 2 == 0 else 0.5
            reveal[j] = mask & (torch.rand(LENGTH, generator=gen, device=device) < fraction)
        _, forward = bind(model, tokenizer, questions, device)
        logits, _ = forward(partial)
        logits[..., MASK_ID] = -torch.inf
        draft = torch.where(reveal, logits.argmax(-1), partial)
        del logits
        logits, hidden = forward(draft)
        del logits
        for j, ref in enumerate(refs):
            eligible = draft[j, : len(ref)] != MASK_ID
            features.append(hidden[j, : len(ref)][eligible].half().cpu())
            labels.append((draft[j, : len(ref)] != clean[j, : len(ref)])[eligible].cpu())
            counts.append(len(features[-1]))
    save_checkpoint(
        path, (torch.cat(features), torch.cat(labels), torch.tensor(counts), torch.tensor(indices))
    )


def fit_head(parts, ridge, device):
    count = torch.tensor(sum(len(y) for x, y in parts), device=device)
    dist.all_reduce(count)
    dimension = parts[0][0].shape[-1]
    theta = torch.zeros(dimension + 1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS(
        [theta],
        max_iter=90,
        lr=1,
        line_search_fn="strong_wolfe",
        tolerance_change=1e-9,
        tolerance_grad=1e-6,
    )
    denominator = int(count.item())

    @torch.no_grad()
    def closure():
        # Accumulate the full objective and gradient before each L-BFGS update.
        acc = torch.zeros(dimension + 2, device=device)
        for features, labels in parts:
            for start in range(0, len(labels), 8192):
                x = normalized(features[start : start + 8192].to(device))
                y = labels[start : start + 8192].to(device, dtype=torch.float32)
                z = x @ theta[:-1] + theta[-1]
                residual = z.sigmoid() - y
                acc[:-2] += x.T @ residual
                acc[-2] += residual.sum()
                acc[-1] += F.binary_cross_entropy_with_logits(z, y, reduction="sum")
        dist.all_reduce(acc)
        acc /= denominator
        theta.grad = acc[:-1].clone()
        theta.grad[:-1] += ridge * theta[:-1]
        return acc[-1] + ridge / 2 * theta[:-1].square().sum()

    opt.step(closure)
    return theta[:-1].detach(), theta[-1].detach(), denominator


def metrics(z, y, question_ids):
    y = y.float()
    order = z.argsort(descending=True, stable=True)
    sorted_scores = z[order]
    ends = torch.where(
        torch.cat((sorted_scores[1:] != sorted_scores[:-1], torch.ones(1, dtype=torch.bool)))
    )[0]
    positives = y[order].cumsum(0)[ends]
    ap = (
        positives.diff(prepend=positives.new_zeros(1)) * positives / (ends + 1)
    ).sum() / positives[-1]
    _, index = question_ids.unique(return_inverse=True)
    loss = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    mean_loss = (
        torch.zeros(index.max() + 1).scatter_add_(0, index, loss) / index.bincount()
    ).mean()
    return dict(
        token_accuracy=((z >= 0) == y.bool()).float().mean().item(),
        average_precision=ap.item(),
        question_mean_log_loss=mean_loss.item(),
        tokens=len(y),
        questions=len(index.unique()),
    )


def validate(shards, w, b, cpu_group):
    logits, labels, questions = [], [], []
    for x, y, counts, ids in shards:
        logits.extend((normalized(chunk.to(w.device)) @ w + b).cpu() for chunk in x.split(8192))
        labels.append(y)
        questions.append(ids.repeat_interleave(counts))
    local = tuple(torch.cat(items) for items in (logits, labels, questions))
    gathered = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
    dist.gather_object(local, gathered, dst=0, group=cpu_group)
    if gathered is not None:
        return metrics(*(torch.cat(items) for items in zip(*gathered)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--sweep", action="store_true", help="Fit all 18 head configurations")
    args, device, rows, tokenizer, model = setup(parser)
    train, valid, _ = question_splits(rows["train"])
    rank, world = dist.get_rank(), dist.get_world_size()
    cpu_group = dist.new_group(backend="gloo", timeout=datetime.timedelta(hours=4))
    raw = {
        task: load_named(args.raw_data, name)[0]
        for task, name in (("gsm8k", "gsm8k-train"), ("math", "math-500-train"))
    }
    shards = {}
    for tag, ids, views in (("valid", valid, 4), ("train", train, 8)):
        shards[tag] = []
        for view in range(views):
            path = args.output / f"{tag}-view{view}-rank{rank}.pt"
            collect(rows["train"], ids[rank::world], view, raw, tokenizer, model, device, path)
            shards[tag].append(torch.load(path, mmap=True, weights_only=True))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    for view, (features, labels, counts, indices) in enumerate(shards["train"]):
        free, _ = torch.cuda.mem_get_info(device)
        if free > features.nbytes + labels.nbytes + 5 * 1024**3:
            shards["train"][view] = (features.to(device), labels.to(device), counts, indices)
    records = []
    for views in (1, 2, 4, 8) if args.sweep else (8,):
        for size in (512, 2048, 8192, 14333) if args.sweep else (14333,):
            parts = []
            for x, y, c, _ in shards["train"][:views]:
                end = int(c[: len(range(rank, size, world))].sum())
                parts.append((x[:end], y[:end]))
            ridges = (0.01, 0.001, 0.1) if (size, views) == (14333, 8) else (0.01,)
            for ridge in ridges if args.sweep else (0.001,):
                w, b, count = fit_head(parts, ridge, device)
                scores = validate(shards["valid"], w, b, cpu_group)
                if rank == 0:
                    name = f"q{size}-x{views}-l{ridge}.pt"
                    save_checkpoint(args.output / name, (w.cpu(), b.cpu()))
                    records.append(
                        dict(
                            head=name,
                            training_questions=size,
                            views=views,
                            training_tokens=count,
                            ridge=ridge,
                            validation=scores,
                        )
                    )
                    save_json(args.output / "validation.json", records)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
