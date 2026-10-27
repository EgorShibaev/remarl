"""Train one fresh policy; checkpoint every collection and evaluate fixed ages."""
import os
from pathlib import Path
import time
import torch
import train_batch_grpo as train
from logging_repair import append, save

BASE = Path(__file__).resolve().parent
ROOT = BASE / 'runs' / os.environ['SCALE_RUN'] / 'source'
train.append, train.save = append, save


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.device_count() == 2
    out = ROOT / 'validation'
    assert not out.exists() and not (ROOT / 'production').exists()
    save(ROOT.parent / 'STARTED.json', dict(time=time.time(), seed=int(os.environ['SCALE_SEED']),
         gpus=os.environ['CUDA_VISIBLE_DEVICES'], pid=os.getpid()))
    t = train.Trainer(ROOT, out, BASE / 'runtime', [0,1])
    try:
        assert t.recipe.nfe == 8 and t.total_batches == 1024 and t.recipe.updates_per_batch == 4
        initial = t.snapshot()
        train.validate(t)
        t.restore(initial)
        t.out = ROOT / 'production'
        t.out.mkdir()
        # The shared initial P2 test results already exist. No duplicate zero-age tests.
        evaluate = t.evaluate
        t.evaluate = lambda split, seed: None if t.completed == 0 else evaluate(split, seed)
        train.production(t)
        save(ROOT.parent / 'COMPLETE.json', dict(time=time.time(), accepted_groups=t.completed*16))
    finally:
        t.close()


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save(ROOT.parent / 'FAILED.json', dict(time=time.time(), type=type(exc).__name__, message=str(exc)))
        raise
