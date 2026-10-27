"""Separate fresh validation/production processes with metadata-only logging repair."""
import evaluate_p2
from logging_repair import append, save

evaluate_p2.save = save
# train_batch_grpo defines its own append, so execute without __main__ then call APIs.
import argparse
from pathlib import Path
import torch
import train_batch_grpo as training

training.append, training.save = append, save
p = argparse.ArgumentParser()
p.add_argument('mode', choices=('validate', 'train'))
p.add_argument('--root', type=Path, required=True)
p.add_argument('--assets', type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)
assert torch.cuda.device_count() == 8
assert all('A100' in torch.cuda.get_device_name(i) for i in range(8))
out = a.root / ('validation' if a.mode == 'validate' else 'production')
assert not out.exists(), 'Never overwrite existing attempt'
t = training.Trainer(a.root, out, a.assets, list(range(8)))
try:
    training.validate(t) if a.mode == 'validate' else training.production(t)
finally:
    t.close()
