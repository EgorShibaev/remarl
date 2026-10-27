# ReMaRL

Code for **ReMaRL: Learning to Unmask and Remask in Frozen Diffusion Language Models**.

ReMaRL learns which answer positions to reveal, retain, or remask while keeping LLaDA-8B-Instruct frozen. A small MLP reads token confidence, mask state, position, and hidden features. GRPO trains the policy using final-answer correctness rewards.

[Public checkpoints](https://drive.google.com/drive/u/0/folders/1UFagh30vsvA-3jK-3sAf_iREaiQ3mxki) · [Training](docs/training.md) · [Evaluation](docs/evaluation.md) · [Inference](docs/inference.md) · [Plotting repository](https://github.com/EgorShibaev/remarl-plots)

## Install

Use Python 3.12 and Linux with an NVIDIA GPU for training and benchmark evaluation.

```bash
git clone git@github.com:EgorShibaev/remarl.git
cd remarl
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[gpu,test]'
```

The default backend uses vLLM's FlashAttention kernel. A PyTorch backend is available with `--backend torch`. See [runtime requirements](docs/training.md#runtime) for hardware and environment details. Code benchmarks also require the [execution sandbox](docs/evaluation.md#code-execution).

## Download checkpoints

```bash
remarl download backbone --output assets/llada
remarl download checkpoints --output checkpoints
```

Checkpoint downloads are public and require no Google account. The command downloads, extracts, and verifies the eight policies automatically. The frozen backbone is downloaded separately from its pinned Hugging Face revision.

| Domain | Training NFE | Checkpoint files |
|---|---|---|
| Math | 4, 8, 16, 32, 64 | `math4-32k.pt`, `math8-32k.pt`, `math16-32k.pt`, `math32-32k.pt`, `math64-32k.pt` |
| Code | 8, 16, 32 | `code8-32k.pt`, `code16-32k.pt`, `code32-32k.pt` |

All released policies were trained for 32k accepted groups. File and policy hashes are in [the checkpoint manifest](checkpoints/manifest.json).

## Reproduce an experiment

Prepare the math data:

```bash
remarl download math-data --output assets/raw/math
remarl prepare math --raw-data assets/raw/math --model assets/llada --output assets/data/math
```

Evaluate the released NFE8 policy on both complete math benchmarks:

```bash
remarl evaluate --policy checkpoints/math8-32k.pt --nfe 8 \
  --model assets/llada --data assets/data/math --output runs/math8-eval
```

Or train a fresh NFE8 policy:

```bash
remarl train --config configs/math/nfe8.json \
  --model assets/llada --data assets/data/math --output runs/math8
```

The trainer initializes the policy directly; training does not require a released checkpoint. It saves checkpoints, training metrics, and the configuration used. The same commands support math, code, component ablations, and training-data-size experiments through their configurations.

## Repository

```text
src/remarl/       Policy, frozen backbone, GRPO trainer, decoders, and task protocols
configs/        Training recipes and baseline evaluation settings
checkpoints/    Public download manifest and checksums
results/        Recorded benchmark scores and source receipts
docs/           Training, inference, and evaluation instructions
tests/          Optimizer, checkpoint, rollout, and decoder-reference tests
environments/   Package inventories for the original experiments and release checks
```

Plot rendering and its numerical inputs live in [remarl-plots](https://github.com/EgorShibaev/remarl-plots). Generated weights, datasets, and run outputs are excluded from Git.

## Verification

```bash
python -m pytest -q
remarl verify checkpoints
```

To include the upstream decoder comparisons:

```bash
remarl download baseline-sources --output assets/baselines
python -m pytest -q
```

Checks cover the original checkpoint outputs, saved dataset rows and prompts, rollout/replay agreement, deterministic training resume, and decoder traces against pinned upstream code. Full GPU training and benchmark evaluation have not been rerun after this reorganization. See [verification details](docs/verification.md).

Third-party code, data, and model weights retain their original terms; see [attribution](THIRD_PARTY.md).
