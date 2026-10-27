# Training

## Runtime

Training uses one complete frozen backbone replica per GPU. An 80 GB NVIDIA GPU provides room for the eight-answer rollout batches; multiple GPUs collect independent question groups. For NFE64 training, allow at least 64 GB of host RAM for replay and model loading.

The paper used BF16 backbone weights, FP32 policy/replay features, fused projections, and vLLM's noncausal FlashAttention kernel. [production-packages.txt](../environments/production-packages.txt) records that environment. `--backend torch` uses PyTorch SDPA. Backend and package changes can affect BF16 rounding and generated answers.

Install with `pip install -e '.[gpu,test]'`. For CPU tests and data preparation, `pip install -e '.[test]'` suffices. This repository uses the vLLM attention kernel directly; it does not start a model server.

## Math

```bash
remarl download backbone --output assets/llada
remarl download math-data --output assets/raw/math
remarl prepare math --raw-data assets/raw/math --model assets/llada --output assets/data/math

remarl train --config configs/math/nfe8.json \
  --model assets/llada --data assets/data/math --output runs/math8 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
```

Use `configs/math/nfe4.json`, `nfe8.json`, `nfe16.json`, `nfe32.json`, or `nfe64.json`. Omit `--devices` to use one GPU.

The mixture contains 7,473 GSM8K and 7,500 MATH training questions. Preparation uses pinned public data files, the original APPLe task prompts and answer extraction, and no prompt truncation. It checks for exact train/test question overlap. The manifest records file hashes, tokenized-row hashes, and split sizes.

## Code

Install and validate the [code execution sandbox](evaluation.md#code-execution) first.

```bash
remarl download code-data --output assets/raw/code
remarl prepare code --raw-data assets/raw/code --model assets/llada \
  --cohort code16 --output assets/data/code16

remarl train --config configs/code/nfe32.json \
  --model assets/llada --data assets/data/code16 --output runs/code32 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
```

`configs/code/nfe16.json` and `nfe32.json` use the `code16` cohort. For `configs/code/nfe8.json`, prepare `--cohort code8` in a separate directory. The saved allowlists retain 9,148 and 9,149 KodCode questions, respectively. Preparation verifies every retained row against its recorded hash. The allowlists include the reference-execution audit and benchmark-overlap exclusions used in the experiments.

HumanEval and MBPP prompts, test code, and sanitization are shared with evaluation. Generated code runs only after the sandbox's isolation and correctness probes pass.

## Recipe and outputs

The main configurations train from a zero-initialized score correction for 32,768 accepted groups. Each group contains eight answers to one question. Groups with constant correctness rewards are discarded. A batch contains 16 accepted groups, followed by four full-batch GRPO updates with the behavior policy fixed. AdamW uses a constant learning rate of `1e-4`, zero weight decay, clipping at `0.2`, and gradient norm clipping at `1.0`. The frozen backbone receives no updates.

| Output | Contents |
|---|---|
| `config.json` | Training settings |
| `initial.pt` | Initial policy and random state |
| `latest.pt` | Latest completed batch, including optimizer and random state |
| `policy-004096.pt`, … | Checkpoints every 4k groups and at the final group count |
| `training.jsonl` | Accuracy, optimization diagnostics, elapsed time, and policy hashes |
| `attempts.jsonl` | Question IDs, group rewards, acceptance, and revision counts |

Resume a run made by this trainer with the same configuration and prepared data:

```bash
remarl train --config configs/math/nfe8.json \
  --model assets/llada --data assets/data/math --output runs/math8 \
  --resume runs/math8/latest.pt
```

Released checkpoints load for inference and evaluation. Their original optimizer-state format is not the new trainer's resume format. A fresh run follows the saved recipe; evaluating the released weights reproduces the original checkpoint selection.

## Ablations

Component ablations use NFE8 and 16k accepted groups:

```bash
remarl train --config configs/ablations/no_hidden.json \
  --model assets/llada --data assets/data/math --output runs/no-hidden
```

Available configurations are `full`, `no_hidden`, `no_position`, `no_revision`, and `bias_only`. The no-revision variant also changes the action support and likelihood so that visible tokens remain visible.

For training-data size:

```bash
remarl train --config configs/data_size/n512-seed7.json \
  --model assets/llada --data assets/data/math --output runs/data512-seed7
```

The filename gives the total question count: 8, 64, 512, or 4,096, split equally between GSM8K and MATH. Within each seed, larger subsets contain the smaller subsets. Seeds 7 and 17 are provided. Subsets repeat when exhausted; the budget remains 16k accepted groups at NFE8. The full-data reference uses the main NFE8 run's 16k checkpoint.

Evaluation is explicit: pass any saved policy to `remarl evaluate`. Test scores do not control training duration or checkpoint saving.
