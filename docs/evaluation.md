# Evaluation

## ReMaRL and P2-prime

Prepare data using the commands in [training](training.md). Then evaluate a checkpoint:

```bash
remarl evaluate --policy checkpoints/math8-32k.pt --nfe 8 \
  --model assets/llada --data assets/data/math --output runs/math8-eval

remarl evaluate --policy checkpoints/code32-32k.pt --nfe 64 \
  --model assets/llada --data assets/data/code16 --output runs/code32-at64
```

Evaluation covers both full benchmarks for the domain and averages three inference seeds. Math uses 1,319 GSM8K questions and 500 MATH500 questions; code uses 164 HumanEval and 500 MBPP tasks. `--split test_gsm8k`, `test_math`, `test_humaneval`, or `test_mbpp` selects one benchmark.

The untrained current-token control uses the same rollout and scoring pipeline:

```bash
remarl evaluate --p2-prime --nfe 8 --model assets/llada \
  --data assets/data/math --output runs/p2-prime8
```

P2-prime scores unmasked positions using their retained token's probability. Original P2 uses the fresh proposal's probability and has a separate baseline configuration.

`--temperature` controls position selection for both trained ReMaRL policies and P2-prime. It defaults to 1; 0 selects positions greedily. For P2-prime, training-question calibration at NFE8 selects 0.25 separately for math and code; the comparison below reuses it at every budget. `--retention-bonus VALUE` applies only to P2-prime and adds a fixed bonus to unmasked positions. Choose control settings on training questions before evaluating the test sets. These options leave token proposals greedy.

To reproduce the trained NFE8 policy's temperature sweep with fixed weights:

```bash
for tau in 0 0.1 0.25 0.5 0.75 0.85 0.9 0.95 1 1.05 1.1 1.15 1.25 1.5 2; do
  seeds=101,202,303
  if [ "$tau" = 0 ]; then seeds=101; fi
  remarl evaluate --policy checkpoints/math8-32k.pt --nfe 8 \
    --temperature "$tau" --seeds "$seeds" \
    --model assets/llada --data assets/data/math \
    --output "runs/math8-temperature/tau-$tau"
done
```

Positive temperatures use three inference seeds; the deterministic setting uses one. Evaluation records `position_temperature` alongside the checkpoint hash. Saved trained-policy evaluations from older releases, which omitted this field, are treated as temperature 1.

Each benchmark directory stores generated token IDs and scored answers in `answers-seed*.jsonl`, a summary per seed, and an aggregate `summary.json`. Accuracy is stored as a fraction and as `accuracy_percent`. Repeating the command resumes an incomplete evaluation. A different checkpoint, dataset, budget, position temperature, retention bonus, seed list, or backend requires a different output directory.

Dream uses the same evaluation command with its own model, tokenizer, prepared data, and `--backend torch`; see [Dream evaluation](dream.md#evaluate).

## Calibrated P2-prime budget sweep

The completed LLaDA sweep holds the position temperature at 0.25 and varies the number of model calls. Calibration uses 256 training questions at eight calls: 128 from GSM8K and 128 from MATH for math, and 256 from KodCode for code. It does not select a new temperature at each test budget.

After preparing both domains as described in [training](training.md), run:

```bash
set -e
for nfe in 4 8 16 32 64 128 256; do
  remarl evaluate --p2-prime --nfe "$nfe" --temperature 0.25 \
    --seeds 101,202,303 --model assets/llada --data assets/data/math \
    --output "runs/p2-prime-temperature-025/math-nfe${nfe}"
done

for nfe in 8 16 32 64 128 256; do
  remarl evaluate --p2-prime --nfe "$nfe" --temperature 0.25 \
    --seeds 101,202,303 --model assets/llada --data assets/data/code16 \
    --output "runs/p2-prime-temperature-025/code-nfe${nfe}"
done
```

Each command evaluates both full benchmarks for its domain with 256-token answers and greedy token proposals. No policy checkpoint or training is required. Existing complete answers are reused when the same command resumes.

[Recorded results](../results/p2-prime-temperature-025/README.md) contain all 26 means and 78 per-seed summaries. The original evaluation reused matching NFE8/16 results and the GSM8K NFE32 trajectory evaluations; their source and answer hashes are retained. The commands above generate the same evaluation grid with the release interface. They were checked against its argument parser; the full GPU sweep has not been rerun through this interface.

## Baselines

Download the pinned upstream source files once:

```bash
remarl download baseline-sources --output assets/baselines
```

Run a configuration from `configs/baselines/` against either prepared domain:

```bash
remarl baseline --config configs/baselines/low_confidence-bl256-steps8-s101.json \
  --model assets/llada --data assets/data/math --output runs/confidence8

remarl baseline --config configs/baselines/pc_sampler-bl256-steps32-s101.json \
  --model assets/llada --data assets/data/code16 --output runs/pc-sampler32-code
```

Configurations cover confidence, random, margin, Fast-dLLM, EB-Sampler, original P2, UNCODE/PC-Sampler, KLASS, WINO, ReMDM's UNCODE adaptation, RCR, CoRe, and DAPD. Random unmasking has configurations for three inference seeds. Inspect the JSON for block length, thresholds, and call budgets.

Adaptive decoders record actual backbone calls per answer and their benchmark mean. ReMDM counts both calls in each revision loop; CoRe includes its extra context checks. Compare adaptive scores at their measured mean NFE. Missing results are not interpolated.

Outputs contain `answers.jsonl` and `summary.json` per benchmark. Each configuration runs the complete test set. Completed answers are reused when the same command is restarted.

APPLe policy, UPO, and SAS values in the recorded paper results are borrowed from their publications. Their source labels are preserved; this release does not provide those authors' trained policy weights. UPO's shorter-answer results and SAS's published protocols remain separate references.

## Code execution

Code scoring requires Linux x86_64, bubblewrap, libseccomp, and permission to create user/PID namespaces. On Debian/Ubuntu:

```bash
sudo apt-get install bubblewrap libseccomp2
remarl check-sandbox
```

The check builds a kernel syscall filter and tests correctness, timeouts, memory limits, read-only mounts, and network denial. Evaluation and training run these checks before executing generated programs and stop if they fail. There is no unsandboxed fallback.

Each candidate runs in a separate process namespace with a private scratch directory, a three-second execution limit, and a 1 GiB memory limit. Home directories and project files are not mounted. Socket and io_uring syscalls are denied by seccomp. `REMARL_SANDBOX_DIR` can set the cache/scratch directory; it defaults to `~/.cache/remarl/sandbox`.

## Recorded results

`results/benchmark-observations.csv` contains the saved observations with method, block geometry, measured NFE, score, and source. `data-receipt.json` retains detailed table inputs and original receipt hashes. Historical receipt paths identify the original experiments; commands do not depend on those paths.

Source labels are `C` for computed results, `A` for APPLe, `U` for UPO, and `S` for SAS. The paper displays computed results without a superscript.
