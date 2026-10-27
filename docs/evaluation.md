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

P2-prime scores visible positions using their retained token's probability. Original P2 uses the fresh proposal's probability and has a separate baseline configuration.

For P2-prime, `--temperature 0` selects greedy positions and `--temperature 0.25` applies the paper's calibrated temperature. `--retention-bonus VALUE` adds a fixed bonus to visible positions. Choose control settings on training questions before evaluating the test sets. These options leave token proposals greedy.

Each benchmark directory stores generated token IDs and scored answers in `answers-seed*.jsonl`, a summary per seed, and an aggregate `summary.json`. Accuracy is stored as a fraction and as `accuracy_percent`. Repeating the command resumes an incomplete evaluation. A different checkpoint, dataset, budget, seed list, or backend requires a different output directory.

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
