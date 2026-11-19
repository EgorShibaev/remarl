# G-Star on GSM8K and MATH500

This directory reproduces the G-Star rows in Appendix A of ReMaRL. We adapt the refinement loop from [Guided Star-Shaped Masked Diffusion](https://proceedings.mlr.press/v306/meshchaninov26a.html) to frozen LLaDA-8B-Instruct using the repository's 256-token math prompts and scorers.

## Evaluate the fitted head

From the repository root, install ReMaRL and follow the math data preparation instructions in the [main README](../../README.md). Run the following on one machine with four CUDA GPUs. The experiment used four H100 GPUs, PyTorch 2.7.1 and Transformers 4.46.3. Use a fresh output path for each invocation.

```bash
torchrun --standalone --nproc-per-node=4 examples/gstar/evaluate.py \
  --model assets/llada --data assets/data/math --output runs/gstar-eval
```

The script loads the included `head.pt`. At each budget (4, 8, 16, 32, 64, 128 and 256 model calls), greedy confidence decoding drafts a full answer. Each refinement loop uses one call to score all positions, remasks the 15 positions with the highest predicted error scores, and uses one call to refill them greedily.

We consider spending 10%, 25% or 50% of the budget on refinement, rounding to whole two-call loops with a minimum of one. We choose the loop count by final-answer accuracy on 128 held-out GSM8K training questions, breaking ties toward fewer loops. We run that schedule on all 1,319 GSM8K and 500 MATH500 test questions. The output contains question-level scores and a `results.json` summary. [Paper results](paper-results.json) contains the selected loop counts, correct-answer counts and archived confidence scores.

## Refit the head

```bash
torchrun --standalone --nproc-per-node=4 examples/gstar/fit.py \
  --model assets/llada --data assets/data/math --raw-data assets/raw/math \
  --output runs/gstar-fit
```

For each of 14,333 training questions we generate eight independently corrupted reference drafts. Each visible reference-answer position contributes a final-layer hidden state and a binary target indicating whether its token differs from the reference. We fit ridge logistic regression (coefficient 0.001) on RMS-normalized features. Allow at least 125 GB of disk space for the cached half-precision features. Pass `--head runs/gstar-fit/q14333-x8-l0.001.pt` to the evaluation command to use the refitted head.

To repeat head selection, add `--sweep` to the fitting command. It evaluates 18 fits on the same 512 held-out questions with four corruption draws each, reporting token accuracy, error average precision and question-mean log loss in `validation.json`. The selected configuration achieved 93.82% accuracy, 79.17% average precision and 0.1672 log loss.
