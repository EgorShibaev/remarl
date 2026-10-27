# Release verification

Checks run locally on 27 October 2026:

- Public checkpoint download without Google cookies or a login; archive and all eight file hashes verified.
- Strict loading of all eight policies. Outputs and seeded initializations match the original implementation exactly on CPU.
- Downloaded math rows match the original files. Both code cohorts verify retained rows against saved hashes.
- Tokenized prompts and scoring inputs match the original evaluator on all four complete test sets.
- GRPO tests cover grouped rewards, fixed behavior probabilities, replay likelihoods, gradient accumulation, and optimizer state.
- A small CPU generator exercises training, saving, resuming, and evaluation. Resuming produces the same next policy state; evaluation resumes partial batches with their original padding.
- Decoder-reference tests compare trajectories and call counts with pinned PC-Sampler, EB-Sampler, KLASS, WINO, and ReMDM implementations. RCR, CoRe, and DAPD tests exercise official decoders and budget accounting.

Full GPU parity after reorganization remains unverified. Released policies and recorded paper scores are unchanged. A complete benchmark run with this package is the next validation step; no GPU training was launched for the cleanup.

The Linux sandbox was retained from the validated experiment implementation, with installation paths made portable. Its probes must pass on the machine used for code evaluation. They cannot run on macOS.

Run `python -m pytest -q`. Upstream-reference tests are skipped until `remarl download baseline-sources --output assets/baselines` has been run. They use small CPU models and do not download backbone weights.
