# Release verification

Checks added on 6 November 2026:

- The public Dream archive downloads without a Google account or cookies. Its archive hash, six checkpoint hashes, and six policy-state hashes match the manifest; all six policies load strictly.
- Dream fixtures check the native one-position shift of logits and hidden states, mask IDs, model/data identity checks, and policy dimensions.
- Confidence decoding matches the pinned APPLe Dream loop at NFE8/16/32 for both full and blockwise decoding. Fast-dLLM, EB-Sampler, and KLASS fixtures check completion, block boundaries, and exact model-call counts. Dream KLASS and EB sources match the historical experiment files.
- The transformer checkpoint loads strictly with 583,187 parameters. Tests cover zero initialization, within-answer attention, separation between answers, gradients, and rollout/replay likelihood agreement.
- The complete CPU suite passes: 100 tests, including the fetched upstream-reference tests. Full GPU benchmarks have not been rerun after packaging.

Checks added on 2 November 2026:

- All 12 data-size recipes cover seeds 7, 17, and 27 and agree with the recorded training costs.
- The public evaluation command supports trained-policy position temperatures, including greedy selection. Tests compare rollout and replay likelihoods, reject invalid temperatures, and prevent resuming into results from another temperature or confidence-residual rule.
- Loading or evaluating a second policy leaves the first policy's ablation and temperature settings unchanged.
- The initialization ablation trains, saves, loads, and resumes through the public commands with a small CPU generator. Resumed weights match uninterrupted training exactly. Its random output heads and absence of a fixed confidence residual are checked explicitly.
- Fifteen CPU parity checks against the pre-fix release and the historical experiment source passed. Default seeded weights and scores are unchanged for all five component variants; random-initialization weights and scores match the historical implementation; isolated ablation sampling and likelihoods agree at temperatures 0, 0.25, 1, and 2.

These checks use CPU fixtures; they do not rerun the full GPU experiments.

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
