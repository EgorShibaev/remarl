# Attribution

- `src/remarl/backbone.py` retains the vLLM Apache-2.0 copyright and SPDX headers. The license is included as `third_party/VLLM-LICENSE`.
- `src/remarl/protocols/official/` contains APPLe prompt, parser, generation, sanitization, and execution functions. The license is in `third_party/APPLE-LICENSE`; original notices remain in source files.
- `src/remarl/apple_policy_layers.py` is the unchanged RoPE DiT implementation from [apple/ml-rl-dllm](https://github.com/apple/ml-rl-dllm), commit `7db77973abb3b9f5bb38e7ab889242f629f3ee8d`, `common/models/policy_layers.py` (SHA256 `2bd2ef882f3ae5ba53a818fd7024ca01e2f68c3f436ebd62f9081f55965890dc`). The APPLe license above applies. The ReMaRL transformer wrapper preserves ReMaRL inputs and training.
- Dream-v0-Instruct-7B weights and model code are downloaded separately from [Dream-org](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) at the revision in the checkpoint manifest. Their original license applies.
- LLaDA-8B-Instruct weights are downloaded separately at the revision in `checkpoints/manifest.json`. The model's original license applies.
- `src/remarl/baselines/upstream.json` records upstream repositories, immutable commits, and file hashes. `remarl download baseline-sources` fetches those files and available license notices into `assets/baselines/`. Upstream code retains its authors' terms.
- Dataset versions are pinned in `src/remarl/resources/datasets.json`. Their authors' licensing and attribution apply.

No new license has been assigned to project-owned code in this repository. Included third-party code and model assets are not relicensed by this release.
