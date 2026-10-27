# Attribution

- `src/remarl/backbone.py` retains the vLLM Apache-2.0 copyright and SPDX headers. The license is included as `third_party/VLLM-LICENSE`.
- `src/remarl/protocols/official/` contains APPLe prompt, parser, generation, sanitization, and execution functions. The license is in `third_party/APPLE-LICENSE`; original notices remain in source files.
- LLaDA-8B-Instruct weights are downloaded separately at the revision in `checkpoints/manifest.json`. The model's original license applies.
- `src/remarl/baselines/upstream.json` records upstream repositories, immutable commits, and file hashes. `remarl download baseline-sources` fetches those files and available license notices into `assets/baselines/`. Upstream code retains its authors' terms.
- Dataset versions are pinned in `src/remarl/resources/datasets.json`. Their authors' licensing and attribution apply.

No new license has been assigned to project-owned code in this repository. Included third-party code and model assets are not relicensed by this release.
