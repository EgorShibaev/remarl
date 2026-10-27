# Attribution

- `native/model.py` and its archived copies are the LLaDA native port. They retain the vLLM Apache-2.0 and copyright headers.
- APPLe prompt, parser, and evaluation source snapshots remain under `official/` and `protocol/official/`. Their license is copied to `third_party/APPLE-LICENSE`. The source commits are recorded in the adjacent data audits and manifests.
- LLaDA-8B-Instruct weights are obtained separately from the named Hugging Face revision; the original model's license applies.
- Baseline adapters in `experiments/` refer to upstream implementations. `UPSTREAM.json` files record those sources where available. They are not bundled as independently licensed replacements for the upstream projects.

Historical source files may use the name P2 for the current-token control. The paper calls that control P2-prime; original P2 scores visible positions with fresh proposals. Internal family names and old comments have been preserved to keep the source snapshot auditable.
