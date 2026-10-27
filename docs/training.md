# Training and evaluation

The exact production sources are included. Start by reading `configs/math8-32k.json` and `native/train_batch_grpo.py`; defaults in legacy classes are not the paper recipe. Main policies use 32k accepted groups. Component and data-size ablations use 16k.

## Math

The production trainer accepts explicit run, asset, and GPU paths:

```sh
PYTHONPATH=native python native/train_batch_grpo.py validate \
  --root /absolute/path/to/prepared-run --assets /absolute/path/to/assets --gpus 0,1,2,3
PYTHONPATH=native python native/train_batch_grpo.py train \
  --root /absolute/path/to/prepared-run --assets /absolute/path/to/assets --gpus 0,1,2,3
```

Use a fresh run directory, copy the contents of `native/` there, and copy the chosen `configs/math*-32k.json` to its `recipe.json`. The run must contain `inputs.pt`, `rows.json`, `data-audit.json`, `source.sha256`, and the `official/` scorers. `prepare_paper.py` builds the input tensors and data audit from the four source Arrow files named in that script. It uses the pinned tokenizer and APPLe prompts without truncating prompts.

The asset directory contains `sota-selector.pt` from the Drive archive, `checkpoint_manifest.json` identifying the backbone files, and `local_root.txt`, whose contents are the local directory containing `models/LLaDA-8B-Instruct`. The seed checkpoint supplies architecture settings for fresh training. The recipe sets `initialization: fresh`; it does not initialize from that checkpoint's trained weights.

The math mixture is GSM8K train plus MATH train. Use the revisions and raw-file hashes in `provenance/*/data-audit.json`. Source manifests are checksums of the copied run files. Recompute manifests for a deliberately relocated run; do not replace historical receipts or disable input checks to force a run through. `--resume` expects the original run state and matching source/data receipts.

The validation stage runs real GPU checks. No training or full GPU validation was launched while creating this repository. For a different CUDA/vLLM installation, run validation before starting a long job.

## Code

`training/code16/train_code.py` contains the production code-training loop. Its `source/` directory contains the corresponding selector, optimizer, and rollout implementation. `protocol/` contains the prompt preparation, sanitization, and isolated execution code; `cohort/` contains retained/excluded source IDs and the benchmark-overlap quarantine. NFE8 has its own recorded source snapshot because its historical cohort differs slightly.

These runners preserve hard-coded job IDs, validation receipts, and launch paths from the original cluster. They are inspectable production code, not portable launch commands. To run on a new allocation, relocate the `BASE`/runtime paths, select the desired GPU IDs and NFE recipe, and perform the sandbox/runtime validation sequence before training. Do not remove the sandbox security checks. `training/code32-training/` preserves the later ordered pipeline used to complete NFE32.

Code rewards execute model-generated programs. Run them in an isolated Linux environment with the supplied resource limits, process isolation, and validated seccomp sandbox. Do not run them on a personal machine with access to credentials or private files.

## Baselines and ablations

`experiments/` contains the scripts used for component removal, training-set size, and remaining baseline evaluations. These also retain original workspace paths. Reuse the recorded input IDs, source revisions, answer length, and measured-NFE conventions when relocating them. APPLe/UPO/SAS published scores are references, not checkpoints trained or reproduced by this release.

The separate `remarl-plots` repository contains all reported numbers and provenance. Rendering those results does not require running these experiments again.

## CPU checks

```sh
python -m pip install -r requirements-test.txt
PYTHONPATH=native python -m pytest native/test_worker.py native/test_batch_grpo.py native/test_evaluate_p2.py -q
python verify_checkpoints.py checkpoints
```

The CPU tests use fake backbones. They check ordered probabilities, transitions, rollout/replay agreement, and training utilities without downloading LLaDA or executing generated code.
