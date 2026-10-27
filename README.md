# ReMaRL

Research code for **ReMaRL: Learning to Unmask and Remask in Frozen Diffusion Language Models**.

ReMaRL trains a small position-selection policy while keeping LLaDA-8B-Instruct frozen. The policy uses token confidence, mask state, position, and hidden features to choose which positions to reveal, retain, or remask. Training uses GRPO with final-answer correctness rewards.

## Start here

- [Checkpoints and hashes](checkpoints/manifest.json): eight final 32k-group policies, with math NFE4/8/16/32/64 and code NFE8/16/32. Download `remarl-checkpoints-32k.zip` from the [private Drive folder](https://drive.google.com/drive/u/0/folders/1UFagh30vsvA-3jK-3sAf_iREaiQ3mxki) and extract it in this repository.
- [Running inference](docs/inference.md).
- [Training and evaluation instructions](docs/training.md).
- [Plot code and saved numbers](https://github.com/EgorShibaev/remarl-plots).

## Contents

| Path | Contents |
|---|---|
| `native/` | Production math trainer, frozen LLaDA port, rollout worker, selector, GRPO, and CPU tests |
| `training/code8/`, `training/code16/` | Original code-training sources, cohort filters, sandbox, and protocol snapshots |
| `training/code32-training/` | Ordered pipeline used to finish the NFE32 code run |
| `configs/` | Recipes read directly from the eight released checkpoints |
| `experiments/` | Data-size/component ablations and baseline reevaluation adapters |
| `provenance/` | Recorded dataset audits, source hashes, and training recipes |
| `environments/` | The production package inventory and the separate local QA environment |

This is a private research snapshot, not a PyPI package. The original training and evaluation runners retain their audit checks and cluster assumptions; the instructions identify what must be supplied on another machine. `generate.py` provides a small local-checkpoint inference entry point. Its rollout code is shared with training, but full GPU generation has not been rerun for this export.

Backbone weights and dataset Arrow files are not included. Use `GSAI-ML/LLaDA-8B-Instruct` at revision `08b83a6feb34df1a6011b80c3c00c7563e963b07`. Checkpoint files are stored on Drive rather than in Git. Access to that folder remains restricted separately from GitHub access.

No new license is assigned to project-owned code in this private export. Existing third-party notices and licenses still apply; see [THIRD_PARTY.md](THIRD_PARTY.md). Do not execute generated code outside an isolated sandbox.
