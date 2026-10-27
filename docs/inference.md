# Inference

Use a GPU environment with PyTorch, NumPy, Transformers, safetensors, and the native port's dependencies. The original CUDA package inventory is in `environments/production-packages.txt`. It is a record of the research environment, not a cross-platform lockfile. Inference defaults to the port's PyTorch attention backend; `--backend vllm` selects the production backend when a compatible vLLM build is installed.

Download the frozen backbone separately and extract the policy archive into `checkpoints/`. Verify it first:

```sh
python verify_checkpoints.py checkpoints
python generate.py --model /absolute/path/to/LLaDA-8B-Instruct \
  --policy checkpoints/math8-32k.pt --nfe 8 --task gsm8k \
  --prompt 'A bag has 3 red balls and 4 blue balls. How many balls are in the bag?'
```

The prompt argument is a raw question for `gsm8k` or `math`; the bundled APPLe template is applied. Use `--task raw` only for an already formatted prompt. `raw` is useful for code generation but is not a benchmark evaluation harness. Outputs contain the answer token IDs, decoded answer, seed, and NFE. Generated code is printed, never executed.

This entry point uses the original current-token scoring, greedy token proposals, linear visibility schedule, and stochastic ordered top-k selection at temperature 1. It uses a caller seed for the single example. For published benchmark results, use the original evaluation harness and its per-question seeds; a new prompt or seed does not reproduce a table entry.

The backbone and answer canvas need enough GPU memory. The checkpoint's position embeddings require 256 answer positions. Full GPU parity for the new command-line wrapper was not run during packaging; the underlying rollout tests and checkpoint loading are checked on CPU.
