# Inference

Download the backbone and policies as described in the README, then run:

```bash
remarl generate --model assets/llada --policy checkpoints/math8-32k.pt \
  --nfe 8 --task gsm8k \
  --prompt 'Mia has 12 apples and gives away 5. How many apples remain?'
```

The command prints the answer, token IDs, inference budget, seed, and policy hash. It uses the same decoding loop as training and benchmark evaluation.

Use `--task math` for the MATH prompt template. `--task raw` accepts an already formatted prompt, including any chat template and generation prefix. Prepare benchmark code prompts through `remarl prepare code` to preserve HumanEval and MBPP formatting.

`--device cuda:0` and `--seed 101` are the defaults. Add `--backend torch` to use PyTorch SDPA. Prompts are never truncated.

The policy's training budget and inference budget can differ:

```bash
remarl generate --model assets/llada --policy checkpoints/math64-32k.pt \
  --nfe 128 --task gsm8k --prompt 'What is 24 multiplied by 7?'
```

All released policies use 256 answer positions. Token proposals are greedy; position choices use stochastic ordered sampling. The final call fills the remaining masks.
