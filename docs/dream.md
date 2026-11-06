# Dream

The release includes six ReMaRL policies trained on frozen Dream-v0-Instruct-7B: math and code at NFE8, NFE16, and NFE32. Each uses 32,768 accepted training groups. These are separate Dream training runs, not LLaDA weights transferred to Dream.

## Environment and downloads

Use a separate environment from the LLaDA vLLM setup. The Dream experiments used PyTorch 2.7.1, Transformers 4.46.3, Accelerate 1.6.0, BF16 weights, and PyTorch SDPA.

```bash
python3.12 -m venv .venv-dream
source .venv-dream/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e '.[dream,test]'
remarl download dream-backbone --output assets/dream
remarl download checkpoints --output checkpoints
```

The model revision is pinned in the checkpoint manifest. Dream has 3,584 hidden features and uses mask token 151666. Its prediction for answer position i uses the preceding decoder output; the policy receives the hidden state aligned with that prediction. The adapter applies this shift to both logits and hidden states. No extra diffusion-model calls are made for the policy.

## Prepare data and train

The task prompts, answer extraction, and code scoring reuse APPLe's setup, with Dream's tokenizer. Prepare Dream data separately from LLaDA data:

```bash
remarl download math-data --output assets/raw/math
remarl prepare math --raw-data assets/raw/math --model assets/dream --output assets/data/dream-math
remarl train --config configs/dream/math/nfe8.json \
  --model assets/dream --data assets/data/dream-math --backend torch \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 --output runs/dream-math8
```

Use the corresponding NFE16 or NFE32 configuration for the other math policies. The mixture contains 14,973 questions. The policy hidden projection changes to accept Dream's 3,584 features; the position features and GRPO recipe are unchanged.

For code, first install the [execution sandbox](evaluation.md#code-execution). Prepare `--cohort code8` for NFE8 and `--cohort code16` for NFE16/32, as in the LLaDA experiments:

```bash
remarl download code-data --output assets/raw/code
remarl prepare code --raw-data assets/raw/code --model assets/dream \
  --cohort code16 --output assets/data/dream-code16
remarl train --config configs/dream/code/nfe16.json \
  --model assets/dream --data assets/data/dream-code16 --backend torch \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 --output runs/dream-code16
```

## Evaluate

```bash
remarl evaluate --policy checkpoints/dream-math8-32k.pt --nfe 8 \
  --model assets/dream --data assets/data/dream-math --backend torch \
  --output runs/dream-math8-eval
remarl evaluate --policy checkpoints/dream-code16-32k.pt --nfe 16 \
  --model assets/dream --data assets/data/dream-code16 --backend torch \
  --output runs/dream-code16-eval
remarl evaluate --p2-prime --nfe 8 --model assets/dream \
  --data assets/data/dream-math --backend torch --output runs/dream-p2-prime8
```

The same evaluation options support greedy P2-prime (`--temperature 0 --seeds 101`) and position-temperature sweeps. Model and prepared-data identities are checked before evaluation.

## Baselines

```bash
remarl download baseline-sources --output assets/baselines
remarl baseline --config configs/dream/baselines/confidence-bl32-nfe8.json \
  --model assets/dream --data assets/data/dream-math --backend torch \
  --output runs/dream-confidence8
remarl baseline --config configs/dream/baselines/klass-math-bl32-p1.0.json \
  --model assets/dream --data assets/data/dream-math --backend torch \
  --output runs/dream-klass
```

Configurations include the observed operating points for Fast-dLLM, EB-Sampler, KLASS, and both DAPD variants. Fixed confidence decoding uses the same block schedule as the historical evaluation. Adaptive methods use the native Dream implementations and record actual model calls. The observed count is checked against each decoder's reported count, and the prompt and complete answer length are checked before scoring.

Dream KLASS uses the block-aware implementation bundled with DAPD at the pinned commit, as in the recorded experiments. The original KLASS Dream file does not implement the block-length option. Both the Dream KLASS and EB source files are verified against the historical experiment copies.

The archived observations in `results/dream/` are the completed experiments. The package adapters preserve their settings, but full GPU benchmarks have not been rerun after packaging; see [verification](verification.md).
