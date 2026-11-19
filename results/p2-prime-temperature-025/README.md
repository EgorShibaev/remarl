# P2-prime at position temperature 0.25

Completed on 16 November 2026 with frozen LLaDA-8B-Instruct, 256-token answers, full diffusion, and greedy token proposals. The position temperature is fixed across inference budgets. It was selected separately for math and code on 256 training questions at eight calls.

Scores are percentages: accuracy for GSM8K and MATH500, pass@1 for HumanEval and MBPP. Each cell is the mean over inference seeds 101, 202, and 303. A dash denotes a budget not evaluated in this sweep.

| Calls | GSM8K | MATH500 | HumanEval | MBPP |
|---:|---:|---:|---:|---:|
| 4 | 27.29 | 13.73 | — | — |
| 8 | 42.30 | 17.87 | 14.23 | 21.47 |
| 16 | 56.25 | 20.93 | 22.56 | 29.33 |
| 32 | 63.79 | 22.60 | 24.19 | 33.67 |
| 64 | 66.87 | 24.67 | 23.17 | 33.73 |
| 128 | 69.12 | 26.27 | 25.20 | 37.00 |
| 256 | 69.42 | 26.07 | 22.97 | 35.73 |

## Files

- `summary.csv` and `summary.json`: all 26 benchmark–budget means and seed ranges.
- `seed-results.json`: all 78 correct-answer counts, dataset hashes, answer-file hashes, and reuse receipts.
- `provenance.json`: source hashes and calibration settings.
- `verify.py`: checks coverage, aggregation, hashes, and agreement with the repository-wide result files.

```bash
python results/p2-prime-temperature-025/verify.py
```

The original run verified every answer-file hash, benchmark count, and reward total. Those complete generated-answer files remain in the experiment archive; this directory publishes their hashes and per-seed summaries. `verify.py` checks the published files and does not fetch or recheck the archived answers.

Matching earlier math/code NFE8/16 evaluations and GSM8K NFE32 trajectory evaluations were reused. Their source identifiers and original hashes are preserved. Existing temperature-1 and greedy results are unchanged.

[Evaluation commands](../../docs/evaluation.md#calibrated-p2-prime-budget-sweep) reproduce the budget grid with the released evaluator.
