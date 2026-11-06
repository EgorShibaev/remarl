# Dream benchmark results

All values are computed in this work on frozen Dream-v0-Instruct-7B, using the APPLe task prompts and scorers. ReMaRL policies are trained separately for math/code and NFE8/16/32, for 32,768 accepted groups. There is no NFE4 or NFE64 ReMaRL training result in these tables.

`observations.json` contains completed operating points. `selected-cells.json` records the observations used in the four LaTeX benchmark tables. `verification.json` records table consistency checks. Historical source paths are relative identifiers; evaluation does not depend on these paths.

Fixed-budget rows use the stated NFE. Adaptive rows select the highest observed test score whose measured mean NFE does not exceed the column budget; ties favor fewer calls. Actual mean NFE is displayed below each adaptive score. These are observed score–budget frontiers, not scores from a decoder forced to use exactly the column budget. No points are interpolated.

ReMaRL has the highest reported score in these comparisons at 8, 16, and 32 calls on math, and at 8 and 16 calls on code. The MBPP comparison at 32 calls does not support the same claim. These experiments demonstrate applicability to another frozen model; the Dream policies were not transferred from LLaDA.

Reproduction commands and operating-point configurations are in `docs/dream.md` and `configs/dream/baselines/`.
