"""Small, paired restoration study: shared (question, NFE) groups only."""

from collections import defaultdict

import torch

CELLS = {
    "restore-control": (False, "trajectory_sum"),
    "restore-expert": (True, "trajectory_sum"),
    "restore-pooled": (False, "pooled"),
    "restore-both": (True, "pooled"),
}
LABELS = {
    "restore-control": "Control · trajectory sum",
    "restore-expert": "Expert · trajectory sum",
    "restore-pooled": "No expert · old pooled loss",
    "restore-both": "Expert · old pooled loss",
}
BUDGETS = (8, 16, 32, 64)
EVALUATIONS = (0, 256, 512)


def make_restoration_plan(seed, *, size=32, passes=16, budgets=BUDGETS, group=8,
                          expert=False):
    if size % len(budgets) or passes % len(budgets):
        raise ValueError("Require balanced budgets per pass and per question")
    question_rng = torch.Generator().manual_seed(seed + 100003)
    budget_rng = torch.Generator().manual_seed(seed + 200003)
    offsets = torch.arange(size) % len(budgets)
    offsets = offsets[torch.randperm(size, generator=budget_rng)]
    plan = []
    for epoch in range(passes):
        for q in torch.randperm(size, generator=question_rng).tolist():
            s = budgets[(int(offsets[q]) + epoch) % len(budgets)]
            row = {"problem_idx": q, "steps": [s] * group}
            if expert:
                row["expert_steps"] = [s]
            plan.append(row)
    return plan


def evaluation_metrics(rows, update):
    groups = defaultdict(list)
    for row in rows:
        groups[row["split"], row["steps"], row["method"]].append(row)
    result = {"progress/attempted_questions": update}
    for (split, steps, method), values in groups.items():
        prefix = f"restoration/{split}/nfe{steps}/{method}/"
        result[prefix + "accuracy"] = sum(r["correctness"] for r in values) / len(values)
        result[prefix + "revisions"] = sum(r["revisions"] for r in values) / len(values)
        result[prefix + "questions"] = len(values)
    return result
