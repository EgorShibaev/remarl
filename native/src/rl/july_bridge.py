"""Bounded conditional follow-ups selected after the completed restoration study."""

CELLS = {
    "factorial-noexpert-linear1": (False, "pooled"),
    "factorial-noexpert-zero": (False, "pooled"),
    "bridge-linear01": (True, "pooled"),
    "bridge-linear1": (True, "pooled"),
    "bridge-power01": (True, "pooled"),
    "bridge-nine-policy": (False, "trajectory_sum"),
}
LABELS = {
    "factorial-noexpert-linear1": "No expert · Linear 1.0 · pooled",
    "factorial-noexpert-zero": "No expert · no revision reward · pooled",
    "bridge-linear01": "Expert + pooled · linear gamma0.1",
    "bridge-linear1": "Expert + pooled · linear gamma1",
    "bridge-power01": "Expert + pooled · power gamma0.1",
    "bridge-nine-policy": "Nine policy samples · no expert",
}
REWARDS = {
    "factorial-noexpert-linear1": (1., "linear"),
    "factorial-noexpert-zero": (0., "linear"),
    "bridge-linear01": (.1, "linear"),
    "bridge-linear1": (1., "linear"),
    "bridge-power01": (.1, "power"),
    "bridge-nine-policy": (0., "linear"),
}


def apply_config(config, cell):
    gamma, form = REWARDS[cell]
    config["grpo"].update(gamma=gamma, remask_form=form, alpha=0., reward_type="multiplicative",
                          group_size=8 if CELLS[cell][0] else 9,
                          revision_normalization="answer_length" if gamma else "none")
    config["tracking"]["group"] = "gsm-july-bridge-20260914"
    config["purpose"] = "Test the revision-reward and extra-policy-sample explanations for the July result"
    config["algorithm_revision"] = "shared-gsm32-july-bridge-v1"
    config["baseline_study"] = "20260914-restoration"
    config["baseline_cells"] = (["restore-control", "restore-expert"] if cell == "bridge-nine-policy"
                                else ["restore-both"])
    config["deviations"] = [
        "Three paired seeds7/13/42;32 GSM questions x16 passes;64 train-derived development questions",
        "Same shared(prompt,NFE) exposure plan at8/16/32/64; frozen Instruct;T1;beta.05;alpha0",
        "Reward cells:8 policy plus1 semi-AR confidence expert and pooled loss; only revision factor changes",
        "Linear factors max(0,1-gamma*D/256); power factor max(0,1-D/256)^.1",
        "Hard-zero rewards for correct heavily revised answers are an intentional historical-reward ablation",
        "Nine-policy control:9 on-policy samples, no expert, trajectory-sum/29 and correctness-only reward",
        "Nine-policy versus8+1 controls sample count, not identical batch-shape numerics or wall time",
        "Reuse completed matched restoration baselines, never retrain identical512-group controls",
        "Corrected verifiers, replay/dropout, LR, horizon features and resume guards retained",
    ]
    if cell.startswith("factorial-"):
        config["tracking"]["group"] = "gsm-expert-reward-20260915"
        config["purpose"] = "Complete expert on/off x Linear1.0/no reward, fixed pooled loss and nine total trajectories"
        config["algorithm_revision"] = "shared-gsm32-expert-reward-v1"
        config["baseline_study"] = "20260914-july-bridge and 20260914-restoration"
        config["baseline_cells"] = ["bridge-linear1", "restore-both"]
        config["deviations"] = [
            "32 identical GSM training questions,64 development questions,512 shared(prompt,NFE) groups,seeds7/13/42",
            "No expert:9 policy trajectories; completed expert-on controls have8 policy plus1 expert",
            "Pooled policy loss,T1,alpha0,beta.05; Linear1.0 or correctness-only reward",
            "All training/decoder/optimizer fixes retained; no longer-data training in this study",
        ]


def split_records(records, group_size):
    if group_size < 1 or not group_size <= len(records) <= group_size + 1:
        raise ValueError("Expected all policy records and at most one expert")
    return records[:group_size], records[group_size:]
