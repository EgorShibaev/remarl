"""One new recipe: correctness GRPO plus bounded revision advantage."""
def configure(config, horizon):
    if horizon not in (512,2048):
        raise ValueError("Only predeclared 512/2048 schedules")
    config["grpo"].update(iterations=horizon,alpha=0.,gamma=0.,gamma_repeat=0.,
        remask_form="linear",separated_revision_weight=.05,separated_revision_allow_expert=True,
        revision_normalization="linear_schedule_opportunity_bound",reward_type="multiplicative")
    config["env"].update(training=32,attempted_questions=horizon)
    config["reward_variant"]=f"bounded005-h{horizon}"
    config["experiment_label"]=f"Bounded revision 0.05 · {horizon}-group schedule"
    config["tracking"].update(group="gsm-bounded-revision-20260916",window_questions=32)
    config["development_attempted_questions"]=[0,512] if horizon==512 else [0,512,2048]
    config["algorithm_revision"]="expert-pooled-bounded-opportunity005-v1"
    config["purpose"]="Replace hard-zero revision reward, preserving expert+pooled recipe"
    config["deviations"]=[
        "32 exact GSM training questions;64 disjoint development;fixed961 test subset",
        "8 policy+1 same-budget semi-AR expert;shared prompt/NFE;pooled loss;T1;beta.05",
        "Correctness-only scalar reward;separate revision advantage weight.05;no restandardization",
        "Correct answers centered among correct trajectories including expert;all-wrong signal zero",
        "Separate512/2048 cosine schedules;retain long-run512 checkpoint",
        "All-correct groups may have weak revision-only signal;skip only zero combined signal",
    ]
