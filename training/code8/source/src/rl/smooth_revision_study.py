"""Restore shaped-reward GRPO without the old zero cutoff or raw-NFE scaling."""
def configure_smooth(config,gamma):
    if gamma not in (0.,.1,.3,1.,2.,5.):raise ValueError('Gamma outside predeclared sweep')
    config['grpo'].update(gamma=gamma,remask_form='exp_opportunity',
       separated_revision_weight=0.,separated_revision_allow_expert=False,
       opportunity_reward_allow_expert=True)
    config['reward_variant']=f'smooth{gamma:g}-h512'
    config['experiment_label']=f'Opportunity exponential {gamma:g} · 512 groups'
    config['tracking']['group']=f'gsm-gamma-sweep-20260917/gamma-{gamma:g}'
    config['algorithm_revision']='expert-pooled-exp-opportunity-v1'
    config['purpose']='Correctness times exp(-gamma R/B); restore joint reward standardization without hard-zero cutoff'
    config['deviations']=[
       'Exact32GSM questions;512groups;expert+pooled;T1;alpha0;beta.05',
       'Reward correctness*exp(-gamma R/B), B=linear-schedule revision opportunities',
       'All9 rewards jointly standardized, including off-policy expert; not separated advantages',
       'Gamma0,.1,.3,1,2,5 predeclared;no long-horizon expansion before development review',
       'All-wrong groups skip;all-correct unequal shaped rewards may update',
    ]
