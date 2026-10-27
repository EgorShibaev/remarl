"""Predeclared one-knob ablations around the pointwise pure-RL MLP."""
from src.decoding.planner_features import HORIZON_SPEC

TOTAL = 2048
EVALUATIONS = (0, 128, 256, 512, 1024, 2048)
VARIANTS = ('base', 'low-lr', 'high-lr', 'wide', 'time', 'skip', 'one-update')

def configure(config, variant, hardware):
    if variant not in VARIANTS or hardware not in ('A100', 'H100'):
        raise ValueError('Unknown overnight variant or hardware')
    config['model'].update(architecture='mlp', hidden_dim=64, linear_skip=False,
                          planner_features=['score','is_masked','pos'])
    config['grpo'].update(iterations=TOTAL,lr=3e-4,ppo_epochs=4,group_size=8,
                         alpha=0.,gamma=0.,beta=0.,weight_decay=0.,
                         expert_steering=False,n_experts=0)
    if variant == 'low-lr': config['grpo']['lr'] = 1e-4
    if variant == 'high-lr': config['grpo']['lr'] = 1e-3
    if variant == 'wide': config['model']['hidden_dim'] = 128
    if variant == 'time': config['model']['planner_features'] = list(HORIZON_SPEC)
    if variant == 'skip': config['model']['linear_skip'] = True
    if variant == 'one-update': config['grpo']['ppo_epochs'] = 1
    config['development_attempted_questions'] = list(EVALUATIONS)
    config['env']['attempted_questions'] = TOTAL
    config['tracking']['group'] = f'gsm32-mlp2048-{hardware}-{variant}-20260918'
    config['tracking']['study_id'] = f'gsm32-mlp2048-{hardware}-20260918'
    config['reward_variant'] = variant  # Tracking condition ID; reward stays binary.
    config['experiment_label'] = f'MLP2048 · {variant} · {hardware}'
    config['algorithm_revision'] = 'mlp-overnight-2048-v1'
    config['overnight_variant'] = variant
    config['hardware_family'] = hardware
    config['deviations'] = [
        'Fresh initialization; 32 GSM8K questions,64 passes,2048 attempted groups; fixed32NFE,G8',
        'Accuracy only; no expert,revision reward,KL or weight decay',
        f'MLP {variant}; inputs={config["model"]["planner_features"]}; hidden={config["model"]["hidden_dim"]}',
        f'LR={config["grpo"]["lr"]}; warmup32; cosine horizon2048 (not continuation of512 schedule)',
        f'{config["grpo"]["ppo_epochs"]} PPO epochs; pooled; T1; ratio clip.2; gradient clip.2; skip zero variance',
        'Greedy/sample101 evaluation on32train/64disjoint dev at0/128/256/512/1024/2048',
        f'Compare within {hardware}; do not pool hardware families',
    ]
