"""Matched 512-group scalar/learned/fixed position experiments."""
TOTAL = 512
EVALUATIONS = (0, 128, 256, 512)
VARIANTS = ('scalar', 'scalar-skip', 'learned16', 'learned16-skip',
            'learned32-skip', 'sinusoidal16-skip')

def configure(config, variant, hardware):
    if variant not in VARIANTS or hardware not in ('A100', 'H100'):
        raise ValueError('Unknown position study variant/hardware')
    scalar = variant.startswith('scalar')
    encoding = 'none' if scalar else 'sinusoidal' if variant.startswith('sinusoidal') else 'learned'
    config['model'].update(architecture='mlp', hidden_dim=64,
        linear_skip=variant.endswith('-skip'), include_position=False,
        planner_features=['score', 'is_masked', 'pos' if scalar else 'position_id'],
        position_encoding=encoding, position_dim=32 if '32' in variant else 16,
        position_count=256)
    config['grpo'].update(iterations=TOTAL, lr=3e-4, ppo_epochs=4, group_size=8,
        alpha=0.,gamma=0.,beta=0.,weight_decay=0.,expert_steering=False,n_experts=0)
    config['env']['attempted_questions'] = TOTAL
    config['development_attempted_questions'] = list(EVALUATIONS)
    config['evaluation_policy_seeds'] = [101,202,303]
    config['tracking']['group'] = f'gsm32-pos512-{hardware}-{variant}-20260918'
    config['tracking']['study_id'] = f'gsm32-pos512-{hardware}-20260918'
    config['reward_variant'] = variant
    config['experiment_label'] = f'Position512 · {variant} · {hardware}'
    config['algorithm_revision'] = 'position-mlp-512-v1'
    config['position_variant'] = variant
    config['hardware_family'] = hardware
    config['deviations'] = [
        'Fresh512-group cosine horizon,warmup32; not a checkpoint from2048 horizon',
        'Same32GSMtrain/64disjointdev; fixed32NFE,length256,G8,4PPOepochs',
        'Binarycorrectness; noexpert,revisionpenalty,KL,weightdecay',
        f'Position encoding={encoding}; dim={config["model"]["position_dim"]}; skip={config["model"]["linear_skip"]}',
        'Position IDs0..255 relative to answer start; scalar removed for embedding variants',
        'Greedy plus separate sampled101/202/303, averagedpass@1, neverbest-of-three',
        'LR3e-4,T1,clip.2,gradclip.2,pooled; skipzero-variancegroups withoutreplacement',
    ]
