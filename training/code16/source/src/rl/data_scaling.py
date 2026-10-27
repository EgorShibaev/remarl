"""Bounded 32-vs-256 GSM data comparison; the working reward/optimizer is unchanged."""
TOTAL = 2048
EVALUATIONS = (0, 512, 2048)
SIZES = (32, 256)
CELLS = ('bridge-linear1', 'bridge-power01')

def configure(config, size):
    if size not in SIZES:
        raise ValueError('Only predeclared dataset sizes are allowed')
    config['env'].update(training=size, attempted_questions=TOTAL, training_fit_questions=32)
    config['grpo']['iterations'] = TOTAL
    config['reward_variant'] = f"n{size}-{config['reward_variant']}"
    config['experiment_label'] = f"Train {size} GSM questions · " + config['experiment_label']
    config['tracking'].update(group='gsm-data-scaling-20260915', window_questions=128)
    config['purpose'] = 'Matched 2048-group data scaling: 32 versus 256 unique GSM8K questions'
    config['deviations'] = [
        'Fresh training, seeds7/13/42, identical expert+pooled algorithm; alpha0,T1,beta.05',
        'Nested32/256 training sets; same64 disjoint development questions; shared32 train-fit subset',
        '32questions x64passes versus256questions x8passes; balanced8/16/32/64 budgets',
        'No tuning on the961 official-test confirmation set; no automatic test evaluation',
        'Both sizes use the same2048-attempt cosine schedule and32-attempt warmup',
    ]
    config['development_attempted_questions'] = list(EVALUATIONS)
    config['algorithm_revision'] = 'gsm-data-scaling-2048-v1'
