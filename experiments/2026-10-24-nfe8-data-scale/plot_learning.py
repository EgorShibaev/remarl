"""Show every completed three-inference-seed test, with training seeds separate."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent
raw = (BASE / 'STATUS.json').read_bytes()
staged_raw = (BASE / 'STAGED.json').read_bytes()
status, staged = json.loads(raw), json.loads(staged_raw)
colors = ('#277da8', '#ed8737', '#00998c', '#a655ab')
styles = {7: '-', 17: '--', 27: ':'}
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'],
                     'font.size': 12, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': '.6',
                     'axes.linewidth': .55})


def point(tests, split, age):
    values = [t for t in tests if t['split'] == split and t['accepted_groups'] == age]
    if len(values) != len(staged['inference_seeds']):
        return None
    assert sorted(t['seed'] for t in values) == sorted(staged['inference_seeds'])
    assert all(t['complete'] and t['count'] == staged['tests'][split] and t['nfe'] == 8
               for t in values)
    assert len({t['policy_sha256'] for t in values}) == 1
    return 100 * sum(t['correct'] for t in values) / sum(t['count'] for t in values)


fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
receipt = {}
for ax, split, title, limits in zip(axes, ('test_gsm8k', 'test_math'),
                                   ('GSM8K', 'MATH500'), ((33, 56), (14, 22))):
    for size, color in zip(staged['per_dataset'], colors):
        for seed in staged['training_seeds']:
            name = f'n{2*size}-seed{seed}'
            tests = status['runs'].get(name, {}).get('tests', [])
            points = [(age, value) for age in staged['full_tests_at_groups']
                      if (value := point(tests, split, age)) is not None]
            if not points:
                continue
            ax.plot([p[0]/1024 for p in points], [p[1] for p in points], color=color,
                    ls=styles[seed], lw=1, marker='o', ms=3,
                    markerfacecolor=color if seed == 7 else 'white', markeredgewidth=.8)
            receipt[f'{name}/{split}'] = points
    reference = point(staged['reference_evaluations'].values(), split, staged['target_groups'])
    ax.axhline(reference, color='.45', ls=(0, (2, 3)), lw=.75)
    ax.annotate(f'Full data, 16k: {reference:.2f}', (4, reference), xytext=(0, 5),
                textcoords='offset points', ha='left', fontsize=10, color='.3')
    ax.set_title(title)
    ax.set_xlabel('Accepted training groups (×1,024)')
    ax.set_ylabel('Full-test accuracy (%)')
    ax.set_xticks([4, 8, 12, 16])
    ax.set_xlim(3.5, 16.5)
    ax.set_ylim(*limits)
    ax.grid(axis='y', color='.9', linewidth=.5)
    ax.set_axisbelow(True)
handles = [Line2D([], [], color=c, lw=1, label=f'{s:,} questions per dataset')
           for s, c in zip(staged['per_dataset'], colors)]
fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False)
visible_seeds = {int(key.split('/')[0].split('-seed')[1]) for key in receipt}
fig.legend(handles=[Line2D([], [], color='.3', lw=1, ls=styles[s], label=f'Seed {i+1}')
                    for i, s in enumerate(staged['training_seeds']) if s in visible_seeds],
           loc='upper center', bbox_to_anchor=(.5, .935), ncol=3, frameon=False)
stamp = datetime.fromtimestamp(status['time'], ZoneInfo('Europe/Paris')).strftime('%d %b, %H:%M %Z')
fig.text(.5, .035, 'NFE8 · each point averages three inference seeds · '
         'incomplete evaluations omitted · ' + stamp, ha='center', fontsize=9)
fig.subplots_adjust(left=.065, right=.985, bottom=.19, top=.75, wspace=.22)
fig.savefig(BASE / 'data-size-learning.png', dpi=180)
plt.close(fig)
(BASE / 'learning-plot-data.json').write_text(json.dumps(dict(
    status_sha256=hashlib.sha256(raw).hexdigest(),
    staged_sha256=hashlib.sha256(staged_raw).hexdigest(),
    individual_training_seeds=True, values=receipt), indent=2) + '\n')
print(BASE / 'data-size-learning.png')
