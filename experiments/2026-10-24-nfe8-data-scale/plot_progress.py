"""Plot saved training progress without treating it as held-out accuracy."""
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
status = json.loads(raw)
staged = json.loads((BASE / 'STAGED.json').read_text())
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'],
                     'font.size': 12, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': '.6',
                     'axes.linewidth': .55})
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
colors = ('#277da8', '#ed8737', '#00998c', '#a655ab')
styles = {7: '-', 17: '--', 27: ':'}
receipt = {}
for size, color in zip((4, 32, 256, 2048), colors):
    for seed in staged['training_seeds']:
        name = f'n{size * 2}-seed{seed}'
        metrics = status['runs'].get(name, {}).get('metrics', [])
        if not metrics:
            continue
        assert [m['accepted_groups'] for m in metrics] == list(range(16, 16 * len(metrics) + 1, 16))
        for ax, task in zip(axes, ('gsm8k', 'math')):
            x, y = [], []
            for i, row in enumerate(metrics):
                window = metrics[max(0, i - 31):i + 1]
                count = sum(m['task_metrics'][task]['trajectories'] for m in window)
                correct = sum(m['task_metrics'][task]['correct'] for m in window)
                assert count > 0
                x.append(row['accepted_groups'] / 1024)
                y.append(100 * correct / count)
            ax.plot(x, y, color=color, lw=.85, alpha=.8, ls=styles[seed])
            ax.plot(x[-1], y[-1], color=color, marker='o', ms=2.5)
            receipt[f'{name}/{task}'] = dict(groups=[int(v * 1024) for v in x], accuracy=y)
for ax, title in zip(axes, ('GSM8K training subset', 'MATH training subset')):
    ax.set_title(title)
    ax.set_xlabel('Accepted training groups (×1,024)')
    ax.set_ylabel('Training accuracy (%)')
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.grid(alpha=.17, lw=.5)
handles = [Line2D([], [], color=c, lw=1, label=f'{s:,} questions per dataset')
           for s, c in zip(staged['per_dataset'], colors)]
fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False)
fig.legend(handles=[Line2D([], [], color='.3', lw=1, ls=styles[s], label=f'Seed {i+1}')
                    for i, s in enumerate(staged['training_seeds'])],
           loc='upper center', bbox_to_anchor=(.5, .935), ncol=3, frameon=False)
stamp = datetime.fromtimestamp(status['time'], ZoneInfo('Europe/Paris')).strftime('%d %b, %H:%M %Z')
fig.text(.5, .035, 'NFE8 · individual training runs · 32-collection trailing means · '
         'training accuracy, not test accuracy · ' + stamp, ha='center', fontsize=9)
fig.subplots_adjust(left=.065, right=.985, bottom=.19, top=.75, wspace=.22)
fig.savefig(BASE / 'data-size-training.png', dpi=180)
plt.close(fig)
(BASE / 'training-plot-data.json').write_text(json.dumps(dict(
    status_sha256=hashlib.sha256(raw).hexdigest(), values=receipt,
    inference_test_results=False, training_seeds=staged['training_seeds'],
    window_collections=32), indent=2) + '\n')
print(BASE / 'data-size-training.png')
