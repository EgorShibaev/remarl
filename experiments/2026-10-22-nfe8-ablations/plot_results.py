"""Plot audited early training and explicitly separate calibration from full tests."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE=Path(__file__).resolve().parent
data_path=BASE/'RESULTS_AUDITED.json'
data=json.loads(data_path.read_text())
reference_path=BASE.parent/'2026-10-07-apple-mixture-native/training-snapshot-latest.json'
reference=next(r for r in json.loads(reference_path.read_text())['runs'] if r['nfe']==8)
COLORS={'full':'#157ca8','no_hidden':'#e7862f','no_position':'#00998c','bias_only':'#a655ab','no_revision':'#c4544b'}
LABELS={'full':'Full policy · archived reference','no_hidden':'No hidden states',
        'no_position':'No position embedding','bias_only':'Position + mask biases only','no_revision':'No revision'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'text.color':'#284653',
    'axes.labelcolor':'#284653','axes.edgecolor':'#bed0df','xtick.color':'#5d768a',
    'ytick.color':'#5d768a','axes.spines.top':False,'axes.spines.right':False})
stamp=data['checked_at'][:16].replace('T',' ')+' CEST'
curves=[]
for name,row in data['variants'].items():
    if row['metrics']:curves.append((name,row['metrics']))
stop_path=BASE/'USER_STOPPED_16K.json'
user_stopped=json.loads(stop_path.read_text()) if stop_path.exists() else None
limit=user_stopped['selected_groups'] if user_stopped else max(ms[-1]['accepted_groups'] for _,ms in curves)
curves=[(name,[m for m in ms if m['accepted_groups']<=limit]) for name,ms in curves]
ref=[m for m in reference['metrics'] if m['accepted_groups']<=limit]
curves=[('full',ref)]+curves
window_size=64 if limit>=8192 else 16
fig,axes=plt.subplots(2,2,figsize=(16,10.5))
for ax,task,title in zip(axes[0],('gsm8k','math'),('GSM8K · training accuracy','MATH · training accuracy')):
    for name,ms in curves:
        x=[];y=[]
        for i in range(window_size-1,len(ms)):
            window=ms[i-window_size+1:i+1]
            correct=sum(m['task_metrics'][task]['correct'] for m in window)
            total=sum(m['task_metrics'][task]['trajectories'] for m in window)
            x.append(ms[i]['accepted_groups']);y.append(100*correct/total)
        if not x:continue
        label=LABELS[name]+(f" · {ms[-1]['accepted_groups']} groups" if name!='full' else '')
        ax.plot(x,y,color=COLORS[name],lw=1.65,alpha=.82,
                linestyle='--' if name=='full' else '-',label=label)
    ax.set_title(title,loc='left',fontweight='bold');ax.set_xlabel('Accepted training groups')
    ax.set_ylabel('Accuracy (%)');ax.grid(alpha=.22);ax.set_xlim(0,max(640,limit+48))
    ax.set_ylim(0,65 if task=='gsm8k' else 40)
test_series={'full':[r['summary'] for r in reference['tests']]}
test_series.update({name:row['tests'] for name,row in data['variants'].items()})
full_test_points={};trained_tests_available=False
for ax,split,title in zip(axes[1],('test_gsm8k','test_math'),('GSM8K · full test, three-seed means','MATH500 · full test, three-seed means')):
    endpoints=[]
    for name,cells in test_series.items():
        points=[]
        for group in sorted({c['accepted_groups'] for c in cells if c['accepted_groups']<=min(limit,32768)}):
            paired={s:[c for c in cells if c['accepted_groups']==group and c['split']==s]
                    for s in ('test_gsm8k','test_math')}
            if not all(sorted(c['seed'] for c in cs)==[101,202,303] for cs in paired.values()):continue
            chosen=paired[split]
            assert all(c['complete'] and c['count']==(1319 if split=='test_gsm8k' else 500) for c in chosen)
            assert len({c['policy_sha256'] for cs in paired.values() for c in cs})==1
            points.append(dict(groups=group,mean=100*sum(c['correct'] for c in chosen)/sum(c['count'] for c in chosen)))
        if not points:continue
        full_test_points.setdefault(name,{})[split]=points
        trained_tests_available |= name!='full' and points[-1]['groups']>0
        ax.plot([p['groups'] for p in points],[p['mean'] for p in points],color=COLORS[name],
                lw=1.6,alpha=.85,marker='o',ms=4,ls='--' if name=='full' else '-')
        endpoints.append((points[-1]['mean'],name,points[-1]['groups']))
    # Keep nearby endpoint scores readable; leaders point to the measured checkpoint.
    last_label=-100
    for score,name,group in sorted(endpoints):
        label_y=max(score,last_label+(1.6 if split=='test_math' else 2.0))
        last_label=label_y
        ax.annotate(f'{score:.2f}%',(group,score),xytext=(limit+350,label_y),textcoords='data',
                    color=COLORS[name],fontsize=9,ha='left',va='center',
                    arrowprops=dict(arrowstyle='-',color=COLORS[name],lw=.7,alpha=.7),
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=.3))
    ax.set_title(title,loc='left',fontweight='bold');ax.set_xlabel('Accepted training groups')
    ax.set_ylabel('Accuracy (%)');ax.grid(alpha=.22);ax.set_ylim(0,65 if split=='test_gsm8k' else 30)
    ax.set_xlim(0,max(4096,limit*1.15))
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.065,.91),ncol=3,frameon=False,fontsize=10)
fig.text(.065,.975,'NFE8 ablations · training and completed full tests',fontsize=23,fontweight='bold')
budget_note='stopped · matched 16,384-group checkpoints' if user_stopped else '32,768-group target'
fig.text(.065,.94,stamp+' · fresh training · '+budget_note+' · one training seed per variant',fontsize=11)
fig.text(.065,.055,f'Training: trailing {window_size} collections ({window_size*16:,} accepted groups), weighted by all attempted trajectories; rejected groups included.',fontsize=10)
fig.text(.065,.030,'Tests: full GSM8K (1,319) / MATH500 (500), complete paired three-seed means only. Dashed reference: original full-policy run.',fontsize=10)
fig.subplots_adjust(left=.065,right=.98,bottom=.115,top=.81,wspace=.22,hspace=.42)
for ext in ('png','svg'):fig.savefig(BASE/f'ablation-training.{ext}',dpi=175,facecolor='white')
plt.close(fig)

temps=[0.,.05,.1,.25,.5,1.,2.]
calibration=data['temperature_selected']['calibration']
matched=data['revision_match_selected']
target=next(c for c in data['controls'] if c['name']=='learned-reference' and c['split']=='calibration')
selected=next(c for c in matched['candidates'] if c['retention_bonus']==matched['retention_bonus'])
fig,axes=plt.subplots(1,3,figsize=(17,6.4),gridspec_kw={'width_ratios':[1,1.13,1]})
ax=axes[0]
y=[100*next(c for c in calibration if c['temperature']==t)['macro_accuracy'] for t in temps]
ax.plot(range(7),y,color='#8e4cb6',marker='o',lw=1.5,alpha=.85,label='Untrained P2')
best=temps.index(data['temperature_selected']['temperature'])
ax.scatter([best],[y[best]],s=135,facecolors='none',edgecolors='#8e4cb6',lw=1.7,zorder=5)
ax.axhline(100*target['macro_accuracy'],color=COLORS['full'],linestyle='--',lw=1.2,alpha=.7,label='Full learned policy · 32k')
ax.annotate(f'T={temps[best]:g} selected\n{y[best]:.2f}%',(best,y[best]),xytext=(0,14),textcoords='offset points',ha='center',fontsize=10)
ax.set_title('Training calibration',loc='left',fontweight='bold');ax.set_ylabel('GSM8K + MATH macro accuracy (%)')
ax.set_ylim(0,55);ax.legend(frameon=False,fontsize=9,loc='lower left')

ax=axes[1];test_points=[];learned_test_points=[]
for split,color,label in [('test_gsm8k','#157ca8','GSM8K · 1,319 questions'),('test_math','#00998c','MATH500 · 500 questions')]:
    xs=[];ys=[]
    for i,t in enumerate(temps):
        cells=[c for c in data['controls'] if c['name']==f'p2-temp-{t:g}' and c['split']==split]
        needed=[101] if t==0 else [101,202,303]
        if sorted(c['seed'] for c in cells)!=needed:continue
        score=100*sum(c['correct'] for c in cells)/sum(c['count'] for c in cells)
        xs.append(i);ys.append(score);test_points.append(dict(temperature=t,split=split,score=score,seeds=needed))
    ax.plot(xs,ys,color=color,marker='o',lw=1.3,alpha=.85,label=label)
    for x,y in zip(xs,ys):ax.annotate(f'{y:.2f}',(x,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9,color=color)
    # Iteration zero is exactly P2 and supplies the same audited full T=1 reference.
    cells=[c for c in data['variants']['no_hidden']['tests'] if c['split']==split and c['accepted_groups']==0]
    assert sorted(c['seed'] for c in cells)==[101,202,303]
    score=100*sum(c['correct'] for c in cells)/sum(c['count'] for c in cells)
    ax.scatter([5],[score],marker='D',facecolors='none',edgecolors=color,s=65,zorder=4)
    ax.annotate(f'{score:.2f}',(5,score),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9,color=color)
    # Fixed 32k-group checkpoint: no selection by the current temperature sweep.
    learned=[r['summary'] for r in reference['tests'] if r['summary']['split']==split
             and r['summary']['accepted_groups']==32768]
    assert sorted(c['seed'] for c in learned)==[101,202,303]
    assert all(c['policy_sha256']==data['config']['reference_policy_sha256'] for c in learned)
    full_score=100*sum(c['correct'] for c in learned)/sum(c['count'] for c in learned)
    learned_test_points.append(dict(split=split,score=full_score,checkpoint_groups=32768,
                                   policy_sha256=data['config']['reference_policy_sha256']))
    ax.axhline(full_score,color=color,ls='--',lw=1.15,alpha=.55)
    ax.annotate(f'Learned {full_score:.2f}',(6.25,full_score),xytext=(0,6),
                textcoords='offset points',ha='right',fontsize=9,color=color)
ax.set_title('Full test · available results',loc='left',fontweight='bold')
ax.set_ylabel('Accuracy (%)');ax.set_ylim(0,60);ax.legend(frameon=False,fontsize=9,loc='upper left')
for ax in axes[:2]:
    ax.set_xticks(range(7),[f'{t:g}' for t in temps]);ax.set_xlim(-.4,6.4);ax.set_xlabel('Position temperature')

ax=axes[2];control_points=[]
control_names=['p2-temp-1',f"p2-temp-{data['temperature_selected']['temperature']:g}",'p2-revision-matched','learned']
for split,color,shift in [('test_gsm8k','#157ca8',-.18),('test_math','#00998c',.18)]:
    for index,name in enumerate(control_names):
        if name=='learned':
            score=next(p['score'] for p in learned_test_points if p['split']==split)
        else:
            cells=[c for c in data['controls'] if c['name']==name and c['split']==split]
            if sorted(c['seed'] for c in cells)!=[101,202,303]:continue
            score=100*sum(c['correct'] for c in cells)/sum(c['count'] for c in cells)
        ax.bar(index+shift,score,width=.32,color=color,alpha=.8,zorder=3)
        ax.annotate(f'{score:.2f}',(index+shift,score),xytext=(0,5),textcoords='offset points',ha='center',fontsize=8,color=color)
        control_points.append(dict(name=name,split=split,score=score))
ax.set_title('Full test · revision control',loc='left',fontweight='bold')
ax.set_xticks(range(4),['P2\nT=1','P2\nT=0.25','Matched\nrevisions','Learned\n32k'])
ax.set_ylabel('Accuracy (%)');ax.set_ylim(0,60);ax.set_xlim(-.6,3.6)
for ax in axes:ax.grid(alpha=.2)
fig.text(.055,.96,'NFE8 P2 controls · calibration and full-test results',fontsize=23,fontweight='bold')
fig.text(.055,.91,stamp+' · temperature and retention bonus selected before test evaluation',fontsize=11)
complete_controls=bool(data['controls_complete'])
fig.text(.055,.08,'Left: 256 TRAINING calibration questions; temperature/retention bonus fixed before tests. Middle/right: full benchmark tests.',fontsize=10)
fig.text(.055,.045,'Complete three-seed means only; greedy T=0 is deterministic. Blue: GSM8K; teal: MATH500. Dashed: fixed 32k learned policy. '+('Controls complete.' if complete_controls else 'Sweep running.'),fontsize=9.5)
fig.subplots_adjust(left=.055,right=.985,bottom=.2,top=.79,wspace=.33)
for ext in ('png','svg'):fig.savefig(BASE/f'p2-controls.{ext}',dpi=175,facecolor='white')
plt.close(fig)
receipt=dict(checked_at=data['checked_at'],inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (data_path,reference_path)},
    training_window_collections=window_size,training_target_groups=16384 if user_stopped else 32768,
    original_training_target_groups=32768,user_stopped=bool(user_stopped),trained_test_results_available=trained_tests_available,
    full_test_training_histories=full_test_points,held_out_control_points=control_points,controls_complete=complete_controls,
    held_out_temperature_points=test_points,learned_reference_full_tests=learned_test_points,calibration_is_held_out=False,
    temperature_selected=data['temperature_selected']['temperature'],revision_match_error=matched['absolute_match_error'])
(BASE/'plot-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(BASE/'ablation-training.png');print(BASE/'p2-controls.png')
