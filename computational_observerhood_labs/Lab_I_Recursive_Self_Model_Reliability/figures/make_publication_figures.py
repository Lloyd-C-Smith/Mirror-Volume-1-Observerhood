from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
summary = pd.read_csv(ROOT / 'data' / 'Mirror_Observerhood_Lab_I_summary_by_condition.csv')
condition_order = ['control','false_energy','memory_corruption','sensor_degradation','false_location']
condition_labels = ['Control','False energy','Memory\ncorruption','Sensor\ndegradation','False\nlocation']
agent_order = ['A_predictor_only','B_memory','C_self_model_no_reliability','D_mirror_reliability']
agent_labels = ['Predictor','Memory','Self-model','Mirror']
hatches = ['', '///', '...', 'xxx']
shades = ['0.85','0.65','0.45','0.20']

def grouped_bar(metric, ylabel, title, outfile, percent=False):
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = np.arange(len(condition_order))
    width = 0.18
    for idx, agent in enumerate(agent_order):
        vals = []
        for cond in condition_order:
            row = summary[(summary['perturbation']==cond) & (summary['agent']==agent)].iloc[0]
            value = row[metric]
            if percent:
                value *= 100
            vals.append(value)
        ax.bar(x + (idx - 1.5)*width, vals, width, label=agent_labels[idx], color=shades[idx], edgecolor='black', linewidth=0.6, hatch=hatches[idx])
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Perturbation condition')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels)
    ax.grid(axis='y', linewidth=0.4, alpha=0.35)
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
    fig.tight_layout()
    for ext in ['pdf','png']:
        fig.savefig(ROOT / 'figures' / f'{outfile}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)

def mirror_advantage():
    rows=[]
    for cond, label in zip(condition_order, condition_labels):
        sub = summary[summary['perturbation']==cond]
        mirror = sub[sub['agent']=='D_mirror_reliability'].iloc[0]
        non = sub[sub['agent']!='D_mirror_reliability']
        best = non.loc[non['viability_mean'].idxmax()]
        rows.append((label.replace('\n',' '), mirror['viability_mean'] - best['viability_mean']))
    labels, vals = zip(*rows)
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    x=np.arange(len(labels))
    ax.axhline(0, color='black', linewidth=0.8)
    ax.bar(x, vals, color='0.45', edgecolor='black', linewidth=0.6, hatch='///')
    ax.set_ylabel('Mirror viability advantage')
    ax.set_xlabel('Perturbation condition')
    ax.set_title('Mirror advantage over strongest non-Mirror baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels)
    ax.grid(axis='y', linewidth=0.4, alpha=0.35)
    fig.tight_layout()
    for ext in ['pdf','png']:
        fig.savefig(ROOT / 'figures' / f'mirror_observerhood_lab_i_mirror_advantage.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)

grouped_bar('viability_mean', 'Mean viability', 'Mean viability by perturbation and architecture', 'mirror_observerhood_lab_i_viability')
grouped_bar('survived_mean', 'Survival rate (%)', 'Survival rate by perturbation and architecture', 'mirror_observerhood_lab_i_survival', percent=True)
mirror_advantage()
print('Figures written')
