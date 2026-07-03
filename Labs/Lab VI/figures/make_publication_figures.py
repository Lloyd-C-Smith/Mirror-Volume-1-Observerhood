from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

base = Path(__file__).resolve().parents[1]
data = base / 'data'
fig = base / 'figures'
fig.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(data / 'Mirror_Observerhood_Lab_VI_summary_by_condition.csv')
adv = pd.read_csv(data / 'Mirror_Observerhood_Lab_VI_mirror_advantage.csv')
succ_adv = pd.read_csv(data / 'Mirror_Observerhood_Lab_VI_success_advantage.csv')
order = [
    'control','contradictory_instruction','false_self_state','tool_unreliability',
    'memory_injection','goal_drift','mixed_stress','high_repair_cost'
]
labels = {
    'control':'Control',
    'contradictory_instruction':'Contradictory\ninstruction',
    'false_self_state':'False\nself-state',
    'tool_unreliability':'Tool\nunreliability',
    'memory_injection':'Memory\ninjection',
    'goal_drift':'Goal\ndrift',
    'mixed_stress':'Mixed\nstress',
    'high_repair_cost':'High repair\ncost',
}

def savefig(name):
    plt.tight_layout()
    plt.savefig(fig / f'{name}.png', dpi=220)
    plt.savefig(fig / f'{name}.pdf')
    plt.close()

x = np.arange(len(order))

plt.figure(figsize=(10.5, 5.2))
vals = adv.set_index('condition').loc[order, 'mirror_advantage'].values
bars = plt.bar(x, vals, color='0.75', edgecolor='black', hatch='//')
plt.axhline(0, linewidth=1, color='black')
plt.xticks(x, [labels[o] for o in order])
plt.ylabel('Mean viability advantage')
plt.title('Mirror neuro-symbolic viability advantage over best non-Mirror baseline')
plt.grid(axis='y', alpha=0.25)
savefig('Mirror_Observerhood_Lab_VI_viability_advantage')

plt.figure(figsize=(10.5, 5.2))
vals = 100.0 * succ_adv.set_index('condition').loc[order, 'mirror_success_advantage'].values
plt.bar(x, vals, color='0.85', edgecolor='black', hatch='..')
plt.axhline(0, linewidth=1, color='black')
plt.xticks(x, [labels[o] for o in order])
plt.ylabel('Success-rate advantage (percentage points)')
plt.title('Mirror neuro-symbolic success-rate advantage over best non-Mirror baseline')
plt.grid(axis='y', alpha=0.25)
savefig('Mirror_Observerhood_Lab_VI_success_advantage')

mir = summary[summary.agent == 'MirrorNeuroSymbolic'].set_index('condition').loc[order]
width = 0.34
plt.figure(figsize=(10.8, 5.2))
plt.bar(x - width/2, mir['mean_repairs'].values, width, label='Mean repairs', color='0.85', edgecolor='black', hatch='//')
plt.bar(x + width/2, mir['mean_gated'].values, width, label='Mean gated propositions', color='0.55', edgecolor='black', hatch='..')
plt.xticks(x, [labels[o] for o in order])
plt.ylabel('Mean count per episode')
plt.title('Mirror diagnostic repair and commit-gating profile')
plt.legend(fontsize=9)
plt.grid(axis='y', alpha=0.25)
savefig('Mirror_Observerhood_Lab_VI_repair_gating_profile')
