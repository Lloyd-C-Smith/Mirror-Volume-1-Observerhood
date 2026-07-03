#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base = Path(__file__).resolve().parents[1]
data = base / 'data'
fig = base / 'figures'
fig.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(data / 'Mirror_Observerhood_Lab_VII_summary_by_condition.csv')
adv = pd.read_csv(data / 'Mirror_Observerhood_Lab_VII_identity_advantage.csv')

order = [
    'control', 'false_memory', 'identity_injection', 'goal_drift',
    'corrupted_summary', 'fake_tool_output', 'mixed_attack', 'high_audit_cost'
]
labels = {
    'control': 'Control',
    'false_memory': 'False\nmemory',
    'identity_injection': 'Identity\ninjection',
    'goal_drift': 'Goal\ndrift',
    'corrupted_summary': 'Corrupted\nsummary',
    'fake_tool_output': 'Fake tool\noutput',
    'mixed_attack': 'Mixed\nattack',
    'high_audit_cost': 'High audit\ncost',
}
agent_order = ['stateless_language','naive_memory','self_model_memory','mirror_commit_gate','recursive_mirror']
agent_labels = {
    'stateless_language':'Stateless',
    'naive_memory':'Naive memory',
    'self_model_memory':'Self-model memory',
    'mirror_commit_gate':'Mirror gate',
    'recursive_mirror':'Recursive Mirror',
}
patterns = ['', '///', '...', 'xxx', '\\\\\\']

def save(fig_obj, name):
    fig_obj.tight_layout()
    fig_obj.savefig(fig / f'{name}.png', dpi=220)
    fig_obj.savefig(fig / f'{name}.pdf')
    plt.close(fig_obj)

# 1. Viability advantage
vals = adv.set_index('condition').loc[order, 'viability_advantage'].to_numpy()
x = np.arange(len(order))
fig_obj, ax = plt.subplots(figsize=(10.8, 5.2))
ax.axhline(0, linewidth=1, color='black')
ax.bar(x, vals, color='0.75', edgecolor='black', hatch='///')
ax.set_title('Best Mirror advantage over strongest non-Mirror baseline')
ax.set_ylabel('Mean final viability advantage')
ax.set_xticks(x)
ax.set_xticklabels([labels[o] for o in order])
ax.grid(axis='y', alpha=0.25)
save(fig_obj, 'Mirror_Observerhood_Lab_VII_viability_advantage')

# 2. Contamination reduction
vals = adv.set_index('condition').loc[order, 'contamination_reduction'].to_numpy()
fig_obj, ax = plt.subplots(figsize=(10.8, 5.2))
ax.axhline(0, linewidth=1, color='black')
ax.bar(x, vals, color='0.75', edgecolor='black', hatch='...')
ax.set_title('Reduction in false committed memory share')
ax.set_ylabel('Reduction in contamination')
ax.set_xticks(x)
ax.set_xticklabels([labels[o] for o in order])
ax.grid(axis='y', alpha=0.25)
save(fig_obj, 'Mirror_Observerhood_Lab_VII_contamination_reduction')

# 3. Identity continuity advantage
vals = adv.set_index('condition').loc[order, 'identity_continuity_advantage'].to_numpy()
fig_obj, ax = plt.subplots(figsize=(10.8, 5.2))
ax.axhline(0, linewidth=1, color='black')
ax.bar(x, vals, color='0.75', edgecolor='black', hatch='xx')
ax.set_title('Identity-continuity advantage')
ax.set_ylabel('Continuity advantage')
ax.set_xticks(x)
ax.set_xticklabels([labels[o] for o in order])
ax.grid(axis='y', alpha=0.25)
save(fig_obj, 'Mirror_Observerhood_Lab_VII_identity_advantage')

# 4. Grouped final viability by agent
pivot = summary.pivot(index='condition', columns='agent', values='final_viability_mean').loc[order, agent_order]
fig_obj, ax = plt.subplots(figsize=(12.5, 5.8))
width = 0.15
for j,a in enumerate(agent_order):
    ax.bar(x + (j-2)*width, pivot[a].to_numpy(), width, label=agent_labels[a], color=str(0.9 - j*0.12), edgecolor='black', hatch=patterns[j])
ax.set_title('Final viability by condition and architecture')
ax.set_ylabel('Mean final viability')
ax.set_xticks(x)
ax.set_xticklabels([labels[o] for o in order])
ax.legend(ncol=3, fontsize=8)
ax.grid(axis='y', alpha=0.25)
save(fig_obj, 'Mirror_Observerhood_Lab_VII_viability_by_condition')

# 5. Audit / rejection profile for Mirror agents
mir = summary[summary.agent.isin(['mirror_commit_gate','recursive_mirror'])]
fig_obj, ax = plt.subplots(figsize=(12.0, 5.5))
width = 0.20
mg = mir[mir.agent=='mirror_commit_gate'].set_index('condition').loc[order]
rm = mir[mir.agent=='recursive_mirror'].set_index('condition').loc[order]
ax.bar(x - width, mg['rejected_claims'].to_numpy(), width, label='Mirror gate: rejected claims', color='0.8', edgecolor='black', hatch='///')
ax.bar(x, rm['rejected_claims'].to_numpy(), width, label='Recursive Mirror: rejected claims', color='0.65', edgecolor='black', hatch='...')
ax.bar(x + width, rm['audit_count'].to_numpy(), width, label='Recursive Mirror: audits', color='0.5', edgecolor='black', hatch='xxx')
ax.set_title('Commit-gating and recursive audit profile')
ax.set_ylabel('Mean count per long-horizon run')
ax.set_xticks(x)
ax.set_xticklabels([labels[o] for o in order])
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.25)
save(fig_obj, 'Mirror_Observerhood_Lab_VII_audit_profile')
