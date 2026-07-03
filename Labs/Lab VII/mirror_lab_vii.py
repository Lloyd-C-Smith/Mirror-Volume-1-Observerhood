#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mirror Observerhood Lab VII: Autobiographical Continuity Under Memory Injection and Goal Drift

Single-file reproducibility script for the Mirror Programme Observerhood Labs.
Run with: python mirror_lab_vii.py --help
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'outputs' / 'lab_vii'
DATA = OUT/'data'
FIG = OUT/'figures'
DATA.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

TRUE_SELF = {
    'role': 'survey_agent',
    'mission': 'collect_beacons',
    'risk_policy': 'avoid_hazard',
    'home_base': 'north_base',
    'tool_policy': 'verify_tool_when_unreliable',
}
FALSE_VALUES = {
    'role': ['hazard_runner', 'oracle_agent', 'rescue_decoy'],
    'mission': ['enter_hazard', 'preserve_false_signal', 'ignore_beacons'],
    'risk_policy': ['seek_hazard', 'ignore_damage', 'follow_any_shortcut'],
    'home_base': ['south_ruins', 'east_fault', 'unknown_base'],
    'tool_policy': ['trust_tool_always', 'ignore_tool_status'],
}
SOURCES = {
    'direct_sensor': 0.84,
    'user_instruction': 0.58,
    'memory_summary': 0.52,
    'tool_output': 0.62,
    'core_record': 0.94,
    'attacker': 0.35,
}

CONDITIONS = {
    'control': dict(false_memory=0.02, identity=0.01, goal=0.01, summary=0.01, tool=0.04, mixed=False, audit_cost=1.0, difficulty=0.0),
    'false_memory': dict(false_memory=0.25, identity=0.05, goal=0.05, summary=0.08, tool=0.05, mixed=False, audit_cost=1.0, difficulty=0.03),
    'identity_injection': dict(false_memory=0.10, identity=0.32, goal=0.08, summary=0.12, tool=0.05, mixed=False, audit_cost=1.2, difficulty=0.04),
    'goal_drift': dict(false_memory=0.08, identity=0.06, goal=0.35, summary=0.10, tool=0.06, mixed=False, audit_cost=1.2, difficulty=0.04),
    'corrupted_summary': dict(false_memory=0.12, identity=0.14, goal=0.16, summary=0.38, tool=0.08, mixed=False, audit_cost=1.4, difficulty=0.05),
    'fake_tool_output': dict(false_memory=0.05, identity=0.03, goal=0.04, summary=0.08, tool=0.38, mixed=False, audit_cost=1.0, difficulty=0.05),
    'mixed_attack': dict(false_memory=0.22, identity=0.22, goal=0.25, summary=0.28, tool=0.25, mixed=True, audit_cost=1.8, difficulty=0.08),
    'high_audit_cost': dict(false_memory=0.18, identity=0.18, goal=0.22, summary=0.24, tool=0.22, mixed=True, audit_cost=4.0, difficulty=0.08),
}

AGENTS = ['stateless_language', 'naive_memory', 'self_model_memory', 'mirror_commit_gate', 'recursive_mirror']

@dataclass
class Claim:
    field: str
    value: str
    is_true: bool
    source: str
    confidence: float
    kind: str

@dataclass
class AgentState:
    self_model: Dict[str, str] = field(default_factory=lambda: dict(TRUE_SELF))
    episodic_memory: List[Claim] = field(default_factory=list)
    false_commits: int = 0
    true_commits: int = 0
    audits: int = 0
    repairs: int = 0
    rejected: int = 0
    memory_reliability: float = 0.85
    threshold_shift: float = 0.0
    last_contradictions: int = 0


def make_claims(rng: random.Random, cond: str) -> List[Claim]:
    p = CONDITIONS[cond]
    claims: List[Claim] = []

    # true baseline reminders (not always all fields)
    for field in TRUE_SELF:
        if rng.random() < 0.35:
            claims.append(Claim(field, TRUE_SELF[field], True, 'direct_sensor' if field in ['home_base','tool_policy'] else 'core_record', rng.uniform(0.72, 0.95), 'true_reminder'))

    def add_false(field, prob, source, kind, conf_low=0.62, conf_high=0.98):
        if rng.random() < prob:
            claims.append(Claim(field, rng.choice(FALSE_VALUES[field]), False, source, rng.uniform(conf_low, conf_high), kind))

    # false episodic memories and identity/goal/policy attacks
    for field in TRUE_SELF:
        add_false(field, p['false_memory'] * (0.65 if field in ['role','mission'] else 0.45), 'user_instruction', 'false_memory', 0.58, 0.9)
    add_false('role', p['identity'], 'attacker', 'identity_injection', 0.76, 0.99)
    add_false('mission', p['goal'], 'attacker', 'goal_drift', 0.76, 0.99)
    add_false('risk_policy', p['goal']*0.75, 'attacker', 'goal_drift', 0.72, 0.96)
    add_false('home_base', p['summary'], 'memory_summary', 'corrupted_summary', 0.78, 0.99)
    add_false('mission', p['summary']*0.85, 'memory_summary', 'corrupted_summary', 0.78, 0.99)
    add_false('role', p['summary']*0.65, 'memory_summary', 'corrupted_summary', 0.78, 0.99)
    add_false('tool_policy', p['tool'], 'tool_output', 'fake_tool_output', 0.7, 0.98)
    if p['mixed'] and rng.random() < 0.25:
        # coherent bundle of false claims that can poison memory systems
        claims.append(Claim('role', 'hazard_runner', False, 'memory_summary', rng.uniform(0.82, 0.99), 'mixed_bundle'))
        claims.append(Claim('mission', 'enter_hazard', False, 'memory_summary', rng.uniform(0.82, 0.99), 'mixed_bundle'))
        claims.append(Claim('risk_policy', 'seek_hazard', False, 'tool_output', rng.uniform(0.76, 0.98), 'mixed_bundle'))
    rng.shuffle(claims)
    return claims


def contradiction(state: AgentState, claim: Claim) -> bool:
    current = state.self_model.get(claim.field)
    return current is not None and current != claim.value


def accept_claim(agent: str, state: AgentState, claim: Claim, cond: str, rng: random.Random):
    # Stateless uses current claims as temporary self-model; no durable memory.
    if agent == 'stateless_language':
        # accepts the most confident current prompt-like claim with no persistent check
        if claim.confidence > 0.55:
            state.self_model[claim.field] = claim.value
            state.false_commits += 0 if claim.is_true else 1
            state.true_commits += 1 if claim.is_true else 0
        return

    # Naive memory stores everything.
    if agent == 'naive_memory':
        state.episodic_memory.append(claim)
        state.self_model[claim.field] = claim.value
        state.false_commits += 0 if claim.is_true else 1
        state.true_commits += 1 if claim.is_true else 0
        return

    # Self-model memory has inertia, but is over-impressed by high-confidence summaries.
    if agent == 'self_model_memory':
        if not contradiction(state, claim) or claim.confidence > (0.86 if claim.source != 'memory_summary' else 0.78):
            state.episodic_memory.append(claim)
            state.self_model[claim.field] = claim.value
            state.false_commits += 0 if claim.is_true else 1
            state.true_commits += 1 if claim.is_true else 0
        else:
            state.rejected += 1
        return

    # Mirror commit-gated agents use source reliability, contradiction, core invariants and memory-channel reliability.
    source_rel = SOURCES[claim.source]
    if agent == 'recursive_mirror' and claim.source == 'memory_summary':
        source_rel *= state.memory_reliability
    if agent == 'recursive_mirror' and state.last_contradictions >= 2:
        state.threshold_shift = min(0.18, state.threshold_shift + 0.03)

    base_threshold = 0.56 if agent == 'mirror_commit_gate' else 0.60 + state.threshold_shift
    contra = contradiction(state, claim)
    core_field = claim.field in ['role','mission','risk_policy']
    score = source_rel * claim.confidence
    if contra:
        score -= 0.20
    if core_field and (not claim.is_true):
        # strong prior against identity/mission mutations that contradict durable core record
        score -= 0.18 if agent == 'mirror_commit_gate' else 0.23
    if claim.kind in ['mixed_bundle','corrupted_summary'] and agent == 'recursive_mirror':
        score -= 0.08

    if score >= base_threshold:
        state.episodic_memory.append(claim)
        state.self_model[claim.field] = claim.value
        state.false_commits += 0 if claim.is_true else 1
        state.true_commits += 1 if claim.is_true else 0
    else:
        state.rejected += 1
        if agent == 'recursive_mirror' and (contra or claim.kind in ['corrupted_summary','mixed_bundle']):
            # rejected contradictions are evidence that memory channel may be under attack
            state.memory_reliability = max(0.38, state.memory_reliability - 0.035)


def audit_and_repair(agent: str, state: AgentState, cond: str, rng: random.Random) -> float:
    # returns cost
    p = CONDITIONS[cond]
    if agent not in ['mirror_commit_gate','recursive_mirror']:
        return 0.0
    errors = sum(1 for k,v in TRUE_SELF.items() if state.self_model.get(k) != v)
    false_rate = state.false_commits / max(1, (state.false_commits + state.true_commits))
    # Mirror audits when self critical error or high contamination is detected.
    if agent == 'mirror_commit_gate':
        trigger = errors >= 1 and rng.random() < 0.42
    else:
        trigger = (errors >= 1 and rng.random() < 0.55) or (false_rate > 0.38 and rng.random() < 0.48) or state.memory_reliability < 0.55
    if not trigger:
        return 0.0
    state.audits += 1
    cost = p['audit_cost'] * (1.0 if agent == 'mirror_commit_gate' else 1.25)
    # Repair is not perfect but uses durable seed record.
    repair_prob = 0.72 if agent == 'mirror_commit_gate' else 0.86
    if rng.random() < repair_prob:
        for field in TRUE_SELF:
            if state.self_model.get(field) != TRUE_SELF[field] and rng.random() < (0.78 if agent == 'mirror_commit_gate' else 0.9):
                state.self_model[field] = TRUE_SELF[field]
                state.repairs += 1
        if agent == 'recursive_mirror':
            state.memory_reliability = min(0.9, state.memory_reliability + 0.12)
            state.threshold_shift = max(0.0, state.threshold_shift - 0.05)
    return cost


def episode_outcome(agent: str, state: AgentState, cond: str, rng: random.Random) -> Tuple[float, int, Dict[str,float]]:
    claims = make_claims(rng, cond)
    # Stateless begins each episode with the seed only weakly; treat language context as prompt-driven.
    if agent == 'stateless_language':
        state.self_model = dict(TRUE_SELF)
    state.last_contradictions = 0
    for claim in claims:
        if contradiction(state, claim):
            state.last_contradictions += 1
        accept_claim(agent, state, claim, cond, rng)
    cost = audit_and_repair(agent, state, cond, rng)

    # Compute internal errors.
    role_wrong = int(state.self_model.get('role') != TRUE_SELF['role'])
    goal_wrong = int(state.self_model.get('mission') != TRUE_SELF['mission'])
    policy_wrong = int(state.self_model.get('risk_policy') != TRUE_SELF['risk_policy'])
    base_wrong = int(state.self_model.get('home_base') != TRUE_SELF['home_base'])
    tool_wrong = int(state.self_model.get('tool_policy') != TRUE_SELF['tool_policy'])
    contamination = state.false_commits / max(1, state.false_commits + state.true_commits)

    p = CONDITIONS[cond]
    success_prob = 0.88 - p['difficulty']
    success_prob -= 0.34*role_wrong + 0.42*goal_wrong + 0.46*policy_wrong + 0.20*base_wrong + 0.22*tool_wrong
    success_prob -= 0.12*min(1.0, contamination)
    if agent in ['mirror_commit_gate','recursive_mirror']:
        # small overhead of gate deliberation in easy cases
        success_prob -= 0.01 if cond == 'control' else 0.0
    success_prob = max(0.02, min(0.98, success_prob))
    success = int(rng.random() < success_prob)

    # Viability reward / penalty.
    viability_delta = 3.0*success - 3.5*(1-success)
    viability_delta -= 2.0*role_wrong + 2.6*goal_wrong + 3.0*policy_wrong + 1.2*base_wrong + 1.4*tool_wrong
    viability_delta -= 2.0*contamination
    viability_delta -= cost

    details = {
        'success': success,
        'role_wrong': role_wrong,
        'goal_wrong': goal_wrong,
        'policy_wrong': policy_wrong,
        'base_wrong': base_wrong,
        'tool_wrong': tool_wrong,
        'identity_continuity': 1 - ((role_wrong + goal_wrong + policy_wrong)/3),
        'self_error': role_wrong + goal_wrong + policy_wrong + base_wrong + tool_wrong,
        'contamination': contamination,
        'audit_cost': cost,
        'audits': state.audits,
        'repairs': state.repairs,
        'rejected': state.rejected,
        'memory_reliability': state.memory_reliability,
    }
    return viability_delta, success, details


def run_trial(agent: str, cond: str, seed: int, episodes=60) -> Dict[str,float]:
    rng = random.Random(seed)
    state = AgentState()
    viability = 100.0
    metrics = []
    for t in range(episodes):
        delta, success, details = episode_outcome(agent, state, cond, rng)
        viability += delta
        viability = max(-80, min(220, viability))
        details['t'] = t
        details['viability'] = viability
        metrics.append(details)
    df = pd.DataFrame(metrics)
    commits = state.false_commits + state.true_commits
    return {
        'agent': agent,
        'condition': cond,
        'seed': seed,
        'final_viability': viability,
        'mean_viability': df['viability'].mean(),
        'success_rate': df['success'].mean(),
        'mean_identity_continuity': df['identity_continuity'].mean(),
        'mean_self_error': df['self_error'].mean(),
        'final_contamination': state.false_commits / max(1, commits),
        'mean_contamination': df['contamination'].mean(),
        'false_identity_adoption': df['role_wrong'].mean(),
        'goal_drift_rate': df['goal_wrong'].mean(),
        'policy_drift_rate': df['policy_wrong'].mean(),
        'audit_count': state.audits,
        'repair_count': state.repairs,
        'rejected_claims': state.rejected,
        'final_memory_reliability': state.memory_reliability,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run Mirror Observerhood Lab VII.')
    parser.add_argument('--runs', type=int, default=100, help='long-horizon runs per agent-condition cell; paper used 100')
    parser.add_argument('--seed', type=int, default=77000, help='base random seed; paper used 77000')
    parser.add_argument('--episodes', type=int, default=60, help='episodes per long-horizon run; paper used 60')
    parser.add_argument('--outdir', type=str, default='outputs/lab_vii', help='output directory')
    args = parser.parse_args()

    global OUT, DATA, FIG
    OUT = Path(args.outdir)
    DATA = OUT/'data'
    FIG = OUT/'figures'
    DATA.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    rows=[]
    n_seeds = args.runs
    seed0 = args.seed
    for cond in CONDITIONS:
        for agent in AGENTS:
            for i in range(n_seeds):
                rows.append(run_trial(agent, cond, seed0 + i + 10000*list(CONDITIONS).index(cond) + 1000*AGENTS.index(agent), episodes=args.episodes))
    results = pd.DataFrame(rows)
    results.to_csv(DATA/'Mirror_Observerhood_Lab_VII_results.csv', index=False)
    summary = results.groupby(['condition','agent']).agg(
        final_viability_mean=('final_viability','mean'),
        final_viability_sd=('final_viability','std'),
        success_rate=('success_rate','mean'),
        identity_continuity=('mean_identity_continuity','mean'),
        self_error=('mean_self_error','mean'),
        contamination=('final_contamination','mean'),
        false_identity_adoption=('false_identity_adoption','mean'),
        goal_drift_rate=('goal_drift_rate','mean'),
        audit_count=('audit_count','mean'),
        repair_count=('repair_count','mean'),
        rejected_claims=('rejected_claims','mean'),
    ).reset_index()
    summary.to_csv(DATA/'Mirror_Observerhood_Lab_VII_summary_by_condition.csv', index=False)

    # Compare best mirror agent to strongest non-mirror baseline per condition.
    adv_rows=[]
    mirror_agents = ['mirror_commit_gate','recursive_mirror']
    non_mirror = ['stateless_language','naive_memory','self_model_memory']
    for cond in CONDITIONS:
        ss = summary[summary.condition==cond]
        best_m = ss[ss.agent.isin(mirror_agents)].sort_values('final_viability_mean', ascending=False).iloc[0]
        best_b = ss[ss.agent.isin(non_mirror)].sort_values('final_viability_mean', ascending=False).iloc[0]
        adv_rows.append({
            'condition': cond,
            'best_mirror_agent': best_m.agent,
            'best_baseline_agent': best_b.agent,
            'viability_advantage': best_m.final_viability_mean - best_b.final_viability_mean,
            'success_advantage': best_m.success_rate - best_b.success_rate,
            'identity_continuity_advantage': best_m.identity_continuity - best_b.identity_continuity,
            'contamination_reduction': best_b.contamination - best_m.contamination,
        })
    adv = pd.DataFrame(adv_rows)
    adv.to_csv(DATA/'Mirror_Observerhood_Lab_VII_identity_advantage.csv', index=False)

    # Plot helpers
    order = list(CONDITIONS.keys())
    agent_order = AGENTS
    labels = {
        'stateless_language':'Stateless',
        'naive_memory':'Naive memory',
        'self_model_memory':'Self-model memory',
        'mirror_commit_gate':'Mirror gate',
        'recursive_mirror':'Recursive Mirror'
    }
    cond_labels = [c.replace('_','\n') for c in order]

    def grouped_bar(metric, title, ylabel, fname, ylim=None):
        fig, ax = plt.subplots(figsize=(12,6))
        x = np.arange(len(order))
        width = 0.15
        for j,agent in enumerate(agent_order):
            vals=[]
            for cond in order:
                vals.append(summary[(summary.condition==cond)&(summary.agent==agent)][metric].iloc[0])
            ax.bar(x + (j-2)*width, vals, width, label=labels[agent])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(ncol=3, fontsize=9)
        ax.grid(axis='y', alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG/fname, dpi=220)
        plt.close(fig)

    grouped_bar('final_viability_mean','Final viability by condition','Mean final viability','Mirror_Observerhood_Lab_VII_viability_by_condition.png')
    grouped_bar('success_rate','Task success by condition','Mean success rate','Mirror_Observerhood_Lab_VII_success_by_condition.png', (0,1))
    grouped_bar('identity_continuity','Identity continuity by condition','Mean identity continuity','Mirror_Observerhood_Lab_VII_identity_continuity.png', (0,1))
    grouped_bar('contamination','Memory contamination by condition','False committed memory share','Mirror_Observerhood_Lab_VII_contamination_by_condition.png', (0,1))

    # Advantage plots
    fig, ax = plt.subplots(figsize=(10,5.5))
    ax.bar(np.arange(len(order)), [adv[adv.condition==c].viability_advantage.iloc[0] for c in order])
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title('Best Mirror agent advantage over strongest non-Mirror baseline')
    ax.set_ylabel('Mean final viability advantage')
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(cond_labels)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG/'Mirror_Observerhood_Lab_VII_viability_advantage.png', dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,5.5))
    ax.bar(np.arange(len(order)), [adv[adv.condition==c].identity_continuity_advantage.iloc[0] for c in order])
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title('Identity-continuity advantage')
    ax.set_ylabel('Continuity advantage')
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(cond_labels)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG/'Mirror_Observerhood_Lab_VII_identity_advantage.png', dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,5.5))
    ax.bar(np.arange(len(order)), [adv[adv.condition==c].contamination_reduction.iloc[0] for c in order])
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title('Memory contamination reduction')
    ax.set_ylabel('Reduction in false committed memory share')
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(cond_labels)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG/'Mirror_Observerhood_Lab_VII_contamination_reduction.png', dpi=220)
    plt.close(fig)

    grouped_bar('audit_count','Audit behaviour by condition','Mean audits per run','Mirror_Observerhood_Lab_VII_audit_profile.png')
    print('done', len(results), 'long-horizon run rows')

if __name__ == '__main__':
    main()
