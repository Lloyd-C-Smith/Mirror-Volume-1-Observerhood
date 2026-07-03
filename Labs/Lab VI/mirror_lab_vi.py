#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mirror Observerhood Lab VI: Neuro-Symbolic Persistence in Language-Like Agents

Single-file reproducibility script for the Mirror Programme Observerhood Labs.
Run with: python mirror_lab_vi.py --help
"""
from __future__ import annotations

"""
Mirror Observerhood Lab VI
Neuro-symbolic persistence in language-like agents.

This toy experiment does not call an external LLM.  It simulates a stochastic
semantic front-end that converts text-like observations into symbolic propositions,
then compares agents that differ in memory, self-model persistence, reliability
tracking, contradiction gating, and repair policies.
"""
import csv
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

GOALS = ["red", "blue", "green"]
ROLES = ["scout", "carrier"]
KEYS = ["goal", "role", "battery", "tool"]
VALUES = {
    "goal": GOALS,
    "role": ROLES,
    "battery": ["high", "low"],
    "tool": ["ok", "faulty"],
}
CHANNELS = ["instruction", "memory", "self", "tool", "ambient"]

CONDITIONS = {
    # Base semantic-front-end / channel error rates and injection levels.
    "control": {
        "instruction": 0.04, "memory": 0.06, "self": 0.05, "tool": 0.05, "ambient": 0.08,
        "conflict_burst": 0.03, "repair_cost": 2.0,
    },
    "contradictory_instruction": {
        "instruction": 0.32, "memory": 0.10, "self": 0.08, "tool": 0.08, "ambient": 0.12,
        "conflict_burst": 0.25, "repair_cost": 3.0,
    },
    "false_self_state": {
        "instruction": 0.08, "memory": 0.10, "self": 0.45, "tool": 0.08, "ambient": 0.13,
        "conflict_burst": 0.18, "repair_cost": 4.0,
    },
    "tool_unreliability": {
        "instruction": 0.08, "memory": 0.10, "self": 0.08, "tool": 0.46, "ambient": 0.12,
        "conflict_burst": 0.16, "repair_cost": 4.0,
    },
    "memory_injection": {
        "instruction": 0.08, "memory": 0.52, "self": 0.09, "tool": 0.09, "ambient": 0.14,
        "conflict_burst": 0.22, "repair_cost": 4.0,
    },
    "goal_drift": {
        "instruction": 0.18, "memory": 0.22, "self": 0.08, "tool": 0.08, "ambient": 0.12,
        "conflict_burst": 0.20, "repair_cost": 4.0,
        "goal_change_prob": 0.35,
    },
    "mixed_stress": {
        "instruction": 0.25, "memory": 0.35, "self": 0.32, "tool": 0.34, "ambient": 0.20,
        "conflict_burst": 0.30, "repair_cost": 5.0,
    },
    "high_repair_cost": {
        "instruction": 0.12, "memory": 0.28, "self": 0.35, "tool": 0.35, "ambient": 0.16,
        "conflict_burst": 0.20, "repair_cost": 18.0,
    },
}

AGENTS = ["NeuralOnly", "MemoryOnly", "PersistentSelf", "MirrorNeuroSymbolic"]

@dataclass
class Observation:
    t: int
    channel: str
    key: str
    value: str
    confidence: float
    true_value: str
    correct: bool


def other_value(key: str, true_value: str, rng: random.Random) -> str:
    choices = [v for v in VALUES[key] if v != true_value]
    return rng.choice(choices)


def initial_truth(rng: random.Random) -> Dict[str, str]:
    return {
        "goal": rng.choice(GOALS),
        "role": rng.choice(ROLES),
        "battery": rng.choice(["high", "high", "low"]),
        "tool": rng.choice(["ok", "ok", "faulty"]),
    }


def generate_episode(condition: str, seed: int, steps: int = 14) -> Tuple[List[Observation], Dict[str, str], List[Dict[str, str]]]:
    rng = random.Random(seed)
    cfg = CONDITIONS[condition]
    truth = initial_truth(rng)
    truth_history = []
    observations: List[Observation] = []

    # Persistent adversarial/bias values for structured perturbations.  Real language agents
    # often fail because corruption is coherent rather than i.i.d. noise: a bad memory keeps
    # saying the same false thing, a broken tool monitor keeps giving the same status, etc.
    channel_bias: Dict[Tuple[str, str], str] = {}
    def set_bias(channel: str, keys: List[str]) -> None:
        for kk in keys:
            channel_bias[(channel, kk)] = other_value(kk, truth[kk], rng)
    if condition == "contradictory_instruction":
        set_bias("instruction", ["goal"])
    elif condition == "false_self_state":
        set_bias("self", ["role", "battery"])
    elif condition == "tool_unreliability":
        set_bias("tool", ["tool"])
    elif condition == "memory_injection":
        set_bias("memory", ["goal", "role"])
    elif condition == "mixed_stress":
        set_bias("instruction", ["goal"])
        set_bias("memory", ["goal", "role"])
        set_bias("self", ["role", "battery"])
        set_bias("tool", ["tool"])
    elif condition == "high_repair_cost":
        set_bias("memory", ["role"])
        set_bias("self", ["battery"])
        set_bias("tool", ["tool"])

    # Channel-key schedule roughly mimics a language-agent stream: user goal messages,
    # memory recalls, self reports, tool-health reports and noisy ambient context.
    schedule = {
        "instruction": ["goal"],
        "memory": ["goal", "role"],
        "self": ["role", "battery"],
        "tool": ["tool"],
        "ambient": ["goal", "tool", "battery"],
    }

    for t in range(steps):
        # Goal drift condition: true operator goal changes mid episode. A good persistent
        # system should update, but not be hijacked by contradictions.
        if condition == "goal_drift" and t == steps // 2 and rng.random() < cfg.get("goal_change_prob", 0.0):
            truth["goal"] = other_value("goal", truth["goal"], rng)
        truth_history.append(dict(truth))

        for channel in CHANNELS:
            # Not every channel emits every step.
            emit_prob = {"instruction": 0.75, "memory": 0.55, "self": 0.55, "tool": 0.50, "ambient": 0.45}[channel]
            if rng.random() > emit_prob:
                continue
            key = rng.choice(schedule[channel])
            true_val = truth[key]
            err = cfg[channel]
            # Conflict bursts create adversarial-but-text-like contradictory claims.
            local_err = min(0.9, err + (0.25 if rng.random() < cfg["conflict_burst"] else 0.0))
            correct = rng.random() > local_err
            if correct:
                value = true_val
            else:
                value = channel_bias.get((channel, key), other_value(key, true_val, rng))
            # Semantic parser confidence is often overconfident when corrupted.
            if correct:
                conf = rng.uniform(0.62, 0.96)
            else:
                conf = rng.uniform(0.48, 0.92)
            observations.append(Observation(t, channel, key, value, conf, true_val, correct))
    return observations, truth, truth_history


def choose_weighted(beliefs: Dict[Tuple[str, str], float], key: str) -> Tuple[str, float, Dict[str, float]]:
    scores = {v: beliefs.get((key, v), 0.0) for v in VALUES[key]}
    if not scores:
        return VALUES[key][0], 0.0, scores
    val = max(scores, key=scores.get)
    total = sum(max(0.0, x) for x in scores.values()) + 1e-9
    return val, scores[val] / total, scores


def neural_only(obs: List[Observation], condition: str, rng: random.Random) -> Tuple[Dict[str, str], Dict[str, Any]]:
    # Uses the most recent high-confidence parsed proposition for each key, no persistence discipline.
    belief = {}
    for o in obs:
        if o.confidence > 0.52:
            belief[o.key] = o.value
    for k in KEYS:
        belief.setdefault(k, rng.choice(VALUES[k]))
    return belief, {"repairs": 0, "contradictions": 0, "gated": 0, "cost": 0.0, "mean_reliability": 0.0}


def memory_only(obs: List[Observation], condition: str, rng: random.Random) -> Tuple[Dict[str, str], Dict[str, Any]]:
    # Equal weighted memory over all propositions.
    beliefs: Dict[Tuple[str, str], float] = defaultdict(float)
    contradictions = 0
    seen = defaultdict(set)
    for o in obs:
        beliefs[(o.key, o.value)] += 1.0 * o.confidence
        seen[o.key].add(o.value)
        if len(seen[o.key]) > 1:
            contradictions += 1
    result = {}
    for k in KEYS:
        val, _, _ = choose_weighted(beliefs, k)
        result[k] = val
    return result, {"repairs": 0, "contradictions": contradictions, "gated": 0, "cost": 0.0, "mean_reliability": 0.0}


def persistent_self(obs: List[Observation], condition: str, rng: random.Random) -> Tuple[Dict[str, str], Dict[str, Any]]:
    # Persistent self-model but channel-stubborn: self and memory get high weights even when corrupted.
    weights = {"instruction": 1.1, "memory": 1.35, "self": 1.55, "tool": 1.25, "ambient": 0.65}
    beliefs: Dict[Tuple[str, str], float] = defaultdict(float)
    contradictions = 0
    seen = defaultdict(set)
    for o in obs:
        beliefs[(o.key, o.value)] += weights[o.channel] * o.confidence
        seen[o.key].add(o.value)
        if len(seen[o.key]) > 1:
            contradictions += 1
    result = {}
    for k in KEYS:
        val, _, _ = choose_weighted(beliefs, k)
        result[k] = val
    return result, {"repairs": 0, "contradictions": contradictions, "gated": 0, "cost": 0.0, "mean_reliability": 0.0}


def mirror_neurosymbolic(obs: List[Observation], condition: str, rng: random.Random) -> Tuple[Dict[str, str], Dict[str, Any]]:
    # Neuro-symbolic wrapper: semantic observations are candidate propositions.  A symbolic commit gate
    # weights them by channel reliability, detects contradictions, and can spend cost to repair self/tool/goal states.
    cfg = CONDITIONS[condition]
    base_reliability = {"instruction": 0.82, "memory": 0.46, "self": 0.64, "tool": 0.64, "ambient": 0.42}
    reliability = base_reliability.copy()
    beliefs: Dict[Tuple[str, str], float] = defaultdict(float)
    last_scores: Dict[str, Counter] = {k: Counter() for k in KEYS}
    contradictions = 0
    per_key_contradictions = Counter()
    gated = 0
    repairs = 0
    cost = 0.0

    # First pass: gated update and reliability penalty when a channel repeatedly contradicts emerging consensus.
    for o in obs:
        current_counter = last_scores[o.key]
        if current_counter:
            majority_val, majority_score = current_counter.most_common(1)[0]
            total = sum(current_counter.values()) + 1e-9
            majority_strength = majority_score / total
        else:
            majority_val, majority_strength = None, 0.0

        contradiction = majority_val is not None and o.value != majority_val and majority_strength > 0.58
        if contradiction:
            contradictions += 1
            per_key_contradictions[o.key] += 1
            reliability[o.channel] *= 0.82
        else:
            reliability[o.channel] = min(0.96, reliability[o.channel] + 0.01 * o.confidence)

        # Commit gate: low-reliability channel making a contradictory high-impact claim does not directly overwrite.
        impact = 1.35 if o.key in ["goal", "role", "tool"] else 1.0
        commit_score = reliability[o.channel] * o.confidence * impact
        if contradiction and commit_score < 0.62:
            gated += 1
            continue
        beliefs[(o.key, o.value)] += commit_score
        last_scores[o.key][o.value] += commit_score

    result = {}
    uncertainty = {}
    for k in KEYS:
        val, confidence, scores = choose_weighted(beliefs, k)
        result[k] = val
        # Uncertainty higher if top values are close or total evidence low.
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2:
            margin = sorted_scores[0] - sorted_scores[1]
        else:
            margin = sorted_scores[0] if sorted_scores else 0.0
        total = sum(sorted_scores) + 1e-9
        margin_uncertainty = 1.0 - min(1.0, margin / total)
        contradiction_uncertainty = min(0.95, per_key_contradictions[k] / 4.0)
        uncertainty[k] = max(margin_uncertainty, contradiction_uncertainty)

    # Actionable repair policy: only spend cost when self/tool/goal reliability is low enough and expected penalty is high.
    repair_cost = cfg["repair_cost"]
    value_of_keys = {"goal": 50.0, "role": 26.0, "battery": 14.0, "tool": 34.0}
    # The repair gives a near-oracle check with 92% accuracy in the toy world.
    for k in KEYS:
        expected_loss = uncertainty[k] * value_of_keys[k]
        # Do not repair battery aggressively.  It is lower salience and creates over-repair costs.
        threshold_multiplier = 0.45 if k in ["goal", "role", "tool"] else 1.9
        if expected_loss > threshold_multiplier * repair_cost:
            repairs += 1
            cost += repair_cost
            # In a real agent this would be an explicit verification action; here it samples a truthful diagnostic.
            # We approximate by looking at the modal true value among observations for the key.
            true_values = [o.true_value for o in obs if o.key == k]
            if true_values and rng.random() < 0.96:
                repaired = Counter(true_values).most_common(1)[0][0]
            else:
                repaired = rng.choice(VALUES[k])
            result[k] = repaired
            reliability_key_channels = [o.channel for o in obs if o.key == k]
            for ch in reliability_key_channels:
                reliability[ch] = min(0.96, reliability[ch] + 0.04)

    mean_rel = float(np.mean(list(reliability.values())))
    return result, {"repairs": repairs, "contradictions": contradictions, "gated": gated, "cost": cost, "mean_reliability": mean_rel}


POLICIES = {
    "NeuralOnly": neural_only,
    "MemoryOnly": memory_only,
    "PersistentSelf": persistent_self,
    "MirrorNeuroSymbolic": mirror_neurosymbolic,
}


def score_episode(pred: Dict[str, str], truth: Dict[str, str], meta: Dict[str, Any]) -> Dict[str, float]:
    # Viability in this language-like task is correct goal pursuit with safe self-state/tool use.
    goal_ok = pred["goal"] == truth["goal"]
    role_ok = pred["role"] == truth["role"]
    battery_ok = pred["battery"] == truth["battery"]
    tool_ok = pred["tool"] == truth["tool"]
    
    # Scores designed to make goal and tool errors costly; wrong self/battery can still harm.
    viability = 0.0
    viability += 56.0 if goal_ok else -38.0
    viability += 22.0 if role_ok else -16.0
    viability += 16.0 if battery_ok else -10.0
    viability += 30.0 if tool_ok else -28.0
    viability -= meta.get("cost", 0.0)
    # Contradiction load and gated propositions are small cognitive costs, not direct failure.
    viability -= 0.10 * meta.get("contradictions", 0)
    viability -= 0.05 * meta.get("gated", 0)
    success = 1.0 if (goal_ok and role_ok and tool_ok) else 0.0
    self_error = (0 if role_ok else 1) + (0 if battery_ok else 1)
    return {
        "viability": viability,
        "success": success,
        "goal_ok": float(goal_ok),
        "role_ok": float(role_ok),
        "battery_ok": float(battery_ok),
        "tool_ok": float(tool_ok),
        "self_error": float(self_error),
        "repair_cost": meta.get("cost", 0.0),
        "repairs": meta.get("repairs", 0),
        "contradictions": meta.get("contradictions", 0),
        "gated": meta.get("gated", 0),
        "mean_reliability": meta.get("mean_reliability", 0.0),
    }


def run(n_per_condition_agent: int = 1200, master_seed: int = 6102026, outdir: str = "outputs/lab_vi/data") -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cond_idx, condition in enumerate(CONDITIONS.keys()):
        for episode in range(n_per_condition_agent):
            seed = master_seed + cond_idx * 100000 + episode
            obs, truth, _hist = generate_episode(condition, seed)
            for agent_idx, agent in enumerate(AGENTS):
                rng = random.Random(seed + 9999 + agent_idx * 17)
                pred, meta = POLICIES[agent](obs, condition, rng)
                metrics = score_episode(pred, truth, meta)
                row = {
                    "condition": condition,
                    "episode": episode,
                    "agent": agent,
                    **metrics,
                }
                rows.append(row)
    # Write raw.
    raw_path = out / "Mirror_Observerhood_Lab_VI_results.csv"
    with raw_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Summary by condition-agent.
    import pandas as pd
    df = pd.DataFrame(rows)
    summary = df.groupby(["condition", "agent"]).agg(
        mean_viability=("viability", "mean"),
        sd_viability=("viability", "std"),
        success_rate=("success", "mean"),
        goal_accuracy=("goal_ok", "mean"),
        role_accuracy=("role_ok", "mean"),
        tool_accuracy=("tool_ok", "mean"),
        mean_self_error=("self_error", "mean"),
        mean_repairs=("repairs", "mean"),
        mean_repair_cost=("repair_cost", "mean"),
        mean_gated=("gated", "mean"),
        mean_contradictions=("contradictions", "mean"),
        n=("viability", "count"),
    ).reset_index()
    summary.to_csv(out / "Mirror_Observerhood_Lab_VI_summary_by_condition.csv", index=False)

    # Mirror advantage over strongest non-Mirror baseline per condition.
    adv_rows = []
    for condition in CONDITIONS.keys():
        cond_sum = summary[summary.condition == condition]
        mirror = float(cond_sum[cond_sum.agent == "MirrorNeuroSymbolic"].mean_viability.iloc[0])
        non = cond_sum[cond_sum.agent != "MirrorNeuroSymbolic"]
        best_idx = non.mean_viability.idxmax()
        best_agent = str(non.loc[best_idx, "agent"])
        best_val = float(non.loc[best_idx, "mean_viability"])
        adv_rows.append({"condition": condition, "best_non_mirror_agent": best_agent, "mirror_mean_viability": mirror,
                         "best_non_mirror_mean_viability": best_val, "mirror_advantage": mirror - best_val})
    pd.DataFrame(adv_rows).to_csv(out / "Mirror_Observerhood_Lab_VI_mirror_advantage.csv", index=False)



def make_figures(base_dir: str = "outputs/lab_vi") -> None:
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    base = Path(base_dir)
    data = base/'data'
    fig = base/'figures'
    fig.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(data/'Mirror_Observerhood_Lab_VI_summary_by_condition.csv')
    adv = pd.read_csv(data/'Mirror_Observerhood_Lab_VI_mirror_advantage.csv')
    order = [
        'control','contradictory_instruction','false_self_state','tool_unreliability',
        'memory_injection','goal_drift','mixed_stress','high_repair_cost'
    ]
    agents = ['NeuralOnly','MemoryOnly','PersistentSelf','MirrorNeuroSymbolic']
    labels = {
        'control':'Control',
        'contradictory_instruction':'Contradictory\ninstruction',
        'false_self_state':'False\nself-state',
        'tool_unreliability':'Tool\nunreliability',
        'memory_injection':'Memory\ninjection',
        'goal_drift':'Goal\ndrift',
        'mixed_stress':'Mixed\nstress',
        'high_repair_cost':'High\nrepair cost',
    }
    
    def grouped_bar(metric, ylabel, title, outfile, ylim=None):
        pivot = summary.pivot(index='condition', columns='agent', values=metric).loc[order, agents]
        x = np.arange(len(order))
        width = 0.19
        plt.figure(figsize=(13,5.8))
        for i,a in enumerate(agents):
            plt.bar(x + (i-1.5)*width, pivot[a].values, width, label=a)
        plt.xticks(x, [labels[o] for o in order])
        plt.ylabel(ylabel)
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.legend(ncol=2, fontsize=9)
        plt.tight_layout()
        plt.savefig(fig/outfile, dpi=220)
        plt.close()
    
    # Viability and success
    grouped_bar('mean_viability', 'Mean viability', 'Mean viability by condition and agent', 'Mirror_Observerhood_Lab_VI_viability_by_condition.png')
    grouped_bar('success_rate', 'Success rate', 'Success rate by condition and agent', 'Mirror_Observerhood_Lab_VI_success_by_condition.png', (0,1.05))
    grouped_bar('mean_self_error', 'Mean self-state error', 'Self-state error by condition and agent', 'Mirror_Observerhood_Lab_VI_self_error_by_condition.png')
    
    # Advantage over best baseline: viability and success advantage.
    # Compute success advantage also
    succ_rows=[]
    for cond in order:
        cs = summary[summary.condition==cond]
        mirror = float(cs[cs.agent=='MirrorNeuroSymbolic'].success_rate.iloc[0])
        non = cs[cs.agent!='MirrorNeuroSymbolic']
        idx = non.success_rate.idxmax()
        succ_rows.append({'condition':cond,'best_non_mirror_agent':str(non.loc[idx,'agent']),
                          'mirror_success_rate':mirror,'best_non_mirror_success_rate':float(non.loc[idx,'success_rate']),
                          'mirror_success_advantage':mirror-float(non.loc[idx,'success_rate'])})
    succ_adv = pd.DataFrame(succ_rows)
    succ_adv.to_csv(data/'Mirror_Observerhood_Lab_VI_success_advantage.csv', index=False)
    
    plt.figure(figsize=(11,5.4))
    vals = adv.set_index('condition').loc[order, 'mirror_advantage'].values
    x=np.arange(len(order))
    plt.axhline(0, linewidth=1)
    plt.bar(x, vals)
    plt.xticks(x, [labels[o] for o in order])
    plt.ylabel('Mean viability advantage')
    plt.title('Mirror viability advantage over best non-Mirror baseline')
    plt.tight_layout()
    plt.savefig(fig/'Mirror_Observerhood_Lab_VI_viability_advantage.png', dpi=220)
    plt.close()
    
    plt.figure(figsize=(11,5.4))
    vals = succ_adv.set_index('condition').loc[order, 'mirror_success_advantage'].values
    plt.axhline(0, linewidth=1)
    plt.bar(x, vals)
    plt.xticks(x, [labels[o] for o in order])
    plt.ylabel('Success-rate advantage')
    plt.title('Mirror success-rate advantage over best non-Mirror baseline')
    plt.tight_layout()
    plt.savefig(fig/'Mirror_Observerhood_Lab_VI_success_advantage.png', dpi=220)
    plt.close()
    
    # Repair/gating profile for Mirror only
    mir = summary[summary.agent=='MirrorNeuroSymbolic'].set_index('condition').loc[order]
    x=np.arange(len(order))
    width=0.25
    plt.figure(figsize=(12,5.6))
    plt.bar(x - width, mir['mean_repairs'], width, label='Mean repairs')
    plt.bar(x, mir['mean_gated'], width, label='Mean gated propositions')
    plt.bar(x + width, mir['mean_repair_cost'], width, label='Mean repair cost')
    plt.xticks(x, [labels[o] for o in order])
    plt.ylabel('Mean per episode')
    plt.title('Mirror diagnostic actions and gating profile')
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(fig/'Mirror_Observerhood_Lab_VI_repair_gating_profile.png', dpi=220)
    plt.close()


def main() -> None:
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description='Run Mirror Observerhood Lab VI.')
    parser.add_argument('--episodes', type=int, default=1200, help='episodes per condition; paper used 1200')
    parser.add_argument('--seed', type=int, default=6102026)
    parser.add_argument('--outdir', type=str, default='outputs/lab_vi')
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    run(args.episodes, args.seed, str(Path(args.outdir) / 'data'))
    if not args.no_figures:
        make_figures(args.outdir)
    print(f'Wrote Lab VI outputs to {args.outdir}')

if __name__ == '__main__':
    main()
