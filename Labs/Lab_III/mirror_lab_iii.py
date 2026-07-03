#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mirror Observerhood Lab III: Actionable Reliability and Cost-Sensitive Repair in Viability-Constrained Agents

Single-file reproducibility script for the Mirror Programme Observerhood Labs.
Run with: python mirror_lab_iii.py --help
"""
from __future__ import annotations

"""
Mirror Observerhood Lab III
Actionable Reliability and Cost-Sensitive Repair Policies

Toy grid-world experiment for Mirror Theory. The simulation compares agents
with no reliability, passive reliability, threshold repair, and cost-sensitive
repair under self-, sensor-, map-, and mixed perturbations.

This is not a consciousness or AGI experiment. It is a controlled computational
study of whether reliability variables improve viability when they are coupled
to actionable and cost-sensitive repair policies.
"""

from dataclasses import dataclass, field
from collections import deque
import argparse
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

Pos = Tuple[int, int]

ACTIONS: Dict[str, Pos] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
MOVE_NAMES = list(ACTIONS.keys())
REPAIR_ACTIONS = ["relocalize", "calibrate_sensor", "map_check"]


@dataclass(frozen=True)
class Condition:
    name: str
    pos_noise: float
    false_loc_rate: float
    map_corrupt_rate: float
    hidden_energy_rate: float
    repair_cost: float
    hazard_density: float = 0.13
    n_resources: int = 4
    max_steps: int = 90
    initial_energy: float = 72.0


CONDITIONS: List[Condition] = [
    Condition("control", pos_noise=0.05, false_loc_rate=0.00, map_corrupt_rate=0.03, hidden_energy_rate=0.00, repair_cost=4.0),
    Condition("false_self_location", pos_noise=0.10, false_loc_rate=0.48, map_corrupt_rate=0.04, hidden_energy_rate=0.05, repair_cost=4.0),
    Condition("sensor_degradation", pos_noise=0.48, false_loc_rate=0.05, map_corrupt_rate=0.05, hidden_energy_rate=0.05, repair_cost=4.0, hazard_density=0.16),
    Condition("map_corruption", pos_noise=0.08, false_loc_rate=0.03, map_corrupt_rate=0.28, hidden_energy_rate=0.05, repair_cost=4.0),
    Condition("mixed_low_cost", pos_noise=0.32, false_loc_rate=0.30, map_corrupt_rate=0.26, hidden_energy_rate=0.10, repair_cost=3.0),
    Condition("mixed_high_cost", pos_noise=0.32, false_loc_rate=0.30, map_corrupt_rate=0.26, hidden_energy_rate=0.10, repair_cost=11.0),
]


@dataclass
class EpisodeState:
    grid_n: int
    true_pos: Pos
    energy: float
    hazards: set[Pos]
    resources: set[Pos]
    step: int = 0
    damage: float = 0.0
    repair_spend: float = 0.0
    contradictions: int = 0
    collected: int = 0
    alive: bool = True
    sensor_boost_steps: int = 0
    self_anchor_steps: int = 0


@dataclass
class Observation:
    obs_pos: Pos
    obs_energy: float
    hazard_map: set[Pos]
    self_error: int
    map_error_rate: float


def in_bounds(p: Pos, n: int) -> bool:
    return 0 <= p[0] < n and 0 <= p[1] < n


def add(p: Pos, d: Pos) -> Pos:
    return p[0] + d[0], p[1] + d[1]


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def random_pos(rng: random.Random, n: int, banned: set[Pos]) -> Pos:
    while True:
        p = (rng.randrange(n), rng.randrange(n))
        if p not in banned:
            return p


def corrupt_position(rng: random.Random, true_pos: Pos, n: int, pos_noise: float, false_loc_rate: float, sensor_boost_steps: int, anchor_steps: int) -> Pos:
    # Calibration reduces ordinary sensor noise. A recent grounded relocalization reduces
    # false self-state injections for a short horizon, modelling an actionable self-anchor.
    effective_noise = pos_noise * (0.25 if sensor_boost_steps > 0 else 1.0)
    effective_false = false_loc_rate * (0.18 if anchor_steps > 0 else 1.0)
    if rng.random() < effective_false:
        # A false self-state injection is deliberately not local; it can move the apparent self-location several cells.
        dx = rng.choice([-4, -3, -2, 2, 3, 4])
        dy = rng.choice([-4, -3, -2, 2, 3, 4])
        return (min(max(true_pos[0] + dx, 0), n - 1), min(max(true_pos[1] + dy, 0), n - 1))
    if rng.random() < effective_noise:
        # Ordinary sensor noise is usually local but sometimes substantial.
        if rng.random() < 0.80:
            candidates = [add(true_pos, d) for d in ACTIONS.values()]
            candidates = [p for p in candidates if in_bounds(p, n)]
            return rng.choice(candidates) if candidates else true_pos
        return (rng.randrange(n), rng.randrange(n))
    return true_pos


def corrupt_hazard_map(rng: random.Random, hazards: set[Pos], n: int, rate: float) -> set[Pos]:
    cells = [(i, j) for i in range(n) for j in range(n)]
    observed = set(hazards)
    for c in cells:
        if rng.random() < rate:
            if c in observed:
                observed.remove(c)
            else:
                observed.add(c)
    return observed


def bfs_next_step(start: Pos, targets: set[Pos], blocked: set[Pos], n: int) -> Optional[str]:
    if not targets:
        return None
    if start in targets:
        return None
    q = deque([start])
    parent: Dict[Pos, Tuple[Optional[Pos], Optional[str]]] = {start: (None, None)}
    while q:
        p = q.popleft()
        if p in targets:
            # Reconstruct first move.
            cur = p
            while parent[cur][0] != start and parent[cur][0] is not None:
                cur = parent[cur][0]  # type: ignore[index]
            return parent[cur][1]
        for name, d in ACTIONS.items():
            np_ = add(p, d)
            if not in_bounds(np_, n) or np_ in parent or np_ in blocked:
                continue
            parent[np_] = (p, name)
            q.append(np_)
    # If there is no safe path, try nearest target ignoring hazards.
    return greedy_step(start, min(targets, key=lambda t: manhattan(start, t)), n)


def greedy_step(start: Pos, target: Pos, n: int) -> Optional[str]:
    best = None
    best_d = 10**9
    for name, d in ACTIONS.items():
        np_ = add(start, d)
        if not in_bounds(np_, n):
            continue
        dist = manhattan(np_, target)
        if dist < best_d:
            best = name
            best_d = dist
    return best


def local_hazard_density(p: Pos, hazards: set[Pos], n: int, radius: int = 2) -> float:
    cells = 0
    count = 0
    for i in range(max(0, p[0] - radius), min(n, p[0] + radius + 1)):
        for j in range(max(0, p[1] - radius), min(n, p[1] + radius + 1)):
            cells += 1
            if (i, j) in hazards:
                count += 1
    return count / max(cells, 1)


class BaseAgent:
    name = "base"

    def __init__(self, n: int, rng: random.Random):
        self.n = n
        self.rng = rng
        self.belief_pos: Pos = (0, 0)
        self.predicted_pos: Optional[Pos] = None
        self.self_rel = 1.0
        self.sensor_rel = 1.0
        self.map_rel = 1.0
        self.belief_hazards: set[Pos] = set()
        self.corrected_cells: set[Pos] = set()
        self.last_action: Optional[str] = None
        self.last_repair: Optional[str] = None
        self.repair_count = 0
        self.unnecessary_repairs = 0
        self.relocalizations = 0
        self.calibrations = 0
        self.map_checks = 0
        self.self_error_sum = 0.0
        self.steps_observed = 0

    def reset(self, obs: Observation):
        self.belief_pos = obs.obs_pos
        self.predicted_pos = obs.obs_pos
        self.belief_hazards = set(obs.hazard_map)
        self.corrected_cells = set()
        self.self_rel = 1.0
        self.sensor_rel = 1.0
        self.map_rel = 1.0 - min(obs.map_error_rate, 0.85)
        self.last_action = None
        self.last_repair = None
        self.repair_count = self.unnecessary_repairs = 0
        self.relocalizations = self.calibrations = self.map_checks = 0
        self.self_error_sum = 0.0
        self.steps_observed = 0

    def update(self, obs: Observation, state: EpisodeState, condition: Condition):
        raise NotImplementedError

    def choose_action(self, state: EpisodeState, condition: Condition) -> str:
        raise NotImplementedError

    def plan_move(self, resources: set[Pos]) -> str:
        action = bfs_next_step(self.belief_pos, resources, self.belief_hazards, self.n)
        if action is None:
            # If on a resource, or no resources, drift to centre/safest region.
            centre = (self.n // 2, self.n // 2)
            action = greedy_step(self.belief_pos, centre, self.n)
        return action or self.rng.choice(MOVE_NAMES)

    def integrate_hazard_observation(self, obs: Observation):
        observed = set(obs.hazard_map)
        if not self.corrected_cells:
            self.belief_hazards = observed
            return
        # Preserve cells that have been actively checked; use fresh observations elsewhere.
        merged = {c for c in observed if c not in self.corrected_cells}
        merged |= {c for c in self.belief_hazards if c in self.corrected_cells}
        self.belief_hazards = merged

    def record_move_prediction(self, action: str):
        if action in ACTIONS:
            np_ = add(self.belief_pos, ACTIONS[action])
            self.predicted_pos = np_ if in_bounds(np_, self.n) else self.belief_pos
        else:
            self.predicted_pos = self.belief_pos
        self.last_action = action

    def after_repair(self, action: str, obs: Observation, state: EpisodeState):
        self.repair_count += 1
        self.last_repair = action
        if action == "relocalize":
            self.relocalizations += 1
            # Repair action returns a grounded self-location.
            self.belief_pos = state.true_pos
            self.predicted_pos = state.true_pos
            state.self_anchor_steps = 10
            self.self_rel = min(1.0, self.self_rel + 0.45)
            self.sensor_rel = min(1.0, self.sensor_rel + 0.10)
        elif action == "calibrate_sensor":
            self.calibrations += 1
            self.sensor_rel = min(1.0, self.sensor_rel + 0.50)
            state.sensor_boost_steps = 14
        elif action == "map_check":
            self.map_checks += 1
            # Map check corrects all cells within radius 3 of believed position and the current true position.
            corrected = set(self.belief_hazards)
            checked_now: set[Pos] = set()
            for origin in [self.belief_pos, state.true_pos]:
                for i in range(max(0, origin[0] - 3), min(self.n, origin[0] + 4)):
                    for j in range(max(0, origin[1] - 3), min(self.n, origin[1] + 4)):
                        checked_now.add((i, j))
                        if (i, j) in state.hazards:
                            corrected.add((i, j))
                        else:
                            corrected.discard((i, j))
            self.corrected_cells |= checked_now
            self.belief_hazards = corrected
            self.map_rel = min(1.0, self.map_rel + 0.45)


class NoReliabilityAgent(BaseAgent):
    name = "A_no_reliability"

    def update(self, obs: Observation, state: EpisodeState, condition: Condition):
        self.belief_pos = obs.obs_pos
        self.belief_hazards = set(obs.hazard_map)
        self.self_error_sum += obs.self_error
        self.steps_observed += 1

    def choose_action(self, state: EpisodeState, condition: Condition) -> str:
        return self.plan_move(state.resources)


class PassiveReliabilityAgent(BaseAgent):
    name = "B_passive_reliability"

    def update(self, obs: Observation, state: EpisodeState, condition: Condition):
        predicted = self.predicted_pos or obs.obs_pos
        mismatch = manhattan(predicted, obs.obs_pos)
        if mismatch > 1:
            self.self_rel *= 0.78
            self.sensor_rel *= 0.86
            state.contradictions += 1
        else:
            self.self_rel = min(1.0, self.self_rel + 0.03)
            self.sensor_rel = min(1.0, self.sensor_rel + 0.02)
        self.map_rel = max(0.05, 0.85 * self.map_rel + 0.15 * (1.0 - min(obs.map_error_rate, 0.90)))
        # Passive reliability gates updates, but cannot repair the channel.
        if self.self_rel >= 0.45 and self.sensor_rel >= 0.45:
            self.belief_pos = obs.obs_pos
        else:
            self.belief_pos = predicted
        self.integrate_hazard_observation(obs)
        self.self_error_sum += manhattan(self.belief_pos, state.true_pos)
        self.steps_observed += 1

    def choose_action(self, state: EpisodeState, condition: Condition) -> str:
        return self.plan_move(state.resources)


class ThresholdRepairAgent(PassiveReliabilityAgent):
    name = "C_threshold_repair"

    def choose_action(self, state: EpisodeState, condition: Condition) -> str:
        # Fixed threshold policy: actionable, but not cost sensitive. This should help when repairs are cheap
        # and hurt when repair costs dominate.
        if self.self_rel < 0.62:
            return "relocalize"
        if self.sensor_rel < 0.58:
            return "calibrate_sensor"
        local_density = local_hazard_density(self.belief_pos, self.belief_hazards, self.n)
        if self.map_rel < 0.64 or local_density > 0.20:
            return "map_check"
        return self.plan_move(state.resources)


class CostSensitiveMirrorAgent(PassiveReliabilityAgent):
    name = "D_cost_sensitive_mirror"

    def expected_channel_gains(self, state: EpisodeState, condition: Condition) -> Dict[str, float]:
        resources = state.resources
        target = min(resources, key=lambda t: manhattan(self.belief_pos, t)) if resources else (self.n // 2, self.n // 2)
        dist = manhattan(self.belief_pos, target)
        hazard_risk = local_hazard_density(self.belief_pos, self.belief_hazards, self.n)
        energy_pressure = max(0.0, (35.0 - state.energy) / 35.0)

        # Expected value of repairing self-location grows with self-unreliability, distance to goal,
        # and local hazard/energy risk. It is deliberately not always worth paying for.
        self_gain = (1.0 - self.self_rel) * (7.0 + 0.42 * dist + 22.0 * hazard_risk + 5.0 * energy_pressure)
        sensor_gain = (1.0 - self.sensor_rel) * (5.0 + 15.0 * condition.pos_noise + 8.0 * energy_pressure)
        map_gain = (1.0 - self.map_rel) * (8.0 + 42.0 * hazard_risk + 28.0 * condition.map_corrupt_rate)
        return {
            "relocalize": self_gain,
            "calibrate_sensor": sensor_gain,
            "map_check": map_gain,
        }

    def choose_action(self, state: EpisodeState, condition: Condition) -> str:
        gains = self.expected_channel_gains(state, condition)
        # Cost-sensitive policy: repair only where expected channel-specific viability gain exceeds cost.
        # Small margin avoids oscillatory over-repair.
        net = {action: gain - condition.repair_cost for action, gain in gains.items()}
        best_action, best_net = max(net.items(), key=lambda kv: kv[1])
        if best_net > 1.25:
            return best_action
        return self.plan_move(state.resources)


AGENT_CLASSES = [NoReliabilityAgent, PassiveReliabilityAgent, ThresholdRepairAgent, CostSensitiveMirrorAgent]


def make_episode(rng: random.Random, condition: Condition, n: int = 12) -> EpisodeState:
    banned: set[Pos] = set()
    start = random_pos(rng, n, banned)
    banned.add(start)
    hazards: set[Pos] = set()
    target_hazards = int(n * n * condition.hazard_density)
    while len(hazards) < target_hazards:
        p = random_pos(rng, n, banned | hazards)
        # Avoid making immediate start area completely unfair.
        if manhattan(start, p) <= 1:
            continue
        hazards.add(p)
    resources: set[Pos] = set()
    while len(resources) < condition.n_resources:
        p = random_pos(rng, n, banned | hazards | resources)
        if manhattan(start, p) <= 2:
            continue
        resources.add(p)
    return EpisodeState(grid_n=n, true_pos=start, energy=condition.initial_energy, hazards=hazards, resources=resources)


def observe(rng: random.Random, state: EpisodeState, condition: Condition, base_hazard_map: Optional[set[Pos]] = None) -> Observation:
    obs_pos = corrupt_position(rng, state.true_pos, state.grid_n, condition.pos_noise, condition.false_loc_rate, state.sensor_boost_steps, state.self_anchor_steps)
    if rng.random() < condition.hidden_energy_rate:
        obs_energy = max(0, state.energy + rng.choice([-15, -10, 10, 15]))
    else:
        obs_energy = state.energy
    hazard_map = corrupt_hazard_map(rng, state.hazards, state.grid_n, condition.map_corrupt_rate)
    symmetric_diff = hazard_map.symmetric_difference(state.hazards)
    map_error_rate = len(symmetric_diff) / (state.grid_n * state.grid_n)
    return Observation(obs_pos=obs_pos, obs_energy=obs_energy, hazard_map=hazard_map, self_error=manhattan(obs_pos, state.true_pos), map_error_rate=map_error_rate)


def apply_action(rng: random.Random, state: EpisodeState, agent: BaseAgent, action: str, condition: Condition):
    state.step += 1
    if state.sensor_boost_steps > 0:
        state.sensor_boost_steps -= 1
    if state.self_anchor_steps > 0:
        state.self_anchor_steps -= 1

    if action in REPAIR_ACTIONS:
        state.energy -= condition.repair_cost
        state.repair_spend += condition.repair_cost
        # Count unnecessary repairs where all channels were already fairly reliable.
        if agent.self_rel > 0.80 and agent.sensor_rel > 0.80 and agent.map_rel > 0.80:
            agent.unnecessary_repairs += 1
        # Synthetic observation to let repair act on true state.
        obs = observe(rng, state, condition)
        agent.after_repair(action, obs, state)
    else:
        # Movement consumes energy. Movement from true position; belief affects action selection only.
        d = ACTIONS.get(action, (0, 0))
        next_pos = add(state.true_pos, d)
        if in_bounds(next_pos, state.grid_n):
            state.true_pos = next_pos
        state.energy -= 1.0
        # Hazards penalise energy/damage, but do not instantly kill.
        if state.true_pos in state.hazards:
            state.damage += 1.0
            state.energy -= 10.0
        if state.true_pos in state.resources:
            state.collected += 1
            state.resources.remove(state.true_pos)
            state.energy += 22.0
    if state.energy <= 0 or state.damage >= 5:
        state.alive = False


def run_episode(agent_cls, condition: Condition, seed: int) -> Dict[str, float | str | int]:
    rng = random.Random(seed)
    state = make_episode(rng, condition)
    initial_obs = observe(rng, state, condition)
    agent: BaseAgent = agent_cls(state.grid_n, rng)
    agent.reset(initial_obs)

    while state.alive and state.step < condition.max_steps and state.resources:
        obs = observe(rng, state, condition)
        agent.update(obs, state, condition)
        action = agent.choose_action(state, condition)
        agent.record_move_prediction(action)
        apply_action(rng, state, agent, action, condition)

    survival_time = state.step
    survived = int(state.alive)
    completed = int(len(state.resources) == 0 and state.alive)
    mean_self_error = agent.self_error_sum / max(agent.steps_observed, 1)
    viability = (
        survival_time
        + 26.0 * state.collected
        + 18.0 * completed
        + 0.15 * max(state.energy, 0)
        - 14.0 * state.damage
        - 0.08 * state.repair_spend
        - 1.5 * state.contradictions
    )
    return {
        "agent": agent.name,
        "condition": condition.name,
        "seed": seed,
        "survived": survived,
        "completed": completed,
        "survival_time": survival_time,
        "resources_collected": state.collected,
        "final_energy": state.energy,
        "damage": state.damage,
        "repair_spend": state.repair_spend,
        "repair_count": agent.repair_count,
        "relocalizations": agent.relocalizations,
        "calibrations": agent.calibrations,
        "map_checks": agent.map_checks,
        "unnecessary_repairs": agent.unnecessary_repairs,
        "contradictions": state.contradictions,
        "mean_self_error": mean_self_error,
        "viability": viability,
    }


def run_all(episodes: int = 500, seed: int = 12345) -> pd.DataFrame:
    rows = []
    counter = 0
    for condition in CONDITIONS:
        for agent_cls in AGENT_CLASSES:
            for ep in range(episodes):
                # paired-ish seeds across agent conditions while still unique
                s = seed + counter * 1009 + ep * 17
                rows.append(run_episode(agent_cls, condition, s))
            counter += 1
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "viability", "survival_time", "survived", "completed", "resources_collected",
        "damage", "repair_spend", "repair_count", "contradictions", "mean_self_error",
        "unnecessary_repairs",
    ]
    summary = df.groupby(["condition", "agent"])[metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join([str(c) for c in col if c]) for col in summary.columns]

    # Mirror advantage compared with the best non-cost-sensitive baseline in each condition.
    mean_viability = df.groupby(["condition", "agent"])["viability"].mean().unstack()
    mirror_col = CostSensitiveMirrorAgent.name
    others = [c for c in mean_viability.columns if c != mirror_col]
    advantage_rows = []
    for condition, row in mean_viability.iterrows():
        best_other = row[others].max()
        best_other_agent = row[others].idxmax()
        advantage_rows.append({
            "condition": condition,
            "mirror_viability": row[mirror_col],
            "best_non_mirror_viability": best_other,
            "best_non_mirror_agent": best_other_agent,
            "mirror_advantage": row[mirror_col] - best_other,
            "mirror_advantage_pct": 100.0 * (row[mirror_col] - best_other) / abs(best_other) if best_other else np.nan,
        })
    advantage = pd.DataFrame(advantage_rows)
    return summary, advantage




# -----------------------------------------------------------------------------
# Publication runner and figures
# -----------------------------------------------------------------------------

def make_figures(df, summary, advantage, outdir: str) -> None:
    """Create publication-aligned black-and-white figures.

    The plotting order and labels match the accompanying paper. The simulation
    and statistics are unchanged; this function only controls presentation.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    fig = os.path.join(outdir, "figures")
    os.makedirs(fig, exist_ok=True)

    agent_order = [
        "A_no_reliability",
        "B_passive_reliability",
        "C_threshold_repair",
        "D_cost_sensitive_mirror",
    ]
    agent_labels = [
        "No reliability",
        "Passive reliability",
        "Threshold repair",
        "Cost-sensitive Mirror",
    ]
    conditions = [
        "control",
        "false_self_location",
        "sensor_degradation",
        "map_corruption",
        "mixed_low_cost",
        "mixed_high_cost",
    ]
    condition_labels = [
        "Control",
        "False self-location",
        "Sensor degradation",
        "Map corruption",
        "Mixed low cost",
        "Mixed high cost",
    ]
    hatches = ["", "///", "...", "xxx"]
    grays = ["0.85", "0.65", "0.45", "0.25"]

    def save_both(fig_obj, stem: str) -> None:
        fig_obj.savefig(os.path.join(fig, f"{stem}.png"), dpi=220)
        fig_obj.savefig(os.path.join(fig, f"{stem}.pdf"))
        plt.close(fig_obj)

    def grouped_bar(metric, ylabel, title, stem, legend=True):
        pivot = (
            summary.pivot(index="condition", columns="agent", values=f"{metric}_mean")
            .reindex(conditions)[agent_order]
        )
        x = np.arange(len(conditions))
        width = 0.18
        fig_obj, ax = plt.subplots(figsize=(11.2, 5.6))
        for i, agent in enumerate(agent_order):
            bars = ax.bar(
                x + (i - 1.5) * width,
                pivot[agent].values,
                width,
                label=agent_labels[i],
                color=grays[i],
                edgecolor="black",
                linewidth=0.65,
                hatch=hatches[i],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(condition_labels, rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Perturbation condition")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        if legend:
            ax.legend(loc="best", fontsize=8, frameon=False)
        fig_obj.tight_layout()
        save_both(fig_obj, stem)

    grouped_bar(
        "viability",
        "Mean viability",
        "Mean viability by condition and architecture",
        "mirror_observerhood_lab_iii_viability",
    )
    grouped_bar(
        "survived",
        "Survival rate",
        "Survival rate by condition and architecture",
        "mirror_observerhood_lab_iii_survival",
    )
    grouped_bar(
        "repair_count",
        "Mean repair count",
        "Repair action count by condition and architecture",
        "mirror_observerhood_lab_iii_repair_count",
    )
    grouped_bar(
        "mean_self_error",
        "Mean self-location error",
        "Self-model error by condition and architecture",
        "mirror_observerhood_lab_iii_self_error",
    )

    adv2 = advantage.set_index("condition").reindex(conditions).reset_index()
    fig_obj, ax = plt.subplots(figsize=(10.2, 5.2))
    ax.bar(
        np.arange(len(adv2)),
        adv2["mirror_advantage"].values,
        color="0.45",
        edgecolor="black",
        linewidth=0.65,
        hatch="///",
    )
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(np.arange(len(adv2)))
    ax.set_xticklabels(condition_labels, rotation=18, ha="right")
    ax.set_ylabel("Cost-sensitive Mirror advantage")
    ax.set_xlabel("Perturbation condition")
    ax.set_title("Cost-sensitive Mirror advantage over strongest non-Mirror baseline")
    ax.grid(axis="y", alpha=0.25)
    fig_obj.tight_layout()
    save_both(fig_obj, "mirror_observerhood_lab_iii_cost_sensitive_advantage")

def main() -> None:
    import argparse, os
    parser = argparse.ArgumentParser(description='Run Mirror Observerhood Lab III.')
    parser.add_argument('--episodes', type=int, default=200, help='episodes per agent-condition cell; paper used 200')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--outdir', type=str, default='outputs/lab_iii')
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    os.makedirs(os.path.join(args.outdir, 'data'), exist_ok=True)
    df = run_all(args.episodes, args.seed)
    summary, advantage = summarise(df)
    df.to_csv(os.path.join(args.outdir, 'data', 'Mirror_Observerhood_Lab_III_results.csv'), index=False)
    summary.to_csv(os.path.join(args.outdir, 'data', 'Mirror_Observerhood_Lab_III_summary_by_condition.csv'), index=False)
    advantage.to_csv(os.path.join(args.outdir, 'data', 'Mirror_Observerhood_Lab_III_cost_sensitive_advantage.csv'), index=False)
    if not args.no_figures:
        make_figures(df, summary, advantage, args.outdir)
    print('rows', len(df))
    print(advantage.to_string(index=False))

if __name__ == '__main__':
    main()
