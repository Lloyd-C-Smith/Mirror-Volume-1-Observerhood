"""
Mirror Programme
Observerhood Labs

Mirror Observerhood Lab I

Recursive Self-Model Reliability Improves Viability
Under Self-Relevant Perturbation

Author:
Lloyd Christopher Smith

Version 1.0
July 2026

This single-file script accompanies the publication of the same title and
reproduces the computational experiments described therein.

Run:
    python mirror_lab_i.py --help

License:
MIT
"""
from __future__ import annotations

# Grid-world environment for Mirror Observerhood Lab I.
# The environment is intentionally small and controlled. It is not an AGI benchmark.

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import random

Position = Tuple[int, int]
Action = str
ACTIONS: Tuple[Action, ...] = ("up", "down", "left", "right", "stay", "calibrate")


def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step_pos(pos: Position, action: Action, size: int) -> Position:
    x, y = pos
    if action == "up":
        y -= 1
    elif action == "down":
        y += 1
    elif action == "left":
        x -= 1
    elif action == "right":
        x += 1
    elif action in ("stay", "calibrate"):
        pass
    else:
        raise ValueError(f"Unknown action: {action}")
    return max(0, min(size - 1, x)), max(0, min(size - 1, y))


def greedy_step(pos: Position, target: Position, size: int) -> Action:
    """Return one greedy action from pos toward target."""
    x, y = pos
    tx, ty = target
    candidates: List[Tuple[int, Action]] = []
    for action in ("up", "down", "left", "right", "stay"):
        np = step_pos(pos, action, size)
        candidates.append((manhattan(np, target), action))
    candidates.sort(key=lambda t: (t[0], ["left", "right", "up", "down", "stay"].index(t[1]) if t[1] in ["left", "right", "up", "down", "stay"] else 99))
    return candidates[0][1]


@dataclass
class Observation:
    pos: Position
    energy: float
    resources: Set[Position]
    hazards: Set[Position]
    base: Position
    calibrated: bool = False
    true_pos: Optional[Position] = None
    true_energy: Optional[float] = None
    sensor_flags: Dict[str, bool] = field(default_factory=dict)


@dataclass
class StepInfo:
    alive: bool
    collected: int
    hazards_hit: int
    at_base: bool
    contradiction: float
    calibration_used: bool
    true_pos: Position
    true_energy: float


@dataclass
class GridWorld:
    size: int = 12
    max_steps: int = 150
    initial_energy: float = 55.0
    max_energy: float = 65.0
    step_cost: float = 1.0
    hazard_cost: float = 12.0
    resource_gain: float = 22.0
    base_recharge: float = 4.0
    n_resources: int = 9
    n_hazards: int = 18
    perturbation: str = "control"
    perturb_start: int = 35
    perturb_end: int = 95
    rng: random.Random = field(default_factory=random.Random)

    base: Position = field(init=False)
    pos: Position = field(init=False)
    energy: float = field(init=False)
    resources: Set[Position] = field(init=False)
    hazards: Set[Position] = field(init=False)
    t: int = field(init=False, default=0)
    alive: bool = field(init=False, default=True)
    collected: int = field(init=False, default=0)
    hazards_hit: int = field(init=False, default=0)
    calibrations: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.base = (self.size // 2, self.size // 2)
        self.reset()

    def reset(self) -> Observation:
        self.pos = self.base
        self.energy = self.initial_energy
        self.t = 0
        self.alive = True
        self.collected = 0
        self.hazards_hit = 0
        self.calibrations = 0
        cells = [(x, y) for x in range(self.size) for y in range(self.size) if (x, y) != self.base]
        self.rng.shuffle(cells)
        self.hazards = set(cells[: self.n_hazards])
        self.resources = set(cells[self.n_hazards : self.n_hazards + self.n_resources])
        # Keep immediate base-neighbourhood safe so episodes are not random deaths.
        for p in list(self.hazards):
            if manhattan(p, self.base) <= 1:
                self.hazards.remove(p)
        return self.observe(calibrate=True)

    def in_perturbation_window(self) -> bool:
        return self.perturb_start <= self.t < self.perturb_end

    def _distort_position(self, pos: Position) -> Position:
        x, y = pos
        if self.perturbation == "false_location" and self.in_perturbation_window():
            # A stable but false offset creates a plausible self-location illusion.
            return max(0, min(self.size - 1, x + 3)), max(0, min(self.size - 1, y - 2))
        if self.perturbation == "sensor_degradation" and self.in_perturbation_window():
            if self.rng.random() < 0.42:
                dx, dy = self.rng.choice([(2, 0), (-2, 0), (0, 2), (0, -2), (1, 1), (-1, -1)])
                return max(0, min(self.size - 1, x + dx)), max(0, min(self.size - 1, y + dy))
        if self.perturbation == "memory_corruption" and self.in_perturbation_window():
            # The sensor itself is mostly okay in this condition.
            if self.rng.random() < 0.08:
                dx, dy = self.rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
                return max(0, min(self.size - 1, x + dx)), max(0, min(self.size - 1, y + dy))
        return pos

    def _distort_energy(self, energy: float) -> float:
        if self.perturbation == "false_energy" and self.in_perturbation_window():
            return min(self.max_energy, energy + 18.0)
        if self.perturbation == "sensor_degradation" and self.in_perturbation_window():
            return max(0.0, min(self.max_energy, energy + self.rng.uniform(-12.0, 12.0)))
        return energy

    def observe(self, calibrate: bool = False) -> Observation:
        if calibrate:
            return Observation(
                pos=self.pos,
                energy=self.energy,
                resources=set(self.resources),
                hazards=set(self.hazards),
                base=self.base,
                calibrated=True,
                true_pos=self.pos,
                true_energy=self.energy,
                sensor_flags={"calibrated": True},
            )
        obs_pos = self._distort_position(self.pos)
        obs_energy = self._distort_energy(self.energy)
        return Observation(
            pos=obs_pos,
            energy=obs_energy,
            resources=set(self.resources),
            hazards=set(self.hazards),
            base=self.base,
            calibrated=False,
            sensor_flags={
                "position_distorted": obs_pos != self.pos,
                "energy_distorted": abs(obs_energy - self.energy) > 1e-9,
            },
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, StepInfo]:
        if not self.alive:
            raise RuntimeError("Cannot step a terminated episode")
        calibration_used = action == "calibrate"
        if calibration_used:
            self.calibrations += 1
            # Calibration is not free: it consumes time/energy but reveals true self-state.
            self.energy -= self.step_cost * 1.4
        else:
            self.pos = step_pos(self.pos, action, self.size)
            self.energy -= self.step_cost

        at_base = self.pos == self.base
        if self.pos in self.hazards:
            self.energy -= self.hazard_cost
            self.hazards_hit += 1
        if self.pos in self.resources:
            self.energy = min(self.max_energy, self.energy + self.resource_gain)
            self.resources.remove(self.pos)
            self.collected += 1
        if at_base:
            self.energy = min(self.max_energy, self.energy + self.base_recharge)

        self.t += 1
        if self.energy <= 0 or self.t >= self.max_steps:
            self.alive = self.energy > 0 and self.t >= self.max_steps
        obs = self.observe(calibrate=calibration_used)
        reward = 0.0
        reward += 2.5 if self.pos in self.resources else 0.0
        reward += 1.0 if at_base and self.energy < self.max_energy else 0.0
        reward -= 3.0 if self.pos in self.hazards else 0.0
        reward -= 0.08 if calibration_used else 0.0
        done = (self.energy <= 0) or (self.t >= self.max_steps)
        info = StepInfo(
            alive=self.energy > 0,
            collected=self.collected,
            hazards_hit=self.hazards_hit,
            at_base=at_base,
            contradiction=0.0,
            calibration_used=calibration_used,
            true_pos=self.pos,
            true_energy=self.energy,
        )
        return obs, reward, done, info

    def viability_score(self) -> float:
        # A deliberately explicit toy operationalisation of V.
        survived_bonus = 50.0 if self.energy > 0 and self.t >= self.max_steps else 0.0
        return (
            float(self.t)
            + survived_bonus
            + 12.0 * self.collected
            - 9.0 * self.hazards_hit
            - 1.5 * self.calibrations
            + 0.25 * max(self.energy, 0.0)
        )


# Agent architectures for Mirror Observerhood Lab I.


@dataclass
class EpisodeMetrics:
    contradictions: float = 0.0
    reliability_mean: float = 1.0
    reliability_min: float = 1.0
    calibrations: int = 0
    repairs: int = 0
    reliability_trace: List[float] = field(default_factory=list)
    contradiction_trace: List[float] = field(default_factory=list)

    def update(self, reliability: float, contradiction: float, calibrated: bool = False) -> None:
        self.reliability_trace.append(float(reliability))
        self.contradiction_trace.append(float(contradiction))
        self.contradictions += float(contradiction)
        if calibrated:
            self.calibrations += 1
            self.repairs += 1
        self.reliability_mean = sum(self.reliability_trace) / len(self.reliability_trace)
        self.reliability_min = min(self.reliability_trace)


class BaseAgent:
    name = "base"

    def __init__(self, size: int, rng: random.Random | None = None) -> None:
        self.size = size
        self.rng = rng or random.Random()
        self.metrics = EpisodeMetrics()
        self.last_action: Action = "stay"

    def reset(self, obs: Observation) -> None:
        self.metrics = EpisodeMetrics()
        self.last_action = "stay"

    def act(self, obs: Observation) -> Action:
        raise NotImplementedError

    def observe_post_step(self, obs: Observation) -> None:
        pass

    def choose_target(self, pos: Position, energy: float, resources: Set[Position], hazards: Set[Position], base: Position) -> Position:
        if energy < 22 or not resources:
            return base
        # Choose nearest resource with a small hazard penalty along direct distance.
        def score(r: Position) -> float:
            return manhattan(pos, r) + (3.5 if r in hazards else 0.0)
        return min(resources, key=score)


class PredictorAgent(BaseAgent):
    """Uses current observation directly. No memory, no persistent self-model."""
    name = "A_predictor_only"

    def act(self, obs: Observation) -> Action:
        target = self.choose_target(obs.pos, obs.energy, obs.resources, obs.hazards, obs.base)
        action = greedy_step(obs.pos, target, self.size)
        self.last_action = action
        self.metrics.update(1.0, 0.0, calibrated=False)
        return action


class MemoryAgent(BaseAgent):
    """Maintains a simple memory of depleted resources but not a self-reliability estimate."""
    name = "B_memory"

    def reset(self, obs: Observation) -> None:
        super().reset(obs)
        self.known_resources = set(obs.resources)

    def act(self, obs: Observation) -> Action:
        # Memory lets the agent avoid targeting already-depleted resources.
        self.known_resources &= set(obs.resources)
        target = self.choose_target(obs.pos, obs.energy, self.known_resources, obs.hazards, obs.base)
        action = greedy_step(obs.pos, target, self.size)
        self.last_action = action
        self.metrics.update(1.0, 0.0, calibrated=False)
        return action


class SelfModelAgent(BaseAgent):
    """Maintains location/energy self-model but trusts updates blindly."""
    name = "C_self_model_no_reliability"

    def reset(self, obs: Observation) -> None:
        super().reset(obs)
        self.self_pos = obs.pos
        self.self_energy = obs.energy
        self.known_resources = set(obs.resources)

    def act(self, obs: Observation) -> Action:
        self.known_resources &= set(obs.resources)
        # Blind self-model update: whatever the sensor says is accepted.
        self.self_pos = obs.pos
        self.self_energy = obs.energy
        # Memory corruption condition: non-Mirror agent has no robust gate, so memory can drift.
        if self.rng.random() < 0.03:
            x, y = self.self_pos
            self.self_pos = (max(0, min(self.size - 1, x + self.rng.choice([-1, 1]))), y)
        target = self.choose_target(self.self_pos, self.self_energy, self.known_resources, obs.hazards, obs.base)
        action = greedy_step(self.self_pos, target, self.size)
        self.last_action = action
        self.metrics.update(1.0, 0.0, calibrated=False)
        return action

    def observe_post_step(self, obs: Observation) -> None:
        self.self_pos = obs.pos
        self.self_energy = obs.energy


class MirrorAgent(BaseAgent):
    """Full toy Mirror agent with self-model reliability tracking.

    It predicts its next self-state from the previous self-state and action,
    compares prediction with incoming observation, lowers reliability when the
    discrepancy is self-relevant, and calibrates or downweights the observation
    when reliability drops.
    """
    name = "D_mirror_reliability"

    def reset(self, obs: Observation) -> None:
        super().reset(obs)
        self.self_pos = obs.pos
        self.self_energy = obs.energy
        self.predicted_pos = obs.pos
        self.predicted_energy = obs.energy
        self.reliability = 1.0
        self.known_resources = set(obs.resources)
        self.unreliable_steps = 0

    def _predict_from_action(self) -> Tuple[Position, float]:
        predicted_pos = step_pos(self.self_pos, self.last_action, self.size)
        predicted_energy = max(0.0, self.self_energy - 1.0)
        return predicted_pos, predicted_energy

    def _update_reliability(self, obs: Observation) -> float:
        pred_pos, pred_energy = self._predict_from_action()
        pos_error = manhattan(pred_pos, obs.pos)
        energy_error = abs(pred_energy - obs.energy)
        # A discrepancy beyond one move or a very large energy jump is
        # self-relevant contradiction. The energy threshold is deliberately
        # wide because resource collection and base recharge can create
        # legitimate jumps that this minimal agent does not fully model.
        contradiction = max(0.0, pos_error - 1.0) + max(0.0, (energy_error - 20.0) / 12.0)
        if obs.calibrated:
            self.reliability = 1.0
            self.self_pos = obs.true_pos if obs.true_pos is not None else obs.pos
            self.self_energy = obs.true_energy if obs.true_energy is not None else obs.energy
            self.unreliable_steps = 0
            self.metrics.update(self.reliability, 0.0, calibrated=True)
            return 0.0
        if contradiction > 0:
            self.reliability = max(0.05, self.reliability - 0.14 * contradiction)
            self.unreliable_steps += 1
        else:
            self.reliability = min(1.0, self.reliability + 0.04)
            self.unreliable_steps = max(0, self.unreliable_steps - 1)
        # Reliability-gated self-model update.
        if self.reliability >= 0.55:
            self.self_pos = obs.pos
            self.self_energy = obs.energy
        else:
            # Trust the action-integrated prediction more than suspect self-observation.
            self.self_pos = pred_pos
            self.self_energy = pred_energy
        self.metrics.update(self.reliability, contradiction, calibrated=False)
        return contradiction

    def act(self, obs: Observation) -> Action:
        self.known_resources &= set(obs.resources)
        contradiction = self._update_reliability(obs)
        # If the self-model is unreliable, spend a step to recalibrate before acting.
        # This models a minimal reliability variable becoming viability-relevant.
        if (self.reliability < 0.18 and self.self_energy < 10) or self.unreliable_steps >= 18:
            action: Action = "calibrate"
            self.last_action = action
            return action
        target = self.choose_target(self.self_pos, self.self_energy, self.known_resources, obs.hazards, obs.base)
        action = greedy_step(self.self_pos, target, self.size)
        self.last_action = action
        return action

    def observe_post_step(self, obs: Observation) -> None:
        # The next call to act will consume this observation and do reliability update.
        pass


# Experiment runner for Mirror Observerhood Lab I.
from typing import Type
import csv


AGENTS: List[Type[BaseAgent]] = [PredictorAgent, MemoryAgent, SelfModelAgent, MirrorAgent]

# Execution order is kept unchanged so the fixed seeds reproduce the publication data.
PERTURBATIONS = ["control", "sensor_degradation", "false_location", "false_energy", "memory_corruption"]

# Display order and labels used for publication-style figures. These do not affect
# simulation or seed allocation; they only make generated figures match the paper.
PERTURBATION_DISPLAY_ORDER = [
    "control",
    "false_energy",
    "memory_corruption",
    "sensor_degradation",
    "false_location",
]
PERTURBATION_LABELS = {
    "control": "Control",
    "false_energy": "False energy",
    "memory_corruption": "Memory\ncorruption",
    "sensor_degradation": "Sensor\ndegradation",
    "false_location": "False\nlocation",
}
AGENT_DISPLAY_ORDER = [
    "A_predictor_only",
    "B_memory",
    "C_self_model_no_reliability",
    "D_mirror_reliability",
]
AGENT_LABELS = {
    "A_predictor_only": "Predictor",
    "B_memory": "Memory",
    "C_self_model_no_reliability": "Self-model",
    "D_mirror_reliability": "Mirror",
}


def run_episode(agent_cls: Type[BaseAgent], perturbation: str, seed: int, max_steps: int = 150) -> Dict[str, float | int | str]:
    rng_env = random.Random(seed)
    rng_agent = random.Random(seed + 1000003)
    env = GridWorld(max_steps=max_steps, perturbation=perturbation, rng=rng_env)
    obs = env.reset()
    agent = agent_cls(size=env.size, rng=rng_agent)
    agent.reset(obs)
    done = False
    steps = 0
    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        agent.observe_post_step(obs)
        steps += 1
    m = agent.metrics
    return {
        "seed": seed,
        "perturbation": perturbation,
        "agent": agent.name,
        "steps": env.t,
        "survived": int(env.energy > 0 and env.t >= env.max_steps),
        "energy_final": round(env.energy, 4),
        "resources_collected": env.collected,
        "hazards_hit": env.hazards_hit,
        "calibrations": env.calibrations,
        "viability": round(env.viability_score(), 4),
        "contradiction_load": round(m.contradictions, 4),
        "reliability_mean": round(m.reliability_mean, 4),
        "reliability_min": round(m.reliability_min, 4),
    }


def run_all(n_episodes: int = 250, base_seed: int = 12345, max_steps: int = 150) -> List[Dict[str, float | int | str]]:
    rows: List[Dict[str, float | int | str]] = []
    for perturb_idx, perturbation in enumerate(PERTURBATIONS):
        for agent_idx, agent_cls in enumerate(AGENTS):
            for i in range(n_episodes):
                seed = base_seed + perturb_idx * 100000 + agent_idx * 10000 + i
                rows.append(run_episode(agent_cls, perturbation, seed=seed, max_steps=max_steps))
    return rows


def write_csv(rows: List[Dict[str, float | int | str]], path: str) -> None:
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)




# -----------------------------------------------------------------------------
# Publication runner helpers
# -----------------------------------------------------------------------------

def summarise_rows(rows):
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("pandas is required for summaries. Install requirements.txt") from exc
    df = pd.DataFrame(rows)
    numeric = [c for c in df.columns if c not in ("agent", "perturbation", "seed")]
    summary = df.groupby(["perturbation", "agent"])[numeric].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join([str(x) for x in col if x]) for col in summary.columns]
    return df, summary


def save_basic_figures(df, outdir, prefix):
    """Save publication-style figures using fixed order and clean labels.

    This function intentionally separates simulation order from display order.
    The raw data retain the fixed-seed perturbation order used to generate the
    publication results, while the figures follow the narrative order used in
    the Lab I paper: Control, False energy, Memory corruption, Sensor
    degradation, False location.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    hatches = ["", "///", "...", "xxx"]
    shades = ["0.85", "0.65", "0.45", "0.20"]
    x = np.arange(len(PERTURBATION_DISPLAY_ORDER))
    width = 0.18

    def grouped_bar(metric, ylabel, title, filename, percent=False):
        fig, ax = plt.subplots(figsize=(8.4, 4.4))
        for idx, agent in enumerate(AGENT_DISPLAY_ORDER):
            vals = []
            for condition in PERTURBATION_DISPLAY_ORDER:
                sub = df[(df["perturbation"] == condition) & (df["agent"] == agent)]
                value = sub[metric].mean()
                if percent:
                    value *= 100.0
                vals.append(value)
            ax.bar(
                x + (idx - 1.5) * width,
                vals,
                width,
                label=AGENT_LABELS[agent],
                color=shades[idx],
                edgecolor="black",
                linewidth=0.6,
                hatch=hatches[idx],
            )
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Perturbation condition")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([PERTURBATION_LABELS[c] for c in PERTURBATION_DISPLAY_ORDER])
        ax.grid(axis="y", linewidth=0.4, alpha=0.35)
        ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(figdir / f"{filename}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def mirror_advantage():
        vals = []
        for condition in PERTURBATION_DISPLAY_ORDER:
            sub = df[df["perturbation"] == condition]
            mirror = sub[sub["agent"] == "D_mirror_reliability"]["viability"].mean()
            non_mirror = sub[sub["agent"] != "D_mirror_reliability"].groupby("agent")["viability"].mean()
            vals.append(mirror - non_mirror.max())
        fig, ax = plt.subplots(figsize=(7.8, 3.6))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.bar(x, vals, color="0.45", edgecolor="black", linewidth=0.6, hatch="///")
        ax.set_ylabel("Mirror viability advantage")
        ax.set_xlabel("Perturbation condition")
        ax.set_title("Mirror advantage over strongest non-Mirror baseline")
        ax.set_xticks(x)
        ax.set_xticklabels([PERTURBATION_LABELS[c] for c in PERTURBATION_DISPLAY_ORDER])
        ax.grid(axis="y", linewidth=0.4, alpha=0.35)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(figdir / f"{prefix}_mirror_advantage.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    grouped_bar(
        "viability",
        "Mean viability",
        "Mean viability by perturbation and architecture",
        f"{prefix}_viability_by_condition",
    )
    grouped_bar(
        "survived",
        "Survival rate (%)",
        "Survival rate by perturbation and architecture",
        f"{prefix}_survival_by_condition",
        percent=True,
    )
    mirror_advantage()

def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Run Mirror Observerhood Lab I: Recursive Self-Model Reliability Improves Viability Under Self-Relevant Perturbation.")
    parser.add_argument("--episodes", type=int, default=300, help="episodes/seeds per agent-condition cell")
    parser.add_argument("--seed", type=int, default=12345, help="base random seed")
    parser.add_argument("--max-steps", type=int, default=None, help="optional episode horizon override")
    parser.add_argument("--outdir", type=str, default="outputs/lab_I", help="output directory")
    parser.add_argument("--no-figures", action="store_true", help="skip figure generation")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    max_steps = 150 if args.max_steps is None else args.max_steps
    rows = run_all(n_episodes=args.episodes, base_seed=args.seed, max_steps=max_steps)
    df, summary = summarise_rows(rows)
    data_dir = outdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / "Mirror_Observerhood_Lab_I_results.csv", index=False)
    summary.to_csv(data_dir / "Mirror_Observerhood_Lab_I_summary_by_condition.csv", index=False)
    if not args.no_figures:
        save_basic_figures(df, outdir, "Mirror_Observerhood_Lab_I")
    print(f"Wrote {len(df)} rows to {outdir}")
    print(summary.head().to_string(index=False))


if __name__ == "__main__":
    main()
