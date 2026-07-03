#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mirror Observerhood Lab II: Channel-Decomposed Reliability and the Limits of Reliability Tracking Alone

Single-file reproducibility script for the Mirror Programme Observerhood Labs.
Run with: python mirror_lab_ii.py --help
"""
from __future__ import annotations

"""Mirror Observerhood Lab II environment.

A controlled grid-world for testing whether decomposed reliability tracking
(self-state, sensor-channel, and world-map confidence) improves viability under
channel-specific perturbation.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import random

Position = Tuple[int, int]
Action = str
ACTIONS: Tuple[Action, ...] = ("up", "down", "left", "right", "stay", "calibrate", "scan")


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
    elif action in ("stay", "calibrate", "scan"):
        pass
    else:
        raise ValueError(f"Unknown action: {action}")
    return max(0, min(size - 1, x)), max(0, min(size - 1, y))


def greedy_step(pos: Position, target: Position, size: int, forbidden: Set[Position] | None = None) -> Action:
    forbidden = forbidden or set()
    order = ["up", "down", "left", "right", "stay"]
    candidates: List[Tuple[int, int, Action]] = []
    for idx, action in enumerate(order):
        np = step_pos(pos, action, size)
        penalty = 5 if np in forbidden else 0
        candidates.append((manhattan(np, target) + penalty, idx, action))
    candidates.sort()
    return candidates[0][2]


@dataclass
class Observation:
    pos: Position
    energy: float
    resources: Set[Position]
    hazards: Set[Position]
    base: Position
    calibrated: bool = False
    scanned: bool = False
    true_pos: Optional[Position] = None
    true_energy: Optional[float] = None
    sensor_flags: Dict[str, bool] = field(default_factory=dict)


@dataclass
class StepInfo:
    alive: bool
    collected: int
    hazards_hit: int
    at_base: bool
    calibration_used: bool
    scan_used: bool
    true_pos: Position
    true_energy: float


@dataclass
class GridWorld:
    size: int = 14
    max_steps: int = 170
    initial_energy: float = 62.0
    max_energy: float = 76.0
    step_cost: float = 1.0
    hazard_cost: float = 15.0
    resource_gain: float = 24.0
    base_recharge: float = 4.5
    calibrate_cost: float = 1.8
    scan_cost: float = 1.3
    n_resources: int = 12
    n_hazards: int = 24
    perturbation: str = "control"
    perturb_start: int = 38
    perturb_end: int = 115
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
    scans: int = field(init=False, default=0)

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
        self.scans = 0
        cells = [(x, y) for x in range(self.size) for y in range(self.size) if (x, y) != self.base]
        self.rng.shuffle(cells)
        self.hazards = set(cells[: self.n_hazards])
        self.resources = set(cells[self.n_hazards : self.n_hazards + self.n_resources])
        # Keep a small radius around base safe.
        self.hazards = {p for p in self.hazards if manhattan(p, self.base) > 1}
        return self.observe(calibrate=True)

    def in_perturbation_window(self) -> bool:
        return self.perturb_start <= self.t < self.perturb_end

    def _clip(self, x: int, y: int) -> Position:
        return max(0, min(self.size - 1, x)), max(0, min(self.size - 1, y))

    def _distort_position(self, pos: Position) -> Position:
        x, y = pos
        if not self.in_perturbation_window():
            return pos
        if self.perturbation == "false_location":
            # Persistent offset: a stable but wrong self-location channel.
            return self._clip(x + 3, y - 2)
        if self.perturbation == "sensor_degradation":
            # Stochastic channel noise: not a stable self-location illusion.
            if self.rng.random() < 0.52:
                dx, dy = self.rng.choice([(2, 0), (-2, 0), (0, 2), (0, -2), (1, 1), (-1, -1), (1, -1), (-1, 1)])
                return self._clip(x + dx, y + dy)
        if self.perturbation == "mixed_self_sensor":
            # Both a smaller persistent bias and stochastic channel noise.
            bx, by = 2, -1
            if self.rng.random() < 0.35:
                dx, dy = self.rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (0, -2)])
                return self._clip(x + bx + dx, y + by + dy)
            return self._clip(x + bx, y + by)
        if self.perturbation == "memory_corruption":
            # Sensors are mostly normal; corruption hits the map observations below.
            if self.rng.random() < 0.06:
                dx, dy = self.rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
                return self._clip(x + dx, y + dy)
        return pos

    def _distort_energy(self, energy: float) -> float:
        if not self.in_perturbation_window():
            return energy
        if self.perturbation == "false_energy":
            return min(self.max_energy, energy + 20.0)
        if self.perturbation == "sensor_degradation":
            return max(0.0, min(self.max_energy, energy + self.rng.uniform(-14.0, 14.0)))
        if self.perturbation == "mixed_self_sensor":
            return max(0.0, min(self.max_energy, energy + 8.0 + self.rng.uniform(-7.0, 7.0)))
        return energy

    def _distort_map(self, resources: Set[Position], hazards: Set[Position]) -> Tuple[Set[Position], Set[Position], bool]:
        if not self.in_perturbation_window() or self.perturbation != "memory_corruption":
            return set(resources), set(hazards), False
        res = set(resources)
        haz = set(hazards)
        distorted = False
        cells = [(x, y) for x in range(self.size) for y in range(self.size) if (x, y) != self.base]
        # False negatives: hide a resource / hazard.
        if res and self.rng.random() < 0.35:
            res.remove(self.rng.choice(tuple(res)))
            distorted = True
        if haz and self.rng.random() < 0.22:
            haz.remove(self.rng.choice(tuple(haz)))
            distorted = True
        # False positives.
        if self.rng.random() < 0.30:
            p = self.rng.choice(cells)
            if p not in res and p != self.pos:
                res.add(p)
                distorted = True
        if self.rng.random() < 0.20:
            p = self.rng.choice(cells)
            if p not in haz and p != self.pos:
                haz.add(p)
                distorted = True
        return res, haz, distorted

    def observe(self, calibrate: bool = False, scan: bool = False) -> Observation:
        if calibrate:
            # Calibration resolves self-channel state; it does not necessarily repair the entire map.
            resources, hazards, map_distorted = self._distort_map(self.resources, self.hazards)
            return Observation(
                pos=self.pos,
                energy=self.energy,
                resources=resources,
                hazards=hazards,
                base=self.base,
                calibrated=True,
                true_pos=self.pos,
                true_energy=self.energy,
                sensor_flags={"calibrated": True, "map_distorted": map_distorted},
            )
        if scan:
            # Scan resolves the external map; it does not reveal true self-state beyond ordinary sensors.
            obs_pos = self._distort_position(self.pos)
            obs_energy = self._distort_energy(self.energy)
            return Observation(
                pos=obs_pos,
                energy=obs_energy,
                resources=set(self.resources),
                hazards=set(self.hazards),
                base=self.base,
                scanned=True,
                sensor_flags={
                    "position_distorted": obs_pos != self.pos,
                    "energy_distorted": abs(obs_energy - self.energy) > 1e-9,
                    "scan": True,
                },
            )
        obs_pos = self._distort_position(self.pos)
        obs_energy = self._distort_energy(self.energy)
        resources, hazards, map_distorted = self._distort_map(self.resources, self.hazards)
        return Observation(
            pos=obs_pos,
            energy=obs_energy,
            resources=resources,
            hazards=hazards,
            base=self.base,
            sensor_flags={
                "position_distorted": obs_pos != self.pos,
                "energy_distorted": abs(obs_energy - self.energy) > 1e-9,
                "map_distorted": map_distorted,
            },
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, StepInfo]:
        if not self.alive:
            raise RuntimeError("Cannot step a terminated episode")
        calibration_used = action == "calibrate"
        scan_used = action == "scan"
        if calibration_used:
            self.calibrations += 1
            self.energy -= self.calibrate_cost
        elif scan_used:
            self.scans += 1
            self.energy -= self.scan_cost
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
        done = (self.energy <= 0) or (self.t >= self.max_steps)
        self.alive = not done or self.energy > 0
        obs = self.observe(calibrate=calibration_used, scan=scan_used)
        reward = self.viability_score() / max(self.max_steps, 1)
        info = StepInfo(
            alive=self.energy > 0,
            collected=self.collected,
            hazards_hit=self.hazards_hit,
            at_base=at_base,
            calibration_used=calibration_used,
            scan_used=scan_used,
            true_pos=self.pos,
            true_energy=self.energy,
        )
        return obs, reward, done, info

    def viability_score(self) -> float:
        survived_bonus = 55.0 if self.energy > 0 and self.t >= self.max_steps else 0.0
        return (
            float(self.t)
            + survived_bonus
            + 11.5 * self.collected
            - 10.0 * self.hazards_hit
            - 1.7 * self.calibrations
            - 1.1 * self.scans
            + 0.22 * max(self.energy, 0.0)
        )


"""Agent architectures for Mirror Observerhood Lab II."""
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
import random



@dataclass
class EpisodeMetrics:
    contradictions: float = 0.0
    calibrations: int = 0
    scans: int = 0
    repairs: int = 0
    r_global_trace: List[float] = field(default_factory=list)
    r_self_trace: List[float] = field(default_factory=list)
    r_sensor_trace: List[float] = field(default_factory=list)
    r_map_trace: List[float] = field(default_factory=list)
    contradiction_trace: List[float] = field(default_factory=list)

    def update(self, contradiction: float, r_global: float = 1.0, r_self: float = 1.0, r_sensor: float = 1.0, r_map: float = 1.0, action: Action = "stay") -> None:
        self.contradictions += float(contradiction)
        self.contradiction_trace.append(float(contradiction))
        self.r_global_trace.append(float(r_global))
        self.r_self_trace.append(float(r_self))
        self.r_sensor_trace.append(float(r_sensor))
        self.r_map_trace.append(float(r_map))
        if action == "calibrate":
            self.calibrations += 1
            self.repairs += 1
        if action == "scan":
            self.scans += 1
            self.repairs += 1

    @property
    def r_global_mean(self) -> float:
        return sum(self.r_global_trace) / len(self.r_global_trace) if self.r_global_trace else 1.0

    @property
    def r_self_mean(self) -> float:
        return sum(self.r_self_trace) / len(self.r_self_trace) if self.r_self_trace else 1.0

    @property
    def r_sensor_mean(self) -> float:
        return sum(self.r_sensor_trace) / len(self.r_sensor_trace) if self.r_sensor_trace else 1.0

    @property
    def r_map_mean(self) -> float:
        return sum(self.r_map_trace) / len(self.r_map_trace) if self.r_map_trace else 1.0


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

    def choose_target(self, pos: Position, energy: float, resources: Set[Position], hazards: Set[Position], base: Position) -> Position:
        if energy < 24 or not resources:
            return base
        def score(r: Position) -> float:
            # Avoid choosing cells marked as hazards and keep some energy margin for return.
            return manhattan(pos, r) + 4.0 * (r in hazards) + 0.15 * manhattan(r, base)
        return min(resources, key=score)

    def _safe_greedy(self, pos: Position, target: Position, hazards: Set[Position]) -> Action:
        return greedy_step(pos, target, self.size, forbidden=hazards)


class PredictorAgent(BaseAgent):
    name = "A_predictor_only"

    def act(self, obs: Observation) -> Action:
        target = self.choose_target(obs.pos, obs.energy, obs.resources, obs.hazards, obs.base)
        action = self._safe_greedy(obs.pos, target, obs.hazards)
        self.last_action = action
        self.metrics.update(0.0, action=action)
        return action


class SelfModelAgent(BaseAgent):
    name = "B_self_model_no_reliability"

    def reset(self, obs: Observation) -> None:
        super().reset(obs)
        self.self_pos = obs.pos
        self.self_energy = obs.energy
        self.known_resources = set(obs.resources)
        self.known_hazards = set(obs.hazards)

    def act(self, obs: Observation) -> Action:
        # Blindly commits both self-state and world-map observations.
        self.self_pos = obs.pos
        self.self_energy = obs.energy
        self.known_resources &= set(obs.resources)
        self.known_resources |= set(obs.resources)
        self.known_hazards = set(obs.hazards)
        # Tiny endogenous drift: memory systems without reliability gates accumulate defects.
        if self.rng.random() < 0.015:
            x, y = self.self_pos
            self.self_pos = (max(0, min(self.size - 1, x + self.rng.choice([-1, 1]))), y)
        target = self.choose_target(self.self_pos, self.self_energy, self.known_resources, self.known_hazards, obs.base)
        action = self._safe_greedy(self.self_pos, target, self.known_hazards)
        self.last_action = action
        self.metrics.update(0.0, action=action)
        return action


class ScalarMirrorAgent(BaseAgent):
    """Lab-I style Mirror agent with a single reliability variable.

    It detects contradiction but uses one global reliability scalar. This is
    useful for self-location perturbation, but it can overgeneralise under
    sensor degradation or map corruption.
    """
    name = "C_scalar_mirror_reliability"

    def reset(self, obs: Observation) -> None:
        super().reset(obs)
        self.self_pos = obs.pos
        self.self_energy = obs.energy
        self.known_resources = set(obs.resources)
        self.known_hazards = set(obs.hazards)
        self.reliability = 1.0
        self.bad_steps = 0

    def _prediction(self) -> Tuple[Position, float]:
        return step_pos(self.self_pos, self.last_action, self.size), max(0.0, self.self_energy - 1.0)

    def _update(self, obs: Observation) -> float:
        pred_pos, pred_energy = self._prediction()
        if obs.calibrated:
            self.self_pos = obs.true_pos if obs.true_pos is not None else obs.pos
            self.self_energy = obs.true_energy if obs.true_energy is not None else obs.energy
            self.reliability = min(1.0, self.reliability + 0.35)
            self.bad_steps = 0
            return 0.0
        pos_error = max(0.0, manhattan(pred_pos, obs.pos) - 1.0)
        energy_error = max(0.0, abs(pred_energy - obs.energy) - 22.0) / 13.0
        # Map contradiction: if many known hazards/resources disappear or appear, treat as global unreliability.
        map_delta = len(self.known_resources.symmetric_difference(obs.resources)) / 18.0 + len(self.known_hazards.symmetric_difference(obs.hazards)) / 35.0
        contradiction = pos_error + energy_error + min(2.5, map_delta)
        if contradiction > 0:
            self.reliability = max(0.04, self.reliability - 0.10 * contradiction)
            self.bad_steps += 1
        else:
            self.reliability = min(1.0, self.reliability + 0.035)
            self.bad_steps = max(0, self.bad_steps - 1)
        if self.reliability >= 0.52:
            self.self_pos = obs.pos
            self.self_energy = obs.energy
            self.known_resources = set(obs.resources)
            self.known_hazards = set(obs.hazards)
        else:
            # Fall back to prediction and stale world model.
            self.self_pos = pred_pos
            self.self_energy = pred_energy
            self.known_resources &= set(obs.resources)
        return contradiction

    def act(self, obs: Observation) -> Action:
        contradiction = self._update(obs)
        # Global reliability does not know which channel failed, so it frequently uses costly repair.
        if self.bad_steps >= 13 or (self.reliability < 0.16 and self.self_energy < 18):
            action: Action = "calibrate"
        else:
            target = self.choose_target(self.self_pos, self.self_energy, self.known_resources, self.known_hazards, obs.base)
            action = self._safe_greedy(self.self_pos, target, self.known_hazards)
        self.last_action = action
        self.metrics.update(contradiction, r_global=self.reliability, r_self=self.reliability, r_sensor=self.reliability, r_map=self.reliability, action=action)
        return action


class DecomposedMirrorAgent(BaseAgent):
    """Mirror agent with separate reliability variables.

    R_self governs commitment to self-location/energy; R_sensor governs how
    much raw sensor observations are trusted; R_map governs commitment to the
    external resource/hazard map. Repair actions are channel-specific: calibrate
    repairs self-state, scan repairs map confidence.
    """
    name = "D_decomposed_mirror_reliability"

    def reset(self, obs: Observation) -> None:
        super().reset(obs)
        self.self_pos = obs.pos
        self.self_energy = obs.energy
        self.known_resources = set(obs.resources)
        self.known_hazards = set(obs.hazards)
        self.r_self = 1.0
        self.r_sensor = 1.0
        self.r_map = 1.0
        self.self_bad = 0
        self.sensor_bad = 0
        self.map_bad = 0
        self.last_innovation: List[float] = []

    def _prediction(self) -> Tuple[Position, float]:
        return step_pos(self.self_pos, self.last_action, self.size), max(0.0, self.self_energy - 1.0)

    def _update_reliabilities(self, obs: Observation) -> float:
        pred_pos, pred_energy = self._prediction()
        if obs.calibrated:
            self.self_pos = obs.true_pos if obs.true_pos is not None else obs.pos
            self.self_energy = obs.true_energy if obs.true_energy is not None else obs.energy
            self.r_self = min(1.0, self.r_self + 0.45)
            self.r_sensor = min(1.0, self.r_sensor + 0.08)
            self.self_bad = 0
            # Calibration does not fix map reliability.
            return 0.0
        if obs.scanned:
            self.known_resources = set(obs.resources)
            self.known_hazards = set(obs.hazards)
            self.r_map = min(1.0, self.r_map + 0.55)
            self.map_bad = 0
            # Scan still leaves self channel ambiguous.
        pos_innov = max(0.0, manhattan(pred_pos, obs.pos) - 1.0)
        energy_innov = max(0.0, abs(pred_energy - obs.energy) - 22.0) / 13.0
        self_innov = pos_innov + energy_innov
        self.last_innovation.append(self_innov)
        if len(self.last_innovation) > 8:
            self.last_innovation.pop(0)
        # If innovations are high but unstable, this looks like noisy sensors.
        mean_i = sum(self.last_innovation) / len(self.last_innovation)
        var_i = sum((x - mean_i) ** 2 for x in self.last_innovation) / len(self.last_innovation)
        unstable_noise = mean_i > 0.45 and var_i > 0.35
        persistent_bias = mean_i > 0.75 and var_i <= 0.35
        if unstable_noise:
            self.r_sensor = max(0.08, self.r_sensor - 0.13 * mean_i)
            self.sensor_bad += 1
            # Do not punish self-model as aggressively: the channel, not the self-model, is suspect.
            self.r_self = max(0.28, self.r_self - 0.010 * mean_i)
        elif persistent_bias:
            self.r_self = max(0.04, self.r_self - 0.16 * mean_i)
            self.self_bad += 2
            self.r_sensor = max(0.18, self.r_sensor - 0.020 * mean_i)
        elif self_innov > 0:
            self.r_self = max(0.10, self.r_self - 0.07 * self_innov)
            self.r_sensor = max(0.12, self.r_sensor - 0.03 * self_innov)
            self.self_bad += 1
        else:
            self.r_self = min(1.0, self.r_self + 0.04)
            self.r_sensor = min(1.0, self.r_sensor + 0.025)
            self.self_bad = max(0, self.self_bad - 1)
            self.sensor_bad = max(0, self.sensor_bad - 1)

        map_delta = len(self.known_resources.symmetric_difference(obs.resources)) / 18.0 + len(self.known_hazards.symmetric_difference(obs.hazards)) / 35.0
        if map_delta > 0.30:
            self.r_map = max(0.08, self.r_map - 0.10 * min(3.0, map_delta))
            self.map_bad += 1
        else:
            self.r_map = min(1.0, self.r_map + 0.03)
            self.map_bad = max(0, self.map_bad - 1)

        # Commitment policy: use observation only if the relevant channel is credible.
        if self.r_sensor >= 0.62 and self.r_self >= 0.50:
            self.self_pos = obs.pos
            self.self_energy = obs.energy
        else:
            # Use action-integrated self-state. This is not magic truth; it is dead reckoning.
            self.self_pos = pred_pos
            # Filter energy rather than blindly accepting noisy energy.
            if self.r_sensor < 0.45:
                self.self_energy = max(pred_energy, 0.65 * pred_energy + 0.35 * obs.energy)
            else:
                self.self_energy = 0.65 * pred_energy + 0.35 * obs.energy
        if self.r_map >= 0.50 or obs.scanned:
            self.known_resources = set(obs.resources)
            self.known_hazards = set(obs.hazards)
        else:
            # Retain stable prior but remove resources that observation also says are absent.
            self.known_resources &= set(obs.resources)
        return self_innov + min(2.5, map_delta)

    def act(self, obs: Observation) -> Action:
        contradiction = self._update_reliabilities(obs)
        # Channel-specific repair. Calibrate if self-state is unreliable; scan if map is unreliable.
        if self.self_bad >= 7 or (self.r_self < 0.18 and self.self_energy < 22):
            action: Action = "calibrate"
        elif self.map_bad >= 5 and self.self_energy > 20:
            action = "scan"
        else:
            target = self.choose_target(self.self_pos, self.self_energy, self.known_resources, self.known_hazards, obs.base)
            action = self._safe_greedy(self.self_pos, target, self.known_hazards)
        self.last_action = action
        r_global = (self.r_self + self.r_sensor + self.r_map) / 3.0
        self.metrics.update(contradiction, r_global=r_global, r_self=self.r_self, r_sensor=self.r_sensor, r_map=self.r_map, action=action)
        return action


"""Experiment runner for Mirror Observerhood Lab II."""
from dataclasses import asdict
from typing import Any, Dict, List, Type
import random


AGENTS: List[Type[BaseAgent]] = [PredictorAgent, SelfModelAgent, ScalarMirrorAgent, DecomposedMirrorAgent]
PERTURBATIONS = ["control", "false_location", "false_energy", "sensor_degradation", "memory_corruption", "mixed_self_sensor"]


def run_episode(agent_cls: Type[BaseAgent], perturbation: str, seed: int, max_steps: int = 170) -> Dict[str, Any]:
    rng_env = random.Random(seed)
    rng_agent = random.Random(seed + 100_003)
    env = GridWorld(perturbation=perturbation, rng=rng_env, max_steps=max_steps)
    obs = env.reset()
    agent = agent_cls(size=env.size, rng=rng_agent)
    agent.reset(obs)
    done = False
    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
    m = agent.metrics
    return {
        "agent": agent.name,
        "perturbation": perturbation,
        "seed": seed,
        "viability": env.viability_score(),
        "survival_steps": env.t,
        "survived": int(env.energy > 0 and env.t >= env.max_steps),
        "collected": env.collected,
        "hazards_hit": env.hazards_hit,
        "final_energy": env.energy,
        "calibrations_env": env.calibrations,
        "scans_env": env.scans,
        "contradictions": m.contradictions,
        "calibrations_agent": m.calibrations,
        "scans_agent": m.scans,
        "repairs": m.repairs,
        "r_global_mean": m.r_global_mean,
        "r_self_mean": m.r_self_mean,
        "r_sensor_mean": m.r_sensor_mean,
        "r_map_mean": m.r_map_mean,
    }


def run_all(n_seeds: int = 320, base_seed: int = 9101) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p_idx, perturbation in enumerate(PERTURBATIONS):
        for a_idx, agent_cls in enumerate(AGENTS):
            for i in range(n_seeds):
                seed = base_seed + 10_000 * p_idx + 1_000 * a_idx + i
                rows.append(run_episode(agent_cls, perturbation, seed))
    return rows




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


PERTURBATION_ORDER = [
    "control",
    "false_location",
    "false_energy",
    "sensor_degradation",
    "memory_corruption",
    "mixed_self_sensor",
]

PERTURBATION_LABELS = {
    "control": "Control",
    "false_location": "False\nlocation",
    "false_energy": "False energy",
    "sensor_degradation": "Sensor\ndegradation",
    "memory_corruption": "Memory\ncorruption",
    "mixed_self_sensor": "Mixed self\n+ sensor",
}

AGENT_ORDER = [
    "A_predictor_only",
    "B_self_model_no_reliability",
    "C_scalar_mirror_reliability",
    "D_decomposed_mirror_reliability",
]

AGENT_LABELS = {
    "A_predictor_only": "Predictor",
    "B_self_model_no_reliability": "Self-model",
    "C_scalar_mirror_reliability": "Scalar Mirror",
    "D_decomposed_mirror_reliability": "Decomposed Mirror",
}


def _ordered_pivot(df, value):
    pivot = df.pivot_table(index="perturbation", columns="agent", values=value, aggfunc="mean")
    pivot = pivot.reindex(PERTURBATION_ORDER)
    pivot = pivot.reindex(columns=AGENT_ORDER)
    pivot.index = [PERTURBATION_LABELS[x] for x in PERTURBATION_ORDER]
    pivot.columns = [AGENT_LABELS[x] for x in AGENT_ORDER]
    return pivot


def _apply_grayscale_patterns(ax):
    hatches = ["", "///", "...", "xxx"]
    colors = ["0.82", "0.65", "0.50", "0.25"]
    for i, patch in enumerate(ax.patches):
        group = i // len(PERTURBATION_ORDER)
        patch.set_facecolor(colors[group % len(colors)])
        patch.set_edgecolor("black")
        patch.set_linewidth(0.55)
        patch.set_hatch(hatches[group % len(hatches)])


def save_basic_figures(df, outdir, prefix):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    viability = _ordered_pivot(df, "viability")
    ax = viability.plot(kind="bar", figsize=(11, 5.5), color=["0.82", "0.65", "0.50", "0.25"], edgecolor="black", linewidth=0.55)
    _apply_grayscale_patterns(ax)
    ax.set_ylabel("Mean viability")
    ax.set_xlabel("Perturbation condition")
    ax.set_title("Mean viability by perturbation and architecture")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.31), ncol=4, frameon=False)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figdir / "mirror_observerhood_lab_ii_viability.png", dpi=300)
    plt.savefig(figdir / "mirror_observerhood_lab_ii_viability.pdf")
    plt.close()

    survival = _ordered_pivot(df, "survived") * 100.0
    ax = survival.plot(kind="bar", figsize=(11, 5.5), color=["0.82", "0.65", "0.50", "0.25"], edgecolor="black", linewidth=0.55)
    _apply_grayscale_patterns(ax)
    ax.set_ylabel("Survival rate (%)")
    ax.set_xlabel("Perturbation condition")
    ax.set_title("Survival rate by perturbation and architecture")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.31), ncol=4, frameon=False)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figdir / "mirror_observerhood_lab_ii_survival.png", dpi=300)
    plt.savefig(figdir / "mirror_observerhood_lab_ii_survival.pdf")
    plt.close()

    # Decomposed reliability advantage: relative to scalar Mirror and strongest non-decomposed baseline.
    rows = []
    for p in PERTURBATION_ORDER:
        sub = df[df["perturbation"] == p]
        means = sub.groupby("agent")["viability"].mean()
        decomp = float(means["D_decomposed_mirror_reliability"])
        scalar = float(means["C_scalar_mirror_reliability"])
        best_non_decomp = float(means[[a for a in AGENT_ORDER if a != "D_decomposed_mirror_reliability"]].max())
        rows.append((PERTURBATION_LABELS[p], decomp - scalar, decomp - best_non_decomp))
    labels = [r[0] for r in rows]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 4.8))
    b1 = ax.bar(x - width/2, [r[1] for r in rows], width, label="vs scalar Mirror", color="0.62", edgecolor="black", hatch="///", linewidth=0.55)
    b2 = ax.bar(x + width/2, [r[2] for r in rows], width, label="vs best non-decomposed", color="0.35", edgecolor="black", hatch="xxx", linewidth=0.55)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean viability advantage")
    ax.set_xlabel("Perturbation condition")
    ax.set_title("Decomposed reliability advantage")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(figdir / "mirror_observerhood_lab_ii_decomposed_advantage.png", dpi=300)
    plt.savefig(figdir / "mirror_observerhood_lab_ii_decomposed_advantage.pdf")
    plt.close()

def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Run Mirror Observerhood Lab II: Channel-Decomposed Reliability and the Limits of Reliability Tracking Alone.")
    parser.add_argument("--episodes", type=int, default=240, help="episodes/seeds per agent-condition cell")
    parser.add_argument("--seed", type=int, default=9101, help="base random seed")
    parser.add_argument("--max-steps", type=int, default=None, help="optional episode horizon override")
    parser.add_argument("--outdir", type=str, default="outputs/lab_II", help="output directory")
    parser.add_argument("--no-figures", action="store_true", help="skip figure generation")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.max_steps is None:
        rows = run_all(n_seeds=args.episodes, base_seed=args.seed)
    else:
        rows = []
        for p_idx, perturbation in enumerate(PERTURBATIONS):
            for a_idx, agent_cls in enumerate(AGENTS):
                for i in range(args.episodes):
                    seed = args.seed + 10_000 * p_idx + 1_000 * a_idx + i
                    rows.append(run_episode(agent_cls, perturbation, seed, max_steps=args.max_steps))
    df, summary = summarise_rows(rows)
    data_dir = outdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / "Mirror_Observerhood_Lab_II_results.csv", index=False)
    summary.to_csv(data_dir / "Mirror_Observerhood_Lab_II_summary_by_condition.csv", index=False)
    if not args.no_figures:
        save_basic_figures(df, outdir, "Mirror_Observerhood_Lab_II")
    print(f"Wrote {len(df)} rows to {outdir}")
    print(summary.head().to_string(index=False))


if __name__ == "__main__":
    main()
