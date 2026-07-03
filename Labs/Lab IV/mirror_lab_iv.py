#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mirror Observerhood Lab IV: Minimal Reliability Thresholds in Viability-Constrained Agents

Single-file reproducibility script for the Mirror Programme Observerhood Labs.
Run with: python mirror_lab_iv.py --help
"""
from __future__ import annotations

"""
Mirror Observerhood Lab IV
Minimal reliability thresholds in viability-constrained agents.

This simulation is intentionally minimal. It tests whether a reliability-gated,
cost-sensitive self-repair policy becomes beneficial only under certain
combinations of self-state perturbation intensity and repair cost.
"""
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

GRID = 12
MAX_STEPS = 80
START = (1, 1)
GOAL = (10, 10)
ENERGY_PACKS = [(2, 9), (5, 5), (9, 2)]
HAZARDS = {(4, 4), (4, 5), (5, 4), (7, 7), (8, 7), (7, 8), (9, 8)}

PERTURB_RATES = [0.00, 0.04, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30]
REPAIR_COSTS = [0, 2, 4, 6, 8, 10, 12, 15]
THRESHOLDS = [0.35, 0.50, 0.65, 0.80]
EPISODES = 90
BASE_SEED = 4404

@dataclass
class EpisodeResult:
    agent: str
    perturb_rate: float
    repair_cost: float
    threshold: float
    seed: int
    viability: float
    survived: int
    success: int
    steps: int
    hazard_hits: int
    repairs: int
    final_energy: float
    mean_self_error: float
    mean_reliability: float


def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def clamp_pos(p: Tuple[int,int]) -> Tuple[int,int]:
    return (max(0, min(GRID-1, p[0])), max(0, min(GRID-1, p[1])))


def step_toward(src: Tuple[int,int], dst: Tuple[int,int]) -> Tuple[int,int]:
    sx, sy = src
    dx, dy = dst
    if sx < dx:
        return (1, 0)
    if sx > dx:
        return (-1, 0)
    if sy < dy:
        return (0, 1)
    if sy > dy:
        return (0, -1)
    return (0, 0)


def add(a: Tuple[int,int], b: Tuple[int,int]) -> Tuple[int,int]:
    return (a[0]+b[0], a[1]+b[1])


def choose_target(believed_pos: Tuple[int,int], energy: float, collected: set) -> Tuple[int,int]:
    # If energy is low, choose nearest remaining energy pack by believed position.
    remaining = [p for p in ENERGY_PACKS if p not in collected]
    if energy < 25 and remaining:
        return min(remaining, key=lambda p: manhattan(believed_pos, p))
    # Otherwise move toward main goal.
    return GOAL


def perturb_belief(rng: random.Random, believed: Tuple[int,int], rate: float) -> Tuple[int,int]:
    if rng.random() >= rate:
        return believed
    # Larger jumps occur more often when perturb_rate is high.
    max_jump = 1 + int(math.ceil(10 * rate))
    dx = rng.randint(-max_jump, max_jump)
    dy = rng.randint(-max_jump, max_jump)
    if dx == 0 and dy == 0:
        dx = rng.choice([-1, 1])
    return clamp_pos((believed[0]+dx, believed[1]+dy))


def local_observation(actual: Tuple[int,int]) -> Dict[str, object]:
    # Minimal percept: hazard adjacency and pack/goal contact.
    adjacent_hazard = any(manhattan(actual, h) == 1 for h in HAZARDS)
    on_hazard = actual in HAZARDS
    on_pack = actual in ENERGY_PACKS
    on_goal = actual == GOAL
    return {"adjacent_hazard": adjacent_hazard, "on_hazard": on_hazard, "on_pack": on_pack, "on_goal": on_goal}


def expected_local_observation(believed: Tuple[int,int]) -> Dict[str, object]:
    adjacent_hazard = any(manhattan(believed, h) == 1 for h in HAZARDS)
    on_hazard = believed in HAZARDS
    on_pack = believed in ENERGY_PACKS
    on_goal = believed == GOAL
    return {"adjacent_hazard": adjacent_hazard, "on_hazard": on_hazard, "on_pack": on_pack, "on_goal": on_goal}


def obs_contradiction(actual_obs: Dict[str, object], expected_obs: Dict[str, object]) -> float:
    mismatches = sum(1 for k in actual_obs if actual_obs[k] != expected_obs[k])
    return mismatches / len(actual_obs)


def run_episode(agent: str, perturb_rate: float, repair_cost: float, threshold: float, seed: int) -> EpisodeResult:
    rng = random.Random(seed)
    actual = START
    believed = START
    reliability = 1.0
    energy = 50.0
    score = 0.0
    collected = set()
    repairs = 0
    hazard_hits = 0
    self_errors: List[float] = []
    reliabilities: List[float] = []
    success = 0
    survived = 1

    for t in range(MAX_STEPS):
        # Random self-location corruption hits the represented self-state, not the world itself.
        believed = perturb_belief(rng, believed, perturb_rate)

        actual_obs = local_observation(actual)
        expected_obs = expected_local_observation(believed)
        contradiction = obs_contradiction(actual_obs, expected_obs)

        # Reliability estimator. A contradiction decreases reliability; smooth confirmation restores it slowly.
        reliability = max(0.02, min(1.0, reliability * (1 - 0.55 * contradiction) + 0.035 * (1 - contradiction)))

        # Mirror agents can perform a diagnostic/self-localisation repair. It costs energy/score.
        if agent == "mirror" and reliability < threshold:
            # Cost-sensitive repair gate. Higher cost requires deeper unreliability.
            estimated_loss = (1 - reliability) * (18 + 70 * perturb_rate)
            if estimated_loss > repair_cost:
                repairs += 1
                energy -= repair_cost
                score -= repair_cost
                believed = actual
                reliability = min(1.0, 0.92 + 0.05 * rng.random())

        target = choose_target(believed, energy, collected)
        action = step_toward(believed, target)

        # Actual movement executes same action, but the action was selected in belief-space.
        old_actual = actual
        old_dist = manhattan(believed, target)
        actual = clamp_pos(add(actual, action))
        believed = clamp_pos(add(believed, action))
        new_dist = manhattan(believed, target)

        # Movement and hazards.
        energy -= 1.0
        if actual in HAZARDS:
            hazard_hits += 1
            energy -= 10.0
            score -= 12.0
            # Being hit is a self-relevant surprise; the agent should distrust itself more.
            reliability = max(0.02, reliability - 0.18)

        if actual in ENERGY_PACKS and actual not in collected:
            collected.add(actual)
            energy += 18.0
            score += 9.0

        if actual == GOAL:
            success = 1
            score += 50.0
            break

        # If the believed plan is not making progress while the agent moves, decrease reliability a bit.
        if new_dist >= old_dist and action != (0,0):
            reliability = max(0.02, reliability - 0.03)
        else:
            reliability = min(1.0, reliability + 0.015)

        self_errors.append(manhattan(actual, believed))
        reliabilities.append(reliability)

        if energy <= 0:
            survived = 0
            break

    steps = t + 1
    # Viability combines persistence, success, remaining resources, and damage.
    viability = score + 0.75 * steps + 0.65 * energy - 9.0 * hazard_hits + 30.0 * success + 15.0 * survived
    if not self_errors:
        self_errors = [manhattan(actual, believed)]
    if not reliabilities:
        reliabilities = [reliability]

    return EpisodeResult(
        agent=agent,
        perturb_rate=perturb_rate,
        repair_cost=repair_cost,
        threshold=threshold,
        seed=seed,
        viability=float(viability),
        survived=int(survived),
        success=int(success),
        steps=int(steps),
        hazard_hits=int(hazard_hits),
        repairs=int(repairs),
        final_energy=float(energy),
        mean_self_error=float(np.mean(self_errors)),
        mean_reliability=float(np.mean(reliabilities)),
    )


def run_all(episodes: int = EPISODES, base_seed: int = BASE_SEED) -> pd.DataFrame:
    rows: List[EpisodeResult] = []
    # Use common random numbers across agent variants within each perturbation/cost cell.
    # This makes the comparison a controlled architectural ablation rather than a comparison
    # between different stochastic worlds.
    for pi, p in enumerate(PERTURB_RATES):
        for ci, c in enumerate(REPAIR_COSTS):
            cell_base = base_seed + 100000 * pi + 1000 * ci
            for ep in range(episodes):
                seed = cell_base + ep
                rows.append(run_episode("baseline", p, c, -1.0, seed))
                for th in THRESHOLDS:
                    rows.append(run_episode("mirror", p, c, th, seed))
    df = pd.DataFrame([r.__dict__ for r in rows])
    return df


def summarize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = df.groupby(["agent", "perturb_rate", "repair_cost", "threshold"], as_index=False).agg(
        viability_mean=("viability", "mean"),
        viability_sem=("viability", lambda x: x.std(ddof=1)/math.sqrt(len(x))),
        survival_rate=("survived", "mean"),
        success_rate=("success", "mean"),
        mean_steps=("steps", "mean"),
        mean_hazard_hits=("hazard_hits", "mean"),
        mean_repairs=("repairs", "mean"),
        mean_self_error=("mean_self_error", "mean"),
        mean_reliability=("mean_reliability", "mean"),
        n=("viability", "count"),
    )
    baseline = summary[summary.agent == "baseline"].copy()
    baseline = baseline.rename(columns={
        "viability_mean": "baseline_viability",
        "survival_rate": "baseline_survival",
        "success_rate": "baseline_success",
        "mean_self_error": "baseline_self_error",
        "mean_hazard_hits": "baseline_hazard_hits",
    })[["perturb_rate", "repair_cost", "baseline_viability", "baseline_survival", "baseline_success", "baseline_self_error", "baseline_hazard_hits"]]
    mirror = summary[summary.agent == "mirror"].copy()
    comp = mirror.merge(baseline, on=["perturb_rate", "repair_cost"], how="left")
    comp["delta_viability"] = comp["viability_mean"] - comp["baseline_viability"]
    comp["delta_survival"] = comp["survival_rate"] - comp["baseline_survival"]
    # Best threshold per perturb/cost cell.
    idx = comp.groupby(["perturb_rate", "repair_cost"])["delta_viability"].idxmax()
    best = comp.loc[idx].reset_index(drop=True)
    best = best.rename(columns={"threshold": "best_threshold"})
    best["mirror_positive"] = (best["delta_viability"] > 0).astype(int)
    return summary, comp, best


def save_plots(summary: pd.DataFrame, comp: pd.DataFrame, best: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    # Heatmap of best Mirror advantage.
    pivot = best.pivot(index="repair_cost", columns="perturb_rate", values="delta_viability").sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="coolwarm", vmin=-max(abs(pivot.values.min()), abs(pivot.values.max())), vmax=max(abs(pivot.values.min()), abs(pivot.values.max())))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(int(y)) for y in pivot.index])
    ax.set_xlabel("Self-state perturbation rate")
    ax.set_ylabel("Repair cost")
    ax.set_title("Best Mirror advantage over baseline (mean viability)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Δ viability")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "mirror_observerhood_lab_iv_phase_diagram.pdf")
    fig.savefig(fig_dir / "mirror_observerhood_lab_iv_phase_diagram.png", dpi=220)
    plt.close(fig)

    # Optimal threshold heatmap.
    pivot_th = best.pivot(index="repair_cost", columns="perturb_rate", values="best_threshold").sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    im = ax.imshow(pivot_th.values, aspect="auto", cmap="viridis", vmin=min(THRESHOLDS), vmax=max(THRESHOLDS))
    ax.set_xticks(range(len(pivot_th.columns)))
    ax.set_xticklabels([f"{x:.2f}" for x in pivot_th.columns])
    ax.set_yticks(range(len(pivot_th.index)))
    ax.set_yticklabels([str(int(y)) for y in pivot_th.index])
    ax.set_xlabel("Self-state perturbation rate")
    ax.set_ylabel("Repair cost")
    ax.set_title("Empirically best reliability threshold")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("threshold τ")
    for i in range(pivot_th.shape[0]):
        for j in range(pivot_th.shape[1]):
            val = pivot_th.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white" if val > 0.57 else "black")
    fig.tight_layout()
    fig.savefig(fig_dir / "mirror_observerhood_lab_iv_threshold_map.pdf")
    fig.savefig(fig_dir / "mirror_observerhood_lab_iv_threshold_map.png", dpi=220)
    plt.close(fig)

    # Curves of advantage against perturbation for selected costs.
    selected = [0, 2, 6, 10, 15]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for c in selected:
        sub = best[best.repair_cost == c].sort_values("perturb_rate")
        ax.plot(sub.perturb_rate, sub.delta_viability, marker="o", label=f"cost={c}")
    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set_xlabel("Self-state perturbation rate")
    ax.set_ylabel("Best Δ viability")
    ax.set_title("Threshold crossing: when Mirror reliability becomes beneficial")
    ax.legend(title="Repair cost", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "mirror_observerhood_lab_iv_threshold_crossing.pdf")
    fig.savefig(fig_dir / "mirror_observerhood_lab_iv_threshold_crossing.png", dpi=220)
    plt.close(fig)

    # Mean repairs for best threshold cells.
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for c in selected:
        sub = best[best.repair_cost == c].sort_values("perturb_rate")
        ax.plot(sub.perturb_rate, sub.mean_repairs, marker="o", label=f"cost={c}")
    ax.set_xlabel("Self-state perturbation rate")
    ax.set_ylabel("Mean repairs per episode")
    ax.set_title("Repair use under the best reliability threshold")
    ax.legend(title="Repair cost", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "mirror_observerhood_lab_iv_repair_profile.pdf")
    fig.savefig(fig_dir / "mirror_observerhood_lab_iv_repair_profile.png", dpi=220)
    plt.close(fig)

    # Self error reduction for best threshold cells.
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for c in selected:
        sub = best[best.repair_cost == c].sort_values("perturb_rate")
        reduction = sub.baseline_self_error - sub.mean_self_error
        ax.plot(sub.perturb_rate, reduction, marker="o", label=f"cost={c}")
    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set_xlabel("Self-state perturbation rate")
    ax.set_ylabel("Self-error reduction")
    ax.set_title("Reliability-gated repair reduces represented self-error")
    ax.legend(title="Repair cost", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "mirror_observerhood_lab_iv_self_error_reduction.pdf")
    fig.savefig(fig_dir / "mirror_observerhood_lab_iv_self_error_reduction.png", dpi=220)
    plt.close(fig)


def threshold_table(best: pd.DataFrame) -> pd.DataFrame:
    # Approximate minimal perturbation rate at which delta becomes positive for each repair cost.
    records = []
    for c, group in best.sort_values("perturb_rate").groupby("repair_cost"):
        pos = group[group.delta_viability > 0]
        records.append({
            "repair_cost": c,
            "minimal_positive_perturb_rate": float(pos.perturb_rate.min()) if len(pos) else np.nan,
            "max_delta_viability": float(group.delta_viability.max()),
            "best_rate_for_max_delta": float(group.loc[group.delta_viability.idxmax(), "perturb_rate"]),
            "threshold_at_max_delta": float(group.loc[group.delta_viability.idxmax(), "best_threshold"]),
        })
    return pd.DataFrame(records)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run Mirror Observerhood Lab IV: Minimal Reliability Thresholds in Viability-Constrained Agents.")
    parser.add_argument("--episodes", type=int, default=EPISODES, help="episodes/seeds per perturbation-cost-threshold cell; paper used 90")
    parser.add_argument("--seed", type=int, default=BASE_SEED, help="base random seed")
    parser.add_argument("--outdir", type=str, default="outputs/lab_iv", help="output directory")
    parser.add_argument("--no-figures", action="store_true", help="skip figure generation")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    data_dir = outdir / "data"
    fig_dir = outdir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Running Mirror Observerhood Lab IV parameter sweep...")
    df = run_all(episodes=args.episodes, base_seed=args.seed)
    summary, comp, best = summarize(df)
    thresh = threshold_table(best)
    df.to_csv(data_dir / "Mirror_Observerhood_Lab_IV_results.csv", index=False)
    summary.to_csv(data_dir / "Mirror_Observerhood_Lab_IV_summary_by_condition.csv", index=False)
    comp.to_csv(data_dir / "Mirror_Observerhood_Lab_IV_all_thresholds_advantage.csv", index=False)
    best.to_csv(data_dir / "Mirror_Observerhood_Lab_IV_best_threshold_advantage.csv", index=False)
    thresh.to_csv(data_dir / "Mirror_Observerhood_Lab_IV_threshold_summary.csv", index=False)
    if not args.no_figures:
        save_plots(summary, comp, best, fig_dir)
    print("Rows:", len(df))
    print("Best advantage by repair cost:")
    print(thresh.to_string(index=False))

if __name__ == "__main__":
    main()
