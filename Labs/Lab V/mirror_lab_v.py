#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mirror Observerhood Lab V: Recursive Reliability Under Estimator Corruption

Single-file reproducibility script for the Mirror Programme Observerhood Labs.
Run with: python mirror_lab_v.py --help
"""
from __future__ import annotations

"""Mirror Observerhood Lab V: Recursive reliability under estimator corruption.

This script implements a minimal viability-constrained decision process. The
system has a hidden self-relevant state z and an internal self-model m. Acting
while m=z produces viability gain; acting while m!=z causes viability loss. A
first-order reliability variable R estimates whether m is accurate. In some
conditions, R itself is corrupted. The recursive agent uses a second-order
consistency signal to decide when R should be trusted, repaired, or ignored.
"""
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("outputs") / "lab_v"
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

HORIZON = 80
AGENTS = ["NoReliability", "FirstOrderR", "RecursiveR", "Oracle"]

@dataclass(frozen=True)
class Condition:
    name: str
    label: str
    self_corrupt_rate: float
    blind_rate: float       # high R despite wrong self-model
    false_alarm_rate: float # low R despite correct self-model
    residual_noise: float
    repair_cost: float
    diagnostic_cost: float
    meta_cost: float

CONDITIONS = [
    Condition("control", "Control", 0.00, 0.00, 0.00, 0.03, 3.0, 0.8, 0.2),
    Condition("truthful_self_perturbation", "Truthful self perturbation", 0.13, 0.00, 0.00, 0.05, 3.0, 0.8, 0.2),
    Condition("reliability_blindness", "Reliability blindness", 0.14, 0.35, 0.00, 0.05, 3.0, 0.8, 0.2),
    Condition("estimator_corruption", "Estimator corruption", 0.18, 0.55, 0.00, 0.06, 3.0, 0.8, 0.2),
    Condition("false_alarm_reliability", "False alarms", 0.04, 0.00, 0.33, 0.05, 3.0, 0.8, 0.2),
    Condition("noisy_residual", "Noisy residual", 0.10, 0.30, 0.10, 0.34, 3.0, 0.8, 0.2),
    Condition("high_meta_cost", "High meta cost", 0.18, 0.55, 0.00, 0.06, 3.0, 4.0, 2.5),
]


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def other_state(z: int) -> int:
    return 1 - z


def sample_reliability(correct: bool, cond: Condition, rng: random.Random) -> float:
    if correct:
        r = rng.uniform(0.78, 0.98)
        if rng.random() < cond.false_alarm_rate:
            r = rng.uniform(0.04, 0.34)
    else:
        r = rng.uniform(0.04, 0.34)
        if rng.random() < cond.blind_rate:
            r = rng.uniform(0.74, 0.98)
    return r


def sample_residual(correct: bool, cond: Condition, rng: random.Random) -> float:
    # residual is a fallible consistency signal: high means action predictions are not matching feedback
    base = rng.uniform(0.02, 0.18) if correct else rng.uniform(0.68, 0.94)
    return clamp01(base + rng.gauss(0.0, cond.residual_noise))


def run_episode(agent: str, cond: Condition, seed: int) -> Dict:
    rng = random.Random(seed)
    z = rng.randrange(2)  # true self-relevant mode
    m = z                 # internal self-model
    viability = 0.0
    energy = 100.0
    repairs = diagnostics = wrong_acts = correct_acts = conflicts = 0
    self_error_sum = 0.0
    r_sum = residual_sum = 0.0

    for t in range(HORIZON):
        # Hidden self-state can change/corrupt the agent's internal self-model.
        if rng.random() < cond.self_corrupt_rate:
            m = other_state(z)

        correct = (m == z)
        r = sample_reliability(correct, cond, rng)
        residual = sample_residual(correct, cond, rng)
        r_sum += r
        residual_sum += residual
        self_error_sum += 0 if correct else 1

        if (r > 0.65 and residual > 0.55) or (r < 0.38 and residual < 0.20):
            conflicts += 1

        # Agent policy.
        if agent == "Oracle":
            if not correct:
                viability -= cond.repair_cost
                energy -= cond.repair_cost
                repairs += 1
                m = z
                correct = True

        elif agent == "FirstOrderR":
            if r < 0.55:
                viability -= cond.repair_cost
                energy -= cond.repair_cost
                repairs += 1
                m = z
                correct = True

        elif agent == "RecursiveR":
            # Maintain the recursive monitor at small cost. It pays off only when R is often wrong.
            viability -= cond.meta_cost
            energy -= cond.meta_cost
            blind_conflict = (r > 0.65 and residual > 0.55)       # R says OK, evidence says not OK
            false_alarm_conflict = (r < 0.38 and residual < 0.20) # R says broken, evidence says OK

            if blind_conflict:
                diagnostics += 1
                viability -= cond.diagnostic_cost
                energy -= cond.diagnostic_cost
                # Diagnostic confirms high residual is likely self-model error, then repair.
                if not correct:
                    repairs += 1
                    viability -= cond.repair_cost
                    energy -= cond.repair_cost
                    m = z
                    correct = True
            elif false_alarm_conflict:
                diagnostics += 1
                viability -= cond.diagnostic_cost
                energy -= cond.diagnostic_cost
                # Suppress costly first-order false repair.
                pass
            elif r < 0.55 and residual > 0.28:
                repairs += 1
                viability -= cond.repair_cost
                energy -= cond.repair_cost
                m = z
                correct = True
            elif r < 0.55 and residual <= 0.28:
                # low R but no evidence of failure: no repair, small audit cost only
                diagnostics += 1
                viability -= cond.diagnostic_cost
                energy -= cond.diagnostic_cost

        # Act according to current self-model. Correct self-model allows coherent action.
        if m == z:
            correct_acts += 1
            viability += 2.0
            energy += 0.25
            # success sometimes changes the true mode; the self-model updates on ordinary feedback
            if rng.random() < 0.035:
                z = other_state(z)
                m = z
        else:
            wrong_acts += 1
            viability -= 7.0
            energy -= 6.0
            # Wrong actions sometimes create enough feedback to correct the self-model next step.
            if rng.random() < 0.18:
                m = z

        if energy <= 0:
            break

    steps = t + 1
    survived = int(energy > 0 and steps == HORIZON)
    return {
        "agent": agent,
        "condition": cond.name,
        "condition_label": cond.label,
        "seed": seed,
        "viability": viability,
        "survived": survived,
        "steps": steps,
        "energy": energy,
        "repairs": repairs,
        "diagnostics": diagnostics,
        "wrong_acts": wrong_acts,
        "correct_acts": correct_acts,
        "mean_self_error": self_error_sum / steps,
        "mean_reliability": r_sum / steps,
        "mean_residual": residual_sum / steps,
        "conflicts": conflicts,
        "self_corrupt_rate": cond.self_corrupt_rate,
        "blind_rate": cond.blind_rate,
        "false_alarm_rate": cond.false_alarm_rate,
        "residual_noise": cond.residual_noise,
        "repair_cost": cond.repair_cost,
        "diagnostic_cost": cond.diagnostic_cost,
        "meta_cost": cond.meta_cost,
    }


def run_all(n_per_cell: int = 1400, seed0: int = 780000) -> pd.DataFrame:
    rows: List[Dict] = []
    k = 0
    for cond in CONDITIONS:
        for agent in AGENTS:
            for _ in range(n_per_cell):
                rows.append(run_episode(agent, cond, seed0 + k))
                k += 1
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["condition", "condition_label", "agent"], as_index=False).agg(
        mean_viability=("viability", "mean"),
        sd_viability=("viability", "std"),
        survival_rate=("survived", "mean"),
        mean_steps=("steps", "mean"),
        mean_repairs=("repairs", "mean"),
        mean_diagnostics=("diagnostics", "mean"),
        mean_wrong_acts=("wrong_acts", "mean"),
        mean_self_error=("mean_self_error", "mean"),
        mean_conflicts=("conflicts", "mean"),
        n=("seed", "count"),
    )


def advantage(summary: pd.DataFrame) -> pd.DataFrame:
    piv = summary.pivot_table(index=["condition", "condition_label"], columns="agent", values="mean_viability").reset_index()
    piv["recursive_minus_first_order"] = piv["RecursiveR"] - piv["FirstOrderR"]
    piv["recursive_minus_no_reliability"] = piv["RecursiveR"] - piv["NoReliability"]
    piv["oracle_gap"] = piv["Oracle"] - piv["RecursiveR"]
    return piv


def bar(summary: pd.DataFrame, metric: str, ylabel: str, filename: str) -> None:
    labels = [c.label for c in CONDITIONS]
    x = np.arange(len(labels))
    width = 0.20
    fig, ax = plt.subplots(figsize=(11.7, 5.3))
    for j, agent in enumerate(AGENTS):
        vals = []
        for c in CONDITIONS:
            vals.append(float(summary[(summary.condition == c.name) & (summary.agent == agent)][metric].iloc[0]))
        ax.bar(x + (j - 1.5) * width, vals, width, label=agent)
    ax.set_title(f"{ylabel} by condition")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=23, ha="right")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=220)
    if filename.lower().endswith('.png'):
        fig.savefig(FIG_DIR / (filename[:-4] + '.pdf'))
    plt.close(fig)


def line_recursive_profile(summary: pd.DataFrame) -> None:
    labels = [c.label for c in CONDITIONS]
    rec = summary[summary.agent == "RecursiveR"].set_index("condition")
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, [float(rec.loc[c.name, "mean_diagnostics"]) for c in CONDITIONS], marker="o", label="Diagnostics")
    ax.plot(x, [float(rec.loc[c.name, "mean_repairs"]) for c in CONDITIONS], marker="o", label="Repairs")
    ax.plot(x, [float(rec.loc[c.name, "mean_conflicts"]) for c in CONDITIONS], marker="o", label="R-residual conflicts")
    ax.set_title("Recursive agent: audits, repairs, and reliability conflicts")
    ax.set_ylabel("Mean count per episode")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=23, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mirror_observerhood_lab_v_recursive_profile.png", dpi=220)
    fig.savefig(FIG_DIR / "mirror_observerhood_lab_v_recursive_profile.pdf")
    plt.close(fig)


def plot_advantage(adv: pd.DataFrame) -> None:
    labels = adv["condition_label"].tolist()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhline(0, linewidth=1)
    ax.bar(x, adv["recursive_minus_first_order"].to_numpy())
    ax.set_title("Recursive reliability advantage over first-order reliability")
    ax.set_ylabel("Mean viability difference")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=23, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mirror_observerhood_lab_v_recursive_advantage.png", dpi=220)
    fig.savefig(FIG_DIR / "mirror_observerhood_lab_v_recursive_advantage.pdf")
    plt.close(fig)


def plot_oracle_gap(adv: pd.DataFrame) -> None:
    labels = adv["condition_label"].tolist()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, adv["oracle_gap"].to_numpy())
    ax.set_title("Remaining gap to oracle repair policy")
    ax.set_ylabel("Oracle minus recursive mean viability")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=23, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mirror_observerhood_lab_v_oracle_gap.png", dpi=220)
    fig.savefig(FIG_DIR / "mirror_observerhood_lab_v_oracle_gap.pdf")
    plt.close(fig)



def make_publication_figures(summary: pd.DataFrame, adv: pd.DataFrame) -> None:
    """Generate the publication figure set in PNG and PDF formats."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Main advantage and oracle/profile figures.
    plot_advantage(adv)
    plot_oracle_gap(adv)
    line_recursive_profile(summary)
    # Agent comparison plots.
    bar(summary, "mean_viability", "Mean viability", "mirror_observerhood_lab_v_viability_by_condition.png")
    bar(summary, "survival_rate", "Survival rate", "mirror_observerhood_lab_v_survival_rate_by_condition.png")
    bar(summary, "mean_self_error", "Mean self-model error", "mirror_observerhood_lab_v_self_error_by_condition.png")


def main() -> None:
    import argparse
    global ROOT, DATA_DIR, FIG_DIR
    parser = argparse.ArgumentParser(description="Run Mirror Observerhood Lab V: recursive reliability under estimator corruption.")
    parser.add_argument("--episodes", type=int, default=1400, help="episodes per agent-condition cell; paper used 1400")
    parser.add_argument("--seed", type=int, default=780000, help="base random seed")
    parser.add_argument("--outdir", type=str, default="outputs/lab_v", help="output directory")
    parser.add_argument("--no-figures", action="store_true", help="skip figure generation")
    args = parser.parse_args()
    ROOT = Path(args.outdir)
    DATA_DIR = ROOT / "data"
    FIG_DIR = ROOT / "figures"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = run_all(n_per_cell=args.episodes, seed0=args.seed)
    summary = summarize(df)
    adv = advantage(summary)
    # Preserve canonical condition order in output CSVs.
    order = {c.name: i for i, c in enumerate(CONDITIONS)}
    df["condition_order"] = df["condition"].map(order)
    df = df.sort_values(["condition_order", "agent", "seed"]).drop(columns=["condition_order"])
    summary["condition_order"] = summary["condition"].map(order)
    summary = summary.sort_values(["condition_order", "agent"]).drop(columns=["condition_order"])
    adv["condition_order"] = adv["condition"].map(order)
    adv = adv.sort_values("condition_order").drop(columns=["condition_order"])
    df.to_csv(DATA_DIR / "Mirror_Observerhood_Lab_V_results.csv", index=False)
    summary.to_csv(DATA_DIR / "Mirror_Observerhood_Lab_V_summary_by_condition.csv", index=False)
    adv.to_csv(DATA_DIR / "Mirror_Observerhood_Lab_V_recursive_advantage.csv", index=False)
    if not args.no_figures:
        make_publication_figures(summary, adv)
    print("episodes", len(df))
    print("\nAdvantage")
    print(adv.to_string(index=False))

if __name__ == "__main__":
    main()
