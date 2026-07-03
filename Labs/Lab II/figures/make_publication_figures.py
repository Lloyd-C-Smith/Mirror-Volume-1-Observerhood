#!/usr/bin/env python3
"""Regenerate publication figures for Mirror Observerhood Lab II from packaged data."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Mirror_Observerhood_Lab_II_results.csv"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

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


def ordered_pivot(df: pd.DataFrame, value: str) -> pd.DataFrame:
    pivot = df.pivot_table(index="perturbation", columns="agent", values=value, aggfunc="mean")
    pivot = pivot.reindex(PERTURBATION_ORDER).reindex(columns=AGENT_ORDER)
    pivot.index = [PERTURBATION_LABELS[x] for x in PERTURBATION_ORDER]
    pivot.columns = [AGENT_LABELS[x] for x in AGENT_ORDER]
    return pivot


def apply_patterns(ax) -> None:
    hatches = ["", "///", "...", "xxx"]
    colors = ["0.82", "0.65", "0.50", "0.25"]
    for i, patch in enumerate(ax.patches):
        group = i // len(PERTURBATION_ORDER)
        patch.set_facecolor(colors[group % len(colors)])
        patch.set_edgecolor("black")
        patch.set_linewidth(0.55)
        patch.set_hatch(hatches[group % len(hatches)])


def save_grouped_bar(df: pd.DataFrame, value: str, ylabel: str, title: str, filename: str, scale: float = 1.0) -> None:
    pivot = ordered_pivot(df, value) * scale
    ax = pivot.plot(kind="bar", figsize=(11, 5.5), color=["0.82", "0.65", "0.50", "0.25"], edgecolor="black", linewidth=0.55)
    apply_patterns(ax)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Perturbation condition")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.31), ncol=4, frameon=False)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT / f"{filename}.png", dpi=300)
    plt.savefig(OUT / f"{filename}.pdf")
    plt.close()


def save_advantage(df: pd.DataFrame) -> None:
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
    ax.bar(x - width/2, [r[1] for r in rows], width, label="vs scalar Mirror", color="0.62", edgecolor="black", hatch="///", linewidth=0.55)
    ax.bar(x + width/2, [r[2] for r in rows], width, label="vs best non-decomposed", color="0.35", edgecolor="black", hatch="xxx", linewidth=0.55)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean viability advantage")
    ax.set_xlabel("Perturbation condition")
    ax.set_title("Decomposed reliability advantage")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(OUT / "mirror_observerhood_lab_ii_decomposed_advantage.png", dpi=300)
    plt.savefig(OUT / "mirror_observerhood_lab_ii_decomposed_advantage.pdf")
    plt.close()


def main() -> None:
    df = pd.read_csv(DATA)
    save_grouped_bar(df, "viability", "Mean viability", "Mean viability by perturbation and architecture", "mirror_observerhood_lab_ii_viability")
    save_grouped_bar(df, "survived", "Survival rate (%)", "Survival rate by perturbation and architecture", "mirror_observerhood_lab_ii_survival", scale=100.0)
    save_advantage(df)


if __name__ == "__main__":
    main()
