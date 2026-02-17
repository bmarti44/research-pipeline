#!/usr/bin/env python3
"""Regenerate fig2_corruption_curves with correct paper M-numbers.

Data files use Lambda-era keys: m3=COCONUT (paper M2), m5=Pause (paper M3).
This script plots forward corruption curves with paper-correct labels.

Usage:
    python regen_fig2_corruption_curves.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

for _style in ["seaborn-v0_8-paper", "seaborn-paper"]:
    try:
        plt.style.use(_style)
        break
    except OSError:
        continue

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "manuscript", "figures")

# Paper M-numbers: M2 = COCONUT (Lambda m3), M3 = Pause (Lambda m5)
MODEL_COLORS = {
    "M2 (COCONUT)": "#DC267F",
    "M3 (Pause)":   "#44AA99",
}


def main():
    path = os.path.join(RESULTS_DIR, "experiments", "corruption", "results.json")
    with open(path) as f:
        data = json.load(f)

    # Lambda-era m3 = paper M2 (COCONUT), m5 = paper M3 (Pause)
    models = [
        ("M2 (COCONUT)", data["m3"]),
        ("M3 (Pause)",   data["m5"]),
    ]

    fig, ax = plt.subplots(figsize=(6, 4))

    for label, mdata in models:
        clean = mdata["clean_accuracy"]
        forward = mdata["forward_corruption"]
        # x-axis: number of positions corrupted (1 through 6)
        x = list(range(1, len(forward) + 1))
        # Prepend clean (0 corrupted)
        x = [0] + x
        y = [clean] + forward

        ax.plot(x, y, "-o", markersize=5, linewidth=2,
                color=MODEL_COLORS[label], label=label)

    ax.set_xlabel("Positions Corrupted (forward)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(range(7))
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    for fmt in ["png", "pdf"]:
        out = os.path.join(FIGURES_DIR, f"fig2_corruption_curves.{fmt}")
        fig.savefig(out, format=fmt)
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
