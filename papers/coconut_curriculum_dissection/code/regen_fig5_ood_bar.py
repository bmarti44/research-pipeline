"""Regenerate fig5_ood_bar.png with all 4 models and paper M-numbers.

Data from paper.yaml and results/experiments/m6/accuracy.json.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Colorblind-friendly palette (IBM Design Library)
MODEL_COLORS = {
    "M1": "#648FFF",   # blue
    "M2": "#DC267F",   # magenta
    "M3": "#44AA99",   # teal
    "M4": "#FE6100",   # orange
}

MODEL_LABELS = {
    "M1": "M1 (CoT)",
    "M2": "M2 (COCONUT)",
    "M3": "M3 (Pause)",
    "M4": "M4 (Pause-Multipass)",
}

# Data from paper.yaml (verified against results JSONs)
# Order: ProsQA (ID), 7-hop, 8-hop, DAG, Dense
OOD_DATA = {
    "M1": [0.830, 0.107, 0.082, 0.282, 0.141],
    "M2": [0.970, 0.660, 0.675, 0.592, 0.612],
    "M3": [0.966, 0.754, 0.751, 0.519, 0.684],
    "M4": [0.948, 0.769, 0.752, 0.598, 0.648],
}

OOD_LABELS = ["ProsQA\n(ID)", "7-hop", "8-hop", "DAG", "Dense"]

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
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

fig, ax = plt.subplots(figsize=(7, 3.5))

models = ["M1", "M2", "M3", "M4"]
n_groups = len(OOD_LABELS)
n_bars = len(models)
bar_width = min(0.15, 0.8 / n_bars)
x = np.arange(n_groups)

for i, model in enumerate(models):
    vals = OOD_DATA[model]
    offset = (i - n_bars / 2 + 0.5) * bar_width
    ax.bar(
        x + offset, vals, bar_width,
        label=MODEL_LABELS[model],
        color=MODEL_COLORS[model],
        edgecolor="white", linewidth=0.5,
        alpha=0.9,
    )

ax.axhline(y=0.1, color="gray", linestyle="--", linewidth=0.8,
           alpha=0.5, label="Chance (~10%)")

ax.set_xlabel("Test Set")
ax.set_ylabel("Accuracy")
ax.set_title("OOD Generalization Across Test Sets")
ax.set_xticks(x)
ax.set_xticklabels(OOD_LABELS)
ax.set_ylim(0, 1.05)
ax.legend(loc="upper right", ncol=2, fontsize=8)
ax.grid(axis="y", alpha=0.3)

import os
output_dir = os.path.join(os.path.dirname(__file__), "..", "manuscript", "figures")
for fmt in ["png", "pdf"]:
    path = os.path.join(output_dir, f"fig5_ood_bar.{fmt}")
    fig.savefig(path, format=fmt)
    print(f"Saved: {path}")
plt.close(fig)
