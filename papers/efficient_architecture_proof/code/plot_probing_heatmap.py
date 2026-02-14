"""
Regenerate fig4_probing_heatmap with Bonferroni significance markers.

Reads corrected permutation test results from m3_linear_perm.json and
m5_linear_perm.json, plots probe accuracy heatmaps with asterisks on
cells that pass Bonferroni correction (p < 0.05/78 = 0.000641).

Usage:
    python plot_probing_heatmap.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Style
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


def load_model(name):
    path = os.path.join(RESULTS_DIR, f"{name}_linear_perm.json")
    with open(path) as f:
        return json.load(f)


def main():
    m3 = load_model("m3")
    m5 = load_model("m5")

    panels = [
        ("M3 (COCONUT)", m3),
        ("M5 (Pause-Curriculum)", m5),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
    im = None

    # Global color range
    all_acc = []
    for _, data in panels:
        all_acc.extend(np.array(data["linear_probe_accuracy"]).flatten().tolist())
    vmin, vmax = 0, max(all_acc)

    for i, (label, data) in enumerate(panels):
        ax = axes[i]
        acc = np.array(data["linear_probe_accuracy"])   # (13 layers, 6 positions)
        sig = np.array(data["significant_cells"])        # (13 layers, 6 positions)

        im = ax.imshow(
            acc, aspect="auto", origin="lower",
            cmap="viridis", vmin=vmin, vmax=vmax,
            interpolation="nearest",
        )

        # Add significance markers
        n_layers, n_pos = acc.shape
        for layer in range(n_layers):
            for pos in range(n_pos):
                if sig[layer, pos]:
                    # White asterisk for significant cells
                    # Use black for light cells, white for dark cells
                    val = acc[layer, pos]
                    color = "black" if val > (vmax * 0.6) else "white"
                    ax.text(
                        pos, layer, "*",
                        ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color=color,
                    )

        ax.set_xlabel("Thought Position")
        if i == 0:
            ax.set_ylabel("Layer")
        ax.set_title(label, fontsize=10)

        # Tick labels
        ax.set_xticks(range(n_pos))
        ax.set_yticks(range(n_layers))

    fig.colorbar(im, ax=list(axes), shrink=0.8, label="Probe Accuracy")

    fig.suptitle(
        "Information Surface: Linear Probe Accuracy by (Layer, Position)",
        fontsize=11, y=1.02,
    )

    # Annotation for significance
    fig.text(
        0.5, -0.06,
        "* Bonferroni-significant (p < 0.05/78 = 0.000641, 2,000 permutations)",
        ha="center", fontsize=8, style="italic", color="0.4",
    )

    os.makedirs(FIGURES_DIR, exist_ok=True)
    for fmt in ["png", "pdf"]:
        path = os.path.join(FIGURES_DIR, f"fig4_probing_heatmap.{fmt}")
        fig.savefig(path, format=fmt)
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
