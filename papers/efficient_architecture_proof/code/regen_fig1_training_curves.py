"""Regenerate fig1_training_curves.png from recovered training logs.

Parses per-epoch validation accuracy from all 4 model logs.
Uses paper M-numbers: M1=CoT, M2=COCONUT, M3=Pause, M4=Pause-Multipass.

Log files (Lambda-era naming → paper naming):
  m1_cot.log        → M1 (CoT)
  m3_coconut.log    → M2 (COCONUT)
  m5_pause.log      → M3 (Pause)
  m4_training.log   → M4 (Pause-Multipass)
"""

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Colorblind-friendly palette (IBM Design Library) — matches generate_figures.py
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

# Lambda-era log filenames → paper M-numbers
LOG_MAP = {
    "M1": "m1_cot.log",
    "M2": "m3_coconut.log",
    "M3": "m5_pause.log",
    "M4": "m4_training.log",
}


def parse_training_log(log_path):
    """Extract per-epoch validation accuracy from a training log."""
    accuracies = []
    with open(log_path, "r") as f:
        for line in f:
            m = re.search(
                r"Accuracy on validation set:\s*(\d+)\s*/\s*(\d+)\s*=\s*([\d.]+)",
                line,
            )
            if m:
                accuracies.append(float(m.group(3)))
    return accuracies


def main():
    logs_dir = os.path.join(
        os.path.dirname(__file__), "..", "results", "logs"
    )
    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "manuscript", "figures"
    )

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
    max_epochs = 0

    for model in ["M1", "M2", "M3", "M4"]:
        log_path = os.path.join(logs_dir, LOG_MAP[model])
        if not os.path.exists(log_path):
            print(f"  WARNING: No log for {model} at {log_path}")
            continue

        accs = parse_training_log(log_path)
        if not accs:
            print(f"  WARNING: No accuracy data in {log_path}")
            continue

        epochs = list(range(1, len(accs) + 1))
        max_epochs = max(max_epochs, len(accs))
        ax.plot(
            epochs, accs, "-", linewidth=1.5,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
        print(f"  {model}: {len(accs)} epochs, final={accs[-1]:.4f}")

    # Stage boundaries (epochs_per_stage = 5)
    stage_boundaries = [5, 10, 15, 20, 25, 30]
    stage_labels = ["S0\u2192S1", "S1\u2192S2", "S2\u2192S3",
                    "S3\u2192S4", "S4\u2192S5", "S5\u2192S6"]
    y_top = ax.get_ylim()[1]
    for boundary, label in zip(stage_boundaries, stage_labels):
        if boundary <= max_epochs + 5:
            ax.axvline(x=boundary + 0.5, color="gray", linestyle="--",
                       linewidth=0.7, alpha=0.5)
            ax.text(boundary + 0.8, y_top * 0.97, label,
                    fontsize=7, color="gray", rotation=90, va="top")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Training Curves: COCONUT Curriculum Stages")
    ax.set_xlim(0, max(max_epochs + 1, 51))
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    for fmt in ["png", "pdf"]:
        path = os.path.join(output_dir, f"fig1_training_curves.{fmt}")
        fig.savefig(path, format=fmt)
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
