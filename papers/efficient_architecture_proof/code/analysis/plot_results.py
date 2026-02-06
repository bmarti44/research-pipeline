"""
Plotting script for LAHR experimental results.

Generates training curves and comparison plots.

Usage:
    python plot_results.py --results_dir checkpoints --output_dir manuscript/figures
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

# Check if matplotlib is available
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plotting disabled.")


def load_all_metrics(results_dir: str) -> Dict[str, Any]:
    """Load all metrics.json files from results directory."""
    results_path = Path(results_dir)
    all_metrics = {}

    for run_dir in results_path.iterdir():
        if not run_dir.is_dir():
            continue

        metrics_file = run_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
                all_metrics[run_dir.name] = metrics

    return all_metrics


def plot_training_curves(all_metrics: Dict[str, Any], output_dir: str):
    """Plot training loss curves for all conditions."""
    if not HAS_MATPLOTLIB:
        print("Skipping plots - matplotlib not available")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Set2(np.linspace(0, 1, 8))
    color_map = {
        'full': colors[0],
        'baseline': colors[1],
        'mod_only': colors[2],
        'latent_only': colors[3],
        'memory_only': colors[4],
        'no_mod': colors[5],
        'no_latent': colors[6],
        'no_memory': colors[7],
    }

    for run_name, metrics in all_metrics.items():
        condition = metrics.get("condition", run_name)
        history = metrics.get("history", [])

        if not history:
            continue

        steps = [h["step"] for h in history]
        losses = [h["loss"] for h in history]

        color = color_map.get(condition, 'gray')
        ax.plot(steps, losses, label=condition, color=color, alpha=0.8)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('LAHR Training Curves by Condition')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved training curves to {output_path / 'training_curves.png'}")


def plot_final_comparison(all_metrics: Dict[str, Any], output_dir: str):
    """Plot bar chart comparing final metrics across conditions."""
    if not HAS_MATPLOTLIB:
        return

    # Group by condition
    by_condition = {}
    for run_name, metrics in all_metrics.items():
        condition = metrics.get("condition", run_name)
        best_loss = metrics.get("best_val_loss", float("inf"))

        if best_loss == float("inf"):
            # Use last training loss if no validation
            history = metrics.get("history", [])
            if history:
                best_loss = history[-1].get("loss", float("inf"))

        if condition not in by_condition:
            by_condition[condition] = []
        if best_loss < float("inf"):
            by_condition[condition].append(best_loss)

    if not by_condition:
        print("No valid results to plot")
        return

    # Compute means and stds
    conditions = sorted(by_condition.keys())
    means = [np.mean(by_condition[c]) if by_condition[c] else 0 for c in conditions]
    stds = [np.std(by_condition[c]) if len(by_condition[c]) > 1 else 0 for c in conditions]

    # Convert to perplexity
    ppl_means = [np.exp(m) if m > 0 else 0 for m in means]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(conditions))
    bars = ax.bar(x, ppl_means, yerr=[np.exp(m) * s for m, s in zip(means, stds)],
                  capsize=5, alpha=0.8)

    ax.set_xlabel('Condition')
    ax.set_ylabel('Perplexity')
    ax.set_title('LAHR Final Perplexity by Condition')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path / 'perplexity_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved comparison to {output_path / 'perplexity_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description="Plot LAHR experiment results")
    parser.add_argument("--results_dir", type=str, default="checkpoints",
                        help="Directory containing training results")
    parser.add_argument("--output_dir", type=str, default="manuscript/figures",
                        help="Directory for output figures")

    args = parser.parse_args()

    # Load all metrics
    print(f"Loading results from {args.results_dir}...")
    all_metrics = load_all_metrics(args.results_dir)

    if not all_metrics:
        print("No results found!")
        return

    print(f"Found {len(all_metrics)} runs")

    # Generate plots
    plot_training_curves(all_metrics, args.output_dir)
    plot_final_comparison(all_metrics, args.output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
