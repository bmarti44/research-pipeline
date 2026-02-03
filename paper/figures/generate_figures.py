#!/usr/bin/env python
"""
Generate Figures for Format Friction Paper

Creates publication-ready figures from experiment results.
Grayscale-friendly with distinguishable patterns.

Usage:
    python paper/figures/generate_figures.py experiments/results/signal_detection_*_judged.json

Output:
    paper/figures/fig1_study1_before_after.png
    paper/figures/fig2_ambiguity_interaction.png
    paper/figures/fig3_measurement_comparison.png
    paper/figures/fig4_hedging_breakdown.png
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Style configuration for publication
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'nl': '#2c3e50',       # Dark blue-gray
    'st': '#7f8c8d',       # Medium gray
    'regex': '#bdc3c7',    # Light gray
    'judge': '#34495e',    # Darker blue-gray
    'hedged': '#95a5a6',   # Medium-light gray
    'not_hedged': '#2c3e50',  # Dark
}
HATCHES = {
    'nl': '',
    'st': '///',
    'regex': '...',
    'judge': '',
}


def load_judged_results(path: Path) -> dict:
    """Load judged results JSON."""
    with open(path) as f:
        return json.load(f)


def fig1_study1_before_after(output_dir: Path):
    """Figure 1: Study 1 before/after prompt correction.

    Simple grouped bar chart showing the 9pp gap with confounded prompts
    and the null after correction.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    # Data (from Study 1 results - Table in §3.3)
    # With Confound: NL 89.8% (229/255), ST 80.8% (206/255)
    # After Correction: NL 100%, ST 100% (n=5 each, preliminary)
    conditions = ['Confounded\nPrompts', 'Corrected\nPrompts*']
    nl_values = [0.898, 1.00]
    st_values = [0.808, 1.00]

    x = np.arange(len(conditions))
    width = 0.35

    bars1 = ax.bar(x - width/2, nl_values, width, label='NL',
                   color=COLORS['nl'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, st_values, width, label='Structured',
                   color=COLORS['st'], edgecolor='black', linewidth=0.5,
                   hatch='///')

    # Add gap annotations
    for i, (nl, st) in enumerate(zip(nl_values, st_values)):
        gap = nl - st
        y_pos = max(nl, st) + 0.02
        ax.annotate(f'Gap: {gap:+.0%}',
                   xy=(i, y_pos),
                   ha='center', fontsize=10,
                   fontweight='bold' if abs(gap) > 0.05 else 'normal')

    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title('Study 1: Memory Persistence\nPrompt Asymmetry Confounds Format Effects', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')

    # Add horizontal line at 80%
    ax.axhline(y=0.80, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    # Add footnote for preliminary correction data
    ax.text(0.5, -0.12, '*Corrected: preliminary validation (n=5 per condition)',
            transform=ax.transAxes, ha='center', fontsize=8, style='italic')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_study1_before_after.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig1_study1_before_after.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: fig1_study1_before_after.png/pdf")


def fig2_ambiguity_interaction(results: list[dict], output_dir: Path):
    """Figure 2: Ambiguity interaction plot.

    Two panels: regex scoring (left) and judge scoring (right).
    Shows how the interaction effect disappears under judge scoring.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    ambiguity_levels = ['EXPLICIT', 'IMPLICIT', 'BORDERLINE']

    for ax_idx, (ax, use_judge, title) in enumerate([
        (axes[0], False, 'Regex Scoring'),
        (axes[1], True, 'Judge Scoring (Primary)')
    ]):
        # Compute recall by ambiguity
        amb_data = {amb: {'nl': [], 'st': []} for amb in ambiguity_levels}

        for r in results:
            amb = r.get('ambiguity')
            if amb not in ambiguity_levels:
                continue
            if r.get('expected_detection') is not True:
                continue

            if use_judge:
                if r.get('nl_judge_detected') is True:
                    amb_data[amb]['nl'].append(1)
                else:
                    amb_data[amb]['nl'].append(0)
                if r.get('st_judge_detected') is True:
                    amb_data[amb]['st'].append(1)
                else:
                    amb_data[amb]['st'].append(0)
            else:
                if r.get('nl_regex_detected') is True:
                    amb_data[amb]['nl'].append(1)
                else:
                    amb_data[amb]['nl'].append(0)
                if r.get('st_regex_detected') is True:
                    amb_data[amb]['st'].append(1)
                else:
                    amb_data[amb]['st'].append(0)

        nl_means = [np.mean(amb_data[amb]['nl']) if amb_data[amb]['nl'] else 0
                    for amb in ambiguity_levels]
        st_means = [np.mean(amb_data[amb]['st']) if amb_data[amb]['st'] else 0
                    for amb in ambiguity_levels]

        x = np.arange(len(ambiguity_levels))

        ax.plot(x, nl_means, 'o-', color=COLORS['nl'], label='NL',
                markersize=8, linewidth=2)
        ax.plot(x, st_means, 's--', color=COLORS['st'], label='Structured',
                markersize=8, linewidth=2)

        ax.set_xticks(x)
        ax.set_xticklabels(ambiguity_levels, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.set_title(title, fontsize=12)
        ax.legend(loc='lower left')

        # Add gap annotations
        for i, (nl, st) in enumerate(zip(nl_means, st_means)):
            gap = nl - st
            y_pos = min(nl, st) - 0.08
            ax.annotate(f'{gap:+.0%}', xy=(i, y_pos), ha='center', fontsize=9)

    axes[0].set_ylabel('Recall', fontsize=12)
    fig.suptitle('Ambiguity Interaction: Regex vs Judge Scoring', fontsize=13, y=1.02)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_ambiguity_interaction.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig2_ambiguity_interaction.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: fig2_ambiguity_interaction.png/pdf")


def fig3_measurement_comparison(results: list[dict], output_dir: Path):
    """Figure 3: Measurement comparison bar chart.

    Grouped bars showing NL and ST recall under regex vs judge scoring.
    Visual demonstration that the gap exists under regex and vanishes under judge.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    # Compute recalls
    with_truth = [r for r in results
                  if r.get('ambiguity') in ['EXPLICIT', 'IMPLICIT']
                  and r.get('expected_detection') is True]

    n = len(with_truth)

    nl_regex = sum(1 for r in with_truth if r.get('nl_regex_detected') is True) / n
    st_regex = sum(1 for r in with_truth if r.get('st_regex_detected') is True) / n
    nl_judge = sum(1 for r in with_truth if r.get('nl_judge_detected') is True) / n
    st_judge = sum(1 for r in with_truth if r.get('st_judge_detected') is True) / n

    x = np.arange(2)  # Regex, Judge
    width = 0.35

    bars_nl = ax.bar(x - width/2, [nl_regex, nl_judge], width,
                     label='NL', color=COLORS['nl'], edgecolor='black', linewidth=0.5)
    bars_st = ax.bar(x + width/2, [st_regex, st_judge], width,
                     label='Structured', color=COLORS['st'], edgecolor='black',
                     linewidth=0.5, hatch='///')

    # Add value labels
    for bars in [bars_nl, bars_st]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1%}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=10)

    # Add gap annotations
    regex_gap = nl_regex - st_regex
    judge_gap = nl_judge - st_judge

    ax.annotate(f'Gap: {regex_gap:+.1%}\np<0.001',
               xy=(0, max(nl_regex, st_regex) + 0.08),
               ha='center', fontsize=10, fontweight='bold',
               color='#c0392b')  # Red for significant

    ax.annotate(f'Gap: {judge_gap:+.1%}\np=0.005',
               xy=(1, max(nl_judge, st_judge) + 0.08),
               ha='center', fontsize=10, fontweight='bold',
               color='#e67e22')  # Orange for significant but smaller

    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title('Measurement Method Determines Apparent Effect', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['Regex Scoring', 'Judge Scoring\n(Primary)'], fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_measurement_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_measurement_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: fig3_measurement_comparison.png/pdf")


def fig4_hedging_breakdown(results: list[dict], output_dir: Path):
    """Figure 4: Hedging breakdown stacked bar.

    Shows structured-condition failures broken down into:
    - No detection (judge also says no)
    - NL acknowledgment present (judge-confirmed)

    The 46% slice is the visual punchline.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # Compute hedging
    with_truth = [r for r in results
                  if r.get('ambiguity') in ['EXPLICIT', 'IMPLICIT']
                  and r.get('expected_detection') is True]

    # Structured trials with NO XML tag
    st_regex_failures = [r for r in with_truth if r.get('st_regex_detected') is not True]
    total_failures = len(st_regex_failures)

    # Of those, judge detected acknowledgment
    hedged = sum(1 for r in st_regex_failures if r.get('st_judge_detected') is True)
    not_hedged = total_failures - hedged

    hedging_rate = hedged / total_failures if total_failures > 0 else 0

    # Create stacked bar
    categories = ['Structured\nRegex Failures']

    ax.bar(categories, [not_hedged], label='No acknowledgment detected',
           color=COLORS['not_hedged'], edgecolor='black', linewidth=0.5)
    ax.bar(categories, [hedged], bottom=[not_hedged],
           label='NL acknowledgment present\n(judge-confirmed)',
           color=COLORS['hedged'], edgecolor='black', linewidth=0.5,
           hatch='///')

    # Add count labels
    ax.annotate(f'{not_hedged}\n({(not_hedged/total_failures):.0%})',
               xy=(0, not_hedged / 2),
               ha='center', va='center', fontsize=11, fontweight='bold')
    ax.annotate(f'{hedged}\n({hedging_rate:.0%})',
               xy=(0, not_hedged + hedged / 2),
               ha='center', va='center', fontsize=11, fontweight='bold')

    ax.set_ylabel('Number of Trials', fontsize=12)
    ax.set_title('Output Compliance Analysis\nStructured "Failures" Often Contain Detection', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)

    # Add text annotation
    ax.text(0.5, 0.02, f'Total structured regex failures: {total_failures}\n'
                       f'{hedging_rate:.0%} contained natural language acknowledgment',
            transform=ax.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_hedging_breakdown.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig4_hedging_breakdown.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: fig4_hedging_breakdown.png/pdf")


def main():
    parser = argparse.ArgumentParser(
        description="Generate figures for format friction paper"
    )
    parser.add_argument(
        "results_file",
        type=Path,
        nargs='?',
        default=None,
        help="Path to judged results JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures"),
        help="Output directory for figures (default: paper/figures)"
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1 doesn't need data file (uses fixed Study 1 numbers)
    print("Generating Figure 1: Study 1 before/after...")
    fig1_study1_before_after(args.output_dir)

    # Figures 2-4 need data file
    if args.results_file is None:
        # Try to find most recent judged file
        results_dir = Path("experiments/results")
        judged_files = list(results_dir.glob("*_judged.json"))
        if judged_files:
            args.results_file = max(judged_files, key=lambda p: p.stat().st_mtime)
            print(f"Using most recent judged file: {args.results_file}")
        else:
            print("No judged results file found. Skipping figures 2-4.")
            print("Run: python experiments/judge_scoring.py <results.json>")
            return

    print(f"Loading results from {args.results_file}...")
    data = load_judged_results(args.results_file)
    results = data.get("results", [])
    print(f"Found {len(results)} trial records")

    print("Generating Figure 2: Ambiguity interaction...")
    fig2_ambiguity_interaction(results, args.output_dir)

    print("Generating Figure 3: Measurement comparison...")
    fig3_measurement_comparison(results, args.output_dir)

    print("Generating Figure 4: Hedging breakdown...")
    fig4_hedging_breakdown(results, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
