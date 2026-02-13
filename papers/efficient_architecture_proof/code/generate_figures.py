"""
Generate Figures for COCONUT Reasoning Study

Generates 7 publication-quality figures from experiment results:
  fig1: Probing heatmap (information surface)
  fig2: OOD generalization bar chart
  fig3: Corruption cascade curves
  fig4: Causal effect heatmap
  fig5: Curriculum training divergence curves
  fig6: Token count ablation
  fig7: Cross-problem transplant matrix (appendix)

Usage:
    python generate_figures.py \
        --results_dir /path/to/results/ \
        --output_dir /path/to/figures/

Looks for experiments/*/results.json and statistical_analysis.json in results_dir.
Parses training logs from logs/ for fig5.
Handles missing data gracefully — skips figures when data is unavailable.

Dependencies: matplotlib, numpy, json, argparse, glob. No GPU, no torch.
"""

import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (safe for headless servers)
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------

_STYLE_APPLIED = False
for _style in ["seaborn-v0_8-paper", "seaborn-paper"]:
    try:
        plt.style.use(_style)
        _STYLE_APPLIED = True
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

# Colorblind-friendly palette (IBM Design Library)
MODEL_COLORS = {
    "m1": "#648FFF",   # blue
    "m2": "#785EF0",   # purple
    "m3": "#DC267F",   # magenta
    "m4": "#FE6100",   # orange
    "m4b": "#FFB000",  # gold
    "m5": "#44AA99",   # teal
}

MODEL_LABELS = {
    "m1": "M1 (CoT)",
    "m2": "M2 (Direct)",
    "m3": "M3 (COCONUT)",
    "m4": "M4 (Pause-Frozen)",
    "m4b": "M4b (Pause-Learned)",
    "m5": "M5 (Pause-Curriculum)",
}

OOD_TEST_SETS = ["prosqa_test", "ood_7hop", "ood_8hop", "ood_dag", "ood_dense"]
OOD_LABELS = ["ProsQA\n(ID)", "7-hop", "8-hop", "DAG", "Dense"]

ALL_MODELS = ["m1", "m3", "m5"]
THOUGHT_MODELS = ["m3", "m5"]  # Models with thought tokens

# Colors for OOD test sets (used in fig6)
OOD_COLORS = {
    "prosqa_test": "#648FFF",
    "ood_7hop": "#DC267F",
    "ood_8hop": "#FE6100",
    "ood_dag": "#785EF0",
    "ood_dense": "#FFB000",
}
OOD_SHORT_LABELS = {
    "prosqa_test": "ProsQA (ID)",
    "ood_7hop": "7-hop",
    "ood_8hop": "8-hop",
    "ood_dag": "DAG",
    "ood_dense": "Dense",
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_json(path):
    """Load JSON file, return None if missing."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def save_figure(fig, output_dir, name):
    """Save figure as both PDF and PNG."""
    for fmt in ["pdf", "png"]:
        path = os.path.join(output_dir, f"{name}.{fmt}")
        fig.savefig(path, format=fmt)
        print(f"  Saved: {path}")
    plt.close(fig)


def parse_coord_dict(d):
    """Parse dict with '(layer, position)' string keys into 2D numpy array."""
    if not d:
        return None
    max_l = max_p = 0
    entries = []
    for k, v in d.items():
        m = re.match(r"\((\d+),\s*(\d+)\)", k)
        if m:
            l, p = int(m.group(1)), int(m.group(2))
            entries.append((l, p, v))
            max_l = max(max_l, l)
            max_p = max(max_p, p)
    if not entries:
        return None
    matrix = np.full((max_l + 1, max_p + 1), np.nan)
    for l, p, v in entries:
        matrix[l, p] = v
    return matrix


# ---------------------------------------------------------------------------
# Figure 1: Probing Heatmap (Information Surface)
# ---------------------------------------------------------------------------

def _extract_probe_matrix(model_data):
    """Extract probe accuracy matrix from model data.

    Handles two formats:
      - nested list: linear_probe_accuracy[layer][position] -> np.ndarray
      - coord dict: probe_accuracies{"(l, p)": acc} -> np.ndarray
    """
    # Try nested list first (v9 format)
    lpa = model_data.get("linear_probe_accuracy")
    if lpa is not None and isinstance(lpa, list):
        return np.array(lpa, dtype=float)
    # Fall back to coord dict
    return parse_coord_dict(model_data.get("probe_accuracies", {}))


def fig1_probing_heatmap(probing_data, output_dir):
    """
    Side-by-side heatmap for all models with probing data.
    X = thought/CoT position, Y = layer, Color = probe accuracy.
    """
    panels = [m for m in THOUGHT_MODELS if m in probing_data]
    if not panels:
        print("  SKIPPED: No probe accuracy data for any model")
        return

    matrices = {}
    for model in panels:
        matrices[model] = _extract_probe_matrix(probing_data[model])

    if all(v is None for v in matrices.values()):
        print("  SKIPPED: No probe accuracy matrices found")
        return

    # Global color range from all panels
    all_vals = []
    for mat in matrices.values():
        if mat is not None:
            all_vals.extend(mat[~np.isnan(mat)].tolist())
    if not all_vals:
        print("  SKIPPED: All probe matrices empty")
        return

    vmin, vmax = min(all_vals), max(all_vals)
    n_panels = len(panels)

    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 3), sharey=True)
    if n_panels == 1:
        axes = [axes]
    im = None

    for i, model in enumerate(panels):
        ax = axes[i]
        mat = matrices[model]
        if mat is not None:
            im = ax.imshow(
                mat, aspect="auto", origin="lower",
                cmap="viridis", vmin=vmin, vmax=vmax,
                interpolation="nearest",
            )
            ax.set_xlabel("Thought Position")
            if i == 0:
                ax.set_ylabel("Layer")
        else:
            ax.text(0.5, 0.5, "No Data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="gray")
        ax.set_title(MODEL_LABELS.get(model, model), fontsize=10)

    if im is not None:
        fig.colorbar(im, ax=list(axes), shrink=0.8, label="Probe Accuracy")

    fig.suptitle(
        "Information Surface: Linear Probe Accuracy by (Layer, Position)",
        fontsize=11, y=1.02,
    )
    save_figure(fig, output_dir, "fig1_probing_heatmap")


# ---------------------------------------------------------------------------
# Figure 2: OOD Generalization Bar Chart
# ---------------------------------------------------------------------------

def _extract_ood_means_stds(ood_data, stat_data):
    """
    Extract per-model per-test-set mean and std.
    Prefers statistical_analysis.json (aggregated); falls back to raw results.
    """
    means = {m: [] for m in ALL_MODELS}
    stds = {m: [] for m in ALL_MODELS}

    for ts in OOD_TEST_SETS:
        for model in ALL_MODELS:
            mean_val = None
            std_val = 0.0

            # Try stat_data first
            if stat_data and "ood" in stat_data:
                m_data = stat_data["ood"].get(ts, {}).get(model, {})
                if isinstance(m_data, dict) and m_data.get("mean") is not None:
                    mean_val = m_data["mean"]
                    std_val = m_data.get("std") or 0.0

            # Fall back to raw ood results
            if mean_val is None and ood_data and model in ood_data:
                model_results = ood_data[model]
                if isinstance(model_results, dict) and ts in model_results:
                    val = model_results[ts]
                    if isinstance(val, (int, float)):
                        mean_val = val
                    elif isinstance(val, dict):
                        mean_val = val.get("accuracy", val.get("mean"))

            means[model].append(mean_val)
            stds[model].append(std_val)

    return means, stds


def fig2_ood_bar(ood_data, stat_data, output_dir):
    """Grouped bar chart: test-set clusters × model bars."""
    means, stds = _extract_ood_means_stds(ood_data, stat_data)

    # Filter to models that actually have data
    active_models = [m for m in ALL_MODELS
                     if any(v is not None for v in means[m])]
    if not active_models:
        print("  SKIPPED: No OOD accuracy data found")
        return

    fig, ax = plt.subplots(figsize=(7, 3.5))

    n_groups = len(OOD_TEST_SETS)
    n_bars = len(active_models)
    bar_width = min(0.15, 0.8 / n_bars)
    x = np.arange(n_groups)

    for i, model in enumerate(active_models):
        vals = [v if v is not None else 0 for v in means[model]]
        errs = stds[model]
        has_err = any(e > 0 for e in errs)
        mask = [v is not None for v in means[model]]

        offset = (i - n_bars / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, vals, bar_width,
            yerr=errs if has_err else None,
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model, "#333333"),
            edgecolor="white", linewidth=0.5,
            capsize=2, alpha=0.9,
        )
        # Dim missing bars
        for bar, present in zip(bars, mask):
            if not present:
                bar.set_alpha(0.1)

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

    save_figure(fig, output_dir, "fig2_ood_bar")


# ---------------------------------------------------------------------------
# Figure 3: Corruption Cascade Curves
# ---------------------------------------------------------------------------

def fig3_corruption_curves(corruption_data, output_dir):
    """3 subplots: forward, reverse, single-position corruption."""
    modes = [
        ("forward_corruption", "Forward Corruption", "Positions Corrupted (1→T)"),
        ("reverse_corruption", "Reverse Corruption", "Positions Corrupted (T→1)"),
        ("single_position", "Single Position", "Position Corrupted"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5), sharey=True)
    has_data = False

    for j, (key, title, xlabel) in enumerate(modes):
        ax = axes[j]
        for model in THOUGHT_MODELS:
            if model not in corruption_data:
                continue
            m_data = corruption_data[model]
            curve = m_data.get(key, [])
            if not curve:
                continue
            has_data = True

            if key == "single_position":
                x = list(range(len(curve)))
            else:
                x = list(range(1, len(curve) + 1))

            ax.plot(x, curve, "-o", markersize=3, linewidth=1.5,
                    color=MODEL_COLORS.get(model, "#333333"),
                    label=MODEL_LABELS.get(model, model))

            # Clean accuracy reference (first subplot only)
            clean = m_data.get("clean_accuracy")
            if clean is not None and j == 0:
                ax.axhline(y=clean, color=MODEL_COLORS[model],
                           linestyle=":", alpha=0.4, linewidth=0.8)

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel("Accuracy")

    if not has_data:
        plt.close(fig)
        print("  SKIPPED: No corruption curves found")
        return

    axes[-1].legend(loc="lower left", fontsize=8)
    fig.suptitle("Corruption Cascade Analysis", fontsize=11, y=1.02)
    save_figure(fig, output_dir, "fig3_corruption_curves")


# ---------------------------------------------------------------------------
# Figure 4: Causal Effect Heatmap
# ---------------------------------------------------------------------------

def _extract_ce_matrix(model_data):
    """Extract CE matrix from causal results; handles list or dict format."""
    ce = model_data.get("ce_matrix")
    if ce is None:
        return None
    if isinstance(ce, list):
        return np.array(ce, dtype=float)
    if isinstance(ce, dict):
        return parse_coord_dict(ce)
    return None


def fig4_causal_heatmap(causal_data, output_dir):
    """3-panel heatmap: M3, M4, M4b. Color = CE(l,p)."""
    matrices = {}
    for model in THOUGHT_MODELS:
        if model in causal_data:
            matrices[model] = _extract_ce_matrix(causal_data[model])
        else:
            matrices[model] = None

    if all(v is None for v in matrices.values()):
        print("  SKIPPED: No CE matrices found")
        return

    all_vals = []
    for mat in matrices.values():
        if mat is not None:
            all_vals.extend(mat[~np.isnan(mat)].tolist())
    if not all_vals:
        print("  SKIPPED: All CE matrices empty")
        return

    vmin = min(0.0, min(all_vals))
    vmax = max(all_vals)

    n_panels = len(THOUGHT_MODELS)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 2.5), sharey=True)
    if n_panels == 1:
        axes = [axes]
    im = None

    for i, model in enumerate(THOUGHT_MODELS):
        ax = axes[i]
        mat = matrices[model]
        if mat is not None:
            im = ax.imshow(
                mat, aspect="auto", origin="lower",
                cmap="RdYlBu_r", vmin=vmin, vmax=vmax,
                interpolation="nearest",
            )
            ax.set_xlabel("Position")
            if i == 0:
                ax.set_ylabel("Layer")
        else:
            ax.text(0.5, 0.5, "No Data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="gray")
        ax.set_title(MODEL_LABELS.get(model, model), fontsize=10)

    if im is not None:
        fig.colorbar(im, ax=list(axes), shrink=0.8, label="Causal Effect CE(l,p)")

    fig.suptitle(
        "Causal Tracing: Effect of Restoring Clean Activations",
        fontsize=11, y=1.02,
    )
    save_figure(fig, output_dir, "fig4_causal_heatmap")


# ---------------------------------------------------------------------------
# Figure 5: Curriculum Divergence (parsed from training logs)
# ---------------------------------------------------------------------------

def parse_training_log(log_path):
    """Extract per-epoch validation accuracy from a training log."""
    if not os.path.exists(log_path):
        return None
    accuracies = []
    with open(log_path, "r") as f:
        for line in f:
            m = re.search(
                r"Accuracy on validation set:\s*(\d+)\s*/\s*(\d+)\s*=\s*([\d.]+)",
                line,
            )
            if m:
                accuracies.append(float(m.group(3)))
    return accuracies if accuracies else None


def fig5_curriculum_divergence(logs_dir, output_dir):
    """Training curves for M3, M4, M4b with stage boundaries."""
    log_files = {
        "m3": os.path.join(logs_dir, "m3_coconut.log"),
        "m5": os.path.join(logs_dir, "m5_pause.log"),
    }

    fig, ax = plt.subplots(figsize=(7, 3.5))
    has_data = False
    max_epochs = 0

    for model, log_path in log_files.items():
        accs = parse_training_log(log_path)
        if accs is None:
            print(f"  WARNING: No log for {model} at {log_path}")
            continue
        has_data = True
        epochs = list(range(1, len(accs) + 1))
        max_epochs = max(max_epochs, len(accs))
        ax.plot(
            epochs, accs, "-", linewidth=1.5,
            color=MODEL_COLORS.get(model, "#333333"),
            label=MODEL_LABELS.get(model, model),
        )

    if not has_data:
        plt.close(fig)
        print("  SKIPPED: No training logs found")
        return

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

    save_figure(fig, output_dir, "fig5_curriculum_divergence")


# ---------------------------------------------------------------------------
# Figure 6: Token Count Ablation
# ---------------------------------------------------------------------------

def fig6_token_count(token_count_data, output_dir):
    """M3 accuracy vs number of thought tokens, one line per OOD test set."""
    # Normalize: accept {token_count: {test_set: acc}} or {m3: {tc: {ts: acc}}}
    data = token_count_data
    if "m3" in data and isinstance(data["m3"], dict):
        data = data["m3"]

    # Identify numeric token count keys
    token_counts = sorted(int(k) for k in data if str(k).isdigit())
    if not token_counts:
        # Try nested structure: {test_set: {token_count: acc}}
        print("  SKIPPED: No token count entries found")
        return

    # Identify test sets from first valid entry
    first_entry = data.get(str(token_counts[0]))
    if not isinstance(first_entry, dict):
        print("  SKIPPED: Unexpected token count data format")
        return

    test_sets = [ts for ts in OOD_TEST_SETS if ts in first_entry]
    if not test_sets:
        # Try whatever keys are present
        test_sets = sorted(first_entry.keys())

    fig, ax = plt.subplots(figsize=(3.5, 3))

    for ts in test_sets:
        xs, ys = [], []
        for tc in token_counts:
            entry = data.get(str(tc), {})
            if isinstance(entry, dict) and ts in entry:
                val = entry[ts]
                if isinstance(val, (int, float)):
                    xs.append(tc)
                    ys.append(val)
        if ys:
            ax.plot(
                xs, ys, "-o", markersize=4, linewidth=1.5,
                color=OOD_COLORS.get(ts, "#333333"),
                label=OOD_SHORT_LABELS.get(ts, ts),
            )

    ax.set_xlabel("Number of Thought Tokens")
    ax.set_ylabel("Accuracy")
    ax.set_title("M3 Accuracy vs Token Count")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    save_figure(fig, output_dir, "fig6_token_count")


# ---------------------------------------------------------------------------
# Figure 7: Cross-Problem Transplant Matrix (Appendix)
# ---------------------------------------------------------------------------

def fig7_cross_transplant(corruption_data, output_dir):
    """Heatmap: source-hop × target-hop transplant accuracy for M3."""
    m3 = corruption_data.get("m3", {})
    details = m3.get("cross_transplant_details", [])
    if not details:
        print("  SKIPPED: No cross_transplant_details in M3 corruption data")
        return

    # Aggregate by (source_hops, target_hops)
    pair_results = {}
    for d in details:
        src_h = d.get("source_hops", d.get("src_hops"))
        tgt_h = d.get("target_hops", d.get("tgt_hops"))
        if src_h is None or tgt_h is None:
            continue
        correct = d.get("correct", d.get("accuracy", 0))
        # Normalise to float
        if isinstance(correct, bool):
            correct = 1.0 if correct else 0.0
        key = (src_h, tgt_h)
        pair_results.setdefault(key, []).append(float(correct))

    if not pair_results:
        print("  SKIPPED: Could not parse transplant pair details")
        return

    hop_values = sorted(set(h for pair in pair_results for h in pair))
    n = len(hop_values)
    hop_idx = {h: i for i, h in enumerate(hop_values)}

    matrix = np.full((n, n), np.nan)
    for (src, tgt), vals in pair_results.items():
        matrix[hop_idx[src], hop_idx[tgt]] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(3.5, 3))

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1,
                   interpolation="nearest", origin="lower")

    tick_labels = [str(h) for h in hop_values]
    ax.set_xticks(range(n))
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(range(n))
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Target Hop Count")
    ax.set_ylabel("Source Hop Count")
    ax.set_title("Cross-Problem Thought Transplant (M3)")

    # Cell annotations
    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                color = "white" if matrix[i, j] < 0.5 else "black"
                ax.text(j, i, f"{matrix[i, j]:.2f}",
                        ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Accuracy")

    save_figure(fig, output_dir, "fig7_cross_transplant")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate figures for COCONUT reasoning study",
    )
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Root results directory (contains experiments/ and logs/)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for figures",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Paths
    exp_dir = os.path.join(args.results_dir, "experiments")
    logs_dir = os.path.join(args.results_dir, "logs")

    # Load all available data
    probing_data = load_json(os.path.join(exp_dir, "probing", "results.json"))
    ood_data = load_json(os.path.join(exp_dir, "ood", "results.json"))
    corruption_data = load_json(os.path.join(exp_dir, "corruption", "results.json"))
    causal_data = load_json(os.path.join(exp_dir, "causal", "results.json"))
    token_count_data = load_json(os.path.join(exp_dir, "token_count", "results.json"))
    stat_data = load_json(os.path.join(args.results_dir, "statistical_analysis.json"))

    available = {
        "probing": probing_data is not None,
        "ood": ood_data is not None,
        "corruption": corruption_data is not None,
        "causal": causal_data is not None,
        "token_count": token_count_data is not None,
        "stat_analysis": stat_data is not None,
    }

    print(f"Results dir: {args.results_dir}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Data loaded: {', '.join(k + '=yes' if v else k + '=no' for k, v in available.items())}")
    print()

    generated = 0

    # --- Fig 1 ---
    print("Fig 1: Probing Heatmap")
    if probing_data:
        fig1_probing_heatmap(probing_data, args.output_dir)
        generated += 1
    else:
        print("  SKIPPED: No probing results")

    # --- Fig 2 ---
    print("\nFig 2: OOD Bar Chart")
    if ood_data or (stat_data and "ood" in stat_data):
        fig2_ood_bar(ood_data, stat_data, args.output_dir)
        generated += 1
    else:
        print("  SKIPPED: No OOD results")

    # --- Fig 3 ---
    print("\nFig 3: Corruption Curves")
    if corruption_data:
        fig3_corruption_curves(corruption_data, args.output_dir)
        generated += 1
    else:
        print("  SKIPPED: No corruption results")

    # --- Fig 4 ---
    print("\nFig 4: Causal Heatmap")
    if causal_data:
        fig4_causal_heatmap(causal_data, args.output_dir)
        generated += 1
    else:
        print("  SKIPPED: No causal results")

    # --- Fig 5 ---
    print("\nFig 5: Curriculum Divergence")
    if os.path.isdir(logs_dir):
        fig5_curriculum_divergence(logs_dir, args.output_dir)
        generated += 1
    else:
        print(f"  SKIPPED: No logs directory at {logs_dir}")

    # --- Fig 6 ---
    print("\nFig 6: Token Count Ablation")
    if token_count_data:
        fig6_token_count(token_count_data, args.output_dir)
        generated += 1
    else:
        print("  SKIPPED: No token count results")

    # --- Fig 7 ---
    print("\nFig 7: Cross-Problem Transplant")
    if corruption_data:
        fig7_cross_transplant(corruption_data, args.output_dir)
        generated += 1
    else:
        print("  SKIPPED: No corruption results")

    print(f"\nDone. Generated {generated}/7 figures.")


if __name__ == "__main__":
    main()
