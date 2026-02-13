"""
Statistical Analysis for COCONUT Reasoning Study (Phase 5)

Aggregates results across seeds and computes:
  - Per-metric mean ± std across seeds
  - McNemar's test for M3 vs M4b (primary comparison)
  - Cohen's d effect sizes for pairwise comparisons
  - Bonferroni-corrected p-values
  - 95% confidence intervals (bootstrap or t-distribution)
  - Decision matrix row assignment

Usage:
    # Single seed (Phase 4):
    python statistical_analysis.py \
        --results_dirs /path/to/experiments/ \
        --output /path/to/statistical_analysis.json

    # Multi-seed (Phase 5):
    python statistical_analysis.py \
        --results_dirs /path/to/seed0/ /path/to/seed1/ /path/to/seed2/ \
        --output /path/to/statistical_analysis.json

Dependencies: scipy, numpy, json, argparse (no GPU, no torch)
"""

import argparse
import json
import math
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAMES = ["m1", "m3", "m5"]
OOD_TEST_SETS = ["prosqa_test", "ood_7hop", "ood_8hop", "ood_dag", "ood_dense"]
PRIMARY_COMPARISON = ("m3", "m5")
SECONDARY_COMPARISONS = [("m3", "m1"), ("m5", "m1")]
ALPHA = 0.05
BOOTSTRAP_N = 10000


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_experiment_results(results_dir):
    """
    Load all experiment results from a single seed's results directory.

    Expected structure:
        results_dir/
            ood/results.json
            corruption/results.json
            probing/results.json
            causal/results.json

    Returns dict with keys: ood, corruption, probing, causal (each may be None).
    """
    experiments = {}
    for exp_name in ["ood", "corruption", "probing", "causal"]:
        path = os.path.join(results_dir, exp_name, "results.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                experiments[exp_name] = json.load(f)
        else:
            experiments[exp_name] = None
    return experiments


def load_all_seeds(results_dirs):
    """Load results from all seed directories."""
    all_seeds = []
    for d in results_dirs:
        data = load_experiment_results(d)
        all_seeds.append(data)
    return all_seeds


# ---------------------------------------------------------------------------
# OOD accuracy extraction
# ---------------------------------------------------------------------------

def extract_ood_accuracies(all_seeds):
    """
    Extract per-model per-test-set accuracies across seeds.

    Returns:
        dict[test_set][model] -> list of floats (one per seed)
    """
    result = defaultdict(lambda: defaultdict(list))

    for seed_data in all_seeds:
        ood = seed_data.get("ood")
        if ood is None:
            continue
        for model in MODEL_NAMES:
            if model not in ood:
                continue
            model_results = ood[model]
            if isinstance(model_results, dict):
                for test_set in OOD_TEST_SETS:
                    if test_set in model_results:
                        result[test_set][model].append(model_results[test_set])
            elif isinstance(model_results, (int, float)):
                # Single test set stored directly
                result["prosqa_test"][model].append(model_results)

    return result


def extract_ood_per_sample(all_seeds):
    """
    Try to extract per-sample correct/incorrect vectors from detailed outputs.

    Returns:
        dict[test_set][model] -> list of lists (outer: seeds, inner: per-sample bool)
        or None if per-sample data unavailable.
    """
    result = defaultdict(lambda: defaultdict(list))
    found_any = False

    for seed_idx, seed_data in enumerate(all_seeds):
        ood = seed_data.get("ood")
        if ood is None:
            continue

        # Check for detailed_outputs key or per-sample data
        # The exp_ood.py saves detailed_outputs.json separately, not in results.json
        # So per-sample data likely isn't in the results.json
        for model in MODEL_NAMES:
            if model not in ood:
                continue
            model_results = ood[model]
            if isinstance(model_results, dict):
                for test_set in OOD_TEST_SETS:
                    if test_set in model_results:
                        val = model_results[test_set]
                        if isinstance(val, dict) and "per_sample" in val:
                            result[test_set][model].append(val["per_sample"])
                            found_any = True

    return result if found_any else None


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def compute_descriptive(values):
    """
    Compute mean, std, and 95% CI for a list of values.

    With n=1: returns point estimate, std=None, ci=None.
    With n>=2: uses t-distribution for CI.
    """
    n = len(values)
    if n == 0:
        return {"mean": None, "std": None, "ci_95": None, "n": 0}

    mean = float(np.mean(values))

    if n == 1:
        return {"mean": round(mean, 6), "std": None, "ci_95": None, "n": 1}

    std = float(np.std(values, ddof=1))
    se = std / math.sqrt(n)
    t_crit = stats.t.ppf(1 - ALPHA / 2, df=n - 1)
    ci_lo = mean - t_crit * se
    ci_hi = mean + t_crit * se

    return {
        "mean": round(mean, 6),
        "std": round(std, 6),
        "ci_95": [round(ci_lo, 6), round(ci_hi, 6)],
        "n": n,
    }


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------

def mcnemar_test(acc_a, n_a, acc_b, n_b, per_sample_a=None, per_sample_b=None):
    """
    McNemar's test for paired nominal data.

    If per-sample data is available, compute exact discordant counts.
    Otherwise, use normal approximation from aggregate accuracies.

    Returns dict with chi2, p_value, approximate (bool).
    """
    if per_sample_a is not None and per_sample_b is not None:
        a = np.array(per_sample_a, dtype=bool)
        b = np.array(per_sample_b, dtype=bool)
        assert len(a) == len(b), f"Sample size mismatch: {len(a)} vs {len(b)}"

        # Discordant pairs
        b_correct_a_wrong = np.sum(~a & b)  # b right, a wrong
        a_correct_b_wrong = np.sum(a & ~b)  # a right, b wrong

        n_discordant = b_correct_a_wrong + a_correct_b_wrong
        if n_discordant == 0:
            return {"chi2": 0.0, "p_value": 1.0, "approximate": False,
                    "b01": int(b_correct_a_wrong), "b10": int(a_correct_b_wrong)}

        chi2 = (abs(b_correct_a_wrong - a_correct_b_wrong) - 1) ** 2 / n_discordant
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

        return {
            "chi2": round(float(chi2), 6),
            "p_value": round(float(p_value), 8),
            "approximate": False,
            "b01": int(b_correct_a_wrong),
            "b10": int(a_correct_b_wrong),
        }
    else:
        # Normal approximation from aggregate stats
        # Estimate discordant cells assuming independence (conservative)
        n = max(n_a, n_b)
        p_a = acc_a
        p_b = acc_b

        # Under independence assumption:
        # b01 ≈ n * p_b * (1 - p_a)
        # b10 ≈ n * p_a * (1 - p_b)
        b01 = n * p_b * (1 - p_a)
        b10 = n * p_a * (1 - p_b)

        n_discordant = b01 + b10
        if n_discordant < 1:
            return {"chi2": 0.0, "p_value": 1.0, "approximate": True,
                    "b01": round(b01, 2), "b10": round(b10, 2),
                    "warning": "Very few estimated discordant pairs"}

        chi2 = (abs(b01 - b10) - 1) ** 2 / n_discordant
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

        return {
            "chi2": round(float(chi2), 6),
            "p_value": round(float(p_value), 8),
            "approximate": True,
            "b01": round(float(b01), 2),
            "b10": round(float(b10), 2),
            "warning": "Approximate: computed from aggregate accuracy, not per-sample data",
        }


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------

def cohens_d(values_a, values_b):
    """
    Compute Cohen's d between two groups.

    With n=1 per group (single seed), returns the raw difference
    divided by a pooled estimate using the difference itself.
    """
    a = np.array(values_a, dtype=float)
    b = np.array(values_b, dtype=float)

    n_a, n_b = len(a), len(b)

    if n_a == 0 or n_b == 0:
        return None

    mean_a = np.mean(a)
    mean_b = np.mean(b)
    diff = mean_a - mean_b

    if n_a == 1 and n_b == 1:
        # Single seed: report raw difference, no pooled SD
        return {
            "d": round(float(diff), 6),
            "interpretation": "single_seed_difference",
            "mean_a": round(float(mean_a), 6),
            "mean_b": round(float(mean_b), 6),
        }

    # Pooled standard deviation
    var_a = np.var(a, ddof=1) if n_a > 1 else 0
    var_b = np.var(b, ddof=1) if n_b > 1 else 0
    pooled_sd = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1))

    if pooled_sd < 1e-10:
        d = 0.0 if abs(diff) < 1e-10 else float('inf') * np.sign(diff)
    else:
        d = diff / pooled_sd

    # Interpretation
    abs_d = abs(d)
    if abs_d < 0.2:
        interp = "negligible"
    elif abs_d < 0.5:
        interp = "small"
    elif abs_d < 0.8:
        interp = "medium"
    else:
        interp = "large"

    return {
        "d": round(float(d), 6),
        "pooled_sd": round(float(pooled_sd), 6),
        "interpretation": interp,
        "mean_a": round(float(mean_a), 6),
        "mean_b": round(float(mean_b), 6),
    }


# ---------------------------------------------------------------------------
# Paired tests (across matched seeds)
# ---------------------------------------------------------------------------

def paired_t_test(values_a, values_b):
    """
    Paired t-test for matched seeds. Requires len(a) == len(b) >= 2.
    Returns None if insufficient data.
    """
    a = np.array(values_a, dtype=float)
    b = np.array(values_b, dtype=float)

    if len(a) != len(b) or len(a) < 2:
        return None

    diffs = a - b
    mean_diff = np.mean(diffs)
    se_diff = np.std(diffs, ddof=1) / math.sqrt(len(diffs))

    if se_diff < 1e-10:
        return {
            "t_statistic": float('inf') if abs(mean_diff) > 1e-10 else 0.0,
            "p_value": 0.0 if abs(mean_diff) > 1e-10 else 1.0,
            "mean_diff": round(float(mean_diff), 6),
            "se_diff": 0.0,
            "df": len(diffs) - 1,
        }

    t_stat, p_value = stats.ttest_rel(a, b)

    return {
        "t_statistic": round(float(t_stat), 6),
        "p_value": round(float(p_value), 8),
        "mean_diff": round(float(mean_diff), 6),
        "se_diff": round(float(se_diff), 6),
        "df": len(diffs) - 1,
    }


# ---------------------------------------------------------------------------
# Bonferroni correction
# ---------------------------------------------------------------------------

def bonferroni_correct(p_values_dict, alpha=ALPHA):
    """
    Apply Bonferroni correction to a flat dict of {label: p_value}.
    Returns dict of {label: {"p_raw": ..., "p_corrected": ..., "significant": ...}}.
    """
    n_tests = len(p_values_dict)
    if n_tests == 0:
        return {}

    corrected = {}
    for label, p_raw in p_values_dict.items():
        if p_raw is None:
            corrected[label] = {"p_raw": None, "p_corrected": None, "significant": None}
            continue
        p_corr = min(p_raw * n_tests, 1.0)
        corrected[label] = {
            "p_raw": round(p_raw, 8),
            "p_corrected": round(p_corr, 8),
            "significant": p_corr < alpha,
            "n_tests": n_tests,
        }

    return corrected


# ---------------------------------------------------------------------------
# Bootstrap CI (for per-sample data)
# ---------------------------------------------------------------------------

def bootstrap_ci(values, n_boot=BOOTSTRAP_N, alpha=ALPHA):
    """
    Bootstrap 95% CI for the mean of values.
    """
    values = np.array(values, dtype=float)
    n = len(values)
    if n < 2:
        return None

    rng = np.random.RandomState(42)
    boot_means = np.array([
        np.mean(rng.choice(values, size=n, replace=True))
        for _ in range(n_boot)
    ])

    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return [round(float(lo), 6), round(float(hi), 6)]


# ---------------------------------------------------------------------------
# Probing analysis
# ---------------------------------------------------------------------------

def analyze_probing(all_seeds):
    """
    Analyze probing results across seeds.

    Key metrics:
      - diagonal_strength: correlation between best-layer index and position index
      - nonlinear_advantage: fraction of cells where MLP > linear by >10pp
    """
    results = {}

    for model in ["m3", "m5"]:
        diag_strengths = []
        nl_advantages = []
        peak_accs = []
        nl_peak_accs = []
        n_sig_cells = []

        for seed_data in all_seeds:
            probing = seed_data.get("probing")
            if probing is None or model not in probing:
                continue

            m = probing[model]
            peak_accs.append(m.get("peak_accuracy", 0))

            # Diagonal strength from diagonal_peak_layers
            diag_peaks = m.get("diagonal_peak_layers", [])
            if len(diag_peaks) >= 3:
                # Spearman correlation between position index and peak layer
                positions = list(range(len(diag_peaks)))
                corr, _ = stats.spearmanr(positions, diag_peaks)
                diag_strengths.append(float(corr) if not np.isnan(corr) else 0.0)
            else:
                diag_strengths.append(0.0)

            # Nonlinear advantage
            nl_adv = m.get("nonlinear_advantage_cells", 0)
            nl_advantages.append(nl_adv)

            nl_peak = m.get("nonlinear_peak_accuracy", 0)
            nl_peak_accs.append(nl_peak)

            n_sig = m.get("n_significant_cells", 0)
            n_sig_cells.append(n_sig)

        results[model] = {
            "diagonal_strength": compute_descriptive(diag_strengths),
            "peak_accuracy": compute_descriptive(peak_accs),
            "nonlinear_peak_accuracy": compute_descriptive(nl_peak_accs),
            "nonlinear_advantage_cells": compute_descriptive(nl_advantages),
            "n_significant_cells": compute_descriptive(n_sig_cells),
            "diagonal_pattern_present": any(
                seed_data.get("probing", {}).get(model, {}).get("diagonal_pattern", False)
                for seed_data in all_seeds
                if seed_data.get("probing") and model in seed_data.get("probing", {})
            ),
        }

    # Compute difference: M3 diagonal_strength vs M5
    m3_diags = []
    m5_diags = []
    for seed_data in all_seeds:
        probing = seed_data.get("probing")
        if probing is None:
            continue
        for model, dest in [("m3", m3_diags), ("m5", m5_diags)]:
            if model in probing:
                dp = probing[model].get("diagonal_peak_layers", [])
                if len(dp) >= 3:
                    corr, _ = stats.spearmanr(range(len(dp)), dp)
                    dest.append(float(corr) if not np.isnan(corr) else 0.0)
                else:
                    dest.append(0.0)

    results["m3_vs_m5_diagonal"] = {
        "effect_size": cohens_d(m3_diags, m5_diags),
        "paired_test": paired_t_test(m3_diags, m5_diags),
    }

    return results


# ---------------------------------------------------------------------------
# Corruption analysis
# ---------------------------------------------------------------------------

def analyze_corruption(all_seeds):
    """
    Analyze corruption results across seeds.

    Key metrics:
      - cascade_rate: how quickly accuracy degrades with forward corruption
        (slope of accuracy vs number of corrupted positions)
      - cross_transplant accuracy delta between M3 and M4b
    """
    results = {}

    for model in ["m3", "m5"]:
        cascade_rates = []
        clean_accs = []
        transplant_accs = []
        sensitivities = []

        for seed_data in all_seeds:
            corruption = seed_data.get("corruption")
            if corruption is None or model not in corruption:
                continue

            m = corruption[model]
            clean = m.get("clean_accuracy", 0)
            clean_accs.append(clean)

            # Cascade rate: linear regression slope of forward corruption accuracy
            fwd = m.get("forward_corruption", [])
            if len(fwd) >= 2:
                x = np.arange(1, len(fwd) + 1)
                slope, _, _, _, _ = stats.linregress(x, fwd)
                cascade_rates.append(float(slope))
            else:
                cascade_rates.append(0.0)

            transplant = m.get("cross_transplant_accuracy", 0)
            transplant_accs.append(transplant)

            sens = m.get("sensitivity", "unknown")
            sensitivities.append(sens)

        results[model] = {
            "clean_accuracy": compute_descriptive(clean_accs),
            "cascade_rate": compute_descriptive(cascade_rates),
            "cross_transplant_accuracy": compute_descriptive(transplant_accs),
            "sensitivity_pattern": sensitivities,
        }

    # M3 vs M5 cascade rate comparison
    m3_rates = []
    m5_rates = []
    m3_transplants = []
    m5_transplants = []

    for seed_data in all_seeds:
        corruption = seed_data.get("corruption")
        if corruption is None:
            continue
        for model, rates, trans in [
            ("m3", m3_rates, m3_transplants),
            ("m5", m5_rates, m5_transplants),
        ]:
            if model in corruption:
                fwd = corruption[model].get("forward_corruption", [])
                if len(fwd) >= 2:
                    slope, _, _, _, _ = stats.linregress(np.arange(1, len(fwd) + 1), fwd)
                    rates.append(float(slope))
                trans.append(corruption[model].get("cross_transplant_accuracy", 0))

    results["m3_vs_m5_cascade"] = {
        "effect_size": cohens_d(m3_rates, m5_rates),
        "paired_test": paired_t_test(m3_rates, m5_rates),
    }
    results["m3_vs_m5_transplant"] = {
        "effect_size": cohens_d(m3_transplants, m5_transplants),
        "paired_test": paired_t_test(m3_transplants, m5_transplants),
    }

    return results


# ---------------------------------------------------------------------------
# Causal tracing analysis
# ---------------------------------------------------------------------------

def analyze_causal(all_seeds):
    """
    Analyze causal tracing results across seeds.
    """
    results = {}

    for model in ["m1", "m3", "m5"]:
        peak_ces = []
        ce_fracs = []
        peak_layers = []

        for seed_data in all_seeds:
            causal = seed_data.get("causal")
            if causal is None or model not in causal:
                continue

            m = causal[model]
            peak_ces.append(m.get("peak_ce", 0))
            ce_fracs.append(m.get("ce_above_03_fraction", 0))
            peak_layers.append(m.get("peak_layer", 0))

        results[model] = {
            "peak_ce": compute_descriptive(peak_ces),
            "ce_above_03_fraction": compute_descriptive(ce_fracs),
            "peak_layer": compute_descriptive(peak_layers),
        }

    return results


# ---------------------------------------------------------------------------
# Decision matrix
# ---------------------------------------------------------------------------

def assign_decision_matrix_row(ood_stats, probing_stats, corruption_stats):
    """
    Assign a decision matrix row based on the pattern of results.

    Row 1 ("reasoning"): M3 > M5 on OOD + probing diagonal + corruption cascade diff
    Row 2 ("buffering"): M3 ≈ M5 on everything (curriculum + compute sufficient)
    Row 3 ("inconclusive"): Mixed signals

    Returns (row_number, justification_string).
    """
    signals = []

    # Signal 1: M3 outperforms M5 on OOD (especially 7-8 hop)
    m3_ood_advantage = False
    m5_ood_advantage = False
    for test_set in ["ood_7hop", "ood_8hop", "ood_dag", "ood_dense"]:
        m3_vals = ood_stats.get(test_set, {}).get("m3", [])
        m5_vals = ood_stats.get(test_set, {}).get("m5", [])
        if m3_vals and m5_vals:
            m3_mean = np.mean(m3_vals)
            m5_mean = np.mean(m5_vals)
            if m3_mean > m5_mean + 0.05:  # >5pp advantage
                m3_ood_advantage = True
            if m5_mean > m3_mean + 0.05:
                m5_ood_advantage = True

    if m3_ood_advantage:
        signals.append("ood_m3_advantage")
    if m5_ood_advantage:
        signals.append("ood_m5_advantage")

    # Check if M3 ≈ M5 on OOD (within 3pp on all sets)
    ood_similar = True
    for test_set in OOD_TEST_SETS:
        m3_vals = ood_stats.get(test_set, {}).get("m3", [])
        m5_vals = ood_stats.get(test_set, {}).get("m5", [])
        if m3_vals and m5_vals:
            if abs(np.mean(m3_vals) - np.mean(m5_vals)) > 0.03:
                ood_similar = False
                break

    # Signal 2: M3 has diagonal probing pattern, M5 doesn't (or vice versa)
    probing_diagonal = False
    if "m3" in probing_stats and "m5" in probing_stats:
        m3_diag = probing_stats["m3"].get("diagonal_pattern_present", False)
        m5_diag = probing_stats["m5"].get("diagonal_pattern_present", False)
        if m3_diag and not m5_diag:
            probing_diagonal = True
            signals.append("probing_m3_diagonal_only")
        elif m3_diag and m5_diag:
            signals.append("probing_both_diagonal")

    # Signal 3: M3 cascade rate differs from M5
    corruption_diff = False
    cascade_data = corruption_stats.get("m3_vs_m5_cascade", {})
    es = cascade_data.get("effect_size")
    if es and isinstance(es, dict):
        d = es.get("d", 0)
        if d is not None and abs(d) > 0.5:
            corruption_diff = True
            signals.append("corruption_cascade_difference")

    # Decision logic
    reasoning_signals = sum([
        m3_ood_advantage,
        probing_diagonal,
        corruption_diff,
    ])

    if reasoning_signals >= 2:
        row = 1
        justification = (
            f"Row 1 (reasoning): {reasoning_signals}/3 signals support M3 performing "
            f"genuine sequential reasoning vs M5. Signals: {signals}"
        )
    elif ood_similar and not probing_diagonal and not corruption_diff:
        row = 2
        justification = (
            f"Row 2 (buffering): M3 ≈ M5 across corruption/probing metrics. "
            f"Curriculum + extra compute is sufficient; hidden-state recycling "
            f"mechanism is unnecessary. Signals: {signals}"
        )
    else:
        row = 3
        justification = (
            f"Row 3 (inconclusive): Mixed signals ({reasoning_signals}/3 reasoning "
            f"indicators). Signals present: {signals}"
        )

    return row, justification


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def run_analysis(results_dirs):
    """
    Run the full statistical analysis pipeline.

    Returns the complete results dict ready for JSON serialization.
    """
    n_seeds = len(results_dirs)
    print(f"Loading results from {n_seeds} seed(s)...")
    all_seeds = load_all_seeds(results_dirs)

    output = {
        "metadata": {
            "n_seeds": n_seeds,
            "results_dirs": results_dirs,
            "alpha": ALPHA,
            "bootstrap_n": BOOTSTRAP_N,
        },
    }

    # -----------------------------------------------------------------------
    # 1. OOD analysis
    # -----------------------------------------------------------------------
    print("Analyzing OOD results...")
    ood_accs = extract_ood_accuracies(all_seeds)
    per_sample = extract_ood_per_sample(all_seeds)

    ood_results = {}
    for test_set in OOD_TEST_SETS:
        ood_results[test_set] = {}
        for model in MODEL_NAMES:
            vals = ood_accs.get(test_set, {}).get(model, [])
            ood_results[test_set][model] = compute_descriptive(vals)

    output["ood"] = ood_results

    # -----------------------------------------------------------------------
    # 2. McNemar's test (M3 vs M4b on each OOD test set)
    # -----------------------------------------------------------------------
    print("Running McNemar's tests...")
    mcnemar_results = {}
    all_p_values = {}

    for test_set in OOD_TEST_SETS:
        m3_vals = ood_accs.get(test_set, {}).get("m3", [])
        m5_vals = ood_accs.get(test_set, {}).get("m5", [])

        if not m3_vals or not m5_vals:
            mcnemar_results[test_set] = {"m3_vs_m5": None}
            continue

        # Average across seeds for aggregate McNemar
        m3_acc = float(np.mean(m3_vals))
        m5_acc = float(np.mean(m5_vals))

        # Try per-sample data
        ps_m3 = None
        ps_m5 = None
        if per_sample is not None:
            ps_m3_list = per_sample.get(test_set, {}).get("m3", [])
            ps_m5_list = per_sample.get(test_set, {}).get("m5", [])
            if ps_m3_list and ps_m5_list:
                # Use first seed's per-sample data
                ps_m3 = ps_m3_list[0]
                ps_m5 = ps_m5_list[0]

        # Estimate n from the test set
        # Standard ProsQA test: 500, OOD sets: varies
        n_samples = 500  # default estimate
        for seed_data in all_seeds:
            ood = seed_data.get("ood")
            if ood and "m3" in ood:
                m3_data = ood["m3"]
                if isinstance(m3_data, dict) and test_set in m3_data:
                    # Check if there's sample count info
                    break

        result = mcnemar_test(m3_acc, n_samples, m5_acc, n_samples,
                              ps_m3, ps_m5)
        mcnemar_results[test_set] = {"m3_vs_m5": result}
        all_p_values[f"mcnemar_{test_set}"] = result["p_value"]

    output["mcnemar"] = mcnemar_results

    # -----------------------------------------------------------------------
    # 3. Effect sizes (Cohen's d)
    # -----------------------------------------------------------------------
    print("Computing effect sizes...")
    effect_sizes = {}

    comparisons = [PRIMARY_COMPARISON] + SECONDARY_COMPARISONS
    for model_a, model_b in comparisons:
        key = f"{model_a}_vs_{model_b}"
        effect_sizes[key] = {}

        for test_set in OOD_TEST_SETS:
            vals_a = ood_accs.get(test_set, {}).get(model_a, [])
            vals_b = ood_accs.get(test_set, {}).get(model_b, [])

            if vals_a and vals_b:
                effect_sizes[key][test_set] = cohens_d(vals_a, vals_b)
            else:
                effect_sizes[key][test_set] = None

    output["effect_sizes"] = effect_sizes

    # -----------------------------------------------------------------------
    # 4. Paired tests (across matched seeds)
    # -----------------------------------------------------------------------
    print("Running paired tests...")
    paired_results = {}

    if n_seeds >= 2:
        for model_a, model_b in comparisons:
            key = f"{model_a}_vs_{model_b}"
            paired_results[key] = {}

            for test_set in OOD_TEST_SETS:
                vals_a = ood_accs.get(test_set, {}).get(model_a, [])
                vals_b = ood_accs.get(test_set, {}).get(model_b, [])

                result = paired_t_test(vals_a, vals_b)
                paired_results[key][test_set] = result
                if result is not None:
                    all_p_values[f"paired_{key}_{test_set}"] = result["p_value"]
    else:
        paired_results["note"] = "Paired tests require >= 2 seeds. Only 1 seed available."

    output["paired_tests"] = paired_results

    # -----------------------------------------------------------------------
    # 5. Bonferroni correction
    # -----------------------------------------------------------------------
    print("Applying Bonferroni correction...")
    output["bonferroni"] = bonferroni_correct(all_p_values)

    # -----------------------------------------------------------------------
    # 6. Probing analysis
    # -----------------------------------------------------------------------
    print("Analyzing probing results...")
    probing_results = analyze_probing(all_seeds)
    output["probing"] = probing_results

    # -----------------------------------------------------------------------
    # 7. Corruption analysis
    # -----------------------------------------------------------------------
    print("Analyzing corruption results...")
    corruption_results = analyze_corruption(all_seeds)
    output["corruption"] = corruption_results

    # -----------------------------------------------------------------------
    # 8. Causal tracing analysis
    # -----------------------------------------------------------------------
    print("Analyzing causal tracing results...")
    causal_results = analyze_causal(all_seeds)
    output["causal"] = causal_results

    # -----------------------------------------------------------------------
    # 9. Decision matrix
    # -----------------------------------------------------------------------
    print("Assigning decision matrix row...")
    row, justification = assign_decision_matrix_row(
        ood_accs, probing_results, corruption_results
    )
    output["decision_matrix_row"] = row
    output["decision_matrix_justification"] = justification

    # -----------------------------------------------------------------------
    # 10. Summary
    # -----------------------------------------------------------------------
    summary = {
        "n_seeds": n_seeds,
        "single_seed_mode": n_seeds == 1,
    }

    # Summarize OOD for primary comparison
    for test_set in OOD_TEST_SETS:
        m3 = ood_results.get(test_set, {}).get("m3", {})
        m5 = ood_results.get(test_set, {}).get("m5", {})
        if m3.get("mean") is not None and m5.get("mean") is not None:
            summary[f"ood_{test_set}_m3_minus_m5"] = round(
                m3["mean"] - m5["mean"], 6
            )

    output["summary"] = summary

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Statistical analysis for COCONUT reasoning study"
    )
    parser.add_argument(
        "--results_dirs", nargs="+", required=True,
        help="One or more experiment results directories (one per seed)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output JSON path",
    )
    args = parser.parse_args()

    # Validate directories
    for d in args.results_dirs:
        if not os.path.isdir(d):
            print(f"WARNING: {d} does not exist or is not a directory")

    output = run_analysis(args.results_dirs)

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(f"\nDecision matrix: Row {output['decision_matrix_row']}")
    print(f"Justification: {output['decision_matrix_justification']}")

    # Print key results
    print("\n" + "=" * 60)
    print("KEY RESULTS")
    print("=" * 60)

    if output.get("ood"):
        print("\nOOD Accuracy (mean across seeds):")
        header = f"  {'Test Set':<16}" + "".join(f"{m:<10}" for m in MODEL_NAMES)
        print(header)
        for ts in OOD_TEST_SETS:
            if ts in output["ood"]:
                row = f"  {ts:<16}"
                for m in MODEL_NAMES:
                    val = output["ood"][ts].get(m, {}).get("mean")
                    row += f"{val:<10.4f}" if val is not None else f"{'N/A':<10}"
                print(row)

    if output.get("bonferroni"):
        print("\nBonferroni-corrected p-values:")
        for label, info in output["bonferroni"].items():
            if info.get("p_corrected") is not None:
                sig = " ***" if info["significant"] else ""
                print(f"  {label}: p_raw={info['p_raw']:.6f}, "
                      f"p_corr={info['p_corrected']:.6f}{sig}")

    print("\nDone.")


if __name__ == "__main__":
    main()
