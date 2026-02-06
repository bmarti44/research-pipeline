#!/usr/bin/env python3
"""
Statistical Analysis Script for A+B+C Architecture Study

Automates:
- Two-sample t-tests (Welch's for unequal variances)
- Effect size calculation (Cohen's d with pooled SD)
- Confidence interval calculation
- Power analysis for minimum detectable effect
- Formatted output for publication

Usage:
    python analyze_results.py --baseline ../results/v3.3_scale/baseline \
                              --treatment ../results/v3.3_scale/memory_only \
                              --metric final_val_ppl
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any


def load_results(results_dir: str, metric: str = "final_val_ppl") -> List[float]:
    """Load metric values from all seed directories."""
    results_path = Path(results_dir)
    values = []

    for seed_dir in sorted(results_path.glob("seed_*")):
        # Find the results JSON file
        json_files = list(seed_dir.glob("*_results.json"))
        if not json_files:
            continue

        with open(json_files[0], 'r') as f:
            data = json.load(f)
            if metric in data:
                values.append(data[metric])

    return values


def mean(values: List[float]) -> float:
    """Calculate mean."""
    return sum(values) / len(values)


def std(values: List[float], ddof: int = 1) -> float:
    """Calculate standard deviation with degrees of freedom adjustment."""
    n = len(values)
    if n <= ddof:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (n - ddof))


def welch_t_test(group1: List[float], group2: List[float]) -> Tuple[float, float, float]:
    """
    Perform Welch's t-test for unequal variances.

    Returns: (t_statistic, p_value, degrees_of_freedom)
    """
    n1, n2 = len(group1), len(group2)
    m1, m2 = mean(group1), mean(group2)
    s1, s2 = std(group1), std(group2)

    # Welch's t-statistic
    se = math.sqrt(s1**2 / n1 + s2**2 / n2)
    if se == 0:
        return 0.0, 1.0, n1 + n2 - 2

    t_stat = (m1 - m2) / se

    # Welch-Satterthwaite degrees of freedom
    v1, v2 = s1**2 / n1, s2**2 / n2
    if v1 + v2 == 0:
        df = n1 + n2 - 2
    else:
        df = (v1 + v2)**2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))

    # Two-tailed p-value (using scipy if available)
    try:
        from scipy import stats
        p_value = 2 * stats.t.sf(abs(t_stat), df)
    except ImportError:
        # Approximate using normal distribution for large df
        p_value = 2 * (1 - _normal_cdf(abs(t_stat)))

    return t_stat, p_value, df


def _normal_cdf(x: float) -> float:
    """Approximate normal CDF using error function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def cohens_d(group1: List[float], group2: List[float]) -> Tuple[float, str]:
    """
    Calculate Cohen's d effect size using pooled standard deviation.

    Formula: d = (M1 - M2) / SD_pooled
    where SD_pooled = sqrt(((n1-1)*s1² + (n2-1)*s2²) / (n1+n2-2))

    Returns: (d_value, interpretation)
    """
    n1, n2 = len(group1), len(group2)
    m1, m2 = mean(group1), mean(group2)
    s1, s2 = std(group1), std(group2)

    # Pooled standard deviation
    pooled_var = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
    pooled_sd = math.sqrt(pooled_var)

    if pooled_sd == 0:
        return 0.0, "undefined"

    d = (m1 - m2) / pooled_sd

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

    return d, interp


def confidence_interval(group1: List[float], group2: List[float],
                       confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for difference in means.

    Returns: (lower_bound, upper_bound) as percentage difference
    """
    n1, n2 = len(group1), len(group2)
    m1, m2 = mean(group1), mean(group2)
    s1, s2 = std(group1), std(group2)

    # Standard error of difference (Welch's)
    se = math.sqrt(s1**2 / n1 + s2**2 / n2)

    # Welch-Satterthwaite df
    v1, v2 = s1**2 / n1, s2**2 / n2
    if v1 + v2 == 0:
        df = n1 + n2 - 2
    else:
        df = (v1 + v2)**2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))

    # t critical value
    try:
        from scipy import stats
        t_crit = stats.t.ppf((1 + confidence) / 2, df)
    except ImportError:
        # Approximate for 95% CI
        t_crit = 2.0 if df > 30 else 2.5

    diff = m1 - m2
    margin = t_crit * se

    # Convert to percentage relative to baseline (group2)
    baseline = m2
    lower_pct = ((diff - margin) / baseline) * 100
    upper_pct = ((diff + margin) / baseline) * 100

    return lower_pct, upper_pct


def minimum_detectable_effect(group1: List[float], group2: List[float],
                              power: float = 0.80, alpha: float = 0.05) -> float:
    """
    Calculate minimum detectable effect size given sample sizes and variances.

    Returns: MDE as percentage of baseline
    """
    try:
        from scipy import stats
    except ImportError:
        return float('nan')

    n1, n2 = len(group1), len(group2)
    s1, s2 = std(group1), std(group2)
    m2 = mean(group2)  # baseline

    # Pooled standard deviation
    pooled_var = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
    pooled_sd = math.sqrt(pooled_var)

    # Standard error
    se = pooled_sd * math.sqrt(1/n1 + 1/n2)

    # Critical values
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    # Minimum detectable difference
    mde = (z_alpha + z_beta) * se

    # Convert to percentage
    mde_pct = (mde / m2) * 100

    return mde_pct


def analyze_comparison(baseline_dir: str, treatment_dir: str,
                       metric: str = "final_val_ppl",
                       baseline_name: str = "Baseline",
                       treatment_name: str = "Treatment") -> Dict[str, Any]:
    """
    Perform full statistical comparison between two conditions.
    """
    baseline = load_results(baseline_dir, metric)
    treatment = load_results(treatment_dir, metric)

    if not baseline or not treatment:
        return {"error": "No data loaded"}

    # Basic statistics
    baseline_mean = mean(baseline)
    baseline_std = std(baseline)
    treatment_mean = mean(treatment)
    treatment_std = std(treatment)

    # Difference
    diff = treatment_mean - baseline_mean
    diff_pct = (diff / baseline_mean) * 100

    # Statistical tests
    t_stat, p_value, df = welch_t_test(treatment, baseline)
    d_value, d_interp = cohens_d(treatment, baseline)
    ci_lower, ci_upper = confidence_interval(treatment, baseline)
    mde = minimum_detectable_effect(baseline, treatment)

    results = {
        "baseline": {
            "name": baseline_name,
            "n": len(baseline),
            "mean": baseline_mean,
            "std": baseline_std,
            "values": baseline
        },
        "treatment": {
            "name": treatment_name,
            "n": len(treatment),
            "mean": treatment_mean,
            "std": treatment_std,
            "values": treatment
        },
        "comparison": {
            "difference": diff,
            "difference_pct": diff_pct,
            "t_statistic": t_stat,
            "p_value": p_value,
            "degrees_of_freedom": df,
            "cohens_d": d_value,
            "cohens_d_interpretation": d_interp,
            "ci_95_lower_pct": ci_lower,
            "ci_95_upper_pct": ci_upper,
            "minimum_detectable_effect_pct": mde
        },
        "interpretation": {
            "significant": p_value < 0.05,
            "ci_excludes_zero": (ci_lower > 0) or (ci_upper < 0),
            "direction": "worse" if diff_pct > 0 else "better" if diff_pct < 0 else "same"
        }
    }

    return results


def format_results(results: Dict[str, Any]) -> str:
    """Format results for publication."""
    if "error" in results:
        return f"Error: {results['error']}"

    b = results["baseline"]
    t = results["treatment"]
    c = results["comparison"]
    i = results["interpretation"]

    output = []
    output.append("=" * 70)
    output.append(f"STATISTICAL COMPARISON: {t['name']} vs {b['name']}")
    output.append("=" * 70)
    output.append("")
    output.append("DESCRIPTIVE STATISTICS:")
    output.append(f"  {b['name']:20s}: {b['mean']:.4f} ± {b['std']:.4f} (n={b['n']})")
    output.append(f"  {t['name']:20s}: {t['mean']:.4f} ± {t['std']:.4f} (n={t['n']})")
    output.append("")
    output.append("COMPARISON:")
    output.append(f"  Difference: {c['difference']:+.4f} ({c['difference_pct']:+.2f}%)")
    output.append(f"  Direction: {i['direction']}")
    output.append("")
    output.append("STATISTICAL TESTS (Welch's t-test for unequal variances):")
    output.append(f"  t-statistic: {c['t_statistic']:.3f}")
    output.append(f"  p-value: {c['p_value']:.4f}")
    output.append(f"  Degrees of freedom: {c['degrees_of_freedom']:.1f}")
    output.append(f"  Significant (α=0.05): {'YES' if i['significant'] else 'NO'}")
    output.append("")
    output.append("EFFECT SIZE (Cohen's d with pooled SD):")
    output.append(f"  Cohen's d: {c['cohens_d']:.2f} ({c['cohens_d_interpretation']})")
    output.append("  Formula: d = (M_treatment - M_baseline) / SD_pooled")
    output.append("  SD_pooled = sqrt(((n1-1)*s1² + (n2-1)*s2²) / (n1+n2-2))")
    output.append("")
    output.append("95% CONFIDENCE INTERVAL (% difference from baseline):")
    output.append(f"  [{c['ci_95_lower_pct']:+.2f}%, {c['ci_95_upper_pct']:+.2f}%]")
    output.append(f"  Excludes zero: {'YES' if i['ci_excludes_zero'] else 'NO'}")
    output.append("")
    output.append("POWER ANALYSIS:")
    output.append(f"  Minimum detectable effect (80% power, α=0.05): ±{c['minimum_detectable_effect_pct']:.2f}%")
    output.append("")
    output.append("INTERPRETATION:")
    if i['significant'] and i['ci_excludes_zero']:
        output.append(f"  ✓ Statistically significant {i['direction']} effect detected")
    else:
        output.append(f"  ✗ No significant effect detected (effect <{c['minimum_detectable_effect_pct']:.1f}%)")
    output.append("=" * 70)

    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis for A+B+C study")
    parser.add_argument("--baseline", required=True, help="Path to baseline results directory")
    parser.add_argument("--treatment", required=True, help="Path to treatment results directory")
    parser.add_argument("--metric", default="final_val_ppl", help="Metric to compare")
    parser.add_argument("--baseline-name", default="Baseline", help="Name for baseline condition")
    parser.add_argument("--treatment-name", default="Treatment", help="Name for treatment condition")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    results = analyze_comparison(
        args.baseline, args.treatment, args.metric,
        args.baseline_name, args.treatment_name
    )

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(format_results(results))


if __name__ == "__main__":
    main()
