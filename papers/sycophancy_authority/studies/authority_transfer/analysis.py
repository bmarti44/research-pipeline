"""
Analysis for Authority Transfer Sycophancy Study.

Statistical tests:
1. Chi-square test for independence (authority × sycophancy)
2. Two-proportion z-test for pairwise comparisons
3. Fisher's exact test for small cell counts
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy import stats


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Calculate Wilson score confidence interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)

    z = stats.norm.ppf(1 - alpha / 2)
    p_hat = successes / n

    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = (z / denominator) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))

    return (max(0, center - margin), min(1, center + margin))


def chi_square_test(contingency_table: np.ndarray) -> dict[str, Any]:
    """Run chi-square test of independence."""
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    return {
        "test": "chi_square",
        "statistic": float(chi2),
        "p_value": float(p_value),
        "dof": int(dof),
        "expected_frequencies": expected.tolist(),
        "significant": bool(p_value < 0.05)
    }


def two_proportion_z_test(x1: int, n1: int, x2: int, n2: int) -> dict[str, Any]:
    """Two-proportion z-test for comparing two groups."""
    p1 = x1 / n1 if n1 > 0 else 0
    p2 = x2 / n2 if n2 > 0 else 0

    # Pooled proportion
    p_pooled = (x1 + x2) / (n1 + n2) if (n1 + n2) > 0 else 0

    # Standard error
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2)) if n1 > 0 and n2 > 0 else 0

    # Z statistic
    z = (p1 - p2) / se if se > 0 else 0

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "test": "two_proportion_z",
        "statistic": float(z),
        "p_value": float(p_value),
        "p1": float(p1),
        "p2": float(p2),
        "difference": float(p1 - p2),
        "significant": bool(p_value < 0.05)
    }


def fishers_exact_test(table_2x2: np.ndarray) -> dict[str, Any]:
    """Fisher's exact test for 2x2 tables."""
    odds_ratio, p_value = stats.fisher_exact(table_2x2)

    return {
        "test": "fishers_exact",
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05)
    }


def run_analysis(
    evaluations: list[dict[str, Any]],
    study_path: Path | None = None
) -> dict[str, Any]:
    """
    Run full statistical analysis on evaluation results.

    Args:
        evaluations: List of evaluation dicts
        study_path: Path to study directory (for loading additional data)

    Returns:
        Complete analysis results
    """
    # Separate by condition
    by_condition = {}
    for ev in evaluations:
        cond = ev.get("condition") or ev.get("authority_level", "unknown")
        if cond not in by_condition:
            by_condition[cond] = []
        by_condition[cond].append(ev)

    # Calculate aggregates
    aggregates = {}
    for cond, evals in by_condition.items():
        n = len(evals)
        sycophantic = sum(1 for e in evals if e.get("is_sycophantic"))
        agreed = sum(1 for e in evals if e.get("agreed"))
        corrected = sum(1 for e in evals if e.get("provided_correction"))

        ci_lower, ci_upper = wilson_ci(sycophantic, n)

        aggregates[cond] = {
            "n": n,
            "sycophantic": sycophantic,
            "sycophancy_rate": round(sycophantic / n, 4) if n > 0 else 0,
            "agreement_rate": round(agreed / n, 4) if n > 0 else 0,
            "correction_rate": round(corrected / n, 4) if n > 0 else 0,
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
        }

    # Build contingency table for chi-square
    # Rows: authority level, Columns: sycophantic vs not
    conditions = ["no_authority", "relevant_authority", "irrelevant_authority"]
    contingency = np.zeros((3, 2), dtype=int)

    for i, cond in enumerate(conditions):
        if cond in by_condition:
            evals = by_condition[cond]
            sycophantic = sum(1 for e in evals if e.get("is_sycophantic"))
            not_sycophantic = len(evals) - sycophantic
            contingency[i, 0] = sycophantic
            contingency[i, 1] = not_sycophantic

    # Run tests
    tests = {}

    # Chi-square test (overall)
    if contingency.sum() > 0 and np.all(contingency.sum(axis=0) > 0):
        tests["chi_square_overall"] = chi_square_test(contingency)

    # Pairwise comparisons
    if "no_authority" in aggregates and "irrelevant_authority" in aggregates:
        no_auth = aggregates["no_authority"]
        irrel = aggregates["irrelevant_authority"]
        tests["cross_domain_transfer"] = two_proportion_z_test(
            irrel["sycophantic"], irrel["n"],
            no_auth["sycophantic"], no_auth["n"]
        )

    if "no_authority" in aggregates and "relevant_authority" in aggregates:
        no_auth = aggregates["no_authority"]
        rel = aggregates["relevant_authority"]
        tests["relevant_vs_none"] = two_proportion_z_test(
            rel["sycophantic"], rel["n"],
            no_auth["sycophantic"], no_auth["n"]
        )

    if "relevant_authority" in aggregates and "irrelevant_authority" in aggregates:
        rel = aggregates["relevant_authority"]
        irrel = aggregates["irrelevant_authority"]
        tests["relevant_vs_irrelevant"] = two_proportion_z_test(
            rel["sycophantic"], rel["n"],
            irrel["sycophantic"], irrel["n"]
        )

    # Effect sizes
    effects = {}
    if "no_authority" in aggregates and "irrelevant_authority" in aggregates:
        baseline = aggregates["no_authority"]["sycophancy_rate"]
        treatment = aggregates["irrelevant_authority"]["sycophancy_rate"]
        effects["cross_domain_effect"] = {
            "baseline": baseline,
            "treatment": treatment,
            "absolute_increase": round(treatment - baseline, 4),
            "relative_increase": round((treatment - baseline) / baseline, 4) if baseline > 0 else None,
        }

    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "total_trials": len(evaluations),
        "aggregates": aggregates,
        "tests": tests,
        "effects": effects,
    }


def generate_results_markdown(analysis: dict[str, Any], hypothesis: str) -> str:
    """Generate markdown results report."""
    md = f"""# authority_transfer: Results

## Hypothesis

{hypothesis}

## Summary

Generated: {analysis['generated']}
Total trials: {analysis['total_trials']}

## Aggregates by Condition

| Condition | n | Sycophancy Rate | 95% CI |
|-----------|---|-----------------|--------|
"""

    for cond in ["no_authority", "relevant_authority", "irrelevant_authority"]:
        if cond in analysis["aggregates"]:
            agg = analysis["aggregates"][cond]
            md += f"| {cond} | {agg['n']} | {agg['sycophancy_rate']:.1%} | [{agg['ci_lower']:.1%}, {agg['ci_upper']:.1%}] |\n"

    md += "\n## Statistical Tests\n\n"

    for test_name, test_result in analysis.get("tests", {}).items():
        md += f"### {test_name}\n\n"
        md += f"- Test: {test_result.get('test')}\n"
        md += f"- Statistic: {test_result.get('statistic', test_result.get('odds_ratio', 'N/A')):.4f}\n"
        md += f"- p-value: {test_result.get('p_value', 0):.6f}\n"
        md += f"- Significant: {test_result.get('significant')}\n\n"

    if analysis.get("effects"):
        md += "## Effect Sizes\n\n"
        for effect_name, effect in analysis["effects"].items():
            md += f"### {effect_name}\n\n"
            md += f"- Baseline: {effect.get('baseline', 0):.1%}\n"
            md += f"- Treatment: {effect.get('treatment', 0):.1%}\n"
            md += f"- Absolute increase: {effect.get('absolute_increase', 0):.1%}\n"
            if effect.get('relative_increase') is not None:
                md += f"- Relative increase: {effect.get('relative_increase', 0):.1%}\n"
            md += "\n"

    md += "\n---\n\n*Generated by Research Pipeline*\n"

    return md


if __name__ == "__main__":
    # Test with mock data
    mock_evals = [
        {"authority_level": "no_authority", "is_sycophantic": False, "agreed": False},
        {"authority_level": "no_authority", "is_sycophantic": True, "agreed": True},
        {"authority_level": "relevant_authority", "is_sycophantic": True, "agreed": True},
        {"authority_level": "relevant_authority", "is_sycophantic": True, "agreed": True},
        {"authority_level": "irrelevant_authority", "is_sycophantic": True, "agreed": True},
        {"authority_level": "irrelevant_authority", "is_sycophantic": False, "agreed": False},
    ]

    results = run_analysis(mock_evals)
    print(json.dumps(results, indent=2))
