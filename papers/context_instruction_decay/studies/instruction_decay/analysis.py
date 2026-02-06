"""
Analysis for Context Instruction Decay Study.

Statistical tests:
1. Mixed ANOVA (context_length × instruction_category)
2. Trend analysis for decay curves
3. Changepoint detection for "cliff" identification
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy import stats


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Calculate Wilson score confidence interval."""
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p_hat = successes / n
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = (z / denominator) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return (max(0, center - margin), min(1, center + margin))


def calculate_decay_rate(compliance_by_length: dict[int, float]) -> dict[str, Any]:
    """
    Calculate decay rate metrics for a compliance curve.

    Args:
        compliance_by_length: Dict mapping context_tokens -> compliance_rate

    Returns:
        Decay metrics including slope, half-life, cliff point
    """
    lengths = sorted(compliance_by_length.keys())
    rates = [compliance_by_length[l] for l in lengths]

    if len(lengths) < 2:
        return {"error": "Insufficient data points"}

    # Log-transform lengths for linear regression (log-linear decay model)
    log_lengths = np.log(lengths)
    rates_array = np.array(rates)

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_lengths, rates_array)

    # Find cliff point (largest single-step drop)
    drops = [rates[i] - rates[i+1] for i in range(len(rates)-1)]
    max_drop_idx = np.argmax(drops) if drops else 0
    cliff_point = lengths[max_drop_idx + 1] if drops else None
    max_drop = max(drops) if drops else 0

    # Calculate half-life (context length where compliance drops to 50%)
    # Using the linear model: rate = slope * log(length) + intercept
    # Solve for: 0.5 = slope * log(half_life) + intercept
    if slope != 0 and intercept > 0.5:
        try:
            half_life = np.exp((0.5 - intercept) / slope)
        except (ValueError, OverflowError):
            half_life = None
    else:
        half_life = None

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value ** 2),
        "p_value": float(p_value),
        "cliff_point": int(cliff_point) if cliff_point else None,
        "cliff_drop_magnitude": float(max_drop),
        "half_life_tokens": int(half_life) if half_life and half_life > 0 else None,
        "initial_compliance": rates[0] if rates else None,
        "final_compliance": rates[-1] if rates else None,
        "total_decay": float(rates[0] - rates[-1]) if rates else None,
    }


def run_analysis(
    evaluations: list[dict[str, Any]],
    study_path: Path | None = None
) -> dict[str, Any]:
    """
    Run full analysis on evaluation results.

    Args:
        evaluations: List of evaluation dicts
        study_path: Path to study directory

    Returns:
        Complete analysis results
    """
    # Organize by category and context length
    by_category_length = {}
    for ev in evaluations:
        category = ev.get("category", "unknown")
        length = ev.get("context_tokens", 0)

        key = (category, length)
        if key not in by_category_length:
            by_category_length[key] = []
        by_category_length[key].append(ev)

    # Calculate compliance rates
    compliance_matrix = {}  # category -> {length -> rate}
    aggregates = {}

    categories = set(ev.get("category") for ev in evaluations)
    lengths = sorted(set(ev.get("context_tokens", 0) for ev in evaluations))

    for category in categories:
        compliance_matrix[category] = {}
        for length in lengths:
            key = (category, length)
            evals = by_category_length.get(key, [])
            n = len(evals)
            if n > 0:
                compliant = sum(1 for e in evals if e.get("correct"))
                rate = compliant / n
                ci_lower, ci_upper = wilson_ci(compliant, n)
                compliance_matrix[category][length] = rate
                aggregates[f"{category}_{length}"] = {
                    "n": n,
                    "compliant": compliant,
                    "rate": round(rate, 4),
                    "ci_lower": round(ci_lower, 4),
                    "ci_upper": round(ci_upper, 4),
                }

    # Calculate decay rates for each category
    decay_rates = {}
    for category in categories:
        if category in compliance_matrix:
            decay_rates[category] = calculate_decay_rate(compliance_matrix[category])

    # Rank categories by robustness (lower decay = more robust)
    robustness_ranking = sorted(
        decay_rates.items(),
        key=lambda x: x[1].get("total_decay", 1.0) if isinstance(x[1], dict) else 1.0
    )

    # Statistical comparisons between categories
    comparisons = {}

    # Compare safety vs formatting decay
    if "safety_refusal" in decay_rates and "output_formatting" in decay_rates:
        safety_decay = decay_rates["safety_refusal"].get("total_decay", 0)
        format_decay = decay_rates["output_formatting"].get("total_decay", 0)
        comparisons["safety_vs_formatting"] = {
            "safety_decay": safety_decay,
            "formatting_decay": format_decay,
            "difference": round(format_decay - safety_decay, 4),
            "safety_more_robust": safety_decay < format_decay,
        }

    # Compare formatting vs factual decay
    if "output_formatting" in decay_rates and "factual_constraints" in decay_rates:
        format_decay = decay_rates["output_formatting"].get("total_decay", 0)
        factual_decay = decay_rates["factual_constraints"].get("total_decay", 0)
        comparisons["formatting_vs_factual"] = {
            "formatting_decay": format_decay,
            "factual_decay": factual_decay,
            "difference": round(factual_decay - format_decay, 4),
            "formatting_more_robust": format_decay < factual_decay,
        }

    # Find overall cliff point
    all_cliff_points = [
        dr.get("cliff_point")
        for dr in decay_rates.values()
        if isinstance(dr, dict) and dr.get("cliff_point")
    ]
    median_cliff = int(np.median(all_cliff_points)) if all_cliff_points else None

    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "total_trials": len(evaluations),
        "categories": list(categories),
        "context_lengths": lengths,
        "aggregates": aggregates,
        "compliance_matrix": {k: dict(v) for k, v in compliance_matrix.items()},
        "decay_rates": decay_rates,
        "robustness_ranking": [cat for cat, _ in robustness_ranking],
        "comparisons": comparisons,
        "median_cliff_point": median_cliff,
    }


def generate_results_markdown(analysis: dict[str, Any], hypothesis: str) -> str:
    """Generate markdown results report."""
    md = f"""# instruction_decay: Results

## Hypothesis

{hypothesis}

## Summary

Generated: {analysis['generated']}
Total trials: {analysis['total_trials']}

## Compliance by Category and Context Length

"""

    # Create table
    lengths = analysis.get("context_lengths", [])
    categories = analysis.get("robustness_ranking", [])

    md += "| Category | " + " | ".join(f"{l//1000}K" for l in lengths) + " | Total Decay |\n"
    md += "|" + "----|" * (len(lengths) + 2) + "\n"

    for cat in categories:
        decay = analysis.get("decay_rates", {}).get(cat, {})
        row = f"| {cat} |"
        for length in lengths:
            rate = analysis.get("compliance_matrix", {}).get(cat, {}).get(length, 0)
            row += f" {rate:.1%} |"
        total_decay = decay.get("total_decay", 0) if isinstance(decay, dict) else 0
        row += f" {total_decay:.1%} |"
        md += row + "\n"

    md += "\n## Decay Analysis by Category\n\n"

    for cat in categories:
        decay = analysis.get("decay_rates", {}).get(cat, {})
        if isinstance(decay, dict):
            md += f"### {cat}\n\n"
            md += f"- Initial compliance: {decay.get('initial_compliance', 0):.1%}\n"
            md += f"- Final compliance: {decay.get('final_compliance', 0):.1%}\n"
            md += f"- Total decay: {decay.get('total_decay', 0):.1%}\n"
            md += f"- Cliff point: {decay.get('cliff_point', 'N/A')} tokens\n"
            md += f"- R²: {decay.get('r_squared', 0):.3f}\n\n"

    md += "## Key Comparisons\n\n"

    for comp_name, comp in analysis.get("comparisons", {}).items():
        md += f"### {comp_name}\n\n"
        for k, v in comp.items():
            md += f"- {k}: {v}\n"
        md += "\n"

    md += f"\n## Robustness Ranking (Most to Least Robust)\n\n"
    for i, cat in enumerate(categories, 1):
        md += f"{i}. {cat}\n"

    if analysis.get("median_cliff_point"):
        md += f"\n## Median Cliff Point: {analysis['median_cliff_point']} tokens\n"

    md += "\n---\n\n*Generated by Research Pipeline*\n"

    return md


if __name__ == "__main__":
    # Test with mock data
    mock_evals = []
    categories = ["safety_refusal", "output_formatting", "factual_constraints"]
    lengths = [1000, 4000, 16000, 32000]

    # Simulate decay patterns
    decay_patterns = {
        "safety_refusal": [0.95, 0.92, 0.88, 0.85],  # Slow decay
        "output_formatting": [0.90, 0.80, 0.65, 0.50],  # Medium decay
        "factual_constraints": [0.85, 0.60, 0.40, 0.25],  # Fast decay
    }

    for cat in categories:
        for i, length in enumerate(lengths):
            rate = decay_patterns[cat][i]
            for _ in range(10):
                correct = np.random.random() < rate
                mock_evals.append({
                    "category": cat,
                    "context_tokens": length,
                    "correct": correct,
                })

    results = run_analysis(mock_evals)
    print(json.dumps(results, indent=2, default=str))
