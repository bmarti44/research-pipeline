"""
Analysis for Multi-Pass Diminishing Returns Study.

Tests competing hypotheses:
1. Geometric Decay: Same detection probability per pass for all error types
2. Error Stratification: Different detection curves for different error types
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


def geometric_model(n: np.ndarray, p: float) -> np.ndarray:
    """Cumulative detection: P(detect by pass n) = 1 - (1-p)^n"""
    return 1 - (1 - p) ** n


def fit_geometric_model(
    pass_numbers: np.ndarray,
    detection_rates: np.ndarray
) -> dict[str, Any]:
    """Fit geometric model and return parameters + fit statistics."""
    try:
        popt, pcov = curve_fit(
            geometric_model, pass_numbers, detection_rates,
            p0=[0.5], bounds=(0.01, 0.99)
        )
        p_fitted = popt[0]
        predicted = geometric_model(pass_numbers, p_fitted)

        # R² and residuals
        ss_res = np.sum((detection_rates - predicted) ** 2)
        ss_tot = np.sum((detection_rates - np.mean(detection_rates)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # AIC (assuming normal errors)
        n = len(detection_rates)
        k = 1  # one parameter
        aic = n * np.log(ss_res / n) + 2 * k if ss_res > 0 else float('inf')

        return {
            "p": float(p_fitted),
            "r_squared": float(r_squared),
            "aic": float(aic),
            "residuals": (detection_rates - predicted).tolist(),
        }
    except Exception as e:
        return {"error": str(e)}


def run_analysis(
    evaluations: list[dict[str, Any]],
    study_path: Path | None = None
) -> dict[str, Any]:
    """
    Run full analysis comparing geometric decay vs stratification.

    Args:
        evaluations: List of multi-pass evaluation results
        study_path: Path to study directory

    Returns:
        Complete analysis results
    """
    # Organize data by error type and pass
    from collections import defaultdict

    by_type_pass = defaultdict(lambda: defaultdict(list))

    for ev in evaluations:
        error_type = ev.get("error_type", "unknown")
        for pr in ev.get("pass_results", []):
            pass_num = pr.get("pass_number", 0)
            detected = 1 if pr.get("cumulative_detected") else 0
            by_type_pass[error_type][pass_num].append(detected)

    # Calculate detection rates
    detection_curves = {}
    for error_type, by_pass in by_type_pass.items():
        curve = {}
        for pass_num in sorted(by_pass.keys()):
            values = by_pass[pass_num]
            curve[pass_num] = sum(values) / len(values) if values else 0
        detection_curves[error_type] = curve

    # Also calculate aggregate curve
    aggregate_by_pass = defaultdict(list)
    for ev in evaluations:
        for pr in ev.get("pass_results", []):
            pass_num = pr.get("pass_number", 0)
            detected = 1 if pr.get("cumulative_detected") else 0
            aggregate_by_pass[pass_num].append(detected)

    aggregate_curve = {
        p: sum(v) / len(v) for p, v in sorted(aggregate_by_pass.items())
    }

    # Fit models
    pass_nums = np.array(sorted(aggregate_curve.keys()))
    aggregate_rates = np.array([aggregate_curve[p] for p in pass_nums])

    # Model 1: Single geometric (same p for all)
    single_geometric = fit_geometric_model(pass_nums, aggregate_rates)

    # Model 2: Stratified geometric (different p per type)
    stratified_fits = {}
    for error_type, curve in detection_curves.items():
        type_pass_nums = np.array(sorted(curve.keys()))
        type_rates = np.array([curve[p] for p in type_pass_nums])
        if len(type_rates) >= 3:
            stratified_fits[error_type] = fit_geometric_model(type_pass_nums, type_rates)

    # Calculate combined AIC for stratified model
    stratified_aic = sum(
        f.get("aic", 0) for f in stratified_fits.values()
        if "aic" in f
    )

    # Model comparison
    model_comparison = {
        "geometric_aic": single_geometric.get("aic"),
        "stratified_aic": stratified_aic,
        "stratified_better": stratified_aic < single_geometric.get("aic", float('inf')),
        "aic_difference": single_geometric.get("aic", 0) - stratified_aic,
    }

    # Test for interaction (error_type × pass_number)
    # Using two-way ANOVA approximation
    # Collect data for ANOVA
    groups_for_anova = []
    for ev in evaluations:
        error_type = ev.get("error_type", "unknown")
        for pr in ev.get("pass_results", []):
            pass_num = pr.get("pass_number", 0)
            detected = 1 if pr.get("cumulative_detected") else 0
            groups_for_anova.append({
                "error_type": error_type,
                "pass_num": pass_num,
                "detected": detected,
            })

    # Find knee point in aggregate curve
    rates_list = [aggregate_curve[p] for p in sorted(aggregate_curve.keys())]
    knee_point = None
    if len(rates_list) >= 3:
        # Find where marginal improvement drops most
        improvements = [rates_list[i+1] - rates_list[i] for i in range(len(rates_list)-1)]
        if improvements:
            min_improvement_idx = improvements.index(min(improvements))
            knee_point = sorted(aggregate_curve.keys())[min_improvement_idx + 1]

    # Calculate per-type detection probabilities
    type_detection_probs = {}
    for error_type, fit in stratified_fits.items():
        if "p" in fit:
            type_detection_probs[error_type] = {
                "p_per_pass": fit["p"],
                "expected_passes_to_90pct": (
                    int(np.ceil(np.log(0.1) / np.log(1 - fit["p"])))
                    if fit["p"] > 0 else None
                ),
            }

    # Rank error types by detectability
    detectability_ranking = sorted(
        type_detection_probs.items(),
        key=lambda x: x[1].get("p_per_pass", 0),
        reverse=True
    )

    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "total_evaluations": len(evaluations),

        "detection_curves": {k: dict(v) for k, v in detection_curves.items()},
        "aggregate_curve": dict(aggregate_curve),

        "model_fits": {
            "single_geometric": single_geometric,
            "stratified": stratified_fits,
        },

        "model_comparison": model_comparison,
        "hypothesis_support": {
            "supports_stratification": model_comparison["stratified_better"],
            "evidence_strength": (
                "strong" if model_comparison["aic_difference"] > 10 else
                "moderate" if model_comparison["aic_difference"] > 4 else
                "weak"
            ),
        },

        "knee_point": knee_point,
        "type_detection_probs": type_detection_probs,
        "detectability_ranking": [t for t, _ in detectability_ranking],
    }


def generate_results_markdown(analysis: dict[str, Any], hypothesis: str) -> str:
    """Generate markdown results report."""
    md = f"""# pass_detection_curves: Results

## Hypothesis

{hypothesis}

## Summary

Generated: {analysis['generated']}
Total evaluations: {analysis['total_evaluations']}

## Aggregate Detection Curve

| Pass | Cumulative Detection Rate |
|------|---------------------------|
"""
    for pass_num, rate in sorted(analysis.get("aggregate_curve", {}).items()):
        md += f"| {pass_num} | {rate:.1%} |\n"

    md += f"\n**Knee Point**: Pass {analysis.get('knee_point', 'N/A')}\n\n"

    md += "## Detection Curves by Error Type\n\n"

    for error_type, curve in analysis.get("detection_curves", {}).items():
        fit = analysis.get("model_fits", {}).get("stratified", {}).get(error_type, {})
        p = fit.get("p", "N/A")
        r2 = fit.get("r_squared", "N/A")

        md += f"### {error_type}\n\n"
        md += f"- Per-pass detection probability: {p:.3f}\n" if isinstance(p, float) else f"- Per-pass detection probability: {p}\n"
        md += f"- Model R²: {r2:.3f}\n" if isinstance(r2, float) else f"- Model R²: {r2}\n"

        prob_info = analysis.get("type_detection_probs", {}).get(error_type, {})
        if prob_info.get("expected_passes_to_90pct"):
            md += f"- Expected passes for 90% detection: {prob_info['expected_passes_to_90pct']}\n"
        md += "\n"

    md += "## Model Comparison\n\n"

    comparison = analysis.get("model_comparison", {})
    md += f"- Single Geometric Model AIC: {comparison.get('geometric_aic', 'N/A'):.2f}\n" if comparison.get('geometric_aic') else ""
    md += f"- Stratified Model AIC: {comparison.get('stratified_aic', 'N/A'):.2f}\n" if comparison.get('stratified_aic') else ""
    md += f"- AIC Difference: {comparison.get('aic_difference', 'N/A'):.2f}\n" if comparison.get('aic_difference') else ""
    md += f"- **Stratified model better fit**: {comparison.get('stratified_better', 'N/A')}\n\n"

    support = analysis.get("hypothesis_support", {})
    md += f"## Hypothesis Support\n\n"
    md += f"- **Supports Stratification Hypothesis**: {support.get('supports_stratification', 'N/A')}\n"
    md += f"- **Evidence Strength**: {support.get('evidence_strength', 'N/A')}\n\n"

    md += "## Detectability Ranking (Highest to Lowest)\n\n"
    for i, error_type in enumerate(analysis.get("detectability_ranking", []), 1):
        md += f"{i}. {error_type}\n"

    md += "\n---\n\n*Generated by Research Pipeline*\n"

    return md


if __name__ == "__main__":
    # Test with mock data
    mock_evals = []

    # Simulate different detection curves by error type
    error_configs = {
        "syntax_error": 0.8,      # High detection per pass
        "type_error": 0.7,
        "logic_error": 0.5,       # Moderate
        "edge_case_error": 0.4,
        "security_error": 0.2,    # Low
        "concurrency_error": 0.15,
    }

    np.random.seed(42)

    for error_type, p in error_configs.items():
        for sample in range(10):
            pass_results = []
            cumulative = False
            for pass_num in range(1, 11):
                if not cumulative:
                    detected = np.random.random() < p
                    if detected:
                        cumulative = True
                pass_results.append({
                    "pass_number": pass_num,
                    "cumulative_detected": cumulative,
                })

            mock_evals.append({
                "task_id": f"{error_type}_{sample}",
                "error_type": error_type,
                "pass_results": pass_results,
            })

    results = run_analysis(mock_evals)
    print(json.dumps(results, indent=2, default=str))
