"""
Analysis for Phase Separation Workflow Study.

Tests whether phase separation, context clearing, and external memory
each contribute to workflow effectiveness, and through what mechanisms.
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy import stats


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score confidence interval."""
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p_hat = successes / n
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = (z / denominator) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return (max(0, center - margin), min(1, center + margin))


def compare_workflows(
    group1_evals: list[dict],
    group2_evals: list[dict],
    metric: str = "task_completed"
) -> dict[str, Any]:
    """
    Compare two workflow groups on a metric.

    Args:
        group1_evals: Evaluations for first workflow
        group2_evals: Evaluations for second workflow
        metric: Metric to compare

    Returns:
        Comparison statistics
    """
    if metric == "task_completed":
        # Binary metric - use chi-square or z-test
        n1, n2 = len(group1_evals), len(group2_evals)
        x1 = sum(1 for e in group1_evals if e.get(metric))
        x2 = sum(1 for e in group2_evals if e.get(metric))

        p1, p2 = x1/n1 if n1 > 0 else 0, x2/n2 if n2 > 0 else 0

        # Two-proportion z-test
        p_pooled = (x1 + x2) / (n1 + n2) if (n1 + n2) > 0 else 0
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2)) if n1 > 0 and n2 > 0 else 0
        z = (p1 - p2) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        return {
            "test": "two_proportion_z",
            "group1_rate": round(p1, 4),
            "group2_rate": round(p2, 4),
            "difference": round(p1 - p2, 4),
            "z_statistic": round(z, 4),
            "p_value": round(p_value, 6),
            "significant": p_value < 0.05,
        }

    else:
        # Continuous metric - use t-test
        values1 = [e.get(metric, 0) for e in group1_evals]
        values2 = [e.get(metric, 0) for e in group2_evals]

        if len(values1) < 2 or len(values2) < 2:
            return {"error": "Insufficient data"}

        t_stat, p_value = stats.ttest_ind(values1, values2)

        return {
            "test": "independent_t",
            "group1_mean": round(np.mean(values1), 4),
            "group2_mean": round(np.mean(values2), 4),
            "difference": round(np.mean(values1) - np.mean(values2), 4),
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "significant": p_value < 0.05,
        }


def run_analysis(
    evaluations: list[dict[str, Any]],
    study_path: Path | None = None
) -> dict[str, Any]:
    """
    Run full analysis comparing workflows.

    Args:
        evaluations: List of workflow evaluation results
        study_path: Path to study directory

    Returns:
        Complete analysis results
    """
    from collections import defaultdict

    # Organize by workflow
    by_workflow = defaultdict(list)
    for ev in evaluations:
        workflow = ev.get("workflow", "unknown")
        by_workflow[workflow].append(ev)

    # Calculate aggregates per workflow
    aggregates = {}
    for workflow, evals in by_workflow.items():
        n = len(evals)
        if n == 0:
            continue

        completed = sum(1 for e in evals if e.get("task_completed"))
        ci_lower, ci_upper = wilson_ci(completed, n)

        code_quality_scores = [
            e.get("code_quality", {}).get("quality_score", 0)
            for e in evals
        ]

        review_scores = [
            e.get("review_analysis", {}).get("quality_score", 0)
            for e in evals if e.get("review_analysis")
        ]

        errors_caught = [e.get("errors_caught_in_review", 0) for e in evals]

        aggregates[workflow] = {
            "n": n,
            "completion_rate": round(completed / n, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "avg_code_quality": round(np.mean(code_quality_scores), 4),
            "avg_review_quality": round(np.mean(review_scores), 4) if review_scores else None,
            "avg_errors_caught": round(np.mean(errors_caught), 2),
        }

    # Planned contrasts
    contrasts = {}

    # Primary: full_workflow vs continuous_context
    if "full_workflow" in by_workflow and "continuous_context" in by_workflow:
        contrasts["full_vs_continuous"] = compare_workflows(
            by_workflow["full_workflow"],
            by_workflow["continuous_context"],
            "task_completed"
        )

    # Effect of context clearing
    if "phases_with_clearing" in by_workflow and "phases_no_clearing" in by_workflow:
        contrasts["clearing_effect"] = compare_workflows(
            by_workflow["phases_with_clearing"],
            by_workflow["phases_no_clearing"],
            "task_completed"
        )

    # Effect of external memory
    if "external_memory_no_clearing" in by_workflow and "phases_no_clearing" in by_workflow:
        contrasts["external_memory_effect"] = compare_workflows(
            by_workflow["external_memory_no_clearing"],
            by_workflow["phases_no_clearing"],
            "task_completed"
        )

    # Value of review phase
    if "full_workflow" in by_workflow and "no_review" in by_workflow:
        contrasts["review_value"] = compare_workflows(
            by_workflow["full_workflow"],
            by_workflow["no_review"],
            "task_completed"
        )

    # Mechanism analysis: critique quality
    if "phases_with_clearing" in by_workflow and "phases_no_clearing" in by_workflow:
        with_clearing = [
            e for e in by_workflow["phases_with_clearing"]
            if e.get("review_analysis")
        ]
        without_clearing = [
            e for e in by_workflow["phases_no_clearing"]
            if e.get("review_analysis")
        ]

        if with_clearing and without_clearing:
            critique_with = [e["review_analysis"]["quality_score"] for e in with_clearing]
            critique_without = [e["review_analysis"]["quality_score"] for e in without_clearing]

            if len(critique_with) >= 2 and len(critique_without) >= 2:
                t_stat, p_value = stats.ttest_ind(critique_with, critique_without)
                contrasts["genuine_critique_test"] = {
                    "test": "independent_t",
                    "with_clearing_mean": round(np.mean(critique_with), 4),
                    "without_clearing_mean": round(np.mean(critique_without), 4),
                    "t_statistic": round(t_stat, 4),
                    "p_value": round(p_value, 6),
                    "significant": p_value < 0.05,
                    "supports_consistency_pressure_hypothesis": (
                        np.mean(critique_with) > np.mean(critique_without)
                    ),
                }

    # Context collapse test
    if "iterative_refine_external" in by_workflow and "iterative_refine_in_context" in by_workflow:
        contrasts["context_collapse_test"] = compare_workflows(
            by_workflow["iterative_refine_external"],
            by_workflow["iterative_refine_in_context"],
            "task_completed"
        )

        # Also compare detail preservation
        external_preservation = [
            e.get("detail_preservation", 1.0)
            for e in by_workflow["iterative_refine_external"]
        ]
        in_context_preservation = [
            e.get("detail_preservation", 1.0)
            for e in by_workflow["iterative_refine_in_context"]
        ]

        if len(external_preservation) >= 2 and len(in_context_preservation) >= 2:
            t_stat, p_value = stats.ttest_ind(external_preservation, in_context_preservation)
            contrasts["detail_preservation_test"] = {
                "test": "independent_t",
                "external_mean": round(np.mean(external_preservation), 4),
                "in_context_mean": round(np.mean(in_context_preservation), 4),
                "t_statistic": round(t_stat, 4),
                "p_value": round(p_value, 6),
                "significant": p_value < 0.05,
                "supports_context_collapse_hypothesis": (
                    np.mean(external_preservation) > np.mean(in_context_preservation)
                ),
            }

    # Mechanism summary
    mechanisms = {
        "mode_interference": {
            "supported": contrasts.get("full_vs_continuous", {}).get("significant", False),
            "evidence": "Full workflow significantly outperforms continuous context",
        },
        "consistency_pressure": {
            "supported": contrasts.get("genuine_critique_test", {}).get("supports_consistency_pressure_hypothesis", False),
            "evidence": "Critiques with clearing are more substantive",
        },
        "error_firewalling": {
            "supported": contrasts.get("review_value", {}).get("significant", False),
            "evidence": "Review phase significantly improves outcomes",
        },
        "context_collapse": {
            "supported": contrasts.get("detail_preservation_test", {}).get("supports_context_collapse_hypothesis", False),
            "evidence": "External memory preserves more details than continuous context",
            "ace_related": "Supports ACE paper's context collapse hypothesis",
        },
    }

    # Rank workflows
    workflow_ranking = sorted(
        aggregates.items(),
        key=lambda x: x[1].get("completion_rate", 0),
        reverse=True
    )

    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "total_evaluations": len(evaluations),
        "aggregates": aggregates,
        "contrasts": contrasts,
        "mechanisms": mechanisms,
        "workflow_ranking": [w for w, _ in workflow_ranking],
        "best_workflow": workflow_ranking[0][0] if workflow_ranking else None,
    }


def generate_results_markdown(analysis: dict[str, Any], hypothesis: str) -> str:
    """Generate markdown results report."""
    md = f"""# workflow_comparison: Results

## Hypothesis

{hypothesis}

## Summary

Generated: {analysis['generated']}
Total evaluations: {analysis['total_evaluations']}

## Workflow Performance

| Workflow | n | Completion Rate | 95% CI | Avg Code Quality |
|----------|---|-----------------|--------|------------------|
"""

    for workflow in analysis.get("workflow_ranking", []):
        agg = analysis.get("aggregates", {}).get(workflow, {})
        md += (
            f"| {workflow} | {agg.get('n', 0)} | "
            f"{agg.get('completion_rate', 0):.1%} | "
            f"[{agg.get('ci_lower', 0):.1%}, {agg.get('ci_upper', 0):.1%}] | "
            f"{agg.get('avg_code_quality', 0):.2f} |\n"
        )

    md += "\n## Planned Contrasts\n\n"

    for contrast_name, result in analysis.get("contrasts", {}).items():
        md += f"### {contrast_name}\n\n"
        if "error" in result:
            md += f"Error: {result['error']}\n\n"
        else:
            md += f"- Test: {result.get('test')}\n"
            if "group1_rate" in result:
                md += f"- Group 1 rate: {result['group1_rate']:.1%}\n"
                md += f"- Group 2 rate: {result['group2_rate']:.1%}\n"
            elif "group1_mean" in result:
                md += f"- Group 1 mean: {result['group1_mean']:.3f}\n"
                md += f"- Group 2 mean: {result['group2_mean']:.3f}\n"
            md += f"- Difference: {result.get('difference', 0):.4f}\n"
            md += f"- p-value: {result.get('p_value', 1):.6f}\n"
            md += f"- **Significant**: {result.get('significant', False)}\n\n"

    md += "## Mechanism Analysis\n\n"

    for mechanism, data in analysis.get("mechanisms", {}).items():
        md += f"### {mechanism.replace('_', ' ').title()}\n\n"
        md += f"- **Supported**: {data.get('supported', 'Unknown')}\n"
        md += f"- Evidence: {data.get('evidence', 'N/A')}\n\n"

    md += f"\n## Best Workflow\n\n**{analysis.get('best_workflow', 'Unknown')}**\n\n"

    md += "\n---\n\n*Generated by Research Pipeline*\n"

    return md


if __name__ == "__main__":
    # Test with mock data
    np.random.seed(42)

    mock_evals = []
    workflows = {
        "continuous_context": 0.50,
        "phases_no_clearing": 0.60,
        "phases_with_clearing": 0.70,
        "external_memory_no_clearing": 0.65,
        "full_workflow": 0.85,
        "no_review": 0.75,
        "iterative_refine_in_context": 0.55,  # Context collapse hurts
        "iterative_refine_external": 0.72,    # External memory helps
    }

    for workflow, success_rate in workflows.items():
        for i in range(20):
            completed = np.random.random() < success_rate
            mock_evals.append({
                "task_id": f"{workflow}_{i}",
                "workflow": workflow,
                "task_completed": completed,
                "code_quality": {"quality_score": np.random.uniform(0.3, 0.9)},
                "review_analysis": {
                    "quality_score": np.random.uniform(0.4, 0.8)
                } if "no_review" not in workflow else None,
                "errors_caught_in_review": np.random.randint(0, 3),
            })

    results = run_analysis(mock_evals)
    print(json.dumps(results, indent=2, default=str))
