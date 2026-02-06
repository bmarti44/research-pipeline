"""
Analysis logic for the study.

Defines statistical tests and aggregation methods.
"""

import json
from pathlib import Path
from scipy import stats
import numpy as np


def run_analysis(scores: list[dict], config: dict, study_path: Path = None) -> dict:
    """
    Run statistical analysis on evaluation scores.

    Args:
        scores: List of score dictionaries from evaluation stage
        config: Study configuration
        study_path: Path to study directory (for loading response metadata)

    Returns:
        Dictionary with:
        - aggregates: Summary statistics
        - tests: Statistical test results
        - assumptions: Assumption check results
    """
    # Extract conditions from config
    conditions = [c["name"] for c in config.get("conditions", [])]

    # Aggregate by condition
    by_condition = {c: [] for c in conditions}

    # Build a map of trial_id to condition from responses
    trial_to_condition = {}
    if study_path:
        responses_path = study_path / "stages" / "3_execute" / "responses"
        if responses_path.exists():
            for resp_file in responses_path.glob("trial_*.json"):
                with open(resp_file) as f:
                    resp_data = json.load(f)
                trial_to_condition[resp_data["trial_id"]] = resp_data.get("condition", "")

    for score in scores:
        # Get trial info from score
        trial_id = score.get("trial_id", "")

        # Get condition from response data
        condition = trial_to_condition.get(trial_id, "")

        modes = score.get("modes", {})
        intent_correct = modes.get("intent", {}).get("correct", False)

        # Append to correct condition only
        if condition in by_condition:
            by_condition[condition].append(1 if intent_correct else 0)

    # Calculate aggregates
    aggregates = {}
    for condition, values in by_condition.items():
        if values:
            aggregates[condition] = {
                "n": len(values),
                "correct": sum(values),
                "rate": sum(values) / len(values),
                "ci_lower": None,
                "ci_upper": None,
            }

            # Wilson confidence interval
            if len(values) > 0:
                from statsmodels.stats.proportion import proportion_confint
                ci = proportion_confint(sum(values), len(values), method="wilson")
                aggregates[condition]["ci_lower"] = ci[0]
                aggregates[condition]["ci_upper"] = ci[1]

    # Run statistical tests
    tests = []

    # Two-proportion z-test (if we have two conditions)
    if len(conditions) >= 2:
        c1, c2 = conditions[0], conditions[1]
        v1, v2 = by_condition[c1], by_condition[c2]

        if v1 and v2:
            n1, n2 = len(v1), len(v2)
            p1, p2 = sum(v1) / n1, sum(v2) / n2

            # Pooled proportion
            p_pool = (sum(v1) + sum(v2)) / (n1 + n2)
            se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

            if se > 0:
                z = (p1 - p2) / se
                p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            else:
                z = 0
                p_value = 1.0

            tests.append({
                "name": "two_proportion_z",
                "description": f"Comparing {c1} vs {c2}",
                "statistic": float(z),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),
                "effect_size": float(p1 - p2),
            })

    # Assumption checks
    assumptions = {
        "sample_sizes": {c: len(v) for c, v in by_condition.items()},
        "min_expected_count": min(
            sum(v) for v in by_condition.values() if v
        ) if any(by_condition.values()) else 0,
        "notes": "Two-proportion z-test assumes independence and sufficient sample size",
    }

    return {
        "aggregates": aggregates,
        "tests": tests,
        "assumptions": assumptions,
    }
