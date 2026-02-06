"""
Analysis module for the research pipeline.

Provides functions for:
- Aggregating evaluation scores
- Running statistical tests
- Generating analysis summaries
- Checking statistical assumptions
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Callable
import numpy as np

from .stats import get_test, list_tests, TestResult
from .utils import now_iso, json_dump, json_load


# =============================================================================
# Analysis Results
# =============================================================================

@dataclass
class ConditionAggregate:
    """Aggregated statistics for a single condition."""
    condition: str
    n: int
    successes: int
    rate: float
    ci_lower: float
    ci_upper: float
    mean: Optional[float] = None
    std: Optional[float] = None
    median: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "condition": self.condition,
            "n": self.n,
            "successes": self.successes,
            "rate": self.rate,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
        }


@dataclass
class AnalysisResult:
    """Complete analysis result."""
    aggregates: dict[str, ConditionAggregate]
    tests: list[TestResult]
    assumptions: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=now_iso)

    def to_dict(self) -> dict:
        return {
            "aggregates": {k: v.to_dict() for k, v in self.aggregates.items()},
            "tests": [t.to_dict() for t in self.tests],
            "assumptions": self.assumptions,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    def save(self, output_dir: Path) -> None:
        """Save analysis results to directory."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save aggregates
        json_dump(
            {k: v.to_dict() for k, v in self.aggregates.items()},
            output_dir / "aggregates.json"
        )

        # Save tests
        json_dump(
            [t.to_dict() for t in self.tests],
            output_dir / "tests.json"
        )

        # Save assumptions
        json_dump(self.assumptions, output_dir / "assumptions.json")

        # Save full result
        json_dump(self.to_dict(), output_dir / "analysis_result.json")


# =============================================================================
# Score Aggregation
# =============================================================================

def aggregate_scores(
    scores: list[dict],
    trial_conditions: dict[str, str],
    mode: str = "intent",
    conditions: Optional[list[str]] = None,
) -> dict[str, ConditionAggregate]:
    """
    Aggregate evaluation scores by condition.

    Args:
        scores: List of score dictionaries from evaluation stage
        trial_conditions: Mapping of trial_id to condition name
        mode: Evaluation mode to use (strict, intent, functional)
        conditions: List of condition names (auto-detected if None)

    Returns:
        Dictionary mapping condition name to ConditionAggregate
    """
    # Auto-detect conditions if not provided
    if conditions is None:
        conditions = list(set(trial_conditions.values()))

    # Collect scores by condition
    by_condition: dict[str, list[int]] = {c: [] for c in conditions}

    for score in scores:
        trial_id = score.get("trial_id", "")
        condition = trial_conditions.get(trial_id, "")

        if condition not in by_condition:
            continue

        modes = score.get("modes", {})
        mode_result = modes.get(mode, {})
        correct = mode_result.get("correct", False)

        by_condition[condition].append(1 if correct else 0)

    # Compute aggregates
    aggregates = {}

    for condition, values in by_condition.items():
        if not values:
            continue

        n = len(values)
        successes = sum(values)
        rate = successes / n if n > 0 else 0

        # Wilson confidence interval
        ci_lower, ci_upper = wilson_ci(successes, n)

        aggregates[condition] = ConditionAggregate(
            condition=condition,
            n=n,
            successes=successes,
            rate=rate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            mean=np.mean(values) if values else None,
            std=np.std(values) if values else None,
            median=np.median(values) if values else None,
        )

    return aggregates


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    Compute Wilson confidence interval for a proportion.

    Args:
        successes: Number of successes
        n: Total trials
        alpha: Significance level

    Returns:
        (lower, upper) confidence interval
    """
    if n == 0:
        return (0.0, 1.0)

    from scipy import stats

    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / n

    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator

    lower = max(0, center - margin)
    upper = min(1, center + margin)

    return (lower, upper)


# =============================================================================
# Statistical Testing
# =============================================================================

def run_statistical_tests(
    aggregates: dict[str, ConditionAggregate],
    test_names: list[str],
    alpha: float = 0.05,
) -> list[TestResult]:
    """
    Run specified statistical tests on aggregated data.

    Args:
        aggregates: Aggregated scores by condition
        test_names: Names of tests to run (from stats registry)
        alpha: Significance level

    Returns:
        List of TestResult objects
    """
    results = []
    conditions = list(aggregates.keys())

    for test_name in test_names:
        try:
            test_fn = get_test(test_name)
        except KeyError:
            # Unknown test, skip
            continue

        # Different tests need different inputs
        if test_name in ["two_proportion_z", "fisher_exact", "chi_square"]:
            # Need two conditions for comparison
            if len(conditions) >= 2:
                c1, c2 = conditions[0], conditions[1]
                agg1, agg2 = aggregates[c1], aggregates[c2]

                result = test_fn(
                    successes1=agg1.successes,
                    n1=agg1.n,
                    successes2=agg2.successes,
                    n2=agg2.n,
                    alpha=alpha,
                )
                result.metadata["conditions"] = [c1, c2]
                results.append(result)

        elif test_name == "mcnemar":
            # McNemar needs paired data - skip if not available
            pass

        elif test_name in ["bootstrap_proportion_diff"]:
            if len(conditions) >= 2:
                c1, c2 = conditions[0], conditions[1]
                agg1, agg2 = aggregates[c1], aggregates[c2]

                result = test_fn(
                    successes1=agg1.successes,
                    n1=agg1.n,
                    successes2=agg2.successes,
                    n2=agg2.n,
                    alpha=alpha,
                )
                result.metadata["conditions"] = [c1, c2]
                results.append(result)

    return results


def check_assumptions(
    aggregates: dict[str, ConditionAggregate],
    test_names: list[str],
) -> dict[str, Any]:
    """
    Check statistical assumptions for the planned tests.

    Args:
        aggregates: Aggregated scores by condition
        test_names: Names of tests to run

    Returns:
        Dictionary of assumption checks
    """
    assumptions = {
        "sample_sizes": {c: agg.n for c, agg in aggregates.items()},
        "checks": [],
        "warnings": [],
    }

    min_n = min(agg.n for agg in aggregates.values()) if aggregates else 0
    total_n = sum(agg.n for agg in aggregates.values())

    # Check sample size
    if min_n < 10:
        assumptions["warnings"].append(
            f"Small sample size (min n={min_n}). Consider Fisher's exact test."
        )

    # Check expected cell counts for chi-square
    if "chi_square" in test_names:
        min_expected = min(agg.successes for agg in aggregates.values()) if aggregates else 0
        assumptions["checks"].append({
            "test": "chi_square",
            "check": "min_expected_count",
            "value": min_expected,
            "threshold": 5,
            "passed": min_expected >= 5,
            "message": "Chi-square requires expected cell counts >= 5"
        })

    # Check for extreme proportions
    for c, agg in aggregates.items():
        if agg.rate < 0.05 or agg.rate > 0.95:
            assumptions["warnings"].append(
                f"Extreme proportion in {c}: {agg.rate:.2%}. Normal approximation may be poor."
            )

    return assumptions


# =============================================================================
# High-Level Analysis Functions
# =============================================================================

def run_analysis(
    scores: list[dict],
    config: dict,
    study_path: Optional[Path] = None,
) -> dict:
    """
    Run complete analysis on evaluation scores.

    This is the main entry point called by the pipeline runner.

    Args:
        scores: List of score dictionaries from evaluation stage
        config: Study configuration
        study_path: Path to study directory (for loading response metadata)

    Returns:
        Dictionary with aggregates, tests, and assumptions
    """
    # Extract conditions from config
    conditions = [c["name"] for c in config.get("conditions", [])]

    # Build trial_id -> condition mapping from responses
    trial_conditions = {}
    if study_path:
        responses_path = study_path / "stages" / "3_execute" / "responses"
        if responses_path.exists():
            for resp_file in responses_path.glob("trial_*.json"):
                with open(resp_file) as f:
                    resp_data = json.load(f)
                trial_conditions[resp_data["trial_id"]] = resp_data.get("condition", "")

    # Get evaluation mode
    eval_mode = config.get("evaluation", {}).get("primary_mode", "intent")

    # Aggregate scores
    aggregates = aggregate_scores(
        scores=scores,
        trial_conditions=trial_conditions,
        mode=eval_mode,
        conditions=conditions,
    )

    # Get tests to run
    analysis_config = config.get("analysis", {})
    test_configs = analysis_config.get("tests", [])
    test_names = [t.get("name", t) if isinstance(t, dict) else t for t in test_configs]

    # Default to two_proportion_z if no tests specified
    if not test_names:
        test_names = ["two_proportion_z"]

    alpha = analysis_config.get("alpha", 0.05)

    # Run tests
    test_results = run_statistical_tests(aggregates, test_names, alpha)

    # Check assumptions
    assumptions = check_assumptions(aggregates, test_names)

    # Convert to serializable format
    return {
        "aggregates": {k: v.to_dict() for k, v in aggregates.items()},
        "tests": [
            {
                "name": t.test_name,
                "description": t.metadata.get("conditions", ""),
                "statistic": float(t.statistic) if t.statistic is not None else None,
                "p_value": float(t.p_value) if t.p_value is not None else None,
                "significant": bool(t.significant),
                "effect_size": float(t.effect_size) if t.effect_size is not None else None,
            }
            for t in test_results
        ],
        "assumptions": assumptions,
    }


def analyze_by_category(
    scores: list[dict],
    trial_metadata: dict[str, dict],
    trial_conditions: dict[str, str],
    category_key: str = "category",
) -> dict[str, dict[str, ConditionAggregate]]:
    """
    Run analysis broken down by task category.

    Args:
        scores: List of score dictionaries
        trial_metadata: Mapping of trial_id to task metadata
        trial_conditions: Mapping of trial_id to condition
        category_key: Key in metadata for category

    Returns:
        Nested dict: category -> condition -> ConditionAggregate
    """
    # Group trials by category
    by_category: dict[str, list[dict]] = {}

    for score in scores:
        trial_id = score.get("trial_id", "")
        metadata = trial_metadata.get(trial_id, {})
        category = metadata.get(category_key, "uncategorized")

        if category not in by_category:
            by_category[category] = []
        by_category[category].append(score)

    # Analyze each category
    results = {}
    for category, category_scores in by_category.items():
        # Filter trial_conditions to this category
        category_trial_ids = {s["trial_id"] for s in category_scores}
        category_conditions = {
            tid: cond for tid, cond in trial_conditions.items()
            if tid in category_trial_ids
        }

        results[category] = aggregate_scores(
            scores=category_scores,
            trial_conditions=category_conditions,
        )

    return results


# =============================================================================
# Analysis Report Generation
# =============================================================================

def generate_analysis_summary(result: AnalysisResult) -> str:
    """
    Generate a human-readable summary of analysis results.

    Args:
        result: AnalysisResult object

    Returns:
        Markdown-formatted summary
    """
    lines = ["# Analysis Summary", ""]

    # Aggregates
    lines.append("## Condition Aggregates")
    lines.append("")
    lines.append("| Condition | N | Successes | Rate | 95% CI |")
    lines.append("|-----------|---|-----------|------|--------|")

    for name, agg in result.aggregates.items():
        ci = f"[{agg.ci_lower:.3f}, {agg.ci_upper:.3f}]"
        lines.append(f"| {name} | {agg.n} | {agg.successes} | {agg.rate:.3f} | {ci} |")

    lines.append("")

    # Statistical tests
    lines.append("## Statistical Tests")
    lines.append("")

    for test in result.tests:
        lines.append(f"### {test.test_name}")
        lines.append("")
        lines.append(f"- **Statistic**: {test.statistic:.4f}" if test.statistic else "- **Statistic**: N/A")
        lines.append(f"- **p-value**: {test.p_value:.4f}" if test.p_value else "- **p-value**: N/A")
        lines.append(f"- **Significant**: {'Yes' if test.significant else 'No'}")
        if test.effect_size is not None:
            lines.append(f"- **Effect size**: {test.effect_size:.4f}")
        if test.ci_lower is not None and test.ci_upper is not None:
            lines.append(f"- **95% CI**: [{test.ci_lower:.4f}, {test.ci_upper:.4f}]")
        lines.append("")

    # Assumptions
    if result.assumptions.get("warnings"):
        lines.append("## Warnings")
        lines.append("")
        for warning in result.assumptions["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")

    lines.append(f"*Generated: {result.timestamp}*")

    return "\n".join(lines)
