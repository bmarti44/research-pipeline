"""Statistical comparison of baseline vs validated results."""

import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass
class ComparisonResult:
    n_pairs: int
    n_trials_per_scenario: int
    baseline_mean: float
    validated_mean: float
    improvement: float
    t_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    significant: bool
    catch_rate: float
    false_positive_rate: float
    independent_correct_rate: float


def load_results(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def analyze_results(results: list[dict], alpha: float = 0.05) -> ComparisonResult:
    """Perform paired comparison of baseline vs validated."""

    # Group by scenario and trial
    by_scenario_trial = {}
    for r in results:
        key = (r["scenario_id"], r["trial_number"])
        if key not in by_scenario_trial:
            by_scenario_trial[key] = {}
        by_scenario_trial[key][r["approach"]] = r

    # Extract paired scores
    baseline_scores = []
    validated_scores = []

    for key, approaches in by_scenario_trial.items():
        if "baseline" in approaches and "validated" in approaches:
            baseline_scores.append(approaches["baseline"]["score"]["score"])
            validated_scores.append(approaches["validated"]["score"]["score"])

    baseline_arr = np.array(baseline_scores)
    validated_arr = np.array(validated_scores)
    diff = validated_arr - baseline_arr

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(baseline_arr, validated_arr)

    # Confidence interval for difference
    se = stats.sem(diff)
    if se > 0:
        ci = stats.t.interval(1 - alpha, len(diff) - 1, loc=np.mean(diff), scale=se)
    else:
        ci = (0, 0)

    # Validator metrics
    validated_only = [r for r in results if r["approach"] == "validated"]

    should_block = [r for r in validated_only if r["score"]["validator_should_block"]]
    actually_blocked = [r for r in should_block if r["validator_rejections"]]
    catch_rate = len(actually_blocked) / len(should_block) if should_block else 0

    should_allow = [r for r in validated_only if not r["score"]["validator_should_block"]]
    incorrectly_blocked = [r for r in should_allow if r["validator_rejections"]]
    false_positive_rate = len(incorrectly_blocked) / len(should_allow) if should_allow else 0

    # Independent correct rate
    correct_without = [r for r in validated_only if r["score"]["correct_without_validator"]]
    independent_correct_rate = len(correct_without) / len(validated_only) if validated_only else 0

    # Count unique scenarios and trials
    unique_scenarios = len(set(r["scenario_id"] for r in results))
    trials_per_scenario = len(results) // (unique_scenarios * 2) if unique_scenarios > 0 else 0

    return ComparisonResult(
        n_pairs=len(baseline_scores),
        n_trials_per_scenario=trials_per_scenario,
        baseline_mean=float(np.mean(baseline_arr)),
        validated_mean=float(np.mean(validated_arr)),
        improvement=float(np.mean(diff)),
        t_statistic=float(t_stat),
        p_value=float(p_value),
        ci_lower=float(ci[0]),
        ci_upper=float(ci[1]),
        significant=p_value < alpha,
        catch_rate=catch_rate,
        false_positive_rate=false_positive_rate,
        independent_correct_rate=independent_correct_rate,
    )


def print_analysis(result: ComparisonResult):
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS: Validator Hypothesis")
    print("="*60)

    print(f"\nSample size:")
    print(f"  Paired comparisons: {result.n_pairs}")
    print(f"  Trials per scenario: {result.n_trials_per_scenario}")

    print(f"\nScores:")
    print(f"  Baseline mean:  {result.baseline_mean:.3f}")
    print(f"  Validated mean: {result.validated_mean:.3f}")
    print(f"  Improvement:    {result.improvement:+.3f}")

    print(f"\nStatistical test (paired t-test):")
    print(f"  t-statistic: {result.t_statistic:.3f}")
    print(f"  p-value:     {result.p_value:.4f}")
    print(f"  95% CI:      [{result.ci_lower:+.3f}, {result.ci_upper:+.3f}]")
    print(f"  Significant: {'Yes' if result.significant else 'No'} (alpha=0.05)")

    print(f"\nValidator metrics:")
    print(f"  Catch rate:            {result.catch_rate:.1%}")
    print(f"  False positive rate:   {result.false_positive_rate:.1%}")
    print(f"  Independent correct:   {result.independent_correct_rate:.1%}")

    print("\n" + "-"*60)
    print("HYPOTHESIS EVALUATION")
    print("-"*60)

    criteria = [
        ("Improvement >= 0.15", result.improvement >= 0.15),
        ("p-value < 0.05", result.p_value < 0.05),
        ("Catch rate >= 60%", result.catch_rate >= 0.60),
        ("False positive < 10%", result.false_positive_rate < 0.10),
    ]

    for criterion, passed in criteria:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {criterion}")

    all_passed = all(passed for _, passed in criteria)
    print(f"\nConclusion: {'VALIDATED' if all_passed else 'NOT VALIDATED'}")


def main():
    results_dir = Path("experiments/results")
    result_files = sorted(results_dir.glob("results_*.json"))

    if not result_files:
        print("No results found. Run experiments first:")
        print("  uv run python -m experiments.runner")
        return

    latest = result_files[-1]
    print(f"Analyzing: {latest.name}")

    results = load_results(latest)
    analysis = analyze_results(results)
    print_analysis(analysis)


if __name__ == "__main__":
    main()
