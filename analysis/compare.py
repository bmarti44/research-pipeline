"""Statistical comparison of baseline vs validated results."""

import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy import stats


@dataclass
class PowerAnalysis:
    """Results of statistical power analysis."""
    observed_effect_size: float  # Cohen's d
    sample_size: int  # Number of paired observations
    achieved_power: float  # Power with observed effect and sample
    required_n_for_80_power: int  # Sample size needed for 80% power
    required_n_for_90_power: int  # Sample size needed for 90% power
    alpha: float  # Significance level used


def compute_power_analysis(
    cohens_d: float,
    n: int,
    alpha: float = 0.05,
) -> PowerAnalysis:
    """
    Compute post-hoc power analysis for paired t-test.

    Uses the non-central t-distribution to compute power.
    Also computes required sample sizes for 80% and 90% power.

    Args:
        cohens_d: Observed effect size (Cohen's d)
        n: Number of paired observations
        alpha: Significance level

    Returns:
        PowerAnalysis with achieved power and required sample sizes
    """
    # Non-centrality parameter for paired t-test
    # ncp = d * sqrt(n)
    ncp = abs(cohens_d) * np.sqrt(n)

    # Critical t-value for two-tailed test
    df = n - 1
    t_crit = stats.t.ppf(1 - alpha / 2, df)

    # Power = P(reject H0 | H1 is true)
    # = P(|t| > t_crit | ncp)
    # = 1 - P(-t_crit < t < t_crit | ncp)
    # Using non-central t distribution
    power = 1 - (stats.nct.cdf(t_crit, df, ncp) - stats.nct.cdf(-t_crit, df, ncp))

    # Compute required sample sizes for target powers
    def required_n_for_power(target_power: float, effect_size: float) -> int:
        """Binary search for required sample size."""
        if abs(effect_size) < 0.001:
            return 10000  # Effectively infinite for zero effect

        low, high = 2, 10000
        while low < high:
            mid = (low + high) // 2
            ncp_mid = abs(effect_size) * np.sqrt(mid)
            df_mid = mid - 1
            t_crit_mid = stats.t.ppf(1 - alpha / 2, df_mid)
            power_mid = 1 - (stats.nct.cdf(t_crit_mid, df_mid, ncp_mid) -
                            stats.nct.cdf(-t_crit_mid, df_mid, ncp_mid))
            if power_mid >= target_power:
                high = mid
            else:
                low = mid + 1
        return low

    return PowerAnalysis(
        observed_effect_size=cohens_d,
        sample_size=n,
        achieved_power=float(power),
        required_n_for_80_power=required_n_for_power(0.80, cohens_d),
        required_n_for_90_power=required_n_for_power(0.90, cohens_d),
        alpha=alpha,
    )


@dataclass
class ComparisonResult:
    n_pairs: int
    n_trials_per_scenario: int
    baseline_mean: float
    validated_mean: float
    improvement: float
    # Parametric test (t-test)
    t_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    significant: bool
    # Non-parametric test (Wilcoxon signed-rank)
    wilcoxon_statistic: float
    wilcoxon_p_value: float
    wilcoxon_significant: bool
    # Effect size
    cohens_d: float
    effect_size_interpretation: str  # "negligible", "small", "medium", "large"
    # Validator metrics
    catch_rate: float
    false_positive_rate: float
    independent_correct_rate: float


def load_results(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> tuple[float, str]:
    """
    Compute Cohen's d effect size for paired samples.

    Returns:
        (effect_size, interpretation)
        interpretation is one of: "negligible", "small", "medium", "large"
    """
    diff = group2 - group1
    d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0

    # Cohen's d interpretation thresholds
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return float(d), interpretation


def analyze_results(results: list[dict], alpha: float = 0.05) -> ComparisonResult:
    """Perform paired comparison of baseline vs validated.

    Uses both parametric (t-test) and non-parametric (Wilcoxon) tests,
    plus Cohen's d effect size for a complete statistical picture.
    """

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

    # Parametric: Paired t-test
    t_stat, p_value = stats.ttest_rel(baseline_arr, validated_arr)

    # Non-parametric: Wilcoxon signed-rank test
    # This is more appropriate for ordinal data like our scores
    # Handle case where all differences are zero
    non_zero_diff = diff[diff != 0]
    if len(non_zero_diff) > 0:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(baseline_arr, validated_arr, alternative='two-sided')
    else:
        # All pairs are tied - no significant difference
        wilcoxon_stat, wilcoxon_p = 0.0, 1.0

    # Effect size: Cohen's d
    cohens_d, effect_interpretation = compute_cohens_d(baseline_arr, validated_arr)

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
        wilcoxon_statistic=float(wilcoxon_stat),
        wilcoxon_p_value=float(wilcoxon_p),
        wilcoxon_significant=wilcoxon_p < alpha,
        cohens_d=cohens_d,
        effect_size_interpretation=effect_interpretation,
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

    print(f"\nParametric test (paired t-test):")
    print(f"  t-statistic: {result.t_statistic:.3f}")
    print(f"  p-value:     {result.p_value:.4f}")
    print(f"  95% CI:      [{result.ci_lower:+.3f}, {result.ci_upper:+.3f}]")
    print(f"  Significant: {'Yes' if result.significant else 'No'} (alpha=0.05)")

    print(f"\nNon-parametric test (Wilcoxon signed-rank):")
    print(f"  W-statistic: {result.wilcoxon_statistic:.3f}")
    print(f"  p-value:     {result.wilcoxon_p_value:.4f}")
    print(f"  Significant: {'Yes' if result.wilcoxon_significant else 'No'} (alpha=0.05)")

    print(f"\nEffect size (Cohen's d):")
    print(f"  d = {result.cohens_d:.3f} ({result.effect_size_interpretation})")

    print(f"\nValidator metrics:")
    print(f"  Catch rate:            {result.catch_rate:.1%}")
    print(f"  False positive rate:   {result.false_positive_rate:.1%}")
    print(f"  Independent correct:   {result.independent_correct_rate:.1%}")

    print("\n" + "-"*60)
    print("HYPOTHESIS EVALUATION")
    print("-"*60)

    criteria = [
        ("Improvement >= 0.15", result.improvement >= 0.15),
        ("t-test p-value < 0.05", result.p_value < 0.05),
        ("Wilcoxon p-value < 0.05", result.wilcoxon_p_value < 0.05),
        ("Catch rate >= 60%", result.catch_rate >= 0.60),
        ("False positive < 10%", result.false_positive_rate < 0.10),
    ]

    for criterion, passed in criteria:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {criterion}")

    # Both statistical tests must pass
    stats_pass = result.significant and result.wilcoxon_significant
    practical_pass = result.improvement >= 0.15 and result.catch_rate >= 0.60 and result.false_positive_rate < 0.10

    if stats_pass and practical_pass:
        conclusion = "VALIDATED (both tests significant, practical criteria met)"
    elif stats_pass:
        conclusion = "PARTIALLY VALIDATED (statistically significant but practical criteria not met)"
    elif practical_pass:
        conclusion = "PARTIALLY VALIDATED (practical criteria met but not statistically significant)"
    else:
        conclusion = "NOT VALIDATED"

    print(f"\nConclusion: {conclusion}")


def analyze_by_rule(results: list[dict]) -> dict[str, dict]:
    """Break down results by failure mode / rule with detailed metrics."""
    validated = [r for r in results if r["approach"] == "validated"]
    baseline = [r for r in results if r["approach"] == "baseline"]

    # Group by scenario prefix (f1_, f4_, etc.)
    rule_stats = {}

    # Get unique scenario prefixes
    prefixes = set()
    for r in validated:
        sid = r["scenario_id"]
        if "_" in sid:
            prefix = sid.split("_")[0]
            prefixes.add(prefix)

    for prefix in sorted(prefixes):
        # Get all results for this prefix
        val_results = [r for r in validated if r["scenario_id"].startswith(prefix + "_")]
        base_results = [r for r in baseline if r["scenario_id"].startswith(prefix + "_")]

        if not val_results:
            continue

        val_scores = [r["score"]["score"] for r in val_results]
        base_scores = [r["score"]["score"] for r in base_results]

        # Compute per-rule metrics
        should_block_results = [r for r in val_results if r["score"]["validator_should_block"]]
        should_allow_results = [r for r in val_results if not r["score"]["validator_should_block"]]

        # True positives: should block and did block
        true_positives = sum(1 for r in should_block_results if r["validator_rejections"])
        # False negatives: should block but didn't
        false_negatives = sum(1 for r in should_block_results if not r["validator_rejections"])
        # True negatives: should allow and did allow
        true_negatives = sum(1 for r in should_allow_results if not r["validator_rejections"])
        # False positives: should allow but blocked
        false_positives = sum(1 for r in should_allow_results if r["validator_rejections"])

        total = len(val_results)
        catch_rate = true_positives / len(should_block_results) if should_block_results else None
        false_positive_rate = false_positives / len(should_allow_results) if should_allow_results else None

        # Precision and recall for this rule
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else None
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall and (precision + recall) > 0 else None

        # Effect size for this rule (Cohen's d)
        if base_scores and val_scores:
            base_arr = np.array(base_scores)
            val_arr = np.array(val_scores)
            diff = val_arr - base_arr
            cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
        else:
            cohens_d = 0

        rule_stats[prefix] = {
            "count": len(val_results),
            "baseline_mean": sum(base_scores) / len(base_scores) if base_scores else 0,
            "validated_mean": sum(val_scores) / len(val_scores),
            "improvement": (sum(val_scores) / len(val_scores)) - (sum(base_scores) / len(base_scores)) if base_scores else 0,
            "cohens_d": float(cohens_d),
            # Confusion matrix
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            # Derived metrics
            "catch_rate": catch_rate,
            "false_positive_rate": false_positive_rate,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return rule_stats


def print_rule_breakdown(rule_stats: dict[str, dict]):
    """Print per-rule breakdown with detailed metrics."""
    print("\n" + "-"*70)
    print("PER-RULE BREAKDOWN")
    print("-"*70)

    # Summary table
    print(f"\n{'Rule':<8} {'N':>4} {'Base':>6} {'Valid':>6} {'Δ':>7} {'d':>6} {'Catch':>7} {'FP':>6}")
    print("-"*70)

    for rule, stats in sorted(rule_stats.items()):
        catch_str = f"{stats['catch_rate']:.0%}" if stats['catch_rate'] is not None else "N/A"
        fp_str = f"{stats['false_positive_rate']:.0%}" if stats['false_positive_rate'] is not None else "N/A"
        d_str = f"{stats['cohens_d']:.2f}" if stats['cohens_d'] else "0.00"

        print(f"{rule:<8} {stats['count']:>4} {stats['baseline_mean']:>6.2f} "
              f"{stats['validated_mean']:>6.2f} {stats['improvement']:>+7.3f} "
              f"{d_str:>6} {catch_str:>7} {fp_str:>6}")

    # Detailed confusion matrix for each rule
    print("\n" + "-"*70)
    print("CONFUSION MATRICES (per-rule)")
    print("-"*70)

    for rule, stats in sorted(rule_stats.items()):
        tp, fp = stats['true_positives'], stats['false_positives']
        fn, tn = stats['false_negatives'], stats['true_negatives']

        if tp + fp + fn + tn == 0:
            continue

        print(f"\n{rule.upper()}:")
        print(f"                  Predicted")
        print(f"                  Block    Allow")
        print(f"  Actual Block  [{tp:>5}]  [{fn:>5}]")
        print(f"  Actual Allow  [{fp:>5}]  [{tn:>5}]")

        if stats['precision'] is not None and stats['recall'] is not None and stats['f1'] is not None:
            print(f"  Precision: {stats['precision']:.2f}, Recall: {stats['recall']:.2f}, F1: {stats['f1']:.2f}")

    # Identify most valuable rules
    print("\n" + "-"*70)
    print("RULE VALUE ANALYSIS")
    print("-"*70)

    # Sort by improvement
    by_improvement = sorted(
        [(r, s) for r, s in rule_stats.items() if s['improvement'] != 0],
        key=lambda x: x[1]['improvement'],
        reverse=True
    )

    if by_improvement:
        print("\nRules ranked by score improvement:")
        for rank, (rule, stats) in enumerate(by_improvement, 1):
            d_interp = "large" if abs(stats['cohens_d']) >= 0.8 else "medium" if abs(stats['cohens_d']) >= 0.5 else "small" if abs(stats['cohens_d']) >= 0.2 else "negligible"
            print(f"  {rank}. {rule.upper()}: {stats['improvement']:+.3f} (d={stats['cohens_d']:.2f}, {d_interp})")


def print_power_analysis(power: PowerAnalysis):
    """Print power analysis results."""
    print("\n" + "-"*60)
    print("STATISTICAL POWER ANALYSIS")
    print("-"*60)

    print(f"\nObserved:")
    print(f"  Effect size (Cohen's d): {power.observed_effect_size:.3f}")
    print(f"  Sample size (n pairs):   {power.sample_size}")
    print(f"  Achieved power:          {power.achieved_power:.1%}")

    print(f"\nPower interpretation:")
    if power.achieved_power >= 0.80:
        print(f"  Adequate power (>= 80%) to detect effects of this size")
    elif power.achieved_power >= 0.50:
        print(f"  Moderate power - some risk of Type II error (false negative)")
    else:
        print(f"  LOW POWER - high risk of missing true effects")
        print(f"  Consider increasing sample size")

    print(f"\nSample size requirements (at alpha={power.alpha}):")
    print(f"  For 80% power: n = {power.required_n_for_80_power} pairs")
    print(f"  For 90% power: n = {power.required_n_for_90_power} pairs")

    if power.sample_size < power.required_n_for_80_power:
        shortfall = power.required_n_for_80_power - power.sample_size
        print(f"\n  WARNING: Current sample ({power.sample_size}) is {shortfall} pairs short of 80% power")


def load_sensitivity_analysis() -> Optional[dict]:
    """Load threshold sensitivity analysis if available."""
    sensitivity_path = Path(__file__).parent.parent / "calibration" / "sensitivity_analysis.json"
    if sensitivity_path.exists():
        with open(sensitivity_path) as f:
            return json.load(f)
    return None


def generate_markdown_report(
    analysis: ComparisonResult,
    power: PowerAnalysis,
    rule_stats: dict[str, dict],
    results_file: Path,
) -> str:
    """Generate a markdown report of the analysis."""
    lines = []
    lines.append("# Validator Experiment Analysis Report")
    lines.append("")
    lines.append(f"**Results file:** `{results_file.name}`")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Paired comparisons | {analysis.n_pairs} |")
    lines.append(f"| Trials per scenario | {analysis.n_trials_per_scenario} |")
    lines.append(f"| Baseline mean | {analysis.baseline_mean:.3f} |")
    lines.append(f"| Validated mean | {analysis.validated_mean:.3f} |")
    lines.append(f"| **Improvement** | **{analysis.improvement:+.3f}** |")
    lines.append("")

    # Statistical Tests
    lines.append("## Statistical Tests")
    lines.append("")
    lines.append("### Parametric (Paired t-test)")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| t-statistic | {analysis.t_statistic:.3f} |")
    lines.append(f"| p-value | {analysis.p_value:.4f} |")
    lines.append(f"| 95% CI | [{analysis.ci_lower:+.3f}, {analysis.ci_upper:+.3f}] |")
    lines.append(f"| Significant | {'Yes ✓' if analysis.significant else 'No ✗'} |")
    lines.append("")

    lines.append("### Non-Parametric (Wilcoxon signed-rank)")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| W-statistic | {analysis.wilcoxon_statistic:.3f} |")
    lines.append(f"| p-value | {analysis.wilcoxon_p_value:.4f} |")
    lines.append(f"| Significant | {'Yes ✓' if analysis.wilcoxon_significant else 'No ✗'} |")
    lines.append("")

    lines.append("### Effect Size")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Cohen's d | {analysis.cohens_d:.3f} |")
    lines.append(f"| Interpretation | {analysis.effect_size_interpretation} |")
    lines.append("")

    # Validator Metrics
    lines.append("## Validator Metrics")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Catch rate | {analysis.catch_rate:.1%} |")
    lines.append(f"| False positive rate | {analysis.false_positive_rate:.1%} |")
    lines.append(f"| Independent correct rate | {analysis.independent_correct_rate:.1%} |")
    lines.append("")

    # Hypothesis Evaluation
    lines.append("## Hypothesis Evaluation")
    lines.append("")
    criteria = [
        ("Improvement >= 0.15", analysis.improvement >= 0.15),
        ("t-test p-value < 0.05", analysis.p_value < 0.05),
        ("Wilcoxon p-value < 0.05", analysis.wilcoxon_p_value < 0.05),
        ("Catch rate >= 60%", analysis.catch_rate >= 0.60),
        ("False positive < 10%", analysis.false_positive_rate < 0.10),
    ]
    lines.append(f"| Criterion | Status |")
    lines.append(f"|-----------|--------|")
    for criterion, passed in criteria:
        status = "✓ PASS" if passed else "✗ FAIL"
        lines.append(f"| {criterion} | {status} |")
    lines.append("")

    stats_pass = analysis.significant and analysis.wilcoxon_significant
    practical_pass = analysis.improvement >= 0.15 and analysis.catch_rate >= 0.60 and analysis.false_positive_rate < 0.10
    if stats_pass and practical_pass:
        conclusion = "**VALIDATED** - Both tests significant, practical criteria met"
    elif stats_pass:
        conclusion = "**PARTIALLY VALIDATED** - Statistically significant but practical criteria not met"
    elif practical_pass:
        conclusion = "**PARTIALLY VALIDATED** - Practical criteria met but not statistically significant"
    else:
        conclusion = "**NOT VALIDATED**"
    lines.append(f"**Conclusion:** {conclusion}")
    lines.append("")

    # Power Analysis
    lines.append("## Power Analysis")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Observed effect size (d) | {power.observed_effect_size:.3f} |")
    lines.append(f"| Sample size (n pairs) | {power.sample_size} |")
    lines.append(f"| Achieved power | {power.achieved_power:.1%} |")
    lines.append(f"| Required n for 80% power | {power.required_n_for_80_power} |")
    lines.append(f"| Required n for 90% power | {power.required_n_for_90_power} |")
    lines.append("")

    if power.achieved_power >= 0.80:
        lines.append("> ✓ Adequate power (>= 80%) to detect effects of this size")
    elif power.achieved_power >= 0.50:
        lines.append("> ⚠ Moderate power - some risk of Type II error")
    else:
        lines.append("> ⚠ LOW POWER - high risk of missing true effects. Consider increasing sample size.")
    lines.append("")

    # Per-Rule Breakdown
    lines.append("## Per-Rule Breakdown")
    lines.append("")
    lines.append(f"| Rule | N | Baseline | Validated | Δ | d | Catch | FP |")
    lines.append(f"|------|---|----------|-----------|---|---|-------|-----|")
    for rule, stats in sorted(rule_stats.items()):
        catch_str = f"{stats['catch_rate']:.0%}" if stats['catch_rate'] is not None else "N/A"
        fp_str = f"{stats['false_positive_rate']:.0%}" if stats['false_positive_rate'] is not None else "N/A"
        d_str = f"{stats['cohens_d']:.2f}" if stats['cohens_d'] else "0.00"
        lines.append(f"| {rule.upper()} | {stats['count']} | {stats['baseline_mean']:.2f} | "
                    f"{stats['validated_mean']:.2f} | {stats['improvement']:+.3f} | "
                    f"{d_str} | {catch_str} | {fp_str} |")
    lines.append("")

    # Rule Value Analysis
    by_improvement = sorted(
        [(r, s) for r, s in rule_stats.items() if s['improvement'] != 0],
        key=lambda x: x[1]['improvement'],
        reverse=True
    )
    if by_improvement:
        lines.append("### Rules Ranked by Improvement")
        lines.append("")
        for rank, (rule, stats) in enumerate(by_improvement, 1):
            d_interp = "large" if abs(stats['cohens_d']) >= 0.8 else "medium" if abs(stats['cohens_d']) >= 0.5 else "small" if abs(stats['cohens_d']) >= 0.2 else "negligible"
            lines.append(f"{rank}. **{rule.upper()}**: {stats['improvement']:+.3f} (d={stats['cohens_d']:.2f}, {d_interp})")
        lines.append("")

    # Diagnostic: Why some rules have 0% catch rate
    lines.append("## Diagnostic: Rule Catch Rate Analysis")
    lines.append("")
    lines.append("Rules with low catch rates may indicate:")
    lines.append("1. **Model competence**: Claude correctly avoids the bad behavior (no tool calls attempted)")
    lines.append("2. **Threshold issues**: Rule thresholds may be too conservative")
    lines.append("3. **Rule logic gaps**: Rule doesn't detect the failure mode")
    lines.append("")
    lines.append("| Rule | Catch | Baseline | Explanation |")
    lines.append("|------|-------|----------|-------------|")
    for rule, stats in sorted(rule_stats.items()):
        if rule.lower() == 'valid':
            continue
        catch = stats.get('catch_rate')
        if catch is None:
            continue
        baseline = stats['baseline_mean']
        if catch == 0:
            # Check if baseline score is high (model already correct)
            if baseline >= 2.5:
                explanation = "Model handles correctly; no bad tool calls to catch"
            elif baseline >= 1.5:
                explanation = "Model partially correct; few tool calls attempted"
            else:
                explanation = "⚠ Rule may not be firing; investigate"
        elif catch < 0.3:
            explanation = "Partial detection; threshold may need tuning"
        elif catch < 0.6:
            explanation = "Moderate detection; room for improvement"
        else:
            explanation = "✓ Good detection rate"
        catch_str = f"{catch:.0%}"
        lines.append(f"| {rule.upper()} | {catch_str} | {baseline:.1f} | {explanation} |")
    lines.append("")

    # Threshold Sensitivity Analysis
    sensitivity = load_sensitivity_analysis()
    if sensitivity:
        lines.append("## Threshold Sensitivity Analysis")
        lines.append("")
        lines.append("How classifier performance changes with threshold variations:")
        lines.append("")

        for category in ["static_knowledge", "memory_reference"]:
            if category in sensitivity:
                cat_data = sensitivity[category]
                cat_name = category.replace("_", " ").title()
                lines.append(f"### {cat_name}")
                lines.append("")
                lines.append(f"Base threshold: **{cat_data['base_threshold']}**")
                lines.append("")
                lines.append("| Threshold | Δ | Accuracy | Precision | Recall | F1 |")
                lines.append("|-----------|---|----------|-----------|--------|-----|")

                for v in cat_data["variations"]:
                    delta_str = f"{v['delta']:+.2f}" if v['delta'] != 0 else "0.00"
                    lines.append(f"| {v['threshold']:.2f} | {delta_str} | {v['accuracy']:.1%} | "
                               f"{v['precision']:.1%} | {v['recall']:.1%} | {v['f1']:.2f} |")
                lines.append("")

        lines.append("> **Note**: Lower thresholds increase recall (catch more) but may increase false positives.")
        lines.append("> Higher thresholds increase precision but may miss valid cases.")
        lines.append("")

    return "\n".join(lines)


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

    # Power analysis
    power = compute_power_analysis(
        cohens_d=analysis.cohens_d,
        n=analysis.n_pairs,
    )
    print_power_analysis(power)

    # Add per-rule breakdown
    rule_stats = analyze_by_rule(results)
    print_rule_breakdown(rule_stats)

    # Generate and save markdown report
    report = generate_markdown_report(analysis, power, rule_stats, latest)

    # Extract timestamp from results filename (results_YYYYMMDD_HHMMSS.json)
    timestamp = latest.stem.replace("results_", "")
    report_path = results_dir / f"analysis_{timestamp}.md"

    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n{'='*60}")
    print(f"Markdown report saved to: {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
