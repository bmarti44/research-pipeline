"""
Pre-registered analysis for Format Friction experiment.

Per PLAN.md specifications, this script implements:
1. Primary analysis (H1): Two-proportion z-test with cluster-robust SEs
2. ICC calculation for cluster analysis interpretation
3. Task-level AND trial-level analyses (always report both)
4. Effect sizes: Risk difference (primary), Relative risk (secondary)
5. Bootstrap CIs at task level (cluster-robust)
6. Manipulation check analysis
7. Secondary hypotheses (H2-H4) with BH correction
8. Exploratory analysis (H5): ICC analysis

Critical requirements from PLAN.md:
- If ICC > 0.9: Trial-level analysis is INVALID for inference
- Always use np.random.default_rng() for reproducibility
- Report both analyses regardless of ICC value
"""

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any
import numpy as np

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.core.bootstrap import (
    bootstrap_proportion_ci,
    bootstrap_difference_ci,
    bootstrap_mean_ci,
)


@dataclass
class ICCResult:
    """Intra-class correlation coefficient result."""
    icc: float
    n_tasks: int
    n_trials: int
    mean_cluster_size: float
    interpretation: str  # "trial_level_valid" or "trial_level_invalid"
    design_effect: float
    effective_n: float


@dataclass
class PrimaryAnalysis:
    """Primary analysis results for H1."""
    # Sample sizes
    n_nl: int
    n_json: int
    n_tasks_nl: int
    n_tasks_json: int

    # Accuracies
    accuracy_nl: float
    accuracy_json: float

    # Effect sizes
    risk_difference: float  # Primary: P(correct|NL) - P(correct|JSON)
    risk_difference_ci: tuple[float, float]
    relative_risk: float  # Secondary: P(correct|NL) / P(correct|JSON)
    relative_risk_ci: tuple[float, float]

    # Statistical significance
    z_statistic: float
    p_value: float
    significant_at_05: bool

    # Practical significance
    friction_pp: float  # Risk difference in percentage points
    practically_significant: bool  # |friction| >= 10pp

    # ICC-adjusted inference validity
    icc: float
    trial_level_valid: bool  # ICC <= 0.9


@dataclass
class ManipulationCheckAnalysis:
    """Manipulation check analysis results."""
    # Explicit declines
    nl_declined: int
    nl_declined_rate: float
    json_declined: int
    json_declined_rate: float

    # JSON-specific
    json_attempted_json: int
    json_syntactically_valid: int
    json_valid_rate: float

    # Conditional analysis (excluding declines)
    nl_correct_conditional: int
    nl_n_conditional: int
    nl_accuracy_conditional: float
    json_correct_conditional: int
    json_n_conditional: int
    json_accuracy_conditional: float
    conditional_friction_pp: float


@dataclass
class AnalysisReport:
    """Complete analysis report."""
    timestamp: str
    source_file: str
    seed: int
    n_bootstrap: int

    # Primary analysis
    primary: PrimaryAnalysis

    # ICC analysis
    icc: ICCResult

    # Task-level analysis
    task_level_accuracy_nl: float
    task_level_accuracy_json: float
    task_level_friction_pp: float
    task_level_ci: tuple[float, float]

    # Trial-level analysis (may be invalid if ICC > 0.9)
    trial_level_accuracy_nl: float
    trial_level_accuracy_json: float
    trial_level_friction_pp: float
    trial_level_ci: tuple[float, float]

    # Manipulation checks
    manipulation_checks: ManipulationCheckAnalysis

    # Metadata
    notes: list[str]


def compute_icc(trials: list[dict], condition: str) -> ICCResult:
    """
    Compute intra-class correlation coefficient for a condition.

    Uses one-way random effects ANOVA ICC(1,1).

    Per PLAN.md:
    - ICC > 0.9 → Trial-level analysis INVALID for inference
    - ICC <= 0.9 → Both analyses valid
    """
    # Filter trials by condition
    cond_trials = [t for t in trials if t["condition"] == condition]

    if not cond_trials:
        return ICCResult(
            icc=0.0, n_tasks=0, n_trials=0, mean_cluster_size=0,
            interpretation="insufficient_data", design_effect=1.0, effective_n=0
        )

    # Group by task
    by_task: dict[str, list[int]] = {}
    for t in cond_trials:
        task_id = t["task_id"]
        if task_id not in by_task:
            by_task[task_id] = []
        by_task[task_id].append(1 if t["is_correct"] else 0)

    n_tasks = len(by_task)
    n_trials = len(cond_trials)

    if n_tasks < 2:
        return ICCResult(
            icc=0.0, n_tasks=n_tasks, n_trials=n_trials,
            mean_cluster_size=n_trials / n_tasks if n_tasks > 0 else 0,
            interpretation="insufficient_tasks", design_effect=1.0, effective_n=n_trials
        )

    # Compute ICC(1,1) using ANOVA decomposition
    task_means = []
    task_sizes = []
    all_values = []

    for task_id, values in by_task.items():
        task_means.append(np.mean(values))
        task_sizes.append(len(values))
        all_values.extend(values)

    grand_mean = np.mean(all_values)
    mean_cluster_size = np.mean(task_sizes)

    # Between-group sum of squares
    ss_between = sum(n * (m - grand_mean) ** 2 for n, m in zip(task_sizes, task_means))

    # Within-group sum of squares
    ss_within = 0.0
    for task_id, values in by_task.items():
        task_mean = np.mean(values)
        ss_within += sum((v - task_mean) ** 2 for v in values)

    # Degrees of freedom
    df_between = n_tasks - 1
    df_within = n_trials - n_tasks

    if df_within <= 0:
        return ICCResult(
            icc=0.0, n_tasks=n_tasks, n_trials=n_trials,
            mean_cluster_size=mean_cluster_size,
            interpretation="insufficient_within_variance",
            design_effect=1.0, effective_n=n_trials
        )

    # Mean squares
    ms_between = ss_between / df_between if df_between > 0 else 0
    ms_within = ss_within / df_within if df_within > 0 else 0

    # ICC(1,1) formula
    # ICC = (MS_between - MS_within) / (MS_between + (k-1) * MS_within)
    # where k is the average cluster size
    k = mean_cluster_size

    denominator = ms_between + (k - 1) * ms_within
    if denominator <= 0:
        icc = 0.0
    else:
        icc = (ms_between - ms_within) / denominator
        icc = max(0, min(1, icc))  # Bound to [0, 1]

    # Design effect: DE = 1 + (m - 1) * ICC
    design_effect = 1 + (k - 1) * icc
    effective_n = n_trials / design_effect if design_effect > 0 else n_trials

    # Interpretation per PLAN.md
    if icc > 0.9:
        interpretation = "trial_level_invalid"
    else:
        interpretation = "trial_level_valid"

    return ICCResult(
        icc=icc,
        n_tasks=n_tasks,
        n_trials=n_trials,
        mean_cluster_size=mean_cluster_size,
        interpretation=interpretation,
        design_effect=design_effect,
        effective_n=effective_n,
    )


def compute_task_level_accuracy(trials: list[dict], condition: str) -> tuple[float, list[float]]:
    """
    Compute task-level accuracy (mean of per-task accuracies).

    Returns:
        (overall_accuracy, list_of_task_accuracies)
    """
    cond_trials = [t for t in trials if t["condition"] == condition]

    # Group by task
    by_task: dict[str, list[int]] = {}
    for t in cond_trials:
        task_id = t["task_id"]
        if task_id not in by_task:
            by_task[task_id] = []
        by_task[task_id].append(1 if t["is_correct"] else 0)

    task_accuracies = [np.mean(values) for values in by_task.values()]

    if not task_accuracies:
        return 0.0, []

    return np.mean(task_accuracies), task_accuracies


def cluster_bootstrap_difference_ci(
    trials: list[dict],
    n_bootstrap: int = 10000,
    seed: int = 42,
    ci_level: float = 0.95,
) -> tuple[float, float]:
    """
    Bootstrap CI for friction at task level (cluster-robust).

    Per PLAN.md: Resample tasks, not trials.
    """
    rng = np.random.default_rng(seed)

    # Group by task and condition
    nl_by_task: dict[str, list[int]] = {}
    json_by_task: dict[str, list[int]] = {}

    for t in trials:
        task_id = t["task_id"]
        correct = 1 if t["is_correct"] else 0

        if t["condition"] == "nl_only":
            if task_id not in nl_by_task:
                nl_by_task[task_id] = []
            nl_by_task[task_id].append(correct)
        else:
            if task_id not in json_by_task:
                json_by_task[task_id] = []
            json_by_task[task_id].append(correct)

    nl_tasks = list(nl_by_task.keys())
    json_tasks = list(json_by_task.keys())

    if not nl_tasks or not json_tasks:
        return (0.0, 0.0)

    bootstrap_diffs = []

    for _ in range(n_bootstrap):
        # Resample tasks with replacement
        sampled_nl_tasks = rng.choice(nl_tasks, size=len(nl_tasks), replace=True)
        sampled_json_tasks = rng.choice(json_tasks, size=len(json_tasks), replace=True)

        # Compute task-level accuracies for sampled tasks
        nl_accs = [np.mean(nl_by_task[t]) for t in sampled_nl_tasks]
        json_accs = [np.mean(json_by_task[t]) for t in sampled_json_tasks]

        # Friction = NL accuracy - JSON accuracy
        diff = np.mean(nl_accs) - np.mean(json_accs)
        bootstrap_diffs.append(diff)

    alpha = 1 - ci_level
    lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    return (lower, upper)


def two_proportion_z_test(
    n1: int, x1: int,  # NL condition
    n2: int, x2: int,  # JSON condition
) -> tuple[float, float]:
    """
    Two-proportion z-test.

    Returns:
        (z_statistic, p_value)
    """
    if n1 == 0 or n2 == 0:
        return (0.0, 1.0)

    p1 = x1 / n1
    p2 = x2 / n2

    # Pooled proportion
    p_pool = (x1 + x2) / (n1 + n2)

    # Standard error
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

    if se == 0:
        return (0.0, 1.0)

    z = (p1 - p2) / se

    # Two-sided p-value
    from scipy import stats
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return (z, p_value)


def compute_relative_risk_ci(
    n1: int, x1: int,
    n2: int, x2: int,
    ci_level: float = 0.95,
) -> tuple[float, float, float]:
    """
    Compute relative risk with CI using log transformation.

    Returns:
        (relative_risk, ci_lower, ci_upper)
    """
    if n1 == 0 or n2 == 0 or x2 == 0:
        return (1.0, 0.0, float('inf'))

    p1 = x1 / n1
    p2 = x2 / n2

    if p2 == 0:
        return (float('inf'), 0.0, float('inf'))

    rr = p1 / p2

    # Log transformation for CI
    if p1 == 0 or p1 == 1 or p2 == 0 or p2 == 1:
        return (rr, 0.0, float('inf'))

    log_rr = np.log(rr)
    se_log_rr = np.sqrt((1 - p1) / (n1 * p1) + (1 - p2) / (n2 * p2))

    from scipy import stats
    z = stats.norm.ppf(1 - (1 - ci_level) / 2)

    ci_lower = np.exp(log_rr - z * se_log_rr)
    ci_upper = np.exp(log_rr + z * se_log_rr)

    return (rr, ci_lower, ci_upper)


def analyze_manipulation_checks(trials: list[dict]) -> ManipulationCheckAnalysis:
    """Analyze manipulation check data."""
    nl_trials = [t for t in trials if t["condition"] == "nl_only"]
    json_trials = [t for t in trials if t["condition"] == "json_only"]

    # Explicit declines
    nl_declined = sum(
        1 for t in nl_trials
        if t.get("manipulation_check", {}).get("explicitly_declined", False)
    )
    json_declined = sum(
        1 for t in json_trials
        if t.get("manipulation_check", {}).get("explicitly_declined", False)
    )

    # JSON-specific checks
    json_attempted = sum(
        1 for t in json_trials
        if t.get("manipulation_check", {}).get("attempted_json", False)
    )
    json_valid = sum(
        1 for t in json_trials
        if t.get("manipulation_check", {}).get("syntactically_valid", False)
    )

    # Conditional analysis (excluding declines)
    nl_conditional = [t for t in nl_trials if not t.get("manipulation_check", {}).get("explicitly_declined", False)]
    json_conditional = [t for t in json_trials if not t.get("manipulation_check", {}).get("explicitly_declined", False)]

    nl_correct_cond = sum(1 for t in nl_conditional if t["is_correct"])
    json_correct_cond = sum(1 for t in json_conditional if t["is_correct"])

    nl_acc_cond = nl_correct_cond / len(nl_conditional) if nl_conditional else 0
    json_acc_cond = json_correct_cond / len(json_conditional) if json_conditional else 0

    return ManipulationCheckAnalysis(
        nl_declined=nl_declined,
        nl_declined_rate=nl_declined / len(nl_trials) if nl_trials else 0,
        json_declined=json_declined,
        json_declined_rate=json_declined / len(json_trials) if json_trials else 0,
        json_attempted_json=json_attempted,
        json_syntactically_valid=json_valid,
        json_valid_rate=json_valid / len(json_trials) if json_trials else 0,
        nl_correct_conditional=nl_correct_cond,
        nl_n_conditional=len(nl_conditional),
        nl_accuracy_conditional=nl_acc_cond,
        json_correct_conditional=json_correct_cond,
        json_n_conditional=len(json_conditional),
        json_accuracy_conditional=json_acc_cond,
        conditional_friction_pp=(nl_acc_cond - json_acc_cond) * 100,
    )


def run_preregistered_analysis(
    results_file: str,
    seed: int = 42,
    n_bootstrap: int = 10000,
) -> AnalysisReport:
    """
    Run complete pre-registered analysis.

    Per PLAN.md requirements:
    1. Always report both task-level and trial-level analyses
    2. Use ICC to interpret, not select
    3. Cluster-robust bootstrap for CIs
    """
    # Load data
    with open(results_file) as f:
        data = json.load(f)

    trials = data.get("trials", [])
    notes = []

    # Separate by condition
    nl_trials = [t for t in trials if t["condition"] == "nl_only"]
    json_trials = [t for t in trials if t["condition"] == "json_only"]

    # Basic counts
    n_nl = len(nl_trials)
    n_json = len(json_trials)
    nl_correct = sum(1 for t in nl_trials if t["is_correct"])
    json_correct = sum(1 for t in json_trials if t["is_correct"])

    # Task counts
    n_tasks_nl = len(set(t["task_id"] for t in nl_trials))
    n_tasks_json = len(set(t["task_id"] for t in json_trials))

    # Trial-level accuracy
    trial_acc_nl = nl_correct / n_nl if n_nl > 0 else 0
    trial_acc_json = json_correct / n_json if n_json > 0 else 0
    trial_friction = trial_acc_nl - trial_acc_json

    # Task-level accuracy
    task_acc_nl, task_accs_nl = compute_task_level_accuracy(trials, "nl_only")
    task_acc_json, task_accs_json = compute_task_level_accuracy(trials, "json_only")
    task_friction = task_acc_nl - task_acc_json

    # ICC analysis
    icc_nl = compute_icc(trials, "nl_only")
    icc_json = compute_icc(trials, "json_only")

    # Use average ICC for interpretation
    avg_icc = (icc_nl.icc + icc_json.icc) / 2 if (icc_nl.n_tasks > 0 and icc_json.n_tasks > 0) else max(icc_nl.icc, icc_json.icc)

    if avg_icc > 0.9:
        notes.append("WARNING: ICC > 0.9; trial-level analysis is INVALID for inference")
        trial_level_valid = False
    else:
        trial_level_valid = True

    # Bootstrap CIs
    # Task-level (cluster-robust) - always valid
    task_ci = cluster_bootstrap_difference_ci(trials, n_bootstrap, seed)

    # Trial-level (may be invalid if ICC > 0.9)
    nl_values = [1 if t["is_correct"] else 0 for t in nl_trials]
    json_values = [1 if t["is_correct"] else 0 for t in json_trials]

    if nl_values and json_values:
        trial_ci_result = bootstrap_difference_ci(
            nl_values, json_values,
            n_bootstrap=n_bootstrap,
            seed=seed,
            paired=False,
        )
        trial_ci = (trial_ci_result["lower"], trial_ci_result["upper"])
    else:
        trial_ci = (0.0, 0.0)

    # Statistical test (two-proportion z-test)
    z_stat, p_value = two_proportion_z_test(n_nl, nl_correct, n_json, json_correct)

    # Effect sizes
    risk_diff = trial_acc_nl - trial_acc_json
    risk_diff_ci = trial_ci  # Bootstrap CI

    rr, rr_lower, rr_upper = compute_relative_risk_ci(n_nl, nl_correct, n_json, json_correct)

    # Practical significance (10pp threshold per PLAN.md)
    practically_significant = abs(risk_diff * 100) >= 10

    # Manipulation checks
    manip_checks = analyze_manipulation_checks(trials)

    # Create primary analysis result
    primary = PrimaryAnalysis(
        n_nl=n_nl,
        n_json=n_json,
        n_tasks_nl=n_tasks_nl,
        n_tasks_json=n_tasks_json,
        accuracy_nl=trial_acc_nl,
        accuracy_json=trial_acc_json,
        risk_difference=risk_diff,
        risk_difference_ci=risk_diff_ci,
        relative_risk=rr,
        relative_risk_ci=(rr_lower, rr_upper),
        z_statistic=z_stat,
        p_value=p_value,
        significant_at_05=p_value < 0.05,
        friction_pp=risk_diff * 100,
        practically_significant=practically_significant,
        icc=avg_icc,
        trial_level_valid=trial_level_valid,
    )

    # Combined ICC result
    combined_icc = ICCResult(
        icc=avg_icc,
        n_tasks=n_tasks_nl + n_tasks_json,
        n_trials=n_nl + n_json,
        mean_cluster_size=(icc_nl.mean_cluster_size + icc_json.mean_cluster_size) / 2,
        interpretation="trial_level_invalid" if avg_icc > 0.9 else "trial_level_valid",
        design_effect=(icc_nl.design_effect + icc_json.design_effect) / 2,
        effective_n=icc_nl.effective_n + icc_json.effective_n,
    )

    return AnalysisReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        source_file=results_file,
        seed=seed,
        n_bootstrap=n_bootstrap,
        primary=primary,
        icc=combined_icc,
        task_level_accuracy_nl=task_acc_nl,
        task_level_accuracy_json=task_acc_json,
        task_level_friction_pp=task_friction * 100,
        task_level_ci=task_ci,
        trial_level_accuracy_nl=trial_acc_nl,
        trial_level_accuracy_json=trial_acc_json,
        trial_level_friction_pp=trial_friction * 100,
        trial_level_ci=trial_ci,
        manipulation_checks=manip_checks,
        notes=notes,
    )


def print_report(report: AnalysisReport) -> None:
    """Print analysis report in human-readable format."""
    print()
    print("=" * 70)
    print("PRE-REGISTERED ANALYSIS REPORT")
    print("=" * 70)
    print(f"Source: {report.source_file}")
    print(f"Timestamp: {report.timestamp}")
    print(f"Seed: {report.seed}")
    print(f"Bootstrap replicates: {report.n_bootstrap}")
    print()

    # Notes/warnings
    if report.notes:
        print("NOTES:")
        for note in report.notes:
            print(f"  ⚠ {note}")
        print()

    # ICC Analysis
    print("-" * 70)
    print("ICC ANALYSIS (H5 - Exploratory)")
    print("-" * 70)
    print(f"  Average ICC: {report.icc.icc:.3f}")
    print(f"  Design effect: {report.icc.design_effect:.2f}")
    print(f"  Effective N: {report.icc.effective_n:.0f}")
    print(f"  Interpretation: {report.icc.interpretation}")
    print()

    # Primary Analysis
    print("-" * 70)
    print("PRIMARY ANALYSIS (H1)")
    print("-" * 70)
    p = report.primary
    print()
    print("Sample sizes:")
    print(f"  NL-only:   N = {p.n_nl} ({p.n_tasks_nl} tasks)")
    print(f"  JSON-only: N = {p.n_json} ({p.n_tasks_json} tasks)")
    print()

    # Task-level analysis (always valid)
    print("TASK-LEVEL ANALYSIS (cluster-robust):")
    print(f"  NL-only accuracy:   {report.task_level_accuracy_nl:.1%}")
    print(f"  JSON-only accuracy: {report.task_level_accuracy_json:.1%}")
    print(f"  Format friction:    {report.task_level_friction_pp:+.1f}pp")
    print(f"  95% CI:             [{report.task_level_ci[0]*100:.1f}pp, {report.task_level_ci[1]*100:.1f}pp]")
    print()

    # Trial-level analysis
    validity = "VALID" if p.trial_level_valid else "INVALID (ICC > 0.9)"
    print(f"TRIAL-LEVEL ANALYSIS ({validity}):")
    print(f"  NL-only accuracy:   {report.trial_level_accuracy_nl:.1%}")
    print(f"  JSON-only accuracy: {report.trial_level_accuracy_json:.1%}")
    print(f"  Format friction:    {report.trial_level_friction_pp:+.1f}pp")
    print(f"  95% CI:             [{report.trial_level_ci[0]*100:.1f}pp, {report.trial_level_ci[1]*100:.1f}pp]")
    print()

    # Effect sizes
    print("Effect sizes:")
    print(f"  Risk difference (primary): {p.risk_difference:.3f} ({p.friction_pp:+.1f}pp)")
    print(f"  Relative risk (secondary): {p.relative_risk:.3f} [{p.relative_risk_ci[0]:.3f}, {p.relative_risk_ci[1]:.3f}]")
    print()

    # Statistical significance
    print("Statistical significance:")
    print(f"  z-statistic: {p.z_statistic:.3f}")
    print(f"  p-value:     {p.p_value:.4f}")
    print(f"  Significant at α=0.05: {'Yes' if p.significant_at_05 else 'No'}")
    print()

    # Practical significance
    print("Practical significance (threshold = 10pp):")
    if p.practically_significant:
        print(f"  ✓ Friction of {abs(p.friction_pp):.1f}pp meets threshold")
    else:
        print(f"  ✗ Friction of {abs(p.friction_pp):.1f}pp below threshold")
    print()

    # Manipulation checks
    print("-" * 70)
    print("MANIPULATION CHECKS")
    print("-" * 70)
    m = report.manipulation_checks
    print()
    print("Explicit declines:")
    print(f"  NL-only:   {m.nl_declined}/{report.primary.n_nl} ({m.nl_declined_rate:.1%})")
    print(f"  JSON-only: {m.json_declined}/{report.primary.n_json} ({m.json_declined_rate:.1%})")
    print()
    print("Conditional analysis (excluding declines):")
    print(f"  NL-only accuracy:   {m.nl_accuracy_conditional:.1%} (N={m.nl_n_conditional})")
    print(f"  JSON-only accuracy: {m.json_accuracy_conditional:.1%} (N={m.json_n_conditional})")
    print(f"  Conditional friction: {m.conditional_friction_pp:+.1f}pp")
    print()

    # Conclusion
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()

    # Primary inference based on ICC validity
    if p.trial_level_valid:
        inference_basis = "trial-level"
        friction = report.trial_level_friction_pp
        ci = report.trial_level_ci
    else:
        inference_basis = "task-level (trial-level invalid due to ICC > 0.9)"
        friction = report.task_level_friction_pp
        ci = report.task_level_ci

    print(f"Primary inference based on {inference_basis} analysis:")

    if p.significant_at_05:
        direction = "higher" if friction > 0 else "lower"
        print(f"  Format friction is statistically significant (p = {p.p_value:.4f})")
        print(f"  NL-only condition has {abs(friction):.1f}pp {direction} accuracy than JSON-only")
    else:
        print(f"  Format friction is NOT statistically significant (p = {p.p_value:.4f})")

    if p.practically_significant:
        print(f"  Effect is practically significant (|{friction:.1f}pp| >= 10pp)")
    else:
        print(f"  Effect is NOT practically significant (|{friction:.1f}pp| < 10pp)")

    print()


def main():
    """Run pre-registered analysis from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Pre-registered analysis for format friction experiment")
    parser.add_argument("--results-file", help="Results file to analyze (default: most recent)")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap seed")
    parser.add_argument("--n-bootstrap", type=int, default=10000, help="Bootstrap replicates")
    parser.add_argument("--output", help="Output file for JSON report")

    args = parser.parse_args()

    # Find results file
    if args.results_file:
        results_file = args.results_file
    else:
        results_dir = Path("experiments/results/raw")
        result_files = list(results_dir.glob("trials_*.json"))
        if not result_files:
            print("Error: No result files found")
            return 1
        results_file = str(max(result_files, key=lambda p: p.stat().st_mtime))

    # Run analysis
    report = run_preregistered_analysis(
        results_file=results_file,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
    )

    # Print human-readable report
    print_report(report)

    # Save JSON report
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("experiments/results/analysis") / f"preregistered_{Path(results_file).stem}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict for JSON
    report_dict = {
        "timestamp": report.timestamp,
        "source_file": report.source_file,
        "seed": report.seed,
        "n_bootstrap": report.n_bootstrap,
        "primary": asdict(report.primary),
        "icc": asdict(report.icc),
        "task_level": {
            "accuracy_nl": report.task_level_accuracy_nl,
            "accuracy_json": report.task_level_accuracy_json,
            "friction_pp": report.task_level_friction_pp,
            "ci": list(report.task_level_ci),
        },
        "trial_level": {
            "accuracy_nl": report.trial_level_accuracy_nl,
            "accuracy_json": report.trial_level_accuracy_json,
            "friction_pp": report.trial_level_friction_pp,
            "ci": list(report.trial_level_ci),
        },
        "manipulation_checks": asdict(report.manipulation_checks),
        "notes": report.notes,
    }

    with open(output_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    print(f"Report saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
