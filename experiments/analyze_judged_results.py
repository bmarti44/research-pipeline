"""
Analyze Signal Detection Results Using Judge Scores

This script runs the full statistical analysis on judge-scored data.
Judge scores are the PRIMARY measure; regex results are kept for comparison.

Usage:
    python experiments/analyze_judged_results.py experiments/results/signal_detection_{timestamp}_judged.json
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional
import math


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 1.0)

    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom

    return (max(0, center - margin), min(1, center + margin))


def mcnemar_test(b: int, c: int) -> tuple[float, float]:
    """McNemar's test for paired binary data.

    Args:
        b: Count where A=1, B=0
        c: Count where A=0, B=1

    Returns:
        (chi_square, p_value)
    """
    from scipy.stats import chi2

    if b + c == 0:
        return (0.0, 1.0)

    chi_sq = (b - c) ** 2 / (b + c)
    p_value = 1 - chi2.cdf(chi_sq, df=1)

    return (chi_sq, p_value)


def sign_test(n_positive: int, n_total: int) -> float:
    """Two-sided sign test p-value."""
    from scipy.stats import binomtest

    if n_total == 0:
        return 1.0

    result = binomtest(n_positive, n_total, 0.5, alternative='two-sided')
    return result.pvalue


def load_judged_results(path: Path) -> dict:
    """Load judged results JSON."""
    with open(path) as f:
        return json.load(f)


def analyze_recall_by_condition(results: list[dict]) -> dict:
    """Compute recall for each condition using judge scores.

    Only includes EXPLICIT and IMPLICIT scenarios (those with ground truth).
    """
    # Filter to scenarios with ground truth
    with_truth = [r for r in results
                  if r.get("ambiguity") in ["EXPLICIT", "IMPLICIT"]
                  and r.get("expected_detection") is True]

    # NL recall
    nl_tp = sum(1 for r in with_truth if r.get("nl_judge_detected") is True)
    nl_n = len(with_truth)
    nl_recall = nl_tp / nl_n if nl_n > 0 else 0
    nl_ci = wilson_ci(nl_tp, nl_n)

    # Structured recall
    st_tp = sum(1 for r in with_truth if r.get("st_judge_detected") is True)
    st_n = len(with_truth)
    st_recall = st_tp / st_n if st_n > 0 else 0
    st_ci = wilson_ci(st_tp, st_n)

    return {
        "nl": {
            "recall": nl_recall,
            "true_positives": nl_tp,
            "n": nl_n,
            "ci_95": nl_ci,
        },
        "structured": {
            "recall": st_recall,
            "true_positives": st_tp,
            "n": st_n,
            "ci_95": st_ci,
        },
        "gap_pp": (nl_recall - st_recall) * 100,
    }


def analyze_by_ambiguity(results: list[dict]) -> dict:
    """Analyze detection rates by ambiguity level."""
    analysis = {}

    for ambiguity in ["EXPLICIT", "IMPLICIT", "BORDERLINE", "CONTROL"]:
        subset = [r for r in results if r.get("ambiguity") == ambiguity]
        if not subset:
            continue

        nl_detected = sum(1 for r in subset if r.get("nl_judge_detected") is True)
        st_detected = sum(1 for r in subset if r.get("st_judge_detected") is True)
        n = len(subset)

        entry = {
            "n": n,
            "nl_rate": nl_detected / n if n > 0 else 0,
            "st_rate": st_detected / n if n > 0 else 0,
            "gap_pp": ((nl_detected - st_detected) / n * 100) if n > 0 else 0,
        }

        # For EXPLICIT and IMPLICIT, these are recall rates
        if ambiguity in ["EXPLICIT", "IMPLICIT"]:
            entry["metric"] = "recall"
        else:
            entry["metric"] = "detection_rate"

        analysis[ambiguity] = entry

    return analysis


def scenario_level_sign_test(results: list[dict]) -> dict:
    """Run sign test at scenario level.

    For each scenario with ground truth, compute NL vs ST recall across trials.
    Count scenarios where NL > ST, ST > NL, tied.
    """
    # Group by scenario
    by_scenario = defaultdict(list)
    for r in results:
        if r.get("ambiguity") in ["EXPLICIT", "IMPLICIT"]:
            by_scenario[r.get("scenario_id")].append(r)

    nl_better = 0
    st_better = 0
    ties = 0

    for scenario_id, trials in by_scenario.items():
        nl_successes = sum(1 for t in trials if t.get("nl_judge_detected") is True)
        st_successes = sum(1 for t in trials if t.get("st_judge_detected") is True)

        if nl_successes > st_successes:
            nl_better += 1
        elif st_successes > nl_successes:
            st_better += 1
        else:
            ties += 1

    n_different = nl_better + st_better
    p_value = sign_test(nl_better, n_different) if n_different > 0 else 1.0

    return {
        "n_scenarios": len(by_scenario),
        "nl_better": nl_better,
        "st_better": st_better,
        "ties": ties,
        "sign_test_p": p_value,
    }


def trial_level_mcnemar(results: list[dict]) -> dict:
    """Run McNemar's test at trial level (secondary analysis)."""
    # Filter to trials with ground truth
    with_truth = [r for r in results
                  if r.get("ambiguity") in ["EXPLICIT", "IMPLICIT"]]

    # Build contingency: NL success vs ST success
    nl_only = sum(1 for r in with_truth
                  if r.get("nl_judge_detected") is True
                  and r.get("st_judge_detected") is not True)
    st_only = sum(1 for r in with_truth
                  if r.get("st_judge_detected") is True
                  and r.get("nl_judge_detected") is not True)
    both = sum(1 for r in with_truth
               if r.get("nl_judge_detected") is True
               and r.get("st_judge_detected") is True)
    neither = sum(1 for r in with_truth
                  if r.get("nl_judge_detected") is not True
                  and r.get("st_judge_detected") is not True)

    chi_sq, p_value = mcnemar_test(nl_only, st_only)

    return {
        "nl_only": nl_only,
        "st_only": st_only,
        "both": both,
        "neither": neither,
        "chi_sq": chi_sq,
        "p_value": p_value,
        "caveat": "Trials within a scenario are correlated; use scenario-level test for valid inference",
    }


def analyze_false_positives(results: list[dict]) -> dict:
    """Analyze false positive rate on CONTROL scenarios."""
    controls = [r for r in results if r.get("ambiguity") == "CONTROL"]

    if not controls:
        return {"n": 0}

    nl_fp = sum(1 for r in controls if r.get("nl_judge_detected") is True)
    st_fp = sum(1 for r in controls if r.get("st_judge_detected") is True)
    n = len(controls)

    return {
        "n": n,
        "nl_false_positives": nl_fp,
        "nl_fp_rate": nl_fp / n if n > 0 else 0,
        "st_false_positives": st_fp,
        "st_fp_rate": st_fp / n if n > 0 else 0,
    }


def analyze_hedging(results: list[dict]) -> dict:
    """Analyze hedging: structured responses that acknowledged signal without XML.

    This is direct evidence of format friction - the model recognized the signal
    but didn't commit to structured output.
    """
    # For hedging, we look at structured responses where:
    # - Judge says YES (signal was acknowledged)
    # - But no XML tag was present (regex/tool detection was NO)

    # Only look at scenarios with signals (not CONTROL)
    with_signal = [r for r in results if r.get("ambiguity") != "CONTROL"]

    hedged = []
    for r in with_signal:
        judge_yes = r.get("st_judge_detected") is True
        xml_present = r.get("st_regex_detected") is True or r.get("tool_called") is True

        if judge_yes and not xml_present:
            hedged.append({
                "scenario_id": r.get("scenario_id"),
                "ambiguity": r.get("ambiguity"),
                "response_preview": (r.get("st_response_text") or "")[:200],
            })

    # Group by ambiguity
    hedging_by_ambiguity = defaultdict(int)
    n_by_ambiguity = defaultdict(int)

    for r in with_signal:
        amb = r.get("ambiguity")
        n_by_ambiguity[amb] += 1
        if any(h["scenario_id"] == r.get("scenario_id") for h in hedged):
            hedging_by_ambiguity[amb] += 1

    return {
        "total_hedged": len(hedged),
        "total_with_signal": len(with_signal),
        "hedging_rate": len(hedged) / len(with_signal) if with_signal else 0,
        "by_ambiguity": {
            amb: {
                "hedged": hedging_by_ambiguity[amb],
                "n": n_by_ambiguity[amb],
                "rate": hedging_by_ambiguity[amb] / n_by_ambiguity[amb] if n_by_ambiguity[amb] > 0 else 0,
            }
            for amb in ["EXPLICIT", "IMPLICIT", "BORDERLINE"]
            if n_by_ambiguity[amb] > 0
        },
        "examples": hedged[:5],  # First 5 examples
    }


def compare_judge_vs_regex(results: list[dict]) -> dict:
    """Compare judge and regex scoring methods."""
    comparison = {
        "nl": {"agree": 0, "judge_only": 0, "regex_only": 0, "neither": 0},
        "st": {"agree": 0, "judge_only": 0, "regex_only": 0, "neither": 0},
    }

    for r in results:
        # NL comparison
        nl_judge = r.get("nl_judge_detected") is True
        nl_regex = r.get("nl_regex_detected") is True

        if nl_judge and nl_regex:
            comparison["nl"]["agree"] += 1
        elif nl_judge and not nl_regex:
            comparison["nl"]["judge_only"] += 1
        elif nl_regex and not nl_judge:
            comparison["nl"]["regex_only"] += 1
        else:
            comparison["nl"]["neither"] += 1

        # ST comparison
        st_judge = r.get("st_judge_detected") is True
        st_regex = r.get("st_regex_detected") is True

        if st_judge and st_regex:
            comparison["st"]["agree"] += 1
        elif st_judge and not st_regex:
            comparison["st"]["judge_only"] += 1
        elif st_regex and not st_judge:
            comparison["st"]["regex_only"] += 1
        else:
            comparison["st"]["neither"] += 1

    # Compute agreement rates
    for cond in ["nl", "st"]:
        total = sum(comparison[cond].values())
        agree = comparison[cond]["agree"] + comparison[cond]["neither"]
        comparison[cond]["total"] = total
        comparison[cond]["agreement_rate"] = agree / total if total > 0 else 0

    # Compute gap under each method
    with_truth = [r for r in results if r.get("ambiguity") in ["EXPLICIT", "IMPLICIT"]]
    n = len(with_truth)

    if n > 0:
        judge_nl = sum(1 for r in with_truth if r.get("nl_judge_detected") is True) / n
        judge_st = sum(1 for r in with_truth if r.get("st_judge_detected") is True) / n
        regex_nl = sum(1 for r in with_truth if r.get("nl_regex_detected") is True) / n
        regex_st = sum(1 for r in with_truth if r.get("st_regex_detected") is True) / n

        comparison["gap_comparison"] = {
            "judge_gap_pp": (judge_nl - judge_st) * 100,
            "regex_gap_pp": (regex_nl - regex_st) * 100,
            "difference_pp": ((judge_nl - judge_st) - (regex_nl - regex_st)) * 100,
        }

    return comparison


def print_report(analysis: dict) -> None:
    """Print formatted analysis report."""

    print("\n" + "=" * 70)
    print("SIGNAL DETECTION ANALYSIS - JUDGE SCORES (PRIMARY)")
    print("=" * 70)

    # Overall recall
    recall = analysis.get("recall", {})
    print("\n1. OVERALL RECALL (EXPLICIT + IMPLICIT scenarios)")
    print("-" * 50)
    nl = recall.get("nl", {})
    st = recall.get("structured", {})
    print(f"   NL:         {nl.get('recall', 0):.1%}  (n={nl.get('n', 0)}, 95% CI: [{nl.get('ci_95', [0,0])[0]:.1%}, {nl.get('ci_95', [0,0])[1]:.1%}])")
    print(f"   Structured: {st.get('recall', 0):.1%}  (n={st.get('n', 0)}, 95% CI: [{st.get('ci_95', [0,0])[0]:.1%}, {st.get('ci_95', [0,0])[1]:.1%}])")
    print(f"   Gap:        {recall.get('gap_pp', 0):+.1f}pp")

    # By ambiguity
    by_amb = analysis.get("by_ambiguity", {})
    print("\n2. AMBIGUITY INTERACTION")
    print("-" * 50)
    for amb in ["EXPLICIT", "IMPLICIT", "BORDERLINE", "CONTROL"]:
        stats = by_amb.get(amb)
        if stats:
            metric = stats.get("metric", "rate")
            print(f"   {amb:12} NL={stats['nl_rate']:.1%}  ST={stats['st_rate']:.1%}  gap={stats['gap_pp']:+.1f}pp  (n={stats['n']}, {metric})")

    # Sign test
    sign = analysis.get("sign_test", {})
    print("\n3. SCENARIO-LEVEL SIGN TEST (PRIMARY)")
    print("-" * 50)
    print(f"   Scenarios: {sign.get('n_scenarios', 0)}")
    print(f"   NL better: {sign.get('nl_better', 0)}")
    print(f"   ST better: {sign.get('st_better', 0)}")
    print(f"   Ties:      {sign.get('ties', 0)}")
    print(f"   p-value:   {sign.get('sign_test_p', 1):.4f}")

    # McNemar (secondary)
    mcn = analysis.get("mcnemar", {})
    print("\n4. TRIAL-LEVEL McNEMAR (SECONDARY)")
    print("-" * 50)
    print(f"   NL-only wins:  {mcn.get('nl_only', 0)}")
    print(f"   ST-only wins:  {mcn.get('st_only', 0)}")
    print(f"   Both succeed:  {mcn.get('both', 0)}")
    print(f"   Both fail:     {mcn.get('neither', 0)}")
    print(f"   χ² = {mcn.get('chi_sq', 0):.2f}, p = {mcn.get('p_value', 1):.4f}")
    print(f"   ⚠️  {mcn.get('caveat', '')}")

    # False positives
    fp = analysis.get("false_positives", {})
    print("\n5. FALSE POSITIVE RATE (CONTROL scenarios)")
    print("-" * 50)
    print(f"   NL:         {fp.get('nl_fp_rate', 0):.1%}  ({fp.get('nl_false_positives', 0)}/{fp.get('n', 0)})")
    print(f"   Structured: {fp.get('st_fp_rate', 0):.1%}  ({fp.get('st_false_positives', 0)}/{fp.get('n', 0)})")
    print(f"   Target:     <5%")

    # Hedging
    hedge = analysis.get("hedging", {})
    print("\n6. HEDGING ANALYSIS (Format Friction Evidence)")
    print("-" * 50)
    print(f"   Structured responses that acknowledged signal WITHOUT XML:")
    print(f"   Total:  {hedge.get('total_hedged', 0)}/{hedge.get('total_with_signal', 0)} ({hedge.get('hedging_rate', 0):.1%})")
    for amb, stats in hedge.get("by_ambiguity", {}).items():
        print(f"   {amb:12} {stats['hedged']}/{stats['n']} ({stats['rate']:.1%})")

    # Measurement comparison
    comp = analysis.get("measurement_comparison", {})
    print("\n7. MEASUREMENT METHOD COMPARISON (Judge vs Regex)")
    print("-" * 50)
    gap_comp = comp.get("gap_comparison", {})
    print(f"   NL-ST gap (judge): {gap_comp.get('judge_gap_pp', 0):+.1f}pp")
    print(f"   NL-ST gap (regex): {gap_comp.get('regex_gap_pp', 0):+.1f}pp")
    print(f"   Difference:        {gap_comp.get('difference_pp', 0):+.1f}pp")

    print(f"\n   NL condition agreement (judge vs regex):")
    nl_comp = comp.get("nl", {})
    print(f"     Both YES:    {nl_comp.get('agree', 0)}")
    print(f"     Judge only:  {nl_comp.get('judge_only', 0)}  (regex false negatives)")
    print(f"     Regex only:  {nl_comp.get('regex_only', 0)}  (regex false positives)")
    print(f"     Both NO:     {nl_comp.get('neither', 0)}")
    print(f"     Agreement:   {nl_comp.get('agreement_rate', 0):.1%}")

    print(f"\n   Structured condition agreement (judge vs regex):")
    st_comp = comp.get("st", {})
    print(f"     Both YES:    {st_comp.get('agree', 0)}")
    print(f"     Judge only:  {st_comp.get('judge_only', 0)}")
    print(f"     Regex only:  {st_comp.get('regex_only', 0)}")
    print(f"     Both NO:     {st_comp.get('neither', 0)}")
    print(f"     Agreement:   {st_comp.get('agreement_rate', 0):.1%}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze signal detection results using judge scores"
    )
    parser.add_argument(
        "judged_file",
        type=Path,
        help="Path to judged results JSON file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for JSON results (optional)"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading judged results from {args.judged_file}")
    data = load_judged_results(args.judged_file)
    results = data.get("results", [])
    print(f"Found {len(results)} trial records")

    # Run analyses
    analysis = {
        "metadata": {
            "source_file": str(args.judged_file),
            "analysis_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "n_trials": len(results),
        },
        "recall": analyze_recall_by_condition(results),
        "by_ambiguity": analyze_by_ambiguity(results),
        "sign_test": scenario_level_sign_test(results),
        "mcnemar": trial_level_mcnemar(results),
        "false_positives": analyze_false_positives(results),
        "hedging": analyze_hedging(results),
        "measurement_comparison": compare_judge_vs_regex(results),
    }

    # Print report
    print_report(analysis)

    # Save JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nSaved analysis to {args.output}")


if __name__ == "__main__":
    main()
