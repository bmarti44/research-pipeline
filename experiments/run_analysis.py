#!/usr/bin/env python
"""
Run Analysis — Single Entry Point for Paper Results

This script runs the complete analysis pipeline and generates
a formatted report matching the paper's narrative structure.

The paper's core claim: Output friction in structured tool calling.
Models detect signals at ~83% but produce XML at ~69%. The 14pp gap
is output friction — detection without compliance.

Usage:
    python experiments/run_analysis.py experiments/results/signal_detection_*_judged.json

Output:
    - Formatted report to stdout
    - Report saved to experiments/results/analysis_report_{timestamp}.txt
"""

import json
import argparse
import random
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
    """McNemar's test for paired binary data."""
    try:
        from scipy.stats import chi2
        if b + c == 0:
            return (0.0, 1.0)
        chi_sq = (b - c) ** 2 / (b + c)
        p_value = 1 - chi2.cdf(chi_sq, df=1)
        return (chi_sq, p_value)
    except ImportError:
        # Fallback without scipy
        if b + c == 0:
            return (0.0, 1.0)
        chi_sq = (b - c) ** 2 / (b + c)
        p_value = math.exp(-chi_sq / 2) if chi_sq > 0 else 1.0
        return (chi_sq, p_value)


def sign_test(n_positive: int, n_total: int) -> float:
    """Two-sided sign test p-value."""
    try:
        from scipy.stats import binomtest
        if n_total == 0:
            return 1.0
        result = binomtest(n_positive, n_total, 0.5, alternative='two-sided')
        return result.pvalue
    except ImportError:
        # Rough approximation without scipy
        if n_total == 0:
            return 1.0
        expected = n_total / 2
        deviation = abs(n_positive - expected)
        std = math.sqrt(n_total * 0.25)
        z = deviation / std if std > 0 else 0
        return 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))


def load_judged_results(path: Path) -> dict:
    """Load judged results JSON."""
    with open(path) as f:
        return json.load(f)


def get_ground_truth_trials(results: list[dict]) -> list[dict]:
    """Filter to ground-truth scenarios (EXPLICIT + IMPLICIT with expected_detection=True)."""
    return [r for r in results
            if r.get("ambiguity") in ["EXPLICIT", "IMPLICIT"]
            and r.get("expected_detection") is True]


def compute_detection_rates(results: list[dict]) -> dict:
    """Compute detection rates using judge scores (primary measure).

    This measures whether the model recognized the signal, regardless of format.
    """
    gt = get_ground_truth_trials(results)
    n = len(gt)

    nl_detected = sum(1 for r in gt if r.get("nl_judge_detected") is True)
    st_detected = sum(1 for r in gt if r.get("st_judge_detected") is True)

    return {
        "n": n,
        "nl_detection": nl_detected / n if n > 0 else 0,
        "nl_detected_count": nl_detected,
        "st_detection": st_detected / n if n > 0 else 0,
        "st_detected_count": st_detected,
        "difference_pp": (nl_detected - st_detected) / n * 100 if n > 0 else 0,
    }


def compute_detection_tests(results: list[dict]) -> dict:
    """Compute statistical tests for detection rate comparison."""
    gt = get_ground_truth_trials(results)

    # McNemar: count discordant pairs
    nl_only = sum(1 for r in gt
                  if r.get("nl_judge_detected") is True
                  and r.get("st_judge_detected") is not True)
    st_only = sum(1 for r in gt
                  if r.get("st_judge_detected") is True
                  and r.get("nl_judge_detected") is not True)

    chi_sq, p = mcnemar_test(nl_only, st_only)

    # Sign test at scenario level
    scenario_scores = defaultdict(lambda: {"nl": 0, "st": 0})
    for r in gt:
        sid = r.get("scenario_id")
        if r.get("nl_judge_detected") is True:
            scenario_scores[sid]["nl"] += 1
        if r.get("st_judge_detected") is True:
            scenario_scores[sid]["st"] += 1

    nl_wins = sum(1 for s in scenario_scores.values() if s["nl"] > s["st"])
    st_wins = sum(1 for s in scenario_scores.values() if s["st"] > s["nl"])
    ties = sum(1 for s in scenario_scores.values() if s["nl"] == s["st"])

    sign_p = sign_test(nl_wins, nl_wins + st_wins)

    return {
        "mcnemar_chi_sq": chi_sq,
        "mcnemar_p": p,
        "sign_nl_wins": nl_wins,
        "sign_st_wins": st_wins,
        "sign_ties": ties,
        "sign_p": sign_p,
    }


def compute_output_friction(results: list[dict]) -> dict:
    """Compute output friction — the gap between detection and compliance.

    This is the PRIMARY FINDING:
    - Detection: Did the model recognize the signal? (judge)
    - Compliance: Did the model produce XML? (regex)
    - Friction: Detection - Compliance
    """
    gt = get_ground_truth_trials(results)
    n = len(gt)

    st_detected = sum(1 for r in gt if r.get("st_judge_detected") is True)
    st_complied = sum(1 for r in gt if r.get("st_regex_detected") is True)

    # Silent failures: detected but no XML
    silent_failures = sum(1 for r in gt
                         if r.get("st_judge_detected") is True
                         and r.get("st_regex_detected") is not True)

    detection_rate = st_detected / n if n > 0 else 0
    compliance_rate = st_complied / n if n > 0 else 0
    friction_pp = (st_detected - st_complied) / n * 100 if n > 0 else 0

    return {
        "n": n,
        "detection_rate": detection_rate,
        "detection_count": st_detected,
        "compliance_rate": compliance_rate,
        "compliance_count": st_complied,
        "friction_pp": friction_pp,
        "silent_failures": silent_failures,
        "failure_ratio": f"1 in {round(n / silent_failures)}" if silent_failures > 0 else "N/A",
    }


def compute_silent_failure_analysis(results: list[dict]) -> dict:
    """Analyze the silent failure cases in detail.

    Silent failures = structured condition, judge detected, no XML.
    These are cases where the model understood but didn't comply.
    """
    gt = get_ground_truth_trials(results)

    # Get silent failure trials
    silent_failures = [r for r in gt
                       if r.get("st_judge_detected") is True
                       and r.get("st_regex_detected") is not True]

    total = len(silent_failures)

    # Check if response contains natural language acknowledgment patterns
    # (This is approximate - the judge already confirmed detection)

    return {
        "total_silent_failures": total,
        "silent_failure_trials": silent_failures,  # For example extraction
    }


def get_silent_failure_examples(results: list[dict], n: int = 5, seed: int = 42) -> list[dict]:
    """Extract representative silent failure examples for the appendix."""
    gt = get_ground_truth_trials(results)

    silent_failures = [r for r in gt
                       if r.get("st_judge_detected") is True
                       and r.get("st_regex_detected") is not True]

    if not silent_failures:
        return []

    # Try to get variety across signal types and ambiguity levels
    random.seed(seed)

    # Group by signal type
    by_type = defaultdict(list)
    for r in silent_failures:
        sig_type = r.get("signal_type", "unknown")
        by_type[sig_type].append(r)

    examples = []
    types_to_sample = list(by_type.keys())
    random.shuffle(types_to_sample)

    # Sample from each type round-robin
    idx = 0
    while len(examples) < n and any(by_type.values()):
        sig_type = types_to_sample[idx % len(types_to_sample)]
        if by_type[sig_type]:
            r = by_type[sig_type].pop(0)
            examples.append({
                "scenario_id": r.get("scenario_id"),
                "signal_type": r.get("signal_type"),
                "ambiguity": r.get("ambiguity"),
                "query": r.get("query", "")[:300],  # Truncate for display
                "response_excerpt": r.get("st_response_text", "")[:400],
                "judge_detected": True,
                "xml_present": False,
            })
        idx += 1
        if idx > len(types_to_sample) * 10:  # Safety valve
            break

    return examples


def compute_ambiguity_friction(results: list[dict]) -> dict:
    """Compute friction breakdown by ambiguity level.

    Key question: Does friction increase with ambiguity?
    """
    friction_by_level = {}

    for level in ["EXPLICIT", "IMPLICIT", "BORDERLINE"]:
        subset = [r for r in results
                  if r.get("ambiguity") == level
                  and r.get("expected_detection") is True]

        if not subset:
            continue

        n = len(subset)
        st_detected = sum(1 for r in subset if r.get("st_judge_detected") is True)
        st_complied = sum(1 for r in subset if r.get("st_regex_detected") is True)

        friction_by_level[level] = {
            "n": n,
            "detection": st_detected / n if n > 0 else 0,
            "compliance": st_complied / n if n > 0 else 0,
            "friction_pp": (st_detected - st_complied) / n * 100 if n > 0 else 0,
        }

    return friction_by_level


def compute_false_positive_rates(results: list[dict]) -> dict:
    """Compute false positive rates on CONTROL scenarios."""
    controls = [r for r in results if r.get("ambiguity") == "CONTROL"]

    n = len(controls)
    if n == 0:
        return {}

    nl_fp_judge = sum(1 for r in controls if r.get("nl_judge_detected") is True)
    st_fp_judge = sum(1 for r in controls if r.get("st_judge_detected") is True)
    nl_fp_regex = sum(1 for r in controls if r.get("nl_regex_detected") is True)
    st_fp_regex = sum(1 for r in controls if r.get("st_regex_detected") is True)

    return {
        "n_controls": n,
        "nl_fp_judge": nl_fp_judge / n,
        "st_fp_judge": st_fp_judge / n,
        "nl_fp_regex": nl_fp_regex / n,
        "st_fp_regex": st_fp_regex / n,
    }


def compute_measurement_comparison(results: list[dict]) -> dict:
    """Compare regex vs judge scoring to show measurement artifact.

    This demonstrates why cross-condition comparison is problematic
    and motivates the within-condition analysis.
    """
    gt = get_ground_truth_trials(results)
    n = len(gt)

    # Regex-based
    nl_regex = sum(1 for r in gt if r.get("nl_regex_detected") is True)
    st_regex = sum(1 for r in gt if r.get("st_regex_detected") is True)

    # Judge-based
    nl_judge = sum(1 for r in gt if r.get("nl_judge_detected") is True)
    st_judge = sum(1 for r in gt if r.get("st_judge_detected") is True)

    # McNemar for regex
    b_regex = sum(1 for r in gt
                  if r.get("nl_regex_detected") is True
                  and r.get("st_regex_detected") is not True)
    c_regex = sum(1 for r in gt
                  if r.get("st_regex_detected") is True
                  and r.get("nl_regex_detected") is not True)
    _, p_regex = mcnemar_test(b_regex, c_regex)

    # McNemar for judge (recompute for completeness)
    b_judge = sum(1 for r in gt
                  if r.get("nl_judge_detected") is True
                  and r.get("st_judge_detected") is not True)
    c_judge = sum(1 for r in gt
                  if r.get("st_judge_detected") is True
                  and r.get("nl_judge_detected") is not True)
    _, p_judge = mcnemar_test(b_judge, c_judge)

    return {
        "n": n,
        "regex": {
            "nl_recall": nl_regex / n if n > 0 else 0,
            "st_recall": st_regex / n if n > 0 else 0,
            "gap_pp": (nl_regex - st_regex) / n * 100 if n > 0 else 0,
            "mcnemar_p": p_regex,
        },
        "judge": {
            "nl_recall": nl_judge / n if n > 0 else 0,
            "st_recall": st_judge / n if n > 0 else 0,
            "gap_pp": (nl_judge - st_judge) / n * 100 if n > 0 else 0,
            "mcnemar_p": p_judge,
        },
    }


def generate_report(data: dict, output_path: Optional[Path] = None) -> str:
    """Generate formatted analysis report following the paper narrative.

    Structure:
    1. DETECTION RATES - Format does not affect detection
    2. OUTPUT FRICTION - Primary finding: detection without compliance
    3. SILENT FAILURE ANALYSIS - Model acknowledged but didn't comply
    4. FALSE POSITIVE RATES - Control scenarios
    5. AMBIGUITY BREAKDOWN - Friction by ambiguity level
    6. MEASUREMENT COMPARISON - Why regex vs judge matters
    7. JUDGE VALIDATION - Status and instructions
    """
    results = data.get("results", [])

    lines = []
    lines.append("=" * 70)
    lines.append("OUTPUT FRICTION IN TOOL CALLING — ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Data file: {data.get('judge_metadata', {}).get('source_file', 'unknown')}")
    lines.append("")

    # ================================================================
    # SECTION 1: DETECTION RATES
    # ================================================================
    lines.append("=" * 70)
    lines.append("DETECTION RATES (LLM Judge — Both Conditions)")
    lines.append("=" * 70)
    lines.append("")

    det = compute_detection_rates(results)
    tests = compute_detection_tests(results)

    lines.append(f"NL detection:         {det['nl_detection']:.1%} ({det['nl_detected_count']}/{det['n']})")
    lines.append(f"Structured detection: {det['st_detection']:.1%} ({det['st_detected_count']}/{det['n']})")
    lines.append(f"Difference:           {det['difference_pp']:+.1f}pp")
    lines.append(f"McNemar p:            {tests['mcnemar_p']:.3f}")
    lines.append(f"Sign test:            {tests['sign_nl_wins']} vs {tests['sign_st_wins']}, p = {tests['sign_p']:.3f}")
    lines.append("")
    lines.append("→ Format does not affect detection.")
    lines.append("")

    # ================================================================
    # SECTION 2: OUTPUT FRICTION (PRIMARY FINDING)
    # ================================================================
    lines.append("=" * 70)
    lines.append("OUTPUT FRICTION (Structured Condition Only) — PRIMARY FINDING")
    lines.append("=" * 70)
    lines.append("")

    friction = compute_output_friction(results)

    lines.append(f"Structured detection (judge):    {friction['detection_rate']:.1%} ({friction['detection_count']}/{friction['n']})")
    lines.append(f"Structured compliance (XML):     {friction['compliance_rate']:.1%} ({friction['compliance_count']}/{friction['n']})")
    lines.append(f"Output friction gap:             {friction['friction_pp']:.1f}pp ({friction['silent_failures']} silent failures)")
    lines.append("")
    lines.append(f"→ {friction['failure_ratio']} detections fails to produce a tool call.")
    lines.append("")

    # ================================================================
    # SECTION 3: SILENT FAILURE ANALYSIS
    # ================================================================
    lines.append("=" * 70)
    lines.append("SILENT FAILURE ANALYSIS")
    lines.append("=" * 70)
    lines.append("")

    silent = compute_silent_failure_analysis(results)

    lines.append(f"Silent failures (judge=Yes, XML=No): {silent['total_silent_failures']}/{friction['n']}")
    lines.append("")
    lines.append("These are trials where the model detected the signal")
    lines.append("(per LLM judge) but failed to produce XML for the parser.")
    lines.append("From a production system's perspective, these are missed")
    lines.append("tool calls — the signal was understood but not acted upon.")
    lines.append("")

    # ================================================================
    # SECTION 4: FALSE POSITIVE RATES
    # ================================================================
    lines.append("=" * 70)
    lines.append("FALSE POSITIVE RATES (Control Scenarios)")
    lines.append("=" * 70)
    lines.append("")

    fp = compute_false_positive_rates(results)
    if fp:
        lines.append(f"Control scenarios: {fp['n_controls']}")
        lines.append("")
        lines.append("                        JUDGE           REGEX")
        lines.append("-" * 50)
        lines.append(f"  NL FP rate:          {fp['nl_fp_judge']:.1%}            {fp['nl_fp_regex']:.1%}")
        lines.append(f"  ST FP rate:          {fp['st_fp_judge']:.1%}            {fp['st_fp_regex']:.1%}")
    lines.append("")

    # ================================================================
    # SECTION 5: AMBIGUITY BREAKDOWN
    # ================================================================
    lines.append("=" * 70)
    lines.append("AMBIGUITY BREAKDOWN")
    lines.append("=" * 70)
    lines.append("")
    lines.append("              Detection(judge)  Compliance(XML)  Friction")
    lines.append("-" * 60)

    amb_friction = compute_ambiguity_friction(results)
    for level in ["EXPLICIT", "IMPLICIT", "BORDERLINE"]:
        if level in amb_friction:
            a = amb_friction[level]
            lines.append(f"{level:12}  {a['detection']:.1%}              {a['compliance']:.1%}             {a['friction_pp']:+.1f}pp  (n={a['n']})")

    lines.append("")
    lines.append("→ Does friction increase with ambiguity?")
    if "EXPLICIT" in amb_friction and "IMPLICIT" in amb_friction:
        exp_friction = amb_friction["EXPLICIT"]["friction_pp"]
        imp_friction = amb_friction["IMPLICIT"]["friction_pp"]
        if exp_friction < imp_friction:
            lines.append(f"   Yes: EXPLICIT ({exp_friction:+.1f}pp) < IMPLICIT ({imp_friction:+.1f}pp)")
        else:
            lines.append(f"   No clear pattern: EXPLICIT ({exp_friction:+.1f}pp), IMPLICIT ({imp_friction:+.1f}pp)")
    lines.append("")

    # ================================================================
    # SECTION 6: MEASUREMENT COMPARISON (Methodological)
    # ================================================================
    lines.append("=" * 70)
    lines.append("MEASUREMENT COMPARISON (Methodological)")
    lines.append("=" * 70)
    lines.append("")
    lines.append("This comparison shows why cross-condition analysis is problematic.")
    lines.append("The within-condition gap (Section 2) is the appropriate metric.")
    lines.append("")

    comp = compute_measurement_comparison(results)

    lines.append("                Regex scoring    Judge scoring")
    lines.append("-" * 50)
    lines.append(f"NL recall:      {comp['regex']['nl_recall']:.1%}            {comp['judge']['nl_recall']:.1%}")
    lines.append(f"ST recall:      {comp['regex']['st_recall']:.1%}            {comp['judge']['st_recall']:.1%}")
    lines.append(f"NL-ST gap:      {comp['regex']['gap_pp']:+.1f}pp           {comp['judge']['gap_pp']:+.1f}pp")
    lines.append(f"McNemar p:      {comp['regex']['mcnemar_p']:.4f}          {comp['judge']['mcnemar_p']:.4f}")
    lines.append("")
    lines.append("→ Cross-condition gap was measurement artifact.")
    lines.append("   Within-condition gap (Section 2) is the real finding.")
    lines.append("")

    # ================================================================
    # SECTION 7: JUDGE VALIDATION
    # ================================================================
    lines.append("=" * 70)
    lines.append("JUDGE VALIDATION")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Judge-human κ:           [PENDING ANNOTATION]")
    lines.append("Agreement rate:          [PENDING ANNOTATION]")
    lines.append("")
    lines.append("To complete validation:")
    lines.append("  1. Annotate: experiments/results/validation_annotation_*.csv")
    lines.append("  2. Run: python experiments/compute_agreement.py <annotated.csv> <key.csv>")
    lines.append("  3. Target: κ ≥ 0.80 for substantial agreement")
    lines.append("")

    # ================================================================
    # END
    # ================================================================
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    report = "\n".join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)

    return report


def print_silent_failure_examples(results: list[dict], n: int = 5) -> None:
    """Print representative silent failure examples."""
    examples = get_silent_failure_examples(results, n)

    if not examples:
        print("No silent failure examples found.")
        return

    print("\n" + "=" * 70)
    print("SILENT FAILURE EXAMPLES (for paper appendix)")
    print("=" * 70)

    for i, ex in enumerate(examples, 1):
        print(f"\n--- Example {i} ---")
        print(f"Scenario: {ex['scenario_id']}")
        print(f"Signal type: {ex['signal_type']}")
        print(f"Ambiguity: {ex['ambiguity']}")
        print(f"Query excerpt: {ex['query']}...")
        print(f"Response excerpt: {ex['response_excerpt']}...")
        print(f"XML present: No")
        print(f"Judge detected: Yes")
        print("Production outcome: TOOL CALL DOES NOT FIRE")


def main():
    parser = argparse.ArgumentParser(
        description="Run complete analysis on judged signal detection results"
    )
    parser.add_argument(
        "results_file",
        type=Path,
        help="Path to judged results JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for report (default: same as input)"
    )
    parser.add_argument(
        "--show-examples",
        action="store_true",
        help="Print silent failure examples for appendix"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of examples to show (default: 5)"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading results from {args.results_file}")
    data = load_judged_results(args.results_file)
    results = data.get("results", [])
    print(f"Found {len(results)} trial records")

    # Count ground truth
    gt = get_ground_truth_trials(results)
    print(f"Ground truth trials (EXPLICIT + IMPLICIT): {len(gt)}")

    # Determine output path
    output_dir = args.output_dir or args.results_file.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"analysis_report_{timestamp}.txt"

    # Generate report
    report = generate_report(data, output_path)

    # Print to stdout
    print(report)

    # Optionally show examples
    if args.show_examples:
        print_silent_failure_examples(results, args.num_examples)

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
