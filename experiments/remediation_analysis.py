"""
Remediation Analysis Script

Computes all statistics required by the REVIEW.md remediation plan:
- Issue 2: Contingency table (Detection × Compliance)
- Issue 4: Missing analyses (signal type, scenario variance, response length)
- Issue 1: Bounded IMPLICIT friction estimates
- Issue 3: HARD scenario CIs
- Issue 9: FPR with Fisher's exact test

Usage:
    python experiments/remediation_analysis.py experiments/results/primary/signal_detection_20260203_074411_judged.json
"""

import json
import argparse
import math
from pathlib import Path
from collections import defaultdict
from typing import Optional
import numpy as np


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 1.0)

    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom

    return (max(0, center - margin), min(1, center + margin))


def load_judged_results(path: Path) -> dict:
    """Load judged results JSON."""
    with open(path) as f:
        return json.load(f)


def compute_contingency_table(results: list[dict], exclude_hard: bool = True) -> dict:
    """
    Issue 2: Compute Detection × Compliance contingency table.

    This clarifies the relationship between "50 compliance gaps" and "10.3pp friction".
    """
    # Filter to ground truth scenarios (EXPLICIT + IMPLICIT)
    with_truth = [r for r in results
                  if r.get("ambiguity") in ["EXPLICIT", "IMPLICIT"]
                  and r.get("expected_detection") is True]

    if exclude_hard:
        hard_ids = ['sig_implicit_frust_007', 'sig_implicit_block_001', 'sig_implicit_block_008']
        with_truth = [r for r in with_truth if r.get("scenario_id") not in hard_ids]

    # Compute 2×2 contingency: Detection (judge) × Compliance (XML)
    both = 0  # Detection YES, XML YES
    detection_only = 0  # Detection YES, XML NO (compliance gaps)
    xml_only = 0  # Detection NO, XML YES (reverse gaps)
    neither = 0  # Detection NO, XML NO

    for r in with_truth:
        detected = r.get("st_judge_detected") is True
        xml_present = r.get("st_regex_detected") is True

        if detected and xml_present:
            both += 1
        elif detected and not xml_present:
            detection_only += 1
        elif not detected and xml_present:
            xml_only += 1
        else:
            neither += 1

    n = len(with_truth)
    total_detected = both + detection_only
    total_compliant = both + xml_only

    # Net friction = (detection - compliance) / n
    net_friction_pp = (total_detected - total_compliant) / n * 100 if n > 0 else 0

    return {
        "n": n,
        "exclude_hard": exclude_hard,
        "contingency": {
            "both_detected_and_xml": both,
            "detection_only_compliance_gaps": detection_only,
            "xml_only_reverse_gaps": xml_only,
            "neither": neither,
        },
        "totals": {
            "total_detected": total_detected,
            "total_xml_present": total_compliant,
            "detection_rate": total_detected / n if n > 0 else 0,
            "compliance_rate": total_compliant / n if n > 0 else 0,
        },
        "metrics": {
            "gross_compliance_gaps": detection_only,
            "reverse_gaps": xml_only,
            "net_difference": total_detected - total_compliant,
            "net_friction_pp": net_friction_pp,
        },
        "explanation": (
            f"{detection_only} trials showed detection without XML (compliance gaps). "
            f"Net friction rate ({net_friction_pp:.1f}pp) differs because {xml_only} trials "
            f"showed XML without judged detection (reverse gaps)."
        ),
    }


def analyze_by_signal_type(results: list[dict], exclude_hard: bool = True) -> dict:
    """
    Issue 4a: Friction breakdown by signal type (frustration, urgency, blocking_issue).
    """
    if exclude_hard:
        hard_ids = ['sig_implicit_frust_007', 'sig_implicit_block_001', 'sig_implicit_block_008']
        results = [r for r in results if r.get("scenario_id") not in hard_ids]

    analysis = {}

    for signal_type in ['frustration', 'urgency', 'blocking_issue']:
        subset = [r for r in results
                  if r.get('signal_type') == signal_type
                  and r.get('ambiguity') in ['EXPLICIT', 'IMPLICIT']
                  and r.get('expected_detection') is True]

        if not subset:
            continue

        detection = sum(1 for r in subset if r.get('st_judge_detected') is True)
        compliance = sum(1 for r in subset if r.get('st_regex_detected') is True)
        n = len(subset)

        det_rate = detection / n if n > 0 else 0
        comp_rate = compliance / n if n > 0 else 0
        friction_pp = (detection - compliance) / n * 100 if n > 0 else 0

        analysis[signal_type] = {
            'n': n,
            'detection_count': detection,
            'compliance_count': compliance,
            'detection_rate': det_rate,
            'detection_ci': wilson_ci(detection, n),
            'compliance_rate': comp_rate,
            'compliance_ci': wilson_ci(compliance, n),
            'friction_pp': friction_pp,
            'compliance_gaps': detection - compliance if detection > compliance else 0,
        }

    return analysis


def scenario_friction_distribution(results: list[dict], exclude_hard: bool = True) -> dict:
    """
    Issue 4b: Compute friction for each IMPLICIT scenario to show variance.
    """
    if exclude_hard:
        hard_ids = ['sig_implicit_frust_007', 'sig_implicit_block_001', 'sig_implicit_block_008']
        results = [r for r in results if r.get("scenario_id") not in hard_ids]

    by_scenario = defaultdict(list)

    for r in results:
        if r.get('ambiguity') == 'IMPLICIT' and r.get('expected_detection') is True:
            by_scenario[r['scenario_id']].append(r)

    frictions = []
    for scenario_id, trials in by_scenario.items():
        det = sum(1 for t in trials if t.get('st_judge_detected') is True)
        comp = sum(1 for t in trials if t.get('st_regex_detected') is True)
        n = len(trials)
        friction = (det - comp) / n * 100 if n > 0 else 0
        frictions.append({
            'scenario_id': scenario_id,
            'n': n,
            'detection': det,
            'compliance': comp,
            'friction_pp': friction,
            'det_rate': det / n if n > 0 else 0,
            'comp_rate': comp / n if n > 0 else 0,
        })

    friction_values = [f['friction_pp'] for f in frictions]

    return {
        'n_scenarios': len(frictions),
        'scenarios': sorted(frictions, key=lambda x: -x['friction_pp']),
        'summary': {
            'mean_friction': float(np.mean(friction_values)) if friction_values else 0,
            'std_friction': float(np.std(friction_values)) if friction_values else 0,
            'median_friction': float(np.median(friction_values)) if friction_values else 0,
            'max_friction': max(friction_values) if friction_values else 0,
            'min_friction': min(friction_values) if friction_values else 0,
            'iqr_25': float(np.percentile(friction_values, 25)) if friction_values else 0,
            'iqr_75': float(np.percentile(friction_values, 75)) if friction_values else 0,
        },
        'distribution': {
            'zero_friction': sum(1 for f in friction_values if f == 0),
            'low_friction_0_10': sum(1 for f in friction_values if 0 < f <= 10),
            'medium_friction_10_30': sum(1 for f in friction_values if 10 < f <= 30),
            'high_friction_30_plus': sum(1 for f in friction_values if f > 30),
        },
    }


def response_length_analysis(results: list[dict], exclude_hard: bool = True) -> dict:
    """
    Issue 4c: Analyze whether longer responses have more compliance gaps.
    """
    if exclude_hard:
        hard_ids = ['sig_implicit_frust_007', 'sig_implicit_block_001', 'sig_implicit_block_008']
        results = [r for r in results if r.get("scenario_id") not in hard_ids]

    with_truth = [r for r in results
                  if r.get('ambiguity') in ['EXPLICIT', 'IMPLICIT']
                  and r.get('expected_detection') is True]

    # Compliance gaps: detection YES, XML NO
    gap_lengths = [len(r.get('st_response_text', ''))
                   for r in with_truth
                   if r.get('st_judge_detected') is True
                   and r.get('st_regex_detected') is not True]

    # Successful compliance: XML YES
    success_lengths = [len(r.get('st_response_text', ''))
                       for r in with_truth
                       if r.get('st_regex_detected') is True]

    # Non-detections: detection NO
    no_detection_lengths = [len(r.get('st_response_text', ''))
                            for r in with_truth
                            if r.get('st_judge_detected') is not True]

    result = {
        'compliance_gaps': {
            'n': len(gap_lengths),
            'mean_length': float(np.mean(gap_lengths)) if gap_lengths else 0,
            'median_length': float(np.median(gap_lengths)) if gap_lengths else 0,
            'std_length': float(np.std(gap_lengths)) if gap_lengths else 0,
        },
        'successful_compliance': {
            'n': len(success_lengths),
            'mean_length': float(np.mean(success_lengths)) if success_lengths else 0,
            'median_length': float(np.median(success_lengths)) if success_lengths else 0,
            'std_length': float(np.std(success_lengths)) if success_lengths else 0,
        },
        'non_detections': {
            'n': len(no_detection_lengths),
            'mean_length': float(np.mean(no_detection_lengths)) if no_detection_lengths else 0,
            'median_length': float(np.median(no_detection_lengths)) if no_detection_lengths else 0,
            'std_length': float(np.std(no_detection_lengths)) if no_detection_lengths else 0,
        },
    }

    # Statistical test
    if gap_lengths and success_lengths:
        from scipy.stats import mannwhitneyu
        stat, pvalue = mannwhitneyu(gap_lengths, success_lengths, alternative='two-sided')
        result['statistical_test'] = {
            'test': 'Mann-Whitney U',
            'statistic': float(stat),
            'p_value': float(pvalue),
            'interpretation': 'significant' if pvalue < 0.05 else 'not significant',
        }

    return result


def compute_bounded_friction(results: list[dict]) -> dict:
    """
    Issue 1: Compute bounded IMPLICIT friction estimates using existing human validation.

    IMPLICIT judge-human disagreement rate is 24% (κ = 0.41).
    """
    hard_ids = ['sig_implicit_frust_007', 'sig_implicit_block_001', 'sig_implicit_block_008']

    implicit = [r for r in results
                if r.get('ambiguity') == 'IMPLICIT'
                and r.get('expected_detection') is True
                and r.get('scenario_id') not in hard_ids]

    if not implicit:
        return {"error": "No IMPLICIT scenarios found"}

    n = len(implicit)

    # Judge-based detection
    det_judge = sum(1 for r in implicit if r.get('st_judge_detected') is True)
    comp = sum(1 for r in implicit if r.get('st_regex_detected') is True)

    # Optimistic (judge-based)
    friction_optimistic = (det_judge - comp) / n * 100

    # Conservative: discount detection by IMPLICIT disagreement rate (24%)
    # This accounts for cases where judge says YES but human would say NO
    disagreement_rate = 0.24  # From validation: κ = 0.41, 76% agreement
    det_conservative = det_judge * (1 - disagreement_rate)
    friction_conservative = (det_conservative - comp) / n * 100

    return {
        'n': n,
        'detection_judge': det_judge,
        'compliance': comp,
        'detection_rate_judge': det_judge / n,
        'compliance_rate': comp / n,
        'friction_optimistic_pp': friction_optimistic,
        'friction_conservative_pp': friction_conservative,
        'uncertainty_pp': friction_optimistic - friction_conservative,
        'interpretation': (
            f"IMPLICIT friction estimates range from {friction_conservative:.1f}pp (conservative) "
            f"to {friction_optimistic:.1f}pp (judge-based), with ±{(friction_optimistic - friction_conservative):.1f}pp "
            f"uncertainty due to judge-human disagreement (κ = 0.41, 24% disagreement rate)."
        ),
    }


def hard_scenario_analysis(results: list[dict]) -> dict:
    """
    Issue 3: Compute CIs for HARD scenarios to show uncertainty.
    """
    hard_ids = ['sig_implicit_frust_007', 'sig_implicit_block_001', 'sig_implicit_block_008']
    hard = [r for r in results if r.get('scenario_id') in hard_ids]

    if not hard:
        return {"error": "No HARD scenarios found"}

    det = sum(1 for r in hard if r.get('st_judge_detected') is True)
    comp = sum(1 for r in hard if r.get('st_regex_detected') is True)
    n = len(hard)

    det_ci = wilson_ci(det, n)
    comp_ci = wilson_ci(comp, n)

    # NL detection for comparison
    nl_det = sum(1 for r in hard if r.get('nl_judge_detected') is True)
    nl_det_ci = wilson_ci(nl_det, n)

    return {
        'n': n,
        'scenario_ids': hard_ids,
        'structured': {
            'detection': det,
            'detection_rate': det / n if n > 0 else 0,
            'detection_ci_95': det_ci,
            'compliance': comp,
            'compliance_rate': comp / n if n > 0 else 0,
            'compliance_ci_95': comp_ci,
            'friction_pp': (det - comp) / n * 100 if n > 0 else 0,
        },
        'natural_language': {
            'detection': nl_det,
            'detection_rate': nl_det / n if n > 0 else 0,
            'detection_ci_95': nl_det_ci,
        },
        'note': (
            f"With n={n}, these CIs are wide: detection [{det_ci[0]:.1%}, {det_ci[1]:.1%}], "
            f"compliance [{comp_ci[0]:.1%}, {comp_ci[1]:.1%}]. "
            f"This is an exploratory observation; inference is unreliable."
        ),
    }


def fpr_with_uncertainty(results: list[dict]) -> dict:
    """
    Issue 9: Compute FPR with appropriate uncertainty for small samples.
    """
    control = [r for r in results if r.get('ambiguity') == 'CONTROL']

    if not control:
        return {"error": "No CONTROL scenarios found"}

    n = len(control)

    nl_fp = sum(1 for r in control if r.get('nl_judge_detected') is True)
    st_fp = sum(1 for r in control if r.get('st_judge_detected') is True)

    nl_fpr = nl_fp / n if n > 0 else 0
    st_fpr = st_fp / n if n > 0 else 0

    nl_ci = wilson_ci(nl_fp, n)
    st_ci = wilson_ci(st_fp, n)

    # Fisher's exact test
    from scipy.stats import fisher_exact
    # Table: [[NL_noFP, NL_FP], [ST_noFP, ST_FP]]
    table = [[n - nl_fp, nl_fp], [n - st_fp, st_fp]]
    odds_ratio, p_value = fisher_exact(table)

    return {
        'n': n,
        'nl': {
            'false_positives': nl_fp,
            'fpr': nl_fpr,
            'fpr_ci_95': nl_ci,
        },
        'structured': {
            'false_positives': st_fp,
            'fpr': st_fpr,
            'fpr_ci_95': st_ci,
        },
        'statistical_test': {
            'test': "Fisher's exact",
            'odds_ratio': float(odds_ratio),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),  # Convert numpy bool to Python bool
        },
        'interpretation': (
            f"We observed {st_fp} false positives in structured vs. {nl_fp} in NL "
            f"(p={p_value:.3f}, Fisher's exact). With only {st_fp} events, this may be noise; "
            f"we note it as an observation for future investigation. "
            f"Structured FPR 95% CI: [{st_ci[0]:.1%}, {st_ci[1]:.1%}]."
        ),
    }


def scenario_sign_test_for_friction(results: list[dict], exclude_hard: bool = True) -> dict:
    """
    Issue 5: Scenario-level sign test specifically for friction (detection > compliance).
    """
    if exclude_hard:
        hard_ids = ['sig_implicit_frust_007', 'sig_implicit_block_001', 'sig_implicit_block_008']
        results = [r for r in results if r.get("scenario_id") not in hard_ids]

    by_scenario = defaultdict(list)
    for r in results:
        if r.get("ambiguity") in ["EXPLICIT", "IMPLICIT"]:
            by_scenario[r.get("scenario_id")].append(r)

    detection_higher = 0
    compliance_higher = 0
    ties = 0

    for scenario_id, trials in by_scenario.items():
        det = sum(1 for t in trials if t.get("st_judge_detected") is True)
        comp = sum(1 for t in trials if t.get("st_regex_detected") is True)

        if det > comp:
            detection_higher += 1
        elif comp > det:
            compliance_higher += 1
        else:
            ties += 1

    from scipy.stats import binomtest
    n_different = detection_higher + compliance_higher
    p_value = binomtest(detection_higher, n_different, 0.5, alternative='two-sided').pvalue if n_different > 0 else 1.0

    return {
        'n_scenarios': len(by_scenario),
        'detection_higher': detection_higher,
        'compliance_higher': compliance_higher,
        'ties': ties,
        'sign_test_p': p_value,
        'interpretation': (
            f"Of {len(by_scenario)} ground-truth scenarios, {detection_higher} showed detection > compliance, "
            f"{compliance_higher} showed compliance > detection, {ties} tied. "
            f"Sign test p = {p_value:.6f}."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Remediation analysis script for REVIEW.md"
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

    # Run all analyses
    analysis = {
        "issue_2_contingency_table": compute_contingency_table(results, exclude_hard=True),
        "issue_4a_signal_type_breakdown": analyze_by_signal_type(results, exclude_hard=True),
        "issue_4b_scenario_variance": scenario_friction_distribution(results, exclude_hard=True),
        "issue_4c_response_length": response_length_analysis(results, exclude_hard=True),
        "issue_1_bounded_implicit_friction": compute_bounded_friction(results),
        "issue_3_hard_scenario_cis": hard_scenario_analysis(results),
        "issue_9_fpr_uncertainty": fpr_with_uncertainty(results),
        "issue_5_friction_sign_test": scenario_sign_test_for_friction(results, exclude_hard=True),
    }

    # Print report
    print("\n" + "=" * 70)
    print("REMEDIATION ANALYSIS RESULTS")
    print("=" * 70)

    # Issue 2: Contingency Table
    print("\n## Issue 2: Detection × Compliance Contingency Table")
    print("-" * 50)
    ct = analysis["issue_2_contingency_table"]
    print(f"N = {ct['n']} (excluding HARD scenarios)")
    print("\n|                    | XML Present | XML Absent |")
    print("|--------------------|-------------|------------|")
    c = ct["contingency"]
    print(f"| Judge: Detected    | {c['both_detected_and_xml']:11} | {c['detection_only_compliance_gaps']:10} (compliance gaps) |")
    print(f"| Judge: Not Detected| {c['xml_only_reverse_gaps']:11} (reverse gaps) | {c['neither']:10} |")
    print(f"\n{ct['explanation']}")

    # Issue 4a: Signal Type
    print("\n## Issue 4a: Friction by Signal Type")
    print("-" * 50)
    st = analysis["issue_4a_signal_type_breakdown"]
    print("| Signal Type     | N   | Detection | Compliance | Friction |")
    print("|-----------------|-----|-----------|------------|----------|")
    for sig_type, stats in st.items():
        print(f"| {sig_type:15} | {stats['n']:3} | {stats['detection_rate']:.1%}     | {stats['compliance_rate']:.1%}      | {stats['friction_pp']:+.1f}pp   |")

    # Issue 4b: Scenario Variance
    print("\n## Issue 4b: IMPLICIT Scenario Friction Distribution")
    print("-" * 50)
    sv = analysis["issue_4b_scenario_variance"]
    s = sv["summary"]
    print(f"N scenarios: {sv['n_scenarios']}")
    print(f"Mean friction: {s['mean_friction']:.1f}pp (SD: {s['std_friction']:.1f})")
    print(f"Median: {s['median_friction']:.1f}pp, Range: [{s['min_friction']:.1f}, {s['max_friction']:.1f}]")
    print(f"IQR: [{s['iqr_25']:.1f}, {s['iqr_75']:.1f}]")
    d = sv["distribution"]
    print(f"\nDistribution: {d['zero_friction']} at 0%, {d['low_friction_0_10']} at 1-10%, "
          f"{d['medium_friction_10_30']} at 11-30%, {d['high_friction_30_plus']} at 30%+")
    print("\nTop 5 highest friction scenarios:")
    for sc in sv["scenarios"][:5]:
        print(f"  {sc['scenario_id']}: {sc['friction_pp']:.0f}pp ({sc['detection']}/{sc['n']} det, {sc['compliance']}/{sc['n']} comp)")

    # Issue 4c: Response Length
    print("\n## Issue 4c: Response Length Analysis")
    print("-" * 50)
    rl = analysis["issue_4c_response_length"]
    print(f"Compliance gaps:      mean {rl['compliance_gaps']['mean_length']:.0f} chars (n={rl['compliance_gaps']['n']})")
    print(f"Successful compliance: mean {rl['successful_compliance']['mean_length']:.0f} chars (n={rl['successful_compliance']['n']})")
    if 'statistical_test' in rl:
        test = rl['statistical_test']
        print(f"Mann-Whitney U: p = {test['p_value']:.4f} ({test['interpretation']})")

    # Issue 1: Bounded Friction
    print("\n## Issue 1: Bounded IMPLICIT Friction Estimates")
    print("-" * 50)
    bf = analysis["issue_1_bounded_implicit_friction"]
    print(f"N = {bf['n']} IMPLICIT trials (excluding HARD)")
    print(f"Detection (judge): {bf['detection_judge']}/{bf['n']} = {bf['detection_rate_judge']:.1%}")
    print(f"Compliance (XML):  {bf['compliance']}/{bf['n']} = {bf['compliance_rate']:.1%}")
    print(f"\nFriction estimates:")
    print(f"  Optimistic (judge-based): {bf['friction_optimistic_pp']:.1f}pp")
    print(f"  Conservative (adj. for 24% disagreement): {bf['friction_conservative_pp']:.1f}pp")
    print(f"  Uncertainty: ±{bf['uncertainty_pp']:.1f}pp")

    # Issue 3: HARD Scenarios
    print("\n## Issue 3: HARD Scenario Analysis")
    print("-" * 50)
    ha = analysis["issue_3_hard_scenario_cis"]
    print(f"N = {ha['n']} trials from 3 HARD scenarios")
    hs = ha["structured"]
    print(f"Structured detection: {hs['detection']}/{ha['n']} = {hs['detection_rate']:.1%} "
          f"95% CI: [{hs['detection_ci_95'][0]:.1%}, {hs['detection_ci_95'][1]:.1%}]")
    print(f"Structured compliance: {hs['compliance']}/{ha['n']} = {hs['compliance_rate']:.1%} "
          f"95% CI: [{hs['compliance_ci_95'][0]:.1%}, {hs['compliance_ci_95'][1]:.1%}]")
    print(f"\n{ha['note']}")

    # Issue 9: FPR
    print("\n## Issue 9: False Positive Rate Analysis")
    print("-" * 50)
    fp = analysis["issue_9_fpr_uncertainty"]
    print(f"N = {fp['n']} CONTROL trials")
    print(f"NL FPR: {fp['nl']['fpr']:.1%} ({fp['nl']['false_positives']}/{fp['n']}) "
          f"95% CI: [{fp['nl']['fpr_ci_95'][0]:.1%}, {fp['nl']['fpr_ci_95'][1]:.1%}]")
    print(f"ST FPR: {fp['structured']['fpr']:.1%} ({fp['structured']['false_positives']}/{fp['n']}) "
          f"95% CI: [{fp['structured']['fpr_ci_95'][0]:.1%}, {fp['structured']['fpr_ci_95'][1]:.1%}]")
    print(f"Fisher's exact p = {fp['statistical_test']['p_value']:.3f}")

    # Issue 5: Sign Test
    print("\n## Issue 5: Friction Sign Test (Scenario-Level)")
    print("-" * 50)
    st = analysis["issue_5_friction_sign_test"]
    print(f"N scenarios: {st['n_scenarios']}")
    print(f"Detection > Compliance: {st['detection_higher']}")
    print(f"Compliance > Detection: {st['compliance_higher']}")
    print(f"Ties: {st['ties']}")
    print(f"Sign test p = {st['sign_test_p']:.6f}")

    print("\n" + "=" * 70)

    # Save JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nSaved analysis to {args.output}")
    else:
        # Default output
        output_path = args.judged_file.parent / "remediation_analysis.json"
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nSaved analysis to {output_path}")


if __name__ == "__main__":
    main()
