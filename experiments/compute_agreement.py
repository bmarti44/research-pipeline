"""
Compute Agreement Between LLM Judge and Human Annotators

This script takes the annotated CSV and key file and computes:
1. Cohen's kappa (judge vs human) - headline metric
2. Raw agreement rate
3. Confusion matrix
4. Agreement by condition (NL vs structured)
5. Agreement by ambiguity level

Usage:
    python experiments/compute_agreement.py <annotated.csv> <key.csv>
"""

import csv
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional


def load_annotated_csv(path: Path) -> dict[str, Optional[bool]]:
    """Load human annotations from CSV.

    Returns:
        Dict mapping sample_id to human label (True=YES, False=NO, None=missing)
    """
    annotations = {}

    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            sample_id = row.get("sample_id", "")
            human_label = row.get("human_label", "").strip().upper()

            if human_label == "YES":
                annotations[sample_id] = True
            elif human_label == "NO":
                annotations[sample_id] = False
            else:
                annotations[sample_id] = None

    return annotations


def load_key_file(path: Path) -> list[dict]:
    """Load key file with judge/regex results."""
    keys = []

    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            judge_val = row.get("judge_detected", "").strip()
            regex_val = row.get("regex_detected", "").strip()

            keys.append({
                "sample_id": row.get("sample_id"),
                "scenario_id": row.get("scenario_id"),
                "ambiguity_level": row.get("ambiguity_level"),
                "condition": row.get("condition"),
                "judge_detected": judge_val.lower() == "true" if judge_val else None,
                "regex_detected": regex_val.lower() == "true" if regex_val else None,
            })

    return keys


def compute_cohens_kappa(
    a_labels: list[bool],
    b_labels: list[bool],
) -> float:
    """Compute Cohen's kappa between two sets of binary labels.

    Args:
        a_labels: First annotator's labels
        b_labels: Second annotator's labels

    Returns:
        Cohen's kappa coefficient
    """
    if len(a_labels) != len(b_labels):
        raise ValueError("Label lists must have same length")

    n = len(a_labels)
    if n == 0:
        return 0.0

    # Count agreement
    agree = sum(1 for a, b in zip(a_labels, b_labels) if a == b)
    p_o = agree / n  # Observed agreement

    # Count marginals
    a_yes = sum(a_labels)
    a_no = n - a_yes
    b_yes = sum(b_labels)
    b_no = n - b_yes

    # Expected agreement by chance
    p_e = (a_yes * b_yes + a_no * b_no) / (n * n)

    # Kappa
    if p_e == 1.0:
        return 1.0  # Perfect agreement on all same label
    return (p_o - p_e) / (1 - p_e)


def compute_confusion_matrix(
    judge_labels: list[bool],
    human_labels: list[bool],
) -> dict[str, int]:
    """Compute confusion matrix.

    Returns:
        Dict with tp, tn, fp, fn counts (judge as predictor, human as gold)
    """
    tp = sum(1 for j, h in zip(judge_labels, human_labels) if j and h)
    tn = sum(1 for j, h in zip(judge_labels, human_labels) if not j and not h)
    fp = sum(1 for j, h in zip(judge_labels, human_labels) if j and not h)
    fn = sum(1 for j, h in zip(judge_labels, human_labels) if not j and h)

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def analyze_agreement(
    annotations: dict[str, Optional[bool]],
    keys: list[dict],
) -> dict:
    """Compute all agreement metrics.

    Returns:
        Dict with all computed metrics
    """
    results = {
        "overall": {},
        "by_condition": {},
        "by_ambiguity": {},
        "flagged_strata": [],
    }

    # Filter to samples with both human and judge labels
    paired = []
    for key in keys:
        sample_id = key["sample_id"]
        human = annotations.get(sample_id)
        judge = key["judge_detected"]

        if human is not None and judge is not None:
            paired.append({
                "human": human,
                "judge": judge,
                "regex": key["regex_detected"],
                "condition": key["condition"],
                "ambiguity": key["ambiguity_level"],
            })

    if not paired:
        print("ERROR: No samples with both human and judge labels")
        return results

    # Overall metrics
    human_labels = [p["human"] for p in paired]
    judge_labels = [p["judge"] for p in paired]

    kappa = compute_cohens_kappa(judge_labels, human_labels)
    agreement = sum(1 for h, j in zip(human_labels, judge_labels) if h == j) / len(paired)
    confusion = compute_confusion_matrix(judge_labels, human_labels)

    results["overall"] = {
        "n": len(paired),
        "kappa": kappa,
        "agreement": agreement,
        "confusion": confusion,
    }

    # By condition
    for condition in ["nl", "structured"]:
        cond_pairs = [p for p in paired if p["condition"] == condition]
        if not cond_pairs:
            continue

        h = [p["human"] for p in cond_pairs]
        j = [p["judge"] for p in cond_pairs]

        cond_kappa = compute_cohens_kappa(j, h)
        cond_agree = sum(1 for hi, ji in zip(h, j) if hi == ji) / len(cond_pairs)

        results["by_condition"][condition] = {
            "n": len(cond_pairs),
            "kappa": cond_kappa,
            "agreement": cond_agree,
        }

        if cond_kappa < 0.70:
            results["flagged_strata"].append(f"condition={condition} (κ={cond_kappa:.2f})")

    # By ambiguity level
    for ambiguity in ["EXPLICIT", "IMPLICIT", "BORDERLINE", "CONTROL"]:
        amb_pairs = [p for p in paired if p["ambiguity"] == ambiguity]
        if not amb_pairs:
            continue

        h = [p["human"] for p in amb_pairs]
        j = [p["judge"] for p in amb_pairs]

        amb_kappa = compute_cohens_kappa(j, h)
        amb_agree = sum(1 for hi, ji in zip(h, j) if hi == ji) / len(amb_pairs)

        results["by_ambiguity"][ambiguity] = {
            "n": len(amb_pairs),
            "kappa": amb_kappa,
            "agreement": amb_agree,
        }

        if amb_kappa < 0.70:
            results["flagged_strata"].append(f"ambiguity={ambiguity} (κ={amb_kappa:.2f})")

    # Regex comparison (bonus analysis)
    regex_pairs = [p for p in paired if p["regex"] is not None]
    if regex_pairs:
        h = [p["human"] for p in regex_pairs]
        r = [p["regex"] for p in regex_pairs]

        regex_kappa = compute_cohens_kappa(r, h)
        regex_agree = sum(1 for hi, ri in zip(h, r) if hi == ri) / len(regex_pairs)

        results["regex_comparison"] = {
            "n": len(regex_pairs),
            "kappa": regex_kappa,
            "agreement": regex_agree,
        }

    return results


def print_report(results: dict) -> None:
    """Print formatted agreement report."""

    print("\n" + "=" * 70)
    print("JUDGE-HUMAN AGREEMENT REPORT")
    print("=" * 70)

    overall = results.get("overall", {})
    print(f"\nOVERALL (n={overall.get('n', 0)})")
    print("-" * 40)
    print(f"  Cohen's κ:     {overall.get('kappa', 0):.3f}")
    print(f"  Agreement:     {overall.get('agreement', 0):.1%}")

    conf = overall.get("confusion", {})
    if conf:
        print(f"\n  Confusion Matrix (Judge vs Human):")
        print(f"                    Human YES    Human NO")
        print(f"    Judge YES       {conf.get('tp', 0):8}     {conf.get('fp', 0):8}")
        print(f"    Judge NO        {conf.get('fn', 0):8}     {conf.get('tn', 0):8}")

        # Interpretation
        fp_rate = conf.get('fp', 0) / (conf.get('fp', 0) + conf.get('tn', 0) + 0.001)
        fn_rate = conf.get('fn', 0) / (conf.get('fn', 0) + conf.get('tp', 0) + 0.001)
        print(f"\n  Judge false positive rate: {fp_rate:.1%}")
        print(f"  Judge false negative rate: {fn_rate:.1%}")

    print(f"\nBY CONDITION")
    print("-" * 40)
    for cond, stats in results.get("by_condition", {}).items():
        print(f"  {cond:12} κ={stats['kappa']:.3f}  agreement={stats['agreement']:.1%}  (n={stats['n']})")

    print(f"\nBY AMBIGUITY LEVEL")
    print("-" * 40)
    for amb, stats in results.get("by_ambiguity", {}).items():
        print(f"  {amb:12} κ={stats['kappa']:.3f}  agreement={stats['agreement']:.1%}  (n={stats['n']})")

    regex = results.get("regex_comparison")
    if regex:
        print(f"\nREGEX-HUMAN COMPARISON (for reference)")
        print("-" * 40)
        print(f"  Cohen's κ:     {regex['kappa']:.3f}")
        print(f"  Agreement:     {regex['agreement']:.1%}")
        print(f"  (n={regex['n']})")

    flagged = results.get("flagged_strata", [])
    if flagged:
        print(f"\n⚠️  FLAGGED STRATA (κ < 0.70)")
        print("-" * 40)
        for flag in flagged:
            print(f"  • {flag}")
        print("\n  These strata may need judge prompt tuning or indicate")
        print("  genuine ambiguity where humans disagree.")

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("  κ > 0.80: Substantial to almost perfect agreement - proceed")
    print("  κ 0.60-0.80: Moderate to substantial - acceptable with caveats")
    print("  κ < 0.60: Fair or worse - investigate disagreements")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compute agreement between LLM judge and human annotators"
    )
    parser.add_argument(
        "annotated_csv",
        type=Path,
        help="Path to human-annotated CSV file"
    )
    parser.add_argument(
        "key_csv",
        type=Path,
        help="Path to key CSV file with judge/regex results"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading annotations from {args.annotated_csv}")
    annotations = load_annotated_csv(args.annotated_csv)
    print(f"Found {sum(1 for v in annotations.values() if v is not None)} labeled samples")

    print(f"Loading key from {args.key_csv}")
    keys = load_key_file(args.key_csv)
    print(f"Found {len(keys)} key entries")

    # Compute agreement
    results = analyze_agreement(annotations, keys)

    # Print report
    print_report(results)

    # Decision guidance
    kappa = results.get("overall", {}).get("kappa", 0)
    if kappa >= 0.70:
        print("✓ κ ≥ 0.70: Proceed with judge scores as primary measure")
    else:
        print("✗ κ < 0.70: Investigate disagreements before using judge scores")


if __name__ == "__main__":
    main()
