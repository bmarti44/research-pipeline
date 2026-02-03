"""
Generate Stratified Validation Sample for Human Annotation

This script extracts a stratified random sample from judged results
for human annotation to validate the LLM judge's reliability.

Usage:
    python experiments/generate_validation_sample.py experiments/results/signal_detection_{timestamp}_judged.json
"""

import json
import csv
import random
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Optional


# Stratification targets
STRATA_TARGETS = {
    ("EXPLICIT", "nl"): 15,
    ("EXPLICIT", "structured"): 15,
    ("IMPLICIT", "nl"): 25,
    ("IMPLICIT", "structured"): 25,
    ("BORDERLINE", "nl"): 15,
    ("BORDERLINE", "structured"): 15,
    ("CONTROL", "nl"): 20,
    ("CONTROL", "structured"): 20,
}


def load_judged_results(path: Path) -> list[dict]:
    """Load judged results file."""
    with open(path) as f:
        data = json.load(f)
    return data.get("results", [])


def stratify_results(results: list[dict]) -> dict[tuple, list[dict]]:
    """Group results by (ambiguity_level, condition) strata."""
    strata = defaultdict(list)

    for trial in results:
        ambiguity = trial.get("ambiguity", "UNKNOWN")

        # Each merged trial has both NL and structured responses
        # Create separate entries for each condition

        # NL entry
        nl_entry = {
            "scenario_id": trial.get("scenario_id"),
            "ambiguity_level": ambiguity,
            "signal_type": trial.get("signal_type"),
            "condition": "nl",
            "query": trial.get("query"),
            "response": trial.get("nl_response_text") or trial.get("response_text"),
            "trial_number": trial.get("trial_number"),
            "judge_detected": trial.get("nl_judge_detected"),
            "regex_detected": trial.get("nl_regex_detected"),
        }
        strata[(ambiguity, "nl")].append(nl_entry)

        # Structured entry
        st_entry = {
            "scenario_id": trial.get("scenario_id"),
            "ambiguity_level": ambiguity,
            "signal_type": trial.get("signal_type"),
            "condition": "structured",
            "query": trial.get("query"),
            "response": trial.get("st_response_text"),
            "trial_number": trial.get("trial_number"),
            "judge_detected": trial.get("st_judge_detected"),
            "regex_detected": trial.get("st_regex_detected"),
        }
        if st_entry["response"]:  # Only add if we have a response
            strata[(ambiguity, "structured")].append(st_entry)

    return dict(strata)


def sample_strata(
    strata: dict[tuple, list[dict]],
    seed: int = 42
) -> tuple[list[dict], dict[str, int]]:
    """Sample from each stratum according to targets.

    Returns:
        Tuple of (sampled entries, actual counts per stratum)
    """
    random.seed(seed)

    sampled = []
    actual_counts = {}

    for stratum_key, target in STRATA_TARGETS.items():
        available = strata.get(stratum_key, [])

        if len(available) <= target:
            # Take all if fewer than target
            selected = available
        else:
            # Random sample
            selected = random.sample(available, target)

        sampled.extend(selected)
        actual_counts[f"{stratum_key[0]}_{stratum_key[1]}"] = len(selected)

    return sampled, actual_counts


def generate_sample_ids(samples: list[dict]) -> list[dict]:
    """Add unique sample IDs and shuffle for blind annotation."""
    # Shuffle to randomize order (annotator shouldn't see condition blocks)
    random.shuffle(samples)

    # Add sample IDs
    for i, sample in enumerate(samples, 1):
        sample["sample_id"] = f"S{i:03d}"

    return samples


def write_annotation_csv(samples: list[dict], output_path: Path) -> None:
    """Write the annotation CSV (without judge/regex results - blind annotation)."""
    fieldnames = [
        "sample_id",
        "scenario_id",
        "ambiguity_level",
        "signal_type",
        "condition",
        "query",
        "response",
        "human_label",  # Empty - annotator fills this in
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for sample in samples:
            row = {
                "sample_id": sample["sample_id"],
                "scenario_id": sample["scenario_id"],
                "ambiguity_level": sample["ambiguity_level"],
                "signal_type": sample["signal_type"] or "none",
                "condition": sample["condition"],
                "query": sample["query"],
                "response": sample["response"],
                "human_label": "",  # Empty for annotator
            }
            writer.writerow(row)


def write_key_file(samples: list[dict], output_path: Path) -> None:
    """Write the key file with judge and regex results (for computing agreement)."""
    fieldnames = [
        "sample_id",
        "scenario_id",
        "trial_number",
        "ambiguity_level",
        "condition",
        "judge_detected",
        "regex_detected",
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for sample in samples:
            row = {
                "sample_id": sample["sample_id"],
                "scenario_id": sample["scenario_id"],
                "trial_number": sample["trial_number"],
                "ambiguity_level": sample["ambiguity_level"],
                "condition": sample["condition"],
                "judge_detected": sample["judge_detected"],
                "regex_detected": sample["regex_detected"],
            }
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Generate stratified validation sample for human annotation"
    )
    parser.add_argument(
        "judged_file",
        type=Path,
        help="Path to judged results JSON file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as input file)"
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading judged results from {args.judged_file}")
    results = load_judged_results(args.judged_file)
    print(f"Found {len(results)} trial records")

    # Stratify
    strata = stratify_results(results)
    print("\nAvailable samples per stratum:")
    for key, samples in sorted(strata.items()):
        target = STRATA_TARGETS.get(key, 0)
        print(f"  {key[0]:12} + {key[1]:10}: {len(samples):4} available, {target:3} target")

    # Sample
    sampled, counts = sample_strata(strata, args.seed)
    print(f"\nSampled {len(sampled)} total responses")

    # Add IDs and shuffle
    sampled = generate_sample_ids(sampled)

    # Determine output paths
    output_dir = args.output_dir or args.judged_file.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    annotation_path = output_dir / f"validation_annotation_{timestamp}.csv"
    key_path = output_dir / f"validation_key_{timestamp}.csv"

    # Write files
    write_annotation_csv(sampled, annotation_path)
    write_key_file(sampled, key_path)

    print(f"\nOutput files:")
    print(f"  Annotation CSV (for human): {annotation_path}")
    print(f"  Key file (for agreement):   {key_path}")

    print("\nActual samples per stratum:")
    for stratum, count in sorted(counts.items()):
        print(f"  {stratum}: {count}")

    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Have a human annotator fill in 'human_label' column in:")
    print(f"   {annotation_path}")
    print("2. Use YES or NO for each response (same criteria as judge)")
    print("3. Save the annotated file")
    print("4. Run: python experiments/compute_agreement.py <annotated.csv> <key.csv>")


if __name__ == "__main__":
    main()
