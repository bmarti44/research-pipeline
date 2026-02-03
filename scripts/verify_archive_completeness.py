#!/usr/bin/env python3
"""
Verify Zenodo archive completeness.

Checks that all required files are present and valid before publication.

Usage:
    python scripts/verify_archive_completeness.py [archive_dir]
"""

import json
import sys
from pathlib import Path


def verify_archive(archive_dir: str) -> dict:
    """Verify all required files are present and valid."""
    archive = Path(archive_dir)

    required_files = [
        "paper/FORMAT_FRICTION.md",
        "experiments/results/primary/signal_detection_20260203_074411_judged.json",
        "experiments/results/primary/signal_detection_20260203_121413.json",
        "experiments/results/primary/two_pass_sonnet_nl_20260203_125603.json",
        "experiments/results/primary/two_pass_qwen7b_nl_20260203_131141.json",
        "experiments/scenarios/signal_detection.py",
        "experiments/analyze_judged_results.py",
        "experiments/results/DATA_MANIFEST.md",
        "requirements.txt",
        "LICENSE",
        "README.md",
    ]

    optional_files = [
        "CITATION.cff",
        ".zenodo.json",
        "paper/LIMITATIONS.md",
        "paper/REVIEW.md",
        "experiments/judge_scoring.py",
        "experiments/two_pass_extraction.py",
        "experiments/remediation_analysis.py",
    ]

    results = {
        "archive_dir": str(archive),
        "missing_required": [],
        "missing_optional": [],
        "present": [],
        "errors": [],
    }

    # Check required files
    for file_path in required_files:
        full_path = archive / file_path
        if full_path.exists():
            results["present"].append(file_path)
            # Validate JSON files
            if file_path.endswith(".json"):
                try:
                    with open(full_path) as f:
                        data = json.load(f)
                    # Check for expected structure
                    if "results" in data or "metadata" in data:
                        pass  # Valid structure
                    else:
                        results["errors"].append(
                            f"{file_path}: JSON valid but unexpected structure"
                        )
                except json.JSONDecodeError as e:
                    results["errors"].append(f"{file_path}: Invalid JSON - {e}")
        else:
            results["missing_required"].append(file_path)

    # Check optional files
    for file_path in optional_files:
        full_path = archive / file_path
        if full_path.exists():
            results["present"].append(file_path)
        else:
            results["missing_optional"].append(file_path)

    return results


def print_report(results: dict) -> bool:
    """Print verification report. Returns True if archive is valid."""
    print("=" * 60)
    print("ZENODO ARCHIVE VERIFICATION REPORT")
    print("=" * 60)
    print(f"\nArchive: {results['archive_dir']}")

    print(f"\nRequired files present: {len(results['present'])} / "
          f"{len(results['present']) + len(results['missing_required'])}")

    if results["missing_required"]:
        print("\nMISSING REQUIRED FILES:")
        for f in results["missing_required"]:
            print(f"  [FAIL] {f}")
    else:
        print("\nAll required files present.")

    if results["missing_optional"]:
        print("\nMissing optional files:")
        for f in results["missing_optional"]:
            print(f"  [WARN] {f}")

    if results["errors"]:
        print("\nERRORS:")
        for e in results["errors"]:
            print(f"  [ERROR] {e}")

    print("\n" + "=" * 60)

    is_valid = (
        len(results["missing_required"]) == 0 and len(results["errors"]) == 0
    )

    if is_valid:
        print("RESULT: Archive is VALID for Zenodo publication")
    else:
        print("RESULT: Archive has ISSUES that need to be fixed")

    return is_valid


def main():
    if len(sys.argv) > 1:
        archive_dir = sys.argv[1]
    else:
        # Default: check current directory structure (not archive)
        archive_dir = "."

    results = verify_archive(archive_dir)
    is_valid = print_report(results)

    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
