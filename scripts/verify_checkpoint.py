#!/usr/bin/env python
"""
Verify checkpoint files and their integrity.

This script validates that:
1. All required checkpoint files exist
2. Checksums in checkpoint files match actual files
3. All phases have passed or been skipped appropriately

Usage:
    python scripts/verify_checkpoint.py [--phase N] [--strict]
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.core.checkpoint import (
    read_checkpoint,
    compute_file_sha256,
    get_all_checkpoints,
)


def verify_single_checkpoint(phase: int, strict: bool = False) -> tuple[bool, list[str]]:
    """
    Verify a single checkpoint.

    Args:
        phase: Phase number to verify
        strict: If True, treat warnings as errors

    Returns:
        Tuple of (passed, list of messages)
    """
    messages = []
    passed = True

    checkpoint = read_checkpoint(phase)
    if checkpoint is None:
        messages.append(f"Phase {phase}: checkpoint not found")
        return False, messages

    status = checkpoint.get("status", "unknown")
    phase_name = checkpoint.get("phase_name", "unknown")

    if status == "passed":
        messages.append(f"Phase {phase} ({phase_name}): PASSED")
    elif status == "skipped":
        messages.append(f"Phase {phase} ({phase_name}): SKIPPED")
    elif status == "failed":
        messages.append(f"Phase {phase} ({phase_name}): FAILED")
        passed = False
    else:
        messages.append(f"Phase {phase} ({phase_name}): UNKNOWN ({status})")
        if strict:
            passed = False

    # Verify input checksums
    inputs = checkpoint.get("inputs_sha256", {})
    for file_path, expected_checksum in inputs.items():
        if expected_checksum is None:
            continue
        if not Path(file_path).exists():
            messages.append(f"  Input file missing: {file_path}")
            if strict:
                passed = False
            continue
        actual = compute_file_sha256(file_path)
        if actual != expected_checksum:
            messages.append(
                f"  Input checksum MISMATCH: {file_path}\n"
                f"    Expected: {expected_checksum}\n"
                f"    Actual:   {actual}"
            )
            passed = False

    # Verify output checksums
    outputs = checkpoint.get("outputs_sha256", {})
    for file_path, expected_checksum in outputs.items():
        if expected_checksum is None:
            continue
        if not Path(file_path).exists():
            messages.append(f"  Output file missing: {file_path}")
            if strict:
                passed = False
            continue
        actual = compute_file_sha256(file_path)
        if actual != expected_checksum:
            messages.append(
                f"  Output checksum MISMATCH: {file_path}\n"
                f"    Expected: {expected_checksum}\n"
                f"    Actual:   {actual}"
            )
            # Output mismatches are warnings (file may have been regenerated)
            if strict:
                passed = False

    return passed, messages


def verify_all_checkpoints(strict: bool = False) -> tuple[bool, list[str]]:
    """
    Verify all checkpoints.

    Args:
        strict: If True, treat warnings as errors

    Returns:
        Tuple of (all_passed, list of messages)
    """
    all_passed = True
    all_messages = []

    for phase in range(-1, 9):
        passed, messages = verify_single_checkpoint(phase, strict)
        all_messages.extend(messages)
        if not passed:
            all_passed = False

    return all_passed, all_messages


def verify_preregistration_lock() -> tuple[bool, list[str]]:
    """
    Verify pre-registration lock file.

    Returns:
        Tuple of (passed, list of messages)
    """
    lock_path = Path("verification/preregistration_lock.json")
    messages = []
    passed = True

    if not lock_path.exists():
        messages.append("Pre-registration lock file not found")
        return False, messages

    with open(lock_path) as f:
        locks = json.load(f)

    for file_path, expected_checksum in locks.items():
        if expected_checksum is None:
            messages.append(f"  {file_path}: not locked (null)")
            continue

        if not Path(file_path).exists():
            messages.append(f"  {file_path}: MISSING")
            passed = False
            continue

        actual = compute_file_sha256(file_path)
        if actual == expected_checksum:
            messages.append(f"  {file_path}: OK")
        else:
            messages.append(
                f"  {file_path}: MODIFIED\n"
                f"    Locked:  {expected_checksum}\n"
                f"    Actual:  {actual}"
            )
            passed = False

    return passed, messages


def main():
    parser = argparse.ArgumentParser(
        description="Verify checkpoint files and integrity"
    )
    parser.add_argument(
        "--phase",
        type=int,
        help="Verify specific phase only",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )
    parser.add_argument(
        "--lock-only",
        action="store_true",
        help="Only verify pre-registration lock",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Checkpoint Verification")
    print("=" * 60)
    print()

    exit_code = 0

    if args.lock_only:
        print("Pre-registration Lock Verification:")
        passed, messages = verify_preregistration_lock()
        for msg in messages:
            print(msg)
        if not passed:
            exit_code = 1
    elif args.phase is not None:
        passed, messages = verify_single_checkpoint(args.phase, args.strict)
        for msg in messages:
            print(msg)
        if not passed:
            exit_code = 1
    else:
        # Verify all checkpoints
        print("Phase Checkpoints:")
        passed, messages = verify_all_checkpoints(args.strict)
        for msg in messages:
            print(msg)
        if not passed:
            exit_code = 1

        print()
        print("Pre-registration Lock:")
        lock_passed, lock_messages = verify_preregistration_lock()
        for msg in lock_messages:
            print(msg)
        if not lock_passed:
            exit_code = 1

    print()
    print("=" * 60)
    if exit_code == 0:
        print("Verification: PASSED")
    else:
        print("Verification: FAILED")
    print("=" * 60)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
