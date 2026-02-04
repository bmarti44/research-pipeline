"""
Checkpoint management for phase verification.

Provides utilities for writing, reading, and verifying checkpoint files.
Each phase creates a checkpoint JSON file with:
- Phase number and name
- Status (passed/failed/skipped)
- Input/output file SHA256 checksums
- Phase-specific metrics
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


CHECKPOINT_DIR = Path("verification")


def compute_file_sha256(file_path: str | Path) -> str:
    """
    Compute SHA256 checksum of a file.

    Args:
        file_path: Path to the file

    Returns:
        Hexadecimal SHA256 checksum string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def compute_content_sha256(content: str | bytes) -> str:
    """
    Compute SHA256 checksum of content.

    Args:
        content: String or bytes to hash

    Returns:
        Hexadecimal SHA256 checksum string
    """
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def verify_checksum(
    file_path: str | Path, expected_checksum: str
) -> tuple[bool, str]:
    """
    Verify a file's checksum matches the expected value.

    Args:
        file_path: Path to the file
        expected_checksum: Expected SHA256 checksum

    Returns:
        Tuple of (match, actual_checksum)
    """
    actual = compute_file_sha256(file_path)
    return actual == expected_checksum, actual


def verify_checksum_from_lock(
    file_path: str | Path, lock_file: str | Path = "verification/preregistration_lock.json"
) -> tuple[bool, str, str]:
    """
    Verify a file's checksum against the pre-registration lock file.

    Args:
        file_path: Path to the file to verify
        lock_file: Path to the lock file containing expected checksums

    Returns:
        Tuple of (match, expected_checksum, actual_checksum)
    """
    file_path = str(file_path)
    with open(lock_file) as f:
        locks = json.load(f)

    expected = locks.get(file_path, "")
    actual = compute_file_sha256(file_path) if os.path.exists(file_path) else ""
    return expected == actual, expected, actual


def write_checkpoint(
    phase: int,
    phase_name: str,
    status: str,
    inputs_sha256: Optional[dict[str, str]] = None,
    outputs_sha256: Optional[dict[str, str]] = None,
    metrics: Optional[dict[str, Any]] = None,
    run_id: Optional[str] = None,
    seed: Optional[int] = None,
) -> Path:
    """
    Write a checkpoint file for a phase.

    Args:
        phase: Phase number (-1 to 8)
        phase_name: Human-readable phase name
        status: Phase status ('passed', 'failed', 'skipped')
        inputs_sha256: Dictionary of input file paths to their checksums
        outputs_sha256: Dictionary of output file paths to their checksums
        metrics: Dictionary of phase-specific metrics
        run_id: Optional deterministic run ID
        seed: Optional random seed used

    Returns:
        Path to the written checkpoint file
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "phase": phase,
        "phase_name": phase_name,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if inputs_sha256:
        checkpoint["inputs_sha256"] = inputs_sha256

    if outputs_sha256:
        checkpoint["outputs_sha256"] = outputs_sha256

    if run_id:
        checkpoint["run_id"] = run_id

    if seed is not None:
        checkpoint["seed"] = seed

    if metrics:
        checkpoint["metrics"] = metrics

    checkpoint_path = CHECKPOINT_DIR / f"checkpoint_{phase}.json"

    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f, indent=2)

    return checkpoint_path


def read_checkpoint(phase: int) -> Optional[dict]:
    """
    Read a checkpoint file for a phase.

    Args:
        phase: Phase number (-1 to 8)

    Returns:
        Checkpoint dictionary or None if not found
    """
    checkpoint_path = CHECKPOINT_DIR / f"checkpoint_{phase}.json"

    if not checkpoint_path.exists():
        return None

    with open(checkpoint_path) as f:
        return json.load(f)


def get_all_checkpoints() -> dict[int, dict]:
    """
    Read all checkpoint files.

    Returns:
        Dictionary mapping phase numbers to checkpoint data
    """
    checkpoints = {}
    for phase in range(-1, 9):
        checkpoint = read_checkpoint(phase)
        if checkpoint:
            checkpoints[phase] = checkpoint
    return checkpoints


def compute_run_id(
    scenario_file: str | Path,
    seed: int,
    n_trials_per_scenario: int,
    conditions: list[str],
    model_name: str,
) -> str:
    """
    Compute deterministic run ID from experiment configuration.

    The run ID is the first 8 characters of SHA256 hash of the
    configuration parameters, ensuring stable filenames across
    re-runs with identical inputs.

    Args:
        scenario_file: Path to scenario definition file
        seed: Random seed
        n_trials_per_scenario: Number of trials per scenario
        conditions: List of experimental conditions
        model_name: Name of the model being tested

    Returns:
        8-character run ID string
    """
    scenario_content = ""
    if os.path.exists(scenario_file):
        with open(scenario_file, "rb") as f:
            scenario_content = f.read()

    config_string = f"{scenario_content}{seed}{n_trials_per_scenario}{sorted(conditions)}{model_name}"
    full_hash = hashlib.sha256(config_string.encode()).hexdigest()
    return full_hash[:8]


def create_preregistration_lock(
    files: list[str | Path],
    output_path: str | Path = "verification/preregistration_lock.json",
) -> Path:
    """
    Create a pre-registration lock file with checksums of specified files.

    Args:
        files: List of file paths to lock
        output_path: Path to write the lock file

    Returns:
        Path to the written lock file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    locks = {}
    for file_path in files:
        file_path = str(file_path)
        if os.path.exists(file_path):
            locks[file_path] = compute_file_sha256(file_path)
        else:
            locks[file_path] = None

    with open(output_path, "w") as f:
        json.dump(locks, f, indent=2)

    return output_path
