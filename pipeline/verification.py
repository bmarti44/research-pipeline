"""
Verification system for pipeline stages.

Each stage has a verification gate that must pass before proceeding.
Verification checks produce deterministic, documented results.
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class CheckResult:
    """Result of a single verification check."""
    name: str
    passed: bool
    message: str
    details: Optional[dict] = None


@dataclass
class VerificationResult:
    """Result of verifying a stage."""
    stage: str
    study: str
    passed: bool
    checks: list[CheckResult]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    inputs_hash: Optional[str] = None
    outputs_hash: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "stage": self.stage,
            "study": self.study,
            "passed": self.passed,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "message": c.message,
                    "details": c.details,
                }
                for c in self.checks
            ],
            "timestamp": self.timestamp,
            "inputs_hash": self.inputs_hash,
            "outputs_hash": self.outputs_hash,
        }

    def save(self, path: Path) -> None:
        """Save verification result to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def hash_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def hash_directory(path: Path, exclude: Optional[list[str]] = None) -> str:
    """
    Compute deterministic hash of directory contents.

    Files are sorted by path for determinism.
    """
    exclude = exclude or []
    hasher = hashlib.sha256()

    for file in sorted(path.rglob("*")):
        if file.is_file():
            # Skip excluded patterns
            if any(ex in str(file) for ex in exclude):
                continue
            # Include relative path in hash for structure
            rel_path = file.relative_to(path)
            hasher.update(str(rel_path).encode())
            hasher.update(file.read_bytes())

    return f"sha256:{hasher.hexdigest()}"


# Stage verification functions

def verify_configure(study_path: Path) -> VerificationResult:
    """Verify the configure stage outputs."""
    checks = []
    stage_path = study_path / "stages" / "1_configure"

    # Check config_resolved.yaml exists
    config_path = stage_path / "config_resolved.yaml"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            checks.append(CheckResult(
                name="config_resolved_exists",
                passed=True,
                message="config_resolved.yaml exists and is valid YAML",
            ))

            # Check model versions are locked (no aliases)
            models = config.get("models", [])
            aliases = ["claude-sonnet", "claude-haiku", "gpt-4o", "gpt-4o-mini", "gemini-flash", "gemini-pro"]
            unlocked = [m.get("model_id") for m in models if m.get("model_id") in aliases]
            if unlocked:
                checks.append(CheckResult(
                    name="models_locked",
                    passed=False,
                    message=f"Model versions not locked: {unlocked}",
                ))
            else:
                checks.append(CheckResult(
                    name="models_locked",
                    passed=True,
                    message="All model versions are locked",
                ))

        except yaml.YAMLError as e:
            checks.append(CheckResult(
                name="config_resolved_exists",
                passed=False,
                message=f"config_resolved.yaml is invalid YAML: {e}",
            ))
    else:
        checks.append(CheckResult(
            name="config_resolved_exists",
            passed=False,
            message="config_resolved.yaml does not exist",
        ))

    # Check environment_lock.json exists
    env_path = stage_path / "environment_lock.json"
    if env_path.exists():
        try:
            with open(env_path) as f:
                env = json.load(f)
            required = ["python_version", "packages", "timestamp"]
            missing = [r for r in required if r not in env]
            if missing:
                checks.append(CheckResult(
                    name="environment_lock_exists",
                    passed=False,
                    message=f"environment_lock.json missing fields: {missing}",
                ))
            else:
                checks.append(CheckResult(
                    name="environment_lock_exists",
                    passed=True,
                    message="environment_lock.json exists with all required fields",
                ))
        except json.JSONDecodeError as e:
            checks.append(CheckResult(
                name="environment_lock_exists",
                passed=False,
                message=f"environment_lock.json is invalid JSON: {e}",
            ))
    else:
        checks.append(CheckResult(
            name="environment_lock_exists",
            passed=False,
            message="environment_lock.json does not exist",
        ))

    # Check seed is recorded
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        seed = config.get("trials", {}).get("seed")
        if seed is not None:
            checks.append(CheckResult(
                name="seed_recorded",
                passed=True,
                message=f"Seed recorded: {seed}",
            ))
        else:
            checks.append(CheckResult(
                name="seed_recorded",
                passed=False,
                message="No seed found in config",
            ))

    passed = all(c.passed for c in checks)

    return VerificationResult(
        stage="configure",
        study=study_path.name,
        passed=passed,
        checks=checks,
        outputs_hash=hash_directory(stage_path) if stage_path.exists() else None,
    )


def verify_generate(study_path: Path) -> VerificationResult:
    """Verify the generate stage outputs."""
    checks = []
    stage_path = study_path / "stages" / "2_generate"

    # Check trials.json exists
    trials_path = stage_path / "trials.json"
    if trials_path.exists():
        try:
            with open(trials_path) as f:
                trials = json.load(f)

            if isinstance(trials, list):
                checks.append(CheckResult(
                    name="trials_exists",
                    passed=True,
                    message=f"trials.json exists with {len(trials)} trials",
                ))

                # Check trial IDs are unique
                ids = [t.get("trial_id") for t in trials]
                if len(ids) == len(set(ids)):
                    checks.append(CheckResult(
                        name="trial_ids_unique",
                        passed=True,
                        message="All trial IDs are unique",
                    ))
                else:
                    checks.append(CheckResult(
                        name="trial_ids_unique",
                        passed=False,
                        message="Duplicate trial IDs found",
                    ))
            else:
                checks.append(CheckResult(
                    name="trials_exists",
                    passed=False,
                    message="trials.json is not a list",
                ))
        except json.JSONDecodeError as e:
            checks.append(CheckResult(
                name="trials_exists",
                passed=False,
                message=f"trials.json is invalid JSON: {e}",
            ))
    else:
        checks.append(CheckResult(
            name="trials_exists",
            passed=False,
            message="trials.json does not exist",
        ))

    # Check coverage report exists
    coverage_path = stage_path / "coverage_report.json"
    if coverage_path.exists():
        checks.append(CheckResult(
            name="coverage_report_exists",
            passed=True,
            message="coverage_report.json exists",
        ))
    else:
        checks.append(CheckResult(
            name="coverage_report_exists",
            passed=False,
            message="coverage_report.json does not exist",
        ))

    passed = all(c.passed for c in checks)

    return VerificationResult(
        stage="generate",
        study=study_path.name,
        passed=passed,
        checks=checks,
        outputs_hash=hash_directory(stage_path) if stage_path.exists() else None,
    )


def verify_execute(study_path: Path) -> VerificationResult:
    """Verify the execute stage outputs."""
    checks = []
    stage_path = study_path / "stages" / "3_execute"

    # Load trials to know expected count
    trials_path = study_path / "stages" / "2_generate" / "trials.json"
    expected_count = 0
    if trials_path.exists():
        with open(trials_path) as f:
            trials = json.load(f)
            expected_count = len(trials)

    # Check responses directory
    responses_path = stage_path / "responses"
    if responses_path.exists():
        response_files = list(responses_path.glob("trial_*.json"))

        if len(response_files) == expected_count:
            checks.append(CheckResult(
                name="all_trials_complete",
                passed=True,
                message=f"{len(response_files)}/{expected_count} trials have responses",
            ))
        else:
            checks.append(CheckResult(
                name="all_trials_complete",
                passed=False,
                message=f"Only {len(response_files)}/{expected_count} trials have responses",
            ))

        # Check response fields
        required_fields = ["response", "timestamp", "model"]
        missing_fields = []
        for rf in response_files[:10]:  # Sample first 10
            with open(rf) as f:
                resp = json.load(f)
            for field in required_fields:
                if field not in resp:
                    missing_fields.append(f"{rf.name}:{field}")

        if not missing_fields:
            checks.append(CheckResult(
                name="response_fields_present",
                passed=True,
                message="All sampled responses have required fields",
            ))
        else:
            checks.append(CheckResult(
                name="response_fields_present",
                passed=False,
                message=f"Missing fields: {missing_fields[:5]}...",
            ))
    else:
        checks.append(CheckResult(
            name="all_trials_complete",
            passed=False,
            message="responses/ directory does not exist",
        ))

    # Check execution log
    log_path = stage_path / "execution_log.json"
    if log_path.exists():
        checks.append(CheckResult(
            name="execution_log_exists",
            passed=True,
            message="execution_log.json exists",
        ))
    else:
        checks.append(CheckResult(
            name="execution_log_exists",
            passed=False,
            message="execution_log.json does not exist",
        ))

    passed = all(c.passed for c in checks)

    return VerificationResult(
        stage="execute",
        study=study_path.name,
        passed=passed,
        checks=checks,
        outputs_hash=hash_directory(stage_path) if stage_path.exists() else None,
    )


def verify_evaluate(study_path: Path) -> VerificationResult:
    """Verify the evaluate stage outputs."""
    checks = []
    stage_path = study_path / "stages" / "4_evaluate"

    # Check scores directory
    scores_path = stage_path / "scores"
    if scores_path.exists():
        score_files = list(scores_path.glob("trial_*.json"))
        checks.append(CheckResult(
            name="scores_exist",
            passed=len(score_files) > 0,
            message=f"{len(score_files)} score files found",
        ))

        # Check evaluation modes present
        if score_files:
            with open(score_files[0]) as f:
                sample = json.load(f)
            modes = sample.get("modes", {}).keys()
            expected_modes = {"strict", "intent", "functional"}
            missing = expected_modes - set(modes)
            if not missing:
                checks.append(CheckResult(
                    name="all_modes_present",
                    passed=True,
                    message="All evaluation modes present",
                ))
            else:
                checks.append(CheckResult(
                    name="all_modes_present",
                    passed=False,
                    message=f"Missing modes: {missing}",
                ))
    else:
        checks.append(CheckResult(
            name="scores_exist",
            passed=False,
            message="scores/ directory does not exist",
        ))

    # Check determinism check
    det_path = stage_path / "determinism_check.json"
    if det_path.exists():
        with open(det_path) as f:
            det = json.load(f)
        if det.get("passed", False):
            checks.append(CheckResult(
                name="determinism_check",
                passed=True,
                message="Determinism check passed",
            ))
        else:
            checks.append(CheckResult(
                name="determinism_check",
                passed=False,
                message=f"Determinism check failed: {det.get('message', 'unknown')}",
            ))
    else:
        checks.append(CheckResult(
            name="determinism_check",
            passed=False,
            message="determinism_check.json does not exist",
        ))

    passed = all(c.passed for c in checks)

    return VerificationResult(
        stage="evaluate",
        study=study_path.name,
        passed=passed,
        checks=checks,
        outputs_hash=hash_directory(stage_path) if stage_path.exists() else None,
    )


def verify_analyze(study_path: Path) -> VerificationResult:
    """Verify the analyze stage outputs."""
    checks = []
    stage_path = study_path / "stages" / "5_analyze"

    # Check aggregates.json
    agg_path = stage_path / "aggregates.json"
    if agg_path.exists():
        checks.append(CheckResult(
            name="aggregates_exist",
            passed=True,
            message="aggregates.json exists",
        ))
    else:
        checks.append(CheckResult(
            name="aggregates_exist",
            passed=False,
            message="aggregates.json does not exist",
        ))

    # Check tests.json
    tests_path = stage_path / "tests.json"
    if tests_path.exists():
        with open(tests_path) as f:
            tests = json.load(f)
        checks.append(CheckResult(
            name="tests_run",
            passed=True,
            message=f"{len(tests)} statistical tests run",
        ))
    else:
        checks.append(CheckResult(
            name="tests_run",
            passed=False,
            message="tests.json does not exist",
        ))

    # Check assumptions.json
    assumptions_path = stage_path / "assumptions.json"
    if assumptions_path.exists():
        checks.append(CheckResult(
            name="assumptions_documented",
            passed=True,
            message="assumptions.json exists",
        ))
    else:
        checks.append(CheckResult(
            name="assumptions_documented",
            passed=False,
            message="assumptions.json does not exist",
        ))

    passed = all(c.passed for c in checks)

    return VerificationResult(
        stage="analyze",
        study=study_path.name,
        passed=passed,
        checks=checks,
        outputs_hash=hash_directory(stage_path) if stage_path.exists() else None,
    )


def verify_report(study_path: Path) -> VerificationResult:
    """Verify the report stage outputs."""
    checks = []
    outputs_path = study_path / "outputs"

    # Check RESULTS.md exists and is substantial
    results_path = outputs_path / "RESULTS.md"
    if results_path.exists():
        size = results_path.stat().st_size
        if size > 500:  # Lowered threshold for small/pilot studies
            checks.append(CheckResult(
                name="results_exists",
                passed=True,
                message=f"RESULTS.md exists ({size} bytes)",
            ))
        else:
            checks.append(CheckResult(
                name="results_exists",
                passed=False,
                message=f"RESULTS.md too small ({size} bytes)",
            ))
    else:
        checks.append(CheckResult(
            name="results_exists",
            passed=False,
            message="RESULTS.md does not exist",
        ))

    # Check archive exists
    archive_path = outputs_path / "archive.zip"
    if archive_path.exists():
        checks.append(CheckResult(
            name="archive_exists",
            passed=True,
            message="archive.zip exists",
        ))
    else:
        checks.append(CheckResult(
            name="archive_exists",
            passed=False,
            message="archive.zip does not exist",
        ))

    passed = all(c.passed for c in checks)

    return VerificationResult(
        stage="report",
        study=study_path.name,
        passed=passed,
        checks=checks,
        outputs_hash=hash_directory(outputs_path) if outputs_path.exists() else None,
    )


# Stage verification dispatcher
STAGE_VERIFIERS = {
    "configure": verify_configure,
    "generate": verify_generate,
    "execute": verify_execute,
    "evaluate": verify_evaluate,
    "analyze": verify_analyze,
    "report": verify_report,
}

STAGE_NUMBERS = {
    "configure": 1,
    "generate": 2,
    "execute": 3,
    "evaluate": 4,
    "analyze": 5,
    "report": 6,
}


def verify_stage(study_path: Path, stage: str) -> VerificationResult:
    """
    Verify a specific stage.

    Args:
        study_path: Path to study directory
        stage: Stage name (configure, generate, execute, evaluate, analyze, report)

    Returns:
        VerificationResult with all checks
    """
    if stage not in STAGE_VERIFIERS:
        return VerificationResult(
            stage=stage,
            study=study_path.name,
            passed=False,
            checks=[CheckResult(
                name="valid_stage",
                passed=False,
                message=f"Unknown stage: {stage}. Valid: {list(STAGE_VERIFIERS.keys())}",
            )],
        )

    return STAGE_VERIFIERS[stage](study_path)


def verify_all_stages(study_path: Path) -> dict[str, VerificationResult]:
    """Verify all stages for a study."""
    results = {}
    for stage in STAGE_VERIFIERS:
        results[stage] = verify_stage(study_path, stage)
    return results
