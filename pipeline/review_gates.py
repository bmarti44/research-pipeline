"""
Automated review gates for the research pipeline.

Implements stage-gated verification with deterministic checks.
Each gate must pass before proceeding to the next stage.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable
import yaml

from .verification import (
    CheckResult,
    VerificationResult,
    hash_file,
    hash_directory,
)
from .preregistration import (
    check_preregistration_exists,
    verify_preregistration,
)
from .pilot import check_pilot_gate, evaluate_pilot, PilotResult
from .adaptive import should_stop, get_adjusted_alpha
from .interview import verify_interview_complete


@dataclass
class GateResult:
    """Result of a review gate check."""
    gate_name: str
    passed: bool
    blocking: bool  # If True, pipeline cannot proceed
    checks: list[CheckResult]
    recommendation: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "gate_name": self.gate_name,
            "passed": self.passed,
            "blocking": self.blocking,
            "checks": [
                {"name": c.name, "passed": c.passed, "message": c.message}
                for c in self.checks
            ],
            "recommendation": self.recommendation,
            "timestamp": self.timestamp,
        }


def check_interview_gate(study_path: Path, config: dict) -> GateResult:
    """
    Gate: Interview must be completed before research planning.

    Blocks: research plan creation if interview not done.
    """
    checks = []

    passed, issues = verify_interview_complete(study_path)

    if passed:
        checks.append(CheckResult(
            name="interview_complete",
            passed=True,
            message="Interview completed with all required questions answered",
        ))
    else:
        for issue in issues:
            checks.append(CheckResult(
                name="interview_complete",
                passed=False,
                message=issue,
            ))

    return GateResult(
        gate_name="interview",
        passed=passed,
        blocking=True,
        checks=checks,
        recommendation="Proceed to research planning" if passed else "Complete the hypothesis interview first",
    )


def check_preregistration_gate(study_path: Path, config: dict) -> GateResult:
    """
    Gate: Study must be preregistered before data collection.

    Blocks: execute stage if preregistration is missing or modified.
    """
    checks = []

    # Check preregistration exists
    has_prereg = check_preregistration_exists(study_path)
    checks.append(CheckResult(
        name="preregistration_exists",
        passed=has_prereg,
        message="Preregistration found" if has_prereg else "No preregistration found",
    ))

    if has_prereg:
        # Verify integrity
        verification = verify_preregistration(study_path)
        checks.append(CheckResult(
            name="preregistration_intact",
            passed=verification.passed,
            message="Preregistration unchanged" if verification.passed else f"Modified: {verification.deviations[:2]}",
            details={"deviations": verification.deviations} if not verification.passed else None,
        ))
    else:
        checks.append(CheckResult(
            name="preregistration_intact",
            passed=False,
            message="Cannot verify - no preregistration",
        ))

    passed = all(c.passed for c in checks)

    return GateResult(
        gate_name="preregistration",
        passed=passed,
        blocking=True,  # Cannot proceed without preregistration
        checks=checks,
        recommendation="Proceed" if passed else "Create preregistration before data collection",
    )


def check_pilot_gate(study_path: Path, config: dict) -> GateResult:
    """
    Gate: Pilot must pass before main study.

    Blocks: main study execution if pilot not completed or failed.
    """
    checks = []

    # Check if pilot is required
    pilot_required = config.get("pilot", {}).get("required", True)

    if not pilot_required:
        return GateResult(
            gate_name="pilot",
            passed=True,
            blocking=False,
            checks=[CheckResult(
                name="pilot_required",
                passed=True,
                message="Pilot not required for this study",
            )],
            recommendation="Proceed without pilot",
        )

    # Check pilot exists and passed
    pilot_path = study_path / "pilot"
    pilot_result_path = pilot_path / "pilot_result.json"

    if not pilot_result_path.exists():
        checks.append(CheckResult(
            name="pilot_completed",
            passed=False,
            message="Pilot study not completed",
        ))
        return GateResult(
            gate_name="pilot",
            passed=False,
            blocking=True,
            checks=checks,
            recommendation="Run pilot study first: python -m pipeline run --pilot",
        )

    # Load pilot result
    with open(pilot_result_path) as f:
        pilot_result = json.load(f)

    recommendation = pilot_result.get("recommendation", "unknown")

    checks.append(CheckResult(
        name="pilot_completed",
        passed=True,
        message="Pilot study completed",
    ))

    passed = recommendation in ["proceed", "adjust"]
    checks.append(CheckResult(
        name="pilot_passed",
        passed=passed,
        message=f"Pilot recommendation: {recommendation}",
        details=pilot_result.get("criteria_met"),
    ))

    return GateResult(
        gate_name="pilot",
        passed=passed,
        blocking=True,
        checks=checks,
        recommendation="Proceed to main study" if passed else "Address pilot issues before proceeding",
    )


def check_execution_integrity_gate(study_path: Path, config: dict) -> GateResult:
    """
    Gate: Execution must complete without critical errors.

    Blocks: evaluate stage if execution is incomplete or has errors.
    """
    checks = []

    # Check execution log
    log_path = study_path / "stages" / "3_execute" / "execution_log.json"
    if not log_path.exists():
        return GateResult(
            gate_name="execution_integrity",
            passed=False,
            blocking=True,
            checks=[CheckResult(
                name="execution_log",
                passed=False,
                message="No execution log found",
            )],
            recommendation="Run execute stage first",
        )

    with open(log_path) as f:
        log = json.load(f)

    # Check completion rate
    completed = log.get("completed", 0)
    failed = log.get("failed", 0)
    total = log.get("total_trials", 1)

    completion_rate = completed / total if total > 0 else 0
    error_rate = failed / total if total > 0 else 0

    checks.append(CheckResult(
        name="completion_rate",
        passed=completion_rate >= 0.95,
        message=f"Completion rate: {completion_rate:.1%} ({completed}/{total})",
    ))

    checks.append(CheckResult(
        name="error_rate",
        passed=error_rate <= 0.05,
        message=f"Error rate: {error_rate:.1%} ({failed}/{total})",
    ))

    # Check model consistency
    responses_path = study_path / "stages" / "3_execute" / "responses"
    if responses_path.exists():
        models_used = set()
        for rf in list(responses_path.glob("trial_*.json"))[:20]:
            with open(rf) as f:
                resp = json.load(f)
            models_used.add(resp.get("model", "unknown"))

        # Should be consistent (unless multi-model study)
        expected_models = len(config.get("models", [{}]))
        checks.append(CheckResult(
            name="model_consistency",
            passed=len(models_used) <= max(1, expected_models),
            message=f"Models used: {models_used}",
        ))

    passed = all(c.passed for c in checks)

    return GateResult(
        gate_name="execution_integrity",
        passed=passed,
        blocking=True,
        checks=checks,
        recommendation="Proceed to evaluation" if passed else "Address execution issues",
    )


def check_evaluation_determinism_gate(study_path: Path, config: dict) -> GateResult:
    """
    Gate: Evaluation must be deterministic.

    Blocks: analyze stage if evaluation is not reproducible.
    """
    checks = []

    det_path = study_path / "stages" / "4_evaluate" / "determinism_check.json"
    if not det_path.exists():
        return GateResult(
            gate_name="evaluation_determinism",
            passed=False,
            blocking=True,
            checks=[CheckResult(
                name="determinism_check",
                passed=False,
                message="No determinism check found",
            )],
            recommendation="Run evaluate stage first",
        )

    with open(det_path) as f:
        det = json.load(f)

    passed = det.get("passed", False)
    checks.append(CheckResult(
        name="determinism",
        passed=passed,
        message=det.get("message", "Unknown"),
        details={"mismatches": det.get("mismatches", [])} if not passed else None,
    ))

    return GateResult(
        gate_name="evaluation_determinism",
        passed=passed,
        blocking=True,
        checks=checks,
        recommendation="Proceed to analysis" if passed else "Fix non-deterministic evaluation",
    )


def check_preregistration_compliance_gate(study_path: Path, config: dict) -> GateResult:
    """
    Gate: Analysis must match preregistration.

    Blocks: report stage if analysis deviates from preregistered plan.
    """
    checks = []

    # Verify preregistration
    verification = verify_preregistration(study_path)

    checks.append(CheckResult(
        name="preregistration_compliance",
        passed=verification.passed,
        message="Analysis matches preregistration" if verification.passed else "Deviations detected",
        details={"deviations": verification.deviations} if verification.deviations else None,
    ))

    # Check that all preregistered analyses were run
    prereg_path = study_path / "preregistration.json"
    tests_path = study_path / "stages" / "5_analyze" / "tests.json"

    if prereg_path.exists() and tests_path.exists():
        with open(prereg_path) as f:
            prereg = json.load(f)
        with open(tests_path) as f:
            tests = json.load(f)

        planned_analyses = prereg.get("analysis_plan", [])
        conducted_tests = [t.get("name", "") for t in tests]

        # Check coverage (rough match)
        all_planned_run = True
        for planned in planned_analyses:
            if not any(planned.lower() in t.lower() for t in conducted_tests):
                all_planned_run = False
                break

        checks.append(CheckResult(
            name="all_analyses_run",
            passed=all_planned_run,
            message="All preregistered analyses conducted" if all_planned_run else "Some planned analyses missing",
        ))

    passed = all(c.passed for c in checks)

    return GateResult(
        gate_name="preregistration_compliance",
        passed=passed,
        blocking=False,  # Warning, not blocking (deviations must be reported)
        checks=checks,
        recommendation="Proceed" if passed else "Document all deviations in report",
    )


def check_adaptive_stopping_gate(study_path: Path, config: dict) -> GateResult:
    """
    Gate: Check if adaptive stopping rules triggered.

    Informs but does not block (stopping is handled elsewhere).
    """
    checks = []

    stop, reason, interim = should_stop(study_path, config)

    if interim:
        checks.append(CheckResult(
            name="stopping_decision",
            passed=True,  # Gate itself passes
            message=f"Decision: {interim.decision} - {reason}",
            details={
                "n_completed": interim.n_completed,
                "effect_size": interim.effect_size,
                "p_value": interim.p_value,
            },
        ))
    else:
        checks.append(CheckResult(
            name="stopping_decision",
            passed=True,
            message=reason,
        ))

    return GateResult(
        gate_name="adaptive_stopping",
        passed=True,  # This gate is informational
        blocking=False,
        checks=checks,
        recommendation="Stop data collection" if stop else "Continue data collection",
    )


# Gate registry
GATES = {
    "pre_planning": [
        ("interview", check_interview_gate),
    ],
    "pre_execute": [
        ("preregistration", check_preregistration_gate),
        ("pilot", check_pilot_gate),
    ],
    "pre_evaluate": [
        ("execution_integrity", check_execution_integrity_gate),
    ],
    "pre_analyze": [
        ("evaluation_determinism", check_evaluation_determinism_gate),
    ],
    "pre_report": [
        ("preregistration_compliance", check_preregistration_compliance_gate),
    ],
    "during_execute": [
        ("adaptive_stopping", check_adaptive_stopping_gate),
    ],
}


def run_gates(
    study_path: Path,
    config: dict,
    stage: str,
) -> tuple[bool, list[GateResult]]:
    """
    Run all gates for a given stage.

    Returns (can_proceed, gate_results).
    """
    gate_key = f"pre_{stage}"
    if gate_key not in GATES:
        return True, []

    results = []
    can_proceed = True

    for gate_name, gate_fn in GATES[gate_key]:
        result = gate_fn(study_path, config)
        results.append(result)

        if not result.passed and result.blocking:
            can_proceed = False

    return can_proceed, results


def save_gate_results(study_path: Path, stage: str, results: list[GateResult]) -> None:
    """Save gate results to disk."""
    gates_path = study_path / "stages" / "gates"
    gates_path.mkdir(parents=True, exist_ok=True)

    output = {
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gates": [r.to_dict() for r in results],
        "all_passed": all(r.passed for r in results),
        "can_proceed": all(r.passed or not r.blocking for r in results),
    }

    with open(gates_path / f"{stage}_gates.json", "w") as f:
        json.dump(output, f, indent=2)


def print_gate_results(results: list[GateResult]) -> None:
    """Print gate results to console."""
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        blocking = " [BLOCKING]" if result.blocking and not result.passed else ""
        print(f"\n  Gate: {result.gate_name} - {status}{blocking}")

        for check in result.checks:
            check_status = "OK" if check.passed else "FAIL"
            print(f"    [{check_status}] {check.name}: {check.message}")

        print(f"    -> {result.recommendation}")
