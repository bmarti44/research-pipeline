"""
Pilot study management for the research pipeline.

Implements automatic pilot → main study progression with verification gates.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class PilotCriteria:
    """Criteria that must be met for pilot to pass."""
    min_trials: int = 20
    min_response_rate: float = 0.95  # At least 95% of trials must get responses
    min_evaluation_determinism: float = 1.0  # 100% deterministic evaluation
    max_error_rate: float = 0.05  # At most 5% API errors
    min_effect_detection_power: Optional[float] = None  # If specified, check if observed effect is detectable


@dataclass
class PilotResult:
    """Result of a pilot study evaluation."""
    passed: bool
    criteria_met: dict[str, bool]
    messages: list[str]
    recommendation: str  # "proceed", "adjust", "abort"
    observed_metrics: dict
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "criteria_met": self.criteria_met,
            "messages": self.messages,
            "recommendation": self.recommendation,
            "observed_metrics": self.observed_metrics,
            "timestamp": self.timestamp,
        }


def load_pilot_criteria(config: dict) -> PilotCriteria:
    """Load pilot criteria from study config."""
    pilot_config = config.get("pilot", {})

    return PilotCriteria(
        min_trials=pilot_config.get("min_trials", 20),
        min_response_rate=pilot_config.get("min_response_rate", 0.95),
        min_evaluation_determinism=pilot_config.get("min_evaluation_determinism", 1.0),
        max_error_rate=pilot_config.get("max_error_rate", 0.05),
        min_effect_detection_power=pilot_config.get("min_effect_detection_power"),
    )


def evaluate_pilot(study_path: Path, config: dict) -> PilotResult:
    """
    Evaluate pilot study results against criteria.

    Must be called after execute and evaluate stages complete.
    """
    criteria = load_pilot_criteria(config)
    criteria_met = {}
    messages = []
    observed = {}

    # Check trial count
    trials_path = study_path / "stages" / "2_generate" / "trials.json"
    if trials_path.exists():
        with open(trials_path) as f:
            trials = json.load(f)
        trial_count = len(trials)
        observed["trial_count"] = trial_count

        criteria_met["min_trials"] = trial_count >= criteria.min_trials
        if not criteria_met["min_trials"]:
            messages.append(f"Insufficient trials: {trial_count} < {criteria.min_trials}")
    else:
        criteria_met["min_trials"] = False
        messages.append("No trials.json found")
        observed["trial_count"] = 0

    # Check response rate
    responses_path = study_path / "stages" / "3_execute" / "responses"
    if responses_path.exists():
        response_files = list(responses_path.glob("trial_*.json"))
        response_count = len(response_files)

        successful = 0
        errors = 0
        for rf in response_files:
            with open(rf) as f:
                resp = json.load(f)
            if resp.get("success", False):
                successful += 1
            else:
                errors += 1

        response_rate = successful / trial_count if trial_count > 0 else 0
        error_rate = errors / trial_count if trial_count > 0 else 1

        observed["response_rate"] = response_rate
        observed["error_rate"] = error_rate
        observed["successful_responses"] = successful

        criteria_met["min_response_rate"] = response_rate >= criteria.min_response_rate
        criteria_met["max_error_rate"] = error_rate <= criteria.max_error_rate

        if not criteria_met["min_response_rate"]:
            messages.append(f"Low response rate: {response_rate:.2%} < {criteria.min_response_rate:.2%}")
        if not criteria_met["max_error_rate"]:
            messages.append(f"High error rate: {error_rate:.2%} > {criteria.max_error_rate:.2%}")
    else:
        criteria_met["min_response_rate"] = False
        criteria_met["max_error_rate"] = False
        messages.append("No responses found")
        observed["response_rate"] = 0
        observed["error_rate"] = 1

    # Check evaluation determinism
    det_path = study_path / "stages" / "4_evaluate" / "determinism_check.json"
    if det_path.exists():
        with open(det_path) as f:
            det = json.load(f)

        determinism_passed = det.get("passed", False)
        observed["evaluation_deterministic"] = determinism_passed

        criteria_met["min_evaluation_determinism"] = (
            determinism_passed or criteria.min_evaluation_determinism < 1.0
        )
        if not criteria_met["min_evaluation_determinism"]:
            messages.append("Evaluation is not deterministic")
    else:
        criteria_met["min_evaluation_determinism"] = False
        messages.append("No determinism check found")
        observed["evaluation_deterministic"] = False

    # Check observed effect (if analysis was run)
    analyze_path = study_path / "stages" / "5_analyze"
    tests_path = analyze_path / "tests.json"
    if tests_path.exists():
        with open(tests_path) as f:
            tests = json.load(f)

        if tests:
            # Look for primary test effect size
            primary_test = tests[0] if tests else {}
            observed["effect_size"] = primary_test.get("effect_size")
            observed["p_value"] = primary_test.get("p_value")

            # Effect detection power check (optional)
            if criteria.min_effect_detection_power is not None:
                from .stats import compute_power
                observed_effect = primary_test.get("effect_size")
                test_name = primary_test.get("test_name", "two_proportion_z")
                if observed_effect is not None and observed_effect > 0:
                    power_result = compute_power(
                        test_name=test_name,
                        effect_size=observed_effect,
                        n=observed.get("completed_trials", 20),
                        alpha=0.05,
                    )
                    pilot_power = power_result["power"]
                    criteria_met["effect_detectable"] = pilot_power >= criteria.min_effect_detection_power
                    if not criteria_met["effect_detectable"]:
                        messages.append(
                            f"Estimated power too low: {pilot_power:.2%} < "
                            f"{criteria.min_effect_detection_power:.2%} "
                            f"(observed effect={observed_effect:.3f})"
                        )
                else:
                    criteria_met["effect_detectable"] = False
                    messages.append("No measurable effect size in pilot data")
    else:
        observed["effect_size"] = None
        observed["p_value"] = None

    # Determine recommendation
    all_critical_passed = all([
        criteria_met.get("min_trials", False),
        criteria_met.get("min_response_rate", False),
        criteria_met.get("min_evaluation_determinism", False),
    ])

    all_passed = all(criteria_met.values())

    if all_passed:
        recommendation = "proceed"
        messages.append("All pilot criteria met. Ready for main study.")
    elif all_critical_passed:
        recommendation = "adjust"
        messages.append("Critical criteria met but some adjustments recommended.")
    else:
        recommendation = "abort"
        messages.append("Critical pilot criteria not met. Review and fix issues before proceeding.")

    return PilotResult(
        passed=all_passed,
        criteria_met=criteria_met,
        messages=messages,
        recommendation=recommendation,
        observed_metrics=observed,
    )


def create_pilot_config(study_path: Path) -> Path:
    """
    Create a pilot configuration from the main study config.

    Pilot uses reduced trial count but same conditions.
    """
    config_path = study_path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.yaml found in {study_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create pilot config
    pilot_config = config.copy()

    # Reduce trials
    main_repetitions = config.get("trials", {}).get("repetitions", 5)
    pilot_repetitions = max(1, main_repetitions // 5)  # 20% of main

    pilot_config["trials"] = pilot_config.get("trials", {}).copy()
    pilot_config["trials"]["repetitions"] = pilot_repetitions
    pilot_config["trials"]["seed"] = config.get("trials", {}).get("seed", 42) + 1000  # Different seed

    # Mark as pilot
    pilot_config["is_pilot"] = True
    pilot_config["main_study_path"] = str(study_path)

    # Add pilot criteria if not present
    if "pilot" not in pilot_config:
        pilot_config["pilot"] = {
            "min_trials": 20,
            "min_response_rate": 0.95,
            "max_error_rate": 0.05,
            "min_evaluation_determinism": 1.0,
        }

    # Save pilot config
    pilot_path = study_path / "pilot"
    pilot_path.mkdir(exist_ok=True)

    pilot_config_path = pilot_path / "config.yaml"
    with open(pilot_config_path, "w") as f:
        yaml.dump(pilot_config, f, default_flow_style=False)

    # Copy study files to pilot
    for filename in ["tasks.py", "evaluation.py", "analysis.py", "prompts.py"]:
        src = study_path / filename
        if src.exists():
            dst = pilot_path / filename
            dst.write_text(src.read_text())

    return pilot_path


def check_pilot_gate(study_path: Path) -> tuple[bool, str]:
    """
    Check if pilot gate allows proceeding to main study.

    Returns (can_proceed, message).
    """
    pilot_path = study_path / "pilot"
    pilot_result_path = pilot_path / "pilot_result.json"

    if not pilot_result_path.exists():
        return False, "Pilot study not completed. Run pilot first."

    with open(pilot_result_path) as f:
        result = json.load(f)

    if result.get("recommendation") == "proceed":
        return True, "Pilot passed. Proceeding to main study."
    elif result.get("recommendation") == "adjust":
        return True, f"Pilot passed with warnings: {'; '.join(result.get('messages', []))}"
    else:
        return False, f"Pilot failed: {'; '.join(result.get('messages', []))}"


def save_pilot_result(study_path: Path, result: PilotResult) -> None:
    """Save pilot evaluation result."""
    pilot_path = study_path / "pilot"
    pilot_path.mkdir(exist_ok=True)

    with open(pilot_path / "pilot_result.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    # Also save human-readable summary
    summary = f"""# Pilot Study Results

## Summary
- **Passed**: {"Yes" if result.passed else "No"}
- **Recommendation**: {result.recommendation.upper()}
- **Timestamp**: {result.timestamp}

## Criteria Evaluation

| Criterion | Met |
|-----------|-----|
{chr(10).join(f"| {k} | {'PASS' if v else 'FAIL'} |" for k, v in result.criteria_met.items())}

## Observed Metrics

```json
{json.dumps(result.observed_metrics, indent=2)}
```

## Messages

{chr(10).join(f"- {m}" for m in result.messages)}

---
*Auto-generated by Research Pipeline*
"""

    with open(pilot_path / "PILOT_RESULTS.md", "w") as f:
        f.write(summary)
