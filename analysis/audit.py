"""
Generate audit reports for experiment verification.

This module creates human-readable reports to verify:
- Every validation decision made
- Semantic classifier scores for each decision
- Tool call sequences
- Convergence state progression
- Any anomalies or outliers
"""

import json
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional


@dataclass
class AuditFlags:
    """Flags for potential issues found during audit."""
    missing_validation_log: bool = False
    unexpected_score: bool = False
    high_rejection_count: bool = False
    semantic_score_near_threshold: bool = False
    error_occurred: bool = False
    long_duration: bool = False


def load_results(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_metadata(path: Path) -> Optional[dict]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def audit_trial(trial: dict, thresholds: dict) -> tuple[list[str], AuditFlags]:
    """Audit a single trial and return findings."""
    findings = []
    flags = AuditFlags()

    scenario_id = trial["scenario_id"]
    approach = trial["approach"]
    score = trial["score"]["score"]

    # Check for errors
    if trial.get("error"):
        flags.error_occurred = True
        findings.append(f"ERROR: {trial['error']}")

    # Check duration (flag if > 30 seconds)
    duration_ms = trial.get("duration_ms", 0)
    if duration_ms > 30000:
        flags.long_duration = True
        findings.append(f"SLOW: Trial took {duration_ms/1000:.1f}s")

    # For validated trials, check validation log
    if approach == "validated":
        validation_log = trial.get("validation_log", [])

        if not validation_log and trial["tools_attempted"]:
            flags.missing_validation_log = True
            findings.append("MISSING: No validation log but tools were attempted")

        # Check each validation decision
        for event in validation_log:
            semantic_scores = event.get("semantic_scores", {})

            # Check if semantic scores are near thresholds (within 0.05)
            for score_type, score_val in semantic_scores.items():
                if score_type in thresholds:
                    threshold = thresholds[score_type]
                    if abs(score_val - threshold) < 0.05:
                        flags.semantic_score_near_threshold = True
                        findings.append(
                            f"NEAR_THRESHOLD: {score_type}={score_val:.3f} "
                            f"(threshold={threshold}, diff={score_val-threshold:+.3f})"
                        )

            # Log denials with details
            if event["decision"] == "deny":
                findings.append(
                    f"DENIED: {event['tool_name']} by {event['rule_id']} "
                    f"(level={event['feedback_level']}, scores={semantic_scores})"
                )

        # Check convergence state
        convergence = trial.get("convergence_state", {})
        if convergence:
            total_rejections = convergence.get("total_rejections", 0)
            if total_rejections >= 3:
                flags.high_rejection_count = True
                findings.append(f"HIGH_REJECTIONS: {total_rejections} total rejections")

            if convergence.get("forced_direct_answer"):
                findings.append(f"FORCED_ANSWER: {convergence.get('termination_reason')}")

    # Check for unexpected scores
    expected_should_block = trial["score"]["validator_should_block"]
    validator_blocked = trial["score"]["validator_blocked"]
    correct_without = trial["score"]["correct_without_validator"]

    if approach == "validated":
        # If should block but didn't, and Claude didn't get it right independently
        if expected_should_block and not validator_blocked and not correct_without:
            flags.unexpected_score = True
            findings.append(
                f"MISSED_BLOCK: Validator should have blocked but didn't, "
                f"and Claude got it wrong (score={score})"
            )

        # If shouldn't block but did (false positive)
        if not expected_should_block and validator_blocked:
            flags.unexpected_score = True
            findings.append(
                f"FALSE_POSITIVE: Validator blocked when it shouldn't have "
                f"(score={score})"
            )

    return findings, flags


def generate_audit_report(results_path: Path, output_path: Path):
    """Generate a comprehensive audit report."""

    results = load_results(results_path)

    # Try to load metadata
    metadata_path = results_path.parent / results_path.name.replace("results_", "metadata_")
    metadata = load_metadata(metadata_path)

    thresholds = {}
    if metadata:
        thresholds = metadata.get("classifier_thresholds", {})

    # Aggregate statistics
    total_trials = len(results)
    trials_with_errors = 0
    trials_with_flags = 0
    flag_counts = defaultdict(int)
    all_findings = []
    trials_by_scenario = defaultdict(list)

    for trial in results:
        findings, flags = audit_trial(trial, thresholds)

        if findings:
            all_findings.append({
                "scenario_id": trial["scenario_id"],
                "trial_number": trial["trial_number"],
                "approach": trial["approach"],
                "findings": findings,
            })

        if flags.error_occurred:
            trials_with_errors += 1

        # Count flags
        has_any_flag = False
        for flag_name, flag_value in vars(flags).items():
            if flag_value:
                flag_counts[flag_name] += 1
                has_any_flag = True

        if has_any_flag:
            trials_with_flags += 1

        # Group by scenario
        trials_by_scenario[trial["scenario_id"]].append(trial)

    # Generate report
    lines = []
    lines.append("=" * 80)
    lines.append("EXPERIMENT AUDIT REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Metadata
    if metadata:
        lines.append("EXPERIMENT METADATA")
        lines.append("-" * 40)
        lines.append(f"  Started: {metadata.get('experiment_started', 'N/A')}")
        lines.append(f"  Completed: {metadata.get('experiment_completed', 'N/A')}")
        lines.append(f"  Total trials: {metadata.get('total_trials', 'N/A')}")
        lines.append(f"  Scenarios: {metadata.get('scenarios_count', 'N/A')}")
        lines.append(f"  Trials per scenario: {metadata.get('trials_per_scenario', 'N/A')}")
        lines.append("")
        lines.append("  Classifier thresholds:")
        for k, v in thresholds.items():
            lines.append(f"    {k}: {v}")
        lines.append("")

    # Summary
    lines.append("AUDIT SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Total trials audited: {total_trials}")
    lines.append(f"  Trials with errors: {trials_with_errors}")
    lines.append(f"  Trials with flags: {trials_with_flags}")
    lines.append("")
    lines.append("  Flag counts:")
    for flag_name, count in sorted(flag_counts.items()):
        lines.append(f"    {flag_name}: {count}")
    lines.append("")

    # Scenario-level summary
    lines.append("PER-SCENARIO SUMMARY")
    lines.append("-" * 40)
    lines.append("")

    for scenario_id in sorted(trials_by_scenario.keys()):
        trials = trials_by_scenario[scenario_id]
        baseline = [t for t in trials if t["approach"] == "baseline"]
        validated = [t for t in trials if t["approach"] == "validated"]

        baseline_scores = [t["score"]["score"] for t in baseline]
        validated_scores = [t["score"]["score"] for t in validated]

        baseline_avg = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0
        validated_avg = sum(validated_scores) / len(validated_scores) if validated_scores else 0

        # Count rejections
        total_rejections = sum(len(t.get("validator_rejections", [])) for t in validated)

        lines.append(f"  {scenario_id}:")
        lines.append(f"    Baseline avg: {baseline_avg:.2f} (n={len(baseline)})")
        lines.append(f"    Validated avg: {validated_avg:.2f} (n={len(validated)})")
        lines.append(f"    Delta: {validated_avg - baseline_avg:+.2f}")
        lines.append(f"    Total rejections: {total_rejections}")
        lines.append("")

    # Detailed findings
    if all_findings:
        lines.append("DETAILED FINDINGS")
        lines.append("-" * 40)
        lines.append("")

        for entry in all_findings:
            lines.append(f"[{entry['scenario_id']}] Trial {entry['trial_number']} ({entry['approach']})")
            for finding in entry["findings"]:
                lines.append(f"  - {finding}")
            lines.append("")

    # Outlier detection
    lines.append("OUTLIER ANALYSIS")
    lines.append("-" * 40)

    # Find scenarios with high variance
    for scenario_id, trials in trials_by_scenario.items():
        validated = [t for t in trials if t["approach"] == "validated"]
        if len(validated) > 1:
            scores = [t["score"]["score"] for t in validated]
            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            if variance > 0.5:  # High variance threshold
                lines.append(f"  HIGH VARIANCE: {scenario_id}")
                lines.append(f"    Scores: {scores}")
                lines.append(f"    Variance: {variance:.2f}")
                lines.append("")

    # Find trials with unusually long duration
    durations = [(t["scenario_id"], t["trial_number"], t["approach"], t.get("duration_ms", 0))
                 for t in results if t.get("duration_ms", 0) > 0]
    if durations:
        avg_duration = sum(d[3] for d in durations) / len(durations)
        outliers = [d for d in durations if d[3] > avg_duration * 2]
        if outliers:
            lines.append("  SLOW TRIALS (>2x average):")
            for scenario_id, trial_num, approach, dur in sorted(outliers, key=lambda x: -x[3])[:10]:
                lines.append(f"    {scenario_id} t{trial_num} ({approach}): {dur/1000:.1f}s")
            lines.append("")

    # Write report
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Audit report written to {output_path}")
    return output_path


def main():
    results_dir = Path("experiments/results")
    result_files = sorted(results_dir.glob("results_*.json"))

    if not result_files:
        print("No results found.")
        return

    latest = result_files[-1]
    print(f"Auditing: {latest.name}")

    output_path = results_dir / latest.name.replace("results_", "audit_").replace(".json", ".txt")
    generate_audit_report(latest, output_path)


if __name__ == "__main__":
    main()
