#!/usr/bin/env python
"""
Format Friction CLI - Tool-call experiment runner.

Usage:
    python -m experiments.cli <command> [options]

Commands:
    setup               Verify environment and create directories
    verify              Run deterministic verification checks
    pilot               Run pilot study (10 tasks x 10 trials)
    run                 Run full experiment
    analyze             Run pre-registered analysis
    validate-extractor  Validate NL extractor against ground truth
    validate-judge      Validate judge against human labels
    summary             Show summary of results

Each command writes checkpoint files and includes deterministic verification.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.core.api_providers import check_api_keys, list_available_models
from experiments.core.config import get_config_manager, get_model_config
from experiments.core.checkpoint import write_checkpoint, compute_file_sha256
from experiments.core.harness import ExperimentHarness, compute_cluster_stats, Task
from experiments.core.prompts import AblationCondition
from experiments.core.extractor import validate_extractor
from experiments.core.judge import validate_judge_human_agreement
from experiments.core.bootstrap import bootstrap_difference_ci, bootstrap_proportion_ci
from experiments.scenarios.tasks import (
    get_all_tasks,
    get_tasks_by_category,
    get_category_counts,
)
from experiments.core.verification import PhaseVerifier, run_verification


def cmd_setup(args: argparse.Namespace) -> int:
    """Verify environment and create directories."""
    print("Setting up experiment environment...")
    print()

    # Check API keys
    keys = check_api_keys()
    print("API Keys:")
    for provider, available in keys.items():
        status = "✓" if available else "✗"
        print(f"  {status} {provider.upper()}")

    available_models = list_available_models()
    print(f"\nAvailable models: {', '.join(available_models) if available_models else 'None'}")

    # Create directories
    dirs = [
        "experiments/results/pilot",
        "experiments/results/primary",
        "experiments/results/raw",
        "experiments/validation",
        "experiments/analysis",
    ]

    print("\nDirectories:")
    for d in dirs:
        path = Path(d)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"  Created: {d}")
        else:
            print(f"  Exists:  {d}")

    # Show task counts
    print("\nTask counts by category:")
    for category, count in get_category_counts().items():
        print(f"  {category}: {count}")

    total = len(get_all_tasks())
    print(f"  Total: {total}")

    # Lock model configuration
    config = get_model_config()
    print("\nModel configuration:")
    for key, value in config.items():
        if key != "locked_at":
            print(f"  {key}: {value}")

    print("\nSetup complete!")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Run deterministic verification checks for all phases."""
    print("Running deterministic verification checks...")
    print()

    verifier = PhaseVerifier()

    if args.phase == "all":
        summary = verifier.run_all_phase_checks()
        verifier.print_summary(summary)
        return 0 if summary['all_passed'] else 1

    elif args.phase == "infrastructure":
        results = []
        results.append(verifier.verify_rng_determinism())
        results.append(verifier.verify_bootstrap_determinism())

        passed = all(r.passed for r in results)
        for r in results:
            status = "✓" if r.passed else "✗"
            print(f"  {status} {r.check_name}: {r.message}")

        return 0 if passed else 1

    elif args.phase == "cleanup":
        results = verifier.verify_phase0_cleanup()
        passed = all(r.passed for r in results)
        for r in results:
            status = "✓" if r.passed else "✗"
            print(f"  {status} {r.check_name}: {r.message}")

        return 0 if passed else 1

    elif args.phase == "validation":
        results = verifier.verify_ground_truth_counts()

        # Also run extractor and judge validation if ground truth exists
        if Path("experiments/validation/extraction_ground_truth.json").exists():
            results.append(verifier.verify_extractor_accuracy())

        passed = all(r.passed for r in results)
        for r in results:
            status = "✓" if r.passed else "✗"
            print(f"  {status} {r.check_name}: {r.actual} (expected {r.expected})")

        return 0 if passed else 1

    else:
        print(f"Unknown phase: {args.phase}")
        return 1


def cmd_pilot(args: argparse.Namespace) -> int:
    """Run pilot study."""
    print(f"Running pilot study...")
    print(f"  Tasks: {args.n_tasks}")
    print(f"  Trials per task: {args.n_trials}")
    print(f"  Seed: {args.seed}")
    print(f"  Model: {args.model}")
    print(f"  Ablation: {args.ablation}")
    print()

    # Check API keys
    keys = check_api_keys()
    if not any(keys.values()):
        print("Error: No API keys configured. Run 'setup' first.")
        return 1

    # Get tasks
    all_tasks = get_all_tasks()
    if args.n_tasks > len(all_tasks):
        print(f"Warning: Requested {args.n_tasks} tasks but only {len(all_tasks)} available")
        args.n_tasks = len(all_tasks)

    # Select tasks
    tasks = all_tasks[:args.n_tasks]

    # Create harness
    harness = ExperimentHarness(
        results_dir="experiments/results",
        seed=args.seed,
    )

    # Create run
    run_id = harness.create_run(
        n_trials_per_task=args.n_trials,
        temperature=args.temperature,
        ablation=AblationCondition(args.ablation),
    )
    print(f"Run ID: {run_id}")
    print()

    # Run experiment
    total_trials = args.n_tasks * args.n_trials
    completed = 0

    try:
        for result in harness.run_experiment(
            tasks=tasks,
            n_trials_per_task=args.n_trials,
            ablation=AblationCondition(args.ablation),
            model=args.model,
        ):
            completed += 1
            correct = "✓" if result.is_correct else "✗"
            print(f"  [{completed}/{total_trials}] {result.task_id} ({result.condition}): {correct}")

    except KeyboardInterrupt:
        print("\nInterrupted!")

    # Save results
    output_path = harness.save_results()
    print(f"\nResults saved to: {output_path}")

    # Show summary
    summary = harness.get_summary()
    print("\nSummary:")
    print(f"  Total trials: {summary['n_trials']}")

    if summary['nl_only']['n'] > 0:
        print(f"  NL-only:   {summary['nl_only']['correct']}/{summary['nl_only']['n']} "
              f"({summary['nl_only']['accuracy']:.1%})")

    if summary['json_only']['n'] > 0:
        print(f"  JSON-only: {summary['json_only']['correct']}/{summary['json_only']['n']} "
              f"({summary['json_only']['accuracy']:.1%})")

    if summary['overall_friction'] is not None:
        friction_pp = summary['overall_friction'] * 100
        print(f"  Friction:  {friction_pp:+.1f} percentage points")

    # Write checkpoint
    write_checkpoint(
        phase=1,
        phase_name="pilot_study",
        status="passed",
        run_id=run_id,
        seed=args.seed,
        metrics={
            "n_tasks": args.n_tasks,
            "n_trials_per_task": args.n_trials,
            "total_trials": summary['n_trials'],
            "nl_accuracy": summary['nl_only']['accuracy'],
            "json_accuracy": summary['json_only']['accuracy'],
            "friction_pp": summary['overall_friction'] * 100 if summary['overall_friction'] else None,
        },
    )

    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run full experiment."""
    print("Running full experiment...")
    print(f"  Trials per task: {args.n_trials}")
    print(f"  Seed: {args.seed}")
    print(f"  Model: {args.model}")
    print()

    # This follows the same pattern as pilot but with all tasks
    all_tasks = get_all_tasks()
    print(f"  Tasks: {len(all_tasks)}")

    # Create harness
    harness = ExperimentHarness(
        results_dir="experiments/results",
        seed=args.seed,
    )

    run_id = harness.create_run(
        n_trials_per_task=args.n_trials,
        temperature=args.temperature,
        ablation=AblationCondition(args.ablation),
    )
    print(f"Run ID: {run_id}")
    print()

    total_trials = len(all_tasks) * args.n_trials
    completed = 0

    try:
        for result in harness.run_experiment(
            tasks=all_tasks,
            n_trials_per_task=args.n_trials,
            ablation=AblationCondition(args.ablation),
            model=args.model,
        ):
            completed += 1
            if completed % 10 == 0:
                print(f"  Progress: {completed}/{total_trials}")

    except KeyboardInterrupt:
        print("\nInterrupted! Saving partial results...")

    output_path = harness.save_results()
    print(f"\nResults saved to: {output_path}")

    summary = harness.get_summary()
    cluster_stats = compute_cluster_stats(harness.trials)

    print("\nSummary:")
    print(f"  Total trials: {summary['n_trials']}")
    print(f"  NL-only accuracy:   {summary['nl_only']['accuracy']:.1%}")
    print(f"  JSON-only accuracy: {summary['json_only']['accuracy']:.1%}")

    if summary['overall_friction'] is not None:
        print(f"  Overall friction:   {summary['overall_friction'] * 100:+.1f}pp")

    print(f"\nCluster stats:")
    print(f"  Tasks: {cluster_stats['n_tasks']}")
    if cluster_stats['task_level_accuracy_nl']:
        print(f"  Task-level NL accuracy: {cluster_stats['task_level_accuracy_nl']:.1%}")
    if cluster_stats['task_level_accuracy_json']:
        print(f"  Task-level JSON accuracy: {cluster_stats['task_level_accuracy_json']:.1%}")

    # Write checkpoint
    write_checkpoint(
        phase=2,
        phase_name="full_experiment",
        status="passed",
        run_id=run_id,
        seed=args.seed,
        outputs_sha256={str(output_path): compute_file_sha256(output_path)},
        metrics={
            "n_tasks": len(all_tasks),
            "n_trials_per_task": args.n_trials,
            "total_trials": summary['n_trials'],
            "nl_accuracy": summary['nl_only']['accuracy'],
            "json_accuracy": summary['json_only']['accuracy'],
            "friction_pp": summary['overall_friction'] * 100 if summary['overall_friction'] else None,
        },
    )

    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Run pre-registered analysis on results."""
    print("Running pre-registered analysis...")

    # Load results
    results_dir = Path("experiments/results/raw")
    result_files = list(results_dir.glob("trials_*.json"))

    if not result_files:
        print("Error: No result files found in experiments/results/raw/")
        return 1

    # Use most recent or specified file
    if args.results_file:
        results_path = Path(args.results_file)
    else:
        results_path = max(result_files, key=lambda p: p.stat().st_mtime)

    print(f"Analyzing: {results_path}")

    with open(results_path) as f:
        data = json.load(f)

    trials = data.get("trials", [])
    print(f"Loaded {len(trials)} trials")

    # Separate by condition
    nl_trials = [t for t in trials if t["condition"] == "nl_only"]
    json_trials = [t for t in trials if t["condition"] == "json_only"]

    nl_correct = sum(1 for t in nl_trials if t["is_correct"])
    json_correct = sum(1 for t in json_trials if t["is_correct"])

    nl_accuracy = nl_correct / len(nl_trials) if nl_trials else 0
    json_accuracy = json_correct / len(json_trials) if json_trials else 0

    friction = nl_accuracy - json_accuracy

    print()
    print("=" * 60)
    print("PRIMARY ANALYSIS (H1)")
    print("=" * 60)
    print()
    print(f"NL-only condition:")
    print(f"  N = {len(nl_trials)}")
    print(f"  Correct = {nl_correct}")
    print(f"  Accuracy = {nl_accuracy:.1%}")
    print()
    print(f"JSON-only condition:")
    print(f"  N = {len(json_trials)}")
    print(f"  Correct = {json_correct}")
    print(f"  Accuracy = {json_accuracy:.1%}")
    print()
    print(f"Format Friction (NL - JSON): {friction * 100:+.1f} percentage points")

    # Bootstrap CI for difference
    nl_values = [1 if t["is_correct"] else 0 for t in nl_trials]
    json_values = [1 if t["is_correct"] else 0 for t in json_trials]

    if nl_values and json_values:
        diff_ci = bootstrap_difference_ci(
            nl_values, json_values,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            paired=False,
        )
        print()
        print(f"95% CI for friction: [{diff_ci['lower'] * 100:.1f}pp, {diff_ci['upper'] * 100:.1f}pp]")
        print(f"Statistically significant: {'Yes' if diff_ci['significant'] else 'No'}")

    # Check practical significance
    print()
    print("Practical significance (threshold = 10pp):")
    if abs(friction) >= 0.10:
        print(f"  Friction meets practical significance threshold")
    else:
        print(f"  Friction below practical significance threshold")

    # Manipulation check analysis
    print()
    print("=" * 60)
    print("MANIPULATION CHECKS")
    print("=" * 60)

    nl_declined = sum(1 for t in nl_trials if t.get("manipulation_check", {}).get("explicitly_declined", False))
    json_declined = sum(1 for t in json_trials if t.get("manipulation_check", {}).get("explicitly_declined", False))

    print()
    print(f"Explicit declines:")
    print(f"  NL-only:   {nl_declined}/{len(nl_trials)} ({100*nl_declined/len(nl_trials) if nl_trials else 0:.1f}%)")
    print(f"  JSON-only: {json_declined}/{len(json_trials)} ({100*json_declined/len(json_trials) if json_trials else 0:.1f}%)")

    # Save analysis results
    analysis_output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_file": str(results_path),
        "n_trials": len(trials),
        "primary_analysis": {
            "nl_n": len(nl_trials),
            "nl_correct": nl_correct,
            "nl_accuracy": nl_accuracy,
            "json_n": len(json_trials),
            "json_correct": json_correct,
            "json_accuracy": json_accuracy,
            "friction_pp": friction * 100,
            "ci_lower_pp": diff_ci['lower'] * 100 if 'diff_ci' in dir() else None,
            "ci_upper_pp": diff_ci['upper'] * 100 if 'diff_ci' in dir() else None,
            "significant": diff_ci['significant'] if 'diff_ci' in dir() else None,
            "practical_significance": abs(friction) >= 0.10,
        },
        "manipulation_checks": {
            "nl_declined": nl_declined,
            "json_declined": json_declined,
        },
        "seed": args.seed,
        "n_bootstrap": args.n_bootstrap,
    }

    output_path = Path("experiments/results/analysis") / f"analysis_{data.get('run_id', 'unknown')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(analysis_output, f, indent=2)

    print()
    print(f"Analysis saved to: {output_path}")

    return 0


def cmd_validate_extractor(args: argparse.Namespace) -> int:
    """Validate NL extractor against ground truth."""
    print("Validating NL extractor...")

    result = validate_extractor(args.ground_truth)

    if "error" in result:
        print(f"Error: {result['error']}")
        return 1

    print()
    print(f"Validation results:")
    print(f"  Accuracy: {result['accuracy']:.1%}")
    print(f"  Correct:  {result['correct']}/{result['total']}")
    print(f"  Threshold: {result['threshold']:.0%}")
    print(f"  Meets threshold: {'Yes' if result['meets_threshold'] else 'No'}")

    if result['errors']:
        print()
        print("Sample errors:")
        for err in result['errors'][:5]:
            print(f"  - Expected {err['expected']}, got {err['actual']}: {err['response'][:50]}...")

    return 0 if result['meets_threshold'] else 1


def cmd_validate_judge(args: argparse.Namespace) -> int:
    """Validate judge against human labels."""
    print("Validating judge-human agreement...")

    result = validate_judge_human_agreement(
        args.ground_truth,
        use_llm=args.use_llm,
        llm_model=args.model,
    )

    if "error" in result:
        print(f"Error: {result['error']}")
        return 1

    print()
    print(f"Validation results:")
    print(f"  Cohen's kappa: {result['kappa']:.3f}")
    print(f"  Raw agreement: {result['agreement']:.1%}")
    print(f"  N examples:    {result['n_examples']}")
    print(f"  Threshold:     {result['threshold']:.2f}")
    print(f"  Meets threshold: {'Yes' if result['meets_threshold'] else 'No'}")
    print(f"  Judge model:   {result['judge_model']}")

    return 0 if result['meets_threshold'] else 1


def cmd_summary(args: argparse.Namespace) -> int:
    """Show summary of all results."""
    print("Experiment Results Summary")
    print("=" * 60)

    results_dir = Path("experiments/results/raw")
    result_files = list(results_dir.glob("trials_*.json"))

    if not result_files:
        print("No results found.")
        return 0

    for path in sorted(result_files, key=lambda p: p.stat().st_mtime, reverse=True):
        with open(path) as f:
            data = json.load(f)

        trials = data.get("trials", [])
        run_id = data.get("run_id", "unknown")

        nl_trials = [t for t in trials if t["condition"] == "nl_only"]
        json_trials = [t for t in trials if t["condition"] == "json_only"]

        nl_acc = sum(1 for t in nl_trials if t["is_correct"]) / len(nl_trials) if nl_trials else 0
        json_acc = sum(1 for t in json_trials if t["is_correct"]) / len(json_trials) if json_trials else 0

        print()
        print(f"Run: {run_id}")
        print(f"  Trials: {len(trials)}")
        print(f"  NL accuracy:   {nl_acc:.1%}")
        print(f"  JSON accuracy: {json_acc:.1%}")
        print(f"  Friction:      {(nl_acc - json_acc) * 100:+.1f}pp")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Format Friction Experiment CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # setup
    p_setup = subparsers.add_parser("setup", help="Verify environment and create directories")

    # verify
    p_verify = subparsers.add_parser("verify", help="Run deterministic verification checks")
    p_verify.add_argument("--phase", default="all",
                          choices=["all", "infrastructure", "cleanup", "validation"],
                          help="Phase to verify (default: all)")

    # pilot
    p_pilot = subparsers.add_parser("pilot", help="Run pilot study")
    p_pilot.add_argument("--n-tasks", type=int, default=10, help="Number of tasks (default: 10)")
    p_pilot.add_argument("--n-trials", type=int, default=10, help="Trials per task (default: 10)")
    p_pilot.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p_pilot.add_argument("--model", default="claude-sonnet", help="Model to use")
    p_pilot.add_argument("--ablation", default="full", choices=["minimal", "tools_security", "tools_style", "full"])
    p_pilot.add_argument("--temperature", type=float, default=0.0, help="Temperature (default: 0.0)")

    # run
    p_run = subparsers.add_parser("run", help="Run full experiment")
    p_run.add_argument("--n-trials", type=int, default=30, help="Trials per task (default: 30)")
    p_run.add_argument("--seed", type=int, default=42, help="Random seed")
    p_run.add_argument("--model", default="claude-sonnet", help="Model to use")
    p_run.add_argument("--ablation", default="full", choices=["minimal", "tools_security", "tools_style", "full"])
    p_run.add_argument("--temperature", type=float, default=0.0, help="Temperature")

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Run pre-registered analysis")
    p_analyze.add_argument("--results-file", help="Results file to analyze (default: most recent)")
    p_analyze.add_argument("--seed", type=int, default=42, help="Bootstrap seed")
    p_analyze.add_argument("--n-bootstrap", type=int, default=10000, help="Bootstrap replicates")

    # validate-extractor
    p_val_ext = subparsers.add_parser("validate-extractor", help="Validate NL extractor")
    p_val_ext.add_argument("--ground-truth", default="experiments/validation/extraction_ground_truth.json")

    # validate-judge
    p_val_judge = subparsers.add_parser("validate-judge", help="Validate judge agreement")
    p_val_judge.add_argument("--ground-truth", default="experiments/validation/judgment_ground_truth.json")
    p_val_judge.add_argument("--use-llm", action="store_true", help="Use LLM-based judging")
    p_val_judge.add_argument("--model", default="claude-sonnet", help="Judge model")

    # summary
    p_summary = subparsers.add_parser("summary", help="Show results summary")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    handlers = {
        "setup": cmd_setup,
        "verify": cmd_verify,
        "pilot": cmd_pilot,
        "run": cmd_run,
        "analyze": cmd_analyze,
        "validate-extractor": cmd_validate_extractor,
        "validate-judge": cmd_validate_judge,
        "summary": cmd_summary,
    }

    handler = handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
