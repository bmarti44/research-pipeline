"""
Stage execution engine for the research pipeline.

Handles running stages, verification gates, and checkpointing.
Integrates preregistration, pilot studies, adaptive stopping, and review gates.
"""

import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Any
import yaml

from .verification import (
    verify_stage,
    VerificationResult,
    STAGE_NUMBERS,
    hash_directory,
)


class StageStatus(Enum):
    """Status of a pipeline stage."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """Result of running a stage."""
    stage: str
    status: StageStatus
    verification: Optional[VerificationResult] = None
    error: Optional[str] = None
    duration_ms: Optional[int] = None


def get_study_status(study_path: Path) -> dict[str, StageStatus]:
    """Get status of all stages for a study."""
    status = {}

    for stage, num in STAGE_NUMBERS.items():
        stage_dir = study_path / "stages" / f"{num}_{stage}"
        manifest_path = stage_dir / "manifest.json"

        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            if manifest.get("completed", False):
                status[stage] = StageStatus.COMPLETED
            elif manifest.get("failed", False):
                status[stage] = StageStatus.FAILED
            else:
                status[stage] = StageStatus.IN_PROGRESS
        else:
            status[stage] = StageStatus.NOT_STARTED

    return status


def load_study_config(study_path: Path) -> dict:
    """Load study configuration."""
    config_path = study_path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.yaml found in {study_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def load_study_module(study_path: Path, module_name: str) -> Any:
    """Dynamically load a module from the study directory."""
    module_path = study_path / f"{module_name}.py"
    if not module_path.exists():
        raise FileNotFoundError(f"No {module_name}.py found in {study_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def save_manifest(stage_path: Path, data: dict) -> None:
    """Save stage manifest."""
    stage_path.mkdir(parents=True, exist_ok=True)
    manifest_path = stage_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(data, f, indent=2)


# Stage implementations

def run_configure(study_path: Path, config: dict) -> StageResult:
    """
    Run the configure stage.

    - Validate config
    - Lock environment versions
    - Resolve model aliases
    """
    import platform

    stage_path = study_path / "stages" / "1_configure"
    stage_path.mkdir(parents=True, exist_ok=True)

    try:
        # Get package versions
        packages = {}
        for pkg in config.get("environment", {}).get("required_packages", []):
            pkg_name = pkg.split(">=")[0].split("==")[0].split("<")[0]
            try:
                import importlib.metadata
                packages[pkg_name] = importlib.metadata.version(pkg_name)
            except Exception:
                packages[pkg_name] = "not installed"

        # Lock environment
        env_lock = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with open(stage_path / "environment_lock.json", "w") as f:
            json.dump(env_lock, f, indent=2)

        # Resolve model aliases
        from .api import MODELS

        resolved_models = []
        for model in config.get("models", []):
            alias = model.get("alias", model.get("model_id"))
            if alias in MODELS:
                model_config = MODELS[alias]
                resolved_models.append({
                    "alias": alias,
                    "model_id": model_config.model_id,
                    "provider": model_config.provider.value,
                    "temperature": model.get("temperature", model_config.temperature),
                    "max_tokens": model.get("max_tokens", model_config.max_tokens),
                })
            else:
                resolved_models.append(model)

        # Create resolved config
        resolved_config = config.copy()
        resolved_config["models"] = resolved_models
        resolved_config["_resolved_at"] = datetime.now(timezone.utc).isoformat()

        with open(stage_path / "config_resolved.yaml", "w") as f:
            yaml.dump(resolved_config, f, default_flow_style=False)

        # Save manifest
        save_manifest(stage_path, {
            "stage": "configure",
            "completed": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Verify
        verification = verify_stage(study_path, "configure")

        return StageResult(
            stage="configure",
            status=StageStatus.COMPLETED if verification.passed else StageStatus.FAILED,
            verification=verification,
        )

    except Exception as e:
        save_manifest(stage_path, {
            "stage": "configure",
            "completed": False,
            "failed": True,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return StageResult(
            stage="configure",
            status=StageStatus.FAILED,
            error=str(e),
        )


def run_generate(study_path: Path, config: dict) -> StageResult:
    """
    Run the generate stage.

    - Load tasks from tasks.py
    - Generate trial matrix
    - Validate coverage
    """
    import numpy as np

    stage_path = study_path / "stages" / "2_generate"
    stage_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load tasks module
        tasks_module = load_study_module(study_path, "tasks")

        if not hasattr(tasks_module, "get_tasks"):
            raise ValueError("tasks.py must define get_tasks() function")

        tasks = tasks_module.get_tasks()

        # Get conditions and trial settings
        conditions = [c["name"] for c in config.get("conditions", [])]
        repetitions = config.get("trials", {}).get("repetitions", 1)
        seed = config.get("trials", {}).get("seed", 42)
        randomize = config.get("trials", {}).get("randomize_order", True)

        # Generate trial matrix
        trials = []
        trial_id = 0

        for task in tasks:
            for condition in conditions:
                for rep in range(repetitions):
                    trials.append({
                        "trial_id": f"trial_{trial_id:04d}",
                        "task_id": task.get("task_id", f"task_{trial_id}"),
                        "task": task,
                        "condition": condition,
                        "repetition": rep,
                    })
                    trial_id += 1

        # Randomize order if requested
        if randomize:
            rng = np.random.default_rng(seed)
            rng.shuffle(trials)

        # Save trials
        with open(stage_path / "trials.json", "w") as f:
            json.dump(trials, f, indent=2)

        # Generate coverage report
        coverage = {
            "total_trials": len(trials),
            "total_tasks": len(tasks),
            "conditions": conditions,
            "repetitions": repetitions,
            "by_condition": {c: sum(1 for t in trials if t["condition"] == c) for c in conditions},
            "by_category": {},
        }

        # Count by category if tasks have categories
        categories = set()
        for task in tasks:
            cat = task.get("category", "uncategorized")
            categories.add(cat)
        for cat in categories:
            coverage["by_category"][cat] = sum(
                1 for t in trials if t["task"].get("category") == cat
            )

        with open(stage_path / "coverage_report.json", "w") as f:
            json.dump(coverage, f, indent=2)

        # Save manifest
        save_manifest(stage_path, {
            "stage": "generate",
            "completed": True,
            "trial_count": len(trials),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Verify
        verification = verify_stage(study_path, "generate")

        return StageResult(
            stage="generate",
            status=StageStatus.COMPLETED if verification.passed else StageStatus.FAILED,
            verification=verification,
        )

    except Exception as e:
        save_manifest(stage_path, {
            "stage": "generate",
            "completed": False,
            "failed": True,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return StageResult(
            stage="generate",
            status=StageStatus.FAILED,
            error=str(e),
        )


def run_execute(study_path: Path, config: dict, resume: bool = False) -> StageResult:
    """
    Run the execute stage.

    - Load trials
    - Call LLM API for each trial
    - Save responses with full metadata
    - Support resume from checkpoint
    """
    from .api import call_model_with_retry, MODELS

    stage_path = study_path / "stages" / "3_execute"
    responses_path = stage_path / "responses"
    responses_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load trials
        trials_path = study_path / "stages" / "2_generate" / "trials.json"
        with open(trials_path) as f:
            trials = json.load(f)

        # Load resolved config for model settings
        config_path = study_path / "stages" / "1_configure" / "config_resolved.yaml"
        with open(config_path) as f:
            resolved_config = yaml.safe_load(f)

        # Get model to use (first model in list)
        model_config = resolved_config.get("models", [{}])[0]
        model_alias = model_config.get("alias", "claude-sonnet")

        # Load prompts module if it exists
        try:
            prompts_module = load_study_module(study_path, "prompts")
            build_prompt = prompts_module.build_prompt
            build_system_prompt = prompts_module.build_system_prompt
        except FileNotFoundError:
            # Default prompt builders
            def build_prompt(task: dict, condition: str) -> str:
                return task.get("user_prompt", str(task))

            def build_system_prompt(task: dict, condition: str) -> str:
                return task.get("system_prompt", "")

        # Track execution
        execution_log = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "model": model_alias,
            "total_trials": len(trials),
            "completed": 0,
            "failed": 0,
            "trials": [],
        }

        # Execute trials
        for i, trial in enumerate(trials):
            trial_id = trial["trial_id"]
            response_path = responses_path / f"{trial_id}.json"

            # Skip if resume and already exists
            if resume and response_path.exists():
                execution_log["completed"] += 1
                continue

            # Build prompts
            prompt = build_prompt(trial["task"], trial["condition"])
            system_prompt = build_system_prompt(trial["task"], trial["condition"])

            # Call API
            response = call_model_with_retry(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model_alias,
            )

            # Save response
            response_data = {
                "trial_id": trial_id,
                "task_id": trial["task_id"],
                "condition": trial["condition"],
                "prompt": prompt,
                "system_prompt": system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt,
                "response": response.response,
                "success": response.success,
                "error": response.error,
                "model": response.model,
                "timestamp": response.timestamp,
                "latency_ms": response.latency_ms,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "retry_count": response.retry_count,
            }

            with open(response_path, "w") as f:
                json.dump(response_data, f, indent=2)

            # Update log
            if response.success:
                execution_log["completed"] += 1
            else:
                execution_log["failed"] += 1

            execution_log["trials"].append({
                "trial_id": trial_id,
                "success": response.success,
                "latency_ms": response.latency_ms,
            })

            # Progress output
            print(f"  [{i+1}/{len(trials)}] {trial_id}: {'OK' if response.success else 'FAIL'}")

        # Finalize log
        execution_log["finished_at"] = datetime.now(timezone.utc).isoformat()

        with open(stage_path / "execution_log.json", "w") as f:
            json.dump(execution_log, f, indent=2)

        # Save manifest
        save_manifest(stage_path, {
            "stage": "execute",
            "completed": True,
            "trials_completed": execution_log["completed"],
            "trials_failed": execution_log["failed"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Verify
        verification = verify_stage(study_path, "execute")

        return StageResult(
            stage="execute",
            status=StageStatus.COMPLETED if verification.passed else StageStatus.FAILED,
            verification=verification,
        )

    except Exception as e:
        save_manifest(stage_path, {
            "stage": "execute",
            "completed": False,
            "failed": True,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return StageResult(
            stage="execute",
            status=StageStatus.FAILED,
            error=str(e),
        )


def run_evaluate(study_path: Path, config: dict) -> StageResult:
    """
    Run the evaluate stage.

    - Load responses
    - Run study's evaluation.py
    - Check determinism
    """
    stage_path = study_path / "stages" / "4_evaluate"
    scores_path = stage_path / "scores"
    scores_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load evaluation module
        eval_module = load_study_module(study_path, "evaluation")

        if not hasattr(eval_module, "evaluate_response"):
            raise ValueError("evaluation.py must define evaluate_response() function")

        evaluate_fn = eval_module.evaluate_response

        # Load responses
        responses_path = study_path / "stages" / "3_execute" / "responses"
        response_files = list(responses_path.glob("trial_*.json"))

        # Load trials to get task data (expected_tool, expected_args)
        trials_path = study_path / "stages" / "2_generate" / "trials.json"
        trial_id_to_task = {}
        if trials_path.exists():
            with open(trials_path) as f:
                trials = json.load(f)
            for trial in trials:
                trial_id_to_task[trial["trial_id"]] = trial.get("task", {})

        # Evaluate each response
        for rf in response_files:
            with open(rf) as f:
                response_data = json.load(f)

            # Load trial info
            trial_id = response_data["trial_id"]

            # Merge response data with original task data to get expected values
            task_data = trial_id_to_task.get(trial_id, {})
            merged_task = {**task_data, **response_data}

            # Run evaluation
            scores = evaluate_fn(
                response=response_data.get("response", ""),
                task=merged_task,
                condition=response_data.get("condition", ""),
            )

            # Save scores
            score_data = {
                "trial_id": trial_id,
                "modes": scores,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            with open(scores_path / f"{trial_id}.json", "w") as f:
                json.dump(score_data, f, indent=2)

        # Determinism check: re-evaluate 10% sample and compare
        import random
        sample_size = max(1, len(response_files) // 10)
        sample_files = random.sample(response_files, sample_size)

        determinism_passed = True
        mismatches = []

        for rf in sample_files:
            with open(rf) as f:
                response_data = json.load(f)

            trial_id = response_data["trial_id"]

            # Merge with original task data
            task_data = trial_id_to_task.get(trial_id, {})
            merged_task = {**task_data, **response_data}

            # Re-evaluate
            scores_reeval = evaluate_fn(
                response=response_data.get("response", ""),
                task=merged_task,
                condition=response_data.get("condition", ""),
            )

            # Load original scores
            with open(scores_path / f"{trial_id}.json") as f:
                original = json.load(f)

            # Compare
            if scores_reeval != original["modes"]:
                determinism_passed = False
                mismatches.append(trial_id)

        # Save determinism check
        det_check = {
            "passed": determinism_passed,
            "sample_size": sample_size,
            "mismatches": mismatches,
            "message": "OK" if determinism_passed else f"Mismatches in {len(mismatches)} trials",
        }

        with open(stage_path / "determinism_check.json", "w") as f:
            json.dump(det_check, f, indent=2)

        # Save manifest
        save_manifest(stage_path, {
            "stage": "evaluate",
            "completed": True,
            "scores_count": len(response_files),
            "determinism_passed": determinism_passed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Verify
        verification = verify_stage(study_path, "evaluate")

        return StageResult(
            stage="evaluate",
            status=StageStatus.COMPLETED if verification.passed else StageStatus.FAILED,
            verification=verification,
        )

    except Exception as e:
        save_manifest(stage_path, {
            "stage": "evaluate",
            "completed": False,
            "failed": True,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return StageResult(
            stage="evaluate",
            status=StageStatus.FAILED,
            error=str(e),
        )


def run_analyze(study_path: Path, config: dict) -> StageResult:
    """
    Run the analyze stage.

    - Aggregate scores
    - Run statistical tests
    - Check assumptions
    """
    stage_path = study_path / "stages" / "5_analyze"
    stage_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load analysis module
        analysis_module = load_study_module(study_path, "analysis")

        if not hasattr(analysis_module, "run_analysis"):
            raise ValueError("analysis.py must define run_analysis() function")

        # Load all scores
        scores_path = study_path / "stages" / "4_evaluate" / "scores"
        score_files = list(scores_path.glob("trial_*.json"))

        scores = []
        for sf in score_files:
            with open(sf) as f:
                scores.append(json.load(f))

        # Run analysis (pass study_path if function accepts it)
        import inspect
        sig = inspect.signature(analysis_module.run_analysis)
        if 'study_path' in sig.parameters:
            results = analysis_module.run_analysis(scores, config, study_path=study_path)
        else:
            results = analysis_module.run_analysis(scores, config)

        # Save results
        with open(stage_path / "aggregates.json", "w") as f:
            json.dump(results.get("aggregates", {}), f, indent=2)

        with open(stage_path / "tests.json", "w") as f:
            json.dump(results.get("tests", []), f, indent=2)

        with open(stage_path / "assumptions.json", "w") as f:
            json.dump(results.get("assumptions", {}), f, indent=2)

        # Save manifest
        save_manifest(stage_path, {
            "stage": "analyze",
            "completed": True,
            "tests_run": len(results.get("tests", [])),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Verify
        verification = verify_stage(study_path, "analyze")

        return StageResult(
            stage="analyze",
            status=StageStatus.COMPLETED if verification.passed else StageStatus.FAILED,
            verification=verification,
        )

    except Exception as e:
        save_manifest(stage_path, {
            "stage": "analyze",
            "completed": False,
            "failed": True,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return StageResult(
            stage="analyze",
            status=StageStatus.FAILED,
            error=str(e),
        )


def run_report(study_path: Path, config: dict) -> StageResult:
    """
    Run the report stage.

    - Generate RESULTS.md
    - Create archive
    """
    import zipfile

    stage_path = study_path / "stages" / "6_report"
    outputs_path = study_path / "outputs"
    stage_path.mkdir(parents=True, exist_ok=True)
    outputs_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load analysis results
        analyze_path = study_path / "stages" / "5_analyze"

        with open(analyze_path / "aggregates.json") as f:
            aggregates = json.load(f)

        with open(analyze_path / "tests.json") as f:
            tests = json.load(f)

        # Generate RESULTS.md
        study_name = config.get("study", {}).get("name", study_path.name)
        hypothesis = config.get("study", {}).get("hypothesis", "")

        report = f"""# {study_name}: Results

## Hypothesis

{hypothesis}

## Summary

Generated: {datetime.now(timezone.utc).isoformat()}

## Aggregates

```json
{json.dumps(aggregates, indent=2)}
```

## Statistical Tests

"""
        for test in tests:
            report += f"### {test.get('name', 'Test')}\n\n"
            report += f"- Statistic: {test.get('statistic', 'N/A')}\n"
            report += f"- p-value: {test.get('p_value', 'N/A')}\n"
            report += f"- Significant: {test.get('significant', 'N/A')}\n\n"

        report += """
---

*Generated by Research Pipeline*
"""

        with open(outputs_path / "RESULTS.md", "w") as f:
            f.write(report)

        # Create archive
        archive_path = outputs_path / "archive.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            stages_dir = study_path / "stages"
            for stage_dir in stages_dir.iterdir():
                if stage_dir.is_dir():
                    for file in stage_dir.rglob("*"):
                        if file.is_file():
                            arcname = file.relative_to(study_path)
                            zf.write(file, arcname)

        # Save manifest
        save_manifest(stage_path, {
            "stage": "report",
            "completed": True,
            "outputs": ["RESULTS.md", "archive.zip"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Verify
        verification = verify_stage(study_path, "report")

        return StageResult(
            stage="report",
            status=StageStatus.COMPLETED if verification.passed else StageStatus.FAILED,
            verification=verification,
        )

    except Exception as e:
        save_manifest(stage_path, {
            "stage": "report",
            "completed": False,
            "failed": True,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return StageResult(
            stage="report",
            status=StageStatus.FAILED,
            error=str(e),
        )


# Stage runner dispatcher
STAGE_RUNNERS = {
    "configure": run_configure,
    "generate": run_generate,
    "execute": run_execute,
    "evaluate": run_evaluate,
    "analyze": run_analyze,
    "report": run_report,
}


def run_stage(
    study_path: Path,
    stage: str,
    resume: bool = False,
) -> StageResult:
    """
    Run a specific stage.

    Args:
        study_path: Path to study directory
        stage: Stage name
        resume: Whether to resume from checkpoint (execute stage only)

    Returns:
        StageResult with status and verification
    """
    if stage not in STAGE_RUNNERS:
        return StageResult(
            stage=stage,
            status=StageStatus.FAILED,
            error=f"Unknown stage: {stage}. Valid: {list(STAGE_RUNNERS.keys())}",
        )

    # Load config
    config = load_study_config(study_path)

    # Check prerequisites
    stage_num = STAGE_NUMBERS[stage]
    if stage_num > 1:
        prev_stage = [s for s, n in STAGE_NUMBERS.items() if n == stage_num - 1][0]
        prev_verification = verify_stage(study_path, prev_stage)
        if not prev_verification.passed:
            return StageResult(
                stage=stage,
                status=StageStatus.SKIPPED,
                error=f"Prerequisite stage '{prev_stage}' has not passed verification",
            )

    # Run stage
    print(f"Running stage: {stage}")

    if stage == "execute":
        return STAGE_RUNNERS[stage](study_path, config, resume=resume)
    else:
        return STAGE_RUNNERS[stage](study_path, config)


def run_all_stages(
    study_path: Path,
    resume: bool = False,
) -> dict[str, StageResult]:
    """
    Run all stages in sequence.

    Stops if any stage fails verification.
    """
    results = {}

    for stage in ["configure", "generate", "execute", "evaluate", "analyze", "report"]:
        print(f"\n{'='*60}")
        print(f"Stage: {stage}")
        print("="*60)

        result = run_stage(study_path, stage, resume=resume)
        results[stage] = result

        if result.status == StageStatus.FAILED:
            print(f"\nStage {stage} failed: {result.error}")
            break

        if result.verification and not result.verification.passed:
            print(f"\nStage {stage} verification failed")
            for check in result.verification.checks:
                status = "PASS" if check.passed else "FAIL"
                print(f"  [{status}] {check.name}: {check.message}")
            break

        print(f"Stage {stage} completed successfully")

    return results


# =============================================================================
# Extended Pipeline with Review Gates
# =============================================================================

def run_stage_with_gates(
    study_path: Path,
    stage: str,
    resume: bool = False,
    skip_gates: bool = False,
) -> StageResult:
    """
    Run a stage with review gates.

    Gates are checked before execution. Blocking gates prevent execution.
    """
    from .review_gates import run_gates, save_gate_results, print_gate_results

    config = load_study_config(study_path)

    if not skip_gates:
        # Run pre-stage gates
        can_proceed, gate_results = run_gates(study_path, config, stage)

        if gate_results:
            print(f"\nReview Gates for {stage}:")
            print_gate_results(gate_results)
            save_gate_results(study_path, stage, gate_results)

            if not can_proceed:
                return StageResult(
                    stage=stage,
                    status=StageStatus.FAILED,
                    error="Blocked by review gate(s)",
                )

    # Run the actual stage
    return run_stage(study_path, stage, resume=resume)


def run_preregistration(study_path: Path) -> StageResult:
    """
    Run preregistration stage.

    Locks the study design before data collection.
    Must be run before execute stage.
    """
    from .preregistration import create_preregistration

    try:
        record = create_preregistration(study_path)

        print(f"  Preregistration created")
        print(f"  Hash: {record.hash}")
        print(f"  Locked files: {list(record.locked_files.keys())}")

        return StageResult(
            stage="preregistration",
            status=StageStatus.COMPLETED,
        )

    except Exception as e:
        return StageResult(
            stage="preregistration",
            status=StageStatus.FAILED,
            error=str(e),
        )


def run_pilot(study_path: Path, resume: bool = False) -> StageResult:
    """
    Run pilot study.

    Creates and runs a reduced version of the study.
    Must pass before main study can run.
    """
    from .pilot import create_pilot_config, evaluate_pilot, save_pilot_result

    config = load_study_config(study_path)

    # Create pilot if not exists
    pilot_path = study_path / "pilot"
    if not (pilot_path / "config.yaml").exists():
        print("  Creating pilot configuration...")
        pilot_path = create_pilot_config(study_path)

    # Run pilot stages
    print("  Running pilot study...")
    pilot_results = run_all_stages(pilot_path, resume=resume)

    # Check if pilot completed
    if pilot_results.get("report", StageResult("", StageStatus.FAILED)).status != StageStatus.COMPLETED:
        return StageResult(
            stage="pilot",
            status=StageStatus.FAILED,
            error="Pilot study did not complete all stages",
        )

    # Evaluate pilot
    print("  Evaluating pilot results...")
    pilot_result = evaluate_pilot(pilot_path, config)
    save_pilot_result(study_path, pilot_result)

    print(f"\n  Pilot Results:")
    print(f"    Passed: {pilot_result.passed}")
    print(f"    Recommendation: {pilot_result.recommendation}")
    for msg in pilot_result.messages:
        print(f"    - {msg}")

    if pilot_result.recommendation == "proceed":
        return StageResult(
            stage="pilot",
            status=StageStatus.COMPLETED,
        )
    elif pilot_result.recommendation == "adjust":
        return StageResult(
            stage="pilot",
            status=StageStatus.COMPLETED,
            error="Pilot passed with warnings - review before proceeding",
        )
    else:
        return StageResult(
            stage="pilot",
            status=StageStatus.FAILED,
            error=f"Pilot failed: {'; '.join(pilot_result.messages)}",
        )


def run_with_adaptive_stopping(
    study_path: Path,
    resume: bool = False,
) -> StageResult:
    """
    Run execute stage with adaptive stopping checks.

    Checks stopping rules at interim points.
    """
    from .adaptive import should_stop, perform_interim_analysis, save_interim_analysis, load_adaptive_config

    config = load_study_config(study_path)
    adaptive_config = load_adaptive_config(config)

    if not adaptive_config.enabled:
        # Just run normal execute
        return run_stage(study_path, "execute", resume=resume)

    # Load trials to know total
    trials_path = study_path / "stages" / "2_generate" / "trials.json"
    with open(trials_path) as f:
        trials = json.load(f)
    n_total = len(trials)

    # Calculate checkpoint numbers
    checkpoints = [int(n_total * f) for f in adaptive_config.interim_looks]

    # Run execute with interim checks
    from .api import call_model_with_retry

    stage_path = study_path / "stages" / "3_execute"
    responses_path = stage_path / "responses"
    responses_path.mkdir(parents=True, exist_ok=True)

    # Load resolved config
    config_path = study_path / "stages" / "1_configure" / "config_resolved.yaml"
    with open(config_path) as f:
        resolved_config = yaml.safe_load(f)

    model_config = resolved_config.get("models", [{}])[0]
    model_alias = model_config.get("alias", "claude-sonnet")

    # Load prompts
    try:
        prompts_module = load_study_module(study_path, "prompts")
        build_prompt = prompts_module.build_prompt
        build_system_prompt = prompts_module.build_system_prompt
    except FileNotFoundError:
        def build_prompt(task: dict, condition: str) -> str:
            return task.get("user_prompt", str(task))
        def build_system_prompt(task: dict, condition: str) -> str:
            return task.get("system_prompt", "")

    # Execute trials
    execution_log = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": model_alias,
        "total_trials": n_total,
        "completed": 0,
        "failed": 0,
        "stopped_early": False,
        "stopping_reason": None,
        "trials": [],
    }

    for i, trial in enumerate(trials):
        trial_id = trial["trial_id"]
        response_path = responses_path / f"{trial_id}.json"

        # Skip if resume and exists
        if resume and response_path.exists():
            execution_log["completed"] += 1
            continue

        # Check stopping at checkpoints
        n_completed = len(list(responses_path.glob("trial_*.json")))
        if n_completed in checkpoints:
            # Run evaluate on current data
            eval_result = run_stage(study_path, "evaluate", resume=True)

            # Check stopping
            stop_now, reason, interim = should_stop(study_path, config)
            if interim:
                save_interim_analysis(study_path, interim)
                print(f"\n  Interim analysis at n={n_completed}: {interim.decision}")

            if stop_now:
                execution_log["stopped_early"] = True
                execution_log["stopping_reason"] = reason
                print(f"\n  STOPPING EARLY: {reason}")
                break

        # Execute trial
        prompt = build_prompt(trial["task"], trial["condition"])
        system_prompt = build_system_prompt(trial["task"], trial["condition"])

        response = call_model_with_retry(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model_alias,
        )

        # Save response
        response_data = {
            "trial_id": trial_id,
            "task_id": trial["task_id"],
            "condition": trial["condition"],
            "prompt": prompt,
            "system_prompt": system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt,
            "response": response.response,
            "success": response.success,
            "error": response.error,
            "model": response.model,
            "timestamp": response.timestamp,
            "latency_ms": response.latency_ms,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "retry_count": response.retry_count,
        }

        with open(response_path, "w") as f:
            json.dump(response_data, f, indent=2)

        if response.success:
            execution_log["completed"] += 1
        else:
            execution_log["failed"] += 1

        execution_log["trials"].append({
            "trial_id": trial_id,
            "success": response.success,
            "latency_ms": response.latency_ms,
        })

        print(f"  [{i+1}/{n_total}] {trial_id}: {'OK' if response.success else 'FAIL'}")

    # Finalize
    execution_log["finished_at"] = datetime.now(timezone.utc).isoformat()
    with open(stage_path / "execution_log.json", "w") as f:
        json.dump(execution_log, f, indent=2)

    save_manifest(stage_path, {
        "stage": "execute",
        "completed": True,
        "stopped_early": execution_log["stopped_early"],
        "stopping_reason": execution_log["stopping_reason"],
        "trials_completed": execution_log["completed"],
        "trials_failed": execution_log["failed"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    verification = verify_stage(study_path, "execute")

    return StageResult(
        stage="execute",
        status=StageStatus.COMPLETED if verification.passed else StageStatus.FAILED,
        verification=verification,
    )


def run_full_pipeline(
    study_path: Path,
    skip_pilot: bool = False,
    resume: bool = False,
) -> dict[str, StageResult]:
    """
    Run the full pipeline with all gates and features.

    1. Configure
    2. Preregistration (lock)
    3. Generate
    4. Pilot (optional)
    5. Execute (with adaptive stopping)
    6. Evaluate
    7. Analyze
    8. Report
    """
    results = {}
    start_time = time.time()

    print("\n" + "="*70)
    print("RESEARCH PIPELINE - FULL EXECUTION")
    print("="*70)

    # Stage 1: Configure
    print(f"\n{'='*60}")
    print("Stage 1: CONFIGURE")
    print("="*60)
    results["configure"] = run_stage(study_path, "configure")
    if results["configure"].status != StageStatus.COMPLETED:
        return results

    # Stage 2: Preregistration
    print(f"\n{'='*60}")
    print("Stage 2: PREREGISTRATION")
    print("="*60)
    results["preregistration"] = run_preregistration(study_path)
    if results["preregistration"].status != StageStatus.COMPLETED:
        print("  Warning: Preregistration failed, but continuing...")

    # Stage 3: Generate
    print(f"\n{'='*60}")
    print("Stage 3: GENERATE")
    print("="*60)
    results["generate"] = run_stage(study_path, "generate")
    if results["generate"].status != StageStatus.COMPLETED:
        return results

    # Stage 4: Pilot (optional)
    if not skip_pilot:
        print(f"\n{'='*60}")
        print("Stage 4: PILOT")
        print("="*60)
        results["pilot"] = run_pilot(study_path, resume=resume)
        if results["pilot"].status == StageStatus.FAILED:
            print("  Pilot failed - cannot proceed to main study")
            return results

    # Stage 5: Execute (with gates and adaptive stopping)
    print(f"\n{'='*60}")
    print("Stage 5: EXECUTE (with adaptive stopping)")
    print("="*60)
    results["execute"] = run_with_adaptive_stopping(study_path, resume=resume)
    if results["execute"].status != StageStatus.COMPLETED:
        return results

    # Stage 6: Evaluate
    print(f"\n{'='*60}")
    print("Stage 6: EVALUATE")
    print("="*60)
    results["evaluate"] = run_stage_with_gates(study_path, "evaluate")
    if results["evaluate"].status != StageStatus.COMPLETED:
        return results

    # Stage 7: Analyze
    print(f"\n{'='*60}")
    print("Stage 7: ANALYZE")
    print("="*60)
    results["analyze"] = run_stage_with_gates(study_path, "analyze")
    if results["analyze"].status != StageStatus.COMPLETED:
        return results

    # Stage 8: Report
    print(f"\n{'='*60}")
    print("Stage 8: REPORT")
    print("="*60)
    results["report"] = run_stage_with_gates(study_path, "report")

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Stages completed: {sum(1 for r in results.values() if r.status == StageStatus.COMPLETED)}/{len(results)}")

    # Save pipeline summary
    summary = {
        "study": study_path.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "stages": {
            name: {
                "status": result.status.value,
                "error": result.error,
            }
            for name, result in results.items()
        },
    }

    with open(study_path / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return results
