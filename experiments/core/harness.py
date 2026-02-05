"""
Between-subjects experiment runner for format friction study.

Per PLAN.md:
- Pure between-subjects design (NL-only vs JSON-only conditions)
- Condition randomization
- Cluster tracking for task-level analysis
- Manipulation check recording
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Iterator
import numpy as np

from .api_providers import call_model_with_retry, APIResponse, RetryConfig
from .config import ConfigManager, get_config_manager, ExperimentConfig
from .prompts import (
    AblationCondition,
    OutputCondition,
    assemble_system_prompt,
)
from .tools import ToolDefinition, get_tools_for_experiment
from .judge import judge_response, JudgmentResult


@dataclass
class ManipulationCheck:
    """
    Manipulation check data per PLAN.md.

    For JSON-only trials:
    - attempted_json: Did the response contain any JSON-like structure?
    - syntactically_valid: Was the JSON parseable?
    - explicitly_declined: Did the model state it cannot/will not produce JSON?
    - refusal_reason: If declined, what reason was given?

    For NL-only trials:
    - attempted_nl_description: Did the response describe a tool intent?
    - tool_identified: Was a specific tool named or unambiguously referenced?
    - args_complete: Were all required arguments specified with values?
    - explicitly_declined: Did the model refuse to engage with the task?
    - refusal_reason: If declined, what reason was given?
    """
    # Common
    explicitly_declined: bool = False
    refusal_reason: Optional[str] = None

    # JSON-only checks
    attempted_json: Optional[bool] = None
    syntactically_valid: Optional[bool] = None

    # NL-only checks
    attempted_nl_description: Optional[bool] = None
    tool_identified: Optional[bool] = None
    args_complete: Optional[bool] = None


@dataclass
class TrialResult:
    """Result from a single trial."""
    trial_id: str
    task_id: str
    condition: str  # "nl_only" or "json_only"
    ablation: str  # AblationCondition value
    model: str

    # Task info
    tool_name: str
    expected_args: dict[str, Any]
    user_prompt: str

    # Response
    response: str
    is_correct: bool

    # Metadata
    timestamp: str
    latency_ms: int
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    retry_count: int

    # Manipulation checks
    manipulation_check: ManipulationCheck

    # Judgment details
    judgment: Optional[dict] = None

    # API response metadata
    request_id: Optional[str] = None
    finish_reason: Optional[str] = None


@dataclass
class Task:
    """A task in the experiment."""
    task_id: str
    tool: ToolDefinition
    user_prompt: str
    expected_args: dict[str, Any]
    category: str  # Complexity category for blind categorization


def detect_manipulation_check(
    response: str,
    is_json_condition: bool,
    judgment: JudgmentResult,
) -> ManipulationCheck:
    """
    Detect manipulation check signals from response.

    Per PLAN.md, trials where explicitly_declined=True are NOT excluded.
    """
    check = ManipulationCheck()

    response_lower = response.lower()

    # Check for explicit refusal
    refusal_phrases = [
        "i cannot", "i can't", "i am unable", "i'm unable",
        "i won't", "i will not", "refuse to",
        "not able to", "unable to produce",
    ]
    for phrase in refusal_phrases:
        if phrase in response_lower:
            check.explicitly_declined = True
            # Try to extract reason
            if "because" in response_lower:
                idx = response_lower.find("because")
                check.refusal_reason = response[idx:idx+100]
            break

    if is_json_condition:
        # JSON-specific checks
        check.attempted_json = "{" in response and "}" in response
        check.syntactically_valid = judgment.json_parseable if hasattr(judgment, 'json_parseable') else None
    else:
        # NL-specific checks
        check.attempted_nl_description = len(response.strip()) > 10
        check.tool_identified = judgment.tool_name_correct
        check.args_complete = judgment.args_complete

    return check


class ExperimentHarness:
    """
    Harness for running between-subjects format friction experiments.

    Implements:
    - Condition randomization at task level
    - Cluster tracking for task-level analysis
    - Full trial logging
    - Manipulation check recording
    """

    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        results_dir: str = "experiments/results",
        seed: int = 42,
    ):
        self.config_manager = config_manager or get_config_manager(results_dir)
        self.results_dir = Path(results_dir)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self._trials: list[TrialResult] = []
        self._current_run_id: Optional[str] = None

    def create_run(
        self,
        n_trials_per_task: int = 30,
        temperature: float = 0.0,
        ablation: AblationCondition = AblationCondition.FULL,
    ) -> str:
        """
        Create a new experiment run.

        Returns:
            Run ID string
        """
        # Create and lock config
        config = self.config_manager.create_config(
            seed=self.seed,
            temperature=temperature,
            n_trials_per_task=n_trials_per_task,
        )
        self.config_manager.lock()

        # Generate run ID
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._current_run_id = f"run_{timestamp}_{self.seed}"

        # Create run directory
        run_dir = self.results_dir / "raw" / self._current_run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save run config
        with open(run_dir / "config.json", "w") as f:
            json.dump(config.to_dict(), f, indent=2)

        return self._current_run_id

    def assign_condition(self, task_id: str) -> OutputCondition:
        """
        Randomly assign a task to NL-only or JSON-only condition.

        Uses reproducible randomization based on task_id and seed.
        """
        # Create task-specific RNG for reproducibility
        task_seed = hash(f"{self.seed}_{task_id}") % (2**32)
        task_rng = np.random.default_rng(task_seed)

        if task_rng.random() < 0.5:
            return OutputCondition.NL_ONLY
        else:
            return OutputCondition.JSON_ONLY

    def run_trial(
        self,
        task: Task,
        condition: OutputCondition,
        ablation: AblationCondition,
        model: str = "claude-sonnet",
        tools: Optional[list[ToolDefinition]] = None,
        trial_number: int = 0,
    ) -> TrialResult:
        """
        Run a single trial.

        Args:
            task: The task to run
            condition: Output condition (NL-only or JSON-only)
            ablation: System prompt ablation condition
            model: Model to use
            tools: Tools to include in prompt (defaults to all)
            trial_number: Trial number within task

        Returns:
            TrialResult with full trial data
        """
        if tools is None:
            tools = get_tools_for_experiment()

        # Assemble system prompt
        system_prompt = assemble_system_prompt(tools, ablation, condition)

        # Call model
        start_time = time.perf_counter()
        api_response = call_model_with_retry(
            prompt=task.user_prompt,
            system_prompt=system_prompt,
            model=model,
        )
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        response_text = api_response.response or ""

        # Judge response
        is_json_condition = (condition == OutputCondition.JSON_ONLY)
        judgment = judge_response(
            response=response_text,
            expected_tool=task.tool.name,
            expected_args=task.expected_args,
            is_json_condition=is_json_condition,
        )

        # Detect manipulation checks
        manipulation_check = detect_manipulation_check(
            response_text, is_json_condition, judgment
        )

        # Create trial ID
        trial_id = f"{task.task_id}_{condition.value}_{trial_number}"

        # Create result
        result = TrialResult(
            trial_id=trial_id,
            task_id=task.task_id,
            condition=condition.value,
            ablation=ablation.value,
            model=model,
            tool_name=task.tool.name,
            expected_args=task.expected_args,
            user_prompt=task.user_prompt,
            response=response_text,
            is_correct=judgment.is_correct,
            timestamp=api_response.timestamp or datetime.now(timezone.utc).isoformat(),
            latency_ms=api_response.latency_ms or latency_ms,
            input_tokens=api_response.input_tokens,
            output_tokens=api_response.output_tokens,
            retry_count=api_response.retry_count,
            manipulation_check=manipulation_check,
            judgment=asdict(judgment),
            request_id=api_response.request_id,
            finish_reason=api_response.finish_reason,
        )

        self._trials.append(result)
        return result

    def run_task(
        self,
        task: Task,
        n_trials: int = 30,
        ablation: AblationCondition = AblationCondition.FULL,
        model: str = "claude-sonnet",
        condition: Optional[OutputCondition] = None,
    ) -> list[TrialResult]:
        """
        Run all trials for a task.

        Args:
            task: The task to run
            n_trials: Number of trials to run
            ablation: System prompt ablation condition
            model: Model to use
            condition: Output condition (if None, assigned randomly)

        Returns:
            List of TrialResults
        """
        if condition is None:
            condition = self.assign_condition(task.task_id)

        results = []
        for i in range(n_trials):
            result = self.run_trial(
                task=task,
                condition=condition,
                ablation=ablation,
                model=model,
                trial_number=i,
            )
            results.append(result)

        return results

    def run_experiment(
        self,
        tasks: list[Task],
        n_trials_per_task: int = 30,
        ablation: AblationCondition = AblationCondition.FULL,
        model: str = "claude-sonnet",
    ) -> Iterator[TrialResult]:
        """
        Run full experiment across all tasks.

        Yields TrialResults as they complete for progress tracking.
        """
        for task in tasks:
            condition = self.assign_condition(task.task_id)

            for i in range(n_trials_per_task):
                result = self.run_trial(
                    task=task,
                    condition=condition,
                    ablation=ablation,
                    model=model,
                    trial_number=i,
                )
                yield result

    def save_results(self, filename: Optional[str] = None) -> Path:
        """
        Save all trial results to file.

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"trials_{self._current_run_id or 'unnamed'}.json"

        output_path = self.results_dir / "raw" / filename

        # Convert to serializable format
        data = {
            "run_id": self._current_run_id,
            "seed": self.seed,
            "n_trials": len(self._trials),
            "trials": [
                {
                    **asdict(trial),
                    "manipulation_check": asdict(trial.manipulation_check),
                }
                for trial in self._trials
            ],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return output_path

    def get_summary(self) -> dict:
        """Get summary statistics for the current run."""
        if not self._trials:
            return {"n_trials": 0}

        nl_trials = [t for t in self._trials if t.condition == "nl_only"]
        json_trials = [t for t in self._trials if t.condition == "json_only"]

        nl_correct = sum(1 for t in nl_trials if t.is_correct)
        json_correct = sum(1 for t in json_trials if t.is_correct)

        # Manipulation check stats
        nl_declined = sum(1 for t in nl_trials if t.manipulation_check.explicitly_declined)
        json_declined = sum(1 for t in json_trials if t.manipulation_check.explicitly_declined)

        return {
            "run_id": self._current_run_id,
            "n_trials": len(self._trials),
            "nl_only": {
                "n": len(nl_trials),
                "correct": nl_correct,
                "accuracy": nl_correct / len(nl_trials) if nl_trials else 0,
                "declined": nl_declined,
            },
            "json_only": {
                "n": len(json_trials),
                "correct": json_correct,
                "accuracy": json_correct / len(json_trials) if json_trials else 0,
                "declined": json_declined,
            },
            "overall_friction": (
                (nl_correct / len(nl_trials) if nl_trials else 0)
                - (json_correct / len(json_trials) if json_trials else 0)
            ) if nl_trials and json_trials else None,
        }

    @property
    def trials(self) -> list[TrialResult]:
        """Get all trial results."""
        return self._trials


def compute_cluster_stats(trials: list[TrialResult]) -> dict:
    """
    Compute cluster (task-level) statistics.

    Per PLAN.md, bootstrap resampling should be at task level, not trial level.
    """
    # Group by task
    by_task: dict[str, list[TrialResult]] = {}
    for trial in trials:
        if trial.task_id not in by_task:
            by_task[trial.task_id] = []
        by_task[trial.task_id].append(trial)

    # Compute per-task accuracy
    task_accuracies_nl = []
    task_accuracies_json = []

    for task_id, task_trials in by_task.items():
        nl_trials = [t for t in task_trials if t.condition == "nl_only"]
        json_trials = [t for t in task_trials if t.condition == "json_only"]

        if nl_trials:
            acc = sum(1 for t in nl_trials if t.is_correct) / len(nl_trials)
            task_accuracies_nl.append(acc)

        if json_trials:
            acc = sum(1 for t in json_trials if t.is_correct) / len(json_trials)
            task_accuracies_json.append(acc)

    return {
        "n_tasks": len(by_task),
        "n_tasks_nl": len(task_accuracies_nl),
        "n_tasks_json": len(task_accuracies_json),
        "task_level_accuracy_nl": np.mean(task_accuracies_nl) if task_accuracies_nl else None,
        "task_level_accuracy_json": np.mean(task_accuracies_json) if task_accuracies_json else None,
    }
